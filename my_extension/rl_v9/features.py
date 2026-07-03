#!/usr/bin/env python3
"""
rl_v9 / features.py  --  the single source of truth for feature extraction.

Everything (dataset collection from replays, the RL environment, and the live
inference bot) builds the *same* `WorldView`, maintains the *same*
`FleetMemory`, and calls the *same* extraction functions -- zero skew between
training-time and play-time representations.

New in rl_v9 vs rl_v8:
  * FleetMemory -- deterministic per-ship recurrent state (sticky homing flag,
    previous executed action, stuck counter).  This is the fix for the
    hidden-FSM-state problem: V71 keeps per-ship state across turns
    (RETURNING + a queued path), which a stateless single-frame policy cannot
    represent (BC match rate capped ~58% in rl_v7/rl_v8).  Every adapter can
    reproduce FleetMemory exactly because it is a pure function of the
    observation/action history.
  * A coarse global map tensor (8x8x4), recentred on the ship (or factory for
    the spawn net) and block-pooled from the full map.
  * Spawn features: a global scalar vector + factory-centred global map, for
    the learned spawn decision.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    MAX_HALITE, EXTRACT_RATIO, MOVE_COST_RATIO, DROPOFF_COST, SHIP_COST,
    INSPIRATION_RADIUS, INSPIRATION_SHIP_COUNT,
    FLEET_NORM, DROPOFF_NORM, BANK_SCALE,
    HOME_TRIGGER_FRAC, STUCK_NORM,
    PATCH_RADIUS, PATCH_SIZE, PATCH_CHANNELS,
    GLOBAL_SIZE, GLOBAL_CHANNELS,
    N_SCALARS, FEATURE_NAMES, N_SPAWN_SCALARS, SPAWN_FEATURE_NAMES,
    N_ACTIONS, ACTION_STAY, ACTION_NORTH, ACTION_EAST, ACTION_SOUTH,
    ACTION_WEST, ACTION_DROPOFF,
    game_max_turns, torus_dist, torus_delta, dropoff_legal, spawn_legal,
)


# ---------------------------------------------------------------------------
# WorldView
# ---------------------------------------------------------------------------

class WorldView:
    """A source-agnostic snapshot of one player's view of the board."""

    __slots__ = (
        'W', 'H', 'turn', 'max_turns', 'halite',
        'my_deposits', 'opp_deposits', 'my_ships', 'opp_ships',
        'my_bank', 'opp_bank',
        '_my_pos', '_opp_pos', '_deposit_set', '_opp_deposit_set',
        '_total_map_halite', '_global_map',
    )

    def __init__(self, W, H, turn, max_turns, halite,
                 my_deposits, opp_deposits, my_ships, opp_ships,
                 my_bank, opp_bank):
        self.W = W
        self.H = H
        self.turn = turn
        self.max_turns = max_turns
        self.halite = halite
        self.my_deposits = my_deposits
        self.opp_deposits = opp_deposits
        self.my_ships = my_ships          # dict id -> (x, y, cargo)
        self.opp_ships = opp_ships        # list of (x, y, cargo)
        self.my_bank = my_bank
        self.opp_bank = opp_bank
        self._finalize()

    def _finalize(self):
        self._my_pos = {(x, y): cargo for (x, y, cargo) in self.my_ships.values()}
        self._opp_pos = {(x, y): cargo for (x, y, cargo) in self.opp_ships}
        self._deposit_set = set(self.my_deposits)
        self._opp_deposit_set = set(self.opp_deposits)
        self._total_map_halite = sum(self.halite.values())
        self._global_map = None      # built lazily, once per turn


# ---------------------------------------------------------------------------
# FleetMemory  --  deterministic per-ship recurrent state (new in rl_v9)
# ---------------------------------------------------------------------------

class FleetMemory:
    """Per-ship memory carried across turns.

    Usage contract (identical in all three adapters):
        1. at the START of each turn, after building the WorldView:
               mem.begin_turn(wv)
        2. extract features (extract_scalars(wv, sid, mem))
        3. once each ship's *executed* action for the turn is known:
               mem.commit_actions({sid: action, ...})

    State per ship:
        homing      : sticky 0/1.  Set when cargo >= HOME_TRIGGER_FRAC, cleared
                      when the ship stands on a friendly deposit.  Mirrors the
                      hidden RETURNING state of V71's FSM.
        prev_action : the action executed last turn (STAY for new ships).
        stuck       : consecutive turns spent at the same position.
    """

    def __init__(self):
        self.homing: Dict[int, int] = {}
        self.prev_action: Dict[int, int] = {}
        self.stuck: Dict[int, int] = {}
        self._last_pos: Dict[int, Tuple[int, int]] = {}

    def begin_turn(self, wv: WorldView):
        live = set(wv.my_ships.keys())
        # prune dead/converted ships
        for d in (self.homing, self.prev_action, self.stuck, self._last_pos):
            for sid in [s for s in d if s not in live]:
                del d[sid]
        for sid, (x, y, cargo) in wv.my_ships.items():
            if (x, y) in wv._deposit_set:
                self.homing[sid] = 0
            elif cargo >= HOME_TRIGGER_FRAC * MAX_HALITE:
                self.homing[sid] = 1
            elif sid not in self.homing:
                self.homing[sid] = 0
            if self._last_pos.get(sid) == (x, y):
                self.stuck[sid] = self.stuck.get(sid, 0) + 1
            else:
                self.stuck[sid] = 0
            self._last_pos[sid] = (x, y)
            if sid not in self.prev_action:
                self.prev_action[sid] = ACTION_STAY

    def commit_actions(self, actions: Dict[int, int]):
        for sid, act in actions.items():
            self.prev_action[sid] = act


# ---------------------------------------------------------------------------
# Core feature extraction
# ---------------------------------------------------------------------------

def _nearest_deposit(sx, sy, deposits, W, H):
    return min(deposits, key=lambda d: torus_dist(sx, sy, d[0], d[1], W, H))


def _richest_in_window(sx, sy, halite, W, H, radius=PATCH_RADIUS):
    """(rx, ry, val) for richest cell in the window.  Deterministic tie-break."""
    best_val, best_pos, best_dist = -1, (sx, sy), 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            cx = (sx + dx) % W
            cy = (sy + dy) % H
            val = halite.get((cx, cy), 0)
            dist = abs(dx) + abs(dy)
            if val > best_val:
                best_val, best_pos, best_dist = val, (cx, cy), dist
            elif val == best_val:
                is_cur = (cx == sx and cy == sy)
                was_cur = (best_pos == (sx, sy))
                if is_cur and not was_cur:
                    best_pos, best_dist = (cx, cy), dist
                elif not is_cur and not was_cur:
                    if dist < best_dist or (dist == best_dist and (cx, cy) < best_pos):
                        best_pos, best_dist = (cx, cy), dist
    return best_pos[0], best_pos[1], best_val


def extract_scalars(wv: WorldView, ship_id: int,
                    mem: Optional[FleetMemory] = None) -> np.ndarray:
    """Build the N_SCALARS float32 feature vector for one of `wv`'s own ships."""
    W, H = wv.W, wv.H
    sx, sy, cargo = wv.my_ships[ship_id]
    halite = wv.halite
    cell_h = halite.get((sx, sy), 0)
    max_turns = wv.max_turns
    turns_left = max_turns - wv.turn
    max_dist = W + H

    # homing
    near = _nearest_deposit(sx, sy, wv.my_deposits, W, H)
    dist_home = torus_dist(sx, sy, near[0], near[1], W, H)
    dhx, dhy = torus_delta(sx, sy, near[0], near[1], W, H)
    on_deposit = 1.0 if dist_home == 0 else 0.0
    return_urgency = 1.0 if (cargo > 0 and turns_left <= dist_home * 1.5 + 1) else 0.0
    turns_slack = (turns_left - dist_home) / max(1, max_turns)

    # prospect
    rx, ry, rval = _richest_in_window(sx, sy, halite, W, H)
    drx, dry = torus_delta(sx, sy, rx, ry, W, H)
    dist_rich = torus_dist(sx, sy, rx, ry, W, H)

    # local halite means
    mean3 = 0.0
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            mean3 += halite.get(((sx + ox) % W, (sy + oy) % H), 0)
    mean3 /= 9.0
    win_sum = 0.0
    win_n = 0
    for oy in range(-PATCH_RADIUS, PATCH_RADIUS + 1):
        for ox in range(-PATCH_RADIUS, PATCH_RADIUS + 1):
            win_sum += halite.get(((sx + ox) % W, (sy + oy) % H), 0)
            win_n += 1
    window_mean = win_sum / win_n

    # danger / contention / inspiration
    enemy_within_1 = enemy_within_2 = 0
    friendly_within_1 = friendly_within_2 = 0
    enemy_count_r4 = 0
    min_enemy_cargo_near = MAX_HALITE
    for (ex, ey, ecargo) in wv.opp_ships:
        d = torus_dist(sx, sy, ex, ey, W, H)
        if d <= INSPIRATION_RADIUS:
            enemy_count_r4 += 1
        if d <= 2:
            enemy_within_2 += 1
            if ecargo < min_enemy_cargo_near:
                min_enemy_cargo_near = ecargo
            if d <= 1:
                enemy_within_1 += 1
    for fid, (fx, fy, fcargo) in wv.my_ships.items():
        if fid == ship_id:
            continue
        d = torus_dist(sx, sy, fx, fy, W, H)
        if d <= 2:
            friendly_within_2 += 1
            if d <= 1:
                friendly_within_1 += 1

    is_inspired = 1.0 if enemy_count_r4 >= INSPIRATION_SHIP_COUNT else 0.0
    has_enemy_near = enemy_within_2 > 0

    # economy / global
    my_ships = len(wv.my_ships)
    opp_ships = len(wv.opp_ships)
    my_bank = wv.my_bank
    opp_bank = wv.opp_bank
    total_bank = my_bank + opp_bank
    num_dropoffs = len(wv.my_deposits) - 1
    cell_owned = (sx, sy) in wv._deposit_set or (sx, sy) in wv._opp_deposit_set
    dcost_here = max(0, DROPOFF_COST - (cell_h + cargo))
    dropoff_aff = 1.0 if my_bank >= dcost_here else 0.0
    d_legal = 1.0 if dropoff_legal(my_bank, cell_h, cargo, cell_owned) else 0.0

    mine_yield = math.ceil(cell_h / EXTRACT_RATIO) if cell_h > 0 else 0
    can_afford_move = 1.0 if cargo >= cell_h // MOVE_COST_RATIO else 0.0

    # memory features
    if mem is not None:
        homing = float(mem.homing.get(ship_id, 0))
        prev = mem.prev_action.get(ship_id, ACTION_STAY)
        stuck = min(mem.stuck.get(ship_id, 0) / STUCK_NORM, 1.0)
    else:
        homing, prev, stuck = 0.0, ACTION_STAY, 0.0
    prev_onehot = [0.0] * 5
    prev_onehot[prev if prev < 5 else ACTION_STAY] = 1.0

    feats = np.array([
        # per-ship economy
        cargo / MAX_HALITE,
        (MAX_HALITE - cargo) / MAX_HALITE,
        1.0 if cargo >= HOME_TRIGGER_FRAC * MAX_HALITE else 0.0,
        cell_h / MAX_HALITE,
        mine_yield / MAX_HALITE,
        can_afford_move,
        is_inspired,
        # homing
        dist_home / max_dist,
        dhx / W,
        dhy / H,
        on_deposit,
        return_urgency,
        turns_slack,
        # prospecting
        drx / W,
        dry / H,
        rval / MAX_HALITE,
        dist_rich / max_dist,
        mean3 / MAX_HALITE,
        window_mean / MAX_HALITE,
        # danger / contention
        min(enemy_within_1 / 4.0, 1.0),
        min(enemy_within_2 / 8.0, 1.0),
        min(friendly_within_1 / 4.0, 1.0),
        min(friendly_within_2 / 8.0, 1.0),
        (min_enemy_cargo_near / MAX_HALITE) if has_enemy_near else 1.0,
        min(enemy_count_r4 / 4.0, 1.0),
        # global / fleet / phase
        wv.turn / max_turns,
        turns_left / max_turns,
        min(my_ships / FLEET_NORM, 1.0),
        min(opp_ships / FLEET_NORM, 1.0),
        min(my_bank / BANK_SCALE, 1.0),
        min(opp_bank / BANK_SCALE, 1.0),
        math.tanh((my_bank - opp_bank) / 5000.0),
        my_bank / (total_bank + 1),
        min(num_dropoffs / DROPOFF_NORM, 1.0),
        wv._total_map_halite / (W * H * MAX_HALITE),
        dropoff_aff,
        d_legal,
        W / 64.0,
        H / 64.0,
        # memory
        homing,
        *prev_onehot,
        stuck,
    ], dtype=np.float32)

    assert feats.shape[0] == N_SCALARS, \
        f"feature count {feats.shape[0]} != N_SCALARS {N_SCALARS}"
    return feats


def extract_patch(wv: WorldView, ship_id: int) -> np.ndarray:
    """PATCH_SIZE x PATCH_SIZE x PATCH_CHANNELS float32, ship-centred, toroidal."""
    W, H = wv.W, wv.H
    sx, sy, _ = wv.my_ships[ship_id]
    r = PATCH_RADIUS
    patch = np.zeros((PATCH_SIZE, PATCH_SIZE, PATCH_CHANNELS), dtype=np.float32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            cx = (sx + dx) % W
            cy = (sy + dy) % H
            wy = dy + r
            wx = dx + r
            patch[wy, wx, 0] = wv.halite.get((cx, cy), 0) / MAX_HALITE
            if (cx, cy) in wv._my_pos:
                patch[wy, wx, 1] = 1.0
                patch[wy, wx, 3] = wv._my_pos[(cx, cy)] / MAX_HALITE
            elif (cx, cy) in wv._opp_pos:
                patch[wy, wx, 2] = 1.0
                patch[wy, wx, 4] = wv._opp_pos[(cx, cy)] / MAX_HALITE
            if (cx, cy) in wv._deposit_set:
                patch[wy, wx, 5] = 1.0
            elif (cx, cy) in wv._opp_deposit_set:
                patch[wy, wx, 5] = -1.0
    return patch


# ---------------------------------------------------------------------------
# Coarse global map (new in rl_v9)
# ---------------------------------------------------------------------------

def _build_global_base(wv: WorldView) -> np.ndarray:
    """Full-resolution [GLOBAL_CHANNELS, H, W] tensor, built once per turn."""
    if wv._global_map is not None:
        return wv._global_map
    W, H = wv.W, wv.H
    g = np.zeros((GLOBAL_CHANNELS, H, W), dtype=np.float32)
    for (x, y), h in wv.halite.items():
        g[0, y, x] = h / MAX_HALITE
    for (x, y) in wv._my_pos:
        g[1, y, x] = 1.0
    for (x, y) in wv._opp_pos:
        g[2, y, x] = 1.0
    for (x, y) in wv._deposit_set:
        g[3, y, x] = 1.0
    for (x, y) in wv._opp_deposit_set:
        g[3, y, x] = -1.0
    wv._global_map = g
    return g


def extract_global(wv: WorldView, cx: int, cy: int) -> np.ndarray:
    """[GLOBAL_CHANNELS, GLOBAL_SIZE, GLOBAL_SIZE] float32, recentred on (cx, cy)
    then block-pooled.  Channel 0 is block-mean halite; channels 1-3 are block
    sums (scaled/clipped)."""
    g = _build_global_base(wv)
    W, H = wv.W, wv.H
    rolled = np.roll(np.roll(g, H // 2 - cy, axis=1), W // 2 - cx, axis=2)
    fh = H // GLOBAL_SIZE
    fw = W // GLOBAL_SIZE
    blocks = rolled.reshape(GLOBAL_CHANNELS, GLOBAL_SIZE, fh, GLOBAL_SIZE, fw)
    out = np.empty((GLOBAL_CHANNELS, GLOBAL_SIZE, GLOBAL_SIZE), dtype=np.float32)
    out[0] = blocks[0].mean(axis=(1, 3))
    out[1] = np.clip(blocks[1].sum(axis=(1, 3)) / 8.0, 0.0, 1.0)
    out[2] = np.clip(blocks[2].sum(axis=(1, 3)) / 8.0, 0.0, 1.0)
    out[3] = np.clip(blocks[3].sum(axis=(1, 3)), -1.0, 1.0)
    return out


def extract_ship_global(wv: WorldView, ship_id: int) -> np.ndarray:
    sx, sy, _ = wv.my_ships[ship_id]
    return extract_global(wv, sx, sy)


def extract_spawn_global(wv: WorldView) -> np.ndarray:
    fx, fy = wv.my_deposits[0]
    return extract_global(wv, fx, fy)


# ---------------------------------------------------------------------------
# Spawn features (global scalar vector, one per turn)
# ---------------------------------------------------------------------------

def extract_spawn_scalars(wv: WorldView) -> np.ndarray:
    W, H = wv.W, wv.H
    max_turns = wv.max_turns
    turns_left = max_turns - wv.turn
    my_ships = len(wv.my_ships)
    opp_ships = len(wv.opp_ships)
    my_bank = wv.my_bank
    opp_bank = wv.opp_bank
    total_bank = my_bank + opp_bank
    fx, fy = wv.my_deposits[0]
    factory_occupied = 1.0 if ((fx, fy) in wv._my_pos or (fx, fy) in wv._opp_pos) else 0.0

    area_sum, area_n = 0.0, 0
    for oy in range(-4, 5):
        for ox in range(-4, 5):
            area_sum += wv.halite.get(((fx + ox) % W, (fy + oy) % H), 0)
            area_n += 1

    ships_total = my_ships + opp_ships
    halite_per_ship = wv._total_map_halite / (ships_total + 1)

    feats = np.array([
        wv.turn / max_turns,
        turns_left / max_turns,
        min(my_ships / FLEET_NORM, 1.0),
        min(opp_ships / FLEET_NORM, 1.0),
        math.tanh((my_ships - opp_ships) / 8.0),
        min(my_bank / BANK_SCALE, 1.0),
        min(opp_bank / BANK_SCALE, 1.0),
        math.tanh((my_bank - opp_bank) / 5000.0),
        my_bank / (total_bank + 1),
        wv._total_map_halite / (W * H * MAX_HALITE),
        min(halite_per_ship / (MAX_HALITE * 8.0), 1.0),
        min((len(wv.my_deposits) - 1) / DROPOFF_NORM, 1.0),
        min(max(0, len(wv.opp_deposits) - 1) / DROPOFF_NORM, 1.0),
        factory_occupied,
        (area_sum / area_n) / MAX_HALITE,
        1.0 if wv.my_bank >= SHIP_COST else 0.0,
        W / 64.0,
        H / 64.0,
    ], dtype=np.float32)
    assert feats.shape[0] == N_SPAWN_SCALARS
    return feats


# ---------------------------------------------------------------------------
# Action masks
# ---------------------------------------------------------------------------

def action_mask(wv: WorldView, ship_id: int) -> np.ndarray:
    """bool[N_ACTIONS]: moves always legal; DROPOFF only physically legal
    (unowned cell + affordable).  No strategy baked in -- that is learned."""
    sx, sy, cargo = wv.my_ships[ship_id]
    cell_h = wv.halite.get((sx, sy), 0)
    cell_owned = (sx, sy) in wv._deposit_set or (sx, sy) in wv._opp_deposit_set
    ok = dropoff_legal(wv.my_bank, cell_h, cargo, cell_owned)
    mask = np.ones(N_ACTIONS, dtype=bool)
    mask[ACTION_DROPOFF] = bool(ok)
    return mask


def spawn_mask(wv: WorldView) -> np.ndarray:
    """bool[2]: NO always legal; YES iff affordable (money is the only gate)."""
    turns_left = wv.max_turns - wv.turn
    ok = spawn_legal(wv.my_bank, turns_left)
    return np.array([True, bool(ok)], dtype=bool)


# ---------------------------------------------------------------------------
# Adapter: build a WorldView from reconstructed replay state
# ---------------------------------------------------------------------------

def world_view_from_replay(pid, W, H, turn, max_turns, halite,
                           factories, dropoffs_by_pid, entities_frame,
                           bank_by_pid, num_players):
    my_deposits = [factories[pid]] + list(dropoffs_by_pid.get(pid, []))
    opp_deposits = []
    for opid in range(num_players):
        if opid == pid:
            continue
        opp_deposits.append(factories[opid])
        opp_deposits.extend(dropoffs_by_pid.get(opid, []))

    my_ships = {}
    opp_ships = []
    for opid_str, ents in entities_frame.items():
        opid = int(opid_str)
        for eid_str, e in ents.items():
            tup = (e['x'], e['y'], e['energy'])
            if opid == pid:
                my_ships[int(eid_str)] = tup
            else:
                opp_ships.append(tup)

    opp_bank = max((bank_by_pid.get(o, 0) for o in range(num_players) if o != pid),
                   default=0)
    return WorldView(W, H, turn, max_turns, halite,
                     my_deposits, opp_deposits, my_ships, opp_ships,
                     bank_by_pid.get(pid, 0), opp_bank)


# ---------------------------------------------------------------------------
# Adapter: build a WorldView from the Python engine (used in PPO training)
# ---------------------------------------------------------------------------

def world_view_from_engine(engine, pid):
    W, H = engine.width, engine.height
    halite = dict(engine.halite)
    factory = engine.players[pid]['factory']
    my_deposits = [factory] + [(dx, dy) for (_d, dx, dy) in engine.players[pid]['dropoffs']]
    opp_deposits = []
    for opid in engine.players:
        if opid == pid:
            continue
        opp_deposits.append(engine.players[opid]['factory'])
        opp_deposits.extend((dx, dy) for (_d, dx, dy) in engine.players[opid]['dropoffs'])

    my_ships = {}
    for eid, (x, y) in engine.player_entities[pid].items():
        my_ships[eid] = (x, y, engine.entities[eid]['cargo'])
    opp_ships = []
    for opid, ents in engine.player_entities.items():
        if opid == pid:
            continue
        for eid, (x, y) in ents.items():
            opp_ships.append((x, y, engine.entities[eid]['cargo']))

    opp_bank = max((engine.players[o]['energy'] for o in engine.players if o != pid),
                   default=0)
    return WorldView(W, H, engine.turn, engine.max_turns, halite,
                     my_deposits, opp_deposits, my_ships, opp_ships,
                     engine.players[pid]['energy'], opp_bank)


# ---------------------------------------------------------------------------
# Adapter: build a WorldView from the hlt API (used in live inference)
# ---------------------------------------------------------------------------

def world_view_from_hlt(game, me):
    gmap = game.game_map
    W, H = gmap.width, gmap.height
    halite = {}
    for x in range(W):
        for y in range(H):
            halite[(x, y)] = gmap[_hlt_pos(x, y)].halite_amount

    my_deposits = [(me.shipyard.position.x, me.shipyard.position.y)]
    my_deposits += [(d.position.x, d.position.y) for d in me.get_dropoffs()]
    opp_deposits = []
    for pid, player in game.players.items():
        if pid == me.id:
            continue
        opp_deposits.append((player.shipyard.position.x, player.shipyard.position.y))
        opp_deposits += [(d.position.x, d.position.y) for d in player.get_dropoffs()]

    my_ships = {s.id: (s.position.x, s.position.y, s.halite_amount)
                for s in me.get_ships()}
    opp_ships = []
    for pid, player in game.players.items():
        if pid == me.id:
            continue
        for s in player.get_ships():
            opp_ships.append((s.position.x, s.position.y, s.halite_amount))

    opp_bank = max((p.halite_amount for pid, p in game.players.items()
                    if pid != me.id), default=0)
    max_turns = game_max_turns(W, H)
    return WorldView(W, H, game.turn_number, max_turns, halite,
                     my_deposits, opp_deposits, my_ships, opp_ships,
                     me.halite_amount, opp_bank)


_HLT_POS = None


def _hlt_pos(x, y):
    global _HLT_POS
    if _HLT_POS is None:
        from hlt.positionals import Position
        _HLT_POS = Position
    return _HLT_POS(x, y)
