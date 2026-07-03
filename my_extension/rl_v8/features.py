#!/usr/bin/env python3
"""
rl_v8 / features.py  --  the single source of truth for feature extraction.

Everything (dataset collection from replays, the RL environment, and the live
inference bot) builds the *same* `WorldView` object and calls the *same*
`extract_scalars` / `extract_patch` functions.  This is what guarantees there
is zero skew between the features a network is trained on and the features it
sees at play time.

A `WorldView` is a cheap, source-agnostic snapshot of one player's view of the
board at the START of a turn (before that turn's commands are applied):

    wv.W, wv.H              map dimensions
    wv.turn                 current turn number (1-based)
    wv.max_turns            true game length for this map size
    wv.halite[(x, y)]       halite on every cell
    wv.my_deposits          list[(x, y)]  (factory first, then dropoffs)
    wv.opp_deposits         list[(x, y)]
    wv.my_ships             dict ship_id -> (x, y, cargo)
    wv.opp_ships            list[(x, y, cargo)]
    wv.my_bank, wv.opp_bank ints

Three adapters build a WorldView from each data source:
    world_view_from_replay(...)   used by collect_dataset.py
    world_view_from_engine(...)   used by rl_env.py (PPO training)
    world_view_from_hlt(...)      used by rl_bot.py (live inference)
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    MAX_HALITE, EXTRACT_RATIO, MOVE_COST_RATIO, DROPOFF_COST,
    INSPIRATION_RADIUS, INSPIRATION_SHIP_COUNT,
    MAX_FLEET, MAX_DROPOFFS, BANK_SCALE, HOME_CARGO_THRESHOLD,
    PATCH_RADIUS, PATCH_SIZE, PATCH_CHANNELS,
    N_SCALARS, FEATURE_NAMES, game_max_turns,
    torus_dist, torus_delta, target_dropoffs, dropoff_legal,
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
        # precomputed lookups (filled by _finalize)
        '_my_pos', '_opp_pos', '_deposit_set', '_opp_deposit_set',
        '_total_map_halite',
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


# ---------------------------------------------------------------------------
# Core feature extraction (operate purely on a WorldView)
# ---------------------------------------------------------------------------

def _nearest_deposit(sx, sy, deposits, W, H):
    return min(deposits, key=lambda d: torus_dist(sx, sy, d[0], d[1], W, H))


def _richest_in_window(sx, sy, halite, W, H, radius=PATCH_RADIUS):
    """Return (rx, ry, val) for richest cell in the (2r+1)^2 window (incl. centre).

    Tie-break: higher halite -> current cell preferred -> shorter Manhattan
    distance -> lower (x, y).  Deterministic, matches rl_v5's prospect scan style.
    """
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


def extract_scalars(wv: WorldView, ship_id: int) -> np.ndarray:
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

    # danger / contention / inspiration  (single scan over ships)
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
    num_dropoffs = len(wv.my_deposits) - 1   # minus the factory
    cell_owned = (sx, sy) in wv._deposit_set or (sx, sy) in wv._opp_deposit_set
    dcost_here = max(0, DROPOFF_COST - (cell_h + cargo))
    dropoff_aff = 1.0 if my_bank >= dcost_here else 0.0
    d_legal = 1.0 if dropoff_legal(my_bank, cell_h, cargo, dist_home,
                                   num_dropoffs, turns_left, cell_owned) else 0.0

    mine_yield = math.ceil(cell_h / EXTRACT_RATIO) if cell_h > 0 else 0
    can_afford_move = 1.0 if cargo >= cell_h // MOVE_COST_RATIO else 0.0

    feats = np.array([
        # per-ship economy
        cargo / MAX_HALITE,
        (MAX_HALITE - cargo) / MAX_HALITE,
        1.0 if cargo >= HOME_CARGO_THRESHOLD * MAX_HALITE else 0.0,
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
        my_ships / MAX_FLEET,
        opp_ships / MAX_FLEET,
        min(my_bank / BANK_SCALE, 1.0),
        min(opp_bank / BANK_SCALE, 1.0),
        math.tanh((my_bank - opp_bank) / 5000.0),
        my_bank / (total_bank + 1),
        num_dropoffs / MAX_DROPOFFS,
        wv._total_map_halite / (W * H * MAX_HALITE),
        dropoff_aff,
        d_legal,
        W / 64.0,
        H / 64.0,
    ], dtype=np.float32)

    assert feats.shape[0] == N_SCALARS, \
        f"feature count {feats.shape[0]} != N_SCALARS {N_SCALARS}"
    return feats


def extract_patch(wv: WorldView, ship_id: int) -> np.ndarray:
    """Build the PATCH_SIZE x PATCH_SIZE x PATCH_CHANNELS float32 local map tensor
    centred on the ship (toroidal wrap)."""
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


def action_mask(wv: WorldView, ship_id: int) -> np.ndarray:
    """bool[N_ACTIONS]: all moves always legal; DROPOFF legal only when sensible."""
    from config import N_ACTIONS, ACTION_DROPOFF
    W, H = wv.W, wv.H
    sx, sy, cargo = wv.my_ships[ship_id]
    cell_h = wv.halite.get((sx, sy), 0)
    near = _nearest_deposit(sx, sy, wv.my_deposits, W, H)
    dist = torus_dist(sx, sy, near[0], near[1], W, H)
    num_dropoffs = len(wv.my_deposits) - 1
    turns_left = wv.max_turns - wv.turn
    cell_owned = (sx, sy) in wv._deposit_set or (sx, sy) in wv._opp_deposit_set
    ok = dropoff_legal(wv.my_bank, cell_h, cargo, dist, num_dropoffs,
                       turns_left, cell_owned)
    mask = np.ones(N_ACTIONS, dtype=bool)
    mask[ACTION_DROPOFF] = bool(ok)
    return mask


# ---------------------------------------------------------------------------
# Adapter: build a WorldView from reconstructed replay state
# ---------------------------------------------------------------------------

def world_view_from_replay(pid, W, H, turn, max_turns, halite,
                           factories, dropoffs_by_pid, entities_frame,
                           bank_by_pid, num_players):
    """Build a WorldView for player `pid` from reconstructed replay state.

    entities_frame is the replay frame's "entities" dict:
        { "0": {"3": {"x":.., "y":.., "energy":.., "is_inspired":..}, ...}, ... }
    factories: list[(x, y)] indexed by player id.
    dropoffs_by_pid: dict pid -> list[(x, y)] of dropoffs built so far.
    bank_by_pid: dict pid -> bank halite at start of this turn.
    """
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
