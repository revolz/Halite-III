#!/usr/bin/env python3
"""
rl_v6 — PURE reinforcement-learning Halite III bot (no rule-based logic).

Unlike rl_v5 (FSM + logit prior + homing/prospect/collision/spawn rules), rl_v6
runs ONLY the neural network:

  per ship   ->  pure features (14ch + 29 base scalars)
             ->  ActorCritic(n_scalars=29, n_actions=6)
             ->  primitive action (STAY/N/E/S/W/DROPOFF), executed VERBATIM

  per turn   ->  compact global features
             ->  SpawnHead -> spawn yes/no

There is deliberately NO collision avoidance, NO homing override, NO endgame
rule and NO spawn economic gate.  Any structure rl_v6 shows must be *learned*.
The only non-net constraint is the engine-legality mask on DROPOFF (an action the
engine would silently reject anyway — not a strategy rule).

Usage
-----
    python rl_bot.py --model checkpoints/model_weights.pt \
                     --spawn-model checkpoints/spawn_weights.pt [--deterministic]
"""

import argparse
import math
import os
import sys
from typing import Tuple

import numpy as np

# ── path setup ──────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))      # rl_v6/
_MY_EXT    = os.path.dirname(_HERE)                           # my_extension/
_REPO_ROOT = os.path.dirname(_MY_EXT)                         # repo root
sys.path.insert(0, _HERE)
sys.path.insert(0, _MY_EXT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'starter_kits', 'Python3'))

import torch
from rl_model    import ActorCritic
from spawn_model import SpawnHead
from rl_config   import (
    N_SCALARS_V6, N_SHIP_ACTIONS_V6, ACTION_DROPOFF_V6, spawn_global_features,
    resolve_macro, resolve_collisions, macro_prior, SHIP_COST,
)
from rl_features import (
    WINDOW_SIZE, N_SPATIAL_CHANNELS,
    ACTION_TO_DIR, torus_dist, torus_delta, overlay_committed, effective_dest,
    ACTION_STAY, ACTION_NORTH, ACTION_SOUTH, ACTION_EAST, ACTION_WEST,
    PROSPECT_RADIUS, dropoff_legal, target_dropoffs, game_max_turns,
    compute_home_cost_field,
)

import hlt
from hlt                import constants
from hlt.positionals    import Direction, Position


# ---------------------------------------------------------------------------
# Pure feature extraction from hlt API objects (no FSM, no rules).
# These are identical to rl_v5's BASE extractors — they read the world only.
# ---------------------------------------------------------------------------

def _inspired(position: Position, game: hlt.Game, me_id: int) -> bool:
    from hlt.constants import INSPIRATION_RADIUS, INSPIRATION_SHIP_COUNT, INSPIRATION_ENABLED
    if not INSPIRATION_ENABLED:
        return False
    count = 0
    for pid, player in game.players.items():
        if pid == me_id:
            continue
        for ship in player.get_ships():
            if game.game_map.calculate_distance(position, ship.position) <= INSPIRATION_RADIUS:
                count += 1
                if count >= INSPIRATION_SHIP_COUNT:
                    return True
    return False


def _nearest_deposit(position: Position, game: hlt.Game, me) -> Position:
    candidates = [me.shipyard.position] + [d.position for d in me.get_dropoffs()]
    return min(candidates, key=lambda p: game.game_map.calculate_distance(position, p))


def _richest_in_prospect_window_hlt(sx, sy, gmap, W, H, radius=PROSPECT_RADIUS):
    best_val, best_pos, best_dist = -1, (sx, sy), 0
    for dy_off in range(-radius, radius + 1):
        for dx_off in range(-radius, radius + 1):
            cx, cy = (sx + dx_off) % W, (sy + dy_off) % H
            val  = gmap[Position(cx, cy)].halite_amount
            dist = abs(dx_off) + abs(dy_off)
            if val > best_val:
                best_val, best_pos, best_dist = val, (cx, cy), dist
            elif val == best_val:
                is_current  = (cx == sx and cy == sy)
                was_current = (best_pos[0] == sx and best_pos[1] == sy)
                if is_current and not was_current:
                    best_pos, best_dist = (cx, cy), dist
                elif not is_current and not was_current:
                    if dist < best_dist or (dist == best_dist and (cx, cy) < best_pos):
                        best_pos, best_dist = (cx, cy), dist
    return best_pos[0], best_pos[1], best_val


def _prospect_scalars_hlt(sx, sy, gmap, W, H, max_dist, MAX_HALITE):
    rx, ry, val = _richest_in_prospect_window_hlt(sx, sy, gmap, W, H)
    dx, dy = torus_delta(sx, sy, rx, ry, W, H)
    dist   = torus_dist(sx, sy, rx, ry, W, H)
    return dx / W, dy / H, val / MAX_HALITE, dist / max_dist


def extract_spatial_hlt(game: hlt.Game, ship_pos: Position, me) -> np.ndarray:
    gmap       = game.game_map
    W, H       = gmap.width, gmap.height
    half       = WINDOW_SIZE // 2
    max_d      = (W + H) / 2.0
    MAX_HALITE = constants.MAX_HALITE
    me_id      = me.id

    my_deposits = {me.shipyard.position}
    for d in me.get_dropoffs():
        my_deposits.add(d.position)

    my_struct_pos, opp_struct_pos = set(), set()
    my_struct_pos.add(me.shipyard.position)
    for d in me.get_dropoffs():
        my_struct_pos.add(d.position)
    for pid, player in game.players.items():
        if pid != me_id:
            opp_struct_pos.add(player.shipyard.position)
            for d in player.get_dropoffs():
                opp_struct_pos.add(d.position)

    my_ships_by_pos  = {s.position: s for s in me.get_ships()}
    opp_ships_by_pos = {}
    for pid, player in game.players.items():
        if pid != me_id:
            for s in player.get_ships():
                opp_ships_by_pos[s.position] = s

    _adj = [(0, 0), (0, -1), (0, 1), (1, 0), (-1, 0)]
    enemy_reachable, friendly_reachable = set(), set()
    enemy_positions, friendly_cargo_map = [], {}
    for pid2, player in game.players.items():
        for s in player.get_ships():
            ex, ey = s.position.x, s.position.y
            if pid2 != me_id:
                enemy_positions.append((ex, ey))
                for ddx, ddy in _adj:
                    enemy_reachable.add(Position((ex + ddx) % W, (ey + ddy) % H))
            else:
                cargo2 = s.halite_amount
                for ddx, ddy in _adj:
                    cell = ((ex + ddx) % W, (ey + ddy) % H)
                    friendly_cargo_map[cell] = friendly_cargo_map.get(cell, 0) + cargo2
                if s.position != ship_pos:
                    for ddx, ddy in _adj:
                        friendly_reachable.add(Position((ex + ddx) % W, (ey + ddy) % H))

    sx, sy  = ship_pos.x, ship_pos.y
    spatial = np.zeros((WINDOW_SIZE, WINDOW_SIZE, N_SPATIAL_CHANNELS), dtype=np.float32)
    for dy_off in range(-half, half + 1):
        for dx_off in range(-half, half + 1):
            cx, cy = (sx + dx_off) % W, (sy + dy_off) % H
            pos    = Position(cx, cy)
            wy, wx = dy_off + half, dx_off + half
            cell   = gmap[pos]
            spatial[wy, wx, 0] = cell.halite_amount / MAX_HALITE
            if pos in my_ships_by_pos:
                s = my_ships_by_pos[pos]
                spatial[wy, wx, 1] = 1.0
                spatial[wy, wx, 2] = s.halite_amount / MAX_HALITE
                if _inspired(pos, game, me_id):
                    spatial[wy, wx, 6] = 1.0
            elif pos in opp_ships_by_pos:
                s = opp_ships_by_pos[pos]
                spatial[wy, wx, 3] = 1.0
                spatial[wy, wx, 8] = s.halite_amount / MAX_HALITE
            if pos in my_struct_pos:
                spatial[wy, wx, 4] = 1.0 if pos == me.shipyard.position else 0.5
            elif pos in opp_struct_pos:
                spatial[wy, wx, 5] = 1.0
            min_d = min(gmap.calculate_distance(pos, dp) for dp in my_deposits)
            spatial[wy, wx, 7] = 1.0 - (min_d / max_d)
            if pos in enemy_reachable:
                spatial[wy, wx, 9] = 1.0
            if pos in friendly_reachable:
                spatial[wy, wx, 10] = 1.0
            mean3 = 0.0
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    mean3 += gmap[Position((cx + ox) % W, (cy + oy) % H)].halite_amount
            spatial[wy, wx, 11] = min(min_d / max_d, 1.0) * (mean3 / 9.0 / MAX_HALITE)
            ins = sum(1 for ex, ey in enemy_positions
                      if torus_dist(cx, cy, ex, ey, W, H) <= 4)
            spatial[wy, wx, 12] = min(ins / 2.0, 1.0)
            spatial[wy, wx, 13] = min(friendly_cargo_map.get((cx, cy), 0) / MAX_HALITE, 1.0)
    return spatial


def extract_scalars_hlt(game: hlt.Game, ship, me) -> np.ndarray:
    """The 29 BASE scalars (no FSM block)."""
    MAX_HALITE = constants.MAX_HALITE
    gmap       = game.game_map
    W, H       = gmap.width, gmap.height
    MAX_TURNS  = game_max_turns(W, H)
    me_id      = me.id

    sx, sy     = ship.position.x, ship.position.y
    cargo      = ship.halite_amount
    is_insp    = _inspired(ship.position, game, me_id)
    near       = _nearest_deposit(ship.position, game, me)
    dist_dep   = gmap.calculate_distance(ship.position, near)
    turns_left = MAX_TURNS - game.turn_number
    max_dist   = W + H
    ddx, ddy   = torus_delta(sx, sy, near.x, near.y, W, H)

    return_urgency = 1.0 if (turns_left <= dist_dep * 1.5 + 1 and cargo > 0) else 0.0
    turns_slack    = (turns_left - dist_dep) / max(1, MAX_TURNS)

    my_ships  = len(list(me.get_ships()))
    opp_ships = sum(len(list(p.get_ships())) for pid, p in game.players.items()
                    if pid != me_id)

    enemy_near = friendly_near = 0
    for pid2, player in game.players.items():
        for s in player.get_ships():
            d = gmap.calculate_distance(ship.position, s.position)
            if d <= 2:
                if pid2 != me_id:
                    enemy_near += 1
                elif s.id != ship.id:
                    friendly_near += 1

    cell_h     = gmap[ship.position].halite_amount
    mine_yield = math.ceil(cell_h / constants.EXTRACT_RATIO) if cell_h > 0 else 0
    my_bank    = me.halite_amount
    total_bank = sum(p.halite_amount for p in game.players.values())
    winning_ratio = my_bank / (total_bank + 1)
    endgame_flag  = 1.0 if turns_left < 50 else 0.0
    num_dropoffs  = len(list(me.get_dropoffs()))

    all_deposits = [me.shipyard.position] + [d.position for d in me.get_dropoffs()]
    my_ship_list = list(me.get_ships())
    if my_ship_list:
        avg_fleet_dist = sum(
            min(gmap.calculate_distance(s.position, dep) for dep in all_deposits)
            for s in my_ship_list) / len(my_ship_list)
    else:
        avg_fleet_dist = 0.0
    bank_can_afford_dropoff = 1.0 if my_bank >= constants.DROPOFF_COST else 0.0

    opp_bank = max((p.halite_amount for pid2, p in game.players.items()
                    if pid2 != me_id), default=0)
    bank_margin = math.tanh((my_bank - opp_bank) / 5000.0)
    dropoff_cost_here  = max(0, constants.DROPOFF_COST - (cell_h + cargo))
    dropoff_affordable = 1.0 if my_bank >= dropoff_cost_here else 0.0
    tgt = target_dropoffs(W, H)
    dropoff_slack = max(0.0, (tgt - num_dropoffs) / max(1, tgt))
    total_map_halite = sum(gmap[Position(x, y)].halite_amount
                           for x in range(W) for y in range(H))
    halite_frac = total_map_halite / (W * H * MAX_HALITE)

    return np.array([
        turns_left / MAX_TURNS, me.halite_amount / MAX_HALITE, cargo / MAX_HALITE,
        my_ships / 30.0, opp_ships / 30.0, float(is_insp), dist_dep / max_dist,
        ddx / W, ddy / H, return_urgency, turns_slack,
        enemy_near / 10.0, friendly_near / 10.0,
        *_prospect_scalars_hlt(sx, sy, gmap, W, H, max_dist, MAX_HALITE),
        cell_h / MAX_HALITE, mine_yield / MAX_HALITE, winning_ratio, endgame_flag,
        num_dropoffs / 5.0, avg_fleet_dist / max_dist, bank_can_afford_dropoff,
        bank_margin, opp_bank / MAX_HALITE, dropoff_affordable, dropoff_slack,
        halite_frac,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Masks & global spawn features
# ---------------------------------------------------------------------------

def action_mask_v6(game: hlt.Game, ship, me) -> np.ndarray:
    """bool[6]: moves 0-4 always legal; DROPOFF legal only when the engine would
    accept it (affordability / distance / cap / turns).  Engine-legality only —
    NOT a strategy rule."""
    gmap   = game.game_map
    cell_h = gmap[ship.position].halite_amount
    deposits = [me.shipyard.position] + [d.position for d in me.get_dropoffs()]
    dist = min(gmap.calculate_distance(ship.position, dp) for dp in deposits)
    turns_left = game_max_turns(gmap.width, gmap.height) - game.turn_number
    ok = dropoff_legal(me.halite_amount, cell_h, ship.halite_amount, dist,
                       len(list(me.get_dropoffs())), turns_left,
                       gmap[ship.position].has_structure)
    mask = np.ones(N_SHIP_ACTIONS_V6, dtype=bool)
    mask[ACTION_DROPOFF_V6] = bool(ok)
    return mask


def spawn_features_hlt(game: hlt.Game, me) -> np.ndarray:
    gmap   = game.game_map
    W, H   = gmap.width, gmap.height
    me_id  = me.id
    my_bank   = me.halite_amount
    opp_bank  = max((p.halite_amount for pid, p in game.players.items()
                     if pid != me_id), default=0)
    my_ships  = len(list(me.get_ships()))
    opp_ships = sum(len(list(p.get_ships())) for pid, p in game.players.items()
                    if pid != me_id)
    map_h = sum(gmap[Position(x, y)].halite_amount for x in range(W) for y in range(H))
    return spawn_global_features(game.turn_number, game_max_turns(W, H),
                                 my_bank, opp_bank, my_ships, opp_ships, map_h, W, H)


# ---------------------------------------------------------------------------
# Bot main
# ---------------------------------------------------------------------------

def main(model_path, spawn_path, device_str='cpu', deterministic=False,
         independent=False):
    device = torch.device(device_str)
    # load_expand tolerates an older 6-action checkpoint (HOME/PROSPECT rows start
    # fresh) as well as the current 8-action macro head.
    model = ActorCritic.load_expand(model_path, device=device_str,
                                    n_scalars=N_SCALARS_V6, n_actions=N_SHIP_ACTIONS_V6)
    model.eval()
    spawn_head = None
    if spawn_path and os.path.exists(spawn_path):
        spawn_head = SpawnHead.load(spawn_path, device=device_str)

    game = hlt.Game()
    game.ready("rl_v6")

    _dir_map = {'n': Direction.North, 's': Direction.South,
                'e': Direction.East,  'w': Direction.West}

    MAX_FLEET = 16
    last_spawn_turn = -10        # spawn cooldown: a freshly-spawned ship isn't
    SPAWN_COOLDOWN = 3           # visible via get_ships() for a turn, so space spawns
    _dir_delta = {ACTION_STAY: (0, 0), ACTION_NORTH: (0, -1), ACTION_SOUTH: (0, 1),
                  ACTION_EAST: (1, 0), ACTION_WEST: (-1, 0)}
    _cardinal = [(0, -1, ACTION_NORTH), (1, 0, ACTION_EAST),
                 (0, 1, ACTION_SOUTH), (-1, 0, ACTION_WEST)]

    while True:
        game.update_frame()
        me   = game.me
        gmap = game.game_map
        W, H = gmap.width, gmap.height
        max_turns  = game_max_turns(W, H)
        turns_left = max_turns - game.turn_number

        halite_dict = {(x, y): gmap[Position(x, y)].halite_amount
                       for x in range(W) for y in range(H)}
        fxy = (me.shipyard.position.x, me.shipyard.position.y)
        deposits = [fxy]
        deposits += [(d.position.x, d.position.y) for d in me.get_dropoffs()]
        cost_field = compute_home_cost_field(halite_dict, deposits, W, H)

        # 1) Net picks a macro per ship (FSM prior guides; net can override) ->
        #    resolved to a primitive via the reused navigation functions.
        ship_resolved = []
        for ship in me.get_ships():
            sx, sy = ship.position.x, ship.position.y
            cell_h = gmap[ship.position].halite_amount
            cargo  = ship.halite_amount
            spatial = extract_spatial_hlt(game, ship.position, me)
            scalars = extract_scalars_hlt(game, ship, me)
            mask    = action_mask_v6(game, ship, me)
            mask[ACTION_DROPOFF_V6] = False    # dropoffs disabled for now
            sp_t = torch.from_numpy(spatial).to(device)
            sc_t = torch.from_numpy(scalars).to(device)
            dist_dep = min(torus_dist(sx, sy, dx, dy, W, H) for dx, dy in deposits)
            prior = macro_prior(cargo, cell_h, dist_dep, turns_left)
            if deterministic:
                a = model.greedy_action(sp_t, sc_t, mask=mask, prior_bonus=prior)
            else:
                a, _, _ = model.select_action(sp_t, sc_t, mask=mask, prior_bonus=prior)
            prim = resolve_macro(a, sx, sy, halite_dict, cost_field, W, H)
            ship_resolved.append((ship, prim))

        # 2) rl_v5's proven 4-phase collision cascade (ported) -----------------
        # Phase 1: enemy threat zone = enemy cells + their neighbours.
        enemy_threat = set()
        for pid, pl in game.players.items():
            if pid != me.id:
                for s in pl.get_ships():
                    ex, ey = s.position.x, s.position.y
                    enemy_threat.add((ex, ey))
                    for ddx, ddy, _ in _cardinal:
                        enemy_threat.add(((ex + ddx) % W, (ey + ddy) % H))
        # Phase 2: initial destinations with move-affordability.  Move cost depends
        # ONLY on the ship's current cell, so a ship can afford ALL cardinal moves
        # or NONE -> precompute can_move and never assign a move to a ship that
        # can't move (else the engine keeps it put while we free its cell -> wreck).
        ov = {}; dest_of = {}; pos_of = {}; can_move = {}
        for ship, act in ship_resolved:
            afford = ship.halite_amount >= gmap[ship.position].halite_amount // 10
            can_move[ship.id] = afford
            if act != ACTION_STAY and not afford:
                act = ACTION_STAY
            ov[ship.id] = act
            ddx, ddy = _dir_delta[act]
            dest_of[ship.id] = ((ship.position.x + ddx) % W, (ship.position.y + ddy) % H)
            pos_of[ship.id]  = (ship.position.x, ship.position.y)
        # Phase 3a: never move into the enemy threat zone.
        for ship, _ in ship_resolved:
            if dest_of[ship.id] in enemy_threat:
                ov[ship.id] = ACTION_STAY; dest_of[ship.id] = pos_of[ship.id]
        # Phase 3b: escape if currently sitting in the threat zone (only if the
        # ship can actually afford to move).
        for ship, _ in ship_resolved:
            if (ov[ship.id] == ACTION_STAY and can_move[ship.id]
                    and pos_of[ship.id] in enemy_threat):
                for ddx, ddy, esc in _cardinal:
                    e = ((ship.position.x + ddx) % W, (ship.position.y + ddy) % H)
                    if e not in enemy_threat:
                        ov[ship.id] = esc; dest_of[ship.id] = e; break
        # Phase 3c: reserve the factory cell when spawning.
        want_spawn = (me.halite_amount >= SHIP_COST and turns_left > 80
                      and len(ship_resolved) < MAX_FLEET)
        if want_spawn:
            for ship, _ in ship_resolved:
                if dest_of[ship.id] == fxy and pos_of[ship.id] != fxy:
                    ov[ship.id] = ACTION_STAY; dest_of[ship.id] = pos_of[ship.id]
        # Phase 4: friendly collision cascade, iterated to a fixpoint.  Resolves
        #  (a) OCCUPANCY/AFFORDABILITY: a ship may move onto a cell occupied by
        #      another ship ONLY if that ship is itself moving AND can CERTAINLY
        #      afford to leave.  Max move cost is MAX_HALITE//10 = 100, so a ship
        #      with cargo >= 100 affords ANY move; one with cargo < 100 MIGHT be
        #      stuck on a rich cell (the engine then keeps it put and the follower
        #      wrecks on it — the real bug).  Cargo is always reliable (unlike the
        #      occasionally-stale gmap halite), so this is robust.  Covers swaps too.
        #  (b) SHARED DESTINATION: stayer owns the cell, else heaviest mover wins.
        SAFE_CARGO = constants.MAX_HALITE // 10
        cargo_of = {s.id: s.halite_amount for s, _ in ship_resolved}
        for _ in range(len(ship_resolved) + 1):
            changed = False
            pos_to_ship = {pos_of[s.id]: s.id for s, _ in ship_resolved}
            # (a) occupancy / affordability
            for ship, _ in ship_resolved:
                sid = ship.id; d = dest_of[sid]
                if d != pos_of[sid] and d in pos_to_ship:
                    o = pos_to_ship[d]
                    safe = (o == sid) or (dest_of[o] != pos_of[o] and cargo_of[o] >= SAFE_CARGO)
                    if not safe:
                        ov[sid] = ACTION_STAY; dest_of[sid] = pos_of[sid]; changed = True
            # (b) shared-destination resolution
            occ = {}
            for ship, _ in ship_resolved:
                occ.setdefault(dest_of[ship.id], []).append((cargo_of[ship.id], ship.id))
            for d, occs in occ.items():
                if len(occs) <= 1:
                    continue
                stayers = [(c, sid) for c, sid in occs if d == pos_of[sid]]
                movers  = [(c, sid) for c, sid in occs if d != pos_of[sid]]
                if stayers:
                    for _, sid in movers:
                        ov[sid] = ACTION_STAY; dest_of[sid] = pos_of[sid]; changed = True
                elif len(movers) > 1:
                    movers.sort(reverse=True)
                    for _, sid in movers[1:]:
                        ov[sid] = ACTION_STAY; dest_of[sid] = pos_of[sid]; changed = True
            if not changed:
                break
        # (The iterative cascade above is rl_v5's proven, collision-free resolver for
        # all VISIBLE ships; ghost cells handle the invisible ones.  No extra
        # single-pass dedup — that re-introduced an affordability/chain race.)

        # 3) Issue commands + spawn (only if the factory ends free).
        commands = []
        for ship, _ in ship_resolved:
            d = ACTION_TO_DIR[ov[ship.id]]
            commands.append(ship.stay_still() if d == 'o' else ship.move(_dir_map[d]))
        # Spawn only when the factory AND its 4 neighbours are clear of friendly
        # ships, and no ship is heading onto the factory.  The engine processes
        # spawn LAST, so a ship that fails to vacate the factory (or a neighbour
        # that slides on) collides with the new ship — this guard removes all such
        # cases at the cost of an occasional skipped spawn.
        factory_area = {fxy}
        for ddx, ddy, _ in _cardinal:
            factory_area.add(((fxy[0] + ddx) % W, (fxy[1] + ddy) % H))
        ship_cells = {pos_of[s.id] for s, _ in ship_resolved}
        dest_cells = {dest_of[s.id] for s, _ in ship_resolved}
        factory_clear = (not (ship_cells & factory_area)) and (fxy not in dest_cells)
        cooled = (game.turn_number - last_spawn_turn) >= SPAWN_COOLDOWN
        spawned = want_spawn and factory_clear and cooled
        if spawned:
            commands.append(me.shipyard.spawn())
            last_spawn_turn = game.turn_number

        game.end_turn(commands)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Halite III rl_v6 pure-RL bot')
    _dm = os.path.join(_HERE, 'checkpoints', 'model_weights.pt')
    _ds = os.path.join(_HERE, 'checkpoints', 'spawn_weights.pt')
    parser.add_argument('--model',         default=_dm)
    parser.add_argument('--spawn-model',   default=_ds, dest='spawn_model')
    parser.add_argument('--device',        default='cpu')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--independent',   action='store_true',
                        help='disable sequential collision-aware decode (A/B baseline)')
    args = parser.parse_args()
    main(args.model, args.spawn_model, args.device, args.deterministic,
         args.independent)
