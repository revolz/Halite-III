#!/usr/bin/env python3
"""
Halite III RL Bot

Loads a trained ActorCritic model and plays Halite III using the standard
hlt protocol.  Communicates with the engine via stdin/stdout, exactly like
any other starter-kit bot.

Behaviour
---------
Each turn the bot:
  1. Runs the neural network for every owned ship to get a raw action index.
  2. Applies home-memory override: ships with cargo ≥ 60 % of MAX_HALITE are
     committed to returning home until they deposit (cargo drops to 0).
  3. Resolves meta-actions (RANDOM → random primitive, HOME → step toward
     nearest deposit).
  4. Runs 4-phase collision prevention:
       Phase 1 – build enemy threat zone (current cell + 4 neighbours per enemy)
       Phase 2 – compute each ship's destination
       Phase 3a – ships moving INTO the threat zone are forced STAY
       Phase 3b – ships already AT a threat-zone cell escape sideways to the
                  first safe adjacent cell (avoids sitting still while enemy
                  walks in)
       Phase 4  – friendly cascade: stayers own their cell; movers yield;
                  heaviest mover wins multi-mover contests; iterated until stable
  5. Spawns a new ship if affordable and the spawn guard is clear (no friendly
     ship at the shipyard or any adjacent cell).

Usage
-----
    # Run via the engine or run_game.py:
    python rl_bot.py --model checkpoints_v9/model_final_weights.pt

    # Register with run_game.py (from repo root):
    python my_extension/run_game.py --bot "python my_extension/rl_v1/rl_bot.py --model my_extension/rl_v1/checkpoints_v9/model_final_weights.pt"
"""

import argparse
import math
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))   # rl_v1/
_MY_EXT    = os.path.dirname(_HERE)                        # my_extension/
_REPO_ROOT = os.path.dirname(_MY_EXT)                      # repo root
sys.path.insert(0, _HERE)
sys.path.insert(0, _MY_EXT)                                # halite_engine.py
sys.path.insert(0, os.path.join(_REPO_ROOT, 'starter_kits', 'Python3'))  # hlt

import torch
from rl_model    import ActorCritic
from rl_features import (
    WINDOW_SIZE, N_SPATIAL_CHANNELS, N_SCALAR_FEATURES, N_SHIP_ACTIONS,
    ACTION_TO_DIR, DIR_TO_ACTION, torus_dist, torus_delta,
    ACTION_STAY, ACTION_NORTH, ACTION_SOUTH, ACTION_EAST, ACTION_WEST,
    ACTION_RANDOM, ACTION_HOME,
)

import hlt
from hlt                 import constants
from hlt.positionals     import Direction, Position


# ---------------------------------------------------------------------------
# Feature extraction from hlt API objects
# ---------------------------------------------------------------------------

def _inspired(position: Position, game: hlt.Game, me_id: int) -> bool:
    """Compute inspiration for a ship at `position` using the hlt API."""
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
    """Return the position of the nearest factory or dropoff owned by `me`."""
    candidates = [me.shipyard.position] + [d.position for d in me.get_dropoffs()]
    return min(candidates, key=lambda p: game.game_map.calculate_distance(position, p))


def extract_spatial_hlt(
    game: hlt.Game,
    ship_pos: Position,
    me,
) -> np.ndarray:
    """
    Build a WINDOW_SIZE × WINDOW_SIZE × N_SPATIAL_CHANNELS float32 array
    centred on the ship, using hlt API objects.
    """
    gmap       = game.game_map
    W, H       = gmap.width, gmap.height
    half       = WINDOW_SIZE // 2
    max_d      = (W + H) / 2.0
    MAX_HALITE = constants.MAX_HALITE
    me_id      = me.id

    my_deposits = {me.shipyard.position}
    for d in me.get_dropoffs():
        my_deposits.add(d.position)

    # Build structure lookup: position → (is_mine, is_factory)
    my_struct_pos = set()
    opp_struct_pos = set()
    my_struct_pos.add(me.shipyard.position)
    for d in me.get_dropoffs():
        my_struct_pos.add(d.position)
    for pid, player in game.players.items():
        if pid != me_id:
            opp_struct_pos.add(player.shipyard.position)
            for d in player.get_dropoffs():
                opp_struct_pos.add(d.position)

    # Ship lookups
    my_ships_by_pos   = {s.position: s for s in me.get_ships()}
    opp_ships_by_pos  = {}
    for pid, player in game.players.items():
        if pid != me_id:
            for s in player.get_ships():
                opp_ships_by_pos[s.position] = s

    # Pre-compute 1-step reachable sets for danger-zone channels
    _adj = [(0, 0), (0, -1), (0, 1), (1, 0), (-1, 0)]

    enemy_reachable: set = set()
    friendly_reachable: set = set()

    for pid2, player in game.players.items():
        for s in player.get_ships():
            ex, ey = s.position.x, s.position.y
            if pid2 != me_id:
                for ddx, ddy in _adj:
                    enemy_reachable.add(Position((ex + ddx) % W, (ey + ddy) % H))
            elif s.position != ship_pos:
                for ddx, ddy in _adj:
                    friendly_reachable.add(Position((ex + ddx) % W, (ey + ddy) % H))

    sx, sy    = ship_pos.x, ship_pos.y
    spatial   = np.zeros((WINDOW_SIZE, WINDOW_SIZE, N_SPATIAL_CHANNELS), dtype=np.float32)

    for dy_off in range(-half, half + 1):
        for dx_off in range(-half, half + 1):
            cx  = (sx + dx_off) % W
            cy  = (sy + dy_off) % H
            pos = Position(cx, cy)
            wy  = dy_off + half
            wx  = dx_off + half

            cell = gmap[pos]
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
                # Ch 8: enemy cargo (0=kamikaze threat, 1=wants to go home)
                spatial[wy, wx, 8] = s.halite_amount / MAX_HALITE

            if pos in my_struct_pos:
                spatial[wy, wx, 4] = 1.0 if pos == me.shipyard.position else 0.5
            elif pos in opp_struct_pos:
                spatial[wy, wx, 5] = 1.0

            # Per-cell distance to nearest own deposit
            min_d = min(gmap.calculate_distance(pos, dp) for dp in my_deposits)
            spatial[wy, wx, 7] = 1.0 - (min_d / max_d)

            if pos in enemy_reachable:
                spatial[wy, wx, 9] = 1.0
            if pos in friendly_reachable:
                spatial[wy, wx, 10] = 1.0

    return spatial


def extract_scalars_hlt(
    game: hlt.Game,
    ship,
    me,
) -> np.ndarray:
    """
    Build an N_SCALAR_FEATURES float32 scalar vector using hlt API objects.
    """
    MAX_HALITE = constants.MAX_HALITE
    MAX_TURNS  = constants.MAX_TURNS
    gmap       = game.game_map
    W, H       = gmap.width, gmap.height
    me_id      = me.id

    sx, sy     = ship.position.x, ship.position.y
    cargo      = ship.halite_amount
    is_insp    = _inspired(ship.position, game, me_id)
    near       = _nearest_deposit(ship.position, game, me)
    dist_dep   = gmap.calculate_distance(ship.position, near)
    turns_left = MAX_TURNS - game.turn_number
    max_dist   = W + H

    # Toroidal delta toward nearest deposit
    ddx, ddy   = torus_delta(sx, sy, near.x, near.y, W, H)

    return_urgency = 1.0 if (turns_left <= dist_dep * 1.5 + 1 and cargo > 0) else 0.0
    turns_slack    = (turns_left - dist_dep) / max(1, MAX_TURNS)

    my_ships   = len(list(me.get_ships()))
    opp_ships  = sum(len(list(p.get_ships())) for pid, p in game.players.items()
                     if pid != me_id)

    # Proximity danger: count ships within 2 steps
    enemy_near    = 0
    friendly_near = 0
    for pid2, player in game.players.items():
        for s in player.get_ships():
            d = gmap.calculate_distance(ship.position, s.position)
            if d <= 2:
                if pid2 != me_id:
                    enemy_near += 1
                elif s.id != ship.id:
                    friendly_near += 1

    return np.array([
        turns_left / MAX_TURNS,
        me.halite_amount / MAX_HALITE,
        cargo      / MAX_HALITE,
        my_ships   / 30.0,
        opp_ships  / 30.0,
        float(is_insp),
        dist_dep   / max_dist,
        ddx / W,
        ddy / H,
        return_urgency,
        turns_slack,
        enemy_near    / 10.0,
        friendly_near / 10.0,
    ], dtype=np.float32)


def _home_dir_hlt(ship, game: hlt.Game, me) -> int:
    """Return the primitive action index (0–4) that moves one step toward
    the nearest deposit structure (shipyard or dropoff)."""
    gmap   = game.game_map
    W, H   = gmap.width, gmap.height
    near   = _nearest_deposit(ship.position, game, me)
    sx, sy = ship.position.x, ship.position.y
    if ship.position == near:
        return ACTION_STAY
    dx, dy = torus_delta(sx, sy, near.x, near.y, W, H)
    if abs(dx) >= abs(dy):
        return ACTION_EAST if dx > 0 else ACTION_WEST
    else:
        return ACTION_NORTH if dy < 0 else ACTION_SOUTH




def _should_spawn(game: hlt.Game, me, max_fleet: int = 12) -> bool:
    """Heuristic: spawn if affordable, fleet is small, and not too late."""
    turns_left = constants.MAX_TURNS - game.turn_number
    n_ships    = len(list(me.get_ships()))
    if not (me.halite_amount >= constants.SHIP_COST
            and n_ships < max_fleet
            and turns_left > 75
            and not game.game_map[me.shipyard].is_occupied):
        return False
    # Don't spawn if a friendly ship is adjacent to the shipyard —
    # it may move in this turn and collide with the spawned ship.
    sy = me.shipyard.position
    my_positions = {s.position for s in me.get_ships()}
    adjacent = {sy.directional_offset(d)
                for d in [Direction.North, Direction.South,
                          Direction.East, Direction.West]}
    return not (my_positions & adjacent)


# ---------------------------------------------------------------------------
# Bot main
# ---------------------------------------------------------------------------

def main(model_path: str, device_str: str = 'cpu', deterministic: bool = False):
    device = torch.device(device_str)

    # Load model
    model = ActorCritic.load(model_path, device=device_str)
    model.eval()

    # Initialise the Halite game
    game = hlt.Game()
    game.ready("RLBot")

    # Ships currently committed to going home (mirrors rl_env home memory)
    homing_ships: set = set()

    while True:
        game.update_frame()
        me      = game.me
        gmap    = game.game_map

        # First pass: resolve all ships' actions and destinations
        ship_resolved: list = []   # (ship, action_idx)
        for ship in me.get_ships():
            spatial = extract_spatial_hlt(game, ship.position, me)
            scalars = extract_scalars_hlt(game, ship, me)

            sp_t = torch.from_numpy(spatial).to(device)
            sc_t = torch.from_numpy(scalars).to(device)

            if deterministic:
                action_idx = model.greedy_action(sp_t, sc_t)
            else:
                action_idx, _, _ = model.select_action(sp_t, sc_t)

            # Auto-trigger home if cargo exceeds 60% capacity
            if ship.halite_amount >= constants.MAX_HALITE * 0.6:
                homing_ships.add(ship.id)
            # Home memory: if this ship committed to going home, keep it going
            if ship.id in homing_ships:
                action_idx = ACTION_HOME

            # Resolve meta-actions to primitives
            if action_idx == ACTION_RANDOM:
                action_idx = random.randint(0, 4)
            elif action_idx == ACTION_HOME:
                homing_ships.add(ship.id)
                action_idx = _home_dir_hlt(ship, game, me)
                if action_idx == ACTION_STAY:   # arrived at deposit — cancel home mode
                    homing_ships.discard(ship.id)

            ship_resolved.append((ship, action_idx))

        # ── Safety override: 4-phase collision prevention ────────────────────
        W, H = gmap.width, gmap.height
        _dir_delta = {
            ACTION_STAY:  (0, 0),  ACTION_NORTH: (0, -1), ACTION_SOUTH: (0, 1),
            ACTION_EAST:  (1, 0),  ACTION_WEST:  (-1, 0),
        }
        # Cardinal deltas paired with their action index.
        _cardinal = [(0, -1, ACTION_NORTH), (1, 0, ACTION_EAST),
                     (0,  1, ACTION_SOUTH), (-1, 0, ACTION_WEST)]

        # Phase 1 — Build enemy threat zone.
        # Covers "enemy stays" AND "enemy moves adjacent": current + 4 neighbours.
        enemy_threat_zone: set = set()
        for pid, player in game.players.items():
            if pid != me.id:
                for s in player.get_ships():
                    ex, ey = s.position.x, s.position.y
                    enemy_threat_zone.add(Position(ex, ey))
                    for ddx, ddy, _ in _cardinal:
                        enemy_threat_zone.add(Position((ex+ddx)%W, (ey+ddy)%H))

        # Phase 2 — Compute initial destinations (with move-affordability check).
        # The engine silently ignores any move a ship cannot afford
        # (cargo < cell_halite // 10).  Mirror this here so that planned
        # destinations reflect what the engine will actually execute.
        overridden_act: dict = {}
        dest_of: dict = {}
        for ship, act in ship_resolved:
            if act != ACTION_STAY:
                cell_h = gmap[ship.position].halite_amount
                if ship.halite_amount < cell_h // 10:
                    act = ACTION_STAY
            overridden_act[ship.id] = act
            ddx, ddy = _dir_delta[act]
            dest_of[ship.id] = Position((ship.position.x+ddx)%W, (ship.position.y+ddy)%H)

        # Phase 3a — Enemy avoidance (MOVE): ships heading into threat zone → STAY.
        for ship, _ in ship_resolved:
            if dest_of[ship.id] in enemy_threat_zone:
                overridden_act[ship.id] = ACTION_STAY
                dest_of[ship.id] = ship.position

        # Phase 3b — Enemy avoidance (ESCAPE): a ship already at a threat-zone cell
        # risks collision if an adjacent enemy moves onto it while the ship stays.
        # Try to move to any safe adjacent cell rather than sitting in the danger zone.
        ship_pos = {ship.id: ship.position for ship, _ in ship_resolved}
        for ship, _ in ship_resolved:
            if overridden_act[ship.id] == ACTION_STAY and ship.position in enemy_threat_zone:
                for ddx, ddy, esc_act in _cardinal:
                    esc = Position((ship.position.x+ddx)%W, (ship.position.y+ddy)%H)
                    if esc not in enemy_threat_zone:
                        overridden_act[ship.id] = esc_act
                        dest_of[ship.id] = esc
                        break   # first safe direction wins; cascade handles friendlies

        # Phase 4 — Friendly collision resolution: no two ships may share a final cell.
        # Rules (applied via cascade until stable):
        #   Stayer (dest == current_pos) owns that cell — every mover is forced STAY.
        #   Multiple movers to same empty cell → heaviest mover wins, rest STAY.
        for _ in range(len(ship_resolved)):
            pos_occupants: dict = {}
            for ship, _ in ship_resolved:
                pos_occupants.setdefault(dest_of[ship.id], []).append(
                    (ship.halite_amount, ship.id))
            changed = False
            for dest, occupants in pos_occupants.items():
                if len(occupants) <= 1:
                    continue
                stayers = [(c, sid) for c, sid in occupants if dest == ship_pos[sid]]
                movers  = [(c, sid) for c, sid in occupants if dest != ship_pos[sid]]
                if stayers:
                    for _, sid in movers:
                        overridden_act[sid] = ACTION_STAY
                        dest_of[sid] = ship_pos[sid]
                        changed = True
                elif len(movers) > 1:
                    movers.sort(reverse=True)
                    for _, sid in movers[1:]:
                        overridden_act[sid] = ACTION_STAY
                        dest_of[sid] = ship_pos[sid]
                        changed = True
            if not changed:
                break

        commands = []
        dir_map = {'n': Direction.North, 's': Direction.South,
                   'e': Direction.East,  'w': Direction.West}
        for ship, _ in ship_resolved:
            action_idx = overridden_act[ship.id]
            direction_str = ACTION_TO_DIR[action_idx]
            if direction_str == 'o':
                commands.append(ship.stay_still())
            else:
                commands.append(ship.move(dir_map[direction_str]))

        if _should_spawn(game, me):
            commands.append(me.shipyard.spawn())

        game.end_turn(commands)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Halite III RL bot')
    parser.add_argument('--model',         required=True,        help='path to model .pt file')
    parser.add_argument('--device',        default='cpu',        help='torch device (cpu or cuda)')
    parser.add_argument('--deterministic', action='store_true',  help='greedy action selection')
    args = parser.parse_args()

    main(args.model, args.device, args.deterministic)
