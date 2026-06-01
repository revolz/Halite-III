"""
Feature extraction for Halite III RL training and inference.

Provides observation extraction from engine internal state (for training)
and a replay-state dict (for imitation learning data collection).

Observation layout
------------------
Spatial : float32[WINDOW_SIZE, WINDOW_SIZE, N_SPATIAL_CHANNELS]
  Window is 5×5 (2-cell radius) centred on the observing ship.
  Ch 0 – cell halite / MAX_HALITE
  Ch 1 – my ship present (binary)
  Ch 2 – my ship cargo / MAX_HALITE
  Ch 3 – opponent ship (binary)
  Ch 4 – my structures: 1.0 = factory, 0.5 = dropoff
  Ch 5 – opponent structures (binary)
  Ch 6 – inspired my-ship (binary)
  Ch 7 – 1 − dist_cell_to_nearest_deposit / max_dist   (per-cell)
  Ch 8 – enemy ship cargo / MAX_HALITE (0 if no enemy here; 0 = kamikaze, 1 = safe)
  Ch 9 – enemy danger zone: 1 if any enemy ship is ≤1 step from this cell
  Ch 10 – friendly danger zone: 1 if any OTHER friendly ship is ≤1 step from this cell

Scalars : float32[N_SCALAR_FEATURES]
  0  – turns remaining / max_turns
  1  – my bank / MAX_HALITE
  2  – ship cargo / MAX_HALITE
  3  – my fleet size / 30
  4  – opponent fleet size / 30
  5  – is_inspired (binary)
  6  – dist to nearest deposit / (W + H)
  7  – toroidal dx to deposit / W
  8  – toroidal dy to deposit / H
  9  – return-urgency flag (1 if turns_left ≤ dist × 1.5 + 1)
  10 – turns slack = (turns_left − dist_deposit) / max_turns
  11 – enemy ships within 2 steps / 10
  12 – other friendly ships within 2 steps / 10
  13 – toroidal dx to richest cell in PROSPECT window / W
  14 – toroidal dy to richest cell in PROSPECT window / H
  15 – richest cell halite in PROSPECT window / MAX_HALITE
  16 – dist to richest cell in PROSPECT window / (W + H)

Actions
-------
  0 STAY    – stay and mine
  1 NORTH   – move north
  2 EAST    – move east
  3 SOUTH   – move south
  4 WEST    – move west
  5 RANDOM  – meta: environment picks a random primitive action (0–4)
  6 HOME    – meta: environment moves one step toward nearest deposit
  7 PROSPECT – meta: environment moves one step toward richest cell in 11×11 window;
               resolves to STAY when already at the richest cell (mine it)
"""

import numpy as np
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Observation shape constants
# ---------------------------------------------------------------------------

WINDOW_SIZE        = 5
N_SPATIAL_CHANNELS = 11
N_SCALAR_FEATURES  = 17

# Radius for PROSPECT window scan: (2*PROSPECT_RADIUS+1) × (2*PROSPECT_RADIUS+1) = 11×11
PROSPECT_RADIUS = 5

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

ACTION_STAY    = 0
ACTION_NORTH   = 1
ACTION_EAST    = 2
ACTION_SOUTH   = 3
ACTION_WEST    = 4
ACTION_RANDOM  = 5   # meta: resolved to random primitive at environment level
ACTION_HOME    = 6   # meta: resolved to one step toward nearest deposit
ACTION_PROSPECT = 7  # meta: resolved to one step toward richest cell in 11×11 window

N_SHIP_ACTIONS = 8

DIR_TO_ACTION = {
    'o': ACTION_STAY,
    'n': ACTION_NORTH,
    'e': ACTION_EAST,
    's': ACTION_SOUTH,
    'w': ACTION_WEST,
}
ACTION_TO_DIR = {v: k for k, v in DIR_TO_ACTION.items()}

# ---------------------------------------------------------------------------
# Toroidal geometry helpers
# ---------------------------------------------------------------------------

def torus_dist(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> int:
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return min(dx, W - dx) + min(dy, H - dy)


def torus_delta(sx: int, sy: int, tx: int, ty: int, W: int, H: int) -> Tuple[int, int]:
    """Signed toroidal vector from (sx, sy) to (tx, ty)."""
    dx = tx - sx
    dy = ty - sy
    if abs(dx) > W // 2:
        dx = dx - W if dx > 0 else dx + W
    if abs(dy) > H // 2:
        dy = dy - H if dy > 0 else dy + H
    return dx, dy


# ---------------------------------------------------------------------------
# Engine-state observation extraction
# ---------------------------------------------------------------------------

def _nearest_deposit(sx: int, sy: int, engine, pid: int) -> Tuple[int, int]:
    candidates = [engine.players[pid]['factory']]
    for _did, dx, dy in engine.players[pid]['dropoffs']:
        candidates.append((dx, dy))
    return min(candidates,
               key=lambda d: torus_dist(sx, sy, d[0], d[1], engine.width, engine.height))


def _richest_in_prospect_window(
    sx: int, sy: int,
    halite_dict: dict,
    W: int, H: int,
    radius: int = PROSPECT_RADIUS,
) -> Tuple[int, int, int]:
    """Return (rx, ry, halite_val) for the richest cell in the prospect window.

    The window is (2*radius+1)×(2*radius+1) centred on (sx, sy), wrapping
    toroidally.  The ship's current cell is included so that PROSPECT → STAY
    fires naturally when the ship is already on the local maximum.

    Tie-breaking order: highest halite → current cell preferred → shorter
    toroidal distance → lower x → lower y (stable, deterministic).
    """
    best_val  = -1
    best_pos  = (sx, sy)
    best_dist = 0

    for dy_off in range(-radius, radius + 1):
        for dx_off in range(-radius, radius + 1):
            cx = (sx + dx_off) % W
            cy = (sy + dy_off) % H
            val  = halite_dict.get((cx, cy), 0)
            dist = abs(dx_off) + abs(dy_off)   # Manhattan within window (not toroidal)

            # Tie-breaking: prefer higher value, then current cell, then closer, then coord
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


def extract_spatial_from_engine(engine, ship_id: int, pid: int = 0) -> np.ndarray:
    """
    Build a WINDOW_SIZE × WINDOW_SIZE × N_SPATIAL_CHANNELS spatial observation
    centred on the given ship.  Channel layout described in module docstring.
    """
    from halite_engine import MAX_HALITE
    W, H   = engine.width, engine.height
    half   = WINDOW_SIZE // 2
    max_d  = (W + H) / 2.0

    sx, sy = engine.player_entities[pid][ship_id]

    # Quick position → (pid, ship_id) lookup
    pos_to_agent: Dict[Tuple, Tuple] = {}
    for p in engine.player_entities:
        for sid2, pos in engine.player_entities[p].items():
            pos_to_agent[pos] = (p, sid2)

    my_factory  = engine.players[pid]['factory']
    my_deposits = {my_factory}
    for _did, dx, dy in engine.players[pid]['dropoffs']:
        my_deposits.add((dx, dy))

    # Pre-compute 1-step reachable sets for danger-zone channels.
    # For each ship, its current position and all 4 adjacent cells form the set
    # of cells it could occupy after one move (including staying).
    _adj = [(0, 0), (0, -1), (0, 1), (1, 0), (-1, 0)]  # stay + N/S/E/W

    enemy_reachable: set = set()   # Ch 9: cells any enemy could reach in 1 step
    friendly_reachable: set = set()  # Ch 10: cells any OTHER friendly could reach

    for p in engine.player_entities:
        for sid2, (ex, ey) in engine.player_entities[p].items():
            if p != pid:
                for ddx, ddy in _adj:
                    enemy_reachable.add(((ex + ddx) % W, (ey + ddy) % H))
            elif sid2 != ship_id:
                for ddx, ddy in _adj:
                    friendly_reachable.add(((ex + ddx) % W, (ey + ddy) % H))

    spatial = np.zeros((WINDOW_SIZE, WINDOW_SIZE, N_SPATIAL_CHANNELS), dtype=np.float32)

    for dy_off in range(-half, half + 1):
        for dx_off in range(-half, half + 1):
            cx = (sx + dx_off) % W
            cy = (sy + dy_off) % H
            wy = dy_off + half
            wx = dx_off + half

            h = engine.halite.get((cx, cy), 0)
            spatial[wy, wx, 0] = h / MAX_HALITE

            if (cx, cy) in pos_to_agent:
                p2, sid2 = pos_to_agent[(cx, cy)]
                if p2 == pid:
                    spatial[wy, wx, 1] = 1.0
                    spatial[wy, wx, 2] = engine.entities[sid2]['cargo'] / MAX_HALITE
                    if engine.entities[sid2]['is_inspired']:
                        spatial[wy, wx, 6] = 1.0
                else:
                    spatial[wy, wx, 3] = 1.0
                    # Ch 8: enemy cargo tells how dangerous this enemy is
                    spatial[wy, wx, 8] = engine.entities[sid2]['cargo'] / MAX_HALITE

            cell_owner = engine.cell_owner.get((cx, cy))
            if cell_owner == pid:
                spatial[wy, wx, 4] = 1.0 if (cx, cy) == my_factory else 0.5
            elif cell_owner is not None:
                spatial[wy, wx, 5] = 1.0

            # Per-cell distance to nearest deposit
            min_d = min(torus_dist(cx, cy, fx, fy, W, H) for fx, fy in my_deposits)
            spatial[wy, wx, 7] = 1.0 - (min_d / max_d)

            # Ch 9/10: danger zones — cells reachable in 1 step by enemy/friendly
            if (cx, cy) in enemy_reachable:
                spatial[wy, wx, 9] = 1.0
            if (cx, cy) in friendly_reachable:
                spatial[wy, wx, 10] = 1.0

    return spatial


def extract_scalars_from_engine(engine, ship_id: int, pid: int = 0) -> np.ndarray:
    """
    Build an N_SCALAR_FEATURES float32 scalar feature vector for the given ship.
    Scalar layout described in module docstring.
    """
    from halite_engine import MAX_HALITE
    W, H = engine.width, engine.height

    sx, sy       = engine.player_entities[pid][ship_id]
    cargo        = engine.entities[ship_id]['cargo']
    is_inspired  = engine.entities[ship_id]['is_inspired']

    near         = _nearest_deposit(sx, sy, engine, pid)
    dist_dep     = torus_dist(sx, sy, near[0], near[1], W, H)
    ddx, ddy     = torus_delta(sx, sy, near[0], near[1], W, H)
    max_dist     = W + H
    turns_left   = engine.max_turns - engine.turn

    return_urgency = 1.0 if (turns_left <= dist_dep * 1.5 + 1 and cargo > 0) else 0.0
    turns_slack    = (turns_left - dist_dep) / max(1, engine.max_turns)

    my_ships   = len(engine.player_entities[pid])
    opp_ships  = sum(len(engine.player_entities[p])
                     for p in range(engine.num_players) if p != pid)

    # Proximity danger scalars: count ships within 2 toroidal steps
    enemy_near   = 0
    friendly_near = 0
    for p in engine.player_entities:
        for sid2, (ex, ey) in engine.player_entities[p].items():
            d = torus_dist(sx, sy, ex, ey, W, H)
            if d <= 2:
                if p != pid:
                    enemy_near += 1
                elif sid2 != ship_id:
                    friendly_near += 1

    return np.array([
        turns_left / engine.max_turns,
        engine.players[pid]['energy'] / MAX_HALITE,
        cargo / MAX_HALITE,
        my_ships  / 30.0,
        opp_ships / 30.0,
        float(is_inspired),
        dist_dep  / max_dist,
        ddx / W,
        ddy / H,
        return_urgency,
        turns_slack,
        enemy_near   / 10.0,
        friendly_near / 10.0,
        # Prospect features (indices 13–16)
        *_prospect_scalars_engine(sx, sy, engine, W, H, max_dist, MAX_HALITE),
    ], dtype=np.float32)


def _prospect_scalars_engine(
    sx: int, sy: int, engine, W: int, H: int, max_dist: float, MAX_HALITE: int,
) -> Tuple[float, float, float, float]:
    """Return (dx/W, dy/H, val/MAX_HALITE, dist/max_dist) for the richest cell
    in the PROSPECT window.  All four are 0 when the window is empty."""
    rx, ry, val = _richest_in_prospect_window(sx, sy, engine.halite, W, H)
    dx, dy = torus_delta(sx, sy, rx, ry, W, H)
    dist   = torus_dist(sx, sy, rx, ry, W, H)
    return dx / W, dy / H, val / MAX_HALITE, dist / max_dist


# ---------------------------------------------------------------------------
# Replay-state observation extraction (for rl_collect.py)
# ---------------------------------------------------------------------------

def extract_spatial_from_replay_state(state: dict, ship_id: str, pid: int) -> np.ndarray:
    """
    Build the spatial observation from a replay-state dict produced by
    ReplayStateReconstructor.get_state().

    The state dict has keys:
      halite         : {(x,y): int}
      entities       : {str(pid): {str(eid): {x,y,energy,is_inspired}}}
      structures     : {pid: [(x,y), ...]}  (includes factories)
      width, height  : int
      factories      : {pid: (x,y)}
    """
    W, H       = state['width'], state['height']
    half       = WINDOW_SIZE // 2
    max_d      = (W + H) / 2.0
    MAX_HALITE = 1000

    ent_pid    = state['entities'].get(str(pid), {})
    ship_data  = ent_pid.get(str(ship_id))
    if ship_data is None:
        return np.zeros((WINDOW_SIZE, WINDOW_SIZE, N_SPATIAL_CHANNELS), dtype=np.float32)

    sx, sy = ship_data['x'], ship_data['y']

    # Position → (pid2, eid, entity_data) lookup
    pos_to_agent = {}
    for p_str, p_ents in state['entities'].items():
        p2 = int(p_str)
        for eid_str, edata in p_ents.items():
            pos_to_agent[(edata['x'], edata['y'])] = (p2, eid_str, edata)

    my_structs = set(state['structures'].get(pid, []))
    my_factory = state['factories'].get(pid)

    opp_structs = set()
    for p2, structs in state['structures'].items():
        if p2 != pid:
            opp_structs.update(structs)

    # Pre-compute 1-step reachable sets for danger-zone channels
    _adj = [(0, 0), (0, -1), (0, 1), (1, 0), (-1, 0)]

    enemy_reachable: set = set()
    friendly_reachable: set = set()

    for p_str, p_ents in state['entities'].items():
        p2 = int(p_str)
        for eid_str, edata in p_ents.items():
            ex, ey = edata['x'], edata['y']
            if p2 != pid:
                for ddx, ddy in _adj:
                    enemy_reachable.add(((ex + ddx) % W, (ey + ddy) % H))
            elif eid_str != str(ship_id):
                for ddx, ddy in _adj:
                    friendly_reachable.add(((ex + ddx) % W, (ey + ddy) % H))

    spatial = np.zeros((WINDOW_SIZE, WINDOW_SIZE, N_SPATIAL_CHANNELS), dtype=np.float32)

    for dy_off in range(-half, half + 1):
        for dx_off in range(-half, half + 1):
            cx = (sx + dx_off) % W
            cy = (sy + dy_off) % H
            wy = dy_off + half
            wx = dx_off + half

            h = state['halite'].get((cx, cy), 0)
            spatial[wy, wx, 0] = h / MAX_HALITE

            if (cx, cy) in pos_to_agent:
                p2, _eid, edata = pos_to_agent[(cx, cy)]
                if p2 == pid:
                    spatial[wy, wx, 1] = 1.0
                    spatial[wy, wx, 2] = edata['energy'] / MAX_HALITE
                    if edata.get('is_inspired', False):
                        spatial[wy, wx, 6] = 1.0
                else:
                    spatial[wy, wx, 3] = 1.0
                    # Ch 8: enemy cargo (0=kamikaze threat, 1=wants to go home)
                    spatial[wy, wx, 8] = edata['energy'] / MAX_HALITE

            if (cx, cy) in my_structs:
                spatial[wy, wx, 4] = 1.0 if (cx, cy) == my_factory else 0.5
            elif (cx, cy) in opp_structs:
                spatial[wy, wx, 5] = 1.0

            min_d = min(torus_dist(cx, cy, fx, fy, W, H) for fx, fy in my_structs) \
                    if my_structs else (W + H)
            spatial[wy, wx, 7] = 1.0 - (min_d / max_d)

            if (cx, cy) in enemy_reachable:
                spatial[wy, wx, 9] = 1.0
            if (cx, cy) in friendly_reachable:
                spatial[wy, wx, 10] = 1.0

    return spatial


def extract_scalars_from_replay_state(state: dict, ship_id: str, pid: int) -> np.ndarray:
    """
    Build the scalar feature vector from a replay-state dict.
    """
    W, H       = state['width'], state['height']
    MAX_HALITE = 1000

    ent_pid   = state['entities'].get(str(pid), {})
    ship_data = ent_pid.get(str(ship_id))
    if ship_data is None:
        return np.zeros(N_SCALAR_FEATURES, dtype=np.float32)

    sx, sy      = ship_data['x'], ship_data['y']
    cargo       = ship_data['energy']
    is_inspired = ship_data.get('is_inspired', False)

    my_structs = list(state['structures'].get(pid, [state['factories'].get(pid, (0, 0))]))
    near       = min(my_structs, key=lambda d: torus_dist(sx, sy, d[0], d[1], W, H))
    dist_dep   = torus_dist(sx, sy, near[0], near[1], W, H)
    ddx, ddy   = torus_delta(sx, sy, near[0], near[1], W, H)
    max_dist   = W + H

    max_turns  = state['max_turns']
    turn       = state['turn']
    turns_left = max_turns - turn

    return_urgency = 1.0 if (turns_left <= dist_dep * 1.5 + 1 and cargo > 0) else 0.0
    turns_slack    = (turns_left - dist_dep) / max(1, max_turns)

    my_ships  = len(ent_pid)
    opp_ships = sum(len(p_ents) for p_str, p_ents in state['entities'].items()
                    if int(p_str) != pid)
    my_bank   = state['player_energy'].get(pid, 5000)

    # Proximity danger: count ships within 2 toroidal steps
    enemy_near    = 0
    friendly_near = 0
    for p_str, p_ents in state['entities'].items():
        p2 = int(p_str)
        for eid_str, edata in p_ents.items():
            d = torus_dist(sx, sy, edata['x'], edata['y'], W, H)
            if d <= 2:
                if p2 != pid:
                    enemy_near += 1
                elif eid_str != str(ship_id):
                    friendly_near += 1

    return np.array([
        turns_left / max_turns,
        my_bank    / MAX_HALITE,
        cargo      / MAX_HALITE,
        my_ships   / 30.0,
        opp_ships  / 30.0,
        float(is_inspired),
        dist_dep   / max_dist,
        ddx / W,
        ddy / H,
        return_urgency,
        turns_slack,
        enemy_near    / 10.0,
        friendly_near / 10.0,
        # Prospect features (indices 13–16)
        *_prospect_scalars_replay(sx, sy, state['halite'], W, H, max_dist, MAX_HALITE),
    ], dtype=np.float32)


def _prospect_scalars_replay(
    sx: int, sy: int, halite_dict: dict,
    W: int, H: int, max_dist: float, MAX_HALITE: int,
) -> Tuple[float, float, float, float]:
    rx, ry, val = _richest_in_prospect_window(sx, sy, halite_dict, W, H)
    dx, dy = torus_delta(sx, sy, rx, ry, W, H)
    dist   = torus_dist(sx, sy, rx, ry, W, H)
    return dx / W, dy / H, val / MAX_HALITE, dist / max_dist
