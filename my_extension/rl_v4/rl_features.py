"""
Feature extraction for Halite III RL training and inference.

Provides observation extraction from engine internal state (for training)
and a replay-state dict (for imitation learning data collection).

Observation layout
------------------
Spatial : float32[WINDOW_SIZE, WINDOW_SIZE, N_SPATIAL_CHANNELS]
  Window is 11×11 (5-cell radius) centred on the observing ship.
  Ch 0  – cell halite / MAX_HALITE
  Ch 1  – my ship present (binary)
  Ch 2  – my ship cargo / MAX_HALITE
  Ch 3  – opponent ship (binary)
  Ch 4  – my structures: 1.0 = factory, 0.5 = dropoff
  Ch 5  – opponent structures (binary)
  Ch 6  – inspired my-ship (binary)
  Ch 7  – 1 − dist_cell_to_nearest_deposit / max_dist   (per-cell)
  Ch 8  – enemy ship cargo / MAX_HALITE (0 if no enemy here; 0 = kamikaze, 1 = safe)
  Ch 9  – enemy danger zone: 1 if any enemy ship is ≤1 step from this cell
  Ch 10 – friendly danger zone: 1 if any OTHER friendly ship is ≤1 step from this cell
  -- New rl_v4 channels --
  Ch 11 – dropoff suitability: (dist_cell_to_nearest_deposit / max_dist)
          × (mean 3×3 halite around cell / MAX_HALITE).  High where a far-out,
          halite-rich region would benefit from a new dropoff.
  Ch 12 – inspiration potential: min(#enemy ships within radius 4 of cell / 2, 1)
  Ch 13 – friendly cargo congestion: Σ cargo of friendly ships ≤1 step from
          cell / MAX_HALITE (clamped to 1).  Marks crowded deposit approaches.

Scalars : float32[N_SCALAR_FEATURES]
  -- Navigation & economy (rl_v2 features) --
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
  -- New rl_v3 features --
  17 – cell halite at ship's current position / MAX_HALITE
  18 – expected mine yield if ship stays this turn / MAX_HALITE
  19 – winning ratio: my_bank / (total_bank + 1)    range [0, 1]
  20 – endgame flag: 1.0 if turns_remaining < 50 else 0.0
  21 – number of own dropoffs built / 5
  22 – average fleet distance to nearest deposit / (W + H)
  23 – bank can afford dropoff: 1.0 if bank >= DROPOFF_COST else 0.0
  -- New rl_v4 features (win-alignment & dropoff support) --
  24 – bank margin vs leading opponent: tanh((my_bank − opp_bank) / 5000)
  25 – opponent bank / MAX_HALITE
  26 – dropoff affordable for THIS ship now: 1.0 if my_bank >= net cost of
       converting this ship (max(0, DROPOFF_COST − (cell_h + cargo))) else 0.0
  27 – target-dropoff slack: max(0, (target_dropoffs − num_dropoffs) / target)
  28 – fraction of map halite remaining: Σ map halite / (W·H·MAX_HALITE)

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
  8 DROPOFF – learned: convert this ship into a dropoff (legal only when the
               action mask permits — see valid_action_mask / dropoff_legal)
"""

import math
import numpy as np
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Observation shape constants
# ---------------------------------------------------------------------------

WINDOW_SIZE        = 11    # 11×11 window (5-cell radius) — expanded from rl_v2's 5×5
N_SPATIAL_CHANNELS = 14    # rl_v3's 11 + 3 rl_v4 channels (dropoff/inspiration/congestion)
N_SCALAR_FEATURES  = 29    # rl_v3's 24 + 5 rl_v4 win-alignment / dropoff features

# Radius for PROSPECT window scan: (2*PROSPECT_RADIUS+1) × (2*PROSPECT_RADIUS+1) = 11×11
PROSPECT_RADIUS = 5

# Inspiration radius (mirrors halite_engine INSPIRATION_RADIUS) for Ch 12.
INSPIRATION_RADIUS = 4

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
ACTION_DROPOFF = 8   # learned: convert this ship into a dropoff (mask-gated)

N_SHIP_ACTIONS = 9

# ---------------------------------------------------------------------------
# Economy / dropoff constants (single source of truth shared by env + bot)
# ---------------------------------------------------------------------------

SHIP_COST                = 1000   # halite to spawn a ship
DROPOFF_COST             = 4000   # gross halite to convert a ship to a dropoff
MAX_DROPOFFS             = 2      # hard cap on dropoffs the policy may build
DROPOFF_MIN_DIST         = 6      # min distance from any deposit to allow a dropoff
DROPOFF_MIN_TURNS_LEFT   = 100    # no dropoffs once this few turns remain
DROPOFF_RESERVE          = 4000   # bank kept free (above SHIP_COST) so a dropoff
                                  # stays affordable while one is still wanted
SPAWN_MIN_TURNS_LEFT     = 100    # stop spawning once this few turns remain — a new
                                  # ship needs ~this long to mine back its 1000 cost
                                  # (raised 75→100: late ships can't pay for themselves
                                  # and only shrink the winning margin)
SPAWN_RESERVE_MIN_SHIPS  = 5      # only reserve bank for a dropoff once the fleet
                                  # has grown to at least this many ships
MAX_FLEET                = 16     # fleet cap on 32×32 (≈ max_dim / 2)

# Game-length constants (mirror halite_engine MIN_TURNS / scaling).  The engine's
# init message advertises MAX_TURNS=500 to bots regardless of map size, but a
# 32×32 game actually ends at 400 turns — so the hlt-side bot must compute the
# true value itself (used for turns_left and the turns_remaining/endgame features,
# keeping inference consistent with training which uses the engine's real value).
MIN_TURNS                = 400
MAX_TURNS_CAP            = 500
MIN_TURN_THRESHOLD       = 32
MAX_TURN_THRESHOLD       = 64


def game_max_turns(width: int, height: int) -> int:
    """True game length for a map of this size (mirrors HaliteEngine)."""
    max_dim = max(width, height)
    turns = MIN_TURNS
    if max_dim > MIN_TURN_THRESHOLD:
        turns += int(((max_dim - MIN_TURN_THRESHOLD)
                      / (MAX_TURN_THRESHOLD - MIN_TURN_THRESHOLD))
                     * (MAX_TURNS_CAP - MIN_TURNS))
    return turns

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
# Dropoff legality, action masking, spawn economy (shared by env + bot)
# ---------------------------------------------------------------------------

def target_dropoffs(W: int, H: int) -> int:
    """How many dropoffs the bot should aim to build on a map of this size."""
    m = max(W, H)
    if m <= 40:
        return 1
    if m <= 48:
        return 2
    if m <= 56:
        return 3
    return 4


def dropoff_legal(
    bank: float, cell_h: int, cargo: int,
    dist_to_nearest: int, num_dropoffs: int,
    turns_left: int, cell_owned: bool,
) -> bool:
    """Whether converting THIS ship into a dropoff is a legal move right now.

    Mirrors the engine's construct validation (halite_engine `_process_commands`):
    cell must be unowned and the player must be able to afford the net cost
    `max(0, DROPOFF_COST − (cell_h + cargo))`.  Adds policy-level guards so the
    learned DROPOFF action is only offered when it is economically sensible.
    """
    if cell_owned:
        return False
    if num_dropoffs >= MAX_DROPOFFS:
        return False
    if dist_to_nearest < DROPOFF_MIN_DIST:
        return False
    if turns_left <= DROPOFF_MIN_TURNS_LEFT:
        return False
    cost = max(0, DROPOFF_COST - (cell_h + cargo))
    return bank >= cost


def build_action_mask(dropoff_ok: bool) -> np.ndarray:
    """Return a bool[N_SHIP_ACTIONS] mask: all move/meta actions always legal;
    DROPOFF legal only when `dropoff_ok`."""
    mask = np.ones(N_SHIP_ACTIONS, dtype=bool)
    mask[ACTION_DROPOFF] = bool(dropoff_ok)
    return mask


def action_mask_from_engine(engine, ship_id: int, pid: int = 0,
                            allow_dropoff: bool = True) -> np.ndarray:
    """Build the action mask for a ship from engine internal state."""
    if not allow_dropoff:
        return build_action_mask(False)
    sx, sy       = engine.player_entities[pid][ship_id]
    bank         = engine.players[pid]['energy']
    cell_h       = engine.halite.get((sx, sy), 0)
    cargo        = engine.entities[ship_id]['cargo']
    num_dropoffs = len(engine.players[pid]['dropoffs'])
    near         = _nearest_deposit(sx, sy, engine, pid)
    dist         = torus_dist(sx, sy, near[0], near[1], engine.width, engine.height)
    turns_left   = engine.max_turns - engine.turn
    cell_owned   = engine.cell_owner.get((sx, sy)) is not None
    ok = dropoff_legal(bank, cell_h, cargo, dist, num_dropoffs, turns_left, cell_owned)
    return build_action_mask(ok)


def spawn_econ_ok(bank: float, n_ships: int, turns_left: int,
                  num_dropoffs: int, tgt_dropoffs: int,
                  max_fleet: int = MAX_FLEET) -> bool:
    """Shared spawn economic gate (factory-occupancy guard handled by caller).

    Reserves DROPOFF_RESERVE halite (above SHIP_COST) so a wanted dropoff stays
    affordable, but only once the fleet is established (>= SPAWN_RESERVE_MIN_SHIPS)
    so early fleet growth is never starved — this is the fix for rl_v3's spawn
    rule draining the bank below the dropoff threshold.
    """
    if bank < SHIP_COST:
        return False
    if n_ships >= max_fleet:
        return False
    if turns_left <= SPAWN_MIN_TURNS_LEFT:
        return False
    if (num_dropoffs < tgt_dropoffs
            and n_ships >= SPAWN_RESERVE_MIN_SHIPS
            and turns_left > 150
            and (bank - SHIP_COST) < DROPOFF_RESERVE):
        return False
    return True


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
    _adj = [(0, 0), (0, -1), (0, 1), (1, 0), (-1, 0)]  # stay + N/S/E/W

    enemy_reachable: set = set()
    friendly_reachable: set = set()
    enemy_positions: list = []                 # for Ch 12 (inspiration potential)
    friendly_cargo_map: Dict[Tuple, int] = {}  # for Ch 13 (friendly cargo congestion)

    for p in engine.player_entities:
        for sid2, (ex, ey) in engine.player_entities[p].items():
            if p != pid:
                enemy_positions.append((ex, ey))
                for ddx, ddy in _adj:
                    enemy_reachable.add(((ex + ddx) % W, (ey + ddy) % H))
            else:
                cargo2 = engine.entities[sid2]['cargo']
                for ddx, ddy in _adj:
                    cell = ((ex + ddx) % W, (ey + ddy) % H)
                    friendly_cargo_map[cell] = friendly_cargo_map.get(cell, 0) + cargo2
                if sid2 != ship_id:
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
                    spatial[wy, wx, 8] = engine.entities[sid2]['cargo'] / MAX_HALITE

            cell_owner = engine.cell_owner.get((cx, cy))
            if cell_owner == pid:
                spatial[wy, wx, 4] = 1.0 if (cx, cy) == my_factory else 0.5
            elif cell_owner is not None:
                spatial[wy, wx, 5] = 1.0

            # Per-cell distance to nearest deposit
            min_d = min(torus_dist(cx, cy, fx, fy, W, H) for fx, fy in my_deposits)
            spatial[wy, wx, 7] = 1.0 - (min_d / max_d)

            if (cx, cy) in enemy_reachable:
                spatial[wy, wx, 9] = 1.0
            if (cx, cy) in friendly_reachable:
                spatial[wy, wx, 10] = 1.0

            # Ch 11 — dropoff suitability: far-from-deposit × locally rich
            mean3 = 0.0
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    mean3 += engine.halite.get(((cx + ox) % W, (cy + oy) % H), 0)
            spatial[wy, wx, 11] = min(min_d / max_d, 1.0) * (mean3 / 9.0 / MAX_HALITE)

            # Ch 12 — inspiration potential: enemy ships within radius 4 of cell
            ins = sum(1 for ex, ey in enemy_positions
                      if torus_dist(cx, cy, ex, ey, W, H) <= INSPIRATION_RADIUS)
            spatial[wy, wx, 12] = min(ins / 2.0, 1.0)

            # Ch 13 — friendly cargo congestion within 1 step of cell
            spatial[wy, wx, 13] = min(friendly_cargo_map.get((cx, cy), 0) / MAX_HALITE, 1.0)

    return spatial


def extract_scalars_from_engine(engine, ship_id: int, pid: int = 0) -> np.ndarray:
    """
    Build an N_SCALAR_FEATURES float32 scalar feature vector for the given ship.
    Scalar layout described in module docstring.
    """
    from halite_engine import MAX_HALITE, EXTRACT_RATIO, DROPOFF_COST
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

    # rl_v3 new scalars -------------------------------------------------------
    cell_h = engine.halite.get((sx, sy), 0)
    mine_yield = math.ceil(cell_h / EXTRACT_RATIO) if cell_h > 0 else 0

    bank = engine.players[pid]['energy']
    total_bank = sum(engine.players[p]['energy'] for p in engine.players)
    winning_ratio = bank / (total_bank + 1)

    endgame_flag = 1.0 if turns_left < 50 else 0.0
    num_dropoffs = len(engine.players[pid]['dropoffs'])

    # Average fleet distance to nearest deposit
    fleet_dists = []
    all_deposits = [engine.players[pid]['factory']] + [
        (dx2, dy2) for _, dx2, dy2 in engine.players[pid]['dropoffs']
    ]
    for sid2, (fx2, fy2) in engine.player_entities[pid].items():
        d2 = min(torus_dist(fx2, fy2, dpx, dpy, W, H) for dpx, dpy in all_deposits)
        fleet_dists.append(d2)
    avg_fleet_dist = sum(fleet_dists) / max(1, len(fleet_dists))

    bank_can_afford_dropoff = 1.0 if bank >= DROPOFF_COST else 0.0

    # rl_v4 win-alignment / dropoff scalars (indices 24–28) -------------------
    opp_bank = max((engine.players[p]['energy']
                    for p in engine.players if p != pid), default=0)
    bank_margin = math.tanh((bank - opp_bank) / 5000.0)
    dropoff_cost_here = max(0, DROPOFF_COST - (cell_h + cargo))
    dropoff_affordable = 1.0 if bank >= dropoff_cost_here else 0.0
    tgt = target_dropoffs(W, H)
    dropoff_slack = max(0.0, (tgt - num_dropoffs) / max(1, tgt))
    total_map_halite = sum(engine.halite.values())
    halite_frac = total_map_halite / (W * H * MAX_HALITE)

    return np.array([
        turns_left / engine.max_turns,
        bank / MAX_HALITE,
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
        # rl_v3 features (indices 17–23)
        cell_h / MAX_HALITE,
        mine_yield / MAX_HALITE,
        winning_ratio,
        endgame_flag,
        num_dropoffs / 5.0,
        avg_fleet_dist / max_dist,
        bank_can_afford_dropoff,
        # rl_v4 features (indices 24–28)
        bank_margin,
        opp_bank / MAX_HALITE,
        dropoff_affordable,
        dropoff_slack,
        halite_frac,
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

    _adj = [(0, 0), (0, -1), (0, 1), (1, 0), (-1, 0)]

    enemy_reachable: set = set()
    friendly_reachable: set = set()
    enemy_positions: list = []
    friendly_cargo_map: Dict[Tuple, int] = {}

    for p_str, p_ents in state['entities'].items():
        p2 = int(p_str)
        for eid_str, edata in p_ents.items():
            ex, ey = edata['x'], edata['y']
            if p2 != pid:
                enemy_positions.append((ex, ey))
                for ddx, ddy in _adj:
                    enemy_reachable.add(((ex + ddx) % W, (ey + ddy) % H))
            else:
                cargo2 = edata['energy']
                for ddx, ddy in _adj:
                    cell = ((ex + ddx) % W, (ey + ddy) % H)
                    friendly_cargo_map[cell] = friendly_cargo_map.get(cell, 0) + cargo2
                if eid_str != str(ship_id):
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

            # Ch 11 — dropoff suitability
            mean3 = 0.0
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    mean3 += state['halite'].get(((cx + ox) % W, (cy + oy) % H), 0)
            spatial[wy, wx, 11] = min(min_d / max_d, 1.0) * (mean3 / 9.0 / MAX_HALITE)

            # Ch 12 — inspiration potential
            ins = sum(1 for ex, ey in enemy_positions
                      if torus_dist(cx, cy, ex, ey, W, H) <= INSPIRATION_RADIUS)
            spatial[wy, wx, 12] = min(ins / 2.0, 1.0)

            # Ch 13 — friendly cargo congestion
            spatial[wy, wx, 13] = min(friendly_cargo_map.get((cx, cy), 0) / MAX_HALITE, 1.0)

    return spatial


def extract_scalars_from_replay_state(state: dict, ship_id: str, pid: int) -> np.ndarray:
    """
    Build the scalar feature vector from a replay-state dict.
    """
    W, H       = state['width'], state['height']
    MAX_HALITE = 1000
    EXTRACT_RATIO = 4
    DROPOFF_COST  = 4000

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

    # rl_v3 new scalars -------------------------------------------------------
    cell_h = state['halite'].get((sx, sy), 0)
    mine_yield = math.ceil(cell_h / EXTRACT_RATIO) if cell_h > 0 else 0

    total_bank = sum(state['player_energy'].get(p2, 0)
                     for p2 in state['player_energy'])
    winning_ratio = my_bank / (total_bank + 1)

    endgame_flag = 1.0 if turns_left < 50 else 0.0

    # num_dropoffs = structures minus the factory
    num_dropoffs = max(0, len(my_structs) - 1)

    # Average fleet distance to nearest deposit
    fleet_dists = []
    for eid_str2, edata2 in ent_pid.items():
        fx2, fy2 = edata2['x'], edata2['y']
        d2 = min(torus_dist(fx2, fy2, s[0], s[1], W, H) for s in my_structs) if my_structs else 0
        fleet_dists.append(d2)
    avg_fleet_dist = sum(fleet_dists) / max(1, len(fleet_dists))

    bank_can_afford_dropoff = 1.0 if my_bank >= DROPOFF_COST else 0.0

    # rl_v4 win-alignment / dropoff scalars (indices 24–28) -------------------
    opp_bank = max((state['player_energy'].get(p2, 0)
                    for p2 in state['player_energy'] if p2 != pid), default=0)
    bank_margin = math.tanh((my_bank - opp_bank) / 5000.0)
    dropoff_cost_here = max(0, DROPOFF_COST - (cell_h + cargo))
    dropoff_affordable = 1.0 if my_bank >= dropoff_cost_here else 0.0
    tgt = target_dropoffs(W, H)
    dropoff_slack = max(0.0, (tgt - num_dropoffs) / max(1, tgt))
    total_map_halite = sum(state['halite'].values())
    halite_frac = total_map_halite / (W * H * MAX_HALITE)

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
        # rl_v3 features (indices 17–23)
        cell_h / MAX_HALITE,
        mine_yield / MAX_HALITE,
        winning_ratio,
        endgame_flag,
        num_dropoffs / 5.0,
        avg_fleet_dist / max_dist,
        bank_can_afford_dropoff,
        # rl_v4 features (indices 24–28)
        bank_margin,
        opp_bank / MAX_HALITE,
        dropoff_affordable,
        dropoff_slack,
        halite_frac,
    ], dtype=np.float32)


def _prospect_scalars_replay(
    sx: int, sy: int, halite_dict: dict,
    W: int, H: int, max_dist: float, MAX_HALITE: int,
) -> Tuple[float, float, float, float]:
    rx, ry, val = _richest_in_prospect_window(sx, sy, halite_dict, W, H)
    dx, dy = torus_delta(sx, sy, rx, ry, W, H)
    dist   = torus_dist(sx, sy, rx, ry, W, H)
    return dx / W, dy / H, val / MAX_HALITE, dist / max_dist
