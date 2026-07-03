#!/usr/bin/env python3
"""
rl_v8 / config.py  --  central constants, action space, feature spec, and the
small pure economy helpers ported verbatim from rl_v5 for economy parity.

This module deliberately has NO heavy dependencies (only numpy) so it can be
imported by every other rl_v7 module: data collection, training, inference,
and the RL environment.  Keeping all the magic numbers and the feature-name
list here is what guarantees the data pipeline, the network, and the live bot
all agree on the exact same representation (zero train/inference skew).

rl_v5, rl_v6, and every other folder in the repo are left untouched.  The
economy functions below are *copied* (not imported) from rl_v5/rl_features.py
so rl_v7 is self-contained while still matching rl_v5's spawn/dropoff economy.
"""

import math
import os
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE       = os.path.dirname(os.path.abspath(__file__))      # .../my_extension/rl_v7
MY_EXT     = os.path.dirname(HERE)                           # .../my_extension
REPO_ROOT  = os.path.dirname(MY_EXT)                         # repo root
STARTER_KIT = os.path.join(REPO_ROOT, 'starter_kits', 'Python3')

REPLAYS_DIR    = os.path.join(HERE, 'replays')
DATASET_DIR    = os.path.join(HERE, 'dataset')
CHECKPOINT_DIR = os.path.join(HERE, 'checkpoints')

FEATURES_CSV   = os.path.join(DATASET_DIR, 'features.csv')
PATCHES_NPY    = os.path.join(DATASET_DIR, 'patches.npy')

# ---------------------------------------------------------------------------
# Game constants (mirror halite_engine.py / rl_v5)
# ---------------------------------------------------------------------------

SHIP_COST      = 1000
DROPOFF_COST   = 4000
MAX_HALITE     = 1000     # ship cargo cap
INITIAL_ENERGY = 5000
EXTRACT_RATIO  = 4
MOVE_COST_RATIO = 10
INSPIRATION_RADIUS     = 4
INSPIRATION_SHIP_COUNT = 2

MIN_TURNS          = 400
MAX_TURNS_CAP      = 500
MIN_TURN_THRESHOLD = 32
MAX_TURN_THRESHOLD = 64


def game_max_turns(width: int, height: int) -> int:
    """True game length for a map of this size (mirrors HaliteEngine)."""
    max_dim = max(width, height)
    turns = MIN_TURNS
    if max_dim > MIN_TURN_THRESHOLD:
        turns += int(((max_dim - MIN_TURN_THRESHOLD)
                      / (MAX_TURN_THRESHOLD - MIN_TURN_THRESHOLD))
                     * (MAX_TURNS_CAP - MIN_TURNS))
    return turns


# ---------------------------------------------------------------------------
# Economy / dropoff constants + helpers (ported verbatim from rl_v5)
# ---------------------------------------------------------------------------

MAX_DROPOFFS           = 2
DROPOFF_MIN_DIST       = 6
DROPOFF_MIN_TURNS_LEFT = 100
DROPOFF_RESERVE        = 4000
SPAWN_MIN_TURNS_LEFT   = 100
SPAWN_RESERVE_MIN_SHIPS = 5
MAX_FLEET              = 16
ENDGAME_COLLAPSE_TURNS = 15
HOME_CARGO_THRESHOLD   = 0.90    # cargo fraction that triggers "go home"
ENDGAME_BUFFER         = 5


def target_dropoffs(W: int, H: int) -> int:
    """How many dropoffs to aim for on a map of this size (rl_v5)."""
    m = max(W, H)
    if m <= 40:
        return 1
    if m <= 48:
        return 2
    if m <= 56:
        return 3
    return 4


def dropoff_legal(bank, cell_h, cargo, dist_to_nearest, num_dropoffs,
                  turns_left, cell_owned) -> bool:
    """Whether converting THIS ship into a dropoff is legal right now (rl_v5)."""
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


def spawn_econ_ok(bank, n_ships, turns_left, num_dropoffs, tgt_dropoffs,
                  max_fleet: int = MAX_FLEET) -> bool:
    """Shared spawn economic gate (factory-occupancy guard handled by caller)."""
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
# Action space  (6 discrete actions; intent only — a deterministic resolver
# deconflicts the moves so no two friendly ships ever collide)
# ---------------------------------------------------------------------------

ACTION_STAY    = 0
ACTION_NORTH   = 1
ACTION_EAST    = 2
ACTION_SOUTH   = 3
ACTION_WEST    = 4
ACTION_DROPOFF = 5
N_ACTIONS      = 6

ACTION_NAMES = ['STAY', 'NORTH', 'EAST', 'SOUTH', 'WEST', 'DROPOFF']

# (dx, dy) for each movement action.  North decreases y (matches engine).
ACTION_DELTA = {
    ACTION_STAY:  (0, 0),
    ACTION_NORTH: (0, -1),
    ACTION_EAST:  (1, 0),
    ACTION_SOUTH: (0, 1),
    ACTION_WEST:  (-1, 0),
}

# engine direction characters
ACTION_TO_DIR = {
    ACTION_STAY:  'o',
    ACTION_NORTH: 'n',
    ACTION_EAST:  'e',
    ACTION_SOUTH: 's',
    ACTION_WEST:  'w',
}
DIR_TO_ACTION = {
    'o': ACTION_STAY,
    'n': ACTION_NORTH,
    'e': ACTION_EAST,
    's': ACTION_SOUTH,
    'w': ACTION_WEST,
}

# ---------------------------------------------------------------------------
# Map patch (local spatial tensor) spec
# ---------------------------------------------------------------------------

PATCH_RADIUS   = 4                       # 9x9 window (covers inspiration radius)
PATCH_SIZE     = 2 * PATCH_RADIUS + 1    # 9
PATCH_CHANNELS = 6
# channel meanings (documented; kept in sync with features.extract_patch):
PATCH_CHANNEL_NAMES = [
    'halite',            # 0: cell halite / MAX_HALITE
    'friendly_ship',     # 1: friendly ship present (binary)
    'enemy_ship',        # 2: enemy ship present (binary)
    'friendly_cargo',    # 3: friendly ship cargo / MAX_HALITE
    'enemy_cargo',       # 4: enemy ship cargo / MAX_HALITE
    'structure',         # 5: +1 my factory/dropoff, -1 enemy
]

# ---------------------------------------------------------------------------
# Scalar feature spec  --  the order here defines the CSV column order and the
# network input order.  Keep this list and features.extract_scalars in lockstep.
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # --- per-ship economy ---
    'cargo_frac',            # cargo / MAX_HALITE
    'cargo_to_full_frac',    # (MAX_HALITE - cargo) / MAX_HALITE
    'cargo_ge_home',         # 1 if cargo >= HOME_CARGO_THRESHOLD * MAX_HALITE
    'cell_halite_frac',      # halite on current cell / MAX_HALITE
    'mine_yield_frac',       # ceil(cell_h / EXTRACT_RATIO) / MAX_HALITE
    'can_afford_move',       # 1 if cargo >= cell_h // MOVE_COST_RATIO
    'is_inspired',           # 1 if >= 2 enemies within radius 4
    # --- homing ---
    'dist_home_frac',        # toroidal dist to nearest deposit / (W+H)
    'dx_home',               # signed toroidal dx toward nearest deposit / W
    'dy_home',               # signed toroidal dy toward nearest deposit / H
    'on_deposit',            # 1 if currently on a friendly deposit cell
    'return_urgency',        # 1 if turns_left <= dist*1.5 + 1 and cargo > 0
    'turns_slack',           # (turns_left - dist) / max_turns
    # --- prospecting (richest cell in PATCH window) ---
    'dx_richest',            # signed toroidal dx to richest cell / W
    'dy_richest',            # signed toroidal dy to richest cell / H
    'richest_halite_frac',   # richest cell halite / MAX_HALITE
    'dist_richest_frac',     # toroidal dist to richest cell / (W+H)
    'local_mean3_frac',      # mean halite of 3x3 around ship / MAX_HALITE
    'window_mean_frac',      # mean halite of PATCH window / MAX_HALITE
    # --- danger / contention ---
    'enemy_within_1',        # enemy ships within 1 step / 4 (clamped)
    'enemy_within_2',        # enemy ships within 2 steps / 8 (clamped)
    'friendly_within_1',     # other friendly ships within 1 step / 4 (clamped)
    'friendly_within_2',     # other friendly ships within 2 steps / 8 (clamped)
    'min_enemy_cargo_near',  # min enemy cargo within 2 / MAX_HALITE (1.0 if none)
    'enemy_count_r4',        # enemy ships within radius 4 / 4 (clamped)
    # --- global / fleet / phase ---
    'turn_frac',             # turn / max_turns
    'turns_left_frac',       # (max_turns - turn) / max_turns
    'my_ships_frac',         # my ship count / MAX_FLEET
    'opp_ships_frac',        # opponent ship count / MAX_FLEET
    'my_bank_frac',          # my bank / BANK_SCALE (clamped)
    'opp_bank_frac',         # opponent bank / BANK_SCALE (clamped)
    'bank_margin_tanh',      # tanh((my_bank - opp_bank) / 5000)
    'winning_ratio',         # my_bank / (total_bank + 1)
    'num_dropoffs_frac',     # my dropoff count / MAX_DROPOFFS
    'map_halite_frac',       # total map halite / (W*H*MAX_HALITE)
    'dropoff_affordable',    # 1 if bank >= net dropoff cost on this cell
    'dropoff_legal',         # 1 if dropoff_legal() for this ship right now
    'map_w_norm',            # W / 64
    'map_h_norm',            # H / 64
]

N_SCALARS = len(FEATURE_NAMES)

BANK_SCALE = 20000.0   # normaliser for raw bank halite

# Extra non-feature columns written to the CSV for inspection / traceability.
META_COLUMNS = ['row_id', 'game_id', 'player_id', 'ship_id', 'turn', 'action']


# ---------------------------------------------------------------------------
# Toroidal geometry helpers (ported from rl_v5)
# ---------------------------------------------------------------------------

def torus_dist(x1, y1, x2, y2, W, H) -> int:
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return min(dx, W - dx) + min(dy, H - dy)


def torus_delta(sx, sy, tx, ty, W, H) -> Tuple[int, int]:
    """Signed toroidal vector from (sx, sy) to (tx, ty)."""
    dx = tx - sx
    dy = ty - sy
    if abs(dx) > W // 2:
        dx = dx - W if dx > 0 else dx + W
    if abs(dy) > H // 2:
        dy = dy - H if dy > 0 else dy + H
    return dx, dy
