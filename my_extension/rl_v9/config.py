#!/usr/bin/env python3
"""
rl_v9 / config.py  --  central constants, action space, and feature spec.

rl_v9's goal: BEAT V71 (the user's 2019 hand-coded bot).  It differs from
rl_v8 in four deliberate ways:

1. NO hard-coded fleet cap and NO hand-coded spawn rule: spawning is a
   *learned* binary action produced by a dedicated SpawnNet (BC-initialised
   from V71's own spawn decisions, then PPO fine-tuned).  The only spawn
   "mask" is affordability (bank >= SHIP_COST).
2. Dropoff construction is *learned*: the only legality mask is physical
   (cell not already owned by any structure) + affordability.  All the old
   rl_v5 heuristics (min distance, turns-left gate, dropoff quota) are gone.
3. Enemy collisions are ALLOWED and learnable: the resolver deconflicts
   *friendly* ships only.  Rewards credit the cargo/ship value exchanged in
   a wreck, so ramming a loaded enemy with an empty ship is learnable, as is
   breaking out of multi-turn traffic jams.
4. Per-ship MEMORY features (sticky homing flag, previous action, stuck
   counter) attack the hidden-FSM-state problem that capped BC match rate
   at ~58% in rl_v7/rl_v8 (V71 is a per-ship FSM with a persistent path
   buffer; single-frame features cannot represent that).

This module has NO heavy dependencies (numpy only) so every other module can
import it: data collection, training, inference, and the RL environment.
"""

import math
import os
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE       = os.path.dirname(os.path.abspath(__file__))      # .../my_extension/rl_v9
MY_EXT     = os.path.dirname(HERE)                           # .../my_extension
REPO_ROOT  = os.path.dirname(MY_EXT)                         # repo root
STARTER_KIT = os.path.join(REPO_ROOT, 'starter_kits', 'Python3')

REPLAYS_DIR    = os.path.join(HERE, 'replays')
DATASET_DIR    = os.path.join(HERE, 'dataset')
CHECKPOINT_DIR = os.path.join(HERE, 'checkpoints')

FEATURES_CSV      = os.path.join(DATASET_DIR, 'features.csv')
PATCHES_NPY       = os.path.join(DATASET_DIR, 'patches.npy')
GLOBALS_NPY       = os.path.join(DATASET_DIR, 'globals.npy')
SPAWN_CSV         = os.path.join(DATASET_DIR, 'spawn.csv')
SPAWN_GLOBALS_NPY = os.path.join(DATASET_DIR, 'spawn_globals.npy')

# ---------------------------------------------------------------------------
# Game constants (mirror halite_engine.py)
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
# Economy: rl_v9 has NO hand-coded spawn rule and almost no dropoff rule.
# ---------------------------------------------------------------------------

FLEET_NORM   = 32          # feature normaliser only -- NOT a cap
DROPOFF_NORM = 4           # feature normaliser only -- NOT a cap

HOME_TRIGGER_FRAC      = 0.80   # cargo fraction that sets the sticky homing flag
                                # (mirrors V71's `enough = 0.8` return threshold)
STUCK_NORM             = 6      # stuck-counter feature saturates here
ENDGAME_COLLAPSE_TURNS = 15     # resolver pile-on exemption window
ENDGAME_BUFFER         = 5      # force-home when turns_left <= dist + buffer


def dropoff_legal(bank, cell_h, cargo, cell_owned) -> bool:
    """Physical/affordability legality only -- the *decision* is learned."""
    if cell_owned:
        return False
    cost = max(0, DROPOFF_COST - (cell_h + cargo))
    return bank >= cost


def spawn_legal(bank, turns_left) -> bool:
    """Affordability only (plus don't burn 1000 on the very last turns when the
    ship can't even reach anything -- a 3-turn floor, not a strategy rule)."""
    return bank >= SHIP_COST and turns_left > 3


# ---------------------------------------------------------------------------
# Action space (6 discrete per-ship actions + a 2-way spawn action per turn)
# ---------------------------------------------------------------------------

ACTION_STAY    = 0
ACTION_NORTH   = 1
ACTION_EAST    = 2
ACTION_SOUTH   = 3
ACTION_WEST    = 4
ACTION_DROPOFF = 5
N_ACTIONS      = 6

ACTION_NAMES = ['STAY', 'NORTH', 'EAST', 'SOUTH', 'WEST', 'DROPOFF']

SPAWN_NO  = 0
SPAWN_YES = 1
N_SPAWN_ACTIONS = 2

# (dx, dy) for each movement action.  North decreases y (matches engine).
ACTION_DELTA = {
    ACTION_STAY:  (0, 0),
    ACTION_NORTH: (0, -1),
    ACTION_EAST:  (1, 0),
    ACTION_SOUTH: (0, 1),
    ACTION_WEST:  (-1, 0),
}

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
# Local map patch spec (same as rl_v8: 9x9x6, covers inspiration radius)
# ---------------------------------------------------------------------------

PATCH_RADIUS   = 4
PATCH_SIZE     = 2 * PATCH_RADIUS + 1    # 9
PATCH_CHANNELS = 6
PATCH_CHANNEL_NAMES = [
    'halite',            # 0: cell halite / MAX_HALITE
    'friendly_ship',     # 1: friendly ship present (binary)
    'enemy_ship',        # 2: enemy ship present (binary)
    'friendly_cargo',    # 3: friendly ship cargo / MAX_HALITE
    'enemy_cargo',       # 4: enemy ship cargo / MAX_HALITE
    'structure',         # 5: +1 my factory/dropoff, -1 enemy
]

# ---------------------------------------------------------------------------
# Coarse GLOBAL map spec (new in rl_v9)
#
# The whole map, recentred on the ship (or the factory for the spawn net),
# block-pooled down to GLOBAL_SIZE x GLOBAL_SIZE.  Gives the net the map-scale
# context the 9x9 patch cannot see (where the big halite regions are, where
# the enemy fleet is massed, where a dropoff would pay off).
# Halite map sizes (32/40/48/56/64) are all divisible by GLOBAL_SIZE=8.
# ---------------------------------------------------------------------------

GLOBAL_SIZE     = 8
GLOBAL_CHANNELS = 4
GLOBAL_CHANNEL_NAMES = [
    'halite',      # 0: block MEAN halite / MAX_HALITE
    'my_ships',    # 1: block SUM of friendly ships / 8, clipped to 1
    'enemy_ships', # 2: block SUM of enemy ships / 8, clipped to 1
    'structures',  # 3: block SUM of +1 my / -1 enemy structures, clipped to [-1,1]
]

# ---------------------------------------------------------------------------
# Per-ship scalar feature spec.  Order here == CSV column order == net input
# order.  Keep in lockstep with features.extract_scalars.
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # --- per-ship economy ---
    'cargo_frac',            # cargo / MAX_HALITE
    'cargo_to_full_frac',    # (MAX_HALITE - cargo) / MAX_HALITE
    'cargo_ge_home',         # 1 if cargo >= HOME_TRIGGER_FRAC * MAX_HALITE
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
    'my_ships_frac',         # my ship count / FLEET_NORM (normaliser, not a cap)
    'opp_ships_frac',        # opponent ship count / FLEET_NORM
    'my_bank_frac',          # my bank / BANK_SCALE (clamped)
    'opp_bank_frac',         # opponent bank / BANK_SCALE (clamped)
    'bank_margin_tanh',      # tanh((my_bank - opp_bank) / 5000)
    'winning_ratio',         # my_bank / (total_bank + 1)
    'num_dropoffs_frac',     # my dropoff count / DROPOFF_NORM
    'map_halite_frac',       # total map halite / (W*H*MAX_HALITE)
    'dropoff_affordable',    # 1 if bank >= net dropoff cost on this cell
    'dropoff_legal',         # 1 if dropoff_legal() for this ship right now
    'map_w_norm',            # W / 64
    'map_h_norm',            # H / 64
    # --- per-ship MEMORY (new in rl_v9; maintained by FleetMemory) ---
    'homing',                # sticky return flag (set at cargo>=0.8, cleared on deposit)
    'prev_stay',             # one-hot: last turn's executed action
    'prev_north',
    'prev_east',
    'prev_south',
    'prev_west',
    'stuck_frac',            # consecutive turns at same position / STUCK_NORM
]

N_SCALARS = len(FEATURE_NAMES)       # 46

BANK_SCALE = 20000.0

# Extra non-feature columns written to the CSV for inspection / traceability.
META_COLUMNS = ['row_id', 'game_id', 'player_id', 'ship_id', 'turn', 'action']

# ---------------------------------------------------------------------------
# Spawn-net scalar feature spec (global, one vector per turn)
# ---------------------------------------------------------------------------

SPAWN_FEATURE_NAMES = [
    'turn_frac',
    'turns_left_frac',
    'my_ships_norm',           # my ship count / FLEET_NORM
    'opp_ships_norm',          # opponent ship count / FLEET_NORM
    'ship_margin_tanh',        # tanh((my_ships - opp_ships) / 8)
    'my_bank_frac',            # / BANK_SCALE clamped
    'opp_bank_frac',
    'bank_margin_tanh',        # tanh((my_bank - opp_bank) / 5000)
    'winning_ratio',
    'map_halite_frac',         # total map halite / (W*H*MAX_HALITE)
    'halite_per_ship_norm',    # map halite / (ships_total+1) / MAX_HALITE / 8, clamped
    'my_dropoffs_norm',        # / DROPOFF_NORM
    'opp_dropoffs_norm',
    'factory_occupied',        # 1 if ANY ship currently sits on my factory
    'factory_area_halite',     # mean halite within radius 4 of factory / MAX_HALITE
    'spawn_affordable',        # 1 if bank >= SHIP_COST
    'map_w_norm',
    'map_h_norm',
]

N_SPAWN_SCALARS = len(SPAWN_FEATURE_NAMES)   # 18

# ---------------------------------------------------------------------------
# Reward weights (PPO)
# ---------------------------------------------------------------------------

REWARD_SCALE   = 0.01   # keeps returns O(tens)

# per-ship stream
W_DEP          = 1.0    # per halite deposited (bank gained) -- dominant signal
W_MINE         = 0.05   # per halite mined, only while cargo < 50% (anti-hoard)
W_DROPOFF_COST = 1.0    # per halite of net bank spend when converting to a dropoff
W_TRADE        = 1.0    # enemy-collision exchange: (enemy losses - own loss)
SHIP_VALUE_TURNS = 200  # a ship is worth SHIP_COST * min(1, turns_left/200)

# terminal (added to the last step of every ship's sequence + spawn stream)
W_WIN          = 200.0
W_MARGIN       = 400.0  # * tanh(margin / 3000)

# spawn stream
W_TEAM_DEP     = 0.25   # per halite the whole team deposits that turn
W_SPAWN_COST   = 1.0    # per halite of SHIP_COST when spawning


# ---------------------------------------------------------------------------
# Toroidal geometry helpers
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
