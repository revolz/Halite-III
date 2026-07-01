"""
rl_v6 configuration — the PURE action/observation space.

rl_v6 is a deliberately *rule-free* bot.  Where rl_v5 wrapped its network in a
finite-state machine + logit prior + homing/prospect/collision/spawn rules, rl_v6
keeps ONLY the network: pure features in, a primitive move out, executed verbatim.

Observation (reused from rl_features, the BASE block only — no FSM block):
    14 spatial channels  (extract_spatial_*)
    29 base scalars      (extract_scalars_*, == N_BASE_SCALAR_FEATURES)

Per-ship action space (6, vs rl_v5's 9):
    0 STAY, 1 NORTH, 2 EAST, 3 SOUTH, 4 WEST   (same primitive indices as rl_v5)
    5 DROPOFF                                   (convert ship -> dropoff)
The rl_v5 meta-actions RANDOM/HOME/PROSPECT do NOT exist here — the net must emit
the resolved primitive directly (that is the whole point of the experiment).

Spawn is a SEPARATE learned binary decision (SpawnHead), fed a compact per-turn
*global* feature vector.  This replaces rl_v5's `spawn_econ_ok` economic rule.
"""

import math

import numpy as np

from rl_features import (
    N_BASE_SCALAR_FEATURES,            # 29
    N_SPATIAL_CHANNELS,                # 14
    WINDOW_SIZE,                       # 11
    ACTION_STAY, ACTION_NORTH, ACTION_EAST, ACTION_SOUTH, ACTION_WEST,
    SHIP_COST,
    torus_delta, torus_dist, least_cost_home_step, _richest_in_prospect_window,
    compute_home_cost_field, effective_dest,
)

# ── pure observation dims (no FSM block) ────────────────────────────────────
N_SCALARS_V6 = N_BASE_SCALAR_FEATURES   # 29

# ── HIERARCHICAL macro action space (2026-06-27 redesign) ───────────────────
# The net picks a high-level macro; proven navigation functions resolve it to a
# primitive move.  Indices 0-5 are unchanged from the old 6-action space so an
# existing checkpoint's actor head transfers 1:1 (HOME/PROSPECT rows are new).
#   0 STAY/MINE, 1 N, 2 E, 3 S, 4 W, 5 DROPOFF   (as before)
#   6 HOME      -> least-cost step toward a deposit (Dijkstra)
#   7 PROSPECT  -> step toward the richest cell in the window
# Still 100% RL: the NET chooses the macro; we only reuse pathfinding mechanics.
ACTION_DROPOFF_V6  = 5
ACTION_HOME_V6     = 6
ACTION_PROSPECT_V6 = 7
N_SHIP_ACTIONS_V6  = 8

# Primitive (non-dropoff) action -> (dx, dy) on the torus.
ACTION_DELTA = {
    ACTION_STAY:  (0, 0),
    ACTION_NORTH: (0, -1),
    ACTION_SOUTH: (0, 1),
    ACTION_EAST:  (1, 0),
    ACTION_WEST:  (-1, 0),
}


def _step_toward(sx, sy, tx, ty, W, H):
    """Primitive action (0-4) that moves one step from (sx,sy) toward (tx,ty)."""
    dx, dy = torus_delta(sx, sy, tx, ty, W, H)
    if dx == 0 and dy == 0:
        return ACTION_STAY
    if abs(dx) >= abs(dy):
        return ACTION_EAST if dx > 0 else ACTION_WEST
    return ACTION_SOUTH if dy > 0 else ACTION_NORTH


def resolve_macro(action, sx, sy, halite_dict, cost_field, W, H):
    """Resolve a macro action to a PRIMITIVE move action (0-4).

    Primitives 0-4 pass through unchanged.  HOME -> least-cost step toward a
    deposit (reuses least_cost_home_step over a precomputed cost_field).
    PROSPECT -> step toward the richest cell in the window.  DROPOFF is handled
    by the caller and never passed here.
    """
    if action == ACTION_HOME_V6:
        return least_cost_home_step(sx, sy, cost_field, W, H)
    if action == ACTION_PROSPECT_V6:
        rx, ry, _ = _richest_in_prospect_window(sx, sy, halite_dict, W, H)
        return _step_toward(sx, sy, rx, ry, W, H)
    return action


# ── collision resolution (2026-06-28 last-chance fix) ───────────────────────
# Self-collisions were the #1 killer.  Resolve them at execution: process ships
# heaviest-cargo-first; each ship takes the FIRST of [desired, stay, N,E,S,W]
# whose destination cell is not already claimed by a heavier ship and not an
# enemy ship's cell.  Guarantees every ship occupies a UNIQUE cell -> no
# friendly collisions, no ramming a stationary enemy.  Strategy stays learned;
# this only stops self-destruction (mechanics, like the reused navigation).
_RESOLVE_ORDER = (ACTION_STAY, ACTION_NORTH, ACTION_EAST, ACTION_SOUTH, ACTION_WEST)


def resolve_collisions(desired_prim, sx, sy, cell_h, cargo, claimed, enemy_cells, W, H):
    """Pick a collision-free primitive for one ship; returns (prim, dest, moved).
    `claimed` is the set of cells already taken this turn (mutated by caller)."""
    for act in (desired_prim,) + _RESOLVE_ORDER:
        dest, moved = effective_dest(act, sx, sy, cell_h, cargo, W, H)
        if dest not in claimed and dest not in enemy_cells:
            return act, dest, moved
    return ACTION_STAY, (sx, sy), False     # boxed in (rare) -> stay


# ── macro prior: bias toward the sensible high-level move (rl_v5-style FSM, but
# only a PRIOR — the net can override).  Makes movement intelligent immediately
# via the proven HOME/PROSPECT navigation; PPO refines.
_HOME_CARGO   = 900
_MIN_HARVEST  = 40
_ENDGAME_BUF  = 6


def suggest_macro(cargo, cell_h, dist_to_deposit, turns_left):
    """The macro a competent player would pick (HOME/MINE/PROSPECT)."""
    if cargo > 0 and turns_left <= dist_to_deposit + _ENDGAME_BUF:
        return ACTION_HOME_V6
    if cargo >= _HOME_CARGO:
        return ACTION_HOME_V6
    if cell_h >= _MIN_HARVEST:
        return ACTION_STAY            # mine here
    return ACTION_PROSPECT_V6         # seek richer halite


def macro_prior(cargo, cell_h, dist_to_deposit, turns_left, beta=100.0):
    """8-dim logit prior: +beta on the suggested macro (0 elsewhere)."""
    pr = np.zeros(N_SHIP_ACTIONS_V6, dtype=np.float32)
    pr[suggest_macro(cargo, cell_h, dist_to_deposit, turns_left)] = beta
    return pr

# ── learned spawn head ──────────────────────────────────────────────────────
SPAWN_FEATURE_DIM = 8
_MAX_HALITE = 1000


def spawn_global_features(
    turn: int,
    max_turns: int,
    my_bank: int,
    opp_bank: int,
    my_ships: int,
    opp_ships: int,
    map_halite_total: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Compact per-turn global feature vector for the learned spawn decision.

    Deliberately small and bank/economy-oriented: spawning is a once-per-turn,
    whole-fleet decision, so it doesn't need the per-ship spatial window.
    """
    mt          = max(1, max_turns)
    turns_left  = max_turns - turn
    cells       = max(1, width * height)
    bank_margin = math.tanh((my_bank - opp_bank) / 5000.0)
    return np.array([
        turn / mt,
        min(my_bank / _MAX_HALITE, 50.0) / 50.0,        # bank, soft-capped
        min(my_ships / 30.0, 1.0),
        min(opp_ships / 30.0, 1.0),
        map_halite_total / (cells * _MAX_HALITE),        # halite remaining frac
        turns_left / mt,
        bank_margin,
        1.0 if my_bank >= SHIP_COST else 0.0,
    ], dtype=np.float32)
