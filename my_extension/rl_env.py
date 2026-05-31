"""
Gym-style Halite III environment for RL training.

The learning agent controls player 0; all other players use a scripted
opponent policy (random, greedy, or idle).

Usage
-----
    env = HaliteEnv(width=32, height=32)
    obs, info = env.reset()
    done = False
    while not done:
        # obs: {ship_id: (spatial[W,W,C], scalars[S])}
        spawn = len(obs) < 6 and env.engine.players[0]['energy'] >= 1000
        actions = {sid: random.randint(0, 4) for sid in obs}
        obs, reward, done, info = env.step(actions, spawn=spawn)
"""

import os
import sys
import random
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from halite_engine import (
    HaliteEngine,
    SHIP_COST, MAX_HALITE, DIRECTIONS,
)
from rl_features import (
    extract_spatial_from_engine,
    extract_scalars_from_engine,
    torus_dist,
    torus_delta,
    _nearest_deposit,
    ACTION_TO_DIR,
    ACTION_STAY,
    ACTION_NORTH,
    ACTION_SOUTH,
    ACTION_EAST,
    ACTION_WEST,
    ACTION_HOME,
    ACTION_RANDOM,
    N_SHIP_ACTIONS,
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class HaliteEnv:
    """
    Single-agent (player 0) Halite III environment.

    Observation  {ship_id: (spatial float32[W,W,C], scalars float32[S])}
    Action       {ship_id: int in [0, N_SHIP_ACTIONS)}
                   0=Stay/Mine  1=N  2=E  3=S  4=W  5=RANDOM  6=HOME
    Reward       (per step)
                   + sum(cargo_after − cargo_before) for each surviving ship
                   + halite deposited this turn   (so depositing ships aren't penalised)
                   − (collision_scale + cargo_lost) for each p0 ship destroyed
    Done         turn >= max_turns  or  engine._game_ended()
    """

    def __init__(
        self,
        width:              int  = 32,
        height:             int  = 32,
        num_players:        int  = 2,
        seed:               Optional[int] = None,
        opponent_policy:    str  = 'greedy',
        collision_scale:    float = 20.0,
    ):
        self.width           = width
        self.height          = height
        self.num_players     = num_players
        self._seed           = seed
        self.opponent_policy = opponent_policy
        self.collision_scale = collision_scale
        self.engine: Optional[HaliteEngine] = None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None):
        s = seed if seed is not None else self._seed
        self.engine = HaliteEngine(
            self.width, self.height, self.num_players,
            seed=s, verbose=False,
        )
        self.engine._init_map_and_players()
        self.engine.turn = 0

        # Track deposited halite across steps
        self._prev_deposited = {pid: 0 for pid in self.engine.players}
        # Cargo snapshot at start of each turn (for reward calculation)
        self._prev_cargo: Dict[int, int] = {}
        # Ships currently committed to going home (memory mechanism)
        self._homing_ships: set = set()

        # Pre-compute inspiration for turn-0 state (no ships → no-op but correct)
        self.engine._update_inspiration()

        obs  = self._get_obs()
        info = {'turn': 0, 'max_turns': self.engine.max_turns}
        return obs, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        ship_actions: Dict[int, int],
        spawn: bool = False,
    ) -> Tuple[dict, float, bool, dict]:
        """
        Execute one game turn.

        Inspiration has already been updated at the end of the previous step
        (or in reset), so we do NOT call it again at the top of this method.

        Returns
        -------
        obs     : {ship_id: (spatial, scalars)} for surviving player-0 ships
        reward  : float
        done    : bool
        info    : dict
        """
        eng        = self.engine
        eng.turn  += 1
        eng._current_events = []
        eng.changed_cells.clear()
        eng._moved_entities.clear()

        # Snapshot owner map BEFORE commands (destroyed ships are removed)
        pre_ship_owners = {eid: eng.entities[eid]['owner'] for eid in eng.entities}
        # Snapshot p0 cargo before commands for collision penalty calculation
        pre_p0_cargo = {sid: eng.entities[sid]['cargo']
                        for sid, owner in pre_ship_owners.items() if owner == 0}

        # Build commands
        p0_cmd, intent_actions = self._build_p0_commands(ship_actions, spawn)
        all_cmds  = {0: p0_cmd}
        for pid in range(1, eng.num_players):
            all_cmds[pid] = self._opponent_command(pid)

        eng._process_commands(all_cmds)
        eng._process_mining()

        # Per-player statistics
        for pid in eng.players:
            if eng.player_entities[pid] or eng.players[pid]['energy'] >= SHIP_COST:
                eng._last_turn_alive[pid] = eng.turn
            n = len(eng.player_entities[pid])
            if n > eng._ships_peak[pid]:
                eng._ships_peak[pid] = n

        # ------------------------------------------------------------------
        # Reward  (v8: per-ship cargo delta + deposited − collision penalty)
        # ------------------------------------------------------------------

        # Cargo snapshot at start of this turn (set in previous step / reset)
        turn_start_cargo = self._prev_cargo
        # Update for next turn
        self._prev_cargo = {sid: eng.entities[sid]['cargo']
                            for sid in eng.player_entities[0]}

        # 1. Sum of (cargo_after − cargo_before) for all surviving ships.
        #    Positive when mining, negative when paying movement costs.
        reward = 0.0
        for sid, cargo_after in self._prev_cargo.items():
            cargo_before = turn_start_cargo.get(sid, 0)
            reward += cargo_after - cargo_before

        # 2. Add halite deposited this turn so depositing ships aren't penalised.
        #    (cargo drops by X when depositing, but bank gains X — net zero for deposit)
        deposited_now = eng._total_deposited.get(0, 0)
        deposited_this_turn = float(deposited_now - self._prev_deposited[0])
        self._prev_deposited[0] = deposited_now
        reward += deposited_this_turn

        # 3. Collision penalty: fixed cost + full cargo lost for every p0 ship destroyed.
        p0_ships_before   = set(pre_p0_cargo.keys())
        destroyed_p0      = p0_ships_before - set(eng.player_entities[0].keys())
        collision_penalty = sum(
            self.collision_scale + pre_p0_cargo[sid]
            for sid in destroyed_p0
        )
        reward -= collision_penalty

        # ------------------------------------------------------------------
        # Done
        # ------------------------------------------------------------------
        done = (eng.turn >= eng.max_turns) or eng._game_ended()

        # ------------------------------------------------------------------
        # Update inspiration for the NEXT step's observation
        # (mirrors engine behaviour: inspiration computed before frame is shown)
        # ------------------------------------------------------------------
        if not done:
            eng._update_inspiration()

        obs  = self._get_obs()
        info = {
            'turn':       eng.turn,
            'max_turns':  eng.max_turns,
            'bank_p0':    eng.players[0]['energy'],
            'ships_p0':   len(eng.player_entities[0]),
        }
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Command helpers
    # ------------------------------------------------------------------

    def _build_p0_commands(self, ship_actions: Dict[int, int], spawn: bool):
        """Build command string for player 0 and return (cmd_str, intent_actions).

        intent_actions maps ship_id → action after home-memory override but
        before direction resolution (still contains ACTION_HOME / ACTION_RANDOM).
        This is what the reward block must use so HOME reward fires on memory turns.
        """
        eng    = self.engine
        W, H   = eng.width, eng.height
        intent_actions: Dict[int, int] = {}
        resolved_prims: Dict[int, int] = {}

        for ship_id, action in ship_actions.items():
            if ship_id not in eng.player_entities[0]:
                continue
            # Auto-trigger home if cargo exceeds 60% capacity
            cargo = eng.entities[ship_id]['cargo']
            if cargo >= MAX_HALITE * 0.6:
                self._homing_ships.add(ship_id)
            # Home memory: if this ship is committed to returning, keep it going
            if ship_id in self._homing_ships:
                action = ACTION_HOME
            intent_actions[ship_id] = action
            # Resolve meta-actions to primitive 0–4
            if action == ACTION_RANDOM:
                action = random.randint(0, 4)
            elif action == ACTION_HOME:
                self._homing_ships.add(ship_id)
                action = self._home_dir(ship_id)
                if action == ACTION_STAY:   # arrived at deposit — cancel home mode
                    self._homing_ships.discard(ship_id)
            resolved_prims[ship_id] = action

        # ── Safety override: 4-phase collision prevention ────────────────────
        _dir_delta = {
            ACTION_STAY: (0, 0), ACTION_NORTH: (0, -1), ACTION_SOUTH: (0, 1),
            ACTION_EAST: (1, 0), ACTION_WEST: (-1, 0),
        }
        # Cardinal-only deltas with their corresponding action index.
        _cardinal = [(0, -1, ACTION_NORTH), (1, 0, ACTION_EAST),
                     (0,  1, ACTION_SOUTH), (-1, 0, ACTION_WEST)]

        # Phase 1 — Build enemy threat zone.
        # Covers "enemy stays" AND "enemy moves adjacent": current + 4 neighbours.
        enemy_threat_zone: set = set()
        for pid in range(1, eng.num_players):
            for _, epos in eng.player_entities[pid].items():
                ex, ey = epos
                enemy_threat_zone.add((ex, ey))
                for ddx, ddy, _ in _cardinal:
                    enemy_threat_zone.add(((ex + ddx) % W, (ey + ddy) % H))

        # Phase 2 — Compute initial destinations from resolved primitive actions.
        dest_map: Dict[int, tuple] = {}
        for sid, prim in resolved_prims.items():
            sx, sy = eng.player_entities[0][sid]
            ddx, ddy = _dir_delta[prim]
            dest_map[sid] = ((sx + ddx) % W, (sy + ddy) % H)

        # Phase 3a — Enemy avoidance (MOVE): ships heading into threat zone → STAY.
        for sid in list(resolved_prims.keys()):
            if dest_map[sid] in enemy_threat_zone:
                resolved_prims[sid] = ACTION_STAY
                dest_map[sid] = eng.player_entities[0][sid]

        # Phase 3b — Enemy avoidance (ESCAPE): a ship already at a threat-zone cell
        # while staying there risks collision if an adjacent enemy moves in.
        # Try to move to any safe adjacent cell instead of staying in the danger zone.
        for sid in list(resolved_prims.keys()):
            sx, sy = eng.player_entities[0][sid]
            if resolved_prims[sid] == ACTION_STAY and (sx, sy) in enemy_threat_zone:
                for ddx, ddy, act in _cardinal:
                    esc = ((sx + ddx) % W, (sy + ddy) % H)
                    if esc not in enemy_threat_zone:
                        resolved_prims[sid] = act
                        dest_map[sid] = esc
                        break   # first safe direction wins; cascade handles friendlies

        # Phase 4 — Friendly collision resolution: no two ships may share a final cell.
        # Rules (applied via cascade until stable):
        #   Stayer (dest == current_pos) owns that cell — every mover is forced STAY.
        #   Multiple movers to same empty cell → heaviest mover wins, rest STAY.
        # A forced STAY creates a new stayer, potentially blocking another ship;
        # iterating up to N times guarantees convergence for any chain of N ships.
        ship_list = list(resolved_prims.keys())
        for _ in range(len(ship_list)):
            pos_occupants: Dict[tuple, list] = {}
            for sid in ship_list:
                pos_occupants.setdefault(dest_map[sid], []).append(
                    (eng.entities[sid]['cargo'], sid)
                )
            changed = False
            for dest, occupants in pos_occupants.items():
                if len(occupants) <= 1:
                    continue
                stayers = [(c, sid) for c, sid in occupants
                           if dest == eng.player_entities[0][sid]]
                movers  = [(c, sid) for c, sid in occupants
                           if dest != eng.player_entities[0][sid]]
                if stayers:
                    for _, sid in movers:
                        resolved_prims[sid] = ACTION_STAY
                        dest_map[sid] = eng.player_entities[0][sid]
                        changed = True
                elif len(movers) > 1:
                    movers.sort(reverse=True)
                    for _, sid in movers[1:]:
                        resolved_prims[sid] = ACTION_STAY
                        dest_map[sid] = eng.player_entities[0][sid]
                        changed = True
            if not changed:
                break

        tokens = [f"m {sid} {ACTION_TO_DIR[prim]}" for sid, prim in resolved_prims.items()]
        fx, fy = eng.players[0]['factory']
        adjacent_to_factory = {
            ((fx + ddx) % W, (fy + ddy) % H)
            for ddx, ddy in [(0, -1), (0, 1), (1, 0), (-1, 0)]
        }
        friendly_positions = set(eng.player_entities[0].values())
        factory_safe = not (friendly_positions & (adjacent_to_factory | {(fx, fy)}))
        if spawn and eng.players[0]['energy'] >= SHIP_COST and factory_safe:
            tokens.append('g')
        return ' '.join(tokens), intent_actions

    def _home_dir(self, ship_id: int) -> int:
        """Return the primitive action (0–4) that moves one step toward the
        nearest deposit structure (factory or dropoff)."""
        eng    = self.engine
        sx, sy = eng.player_entities[0][ship_id]
        nx, ny = _nearest_deposit(sx, sy, eng, 0)
        if (sx, sy) == (nx, ny):
            return ACTION_STAY
        dx, dy = torus_delta(sx, sy, nx, ny, eng.width, eng.height)
        if abs(dx) >= abs(dy):
            return ACTION_EAST if dx > 0 else ACTION_WEST
        else:
            return ACTION_NORTH if dy < 0 else ACTION_SOUTH

    def _opponent_command(self, pid: int) -> str:
        eng    = self.engine
        tokens = []

        if self.opponent_policy == 'idle':
            return ''

        for ship_id, (sx, sy) in list(eng.player_entities[pid].items()):
            if self.opponent_policy == 'random':
                direction = random.choice(list(ACTION_TO_DIR.values()))
                tokens.append(f"m {ship_id} {direction}")
            elif self.opponent_policy == 'greedy':
                tokens.append(f"m {ship_id} {self._greedy_dir(pid, ship_id, sx, sy)}")

        if eng.players[pid]['energy'] >= SHIP_COST and len(eng.player_entities[pid]) < 10:
            tokens.append('g')

        return ' '.join(tokens)

    def _greedy_dir(self, pid: int, ship_id: int, sx: int, sy: int) -> str:
        """Simple greedy: return home when full or time-critical, else mine."""
        eng        = self.engine
        cargo      = eng.entities[ship_id]['cargo']
        factory    = eng.players[pid]['factory']
        turns_left = eng.max_turns - eng.turn
        dist_home  = torus_dist(sx, sy, factory[0], factory[1], eng.width, eng.height)

        if cargo >= MAX_HALITE * 0.9 or (cargo > 0 and turns_left <= dist_home + 5):
            return self._dir_toward(sx, sy, factory[0], factory[1])

        cell_h = eng.halite.get((sx, sy), 0)
        if cell_h > MAX_HALITE * 0.1:
            return 'o'  # stay and mine

        best_h, best_dir = cell_h, 'o'
        for d, (ddx, ddy) in [('n', (0, -1)), ('s', (0, 1)), ('e', (1, 0)), ('w', (-1, 0))]:
            nx = (sx + ddx) % eng.width
            ny = (sy + ddy) % eng.height
            h  = eng.halite.get((nx, ny), 0)
            if h > best_h:
                best_h, best_dir = h, d
        return best_dir

    def _dir_toward(self, sx: int, sy: int, tx: int, ty: int) -> str:
        W, H = self.engine.width, self.engine.height
        dx   = tx - sx
        dy   = ty - sy
        if abs(dx) > W // 2:
            dx = dx - W if dx > 0 else dx + W
        if abs(dy) > H // 2:
            dy = dy - H if dy > 0 else dy + H
        if abs(dx) >= abs(dy):
            return 'e' if dx > 0 else ('w' if dx < 0 else 'o')
        return 's' if dy > 0 else ('n' if dy < 0 else 'o')

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        obs = {}
        for ship_id in self.engine.player_entities[0]:
            spatial = extract_spatial_from_engine(self.engine, ship_id, pid=0)
            scalars = extract_scalars_from_engine(self.engine, ship_id, pid=0)
            obs[ship_id] = (spatial, scalars)
        return obs

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def close(self):
        self.engine = None

    def get_action_meanings(self):
        return ['Stay/Mine', 'North', 'East', 'South', 'West']
