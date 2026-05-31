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
    Reward       (per step, per ship averaged)
                   + halite deposited this turn                  (primary signal)
                   + stay_scale × (1−cargo) if STAY and mined   (mine when empty)
                   + home_scale × cargo if HOME                  (return when full)
                   + explore_scale if RANDOM and bank_stuck      (break loops)
                   − collision_scale × (1+cargo) per p0 ship destroyed  (avoid crashes)
    Done         turn >= max_turns  or  engine._game_ended()
    """

    def __init__(
        self,
        width:              int  = 32,
        height:             int  = 32,
        num_players:        int  = 2,
        seed:               Optional[int] = None,
        opponent_policy:    str  = 'greedy',
        stay_scale:         float = 1.0,
        home_scale:         float = 2.0,
        explore_scale:      float = 0.5,
        explore_window:     int   = 30,
        collision_scale:    float = 20.0,
    ):
        self.width           = width
        self.height          = height
        self.num_players     = num_players
        self._seed           = seed
        self.opponent_policy = opponent_policy
        self.stay_scale      = stay_scale
        self.home_scale      = home_scale
        self.explore_scale   = explore_scale
        self.explore_window  = explore_window
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
        # Track cargo per ship for per-action shaping (snapshot at start of each turn)
        self._prev_cargo: Dict[int, int] = {}
        # Track when bank last grew (for explore bonus)
        self._last_deposit_turn: int = 0
        self._bank_at_check: int = 0
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
        # Reward
        # ------------------------------------------------------------------

        # Snapshot cargo BEFORE overwriting _prev_cargo (gives cargo at
        # the time each ship decided its action this turn).
        turn_start_cargo = self._prev_cargo
        self._prev_cargo = {sid: eng.entities[sid]['cargo']
                            for sid in eng.player_entities[0]}

        # 1. Deposit reward: halite actually banked this turn (primary signal).
        deposited_now  = eng._total_deposited.get(0, 0)
        deposit_reward = float(deposited_now - self._prev_deposited[0])
        self._prev_deposited[0] = deposited_now

        # Update bank-stuck tracker
        if deposited_now > self._bank_at_check:
            self._last_deposit_turn = eng.turn
            self._bank_at_check     = deposited_now
        bank_stuck = (eng.turn - self._last_deposit_turn) >= self.explore_window

        # 2. Per-action state-based shaping.
        #    Uses intent_actions (post-memory-override, pre-direction-resolution)
        #    so HOME reward fires on memory-driven turns too.
        #    STAY  → reward only if ship actually mined this turn (cargo increased).
        #            Scales with empty cargo — encourages mining when hold is empty.
        #            Guard against factory-camping hack: sitting on a cell with no
        #            halite gives zero cargo gain → zero stay reward.
        #    HOME  → reward scales with full cargo (encourage returning when full)
        #    RANDOM → exploration bonus when bank is stuck
        action_reward = 0.0
        n_acting = max(1, len(intent_actions))
        for sid, action in intent_actions.items():
            cargo_pre_raw = turn_start_cargo.get(sid, 0)
            cargo_pre     = cargo_pre_raw / MAX_HALITE  # 0–1
            if action == ACTION_STAY:
                # Only reward if the ship actually mined something this turn
                cargo_post_raw = self._prev_cargo.get(sid, cargo_pre_raw)
                if cargo_post_raw > cargo_pre_raw:
                    action_reward += self.stay_scale * max(0.0, 1.0 - cargo_pre)
            elif action == ACTION_HOME:
                action_reward += self.home_scale * cargo_pre
            elif action == ACTION_RANDOM and bank_stuck:
                action_reward += self.explore_scale

        # 3. Collision penalty: penalise every p0 ship destroyed this turn.
        #    Scales with lost cargo so the bot learns cargo loss is doubly bad.
        p0_ships_before = set(pre_p0_cargo.keys())
        destroyed_p0    = p0_ships_before - set(eng.player_entities[0].keys())
        collision_penalty = sum(
            self.collision_scale * (1.0 + pre_p0_cargo[sid] / MAX_HALITE)
            for sid in destroyed_p0
        )

        reward = deposit_reward + action_reward / n_acting - collision_penalty

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

        # Safety override: prevent ships moving into collisions.
        # Covers two cases in one pass:
        #   (a) Destination is current enemy ship position → STAY
        #   (b) Two friendly ships target the same cell → heaviest keeps going, others wait
        #   (c) Deposit stagger: same cell as own deposit handled by (b)
        _dir_delta = {
            ACTION_STAY: (0, 0), ACTION_NORTH: (0, -1), ACTION_SOUTH: (0, 1),
            ACTION_EAST: (1, 0), ACTION_WEST: (-1, 0),
        }
        enemy_positions = set()
        for pid in range(1, eng.num_players):
            for _, epos in eng.player_entities[pid].items():
                enemy_positions.add(epos)

        dest_map: Dict[int, tuple] = {}   # ship_id → destination cell
        for sid, prim in resolved_prims.items():
            sx, sy = eng.player_entities[0][sid]
            ddx, ddy = _dir_delta[prim]
            dest_map[sid] = ((sx + ddx) % W, (sy + ddy) % H)

        # (a) Enemy at destination → override to STAY
        for sid, dest in dest_map.items():
            if dest in enemy_positions:
                resolved_prims[sid] = ACTION_STAY
                dest_map[sid] = eng.player_entities[0][sid]   # update to current pos

        # (b) Multiple friendlies targeting same cell → heaviest keeps going
        cell_arrivals: Dict[tuple, list] = {}
        for sid, dest in dest_map.items():
            if dest != eng.player_entities[0][sid]:   # only ships that are actually moving
                cell_arrivals.setdefault(dest, []).append(
                    (eng.entities[sid]['cargo'], sid)
                )
        for dest, arrivals in cell_arrivals.items():
            if len(arrivals) > 1:
                arrivals.sort(reverse=True)   # heaviest cargo gets priority
                for _, sid in arrivals[1:]:
                    resolved_prims[sid] = ACTION_STAY

        tokens = [f"m {sid} {ACTION_TO_DIR[prim]}" for sid, prim in resolved_prims.items()]
        if spawn and eng.players[0]['energy'] >= SHIP_COST:
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
