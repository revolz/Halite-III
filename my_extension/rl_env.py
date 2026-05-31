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
    ACTION_TO_DIR,
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
                   0=Stay/Mine  1=N  2=E  3=S  4=W
    Reward       (per step)
                   + halite deposited this turn          (direct objective progress)
                   + cargo_reward_scale × halite mined  (partial credit for potential)
                   − death_penalty_scale × cargo lost   (proportional to actual loss)
                 (terminal bonus)
                   + 0.01 × total halite deposited this game
    Done         turn >= max_turns  or  engine._game_ended()
    """

    def __init__(
        self,
        width:              int  = 32,
        height:             int  = 32,
        num_players:        int  = 2,
        seed:               Optional[int] = None,
        opponent_policy:    str  = 'greedy',
        death_penalty_scale: float = 0.5,
        cargo_reward_scale: float = 0.3,
    ):
        self.width               = width
        self.height              = height
        self.num_players         = num_players
        self._seed               = seed
        self.opponent_policy     = opponent_policy
        self.death_penalty_scale = death_penalty_scale
        self.cargo_reward_scale  = cargo_reward_scale
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
        # Track cargo per ship for dense mining reward
        self._prev_cargo: Dict[int, int] = {}

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

        # Build commands
        p0_cmd    = self._build_p0_commands(ship_actions, spawn)
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

        # 1. Deposit reward: halite actually banked this turn (primary signal).
        deposited_now  = eng._total_deposited.get(0, 0)
        deposit_reward = float(deposited_now - self._prev_deposited[0])
        self._prev_deposited[0] = deposited_now

        # 2. Cargo (mining) reward: partial credit for halite collected this
        #    turn.  Scaled < 1.0 because cargo is only *potential* — it must
        #    still be deposited to count toward the real objective.
        cargo_gained = 0.0
        for sid in eng.player_entities[0]:
            cur  = eng.entities[sid]['cargo']
            prev = self._prev_cargo.get(sid, 0)
            if cur > prev:
                cargo_gained += cur - prev

        # Snapshot cargo BEFORE overwriting _prev_cargo so the death penalty
        # below can look up the cargo each ship was carrying when it died.
        turn_start_cargo = self._prev_cargo
        self._prev_cargo = {sid: eng.entities[sid]['cargo']
                            for sid in eng.player_entities[0]}

        # 3. Death penalty: proportional to the cargo actually lost.
        #    A ship dying with 0 cargo is barely penalised; dying with 900
        #    halite is penalised heavily.  Using `death_penalty_scale × cargo`
        #    (not a flat constant) avoids the failure mode where the bot learns
        #    to sit still and never risk movement.
        cargo_lost = 0.0
        for ev in eng._current_events:
            if ev['type'] == 'shipwreck':
                for sid in ev['ships']:
                    if pre_ship_owners.get(sid) == 0:
                        cargo_lost += turn_start_cargo.get(sid, 0)

        reward = (deposit_reward
                  + self.cargo_reward_scale * cargo_gained
                  - self.death_penalty_scale * cargo_lost)

        # ------------------------------------------------------------------
        # Done
        # ------------------------------------------------------------------
        done = (eng.turn >= eng.max_turns) or eng._game_ended()

        if done:
            # Terminal bonus: modest reward proportional to total halite
            # banked over the whole game.  This anchors the critic's
            # long-range value estimates to the actual episode outcome so
            # that early strategic actions (spawn, dropoff) can be properly
            # credited through GAE back-propagation.
            total_deposited = eng._total_deposited.get(0, 0)
            reward += total_deposited * 0.01

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

    def _build_p0_commands(self, ship_actions: Dict[int, int], spawn: bool) -> str:
        eng    = self.engine
        tokens = []
        for ship_id, action in ship_actions.items():
            if ship_id not in eng.player_entities[0]:
                continue
            tokens.append(f"m {ship_id} {ACTION_TO_DIR[action]}")
        if spawn and eng.players[0]['energy'] >= SHIP_COST:
            tokens.append('g')
        return ' '.join(tokens)

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
