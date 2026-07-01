#!/usr/bin/env python3
"""
rl_v7 / rl_env.py  --  Gym-style single-player RL environment.

Wraps the Python Halite engine for PPO training.  Player 0 is the rl_v7
agent; player 1 is a frozen rl_v5 opponent driven via FrozenV5.

The agent provides one action per live ship each turn.  The environment
handles spawning (rl_v5's economy rule) and collision resolution (the
deterministic resolver) internally, so the agent only ever chooses strategy.

Observation per ship-turn:
    scalars : float32[N_SCALARS]
    patch   : float32[PATCH_CHANNELS, PATCH_SIZE, PATCH_SIZE]  (CHW)
    mask    : bool[N_ACTIONS]

Reward per ship-turn:
    W_DEP * halite_deposited_by_ship_this_turn
    + potential shaping (Φ_t+1 - Φ_t)   where Φ = total_deposited_so_far / scale
  Terminal bonus (added once when game ends):
    ± W_WIN  (sign depends on whether agent won)
    + W_MARGIN * tanh(margin / 3000)
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE   = os.path.dirname(os.path.abspath(__file__))
MY_EXT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, MY_EXT)

from halite_engine import HaliteEngine              # noqa: E402
import config                                       # noqa: E402
from config import (                                # noqa: E402
    N_ACTIONS, ACTION_STAY, ACTION_DROPOFF, ACTION_TO_DIR, ACTION_DELTA,
    game_max_turns, target_dropoffs, spawn_econ_ok,
)
import features as featmod                          # noqa: E402
from resolver import resolve                        # noqa: E402

# RL reward weights
W_MINE   = 0.05     # per halite mined — only fires when cargo < 50%
W_DEP    = 1.0      # per halite deposited by own ship this turn (dominant)
W_DEATH  = 0.5      # penalty for dying with cargo
W_WIN    = 200.0    # terminal ±win bonus
W_MARGIN = 400.0    # terminal margin bonus (tanh-scaled)
REWARD_SCALE = 0.01  # keeps value targets O(tens), not O(thousands)

import math


class FrozenV5:
    """Drive the frozen rl_v5 bot in a subprocess (same protocol as run_game.py)."""

    def __init__(self, model_path: Optional[str] = None):
        rl_v5_dir = os.path.join(MY_EXT, 'rl_v5')
        if model_path is None:
            model_path = os.path.join(rl_v5_dir, 'checkpoints', 'model_final_weights.pt')
        v5_script = os.path.join(HERE, 'v5_opponent.py')
        self.cmd = (f'python -u "{v5_script}" --model "{model_path}" --deterministic')

    def get_cmd(self):
        return self.cmd


class HaliteEnvV7:
    """Wraps HaliteEngine for single-agent PPO (player 0 = rl_v7)."""

    def __init__(self, width=32, height=32, seed=None,
                 v5_model_path=None, agent_pid=0):
        self.width  = width
        self.height = height
        self.seed   = seed
        self.agent_pid = agent_pid
        self.v5_cmd = FrozenV5(v5_model_path).get_cmd()

        self.engine: Optional[HaliteEngine] = None
        self._prev_deposited = 0
        self._current_wv = None

    # ------------------------------------------------------------------
    def reset(self, seed=None):
        """Reset the environment and return the first observations."""
        if seed is not None:
            self.seed = seed
        actual_seed = self.seed
        self.engine = HaliteEngine(
            width=self.width, height=self.height,
            num_players=2, seed=actual_seed, verbose=False,
        )
        self.engine._init_map_and_players()
        self._prev_deposited = 0
        self._current_wv = None
        return None   # first obs built on first step

    def close(self):
        """Nothing to clean up (engine is pure Python)."""
        pass

    # ------------------------------------------------------------------
    # Running a full game and collecting (obs, act, reward, ...) tuples
    # The env runs the FULL game using the engine's subprocess bot for
    # the opponent, and our policy for player 0.
    # Returns a dict of trajectory lists.
    # ------------------------------------------------------------------

    def run_episode(self, policy_fn, deterministic=False):
        """Run one complete game.  policy_fn(wv, sid) -> action_idx.

        Returns:
            trajectory: list of dicts, one per ship-turn, with keys:
                scalars, patch, mask, action, log_prob, value, reward,
                done, ship_id, turn.
            info: dict with final_deposited_agent, final_deposited_opp, winner.
        """
        import subprocess, json

        engine = HaliteEngine(
            width=self.width, height=self.height,
            num_players=2, seed=self.seed, verbose=False,
        )
        engine._init_map_and_players()

        # Launch opponent bot subprocess
        import os as _os
        env = {**_os.environ, 'PYTHONUNBUFFERED': '1'}
        opp = subprocess.Popen(
            self.v5_cmd, shell=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, env=env,
        )
        # Handshake: send init message to opponent (pid=1)
        opp.stdin.write(engine._init_message(1))
        opp.stdin.flush()
        _opp_name = opp.stdout.readline().strip()   # receive bot name

        trajectory = []
        prev_deposited = 0

        for turn in range(1, engine.max_turns + 1):
            engine._update_inspiration()
            engine.turn = turn

            # Build agent WorldView BEFORE commands
            agent_pid = self.agent_pid
            wv = featmod.world_view_from_engine(engine, agent_pid)

            # per-ship observations and intents
            ship_data = {}   # sid -> (scalars, patch, mask, action, log_prob, value)
            intents = {}
            for sid in list(engine.player_entities[agent_pid].keys()):
                scal = featmod.extract_scalars(wv, sid)
                patch = featmod.extract_patch(wv, sid)
                mask = featmod.action_mask(wv, sid)
                act, lp, val = policy_fn(wv, sid, scal, patch, mask)
                intents[sid] = act
                ship_data[sid] = (scal, patch, mask, act, lp, val)

            # resolver -> final actions
            want_spawn = _want_spawn_engine(engine, agent_pid)
            final, spawn_issued, dropoff_sid = resolve(wv, intents, want_spawn)

            # Build command string for player 0
            agent_cmds = _build_cmd_str(engine, agent_pid, final,
                                        spawn_issued, dropoff_sid)

            # get opponent command
            turn_msg = engine._turn_message()
            try:
                opp.stdin.write(turn_msg)
                opp.stdin.flush()
                opp_line = opp.stdout.readline()
                opp_cmds = opp_line.strip() if opp_line else ''
            except OSError:
                opp_cmds = ''

            # process commands
            engine._current_events = []
            engine.changed_cells.clear()
            engine._moved_entities.clear()

            # snapshot cargo before commands (to measure deposits)
            pre_cmd_cargo = {
                sid: engine.entities[sid]['cargo']
                for sid in engine.player_entities[agent_pid]
            }

            engine._process_commands({agent_pid: agent_cmds, 1: opp_cmds})

            # per-ship deposit: cargo drop on surviving ships (dead ships get 0)
            ship_deposited = {}
            for sid, pre in pre_cmd_cargo.items():
                if sid in engine.player_entities[agent_pid]:
                    post = engine.entities[sid]['cargo']
                    ship_deposited[sid] = max(0, pre - post)
                else:
                    ship_deposited[sid] = 0  # ship died, no deposit credit

            # snapshot cargo before mining (to measure harvest)
            pre_mine_cargo = {
                sid: engine.entities[sid]['cargo']
                for sid in engine.player_entities[agent_pid]
            }

            engine._process_mining()

            # per-ship mining: cargo gain on surviving ships
            ship_mined = {}
            for sid, pre in pre_mine_cargo.items():
                if sid in engine.player_entities[agent_pid]:
                    post = engine.entities[sid]['cargo']
                    ship_mined[sid] = max(0, post - pre)
                else:
                    ship_mined[sid] = 0

            done = engine._game_ended() or turn == engine.max_turns
            term_bonus = 0.0
            if done:
                my_final = engine.players[agent_pid]['energy']
                opp_final = engine.players[1]['energy']
                won = my_final > opp_final
                margin = my_final - opp_final
                term_bonus = ((W_WIN if won else -W_WIN)
                              + W_MARGIN * math.tanh(margin / 3000)) * REWARD_SCALE

            n_ships = max(1, len(ship_data))
            # record trajectory: each ship gets its own shaped reward
            for sid, (scal, patch, mask, act, lp, val) in ship_data.items():
                # pre-mine cargo_frac decides conditional mine reward
                pre_cargo = pre_mine_cargo.get(sid, 0)
                cargo_frac = pre_cargo / 1000.0
                mine_rew = (W_MINE * ship_mined.get(sid, 0) * REWARD_SCALE
                            if cargo_frac < 0.5 else 0.0)
                dep_rew = W_DEP * ship_deposited.get(sid, 0) * REWARD_SCALE
                # death penalty: ship was alive before commands but died
                death_rew = (-W_DEATH * pre_cmd_cargo.get(sid, 0) * REWARD_SCALE
                             if sid not in engine.player_entities[agent_pid] else 0.0)
                trajectory.append({
                    'scalars': scal,
                    'patch': patch,
                    'mask': mask,
                    'action': act,
                    'log_prob': lp,
                    'value': val,
                    'reward': dep_rew + mine_rew + death_rew + term_bonus / n_ships,
                    'done': done,
                    'ship_id': sid,
                    'turn': turn,
                })

            # update engine stats
            for pid in engine.players:
                num_ships = len(engine.player_entities[pid])
                if num_ships > engine._ships_peak[pid]:
                    engine._ships_peak[pid] = num_ships

            if done:
                break

        # cleanup
        try:
            opp.stdin.close()
            opp.terminate()
            opp.wait(timeout=2)
        except Exception:
            pass

        info = {
            'final_deposited_agent': engine.players[agent_pid]['energy'],
            'final_deposited_opp':   engine.players[1]['energy'],
            'winner': agent_pid if engine.players[agent_pid]['energy'] >
                      engine.players[1]['energy'] else 1,
            'turns': turn,
        }
        return trajectory, info


def _want_spawn_engine(engine, pid: int) -> bool:
    bank = engine.players[pid]['energy']
    n_ships = len(engine.player_entities[pid])
    turns_left = engine.max_turns - engine.turn
    num_dropoffs = len(engine.players[pid]['dropoffs'])
    W, H = engine.width, engine.height
    tgt = target_dropoffs(W, H)
    return spawn_econ_ok(bank, n_ships, turns_left, num_dropoffs, tgt)


def _build_cmd_str(engine, pid: int, final: Dict[int, int],
                   spawn_issued: bool, dropoff_sid: Optional[int]) -> str:
    parts = []
    for sid, act in final.items():
        dir_ch = ACTION_TO_DIR[act]
        if dir_ch != 'o':
            parts.append(f"m {sid} {dir_ch}")
    if dropoff_sid is not None:
        parts.append(f"c {dropoff_sid}")
    if spawn_issued:
        parts.append("g")
    return ' '.join(parts)
