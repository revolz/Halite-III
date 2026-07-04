#!/usr/bin/env python3
"""
rl_v9 / rl_env.py  --  RL environment: full games vs a frozen opponent bot.

Player 0 is the rl_v9 agent (in-process policy); player 1 is any bot speaking
the standard hlt stdin/stdout protocol in a subprocess (V71 by default).

Differences vs rl_v8 (all aimed at "PPO must not make the bot worse"):

* Trajectories are returned PER SHIP (dict ship_id -> ordered step list) plus
  a separate per-turn spawn stream.  rl_v8 flattened all ships interleaved,
  so GAE bootstrapped each ship's advantage from a DIFFERENT ship's value --
  that noise is the prime suspect for PPO degrading the BC policy.
* Every ship's sequence ends with done=1 (on death, conversion, or game end).
* Spawning is a learned action (SpawnPolicy) -- no hand-coded fleet cap.
* Enemy collisions are allowed by the resolver; the reward prices the trade:
      + (enemy cargo + enemy ship value)  - (own cargo + own ship value)
  where ship value = SHIP_COST * min(1, turns_left / SHIP_VALUE_TURNS).
  A wreck on our OWN structure banks the cargo (endgame pile-on): that cargo
  is credited as a deposit instead of penalised.
* Dropoff conversion is priced honestly: reward = (cell_h + cargo - cost),
  i.e. the actual bank delta of the construct; the payoff arrives later as
  deposit rewards at the new dropoff.

Reward summary (x REWARD_SCALE):
  ship step  : W_DEP*deposited + W_MINE*mined [if cargo<50%]
               + dropoff bank delta + collision trade + terminal (last step)
  spawn step : W_TEAM_DEP*team_deposited_this_turn - W_SPAWN_COST*SHIP_COST
               [if spawned] + terminal (last step)
"""

import math
import os
import subprocess
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
    SPAWN_YES, SHIP_COST, DROPOFF_COST,
    REWARD_SCALE, W_DEP, W_MINE, W_DROPOFF_COST, W_TRADE, SHIP_VALUE_TURNS,
    W_WIN, W_MARGIN, W_TEAM_DEP, W_SPAWN_COST,
)
import features as featmod                          # noqa: E402
from features import FleetMemory                    # noqa: E402
from resolver import resolve                        # noqa: E402

V71_SCRIPT = os.path.join(MY_EXT, 'Year 2019', 'MyBot - V71', 'MyBot.py')
RL_V5_DEFAULT_MODEL = os.path.join(MY_EXT, 'rl_v5', 'checkpoints', 'best.pt')


def opponent_cmd_for(name: str, model_path: Optional[str] = None) -> str:
    if name == 'v71':
        return f'python -u "{V71_SCRIPT}"'
    elif name == 'rl_v5':
        mp = model_path or RL_V5_DEFAULT_MODEL
        script = os.path.join(MY_EXT, 'rl_v5', 'rl_bot.py')
        return f'python -u "{script}" --model "{mp}" --deterministic'
    elif name == 'rl_v8':
        mp = model_path or os.path.join(MY_EXT, 'rl_v8', 'checkpoints', 'best.pt')
        script = os.path.join(MY_EXT, 'rl_v8', 'rl_bot.py')
        return f'python -u "{script}" --model "{mp}" --deterministic'
    else:
        raise ValueError(f"unknown opponent {name!r}")


class HaliteEnvV9:
    """Runs full games; player 0 = rl_v9 (in-process policy)."""

    def __init__(self, width=32, height=32, seed=None, opponent_cmd=None):
        self.width  = width
        self.height = height
        self.seed   = seed
        self.agent_pid = 0
        self.opp_pid = 1
        self.opponent_cmd = opponent_cmd or opponent_cmd_for('v71')

    # ------------------------------------------------------------------
    def run_episode(self, ship_policy_fn, spawn_policy_fn):
        """Run one complete game.

        ship_policy_fn(scals, patches, gmaps, masks) -> (acts, lps, vals)
            batched over the fleet: numpy arrays in, three sequences out.
        spawn_policy_fn(sscal, sglob, smask) -> (act, lp, val)

        Returns (ship_trajs, spawn_traj, info):
            ship_trajs: dict sid -> ordered list of step dicts
                        (scalars, patch, gmap, mask, action, log_prob, value,
                         reward, done, turn)
            spawn_traj: ordered list of step dicts
                        (scalars, gmap, mask, action, log_prob, value,
                         reward, done, turn)
            info: final banks, winner, wreck/kill/dropoff/spawn counts.
        """
        engine = HaliteEngine(
            width=self.width, height=self.height,
            num_players=2, seed=self.seed, verbose=False,
        )
        engine._init_map_and_players()

        env_os = {**os.environ, 'PYTHONUNBUFFERED': '1'}
        opp = subprocess.Popen(
            self.opponent_cmd, shell=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env_os,
        )
        opp.stdin.write(engine._init_message(self.opp_pid))
        opp.stdin.flush()
        opp.stdout.readline()               # opponent name

        apid = self.agent_pid
        mem = FleetMemory()
        ship_trajs: Dict[int, List[dict]] = {}
        spawn_traj: List[dict] = []
        n_my_wrecks = n_enemy_kills = n_dropoffs_built = n_spawns = 0
        peak_ships = 0

        turn = 0
        for turn in range(1, engine.max_turns + 1):
            engine._update_inspiration()
            engine.turn = turn

            wv = featmod.world_view_from_engine(engine, apid)
            mem.begin_turn(wv)

            # ---- per-ship observations + batched policy call ----
            sids = list(engine.player_entities[apid].keys())
            peak_ships = max(peak_ships, len(sids))
            intents: Dict[int, int] = {}
            ship_data = {}
            if sids:
                scals = np.stack([featmod.extract_scalars(wv, s, mem) for s in sids])
                patches = np.stack([featmod.extract_patch(wv, s) for s in sids])
                gmaps = np.stack([featmod.extract_ship_global(wv, s) for s in sids])
                masks = np.stack([featmod.action_mask(wv, s) for s in sids])
                acts, lps, vals = ship_policy_fn(scals, patches, gmaps, masks)
                for i, sid in enumerate(sids):
                    intents[sid] = int(acts[i])
                    ship_data[sid] = (scals[i], patches[i], gmaps[i], masks[i],
                                      int(acts[i]), float(lps[i]), float(vals[i]))

            # ---- learned spawn decision ----
            sscal = featmod.extract_spawn_scalars(wv)
            sglob = featmod.extract_spawn_global(wv)
            smask = featmod.spawn_mask(wv)
            s_act, s_lp, s_val = spawn_policy_fn(sscal, sglob, smask)
            want_spawn = (int(s_act) == SPAWN_YES)

            # ---- resolve friendly conflicts ----
            final, spawn_issued, dropoff_sid = resolve(wv, intents, want_spawn)
            mem.commit_actions(final)
            if dropoff_sid is not None:
                mem.commit_actions({dropoff_sid: ACTION_DROPOFF})

            agent_cmds = _build_cmd_str(final, spawn_issued, dropoff_sid)

            # ---- opponent commands ----
            turn_msg = engine._turn_message()
            try:
                opp.stdin.write(turn_msg)
                opp.stdin.flush()
                opp_line = opp.stdout.readline()
                opp_cmds = opp_line.strip() if opp_line else ''
            except OSError:
                opp_cmds = ''

            # ---- process turn ----
            engine._current_events = []
            engine.changed_cells.clear()
            engine._moved_entities.clear()

            # snapshots for reward computation
            pre_entities = {eid: (e['owner'], e['cargo'])
                            for eid, e in engine.entities.items()}
            pre_deposited = engine._total_deposited[apid]
            pre_bank = engine.players[apid]['energy']
            turns_left = engine.max_turns - turn
            ship_value = SHIP_COST * min(1.0, turns_left / SHIP_VALUE_TURNS)

            engine._process_commands({apid: agent_cmds, self.opp_pid: opp_cmds})

            # --- deposits by surviving ships (cargo drop across commands) ---
            ship_deposited: Dict[int, int] = {}
            for sid in ship_data:
                if sid in engine.player_entities[apid]:
                    post = engine.entities[sid]['cargo']
                    pre = pre_entities[sid][1]
                    ship_deposited[sid] = max(0, pre - post)

            # --- dropoff construct pricing (actual bank delta) ---
            dropoff_reward: Dict[int, float] = {}
            for ev in engine._current_events:
                if ev.get('type') == 'construct' and ev.get('owner_id') == apid:
                    sid = ev['id']
                    if sid in pre_entities:
                        n_dropoffs_built += 1
                        cargo = pre_entities[sid][1]
                        # engine zeroed the cell; reconstruct credit from cost:
                        # bank delta = credit - cost, both derivable pre-turn,
                        # but cell halite was consumed -- use recorded values:
                        # credit = cell_h + cargo; cost = max(0, 4000 - credit)
                        # We can't read cell_h post-hoc, so approximate via
                        # bank flow: it is exact when bank >= cost (always true
                        # for issued constructs).
                        # bank_delta = credit - cost
                        # Simplest exact path: engine credited then deducted;
                        # we recompute from the pre-command halite snapshot.
                        cell_xy = (ev['location']['x'], ev['location']['y'])
                        cell_h = wv.halite.get(cell_xy, 0)
                        credit = cell_h + cargo
                        cost = max(0, DROPOFF_COST - credit)
                        dropoff_reward[sid] = (credit - cost) * W_DROPOFF_COST * REWARD_SCALE

            # --- collisions: price the exchange ---
            death_reward: Dict[int, float] = {}
            dead_sids = set()
            for ev in engine._current_events:
                if ev.get('type') != 'shipwreck':
                    continue
                ships = ev.get('ships', [])
                ours = [e for e in ships
                        if e in pre_entities and pre_entities[e][0] == apid]
                theirs = [e for e in ships
                          if e in pre_entities and pre_entities[e][0] != apid]
                if not ours:
                    continue
                loc = (ev['location']['x'], ev['location']['y'])
                on_our_structure = engine.cell_owner.get(loc) == apid
                enemy_loss = sum(pre_entities[e][1] + ship_value for e in theirs)
                n_enemy_kills += len(theirs)
                for sid in ours:
                    dead_sids.add(sid)
                    n_my_wrecks += 1
                    cargo = pre_entities[sid][1]
                    if on_our_structure:
                        # cargo banked by the engine -> deposit credit, and the
                        # hull loss is real but usually an endgame sacrifice
                        own_loss = ship_value
                        dep_credit = W_DEP * cargo
                    else:
                        own_loss = cargo + ship_value
                        dep_credit = 0.0
                    r = (dep_credit
                         + W_TRADE * (enemy_loss / len(ours) - own_loss))
                    death_reward[sid] = r * REWARD_SCALE

            # --- mining (conditional reward) ---
            pre_mine_cargo = {
                sid: engine.entities[sid]['cargo']
                for sid in engine.player_entities[apid]
            }
            engine._process_mining()
            ship_mined: Dict[int, int] = {}
            for sid, pre in pre_mine_cargo.items():
                if sid in engine.player_entities[apid]:
                    ship_mined[sid] = max(0, engine.entities[sid]['cargo'] - pre)

            done = engine._game_ended() or turn == engine.max_turns
            term_bonus = 0.0
            if done:
                my_final = engine.players[apid]['energy']
                opp_final = engine.players[self.opp_pid]['energy']
                won = my_final > opp_final
                margin = my_final - opp_final
                term_bonus = ((W_WIN if won else -W_WIN)
                              + W_MARGIN * math.tanh(margin / 3000.0)) * REWARD_SCALE

            # ---- record ship steps ----
            for sid, (scal, patch, gmap, mask, act, lp, val) in ship_data.items():
                alive = sid in engine.player_entities[apid]
                converted = sid in dropoff_reward
                ship_done = done or (not alive)
                pre_cargo = pre_mine_cargo.get(sid, pre_entities[sid][1])
                mine_rew = (W_MINE * ship_mined.get(sid, 0) * REWARD_SCALE
                            if pre_cargo < 0.5 * config.MAX_HALITE else 0.0)
                r = (W_DEP * ship_deposited.get(sid, 0) * REWARD_SCALE
                     + mine_rew
                     + dropoff_reward.get(sid, 0.0)
                     + death_reward.get(sid, 0.0))
                step = {
                    'scalars': scal, 'patch': patch, 'gmap': gmap, 'mask': mask,
                    'action': act, 'log_prob': lp, 'value': val,
                    'reward': r, 'done': ship_done, 'turn': turn,
                }
                ship_trajs.setdefault(sid, []).append(step)

            # ---- record spawn step ----
            # The policy's CHOSEN action is recorded with its own log-prob;
            # a resolver veto (factory blocked) is environment dynamics.
            actually_spawned = spawn_issued and (pre_bank >= SHIP_COST)
            if actually_spawned:
                n_spawns += 1
            team_dep = engine._total_deposited[apid] - pre_deposited
            s_r = (W_TEAM_DEP * team_dep
                   - (W_SPAWN_COST * SHIP_COST if actually_spawned else 0.0)) * REWARD_SCALE
            spawn_traj.append({
                'scalars': sscal, 'gmap': sglob, 'mask': smask,
                'action': int(s_act),
                'log_prob': s_lp, 'value': s_val,
                'reward': s_r, 'done': done, 'turn': turn,
            })

            if done:
                break

        # Terminal bonus onto sequences that reached the final turn.  Ships
        # that died mid-game already received their outcome signal (the death
        # trade); crediting them with the end-of-game result across a long
        # gap would only add advantage noise.
        for sid, steps in ship_trajs.items():
            if steps[-1]['turn'] == turn:
                steps[-1]['reward'] += term_bonus
            steps[-1]['done'] = True
        if spawn_traj:
            spawn_traj[-1]['reward'] += term_bonus
            spawn_traj[-1]['done'] = True

        try:
            opp.stdin.close()
            opp.terminate()
            opp.wait(timeout=2)
        except Exception:
            pass

        my_final = engine.players[apid]['energy']
        opp_final = engine.players[self.opp_pid]['energy']
        info = {
            'final_deposited_agent': my_final,
            'final_deposited_opp': opp_final,
            'winner': apid if my_final > opp_final else self.opp_pid,
            'turns': turn,
            'my_wrecks': n_my_wrecks,
            'enemy_kills': n_enemy_kills,
            'dropoffs_built': n_dropoffs_built,
            'spawns': n_spawns,
            'peak_ships': peak_ships,
        }
        return ship_trajs, spawn_traj, info


def _build_cmd_str(final: Dict[int, int], spawn_issued: bool,
                   dropoff_sid: Optional[int]) -> str:
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
