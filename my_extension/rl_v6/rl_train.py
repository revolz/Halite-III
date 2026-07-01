#!/usr/bin/env python3
"""
PPO fine-tuning for rl_v6 — push the pure policy past rl_v5.

Warm-starts from the BC/DAgger policy and refines player-0's PURE movement policy
with PPO against a frozen opponent (rl_v5 by default, the bot to beat; or rl_v4 /
greedy).  No FSM, no logit prior — the network's raw logits are the policy.

Spawn: the BC-trained spawn head is loaded FROZEN and decides spawning each turn,
so this stage focuses the policy gradient on movement (the hard part).  Spawn can
be refined separately later; keeping it fixed avoids a second, noisier gradient.

Reward (HaliteEnvV6): deposited-anchored, scaled by reward_scale for learning so
value targets stay O(tens) — the entropy-collapse guard hard-won in rl_v4/v5.
Watch training_log.csv: deposited rising, mean_entropy NOT trending to 0, and a
non-uniform action distribution.

Usage
-----
    python rl_train.py --resume checkpoints/model_weights.pt \
        --spawn checkpoints/spawn_weights.pt --opponent rl_v5 \
        --episodes 4000 --checkpoint-dir checkpoints/
"""

import argparse
import csv
import os
import sys
import time
from collections import deque
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from rl_env      import HaliteEnvV6
from rl_model    import ActorCritic
from rl_features import effective_dest, compute_home_cost_field
from spawn_model import SpawnHead
from experts     import FrozenBotDriver
from rl_config   import (N_SCALARS_V6, N_SHIP_ACTIONS_V6, ACTION_DROPOFF_V6,
                         resolve_macro, SHIP_COST)


class ShipTrajectory:
    __slots__ = ('sp', 'sc', 'mk', 'ac', 'rw', 'vl', 'lp', 'dn')

    def __init__(self):
        self.sp, self.sc, self.mk, self.ac = [], [], [], []
        self.rw, self.vl, self.lp, self.dn = [], [], [], []

    def add(self, sp, sc, mk, ac, rw, vl, lp, dn):
        self.sp.append(sp); self.sc.append(sc); self.mk.append(mk); self.ac.append(ac)
        self.rw.append(rw); self.vl.append(vl); self.lp.append(lp); self.dn.append(dn)


class SpawnTrajectory:
    """Per-turn spawn decisions for one episode (Bernoulli action + value)."""
    __slots__ = ('feat', 'a', 'lp', 'vl', 'rw', 'dn')

    def __init__(self):
        self.feat, self.a, self.lp, self.vl, self.rw, self.dn = [], [], [], [], [], []

    def add(self, feat, a, lp, vl):
        self.feat.append(feat); self.a.append(a); self.lp.append(lp); self.vl.append(vl)
        self.rw.append(0.0); self.dn.append(False)


def compute_gae(traj, gamma, lam):
    T = len(traj.rw); adv = [0.0] * T; gae = 0.0; next_val = 0.0
    for t in reversed(range(T)):
        m = 1.0 - float(traj.dn[t])
        delta = traj.rw[t] + gamma * next_val * m - traj.vl[t]
        gae = delta + gamma * lam * m * gae
        adv[t] = gae
        next_val = traj.vl[t]
    return adv, [a + v for a, v in zip(adv, traj.vl)]


# 2026-06-28 stability pass (macro run collapsed to a single-action South-spam,
# entropy -> 0): gentler lr, much stronger entropy floor, and — the big one —
# multiple games per PPO update (games_per_update) to kill single-game variance.
DEFAULTS = dict(
    gamma=0.999, lam=0.95, clip_eps=0.2, vf_coef=0.5,
    # entropy: ent_floor 0.5/coef 0.2 over-regularised (policy went random,
    # deposited->0); games_per_update is the real anti-collapse lever, so dial
    # entropy back to a soft floor that still lets the policy EXPLOIT depositing.
    ent_coef=0.01, ent_floor=0.3, ent_floor_coef=0.08,
    reward_scale=0.01, lr=1e-4, n_epochs=4, minibatch_size=128,
    max_grad_norm=0.5, games_per_update=6,
)


def _last_logged_episode(log_path: str) -> int:
    """Episode number on the LAST data row of training_log.csv (0 if none).

    Uses the last row, not the max, so we continue the MOST RECENT session even if
    an earlier run left higher numbers behind (the log can contain restarts)."""
    if not os.path.isfile(log_path):
        return 0
    last = 0
    with open(log_path, newline='') as f:
        for row in csv.reader(f):
            if row and row[0].lstrip('-').isdigit():
                last = int(row[0])
    return last


class PPOTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg['device'])
        if cfg.get('resume') and os.path.isfile(cfg['resume']):
            # load_expand transfers all of the body/critic and the overlapping
            # action rows, so a 6-action checkpoint warm-starts the 8-action
            # (macro) head; HOME/PROSPECT rows start fresh.
            self.model = ActorCritic.load_expand(
                cfg['resume'], device=cfg['device'],
                n_scalars=N_SCALARS_V6, n_actions=N_SHIP_ACTIONS_V6).to(self.device)
            print(f"Warm-started (expand) policy from {cfg['resume']}")
        else:
            self.model = ActorCritic(n_scalars=N_SCALARS_V6,
                                     n_actions=N_SHIP_ACTIONS_V6).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg['lr'])
        self.train_spawn = cfg.get('train_spawn', True)
        self.spawn_head = None
        self.spawn_optim = None
        if cfg.get('spawn') and os.path.isfile(cfg['spawn']):
            self.spawn_head = SpawnHead.load(cfg['spawn'], device=cfg['device']).to(self.device)
            if self.train_spawn:
                self.spawn_head.train()
                self.spawn_optim = torch.optim.Adam(self.spawn_head.parameters(), lr=cfg['lr'])
                print(f"Loaded TRAINABLE spawn head from {cfg['spawn']} (learning to spawn)")
            else:
                self.spawn_head.eval()
                print(f"Loaded frozen spawn head from {cfg['spawn']}")
        self._ep_rewards = deque(maxlen=100)

    def _spawn_decision(self, env):
        eng = env.engine
        if eng.players[0]['energy'] < 1000:
            return False
        if self.spawn_head is None:
            return len(eng.player_entities[0]) < 12 and (eng.max_turns - eng.turn) > 100
        feats = torch.from_numpy(env.spawn_features(0)).to(self.device)
        return self.spawn_head.spawn_prob(feats) >= 0.5

    def _collect(self, env):
        env.reset()
        done = False; ep_r = 0.0
        counts = [0] * N_SHIP_ACTIONS_V6
        diag = dict(mined=0.0, move_cost=0.0, collisions=0, offences=0)
        active: Dict[int, ShipTrajectory] = {}
        finished: List[ShipTrajectory] = []
        sptraj = SpawnTrajectory()
        while not done:
            eng = env.engine
            W, H = eng.width, eng.height
            # Sequential heaviest-cargo-first decode WITH the Flavor-A overlay, so
            # train == inference: each ship sees teammates already-committed cells.
            ships = sorted(eng.player_entities[0].keys(),
                           key=lambda s: (-eng.entities[s]['cargo'], s))
            committed = set(); vacated = set(); acts = {}; offenders = set()
            # Per-turn macro context: HOME needs a least-cost field to the deposits.
            deposits = [eng.players[0]['factory']]
            deposits += [(dx, dy) for _d, dx, dy in eng.players[0]['dropoffs']]
            cost_field = compute_home_cost_field(eng.halite, deposits, W, H)
            for sid in ships:
                sp, sc, mk = env.obs_for_ship(sid, committed, vacated)
                active.setdefault(sid, ShipTrajectory())
                spt = torch.from_numpy(sp).to(self.device)
                sct = torch.from_numpy(sc).to(self.device)
                a, lp, v = self.model.select_action(spt, sct, mask=mk)
                counts[a] += 1
                # Store the MACRO the net chose (for PPO); execute the RESOLVED
                # primitive (HOME/PROSPECT -> a move via the reused nav functions).
                active[sid].add(sp, sc, mk, a, 0.0, v, lp, False)
                sx, sy = eng.player_entities[0][sid]
                prim = (ACTION_DROPOFF_V6 if a == ACTION_DROPOFF_V6
                        else resolve_macro(a, sx, sy, eng.halite, cost_field, W, H))
                acts[sid] = prim
                dest, moved = effective_dest(prim, sx, sy, eng.halite.get((sx, sy), 0),
                                             eng.entities[sid]['cargo'], W, H)
                if dest in committed:           # a teammate already claimed this cell
                    offenders.add(sid)
                committed.add(dest)
                if moved:
                    vacated.add((sx, sy))
            # Spawn decision (learned).  Only a real choice when the bank can
            # afford it; record it so the spawn head can be PPO-trained.
            spawn = False; spawn_recorded = False
            if self.spawn_head is not None and eng.players[0]['energy'] >= SHIP_COST:
                feat = env.spawn_features(0)
                if self.train_spawn:
                    sa, slp, sval = self.spawn_head.select_spawn(
                        torch.from_numpy(feat).to(self.device))
                    spawn = bool(sa)
                    sptraj.add(feat, float(sa), slp, sval); spawn_recorded = True
                else:
                    spawn = self.spawn_head.spawn_prob(
                        torch.from_numpy(feat).to(self.device)) >= 0.5
            elif self.spawn_head is None:
                spawn = (len(eng.player_entities[0]) < 12
                         and (eng.max_turns - eng.turn) > 100)

            obs, r, done, info = env.step(acts, spawn=spawn, offenders=offenders)
            ep_r += r
            if spawn_recorded:
                sptraj.rw[-1] = r * self.cfg['reward_scale']
                sptraj.dn[-1] = done
            sr = info['ship_rewards']
            cur = set(acts.keys())
            for sid in cur:
                if active[sid].rw:
                    active[sid].rw[-1] = sr.get(sid, 0.0) * self.cfg['reward_scale']
            for k in diag:
                diag[k] += info[k]
            for sid in cur - set(obs.keys()):
                if active[sid].dn:
                    active[sid].dn[-1] = True
                    finished.append(active.pop(sid))
        for tr in active.values():
            if tr.rw:
                tr.dn[-1] = True
            finished.append(tr)
        return [t for t in finished if t.rw], ep_r, counts, info, diag, sptraj

    def _update(self, trajs):
        cfg = self.cfg
        sp, sc, mk, ac, ret, adv, lp = [], [], [], [], [], [], []
        for tr in trajs:
            a, r = compute_gae(tr, cfg['gamma'], cfg['lam'])
            sp += tr.sp; sc += tr.sc; mk += tr.mk; ac += tr.ac
            ret += r; adv += a; lp += tr.lp
        if not sp:
            return 0.0
        sp_t = torch.tensor(np.array(sp), dtype=torch.float32, device=self.device)
        sc_t = torch.tensor(np.array(sc), dtype=torch.float32, device=self.device)
        mk_t = torch.tensor(np.array(mk), dtype=torch.bool, device=self.device)
        ac_t = torch.tensor(ac, dtype=torch.long, device=self.device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
        lp_t = torch.tensor(lp, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        N = len(sp); bs = cfg['minibatch_size']
        ent_floor = torch.tensor(cfg['ent_floor'], device=self.device)
        tot_ent = tot_n = 0
        for _ in range(cfg['n_epochs']):
            idx = torch.randperm(N, device=self.device)
            for s in range(0, N, bs):
                mb = idx[s:s + bs]
                lp_new, val, ent = self.model.evaluate(sp_t[mb], sc_t[mb], ac_t[mb], masks=mk_t[mb])
                ratio = torch.exp(lp_new - lp_t[mb])
                a_mb = adv_t[mb]
                pg = torch.max(-a_mb * ratio,
                               -a_mb * ratio.clamp(1 - cfg['clip_eps'], 1 + cfg['clip_eps'])).mean()
                vf = F.mse_loss(val, ret_t[mb])
                mean_ent = ent.mean()
                floor_loss = cfg['ent_floor_coef'] * F.relu(ent_floor - mean_ent)
                loss = pg + cfg['vf_coef'] * vf - cfg['ent_coef'] * mean_ent + floor_loss
                self.optim.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg['max_grad_norm'])
                self.optim.step()
                tot_ent += mean_ent.item() * mb.shape[0]; tot_n += mb.shape[0]
        return tot_ent / max(tot_n, 1)

    def _update_spawn(self, sptrajs):
        """PPO update for the spawn head over per-turn spawn decisions."""
        if not self.train_spawn or self.spawn_optim is None:
            return
        cfg = self.cfg
        feat, act, ret, adv, lp = [], [], [], [], []
        for tr in sptrajs:
            if not tr.rw:
                continue
            a, r = compute_gae(tr, cfg['gamma'], cfg['lam'])
            feat += tr.feat; act += tr.a; ret += r; adv += a; lp += tr.lp
        if not feat:
            return
        ft  = torch.tensor(np.array(feat), dtype=torch.float32, device=self.device)
        at  = torch.tensor(act,  dtype=torch.float32, device=self.device)
        ret = torch.tensor(ret,  dtype=torch.float32, device=self.device)
        adv = torch.tensor(adv,  dtype=torch.float32, device=self.device)
        lpt = torch.tensor(lp,   dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        N = len(feat); bs = cfg['minibatch_size']
        for _ in range(cfg['n_epochs']):
            idx = torch.randperm(N, device=self.device)
            for s in range(0, N, bs):
                mb = idx[s:s + bs]
                lp_new, val, ent = self.spawn_head.evaluate(ft[mb], at[mb])
                ratio = torch.exp(lp_new - lpt[mb])
                a_mb = adv[mb]
                pg = torch.max(-a_mb * ratio,
                               -a_mb * ratio.clamp(1 - cfg['clip_eps'], 1 + cfg['clip_eps'])).mean()
                vf = F.mse_loss(val, ret[mb])
                loss = pg + cfg['vf_coef'] * vf - cfg['ent_coef'] * ent.mean()
                self.spawn_optim.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.spawn_head.parameters(), cfg['max_grad_norm'])
                self.spawn_optim.step()

    def train(self):
        cfg = self.cfg
        os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
        opp = None
        if cfg['opponent'] in ('rl_v5', 'rl_v4'):
            opp = FrozenBotDriver(cfg['opponent'])
            print(f"Opponent = frozen {cfg['opponent']} bot")
        log_path = os.path.join(cfg['checkpoint_dir'], 'training_log.csv')
        new = not os.path.isfile(log_path)

        # Episode numbering: default (<=0) auto-continues from the last logged
        # episode so a resume keeps training_log.csv contiguous without the user
        # having to look the number up.  An explicit --start-episode wins.
        start = cfg['start_episode']
        if start <= 0:
            start = _last_logged_episode(log_path) + 1
            if not new:
                print(f"Auto-continuing from episode {start} "
                      f"(last logged {start - 1})")

        lf = open(log_path, 'a', newline=''); lw = csv.writer(lf)
        names = ['stay', 'north', 'east', 'south', 'west', 'dropoff', 'home', 'prospect']
        if new:
            lw.writerow(['episode', 'reward', 'deposited', 'ships_end',
                         'mean_entropy'] + [f'act_{n}' for n in names]
                        + ['mined', 'move_cost', 'collisions', 'offences'])

        total = start + cfg['episodes'] - 1
        gpu = max(1, int(cfg['games_per_update']))
        t0 = time.time()
        traj_buf = []          # trajectories accumulated across games_per_update games
        sptraj_buf = []
        last_ent = 0.0
        for ep in range(start, total + 1):
            env = HaliteEnvV6(cfg['width'], cfg['height'], seed=cfg['seed'],
                              opponent_policy='greedy', allow_dropoff=True)
            if opp is not None:
                env.opponent_command_fn = lambda e, pid, _o=opp: _o.command(e.engine, pid)
            trajs, ep_r, counts, info, diag, sptraj = self._collect(env)
            traj_buf.extend(trajs)
            sptraj_buf.append(sptraj)
            # PPO update only once every `gpu` games -> a much larger, lower-variance
            # batch (the single-game updates collapsed the policy to one action).
            if (ep - start + 1) % gpu == 0 or ep == total:
                last_ent = self._update(traj_buf)
                self._update_spawn(sptraj_buf)
                traj_buf = []; sptraj_buf = []
            self._ep_rewards.append(ep_r)
            dep = info.get('deposited_p0', 0)
            lw.writerow([ep, round(ep_r, 1), dep, info['ships_p0'], round(last_ent, 4)] + counts
                        + [round(diag['mined'], 1), round(diag['move_cost'], 1),
                           diag['collisions'], diag['offences']])
            if ep % 10 == 0 or ep == total:
                lf.flush()
                avg = sum(self._ep_rewards) / len(self._ep_rewards)
                rate = (ep - start + 1) / (time.time() - t0)
                print(f"ep {ep}/{total}  reward={ep_r:.0f} avg100={avg:.0f}  "
                      f"deposited={dep}  ships={info['ships_p0']}  ent={last_ent:.3f}  "
                      f"({rate:.1f} ep/s)", flush=True)
            if ep % cfg['checkpoint_interval'] == 0 or ep == total:
                self.model.save(os.path.join(cfg['checkpoint_dir'], 'model_weights.pt'))
                # Also keep a numbered snapshot so a good policy is never overwritten
                # away (deposited can peak mid-run then drift).
                self.model.save(os.path.join(cfg['checkpoint_dir'], f'model_ep{ep}.pt'))
                if self.train_spawn and self.spawn_head is not None:
                    self.spawn_head.save(os.path.join(cfg['checkpoint_dir'], 'spawn_weights.pt'))
                    self.spawn_head.save(os.path.join(cfg['checkpoint_dir'], f'spawn_ep{ep}.pt'))
        lf.close()
        self.model.save(os.path.join(cfg['checkpoint_dir'], 'model_weights.pt'))
        if self.train_spawn and self.spawn_head is not None:
            self.spawn_head.save(os.path.join(cfg['checkpoint_dir'], 'spawn_weights.pt'))
        print("Done.")


def main():
    ap = argparse.ArgumentParser(description='PPO fine-tune rl_v6')
    ap.add_argument('--resume', default=os.path.join(_HERE, 'checkpoints', 'model_weights.pt'))
    ap.add_argument('--spawn', default=os.path.join(_HERE, 'checkpoints', 'spawn_weights.pt'))
    ap.add_argument('--opponent', choices=['rl_v5', 'rl_v4', 'greedy'], default='rl_v5')
    ap.add_argument('--episodes', type=int, default=4000)
    ap.add_argument('--checkpoint-dir', dest='checkpoint_dir',
                    default=os.path.join(_HERE, 'checkpoints'))
    ap.add_argument('--checkpoint-interval', type=int, default=50)
    ap.add_argument('--games-per-update', dest='games_per_update', type=int, default=6,
                    help='games collected per PPO update (>1 reduces variance)')
    ap.add_argument('--no-train-spawn', dest='train_spawn', action='store_false',
                    help='keep the spawn head frozen (default: train it with PPO)')
    ap.set_defaults(train_spawn=True)
    ap.add_argument('--start-episode', type=int, default=0,
                    help='0/omitted = auto-continue from last logged episode; '
                         'set explicitly to override')
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--height', type=int, default=32)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    cfg = dict(DEFAULTS)
    cfg.update(vars(args))
    PPOTrainer(cfg).train()


if __name__ == '__main__':
    main()
