#!/usr/bin/env python3
"""
rl_v9 / rl_train.py  --  PPO fine-tuning from the BC warm start, vs V71.

Everything here is engineered around one observation from rl_v8: the BC
policy already played V71 close to even, and PPO then made it WORSE.  The
fixes:

1. PER-SHIP GAE.  rl_v8 flattened all ships' steps interleaved by turn and
   bootstrapped values across ship boundaries -- advantage noise.  Here every
   ship's sequence (and the spawn stream) gets its own GAE pass.
2. SEPARATE value networks (fresh, not sharing the BC policy trunk), plus a
   value-only WARMUP phase before any policy gradient is taken, so early
   garbage advantages never touch the BC policy.
3. KL EARLY STOPPING per update (target-kl) + BC-anchor KL regularisation
   (annealed), so the policy cannot run away from the BC solution faster
   than the reward justifies.
4. EVAL GATING: every eval-interval episodes, play deterministic games vs
   V71 and keep best.pt = the best evaluated checkpoint ever.  PPO can then
   never end worse than BC silently.

Usage:
    python rl_v9/rl_train.py --bc-ckpt rl_v9/checkpoints/bc.pt --episodes 300 --device cuda
    python rl_v9/rl_train.py --resume auto --episodes 200
"""

import argparse
import copy
import csv
import glob as globmod
import math
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import config                                        # noqa: E402
from net import (ShipPolicy, ShipValue, SpawnPolicy, SpawnValue,   # noqa: E402
                 save_bundle, load_bundle, mask_logits, NEG_INF)
from rl_env import HaliteEnvV9, opponent_cmd_for     # noqa: E402

# --- PPO hyperparameters ---
GAMMA          = 0.999
LAM            = 0.95
CLIP_EPS       = 0.2
TARGET_KL      = 0.02
N_EPOCHS       = 3
MINIBATCH      = 256
MAX_GRAD_NORM  = 0.5
ENT_FLOOR      = 0.15
ENT_FLOOR_COEF = 0.05
ENT_CEIL       = 1.2
ENT_CEIL_COEF  = 0.05


# ---------------------------------------------------------------------------
# GAE over ONE ship's (or the spawn stream's) ordered step list
# ---------------------------------------------------------------------------

def gae_sequence(steps: List[dict]):
    n = len(steps)
    adv = np.zeros(n, dtype=np.float32)
    last = 0.0
    for t in reversed(range(n)):
        next_val = steps[t + 1]['value'] if t + 1 < n else 0.0
        nonterm = 0.0 if steps[t]['done'] else 1.0
        delta = steps[t]['reward'] + GAMMA * next_val * nonterm - steps[t]['value']
        last = delta + GAMMA * LAM * nonterm * last
        adv[t] = last
    ret = adv + np.array([s['value'] for s in steps], dtype=np.float32)
    return adv, ret


# ---------------------------------------------------------------------------
# policy wrappers used by the env
# ---------------------------------------------------------------------------

def make_ship_policy_fn(policy, value, device, deterministic=False):
    @torch.no_grad()
    def fn(scals, patches, gmaps, masks):
        s = torch.as_tensor(scals, dtype=torch.float32, device=device)
        p = torch.as_tensor(np.transpose(patches, (0, 3, 1, 2)),
                            dtype=torch.float32, device=device)
        g = torch.as_tensor(gmaps, dtype=torch.float32, device=device)
        m = torch.as_tensor(masks, dtype=torch.bool, device=device)
        logits = policy(s, p, g)
        logits = torch.where(m, logits, torch.full_like(logits, NEG_INF))
        vals = value(s, p, g) if value is not None else torch.zeros(len(scals), device=device)
        if deterministic:
            acts = torch.argmax(logits, dim=1)
            lps = torch.zeros(len(scals), device=device)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            acts = dist.sample()
            lps = dist.log_prob(acts)
        return (acts.cpu().numpy(), lps.cpu().numpy(), vals.cpu().numpy())
    return fn


def make_spawn_policy_fn(policy, value, device, deterministic=False):
    @torch.no_grad()
    def fn(sscal, sglob, smask):
        s = torch.as_tensor(sscal[None, ...], dtype=torch.float32, device=device)
        g = torch.as_tensor(sglob[None, ...], dtype=torch.float32, device=device)
        m = torch.as_tensor(smask[None, ...], dtype=torch.bool, device=device)
        logits = policy(s, g)
        logits = torch.where(m, logits, torch.full_like(logits, NEG_INF))
        v = value(s, g)[0] if value is not None else torch.tensor(0.0)
        if deterministic:
            a = torch.argmax(logits[0])
            return int(a.item()), 0.0, float(v.item())
        dist = torch.distributions.Categorical(logits=logits[0])
        a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(v.item())
    return fn


# ---------------------------------------------------------------------------
# batch assembly
# ---------------------------------------------------------------------------

def flatten_ship_batch(all_ship_trajs, device):
    """all_ship_trajs: list over games of dict sid->steps.  GAE per ship."""
    steps, advs, rets = [], [], []
    for trajs in all_ship_trajs:
        for sid, seq in trajs.items():
            a, r = gae_sequence(seq)
            steps.extend(seq)
            advs.append(a)
            rets.append(r)
    if not steps:
        return None
    advs = np.concatenate(advs)
    rets = np.concatenate(rets)
    batch = {
        'scalars': torch.tensor(np.stack([s['scalars'] for s in steps]),
                                dtype=torch.float32, device=device),
        'patch': torch.tensor(np.transpose(np.stack([s['patch'] for s in steps]),
                                           (0, 3, 1, 2)),
                              dtype=torch.float32, device=device),
        'gmap': torch.tensor(np.stack([s['gmap'] for s in steps]),
                             dtype=torch.float32, device=device),
        'mask': torch.tensor(np.stack([s['mask'] for s in steps]),
                             dtype=torch.bool, device=device),
        'action': torch.tensor([s['action'] for s in steps],
                               dtype=torch.long, device=device),
        'old_lp': torch.tensor([s['log_prob'] for s in steps],
                               dtype=torch.float32, device=device),
        'adv': torch.tensor(advs, dtype=torch.float32, device=device),
        'ret': torch.tensor(rets, dtype=torch.float32, device=device),
    }
    batch['adv'] = (batch['adv'] - batch['adv'].mean()) / (batch['adv'].std() + 1e-8)
    return batch


def flatten_spawn_batch(all_spawn_trajs, device):
    steps, advs, rets = [], [], []
    for seq in all_spawn_trajs:
        if not seq:
            continue
        a, r = gae_sequence(seq)
        steps.extend(seq)
        advs.append(a)
        rets.append(r)
    if not steps:
        return None
    advs = np.concatenate(advs)
    rets = np.concatenate(rets)
    batch = {
        'scalars': torch.tensor(np.stack([s['scalars'] for s in steps]),
                                dtype=torch.float32, device=device),
        'gmap': torch.tensor(np.stack([s['gmap'] for s in steps]),
                             dtype=torch.float32, device=device),
        'mask': torch.tensor(np.stack([s['mask'] for s in steps]),
                             dtype=torch.bool, device=device),
        'action': torch.tensor([s['action'] for s in steps],
                               dtype=torch.long, device=device),
        'old_lp': torch.tensor([s['log_prob'] for s in steps],
                               dtype=torch.float32, device=device),
        'adv': torch.tensor(advs, dtype=torch.float32, device=device),
        'ret': torch.tensor(rets, dtype=torch.float32, device=device),
    }
    batch['adv'] = (batch['adv'] - batch['adv'].mean()) / (batch['adv'].std() + 1e-8)
    return batch


# ---------------------------------------------------------------------------
# PPO updates
# ---------------------------------------------------------------------------

def ppo_update_ship(policy, value, opt_pol, opt_val, batch,
                    bc_policy=None, lambda_bc=0.0, policy_updates=True):
    n = batch['action'].shape[0]
    device = batch['action'].device
    stats = {'kl': 0.0, 'entropy': 0.0, 'v_loss': 0.0, 'clipfrac': 0.0,
             'early_stop': 0}
    n_pol_batches = 0

    for epoch in range(N_EPOCHS):
        perm = torch.randperm(n, device=device)
        epoch_kl = []
        for i in range(0, n, MINIBATCH):
            b = perm[i:i + MINIBATCH]
            s, p, g = batch['scalars'][b], batch['patch'][b], batch['gmap'][b]
            m = batch['mask'][b]

            # ---- value update (always) ----
            v = value(s, p, g)
            v_loss = F.mse_loss(v, batch['ret'][b])
            opt_val.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(value.parameters(), MAX_GRAD_NORM)
            opt_val.step()
            stats['v_loss'] += float(v_loss.item())

            if not policy_updates:
                continue

            # ---- policy update ----
            raw_logits = policy(s, p, g)
            logits = torch.where(m, raw_logits, torch.full_like(raw_logits, NEG_INF))
            dist = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(batch['action'][b])
            ent = dist.entropy().mean()

            ratio = (new_lp - batch['old_lp'][b]).exp()
            adv = batch['adv'][b]
            pg1 = ratio * adv
            pg2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
            actor_loss = -torch.min(pg1, pg2).mean()

            ent_loss = (ENT_FLOOR_COEF * torch.clamp(ENT_FLOOR - ent, min=0)
                        + ENT_CEIL_COEF * torch.clamp(ent - ENT_CEIL, min=0))

            kl_bc = torch.tensor(0.0, device=device)
            if bc_policy is not None and lambda_bc > 0.0:
                with torch.no_grad():
                    bc_logits = bc_policy(s, p, g)
                    bc_logits = torch.where(m, bc_logits,
                                            torch.full_like(bc_logits, NEG_INF))
                    bc_logp = F.log_softmax(bc_logits, dim=-1)
                cur_logp = F.log_softmax(logits, dim=-1)
                kl_bc = F.kl_div(bc_logp, cur_logp, reduction='batchmean',
                                 log_target=True)

            loss = actor_loss + ent_loss + lambda_bc * kl_bc
            opt_pol.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            opt_pol.step()

            approx_kl = float((batch['old_lp'][b] - new_lp).mean().item())
            epoch_kl.append(approx_kl)
            stats['kl'] += approx_kl
            stats['entropy'] += float(ent.item())
            stats['clipfrac'] += float(((ratio - 1).abs() > CLIP_EPS).float().mean().item())
            n_pol_batches += 1

        if policy_updates and epoch_kl and np.mean(epoch_kl) > TARGET_KL:
            stats['early_stop'] = epoch + 1
            break

    total_batches = max(1, N_EPOCHS * math.ceil(n / MINIBATCH))
    stats['v_loss'] /= total_batches
    if n_pol_batches:
        for k in ('kl', 'entropy', 'clipfrac'):
            stats[k] /= n_pol_batches
    stats['n'] = n
    return stats


def ppo_update_spawn(policy, value, opt_pol, opt_val, batch,
                     bc_policy=None, lambda_bc=0.0, policy_updates=True):
    n = batch['action'].shape[0]
    device = batch['action'].device
    stats = {'kl': 0.0, 'entropy': 0.0, 'v_loss': 0.0}
    n_pol = 0
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(n, device=device)
        epoch_kl = []
        for i in range(0, n, MINIBATCH):
            b = perm[i:i + MINIBATCH]
            s, g, m = batch['scalars'][b], batch['gmap'][b], batch['mask'][b]

            v = value(s, g)
            v_loss = F.mse_loss(v, batch['ret'][b])
            opt_val.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(value.parameters(), MAX_GRAD_NORM)
            opt_val.step()
            stats['v_loss'] += float(v_loss.item())

            if not policy_updates:
                continue

            raw_logits = policy(s, g)
            logits = torch.where(m, raw_logits, torch.full_like(raw_logits, NEG_INF))
            dist = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(batch['action'][b])
            ent = dist.entropy().mean()
            ratio = (new_lp - batch['old_lp'][b]).exp()
            adv = batch['adv'][b]
            pg1 = ratio * adv
            pg2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
            actor_loss = -torch.min(pg1, pg2).mean()

            kl_bc = torch.tensor(0.0, device=device)
            if bc_policy is not None and lambda_bc > 0.0:
                with torch.no_grad():
                    bc_logits = bc_policy(s, g)
                    bc_logits = torch.where(m, bc_logits,
                                            torch.full_like(bc_logits, NEG_INF))
                    bc_logp = F.log_softmax(bc_logits, dim=-1)
                cur_logp = F.log_softmax(logits, dim=-1)
                kl_bc = F.kl_div(bc_logp, cur_logp, reduction='batchmean',
                                 log_target=True)

            loss = actor_loss + lambda_bc * kl_bc
            opt_pol.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            opt_pol.step()

            approx_kl = float((batch['old_lp'][b] - new_lp).mean().item())
            epoch_kl.append(approx_kl)
            stats['kl'] += approx_kl
            stats['entropy'] += float(ent.item())
            n_pol += 1
        if policy_updates and epoch_kl and np.mean(epoch_kl) > TARGET_KL:
            break
    if n_pol:
        stats['kl'] /= n_pol
        stats['entropy'] /= n_pol
    return stats


# ---------------------------------------------------------------------------
# evaluation gating
# ---------------------------------------------------------------------------

def evaluate(env, policy, value, spawn_policy, spawn_value, device,
             n_games, seed_base):
    ship_fn = make_ship_policy_fn(policy, None, device, deterministic=True)
    spawn_fn = make_spawn_policy_fn(spawn_policy, None, device, deterministic=True)
    wins, margins, deps = 0, [], []
    for i in range(n_games):
        env.seed = seed_base + i
        _, _, info = env.run_episode(ship_fn, spawn_fn)
        if info['winner'] == 0:
            wins += 1
        margins.append(info['final_deposited_agent'] - info['final_deposited_opp'])
        deps.append(info['final_deposited_agent'])
    return {
        'win_rate': wins / n_games,
        'mean_margin': float(np.mean(margins)),
        'mean_dep': float(np.mean(deps)),
    }


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='PPO fine-tune rl_v9 vs V71.')
    ap.add_argument('--bc-ckpt', default=os.path.join(config.CHECKPOINT_DIR, 'bc.pt'))
    ap.add_argument('--resume', default=None,
                    help='PPO checkpoint to resume, or "auto" for the latest')
    ap.add_argument('--start-ep', type=int, default=None)
    ap.add_argument('--episodes', type=int, default=300)
    ap.add_argument('--games-per-update', type=int, default=6)
    ap.add_argument('--lr-pol', type=float, default=1e-4)
    ap.add_argument('--lr-val', type=float, default=5e-4)
    ap.add_argument('--lr-spawn', type=float, default=5e-5)
    ap.add_argument('--value-warmup', type=int, default=15,
                    help='episodes of value-only training before any policy update')
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--height', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--save-interval', type=int, default=25)
    ap.add_argument('--eval-interval', type=int, default=20)
    ap.add_argument('--eval-games', type=int, default=9)
    ap.add_argument('--opponent', choices=['v71', 'rl_v5', 'rl_v8'], default='v71')
    ap.add_argument('--lambda-bc', type=float, default=1.0)
    ap.add_argument('--lambda-bc-min', type=float, default=0.1)
    ap.add_argument('--lambda-bc-episodes', type=int, default=150)
    args = ap.parse_args()

    device = args.device
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    if args.resume == 'auto':
        cands = globmod.glob(os.path.join(config.CHECKPOINT_DIR, 'ppo_ep*.pt'))
        def _ep(p):
            try:
                return int(os.path.splitext(os.path.basename(p))[0].replace('ppo_ep', ''))
            except ValueError:
                return -1
        if cands:
            latest = max(cands, key=_ep)
            args.resume = latest
            if args.start_ep is None:
                args.start_ep = _ep(latest) + 1
            print(f"--resume auto: {latest} (episode {_ep(latest)})")
        else:
            args.resume = None
    if args.start_ep is None:
        args.start_ep = 1

    src = args.resume or args.bc_ckpt
    print(f"Loading {'PPO resume' if args.resume else 'BC warm start'}: {src}")
    policy, spawn_policy, value, spawn_value = load_bundle(src, device=device,
                                                           need_values=True)
    for mdl in (policy, spawn_policy, value, spawn_value):
        mdl.train()

    # frozen BC anchors
    bc_policy, bc_spawn, _, _ = load_bundle(args.bc_ckpt, device=device)
    for mdl in (bc_policy, bc_spawn):
        mdl.eval()
        for prm in mdl.parameters():
            prm.requires_grad_(False)

    opt_pol = torch.optim.Adam(policy.parameters(), lr=args.lr_pol)
    opt_val = torch.optim.Adam(value.parameters(), lr=args.lr_val)
    opt_spol = torch.optim.Adam(spawn_policy.parameters(), lr=args.lr_spawn)
    opt_sval = torch.optim.Adam(spawn_value.parameters(), lr=args.lr_val)

    opponent_cmd = opponent_cmd_for(args.opponent)
    env = HaliteEnvV9(width=args.width, height=args.height,
                      opponent_cmd=opponent_cmd)

    log_path = os.path.join(config.CHECKPOINT_DIR, 'ppo_log.csv')
    log_exists = os.path.exists(log_path)
    log_f = open(log_path, 'a', newline='')
    log_w = csv.writer(log_f)
    if not log_exists:
        log_w.writerow(['episode', 'phase', 'dep_agent', 'dep_opp', 'win_rate',
                        'ship_kl', 'ship_entropy', 'ship_vloss', 'clipfrac',
                        'spawn_kl', 'lambda_bc', 'my_wrecks', 'enemy_kills',
                        'dropoffs', 'spawns', 'peak_ships', 'n_steps', 'secs'])

    # best-checkpoint gating state (survives --resume via best.pt's metadata)
    best_path = os.path.join(config.CHECKPOINT_DIR, 'best.pt')
    best_score = (-1.0, -1e18)          # (win_rate, mean_margin)
    if os.path.exists(best_path):
        try:
            meta = torch.load(best_path, map_location='cpu',
                              weights_only=False).get('meta', {})
            if 'win_rate' in meta:
                best_score = (meta['win_rate'], meta.get('mean_margin', -1e18))
                print(f"Existing best.pt score: wr={best_score[0]:.2f} "
                      f"margin={best_score[1]:+.0f}")
        except Exception:
            pass
    eval_log_path = os.path.join(config.CHECKPOINT_DIR, 'eval_log.csv')
    ev_exists = os.path.exists(eval_log_path)
    ev_f = open(eval_log_path, 'a', newline='')
    ev_w = csv.writer(ev_f)
    if not ev_exists:
        ev_w.writerow(['episode', 'win_rate', 'mean_margin', 'mean_dep', 'is_best'])

    seed_counter = args.seed + (args.start_ep - 1) * args.games_per_update
    EVAL_SEED_BASE = 900000

    def run_eval_and_gate(ep):
        nonlocal best_score
        policy.eval(); spawn_policy.eval()
        res = evaluate(env, policy, value, spawn_policy, spawn_value, device,
                       args.eval_games, EVAL_SEED_BASE)
        policy.train(); spawn_policy.train()
        score = (res['win_rate'], res['mean_margin'])
        is_best = score > best_score
        if is_best:
            best_score = score
            save_bundle(best_path, policy, spawn_policy, value, spawn_value,
                        extra={'episode': ep, **res})
        print(f"  [eval ep {ep}] wr={res['win_rate']:.2f} "
              f"margin={res['mean_margin']:+.0f} dep={res['mean_dep']:.0f}"
              f"{'  *** new best ***' if is_best else ''}")
        ev_w.writerow([ep, res['win_rate'], res['mean_margin'],
                       res['mean_dep'], int(is_best)])
        ev_f.flush()
        return res

    # baseline eval of the starting policy (BC or resume point)
    if not args.resume:
        print("Baseline evaluation of the BC policy...")
        run_eval_and_gate(args.start_ep - 1)

    for ep in range(args.start_ep, args.start_ep + args.episodes):
        t0 = time.time()
        warming = (ep - args.start_ep) < args.value_warmup and not args.resume

        ship_fn = make_ship_policy_fn(policy, value, device)
        spawn_fn = make_spawn_policy_fn(spawn_policy, spawn_value, device)

        all_ship_trajs, all_spawn_trajs, infos = [], [], []
        for gi in range(args.games_per_update):
            env.seed = seed_counter
            seed_counter += 1
            st, sp, info = env.run_episode(ship_fn, spawn_fn)
            all_ship_trajs.append(st)
            all_spawn_trajs.append(sp)
            infos.append(info)

        ship_batch = flatten_ship_batch(all_ship_trajs, device)
        spawn_batch = flatten_spawn_batch(all_spawn_trajs, device)

        rel = ep - args.start_ep
        if args.lambda_bc_episodes > 0:
            frac = min(1.0, rel / args.lambda_bc_episodes)
            lambda_bc = args.lambda_bc * (1 - frac) + args.lambda_bc_min * frac
        else:
            lambda_bc = args.lambda_bc

        s_stats = {'kl': 0, 'entropy': 0, 'v_loss': 0, 'clipfrac': 0, 'n': 0}
        if ship_batch is not None:
            s_stats = ppo_update_ship(policy, value, opt_pol, opt_val,
                                      ship_batch, bc_policy, lambda_bc,
                                      policy_updates=not warming)
        sp_stats = {'kl': 0, 'entropy': 0, 'v_loss': 0}
        if spawn_batch is not None:
            sp_stats = ppo_update_spawn(spawn_policy, spawn_value, opt_spol,
                                        opt_sval, spawn_batch, bc_spawn,
                                        lambda_bc, policy_updates=not warming)

        wins = sum(1 for i in infos if i['winner'] == 0)
        mean_dep_a = np.mean([i['final_deposited_agent'] for i in infos])
        mean_dep_o = np.mean([i['final_deposited_opp'] for i in infos])
        wr = wins / len(infos)
        secs = time.time() - t0
        phase = 'warmup' if warming else 'ppo'

        print(f"ep {ep:4d} [{phase}]  dep={mean_dep_a:7.0f}/{mean_dep_o:7.0f}  "
              f"wr={wr:.2f}  kl={s_stats['kl']:.4f}  ent={s_stats['entropy']:.3f}  "
              f"vloss={s_stats['v_loss']:.3f}  lbc={lambda_bc:.2f}  "
              f"wrecks={sum(i['my_wrecks'] for i in infos)}  "
              f"kills={sum(i['enemy_kills'] for i in infos)}  "
              f"drop={sum(i['dropoffs_built'] for i in infos)}  "
              f"spawn={sum(i['spawns'] for i in infos)}  t={secs:.0f}s")
        log_w.writerow([ep, phase, mean_dep_a, mean_dep_o, wr,
                        s_stats['kl'], s_stats['entropy'], s_stats['v_loss'],
                        s_stats.get('clipfrac', 0), sp_stats['kl'], lambda_bc,
                        sum(i['my_wrecks'] for i in infos),
                        sum(i['enemy_kills'] for i in infos),
                        sum(i['dropoffs_built'] for i in infos),
                        sum(i['spawns'] for i in infos),
                        max(i.get('peak_ships', 0) for i in infos),
                        s_stats.get('n', 0), round(secs, 1)])
        log_f.flush()

        if ep % args.save_interval == 0:
            out = os.path.join(config.CHECKPOINT_DIR, f'ppo_ep{ep:04d}.pt')
            save_bundle(out, policy, spawn_policy, value, spawn_value)
            print(f"  saved {out}")
        if (not warming) and ep % args.eval_interval == 0:
            run_eval_and_gate(ep)

    save_bundle(os.path.join(config.CHECKPOINT_DIR, 'ppo_final.pt'),
                policy, spawn_policy, value, spawn_value)
    run_eval_and_gate(args.start_ep + args.episodes - 1)
    print(f"\nDone.  best.pt score: win_rate={best_score[0]:.2f} "
          f"margin={best_score[1]:+.0f}")
    log_f.close()
    ev_f.close()


if __name__ == '__main__':
    main()
