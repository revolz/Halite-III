#!/usr/bin/env python3
"""
rl_v7 / rl_train.py  --  PPO fine-tuning from a BC warm-start vs frozen rl_v5.

Loads a BC checkpoint, then runs PPO against the frozen rl_v5 opponent.
The deterministic collision resolver stays active so exploration never blows
up the fleet.  The agent only searches over strategy.

Key stability choices from rl_v5's training (adapted):
  - Multi-game batches (games_per_update) reduce gradient variance.
  - Soft entropy floor guards against collapse.
  - LR decay after a burn-in period.
  - Reward is halite-deposited-based (primary) + terminal win/margin.

Usage:
    python rl_v7/rl_train.py --bc-ckpt checkpoints/bc.pt --episodes 500
    python rl_v7/rl_train.py --resume checkpoints/ppo_ep100.pt --start-ep 101
"""

import argparse
import csv
import math
import os
import random
import sys
import time
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import config                                       # noqa: E402
from config import N_ACTIONS, PATCH_SIZE, PATCH_CHANNELS  # noqa: E402
from net import ActorCritic                         # noqa: E402
import features as featmod                          # noqa: E402
from rl_env import HaliteEnvV7                      # noqa: E402

NEG_INF = -1e9

# --- PPO hyperparameters ---
GAMMA          = 0.999
LAM            = 0.95
CLIP_EPS       = 0.2
VF_COEF        = 0.5
ENT_COEF       = 0.0    # no upward entropy push (was 0.01, caused explosion)
ENT_FLOOR      = 0.3    # nats; penalise if entropy drops below this
ENT_FLOOR_COEF = 0.05
ENT_CEIL       = 1.0    # nats; penalise if entropy rises above this
ENT_CEIL_COEF  = 0.05
MAX_GRAD_NORM  = 0.5
N_EPOCHS       = 2
MINIBATCH      = 256


def gae(rewards, values, dones, gamma, lam):
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae_val = 0.0
    for t in reversed(range(n)):
        next_val = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae_val = delta + gamma * lam * (1 - dones[t]) * gae_val
        advantages[t] = gae_val
    returns = advantages + np.array(values[:n], dtype=np.float32)
    return advantages, returns


def make_policy_fn(model, device, deterministic=False):
    def policy_fn(wv, sid, scal, patch, mask):
        if deterministic:
            act = model.greedy_action(scal, patch, mask=mask, device=device)
            return act, 0.0, 0.0
        else:
            act, lp, val = model.select_action(scal, patch, mask=mask, device=device)
            return act, lp, val
    return policy_fn


def collect_trajectories(env, policy_fn, n_games, seed_base):
    trajs, infos = [], []
    for i in range(n_games):
        env.seed = seed_base + i
        traj, info = env.run_episode(policy_fn)
        trajs.extend(traj)
        infos.append(info)
    return trajs, infos


def ppo_update(model, opt, trajs, device, n_epochs, minibatch,
               ent_coef=ENT_COEF, ent_floor_coef=ENT_FLOOR_COEF,
               ent_ceil=ENT_CEIL, ent_ceil_coef=ENT_CEIL_COEF,
               bc_model=None, lambda_bc=0.0):
    scalars = torch.tensor(np.stack([t['scalars'] for t in trajs]),
                           dtype=torch.float32, device=device)
    patches = torch.tensor(
        np.transpose(np.stack([t['patch'] for t in trajs]), (0, 3, 1, 2)),
        dtype=torch.float32, device=device)
    actions = torch.tensor([t['action'] for t in trajs], dtype=torch.long, device=device)
    masks   = torch.tensor(np.stack([t['mask'] for t in trajs]), dtype=torch.bool, device=device)
    old_lp  = torch.tensor([t['log_prob'] for t in trajs], dtype=torch.float32, device=device)
    rewards = np.array([t['reward'] for t in trajs], dtype=np.float32)
    dones   = np.array([t['done'] for t in trajs], dtype=np.float32)
    values  = np.array([t['value'] for t in trajs], dtype=np.float32)

    advs, rets = gae(rewards.tolist(), values.tolist(), dones.tolist(), GAMMA, LAM)
    advs_t = torch.tensor(advs, dtype=torch.float32, device=device)
    advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)
    rets_t = torch.tensor(rets, dtype=torch.float32, device=device)

    n = len(trajs)
    total_kl = 0.0
    use_bc = (bc_model is not None) and (lambda_bc > 0.0)
    for _ in range(n_epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, minibatch):
            b = perm[i:i + minibatch]
            # forward pass to get raw logits (needed for KL)
            model_logits, new_val = model(scalars[b], patches[b])
            masked_logits = torch.where(masks[b], model_logits,
                                        torch.full_like(model_logits, NEG_INF))
            dist = torch.distributions.Categorical(logits=masked_logits)
            new_lp  = dist.log_prob(actions[b])
            entropy = dist.entropy()

            ratio = (new_lp - old_lp[b]).exp()
            pg1 = ratio * advs_t[b]
            pg2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advs_t[b]
            actor_loss = -torch.min(pg1, pg2).mean()
            value_loss = F.mse_loss(new_val, rets_t[b])
            ent = entropy.mean()
            ent_bonus      = -ent_coef      * ent
            ent_floor_loss =  ent_floor_coef * torch.clamp(ENT_FLOOR - ent, min=0)
            ent_ceil_loss  =  ent_ceil_coef  * torch.clamp(ent - ent_ceil,  min=0)

            # BC regularization: KL(current || BC_frozen), computed per minibatch
            kl_loss = torch.tensor(0.0, device=device)
            if use_bc:
                with torch.no_grad():
                    bc_logits_b, _ = bc_model(scalars[b], patches[b])
                    bc_log_probs_b = F.log_softmax(bc_logits_b, dim=-1)
                curr_log_probs = F.log_softmax(model_logits, dim=-1)
                # KL(current||BC) in log space: F.kl_div(input=BC_log, target=curr_log, log_target=True)
                kl_loss = F.kl_div(bc_log_probs_b, curr_log_probs,
                                   reduction='batchmean', log_target=True)
                total_kl += float(kl_loss.item())

            loss = (actor_loss + VF_COEF * value_loss
                    + ent_bonus + ent_floor_loss + ent_ceil_loss
                    + lambda_bc * kl_loss)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step()

    n_batches = n_epochs * math.ceil(n / minibatch)
    # return summary stats
    with torch.no_grad():
        new_lp_all, ent_all, new_val_all = model.evaluate_batch(
            scalars, patches, actions, masks=masks)
    return {
        'mean_entropy': float(ent_all.mean()),
        'mean_value': float(new_val_all.mean()),
        'mean_kl': total_kl / max(1, n_batches),
        'n_steps': n,
    }


def main():
    ap = argparse.ArgumentParser(description='PPO fine-tune rl_v7 vs rl_v5.')
    ap.add_argument('--bc-ckpt',  default=os.path.join(config.CHECKPOINT_DIR, 'bc.pt'))
    ap.add_argument('--resume',   default=None,
                    help='path to PPO checkpoint to resume from, or "auto" to find the latest')
    ap.add_argument('--start-ep', type=int, default=None,
                    help='episode number to start from (inferred automatically when --resume auto)')
    ap.add_argument('--episodes', type=int, default=500)
    ap.add_argument('--games-per-update', type=int, default=6)
    ap.add_argument('--lr',       type=float, default=2e-4)
    ap.add_argument('--width',    type=int, default=32)
    ap.add_argument('--height',   type=int, default=32)
    ap.add_argument('--seed',     type=int, default=0)
    ap.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--save-interval', type=int, default=25)
    ap.add_argument('--v5-model', default=None)
    ap.add_argument('--ent-coef',           type=float, default=ENT_COEF)
    ap.add_argument('--ent-ceil',           type=float, default=ENT_CEIL)
    ap.add_argument('--ent-ceil-coef',      type=float, default=ENT_CEIL_COEF)
    ap.add_argument('--lambda-bc',          type=float, default=0.0,
                    help='initial BC-regularization KL weight (0 = disabled)')
    ap.add_argument('--lambda-bc-min',      type=float, default=0.05,
                    help='minimum BC-reg weight after annealing')
    ap.add_argument('--lambda-bc-episodes', type=int,   default=200,
                    help='episodes over which lambda_bc anneals from start to min')
    args = ap.parse_args()

    device = args.device
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # resolve --resume auto: find the highest-numbered ppo_ep*.pt in checkpoints/
    if args.resume == 'auto':
        import glob as _glob
        pattern = os.path.join(config.CHECKPOINT_DIR, 'ppo_ep*.pt')
        candidates = _glob.glob(pattern)
        if not candidates:
            print("--resume auto: no PPO checkpoints found, warm-starting from BC checkpoint")
            args.resume = None
        else:
            def _ep_num(p):
                base = os.path.splitext(os.path.basename(p))[0]  # e.g. ppo_ep0025
                try:
                    return int(base.replace('ppo_ep', ''))
                except ValueError:
                    return -1
            latest = max(candidates, key=_ep_num)
            inferred_ep = _ep_num(latest)
            args.resume = latest
            if args.start_ep is None:
                args.start_ep = inferred_ep + 1
            print(f"--resume auto: found {latest}  (episode {inferred_ep})")

    if args.start_ep is None:
        args.start_ep = 1

    # load model
    if args.resume:
        print(f"Resuming from {args.resume}  (start_ep={args.start_ep})")
        model = ActorCritic.load(args.resume, device=device)
    else:
        print(f"Warm-starting from BC checkpoint {args.bc_ckpt}")
        model = ActorCritic.load(args.bc_ckpt, device=device)
    model.train()

    # frozen BC reference model for KL regularization
    bc_model = None
    if args.lambda_bc > 0.0:
        print(f"Loading frozen BC model from {args.bc_ckpt} for KL regularization")
        bc_model = ActorCritic.load(args.bc_ckpt, device=device)
        for p in bc_model.parameters():
            p.requires_grad_(False)
        bc_model.eval()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # LR schedule: decay 3% every 50 episodes after episode 100
    def lr_lambda(ep):
        if ep < 100:
            return 1.0
        return 0.97 ** ((ep - 100) // 50)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    env = HaliteEnvV7(width=args.width, height=args.height,
                      v5_model_path=args.v5_model)

    log_path = os.path.join(config.CHECKPOINT_DIR, 'ppo_log.csv')
    log_exists = os.path.exists(log_path)
    log_f = open(log_path, 'a', newline='')
    log_w = csv.writer(log_f)
    if not log_exists:
        log_w.writerow(['episode', 'deposited_agent', 'deposited_opp',
                        'win_rate', 'mean_entropy', 'mean_value', 'mean_kl',
                        'lambda_bc', 'lr', 'n_steps'])

    seed_counter = args.seed + (args.start_ep - 1) * args.games_per_update

    for ep in range(args.start_ep, args.start_ep + args.episodes):
        t0 = time.time()
        policy_fn = make_policy_fn(model, device, deterministic=False)
        trajs, infos = collect_trajectories(env, policy_fn, args.games_per_update, seed_counter)
        seed_counter += args.games_per_update

        # lambda_bc schedule: linear anneal from start to min over lambda_bc_episodes
        rel_ep = ep - args.start_ep
        if args.lambda_bc > 0.0 and args.lambda_bc_episodes > 0:
            frac = min(1.0, rel_ep / args.lambda_bc_episodes)
            lambda_bc = max(args.lambda_bc_min,
                            args.lambda_bc * (1.0 - frac) + args.lambda_bc_min * frac)
        else:
            lambda_bc = args.lambda_bc

        stats = ppo_update(model, opt, trajs, device, N_EPOCHS, MINIBATCH,
                           ent_coef=args.ent_coef,
                           ent_ceil=args.ent_ceil,
                           ent_ceil_coef=args.ent_ceil_coef,
                           bc_model=bc_model, lambda_bc=lambda_bc)
        scheduler.step()

        wins = sum(1 for info in infos if info['winner'] == env.agent_pid)
        mean_dep_a = sum(i['final_deposited_agent'] for i in infos) / len(infos)
        mean_dep_o = sum(i['final_deposited_opp']   for i in infos) / len(infos)
        win_rate = wins / len(infos)
        lr_now = opt.param_groups[0]['lr']
        elapsed = time.time() - t0

        print(f"ep {ep:4d}  dep={mean_dep_a:7.0f}/{mean_dep_o:7.0f}  "
              f"wr={win_rate:.2f}  ent={stats['mean_entropy']:.3f}  "
              f"kl={stats['mean_kl']:.4f}  lbc={lambda_bc:.3f}  "
              f"lr={lr_now:.2e}  t={elapsed:.1f}s")
        log_w.writerow([ep, mean_dep_a, mean_dep_o, win_rate,
                        stats['mean_entropy'], stats['mean_value'],
                        stats['mean_kl'], lambda_bc,
                        lr_now, stats['n_steps']])
        log_f.flush()

        if ep % args.save_interval == 0:
            out = os.path.join(config.CHECKPOINT_DIR, f'ppo_ep{ep:04d}.pt')
            model.save(out)
            print(f"  saved {out}")

    best_out = os.path.join(config.CHECKPOINT_DIR, 'ppo_final.pt')
    model.save(best_out)
    print(f"\nTraining done.  Final checkpoint: {best_out}")
    log_f.close()


if __name__ == '__main__':
    main()
