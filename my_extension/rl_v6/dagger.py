#!/usr/bin/env python3
"""
dagger.py — DAgger data collection for rl_v6 (fixes BC distribution shift).

Plain behavioral cloning drifts: rl_v6 visits states rl_v5 never did (collisions
it fails to avoid, ships it stranded), where it has no expert guidance.  DAgger
fixes this by collecting ON-POLICY states (rl_v6 drives player 0) but labelling
them with the EXPERT's action (what rl_v5 would do there).

Each turn of each game:
  1. rl_v6's CURRENT policy + spawn head choose player-0 actions  (on-policy state)
  2. the rl_v5 expert is queried for the SAME state -> per-ship labels + spawn label
  3. (rl_v6 pure features, rl_v5 label) is recorded
  4. the engine is stepped with rl_v6's actions (so the next state is on rl_v6's
     own trajectory)
Opponent (player 1) is the rl_v5 bot by default, so rl_v6 learns the matchup it
will actually face.

Shards are written in the same .npz format as rl_collect.py, so they drop into
the dataset dir and `bc_train.py` retrains on BC + DAgger data together.  The
DAgger outer loop is: dagger.py (collect) -> bc_train.py (retrain) -> repeat,
which `--retrain` automates.

Usage
-----
    python dagger.py --policy checkpoints/model_weights.pt \
        --spawn checkpoints/spawn_weights.pt --n-games 20 \
        --out dataset/ --opponent rl_v5 [--retrain --epochs 10]
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from rl_env      import HaliteEnvV6
from rl_model    import ActorCritic
from spawn_model import SpawnHead
from experts     import FrozenBotDriver
from rl_config   import N_SCALARS_V6, N_SHIP_ACTIONS_V6


def collect_game(env, policy, spawn_head, expert, device, deterministic=False):
    """Play one game; return per-ship (obs,label) arrays + per-turn spawn data."""
    obs, _ = env.reset()
    sp, sc, ac, tn, sids, sf, sl = [], [], [], [], [], [], []
    done = False
    while not done:
        engine = env.engine
        # 1+2. Expert labels for the CURRENT (rl_v6-visited) state.
        exp_acts, exp_spawn = expert.expert_actions(engine, 0)
        # record (rl_v6 pure features, expert label) per p0 ship
        for sid, (spatial, scalars, mask) in obs.items():
            sp.append(spatial); sc.append(scalars)
            ac.append(exp_acts.get(sid, 0))
            tn.append(engine.turn); sids.append(sid)
        sf.append(env.spawn_features(0)); sl.append(1 if exp_spawn else 0)

        # 1. rl_v6's OWN actions drive the step (on-policy state distribution).
        v6_acts = {}
        for sid, (spatial, scalars, mask) in obs.items():
            spt = torch.from_numpy(spatial).to(device)
            sct = torch.from_numpy(scalars).to(device)
            if deterministic:
                v6_acts[sid] = policy.greedy_action(spt, sct, mask=mask)
            else:
                a, _, _ = policy.select_action(spt, sct, mask=mask)
                v6_acts[sid] = a
        do_spawn = False
        if spawn_head is not None and engine.players[0]['energy'] >= 1000:
            p = spawn_head.spawn_prob(torch.from_numpy(env.spawn_features(0)).to(device))
            do_spawn = (p >= 0.5) if deterministic else (np.random.random() < p)

        obs, _, done, info = env.step(v6_acts, spawn=do_spawn)
    return sp, sc, ac, tn, sids, sf, sl, info


def main():
    ap = argparse.ArgumentParser(description='DAgger collection for rl_v6')
    ap.add_argument('--policy', default=os.path.join(_HERE, 'checkpoints', 'model_weights.pt'))
    ap.add_argument('--spawn',  default=os.path.join(_HERE, 'checkpoints', 'spawn_weights.pt'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'dataset'))
    ap.add_argument('--n-games', type=int, default=20)
    ap.add_argument('--opponent', choices=['rl_v5', 'rl_v4', 'greedy'], default='rl_v5')
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--height', type=int, default=32)
    ap.add_argument('--seed-start', type=int, default=50000)
    ap.add_argument('--tag', default='dagger')
    ap.add_argument('--deterministic', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    # convenience: retrain BC after collecting
    ap.add_argument('--retrain', action='store_true')
    ap.add_argument('--epochs', type=int, default=10)
    args = ap.parse_args()

    device = torch.device(args.device)
    policy = ActorCritic(n_scalars=N_SCALARS_V6, n_actions=N_SHIP_ACTIONS_V6).to(device)
    policy.load_state_dict(torch.load(args.policy, map_location=device))
    policy.eval()
    spawn_head = (SpawnHead.load(args.spawn, device=args.device).to(device)
                  if os.path.exists(args.spawn) else None)

    expert = FrozenBotDriver('rl_v5')                       # always the label source
    opp = None if args.opponent == 'greedy' else FrozenBotDriver(args.opponent)

    os.makedirs(args.out, exist_ok=True)
    for i in range(args.n_games):
        seed = args.seed_start + i
        env = HaliteEnvV6(args.width, args.height, seed=seed,
                          opponent_policy='greedy', allow_dropoff=True)
        if opp is not None:
            env.opponent_command_fn = lambda e, pid, _o=opp: _o.command(e.engine, pid)
        sp, sc, ac, tn, sids, sf, sl, info = collect_game(
            env, policy, spawn_head, expert, device, args.deterministic)
        if not sp:
            print(f"[{i+1}/{args.n_games}] seed={seed} no steps, skip"); continue
        path = os.path.join(args.out, f'{args.tag}_{seed}.npz')
        np.savez_compressed(
            path,
            obs_spatial=np.array(sp, np.float32), obs_scalars=np.array(sc, np.float32),
            actions=np.array(ac, np.int8), turns=np.array(tn, np.int16),
            ship_ids=np.array(sids, np.int32),
            spawn_feats=np.array(sf, np.float32), spawn_label=np.array(sl, np.int8),
            player_id=np.int8(0), map_width=np.int16(args.width),
            map_height=np.int16(args.height))
        print(f"[{i+1}/{args.n_games}] seed={seed} {len(ac)} labelled steps, "
              f"rl_v6 ships_end={info['ships_p0']} deposited={info['deposited_p0']} "
              f"-> {os.path.basename(path)}", flush=True)

    if args.retrain:
        print("\nRetraining BC on BC+DAgger data…")
        subprocess.run([sys.executable, os.path.join(_HERE, 'bc_train.py'),
                        '--data', args.out, '--epochs', str(args.epochs),
                        '--out', args.policy, '--spawn-out', args.spawn,
                        '--resume', args.policy], check=True)


if __name__ == '__main__':
    main()
