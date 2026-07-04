#!/usr/bin/env python3
"""
gen_data.py — generate rl_v5 gameplay replays for rl_v6 imitation learning.

Runs many games on the Python engine and saves .hlt replays, which
`rl_collect.py` then turns into (obs, action) + spawn datasets.

Modes:
  vs_v4     : rl_v5 (player 0) vs rl_v4 (player 1)   — extract player 0
  selfplay  : rl_v5 vs rl_v5                          — extract --both
  both      : alternate the two each game

rl_v5 plays STOCHASTICALLY by default (its FSM logit prior still dominates, so
moves are mostly canonical, but sampling diversifies the visited states — good
coverage for behavioral cloning).  Pass --deterministic for the canonical policy.

Usage
-----
    python gen_data.py --n-games 50 --mode vs_v4 --out-dir replays_v6/
"""

import argparse
import datetime
import os
import sys

_HERE      = os.path.dirname(os.path.abspath(__file__))
_MY_EXT    = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_MY_EXT)
STARTER    = os.path.join(_REPO_ROOT, 'starter_kits', 'Python3')
sys.path.insert(0, _MY_EXT)
from halite_engine import HaliteEngine

RL_V5 = os.path.join(_MY_EXT, 'rl_v5')
RL_V4 = os.path.join(_MY_EXT, 'rl_v4')


def _bot_cmd(folder: str, deterministic: bool) -> str:
    model = os.path.join(folder, 'checkpoints', 'best.pt')
    det   = ' --deterministic' if deterministic else ''
    return f'python -u "{os.path.join(folder, "rl_bot.py")}" --model "{model}"{det}'


def main():
    ap = argparse.ArgumentParser(description='Generate rl_v5 replays for rl_v6')
    ap.add_argument('--n-games', type=int, default=50)
    ap.add_argument('--mode', choices=['vs_v4', 'selfplay', 'both'], default='vs_v4')
    ap.add_argument('--out-dir', default=os.path.join(_HERE, 'replays_v6'))
    ap.add_argument('--width',  type=int, default=32)
    ap.add_argument('--height', type=int, default=32)
    ap.add_argument('--seed-start', type=int, default=10000)
    ap.add_argument('--deterministic', action='store_true',
                    help='rl_v5 plays its canonical (greedy) policy')
    args = ap.parse_args()

    os.environ['PYTHONPATH'] = STARTER + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    os.makedirs(args.out_dir, exist_ok=True)

    v5 = _bot_cmd(RL_V5, args.deterministic)
    v4 = _bot_cmd(RL_V4, False)

    for i in range(args.n_games):
        seed = args.seed_start + i
        mode = args.mode
        if mode == 'both':
            mode = 'vs_v4' if i % 2 == 0 else 'selfplay'
        bots = [v5, v5] if mode == 'selfplay' else [v5, v4]

        ts = datetime.datetime.now().strftime('%H%M%S')
        path = os.path.join(args.out_dir,
                            f'g{i:05d}-{mode}-{seed}-{args.width}x{args.height}-{ts}.hlt')
        eng = HaliteEngine(width=args.width, height=args.height,
                           num_players=2, seed=seed, verbose=False)
        try:
            results = eng.run(bots, replay_file=path)
            win_pid, win_h = results[0]
            print(f"[{i+1}/{args.n_games}] {mode} seed={seed} "
                  f"winner=p{win_pid} ({win_h:,}) -> {os.path.basename(path)}",
                  flush=True)
        except Exception as e:
            print(f"[{i+1}/{args.n_games}] {mode} seed={seed} FAILED: {e}", flush=True)

    print(f"\nDone. Replays in {args.out_dir}")


if __name__ == '__main__':
    main()
