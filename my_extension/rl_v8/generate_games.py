#!/usr/bin/env python3
"""
rl_v8 / generate_games.py  --  have the 2019 bot (MyBot V71) play many games
and save the replays.

These replays are the raw material for the behavioral-cloning dataset: we will
parse them (collect_dataset.py) to recover, for every V71 ship-turn, the
(features, action) pair we want rl_v8 to imitate.

Default mode: V71 vs rl_v5 (not V71 self-play). Self-play only visits states
V71 itself creates; playing against rl_v5 pushes V71 into a wider variety of
board states (different fleet sizes/positions/pressure) closer to what rl_v8
will actually face at evaluation time, which should make for a more useful
imitation dataset. rl_v5 runs STOCHASTICALLY by default (like rl_v7's original
rl_v5 self-play) for extra state diversity; pass --v5-deterministic to disable
that. collect_dataset.py already filters to the seat named "RevolzBot", so
only V71's ship-turns are ever collected -- rl_v5's own moves are never
imitated.

Optional extra sources: --selfplay N (V71 vs itself) and --vs-greedy N
(V71 vs the simple starter-kit bot), both off by default.

Usage:
    python rl_v8/generate_games.py --games 60
    python rl_v8/generate_games.py --games 100 --selfplay 20 --vs-greedy 20
"""

import argparse
import datetime
import os
import random
import sys

HERE      = os.path.dirname(os.path.abspath(__file__))
MY_EXT    = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(MY_EXT)
sys.path.insert(0, MY_EXT)

from halite_engine import HaliteEngine          # noqa: E402
import config                                    # noqa: E402

V71_BOT = 'python -u "' + os.path.join(MY_EXT, 'Year 2019', 'MyBot - V71', 'MyBot.py') + '"'
GREEDY_BOT = f'python -u "{os.path.join(REPO_ROOT, "starter_kits", "Python3", "MyBot.py")}"'
RL_V5_MODEL_DEFAULT = os.path.join(MY_EXT, 'rl_v5', 'checkpoints', 'best.pt')


def run_batch(n_games, width, height, bots, tag, seed0):
    os.makedirs(config.REPLAYS_DIR, exist_ok=True)
    for i in range(n_games):
        seed = seed0 + i
        engine = HaliteEngine(width=width, height=height, num_players=2,
                              seed=seed, verbose=False)
        ts = datetime.datetime.now().strftime('%H%M%S')
        out = os.path.join(config.REPLAYS_DIR,
                           f'{tag}-{width}x{height}-{seed}-{ts}.hlt')
        try:
            results = engine.run(bots, replay_file=out)
        except Exception as e:               # a crashed bot shouldn't kill the batch
            print(f"  [{tag} {i+1}/{n_games}] seed {seed} FAILED: {e}")
            continue
        (w_pid, w_hal) = results[0]
        print(f"  [{tag} {i+1}/{n_games}] seed {seed}  ->  "
              f"P{results[0][0]}={results[0][1]:,}  P{results[1][0]}={results[1][1]:,}  "
              f"saved {os.path.basename(out)}")


def main():
    ap = argparse.ArgumentParser(description='Generate V71 replays for rl_v8 BC.')
    ap.add_argument('--games', type=int, default=60,
                    help='number of V71-vs-rl_v5 games (default 60)')
    ap.add_argument('--selfplay', type=int, default=0,
                    help='additional V71-vs-V71 self-play games (default 0)')
    ap.add_argument('--vs-greedy', type=int, default=0,
                    help='additional games vs the starter-kit bot (default 0)')
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--height', type=int, default=32)
    ap.add_argument('--seed', type=int, default=None,
                    help='base seed (default: random)')
    ap.add_argument('--v5-model', default=RL_V5_MODEL_DEFAULT,
                    help='rl_v5 checkpoint to play against')
    ap.add_argument('--v5-deterministic', action='store_true',
                    help='run rl_v5 deterministically (less state diversity)')
    args = ap.parse_args()

    seed0 = args.seed if args.seed is not None else random.randint(0, 2**30)
    v71 = V71_BOT
    v5_det = ' --deterministic' if args.v5_deterministic else ''
    v5 = f'python -u "{os.path.join(MY_EXT, "rl_v5", "rl_bot.py")}" --model "{args.v5_model}"{v5_det}'

    print(f"=== rl_v8 data generation ===")
    print(f"V71-vs-rl_v5 games : {args.games}")
    print(f"V71 self-play games: {args.selfplay}")
    print(f"vs-greedy games    : {args.vs_greedy}")
    print(f"map                : {args.width}x{args.height}")
    print(f"base seed          : {seed0}")
    print(f"replays -> {config.REPLAYS_DIR}\n")

    if args.games > 0:
        print("V71 vs rl_v5:")
        run_batch(args.games, args.width, args.height, [v71, v5],
                  tag='v71vsv5', seed0=seed0)
    if args.selfplay > 0:
        print("\nV71 self-play:")
        run_batch(args.selfplay, args.width, args.height, [v71, v71],
                  tag='v71self', seed0=seed0 + 100000)
    if args.vs_greedy > 0:
        print("\nV71 vs greedy:")
        run_batch(args.vs_greedy, args.width, args.height, [v71, GREEDY_BOT],
                  tag='v71greedy', seed0=seed0 + 200000)

    print("\nDone.")


if __name__ == '__main__':
    main()
