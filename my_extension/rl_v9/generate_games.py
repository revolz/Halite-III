#!/usr/bin/env python3
"""
rl_v9 / generate_games.py  --  have the 2019 bot (MyBot V71) play many games
and save the replays (raw material for behavioral cloning).

Default: V71 vs rl_v5 (stochastic) for state diversity, same rationale as
rl_v8.  --selfplay N adds V71-vs-V71 games (also the source of *both-seat*
spawn/dropoff labels), --vs-greedy N adds games vs the starter-kit bot.

Usage:
    python rl_v9/generate_games.py --games 150 --selfplay 50
"""

import argparse
import datetime
import os
import random
import sys

HERE      = os.path.dirname(os.path.abspath(__file__))
MY_EXT    = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(MY_EXT)
sys.path.insert(0, HERE)
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
        print(f"  [{tag} {i+1}/{n_games}] seed {seed}  ->  "
              f"P{results[0][0]}={results[0][1]:,}  P{results[1][0]}={results[1][1]:,}  "
              f"saved {os.path.basename(out)}")


def main():
    ap = argparse.ArgumentParser(description='Generate V71 replays for rl_v9 BC.')
    ap.add_argument('--games', type=int, default=150,
                    help='number of V71-vs-rl_v5 games')
    ap.add_argument('--selfplay', type=int, default=50,
                    help='additional V71-vs-V71 self-play games')
    ap.add_argument('--vs-greedy', type=int, default=0)
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--height', type=int, default=32)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--v5-model', default=RL_V5_MODEL_DEFAULT)
    ap.add_argument('--v5-deterministic', action='store_true')
    args = ap.parse_args()

    seed0 = args.seed if args.seed is not None else random.randint(0, 2**30)
    v71 = V71_BOT
    v5_det = ' --deterministic' if args.v5_deterministic else ''
    v5 = f'python -u "{os.path.join(MY_EXT, "rl_v5", "rl_bot.py")}" --model "{args.v5_model}"{v5_det}'

    print(f"=== rl_v9 data generation ===")
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
