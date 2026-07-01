#!/usr/bin/env python3
"""
rl_v7 / generate_games.py  --  have rl_v5 play many games and save the replays.

These replays are the raw material for the behavioral-cloning dataset: we will
parse them (collect_dataset.py) to recover, for every rl_v5 ship-turn, the
(features, action) pair we want rl_v7 to imitate.

rl_v5 is run via v5_opponent.py (the pristine rl_v5 bot, reported as "rl_v5").
By default both seats are rl_v5 (self-play) which keeps the visited states on
rl_v5's own distribution; pass --vs-greedy to also generate games against the
simple starter-kit bot for extra state diversity.

rl_v5 is run STOCHASTICALLY by default (it samples actions) so the dataset
covers a wider slice of states than a single deterministic line of play.

Usage:
    python rl_v7/generate_games.py --games 60
    python rl_v7/generate_games.py --games 100 --vs-greedy 20 --width 32
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

V5_BOT = f'python -u "{os.path.join(HERE, "v5_opponent.py")}"'
GREEDY_BOT = f'python -u "{os.path.join(REPO_ROOT, "starter_kits", "Python3", "MyBot.py")}"'


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
    ap = argparse.ArgumentParser(description='Generate rl_v5 replays for rl_v7 BC.')
    ap.add_argument('--games', type=int, default=60,
                    help='number of rl_v5 self-play games (default 60)')
    ap.add_argument('--vs-greedy', type=int, default=0,
                    help='additional games vs the starter-kit bot (default 0)')
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--height', type=int, default=32)
    ap.add_argument('--seed', type=int, default=None,
                    help='base seed (default: random)')
    ap.add_argument('--deterministic', action='store_true',
                    help='run rl_v5 deterministically (less state diversity)')
    args = ap.parse_args()

    seed0 = args.seed if args.seed is not None else random.randint(0, 2**30)
    det = ' --deterministic' if args.deterministic else ''
    v5 = V5_BOT + det

    print(f"=== rl_v7 data generation ===")
    print(f"self-play games : {args.games}")
    print(f"vs-greedy games : {args.vs_greedy}")
    print(f"map             : {args.width}x{args.height}")
    print(f"base seed       : {seed0}")
    print(f"replays -> {config.REPLAYS_DIR}\n")

    if args.games > 0:
        print("rl_v5 self-play:")
        run_batch(args.games, args.width, args.height, [v5, v5],
                  tag='v5self', seed0=seed0)
    if args.vs_greedy > 0:
        print("\nrl_v5 vs greedy:")
        run_batch(args.vs_greedy, args.width, args.height, [v5, GREEDY_BOT],
                  tag='v5greedy', seed0=seed0 + 100000)

    print("\nDone.")


if __name__ == '__main__':
    main()
