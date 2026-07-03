#!/usr/bin/env python3
"""
rl_v9 / rl_eval.py  --  head-to-head win-rate evaluation, subprocess vs
subprocess through the real engine loop (both bots on the hlt protocol),
so the numbers reflect exactly what a real match would produce.

**V71 is the primary objective**: rl_v9's goal is >50% win rate vs V71.

Usage:
    python rl_v9/rl_eval.py --model rl_v9/checkpoints/bc.pt --games 20
    python rl_v9/rl_eval.py --model rl_v9/checkpoints/best.pt --games 50 --save-replays
"""

import argparse
import datetime
import json
import os
import random
import sys

HERE   = os.path.dirname(os.path.abspath(__file__))
MY_EXT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, MY_EXT)

from halite_engine import HaliteEngine  # noqa: E402
import config                           # noqa: E402
from rl_env import opponent_cmd_for     # noqa: E402


def run_match(opponent_name, opponent_cmd, agent_cmd, args):
    os.makedirs(config.REPLAYS_DIR, exist_ok=True)
    seed0 = args.seed if args.seed is not None else random.randint(0, 2**30)

    results = []
    print(f"\n=== rl_v9 eval vs {opponent_name} ({args.games} games, "
          f"{args.width}x{args.height}) ===")
    print(f"model: {args.model}")
    print(f"{'game':>4}  {'rl_v9':>7}  {opponent_name:>7}  result")
    print("-" * 44)

    for i in range(args.games):
        seed = seed0 + i
        replay_path = None
        if args.save_replays:
            ts = datetime.datetime.now().strftime('%H%M%S')
            replay_path = os.path.join(
                config.REPLAYS_DIR,
                f'eval-v9vs{opponent_name}-{seed}-{ts}.hlt')

        engine = HaliteEngine(width=args.width, height=args.height,
                              num_players=2, seed=seed, verbose=False)
        try:
            raw = engine.run([agent_cmd, opponent_cmd], replay_file=replay_path)
        except Exception as e:
            print(f"  game {i+1} FAILED: {e}")
            continue

        hal_v9  = dict(raw)[0]
        hal_opp = dict(raw)[1]
        winner = 'rl_v9' if hal_v9 > hal_opp else opponent_name
        results.append({'hal_v9': hal_v9, 'hal_opp': hal_opp,
                        'winner': winner, 'seed': seed})
        print(f"{i+1:4d}  {hal_v9:7,}  {hal_opp:7,}  {winner}")

    if not results:
        print("No results.")
        return None

    n = len(results)
    wins = sum(1 for r in results if r['winner'] == 'rl_v9')
    mean_v9  = sum(r['hal_v9'] for r in results) / n
    mean_opp = sum(r['hal_opp'] for r in results) / n
    wr = wins / n

    print("-" * 44)
    print(f"win rate  : {wins}/{n} = {wr*100:.1f}%")
    print(f"mean halite: rl_v9={mean_v9:,.0f}  {opponent_name}={mean_opp:,.0f}")
    if wr > 0.50:
        print(f"*** rl_v9 beats {opponent_name}! ({wr*100:.1f}% win rate) ***")
    else:
        print(f"rl_v9 not yet beating {opponent_name} (need >50%, got {wr*100:.1f}%)")

    return {'opponent': opponent_name, 'n': n, 'wins': wins, 'win_rate': wr,
            'mean_v9': mean_v9, 'mean_opp': mean_opp}


def main():
    ap = argparse.ArgumentParser(description='Evaluate rl_v9 (default vs V71).')
    default_model = os.path.join(HERE, 'checkpoints', 'bc.pt')
    ap.add_argument('--model', default=default_model)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--stochastic', action='store_true',
                    help='sample instead of greedy (default greedy)')
    ap.add_argument('--games', type=int, default=20)
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--height', type=int, default=32)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--save-replays', action='store_true')
    ap.add_argument('--opponent', choices=['v71', 'rl_v5', 'rl_v8'], default='v71')
    args = ap.parse_args()

    det = '' if args.stochastic else ' --deterministic'
    agent_cmd = (f'python -u "{os.path.join(HERE, "rl_bot.py")}" '
                 f'--model "{args.model}" --device {args.device}{det}')

    cmd = opponent_cmd_for(args.opponent)
    run_match(args.opponent, cmd, agent_cmd, args)


if __name__ == '__main__':
    main()
