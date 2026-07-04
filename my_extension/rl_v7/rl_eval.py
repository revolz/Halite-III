#!/usr/bin/env python3
"""
rl_v7 / rl_eval.py  --  head-to-head win-rate evaluation vs rl_v5.

Runs N games between rl_v7 (via the live inference bot) and rl_v5, then
reports win rate, mean deposited halite, and verifies the replay bot-name
entries read "rl_v7" and "rl_v5".

Usage:
    python rl_v7/rl_eval.py --model checkpoints/best.pt --games 20
    python rl_v7/rl_eval.py --model checkpoints/ppo_final.pt --games 50
"""

import argparse
import datetime
import json
import os
import sys
import random

HERE   = os.path.dirname(os.path.abspath(__file__))
MY_EXT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, MY_EXT)

from halite_engine import HaliteEngine  # noqa: E402
import config                           # noqa: E402


def parse_replay_names(path):
    """Return the list of player names from a .hlt replay."""
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        try:
            import zstd
            data = zstd.decompress(raw)
        except Exception:
            data = raw
        replay = json.loads(data)
        return [p['name'] for p in replay['players']]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser(description='Evaluate rl_v7 vs rl_v5.')
    default_model = os.path.join(HERE, 'checkpoints', 'best.pt')
    ap.add_argument('--model', default=default_model)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--deterministic', action='store_true')
    ap.add_argument('--games', type=int, default=20)
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--height', type=int, default=32)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--save-replays', action='store_true',
                    help='save .hlt replays for each game')
    args = ap.parse_args()

    seed0 = args.seed if args.seed is not None else random.randint(0, 2**30)
    det   = ' --deterministic' if args.deterministic else ''
    v7_cmd = (f'python -u "{os.path.join(HERE, "rl_bot.py")}" '
              f'--model "{args.model}" --device {args.device}{det}')
    v5_cmd = f'python -u "{os.path.join(HERE, "v5_opponent.py")}" --deterministic'

    os.makedirs(config.REPLAYS_DIR, exist_ok=True)

    results = []
    print(f"=== rl_v7 eval vs rl_v5 ({args.games} games, 32x32) ===")
    print(f"model: {args.model}")
    print(f"{'game':>4}  {'rl_v7':>7}  {'rl_v5':>7}  result  names-ok")
    print("-" * 50)

    for i in range(args.games):
        seed = seed0 + i
        replay_path = None
        if args.save_replays:
            ts = datetime.datetime.now().strftime('%H%M%S')
            replay_path = os.path.join(
                config.REPLAYS_DIR,
                f'eval-v7vsv5-{seed}-{ts}.hlt')

        engine = HaliteEngine(width=args.width, height=args.height,
                              num_players=2, seed=seed, verbose=False)
        try:
            raw = engine.run([v7_cmd, v5_cmd], replay_file=replay_path)
        except Exception as e:
            print(f"  game {i+1} FAILED: {e}")
            continue

        hal_v7 = dict(raw)[0]
        hal_v5 = dict(raw)[1]
        winner = 'rl_v7' if hal_v7 > hal_v5 else 'rl_v5'

        names_ok = 'n/a'
        if replay_path:
            names = parse_replay_names(replay_path)
            if len(names) >= 2:
                ok = (names[0] == 'rl_v7' and names[1] == 'rl_v5')
                names_ok = 'OK' if ok else f'FAIL:{names}'

        results.append({'hal_v7': hal_v7, 'hal_v5': hal_v5,
                        'winner': winner, 'seed': seed})
        print(f"{i+1:4d}  {hal_v7:7,}  {hal_v5:7,}  {winner:6}  {names_ok}")

    if not results:
        print("No results.")
        return

    n = len(results)
    wins = sum(1 for r in results if r['winner'] == 'rl_v7')
    mean_v7 = sum(r['hal_v7'] for r in results) / n
    mean_v5 = sum(r['hal_v5'] for r in results) / n
    wr = wins / n

    print("-" * 50)
    print(f"win rate  : {wins}/{n} = {wr*100:.1f}%")
    print(f"mean halite: rl_v7={mean_v7:,.0f}  rl_v5={mean_v5:,.0f}")
    if wr >= 0.50:
        print(f"\n*** rl_v7 beats rl_v5! ({wr*100:.1f}% win rate) ***")
    else:
        print(f"\nrl_v7 not yet beating rl_v5 (need >50%, got {wr*100:.1f}%)")


if __name__ == '__main__':
    main()
