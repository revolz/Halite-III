#!/usr/bin/env python3
"""
rl_v8 / rl_eval.py  --  head-to-head win-rate evaluation vs V71, rl_v5, and rl_v7.

Runs N games between rl_v8 (via the live inference bot) and the chosen
opponent(s), reports win rate + mean deposited halite for each matchup, and
verifies the replay bot-name entries are correct.

**V71 is the primary objective bot** (default --opponent), since rl_v8's goal
is to become a better bot than the one it imitated. rl_v5/rl_v7 remain
available for secondary benchmarking.

Usage:
    python rl_v8/rl_eval.py --model checkpoints/best.pt --games 20            # vs V71 (default)
    python rl_v8/rl_eval.py --model checkpoints/ppo_final.pt --games 50 --save-replays
    python rl_v8/rl_eval.py --model checkpoints/ppo_final.pt --opponent rl_v5   # single matchup
    python rl_v8/rl_eval.py --model checkpoints/ppo_final.pt --opponent all     # v71 + rl_v5 + rl_v7
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
from rl_env import opponent_cmd_for     # noqa: E402


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


def run_match(opponent_name, opponent_cmd, agent_cmd, args):
    os.makedirs(config.REPLAYS_DIR, exist_ok=True)
    seed0 = args.seed if args.seed is not None else random.randint(0, 2**30)

    results = []
    print(f"\n=== rl_v8 eval vs {opponent_name} ({args.games} games, "
          f"{args.width}x{args.height}) ===")
    print(f"model: {args.model}")
    print(f"{'game':>4}  {'rl_v8':>7}  {opponent_name:>7}  result  names-ok")
    print("-" * 50)

    for i in range(args.games):
        seed = seed0 + i
        replay_path = None
        if args.save_replays:
            ts = datetime.datetime.now().strftime('%H%M%S')
            replay_path = os.path.join(
                config.REPLAYS_DIR,
                f'eval-v8vs{opponent_name}-{seed}-{ts}.hlt')

        engine = HaliteEngine(width=args.width, height=args.height,
                              num_players=2, seed=seed, verbose=False)
        try:
            raw = engine.run([agent_cmd, opponent_cmd], replay_file=replay_path)
        except Exception as e:
            print(f"  game {i+1} FAILED: {e}")
            continue

        hal_v8  = dict(raw)[0]
        hal_opp = dict(raw)[1]
        winner = 'rl_v8' if hal_v8 > hal_opp else opponent_name

        names_ok = 'n/a'
        if replay_path:
            names = parse_replay_names(replay_path)
            if len(names) >= 2:
                names_ok = f'{names}'

        results.append({'hal_v8': hal_v8, 'hal_opp': hal_opp,
                        'winner': winner, 'seed': seed})
        print(f"{i+1:4d}  {hal_v8:7,}  {hal_opp:7,}  {winner:6}  {names_ok}")

    if not results:
        print("No results.")
        return None

    n = len(results)
    wins = sum(1 for r in results if r['winner'] == 'rl_v8')
    mean_v8  = sum(r['hal_v8'] for r in results) / n
    mean_opp = sum(r['hal_opp'] for r in results) / n
    wr = wins / n

    print("-" * 50)
    print(f"win rate  : {wins}/{n} = {wr*100:.1f}%")
    print(f"mean halite: rl_v8={mean_v8:,.0f}  {opponent_name}={mean_opp:,.0f}")
    if wr >= 0.50:
        print(f"*** rl_v8 beats {opponent_name}! ({wr*100:.1f}% win rate) ***")
    else:
        print(f"rl_v8 not yet beating {opponent_name} (need >50%, got {wr*100:.1f}%)")

    return {'opponent': opponent_name, 'n': n, 'wins': wins, 'win_rate': wr,
            'mean_v8': mean_v8, 'mean_opp': mean_opp}


def main():
    ap = argparse.ArgumentParser(description='Evaluate rl_v8 vs V71 (default), rl_v5, and/or rl_v7.')
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
    ap.add_argument('--opponent', choices=['v71', 'rl_v5', 'rl_v7', 'both', 'all'], default='v71',
                    help="which opponent(s) to evaluate against (default 'v71', the "
                         "primary objective bot; 'both' = rl_v5+rl_v7; 'all' = v71+rl_v5+rl_v7)")
    ap.add_argument('--v5-model', default=None)
    ap.add_argument('--v7-model', default=None)
    args = ap.parse_args()

    det = ' --deterministic' if args.deterministic else ''
    agent_cmd = (f'python -u "{os.path.join(HERE, "rl_bot.py")}" '
                 f'--model "{args.model}" --device {args.device}{det}')

    if args.opponent == 'both':
        opponents = ['rl_v5', 'rl_v7']
    elif args.opponent == 'all':
        opponents = ['v71', 'rl_v5', 'rl_v7']
    else:
        opponents = [args.opponent]

    summaries = []
    for name in opponents:
        model_override = {'rl_v5': args.v5_model, 'rl_v7': args.v7_model}.get(name)
        cmd = opponent_cmd_for(name, model_override)
        summary = run_match(name, cmd, agent_cmd, args)
        if summary:
            summaries.append(summary)

    if len(summaries) > 1:
        print("\n=== combined summary ===")
        for s in summaries:
            print(f"  vs {s['opponent']:6s}: {s['wins']}/{s['n']} "
                  f"({s['win_rate']*100:.1f}%)  mean {s['mean_v8']:,.0f} vs {s['mean_opp']:,.0f}")


if __name__ == '__main__':
    main()
