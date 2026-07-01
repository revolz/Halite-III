#!/usr/bin/env python3
"""
rl_eval.py — head-to-head benchmark for rl_v6.

Runs rl_v6 (the pure bot) vs a frozen opponent (rl_v5 the target, or rl_v4) over
many seeds on the Python engine and reports win-rate and mean final halite.  Both
bots run as their real subprocess bots through the engine, so this measures the
deployed pure-inference policy (including any self-collisions — the cost of 100 %
purity we accept).

Usage
-----
    python rl_eval.py --opponent rl_v5 --games 30 --deterministic
"""

import argparse
import os
import sys

_HERE      = os.path.dirname(os.path.abspath(__file__))
_MY_EXT    = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_MY_EXT)
STARTER    = os.path.join(_REPO_ROOT, 'starter_kits', 'Python3')
sys.path.insert(0, _MY_EXT)
from halite_engine import HaliteEngine


def v6_cmd(model, spawn, deterministic, independent=False):
    det = ' --deterministic' if deterministic else ''
    ind = ' --independent' if independent else ''
    return (f'python -u "{os.path.join(_HERE, "rl_bot.py")}" '
            f'--model "{model}" --spawn-model "{spawn}"{det}{ind}')


def opp_cmd(version):
    folder = os.path.join(_MY_EXT, version)
    model = os.path.join(folder, 'checkpoints', 'model_final_weights.pt')
    return f'python -u "{os.path.join(folder, "rl_bot.py")}" --model "{model}"'


def run_match(model, spawn, opponent='rl_v5', games=30, width=32, height=32,
              seed_start=20000, deterministic=True, verbose=True, independent=False):
    """Run `games` head-to-head games; return a metrics dict.  Importable entry
    point shared by main() and run_pipeline.py."""
    os.environ['PYTHONPATH'] = STARTER + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    bots = [v6_cmd(model, spawn, deterministic, independent), opp_cmd(opponent)]
    wins = my_tot = opp_tot = 0
    for i in range(games):
        seed = seed_start + i
        eng = HaliteEngine(width, height, 2, seed=seed, verbose=False)
        results = dict(eng.run(bots))
        my_h, opp_h = results[0], results[1]
        my_tot += my_h; opp_tot += opp_h
        win = my_h > opp_h; wins += win
        if verbose:
            print(f"[{i+1}/{games}] seed={seed}  rl_v6={my_h:,}  {opponent}={opp_h:,}"
                  f"  {'WIN' if win else 'loss'}", flush=True)
    g = max(1, games)
    return {'opponent': opponent, 'games': games, 'wins': wins,
            'winrate': wins / g, 'mean_v6': my_tot // g, 'mean_opp': opp_tot // g}


def main():
    ap = argparse.ArgumentParser(description='rl_v6 head-to-head benchmark')
    ap.add_argument('--model', default=os.path.join(_HERE, 'checkpoints', 'model_weights.pt'))
    ap.add_argument('--spawn', default=os.path.join(_HERE, 'checkpoints', 'spawn_weights.pt'))
    ap.add_argument('--opponent', choices=['rl_v5', 'rl_v4'], default='rl_v5')
    ap.add_argument('--games', type=int, default=30)
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--height', type=int, default=32)
    ap.add_argument('--seed-start', type=int, default=20000)
    ap.add_argument('--deterministic', action='store_true')
    ap.add_argument('--independent', action='store_true',
                    help='disable sequential collision-aware decode (A/B baseline)')
    args = ap.parse_args()
    r = run_match(args.model, args.spawn, args.opponent, args.games, args.width,
                  args.height, args.seed_start, args.deterministic,
                  independent=args.independent)
    print(f"\nrl_v6 vs {r['opponent']}: {r['wins']}/{r['games']} wins "
          f"({100*r['winrate']:.0f}%)  mean rl_v6={r['mean_v6']:,}  "
          f"mean {r['opponent']}={r['mean_opp']:,}")


if __name__ == '__main__':
    main()
