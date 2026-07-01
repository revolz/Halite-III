#!/usr/bin/env python3
"""
gen_replay.py — render a watchable .hlt of the CURRENT rl_v6 policy.

PPO training and rl_eval.py run games only for reward/metrics and never write a
replay.  This runs one game with the current checkpoint (player 0 = rl_v6) vs a
frozen opponent and saves a .hlt you can open in replay_viewer.py / the web
viewer.

Usage
-----
    python gen_replay.py                       # latest PPO ckpt vs rl_v5, deterministic
    python gen_replay.py --opponent rl_v4 --seed 12345
    python gen_replay.py --stochastic          # sample actions (see policy variety)
"""

import argparse
import datetime
import os
import sys

_HERE   = os.path.dirname(os.path.abspath(__file__))
_MY_EXT = os.path.dirname(_HERE)
sys.path.insert(0, _MY_EXT)
from halite_engine import HaliteEngine
from rl_eval import v6_cmd, opp_cmd, STARTER


def main():
    ap = argparse.ArgumentParser(description='Render a .hlt of the current rl_v6 policy')
    ap.add_argument('--model', default=os.path.join(_HERE, 'checkpoints', 'model_weights.pt'))
    ap.add_argument('--spawn', default=os.path.join(_HERE, 'checkpoints', 'spawn_weights.pt'))
    ap.add_argument('--opponent', choices=['rl_v5', 'rl_v4'], default='rl_v5')
    ap.add_argument('--seed', type=int, default=20000)
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--height', type=int, default=32)
    ap.add_argument('--out-dir', default=os.path.join(_HERE, 'replays_v6', 'watch'))
    ap.add_argument('--stochastic', action='store_true',
                    help='sample actions instead of greedy (default deterministic)')
    ap.add_argument('--independent', action='store_true',
                    help='disable sequential collision-aware decode (A/B baseline)')
    args = ap.parse_args()

    os.environ['PYTHONPATH'] = STARTER + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    os.makedirs(args.out_dir, exist_ok=True)

    bots = [v6_cmd(args.model, args.spawn, not args.stochastic, args.independent),
            opp_cmd(args.opponent)]
    ts = datetime.datetime.now().strftime('%H%M%S')
    path = os.path.join(args.out_dir,
                        f'v6-vs-{args.opponent}-{args.seed}-{args.width}x{args.height}-{ts}.hlt')

    eng = HaliteEngine(args.width, args.height, 2, seed=args.seed, verbose=False)
    results = dict(eng.run(bots, replay_file=path))
    v6_h, opp_h = results[0], results[1]
    print(f"\nrl_v6={v6_h:,}  {args.opponent}={opp_h:,}  "
          f"{'WIN' if v6_h > opp_h else 'loss'}")
    print(f"Replay: {path}")
    print(f"View:   python \"{os.path.join(_MY_EXT, 'replay_viewer.py')}\" \"{path}\"")


if __name__ == '__main__':
    main()
