#!/usr/bin/env python3
"""
rl_v10 / evaluate.py -- head-to-head matches through the real engine
(halite_engine.py, bots as subprocesses), seats alternating, games in parallel.

Usage:
  python rl_v10/evaluate.py --opponent v71 --games 100 --seed0 100000
  python rl_v10/evaluate.py --opponent rl_v9 --games 100 --model rl_v10/checkpoints/best.pt
"""
import argparse
import csv
import math
import os
import sys
import tempfile
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
MY_EXT = os.path.dirname(HERE)
sys.path.insert(0, MY_EXT)
PY = sys.executable


def bot_cmd(name, model=None, greedy=None, label='rl_v10'):
    if name == 'rl_v10':
        g = {None: '', True: ' --greedy', False: ' --sample'}[greedy]
        model = model or f'{HERE}/checkpoints/best.pt'
        return f'"{PY}" -u "{HERE}/bot.py" --model "{model}" --name {label}{g}'
    if name == 'v71':
        return f'"{PY}" -u "{MY_EXT}/Year 2019/MyBot - V71/MyBot.py"'
    if name == 'rl_v9':
        return (f'"{PY}" -u "{MY_EXT}/rl_v9/rl_bot.py" '
                f'--model "{MY_EXT}/rl_v9/checkpoints/best.pt" --deterministic')
    raise ValueError(name)


def play(job):
    seed, v10_seat, cmd_v10, cmd_opp, replay_dir = job
    from halite_engine import HaliteEngine
    cmds = [cmd_v10, cmd_opp] if v10_seat == 0 else [cmd_opp, cmd_v10]
    wd = tempfile.mkdtemp(prefix='v10eval_')
    os.chdir(wd)
    rp = os.path.join(replay_dir, f'v10-seed{seed}-seat{v10_seat}.hlt') if replay_dir else None
    t = time.time()
    try:
        res = dict(HaliteEngine(32, 32, 2, seed=seed).run(cmds, replay_file=rp))
        err = ''
    except Exception as e:  # noqa: BLE001
        res, err = {0: -1, 1: -1}, str(e)[:200]
    return seed, v10_seat, res[v10_seat], res[1 - v10_seat], round(time.time() - t, 1), err


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(HERE, 'checkpoints', 'best.pt'))
    ap.add_argument('--opponent', required=True,
                    help="v71, rl_v9, rl_v10, or a label for --opp-model (e.g. rl_v11_sampled)")
    ap.add_argument('--opp-model', default=None, help='checkpoint for a network opponent other than rl_v10')
    om = ap.add_mutually_exclusive_group()
    om.add_argument('--opp-greedy', dest='opp_greedy', action='store_const', const=True, default=None)
    om.add_argument('--opp-sample', dest='opp_greedy', action='store_const', const=False)
    ap.add_argument('--name', default='rl_v10', help='label of the evaluated bot (e.g. rl_v11)')
    ap.add_argument('--games', type=int, default=20)
    ap.add_argument('--seed0', type=int, default=500000)
    ap.add_argument('--workers', type=int, default=4)
    gm = ap.add_mutually_exclusive_group()
    gm.add_argument('--greedy', dest='greedy', action='store_const', const=True, default=None,
                    help='force greedy play (default: the checkpoint setting)')
    gm.add_argument('--sample', dest='greedy', action='store_const', const=False, help='force sampled play')
    ap.add_argument('--replays', default=None, help='directory to save .hlt replays')
    ap.add_argument('--csv', default=None)
    args = ap.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    if args.replays:
        os.makedirs(args.replays, exist_ok=True)
    c10 = bot_cmd('rl_v10', os.path.abspath(args.model), args.greedy, args.name)
    if args.opp_model:
        co = bot_cmd('rl_v10', os.path.abspath(args.opp_model), args.opp_greedy, args.opponent)
    else:
        co = bot_cmd(args.opponent)
    jobs = [(args.seed0 + i // 2, i % 2, c10, co, args.replays) for i in range(args.games)]
    wins = n = 0
    tot10 = tot_o = 0
    rows = []
    with Pool(args.workers) as pool:
        for seed, seat, h10, ho, sec, err in pool.imap_unordered(play, jobs):
            if err:
                print(f'seed {seed} seat {seat} ERROR {err}', flush=True)
                continue
            n += 1
            wins += h10 > ho
            tot10 += h10
            tot_o += ho
            rows.append((seed, seat, h10, ho, sec))
            print(f'[{n}/{args.games}] seed {seed} seat {seat}: {args.name} {h10:,} vs {args.opponent} {ho:,} '
                  f'-> {"WIN" if h10 > ho else "loss"}  ({sec}s)  running {wins}/{n}', flush=True)
    lo, hi = wilson(wins, n)
    print(f'RESULT {args.name} vs {args.opponent}: {wins}/{n} = {100 * wins / max(n, 1):.1f}%  '
          f'95% CI [{100 * lo:.1f}, {100 * hi:.1f}]  mean halite {tot10 / max(n, 1):,.0f} vs {tot_o / max(n, 1):,.0f}')
    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(['model', 'opponent', 'seed', 'v10_seat', 'v10', 'opp', 'seconds'])
            for r in rows:
                w.writerow([os.path.basename(args.model), args.opponent, *r])


if __name__ == '__main__':
    main()
