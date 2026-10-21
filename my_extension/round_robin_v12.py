#!/usr/bin/env python3
"""Round robin of the strongest bots (rl_v12, rl_v11, rl_v10, rl_v9, V71) through halite_engine.py.

Every pair plays the same 10 maps, each bot taking both seats on every map: 20 games per pair,
200 games in total. Each bot plays in its normal mode (rl_v12/rl_v11 greedy from their checkpoints,
rl_v10 sampled, rl_v9 deterministic, V71 as written).

Usage (from my_extension/): python round_robin_v12.py [round_robin_v12_results.csv]
Summary of the 2026-10-21 run: README.md, "Final round robin".
"""
import csv, itertools, os, sys, tempfile, time
from multiprocessing import Pool
ME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ME)
PY = sys.executable
BOTS = {
    'rl_v12': f'"{PY}" -u "{ME}/rl_v10/bot.py" --model "{ME}/rl_v12/checkpoints/best.pt" --name rl_v12',
    'rl_v11': f'"{PY}" -u "{ME}/rl_v10/bot.py" --model "{ME}/rl_v11/checkpoints/best.pt" --name rl_v11',
    'rl_v10': f'"{PY}" -u "{ME}/rl_v10/bot.py" --model "{ME}/rl_v10/checkpoints/best.pt" --name rl_v10',
    'rl_v9':  f'"{PY}" -u "{ME}/rl_v9/rl_bot.py" --model "{ME}/rl_v9/checkpoints/best.pt" --deterministic',
    'V71':    f'"{PY}" -u "{ME}/Year 2019/MyBot - V71/MyBot.py"',
}
SEEDS = range(9_100_001, 9_100_011)

def play(job):
    a, b, seed, swap = job
    from halite_engine import HaliteEngine
    os.chdir(tempfile.mkdtemp(prefix='rr5_'))
    p0, p1 = (b, a) if swap else (a, b)
    res = dict(HaliteEngine(32, 32, 2, seed=seed).run([BOTS[p0], BOTS[p1]]))
    return {'a': a, 'b': b, 'seed': seed, 'a_seat': 1 if swap else 0,
            'a_halite': res[1 if swap else 0], 'b_halite': res[0 if swap else 1]}

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    jobs = [(a, b, s, sw) for a, b in itertools.combinations(BOTS, 2) for s in SEEDS for sw in (0, 1)]
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ME, 'round_robin_v12_results.csv'))
    t = time.time()
    with Pool(4) as pool, open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['a', 'b', 'seed', 'a_seat', 'a_halite', 'b_halite'])
        w.writeheader()
        for i, r in enumerate(pool.imap_unordered(play, jobs), 1):
            w.writerow(r); f.flush()
            if i % 20 == 0:
                print(f'{i}/{len(jobs)} games, {time.time() - t:.0f}s', flush=True)
