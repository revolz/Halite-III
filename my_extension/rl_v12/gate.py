#!/usr/bin/env python3
"""
rl_v12 / gate.py -- champion gate for continual training.

Watches rl_v12/checkpoints for new snapshots (every --every iterations), plays
each candidate (greedy) against the current champion (greedy) through the real
engine, and promotes it to checkpoints/best.pt only if it wins at least
--need of --games.  The training league reads best.pt as "champion", so every
promotion also hands the learner a stronger opponent.  Runs until stopped.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MY_EXT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(MY_EXT, 'rl_v10'))
from model import load, save  # noqa: E402

CK = os.path.join(HERE, 'checkpoints')


def match(cand, champ, games, seed0, workers):
    out = subprocess.run(
        [sys.executable, os.path.join(MY_EXT, 'rl_v10', 'evaluate.py'), '--model', cand, '--greedy',
         '--name', 'candidate', '--opponent', 'champion', '--opp-model', champ, '--opp-greedy',
         '--games', str(games), '--seed0', str(seed0), '--workers', str(workers),
         '--csv', os.path.join(CK, 'gate_games.csv')],
        cwd=MY_EXT, capture_output=True, text=True).stdout
    m = re.search(r'RESULT .*?: (\d+)/(\d+)', out)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--every', type=int, default=20)
    ap.add_argument('--games', type=int, default=48)
    ap.add_argument('--need', type=int, default=29, help='wins needed to promote (29/48 = 60%%)')
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    champ = os.path.join(CK, 'best.pt')
    log = os.path.join(CK, 'gate_log.jsonl')
    seen = set()
    if os.path.exists(log):
        seen = {json.loads(l)['candidate'] for l in open(log)}
    while True:
        for s in sorted(glob.glob(os.path.join(CK, 'snap_*.pt'))):
            it = int(re.search(r'snap_(\d+)', s).group(1))
            if it % a.every or os.path.basename(s) in seen:
                continue
            w, n = match(s, champ, a.games, 6_000_000 + it * 100, a.workers)
            promoted = n > 0 and w >= a.need
            if promoted:
                net, extra = load(s)
                save(net, champ, dict(extra, greedy=True, promoted_at_iter=it))
            rec = {'candidate': os.path.basename(s), 'iter': it, 'wins': w, 'games': n,
                   'promoted': promoted, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
            open(log, 'a').write(json.dumps(rec) + '\n')
            print(rec, flush=True)
            seen.add(os.path.basename(s))
        time.sleep(120)


if __name__ == '__main__':
    main()
