#!/usr/bin/env python3
"""
run_pipeline.py — ONE command to train rl_v6 and track its improvement.

Runs the whole pipeline end to end:
    generate rl_v5 games -> extract dataset -> behavioral cloning
        -> N DAgger cycles (collect on-policy + retrain)
        -> optional PPO fine-tune
After every training milestone it benchmarks rl_v6 head-to-head vs rl_v5 (and
rl_v4), appends the result to checkpoints/progress.csv, and reprints the full
improvement table so you can watch it get better.

Typical use
-----------
    # imitate rl_v5 (BC + DAgger), no PPO — fastest path to a working bot
    python run_pipeline.py --games 80 --selfplay-games 20 --dagger-cycles 4

    # also try to BEAT rl_v5 with PPO afterwards
    python run_pipeline.py --games 80 --dagger-cycles 4 --ppo-episodes 2000

    # reuse already-generated replays / dataset (skip straight to training)
    python run_pipeline.py --skip-gen --dagger-cycles 4

Resumability: every stage writes checkpoints/model_weights.pt; rerun with
--skip-gen --skip-bc to continue from more DAgger cycles or PPO.
"""

import argparse
import csv
import datetime
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

PY        = sys.executable
REPLAYS   = os.path.join(_HERE, 'replays_v6')
DATASET   = os.path.join(_HERE, 'dataset')
CKPT      = os.path.join(_HERE, 'checkpoints')
MODEL     = os.path.join(CKPT, 'model_weights.pt')
SPAWN     = os.path.join(CKPT, 'spawn_weights.pt')
PROGRESS  = os.path.join(CKPT, 'progress.csv')

from bc_train import run_bc
from rl_eval  import run_match


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def banner(msg):
    print("\n" + "=" * 70 + f"\n  {msg}\n" + "=" * 70, flush=True)


def sh(args):
    """Run a sub-script with live output; abort on failure."""
    args = [str(a) for a in args]
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    print(f"$ {' '.join(args)}", flush=True)
    subprocess.run([PY] + args, check=True, env=env)


def record(stage, bc_val, eval_games, opponents):
    """Benchmark the current model vs each opponent, append a progress row, and
    reprint the whole improvement table."""
    row = {'time': datetime.datetime.now().strftime('%H:%M:%S'),
           'stage': stage, 'bc_val_match': '' if bc_val is None else round(bc_val, 3)}
    for opp in opponents:
        banner(f"benchmark @ {stage}: rl_v6 vs {opp} ({eval_games} games)")
        r = run_match(MODEL, SPAWN, opponent=opp, games=eval_games,
                      deterministic=True, verbose=True)
        row[f'winrate_{opp}'] = round(r['winrate'], 3)
        row[f'mean_v6_vs_{opp}'] = r['mean_v6']
        row[f'mean_{opp}'] = r['mean_opp']

    new = not os.path.isfile(PROGRESS)
    fields = ['time', 'stage', 'bc_val_match']
    for opp in opponents:
        fields += [f'winrate_{opp}', f'mean_v6_vs_{opp}', f'mean_{opp}']
    # rewrite with a stable header (handles added opponents across runs)
    rows = []
    if not new:
        with open(PROGRESS, newline='') as f:
            rows = list(csv.DictReader(f))
    rows.append(row)
    allfields = list(dict.fromkeys(sum([list(r.keys()) for r in rows], []) + fields))
    with open(PROGRESS, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=allfields)
        w.writeheader(); w.writerows(rows)
    _print_table(rows, opponents)


def _print_table(rows, opponents):
    banner("PROGRESS so far (rl_v6 improvement)")
    hdr = f"{'stage':<16}{'bc_match':>9}"
    for opp in opponents:
        hdr += f"{'win%_' + opp:>12}{'v6h_' + opp:>11}{opp + 'h':>11}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = f"{r.get('stage',''):<16}{str(r.get('bc_val_match','')):>9}"
        for opp in opponents:
            wr = r.get(f'winrate_{opp}', '')
            wr = f"{float(wr)*100:.0f}%" if wr != '' else ''
            line += (f"{wr:>12}{str(r.get(f'mean_v6_vs_{opp}','')):>11}"
                     f"{str(r.get(f'mean_{opp}','')):>11}")
        print(line)
    print(flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='One-command rl_v6 training + tracking',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    # data
    ap.add_argument('--games', type=int, default=80, help='rl_v5-vs-rl_v4 games to generate')
    ap.add_argument('--selfplay-games', type=int, default=20, help='rl_v5 self-play games')
    ap.add_argument('--skip-gen', action='store_true', help='reuse existing replays+dataset')
    # BC
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--skip-bc', action='store_true', help='reuse existing BC weights')
    # DAgger
    ap.add_argument('--dagger-cycles', type=int, default=4)
    ap.add_argument('--dagger-games', type=int, default=20)
    ap.add_argument('--dagger-epochs', type=int, default=12)
    # PPO (optional — off by default)
    ap.add_argument('--ppo-episodes', type=int, default=0)
    ap.add_argument('--ppo-opponent', choices=['rl_v5', 'rl_v4', 'greedy'], default='rl_v4')
    # eval / tracking
    ap.add_argument('--eval-games', type=int, default=8)
    ap.add_argument('--eval-opponents', default='rl_v5,rl_v4',
                    help='comma list to benchmark against each milestone')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    opponents = [o.strip() for o in args.eval_opponents.split(',') if o.strip()]
    os.makedirs(CKPT, exist_ok=True)
    t0 = time.time()

    # ---- 1. generate + extract -------------------------------------------
    if not args.skip_gen:
        if args.games > 0:
            banner(f"generate {args.games} rl_v5-vs-rl_v4 games")
            sh(['gen_data.py', '--n-games', args.games, '--mode', 'vs_v4',
                '--out-dir', os.path.join(REPLAYS, 'vs_v4'), '--seed-start', 10000])
            sh(['rl_collect.py', os.path.join(REPLAYS, 'vs_v4'),
                '--output', os.path.join(DATASET, 'vs_v4'), '--player', 0])
        if args.selfplay_games > 0:
            banner(f"generate {args.selfplay_games} rl_v5 self-play games")
            sh(['gen_data.py', '--n-games', args.selfplay_games, '--mode', 'selfplay',
                '--out-dir', os.path.join(REPLAYS, 'selfplay'), '--seed-start', 30000])
            sh(['rl_collect.py', os.path.join(REPLAYS, 'selfplay'),
                '--output', os.path.join(DATASET, 'selfplay'), '--both'])
    else:
        print("Skipping generation (reusing dataset/).")

    # ---- 2. behavioral cloning -------------------------------------------
    if not args.skip_bc:
        banner("behavioral cloning")
        m = run_bc(DATASET, MODEL, SPAWN, epochs=args.epochs, device=args.device)
        record('BC', m['val_match'], args.eval_games, opponents)
    else:
        print("Skipping BC (reusing checkpoints/).")

    # ---- 3. DAgger cycles -------------------------------------------------
    for c in range(1, args.dagger_cycles + 1):
        banner(f"DAgger cycle {c}/{args.dagger_cycles}: collect {args.dagger_games} "
               f"on-policy games labelled by rl_v5")
        sh(['dagger.py', '--policy', MODEL, '--spawn', SPAWN,
            '--n-games', args.dagger_games, '--opponent', 'rl_v5',
            '--out', os.path.join(DATASET, 'dagger'),
            '--tag', f'dagger{c}', '--seed-start', 50000 + 1000 * c])
        banner(f"DAgger cycle {c}: retrain on BC+DAgger data")
        m = run_bc(DATASET, MODEL, SPAWN, epochs=args.dagger_epochs,
                   device=args.device, resume=MODEL)
        record(f'DAgger{c}', m['val_match'], args.eval_games, opponents)

    # ---- 4. optional PPO --------------------------------------------------
    if args.ppo_episodes > 0:
        banner(f"PPO fine-tune: {args.ppo_episodes} episodes vs {args.ppo_opponent}")
        sh(['rl_train.py', '--resume', MODEL, '--spawn', SPAWN,
            '--opponent', args.ppo_opponent, '--episodes', args.ppo_episodes,
            '--checkpoint-dir', CKPT])
        record('PPO', None, args.eval_games, opponents)

    banner(f"pipeline complete in {(time.time()-t0)/60:.1f} min  "
           f"(progress: {PROGRESS})")


if __name__ == '__main__':
    main()
