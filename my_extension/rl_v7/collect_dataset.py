#!/usr/bin/env python3
"""
rl_v7 / collect_dataset.py  --  turn rl_v5 replays into a behavioral-cloning
dataset.

For every replay in the replays directory we walk the game turn by turn,
reconstruct the board state at the START of each turn, and for each rl_v5 ship
emit one row:

    features (the ~45 named scalars)  +  a 9x9x6 local map patch  +  the action
    rl_v5 actually took that turn (STAY / N / E / S / W / DROPOFF).

Outputs (in rl_v7/dataset/):
    features.csv   inspectable: row_id, game_id, player_id, ship_id, turn,
                   <45 feature columns>, action
    patches.npy    float16 [N, 9, 9, 6]; patches[row_id] matches CSV row_id

Only players whose recorded replay name is "rl_v5" are collected, so self-play
replays contribute both seats and rl_v5-vs-greedy replays contribute only the
rl_v5 seat -- automatically.

An alignment sanity check verifies, on a random sample, that a ship labelled
"move <dir>" is actually found one step in <dir> on the next frame.  If this
fails, features and labels are misaligned and you must NOT train on the data.

Usage:
    python rl_v7/collect_dataset.py
    python rl_v7/collect_dataset.py --stay-keep 0.4     # subsample STAY rows
"""

import argparse
import glob
import json
import os
import random
import sys

import numpy as np

HERE   = os.path.dirname(os.path.abspath(__file__))
MY_EXT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, MY_EXT)

import config                                       # noqa: E402
from config import (                                # noqa: E402
    INITIAL_ENERGY, ACTION_STAY, ACTION_DROPOFF, DIR_TO_ACTION,
    ACTION_DELTA, ACTION_NAMES, FEATURE_NAMES, game_max_turns,
)
import features as featmod                           # noqa: E402


def load_replay(path):
    with open(path, 'rb') as f:
        raw = f.read()
    try:
        import zstd
        data = zstd.decompress(raw)
    except Exception:
        data = raw           # maybe stored as plain JSON
    return json.loads(data)


def parse_moves(player_moves):
    """Map ship_id -> action index from a frame's per-player move list."""
    out = {}
    for mv in player_moves:
        t = mv.get('type')
        if t == 'm':
            out[int(mv['id'])] = DIR_TO_ACTION.get(mv['direction'], ACTION_STAY)
        elif t == 'c':
            out[int(mv['id'])] = ACTION_DROPOFF
        # 'g' (spawn) is a per-player action, not a per-ship label
    return out


def collect_replay(path, rows, patches, counts, align, rng, stay_keep, game_id):
    replay = load_replay(path)
    pm = replay['production_map']
    W, H = pm['width'], pm['height']
    grid = pm['grid']
    num_players = replay['number_of_players']
    max_turns = game_max_turns(W, H)

    # which seats are rl_v5
    collect_pids = [p['player_id'] for p in replay['players']
                    if p.get('name') == 'rl_v5']
    if not collect_pids:
        # fall back: if nothing is named rl_v5, collect player 0
        collect_pids = [0]

    factories = [None] * num_players
    for p in replay['players']:
        fl = p['factory_location']
        factories[p['player_id']] = (fl['x'], fl['y'])

    # initial state
    halite = {(x, y): grid[y][x]['energy'] for y in range(H) for x in range(W)}
    dropoffs_by_pid = {pid: [] for pid in range(num_players)}
    bank = {pid: INITIAL_ENERGY for pid in range(num_players)}

    frames = replay['full_frames']
    n_turns = len(frames) - 1                      # frame 0 is the empty pre-game frame

    for t in range(1, n_turns + 1):
        frame = frames[t]
        entities = frame['entities']
        moves = frame['moves']

        for pid in collect_pids:
            wv = featmod.world_view_from_replay(
                pid, W, H, t, max_turns, halite, factories,
                dropoffs_by_pid, entities, bank, num_players)
            label_map = parse_moves(moves.get(str(pid), []))

            for ship_id, (sx, sy, cargo) in wv.my_ships.items():
                action = label_map.get(ship_id, ACTION_STAY)

                # optional STAY subsampling to balance the heavily STAY-skewed data
                if action == ACTION_STAY and stay_keep < 1.0 and rng.random() > stay_keep:
                    continue

                scal = featmod.extract_scalars(wv, ship_id)
                patch = featmod.extract_patch(wv, ship_id)
                row_id = len(rows)
                rows.append((row_id, game_id, pid, ship_id, t, scal, action))
                patches.append(patch.astype(np.float16))
                counts[action] += 1

                # alignment sanity check on a sample of MOVE rows
                if action in ACTION_DELTA and action != ACTION_STAY and rng.random() < 0.02:
                    dxy = ACTION_DELTA[action]
                    nx, ny = (sx + dxy[0]) % W, (sy + dxy[1]) % H
                    nxt = frames[t + 1]['entities'].get(str(pid), {}) if t < n_turns else {}
                    e = nxt.get(str(ship_id))
                    if e is not None:               # ship survived; verify it moved as labelled
                        align['total'] += 1
                        if e['x'] == nx and e['y'] == ny:
                            align['match'] += 1

        # advance state to the END of turn t (-> start of t+1)
        for ev in frame.get('events', []):
            if ev.get('type') == 'construct':
                owner = ev['owner_id']
                loc = ev['location']
                dropoffs_by_pid[owner].append((loc['x'], loc['y']))
        for c in frame.get('cells', []):
            halite[(c['x'], c['y'])] = c['production']
        if frame.get('energy'):
            bank = {int(k): v for k, v in frame['energy'].items()}


def main():
    ap = argparse.ArgumentParser(description='Build the rl_v7 BC dataset from replays.')
    ap.add_argument('--replays-dir', default=config.REPLAYS_DIR)
    ap.add_argument('--stay-keep', type=float, default=1.0,
                    help='fraction of STAY rows to keep (default 1.0 = keep all)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    paths = sorted(glob.glob(os.path.join(args.replays_dir, '*.hlt')))
    if not paths:
        print(f"No replays found in {args.replays_dir}. Run generate_games.py first.")
        return

    print(f"Found {len(paths)} replays in {args.replays_dir}")
    rows, patches = [], []
    counts = {a: 0 for a in range(len(ACTION_NAMES))}
    align = {'match': 0, 'total': 0}

    for gi, path in enumerate(paths):
        try:
            collect_replay(path, rows, patches, counts, align, rng,
                           args.stay_keep, gi)
        except Exception as e:
            print(f"  skipped {os.path.basename(path)}: {e}")
            continue
        if (gi + 1) % 10 == 0:
            print(f"  processed {gi+1}/{len(paths)} replays, {len(rows)} rows so far")

    if not rows:
        print("No rows collected.")
        return

    os.makedirs(config.DATASET_DIR, exist_ok=True)

    # --- write patches.npy ---
    patch_arr = np.stack(patches).astype(np.float16)
    np.save(config.PATCHES_NPY, patch_arr)

    # --- write features.csv ---
    header = (['row_id', 'game_id', 'player_id', 'ship_id', 'turn']
              + FEATURE_NAMES + ['action'])
    import csv
    with open(config.FEATURES_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for (row_id, game_id, pid, ship_id, turn, scal, action) in rows:
            w.writerow([row_id, game_id, pid, ship_id, turn]
                       + [f"{v:.5g}" for v in scal] + [action])

    # --- report ---
    total = len(rows)
    print(f"\n=== dataset written ===")
    print(f"rows         : {total}")
    print(f"features.csv : {config.FEATURES_CSV}")
    print(f"patches.npy  : {config.PATCHES_NPY}  shape={patch_arr.shape} dtype={patch_arr.dtype}")
    print(f"\naction distribution:")
    for a in range(len(ACTION_NAMES)):
        print(f"  {ACTION_NAMES[a]:8s}: {counts[a]:8d}  ({100*counts[a]/total:5.1f}%)")
    if align['total']:
        frac = align['match'] / align['total']
        status = "OK" if frac > 0.98 else "*** MISALIGNED ***"
        print(f"\nalignment sanity check: {align['match']}/{align['total']} "
              f"moves landed as labelled ({100*frac:.1f}%)  {status}")
    else:
        print("\nalignment sanity check: no sampled move rows to verify")


if __name__ == '__main__':
    main()
