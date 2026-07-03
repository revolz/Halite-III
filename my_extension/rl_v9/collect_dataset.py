#!/usr/bin/env python3
"""
rl_v9 / collect_dataset.py  --  turn V71 replays into BC datasets.

Two datasets are produced (both in rl_v9/dataset/):

SHIP dataset (one row per V71 ship-turn):
    features.csv   row_id, game_id, player_id, ship_id, turn,
                   <46 scalar features incl. memory features>, action
    patches.npy    float16 [N, 9, 9, 6]     local patch  (HWC)
    globals.npy    float16 [N, 4, 8, 8]     ship-centred coarse global (CHW)

SPAWN dataset (one row per V71 turn where spawning was AFFORDABLE):
    spawn.csv           row_id, game_id, player_id, turn,
                        <18 spawn scalars>, action (0=no, 1=spawn)
    spawn_globals.npy   float16 [M, 4, 8, 8]  factory-centred coarse global

Memory features (homing / prev-action / stuck) are reconstructed by running
the exact same FleetMemory over the replay turn sequence that the live bot
and the RL environment run at play time -- V71's executed action at turn t-1
becomes the prev-action feature at turn t.

Only seats named "RevolzBot" (V71's game.ready() name) are collected.

Usage:
    python rl_v9/collect_dataset.py
    python rl_v9/collect_dataset.py --stay-keep 0.5
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
    INITIAL_ENERGY, SHIP_COST, ACTION_STAY, ACTION_DROPOFF, DIR_TO_ACTION,
    ACTION_DELTA, ACTION_NAMES, FEATURE_NAMES, SPAWN_FEATURE_NAMES,
    SPAWN_NO, SPAWN_YES, game_max_turns,
)
import features as featmod                           # noqa: E402
from features import FleetMemory                     # noqa: E402


def load_replay(path):
    with open(path, 'rb') as f:
        raw = f.read()
    try:
        import zstd
        data = zstd.decompress(raw)
    except Exception:
        data = raw
    return json.loads(data)


def parse_moves(player_moves):
    """ship_id -> action index; also returns whether a spawn was issued."""
    out = {}
    spawned = False
    for mv in player_moves:
        t = mv.get('type')
        if t == 'm':
            out[int(mv['id'])] = DIR_TO_ACTION.get(mv['direction'], ACTION_STAY)
        elif t == 'c':
            out[int(mv['id'])] = ACTION_DROPOFF
        elif t == 'g':
            spawned = True
    return out, spawned


def collect_replay(path, ship_rows, patches, globals_list,
                   spawn_rows, spawn_globals, counts, spawn_counts,
                   align, rng, stay_keep, game_id):
    replay = load_replay(path)
    pm = replay['production_map']
    W, H = pm['width'], pm['height']
    grid = pm['grid']
    num_players = replay['number_of_players']
    max_turns = game_max_turns(W, H)

    collect_pids = [p['player_id'] for p in replay['players']
                    if p.get('name') == 'RevolzBot']
    if not collect_pids:
        collect_pids = [0]

    factories = [None] * num_players
    for p in replay['players']:
        fl = p['factory_location']
        factories[p['player_id']] = (fl['x'], fl['y'])

    halite = {(x, y): grid[y][x]['energy'] for y in range(H) for x in range(W)}
    dropoffs_by_pid = {pid: [] for pid in range(num_players)}
    bank = {pid: INITIAL_ENERGY for pid in range(num_players)}
    memories = {pid: FleetMemory() for pid in collect_pids}

    frames = replay['full_frames']
    n_turns = len(frames) - 1              # frame 0 is the empty pre-game frame

    for t in range(1, n_turns + 1):
        frame = frames[t]
        entities = frame['entities']
        moves = frame['moves']

        for pid in collect_pids:
            wv = featmod.world_view_from_replay(
                pid, W, H, t, max_turns, halite, factories,
                dropoffs_by_pid, entities, bank, num_players)
            mem = memories[pid]
            mem.begin_turn(wv)
            label_map, spawned = parse_moves(moves.get(str(pid), []))

            # ---- spawn row (only when the choice actually existed) ----
            if wv.my_bank >= SHIP_COST:
                srow_id = len(spawn_rows)
                sscal = featmod.extract_spawn_scalars(wv)
                sglob = featmod.extract_spawn_global(wv)
                spawn_rows.append((srow_id, game_id, pid, t, sscal,
                                   SPAWN_YES if spawned else SPAWN_NO))
                spawn_globals.append(sglob.astype(np.float16))
                spawn_counts[SPAWN_YES if spawned else SPAWN_NO] += 1

            # ---- ship rows ----
            for ship_id, (sx, sy, cargo) in wv.my_ships.items():
                action = label_map.get(ship_id, ACTION_STAY)

                if action == ACTION_STAY and stay_keep < 1.0 and rng.random() > stay_keep:
                    continue

                scal = featmod.extract_scalars(wv, ship_id, mem)
                patch = featmod.extract_patch(wv, ship_id)
                gmap = featmod.extract_ship_global(wv, ship_id)
                row_id = len(ship_rows)
                ship_rows.append((row_id, game_id, pid, ship_id, t, scal, action))
                patches.append(patch.astype(np.float16))
                globals_list.append(gmap.astype(np.float16))
                counts[action] += 1

                # alignment sanity check on a sample of MOVE rows
                if action in ACTION_DELTA and action != ACTION_STAY and rng.random() < 0.02:
                    dxy = ACTION_DELTA[action]
                    nx, ny = (sx + dxy[0]) % W, (sy + dxy[1]) % H
                    nxt = frames[t + 1]['entities'].get(str(pid), {}) if t < n_turns else {}
                    e = nxt.get(str(ship_id))
                    if e is not None:
                        align['total'] += 1
                        if e['x'] == nx and e['y'] == ny:
                            align['match'] += 1

            # commit executed actions -> next turn's prev-action feature
            commits = {sid: label_map.get(sid, ACTION_STAY)
                       for sid in wv.my_ships}
            mem.commit_actions(commits)

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
    ap = argparse.ArgumentParser(description='Build the rl_v9 BC datasets from replays.')
    ap.add_argument('--replays-dir', default=config.REPLAYS_DIR)
    ap.add_argument('--stay-keep', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    paths = sorted(glob.glob(os.path.join(args.replays_dir, '*.hlt')))
    if not paths:
        print(f"No replays found in {args.replays_dir}. Run generate_games.py first.")
        return

    print(f"Found {len(paths)} replays in {args.replays_dir}")
    ship_rows, patches, globals_list = [], [], []
    spawn_rows, spawn_globals = [], []
    counts = {a: 0 for a in range(len(ACTION_NAMES))}
    spawn_counts = {SPAWN_NO: 0, SPAWN_YES: 0}
    align = {'match': 0, 'total': 0}

    for gi, path in enumerate(paths):
        try:
            collect_replay(path, ship_rows, patches, globals_list,
                           spawn_rows, spawn_globals, counts, spawn_counts,
                           align, rng, args.stay_keep, gi)
        except Exception as e:
            print(f"  skipped {os.path.basename(path)}: {e}")
            continue
        if (gi + 1) % 10 == 0:
            print(f"  processed {gi+1}/{len(paths)} replays, "
                  f"{len(ship_rows)} ship rows, {len(spawn_rows)} spawn rows")

    if not ship_rows:
        print("No rows collected.")
        return

    os.makedirs(config.DATASET_DIR, exist_ok=True)

    np.save(config.PATCHES_NPY, np.stack(patches).astype(np.float16))
    np.save(config.GLOBALS_NPY, np.stack(globals_list).astype(np.float16))
    np.save(config.SPAWN_GLOBALS_NPY, np.stack(spawn_globals).astype(np.float16))

    import csv
    header = (['row_id', 'game_id', 'player_id', 'ship_id', 'turn']
              + FEATURE_NAMES + ['action'])
    with open(config.FEATURES_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for (row_id, game_id, pid, ship_id, turn, scal, action) in ship_rows:
            w.writerow([row_id, game_id, pid, ship_id, turn]
                       + [f"{v:.5g}" for v in scal] + [action])

    sheader = (['row_id', 'game_id', 'player_id', 'turn']
               + SPAWN_FEATURE_NAMES + ['action'])
    with open(config.SPAWN_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(sheader)
        for (row_id, game_id, pid, turn, scal, action) in spawn_rows:
            w.writerow([row_id, game_id, pid, turn]
                       + [f"{v:.5g}" for v in scal] + [action])

    total = len(ship_rows)
    print(f"\n=== datasets written ===")
    print(f"ship rows    : {total}")
    print(f"spawn rows   : {len(spawn_rows)}")
    print(f"\nship action distribution:")
    for a in range(len(ACTION_NAMES)):
        print(f"  {ACTION_NAMES[a]:8s}: {counts[a]:8d}  ({100*counts[a]/total:5.1f}%)")
    st = max(1, len(spawn_rows))
    print(f"spawn distribution: NO={spawn_counts[SPAWN_NO]} "
          f"({100*spawn_counts[SPAWN_NO]/st:.1f}%)  "
          f"YES={spawn_counts[SPAWN_YES]} ({100*spawn_counts[SPAWN_YES]/st:.1f}%)")
    if align['total']:
        frac = align['match'] / align['total']
        status = "OK" if frac > 0.98 else "*** MISALIGNED ***"
        print(f"\nalignment sanity check: {align['match']}/{align['total']} "
              f"moves landed as labelled ({100*frac:.1f}%)  {status}")
    else:
        print("\nalignment sanity check: no sampled move rows to verify")


if __name__ == '__main__':
    main()
