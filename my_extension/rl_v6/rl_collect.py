"""
rl_v6 imitation-learning dataset extractor for Halite III.

Reads .hlt replay files of rl_v5 games and converts a chosen player's
trajectories into PURE (observation, action) pairs for behavioral cloning of
rl_v5 — plus a per-turn spawn-decision dataset for the learned spawn head.

Differences from the rl_v5 collector:
  * targets a SPECIFIC player slot (the rl_v5 bot), not just `best_player`;
  * keeps construct ('c') ships and labels them ACTION_DROPOFF (5), giving the
    full 6-action rl_v6 space (0-4 moves + dropoff);
  * additionally emits a per-turn spawn dataset (global features + 0/1 label).

Usage
-----
    # Extract player 0 from a single replay
    python rl_collect.py replay.hlt --output dataset/ep_001.npz --player 0

    # Batch extract all replays in a directory (player 0)
    python rl_collect.py replays/ --output dataset/ --player 0

    # Self-play replays: extract BOTH players
    python rl_collect.py replays/ --output dataset/ --both

Dataset format (per .npz file)
-------------------------------
    obs_spatial : float32[N, WINDOW, WINDOW, C]
    obs_scalars : float32[N, S]            (29 base scalars — no FSM block)
    actions     : int8[N]                  (0-4 moves, 5 = DROPOFF)
    turns       : int16[N]
    ship_ids    : int32[N]
    spawn_feats : float32[T, SPAWN_FEATURE_DIM]   (one row per turn)
    spawn_label : int8[T]                  (1 = player spawned that turn)
    player_id   : int8 scalar              (player whose moves were extracted)
    map_width   : int16 scalar
    map_height  : int16 scalar
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE   = os.path.dirname(os.path.abspath(__file__))
_MY_EXT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)     # rl_v6/ — finds rl_features, rl_config
sys.path.insert(0, _MY_EXT)  # my_extension/ — finds halite_engine (if needed)

from rl_features import (
    extract_spatial_from_replay_state,
    extract_scalars_from_replay_state,
    DIR_TO_ACTION,
    ACTION_STAY,
)
from rl_config import ACTION_DROPOFF_V6, SPAWN_FEATURE_DIM, spawn_global_features


# ---------------------------------------------------------------------------
# Replay loading
# ---------------------------------------------------------------------------

def load_replay(path: str) -> dict:
    """Load a .hlt replay file (zstd or plain JSON)."""
    with open(path, 'rb') as f:
        raw = f.read()

    try:
        import zstd
        data = zstd.decompress(raw)
    except (ImportError, Exception):
        data = raw  # assume plain JSON

    return json.loads(data.decode('utf-8'))


# ---------------------------------------------------------------------------
# State reconstruction
# ---------------------------------------------------------------------------

class ReplayStateReconstructor:
    """
    Incrementally reconstructs the game state for each turn from a replay dict,
    providing the same data structure used by the feature extractors.
    """

    def __init__(self, replay: dict):
        self.replay  = replay
        gc           = replay['GAME_CONSTANTS']
        self.width   = gc['DEFAULT_MAP_WIDTH']
        self.height  = gc['DEFAULT_MAP_HEIGHT']
        self.max_turns = gc['MAX_TURNS']
        self.n_players = replay['number_of_players']

        # Initial halite grid
        self.initial_halite = {}
        for y, row in enumerate(replay['production_map']['grid']):
            for x, cell in enumerate(row):
                self.initial_halite[(x, y)] = cell['energy']

        # Factory positions per player
        self.factories = {}
        for p in replay['players']:
            pid = p['player_id']
            fx  = p['factory_location']['x']
            fy  = p['factory_location']['y']
            self.factories[pid] = (fx, fy)

        self._precompute()

    def _precompute(self):
        """Build per-turn halite maps and structure sets."""
        frames = self.replay['full_frames']
        N      = len(frames)

        # Cumulative halite state
        halite = dict(self.initial_halite)

        # Structures per player (start with factories)
        structures = {pid: [self.factories[pid]] for pid in range(self.n_players)}

        # Player bank at end of previous frame (approximation; use 'energy' field)
        player_energy = {pid: self.replay['players'][pid].get('energy', 5000)
                         for pid in range(self.n_players)}

        self._states = []

        for i, frame in enumerate(frames):
            state = {
                'halite':        dict(halite),
                'entities':      frame.get('entities', {}),
                'structures':    {pid: list(s) for pid, s in structures.items()},
                'factories':     dict(self.factories),
                'player_energy': dict(player_energy),
                'turn':          i,
                'max_turns':     self.max_turns,
                'width':         self.width,
                'height':        self.height,
            }
            self._states.append(state)

            # Apply cell changes for next turn
            for cell in frame.get('cells', []):
                halite[(cell['x'], cell['y'])] = cell['production']

            # Apply construct events → new dropoffs
            for ev in frame.get('events', []):
                if ev['type'] == 'construct':
                    pid = ev['owner_id']
                    pos = (ev['location']['x'], ev['location']['y'])
                    if pos not in structures[pid]:
                        structures[pid].append(pos)

            # Update player energy from frame
            energy_snap = frame.get('energy', {})
            for pid_str, val in energy_snap.items():
                player_energy[int(pid_str)] = val

    def num_frames(self) -> int:
        return len(self._states)

    def get_state(self, turn: int) -> dict:
        return self._states[turn]


# ---------------------------------------------------------------------------
# Action inference from moves data
# ---------------------------------------------------------------------------

def _parse_moves(frame: dict, pid: int):
    """
    Return ({ship_id: action_int}, spawned_bool) for player `pid`.

    'm' moves map to their direction action; 'c' (construct) ships are labelled
    ACTION_DROPOFF (5); ships with no explicit entry are ACTION_STAY (mine).
    `spawned_bool` is True iff a spawn ('g') command was issued this turn.
    """
    pid_str   = str(pid)
    moves_raw = frame.get('moves', {}).get(pid_str, [])

    move_by_ship = {}
    spawned = False
    for m in moves_raw:
        t = m.get('type')
        if t == 'm':
            move_by_ship[m['id']] = DIR_TO_ACTION.get(m.get('direction', 'o'), ACTION_STAY)
        elif t == 'c':
            move_by_ship[m['id']] = ACTION_DROPOFF_V6
        elif t == 'g':
            spawned = True

    return move_by_ship, spawned


# ---------------------------------------------------------------------------
# Dataset extraction
# ---------------------------------------------------------------------------

def _spawn_features_from_state(state: dict, pid: int) -> np.ndarray:
    """Build the compact global spawn feature vector from a replay state."""
    W, H = state['width'], state['height']
    energy = state['player_energy']
    my_bank   = energy.get(pid, 0)
    opp_bank  = max((energy.get(p, 0) for p in energy if p != pid), default=0)
    my_ships  = len(state['entities'].get(str(pid), {}))
    opp_ships = sum(len(e) for p, e in state['entities'].items() if int(p) != pid)
    map_h     = sum(state['halite'].values())
    return spawn_global_features(state['turn'], state['max_turns'],
                                 my_bank, opp_bank, my_ships, opp_ships, map_h, W, H)


def extract_episode(replay: dict, pid: int) -> dict:
    """
    Extract PURE (obs, action) pairs + a per-turn spawn dataset for player `pid`.

    Ship actions are 0-4 (moves) or 5 (DROPOFF/construct).  One spawn sample is
    emitted per turn (global features + 0/1 spawned label).  Returns None if no
    usable ship steps were found.
    """
    rec    = ReplayStateReconstructor(replay)
    frames = replay['full_frames']

    all_sp, all_sc, all_ac, all_turns, all_sids = [], [], [], [], []
    spawn_feats, spawn_label = [], []

    for turn_idx in range(1, rec.num_frames()):
        state = rec.get_state(turn_idx)
        frame = frames[turn_idx]

        move_map, spawned = _parse_moves(frame, pid)
        pid_str    = str(pid)
        entities_t = state['entities'].get(pid_str, {})

        # Per-turn spawn decision sample (the global, once-per-turn choice).
        spawn_feats.append(_spawn_features_from_state(state, pid))
        spawn_label.append(1 if spawned else 0)

        for eid_str, edata in entities_t.items():
            eid    = int(eid_str)
            action = move_map.get(eid, ACTION_STAY)   # construct -> DROPOFF, kept

            spatial = extract_spatial_from_replay_state(state, eid_str, pid)
            scalars = extract_scalars_from_replay_state(state, eid_str, pid)

            all_sp.append(spatial)
            all_sc.append(scalars)
            all_ac.append(action)
            all_turns.append(turn_idx)
            all_sids.append(eid)

    if not all_sp:
        return None

    return {
        'obs_spatial': np.array(all_sp,    dtype=np.float32),
        'obs_scalars': np.array(all_sc,    dtype=np.float32),
        'actions':     np.array(all_ac,    dtype=np.int8),
        'turns':       np.array(all_turns, dtype=np.int16),
        'ship_ids':    np.array(all_sids,  dtype=np.int32),
        'spawn_feats': np.array(spawn_feats, dtype=np.float32),
        'spawn_label': np.array(spawn_label, dtype=np.int8),
        'player_id':   np.int8(pid),
        'map_width':   np.int16(rec.width),
        'map_height':  np.int16(rec.height),
    }


def best_player(replay: dict) -> int:
    """Return the player id with the highest final rank (rank 1 wins)."""
    stats = replay['game_statistics']['player_statistics']
    best  = min(stats, key=lambda s: s['rank'])
    return best['player_id']


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def process_replay(src_path: str, dst_path: str, players, verbose: bool = True):
    """Extract one .npz per requested player slot.  `players` is a list of pids,
    or None to auto-pick the best (winning) player."""
    if verbose:
        print(f"  Loading {src_path} … ", end='', flush=True)
    try:
        replay = load_replay(src_path)
    except Exception as e:
        print(f"SKIP ({e})")
        return 0

    pids = players if players is not None else [best_player(replay)]
    base, ext = os.path.splitext(dst_path)
    ok = 0
    for pid in pids:
        data = extract_episode(replay, pid)
        if data is None:
            continue
        out = dst_path if len(pids) == 1 else f"{base}_p{pid}{ext}"
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        np.savez_compressed(out, **data)
        ok += 1
        if verbose:
            print(f"\n    p{pid}: {len(data['actions'])} ship-steps, "
                  f"{len(data['spawn_label'])} turns → {out}", end='')
    if verbose:
        print()
    return ok


def main():
    parser = argparse.ArgumentParser(description='Extract rl_v6 imitation dataset from .hlt replays')
    parser.add_argument('source', help='path to a .hlt file or directory of .hlt files')
    parser.add_argument('--output', '-o', default='dataset/', help='output .npz file or directory')
    parser.add_argument('--player', type=int, default=0,
                        help='player slot to extract (default 0 = the rl_v5 bot)')
    parser.add_argument('--both', action='store_true',
                        help='extract BOTH players (use for rl_v5 self-play replays)')
    args = parser.parse_args()

    players = [0, 1] if args.both else [args.player]
    src, out = args.source, args.output

    if os.path.isfile(src):
        dst = out if out.endswith('.npz') else os.path.join(out, 'ep_000001.npz')
        process_replay(src, dst, players)
    elif os.path.isdir(src):
        replays = sorted(f for f in os.listdir(src) if f.endswith('.hlt'))
        if not replays:
            print(f"No .hlt files found in {src}")
            return
        print(f"Processing {len(replays)} replay(s) from {src} (players={players})")
        ok = 0
        for i, name in enumerate(replays, 1):
            dst = os.path.join(out, f'ep_{i:06d}.npz')
            ok += process_replay(os.path.join(src, name), dst, players)
        print(f"\nDone: {ok} shard(s) written to {out}")
    else:
        print(f"Source not found: {src}")


if __name__ == '__main__':
    main()
