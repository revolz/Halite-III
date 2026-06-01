"""
Imitation learning dataset extractor for Halite III.

Reads .hlt replay files and converts the best player's trajectories into
(observation, action) pairs suitable for supervised pre-training.

Usage
-----
    # Extract from a single replay
    python rl_collect.py replay.hlt --output dataset/ep_001.npz

    # Batch extract all replays in a directory
    python rl_collect.py replays/ --output dataset/

Dataset format (per .npz file)
-------------------------------
    obs_spatial : float32[N, WINDOW, WINDOW, C]
    obs_scalars : float32[N, S]
    actions     : int8[N]               (0-4 per ship action)
    turns       : int16[N]
    ship_ids    : int32[N]
    player_id   : int8 scalar           (player whose moves were extracted)
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
sys.path.insert(0, _HERE)     # rl_v1/ — finds rl_features
sys.path.insert(0, _MY_EXT)  # my_extension/ — finds halite_engine (if needed)

from rl_features import (
    extract_spatial_from_replay_state,
    extract_scalars_from_replay_state,
    DIR_TO_ACTION,
    ACTION_STAY,
)


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
    Return {ship_id: action_int} for player `pid` from a full_frame dict.

    Ships that do not have an explicit 'm' move entry are assigned ACTION_STAY
    (mine / stay still).  Construct ('c') and spawn ('g') moves are excluded
    (ships converted to dropoffs have no ship action to learn from).
    """
    pid_str   = str(pid)
    moves_raw = frame.get('moves', {}).get(pid_str, [])

    move_by_ship = {}
    for m in moves_raw:
        if m.get('type') == 'm':
            sid = m['id']
            d   = m.get('direction', 'o')
            move_by_ship[sid] = DIR_TO_ACTION.get(d, ACTION_STAY)

    return move_by_ship


# ---------------------------------------------------------------------------
# Dataset extraction
# ---------------------------------------------------------------------------

def extract_episode(replay: dict, pid: int) -> dict:
    """
    Extract (obs, action) pairs for player `pid` from a single replay.

    Returns a dict with arrays:
        obs_spatial, obs_scalars, actions, turns, ship_ids
    or None if no usable steps were found.
    """
    rec    = ReplayStateReconstructor(replay)
    frames = replay['full_frames']

    all_sp, all_sc, all_ac, all_turns, all_sids = [], [], [], [], []

    for turn_idx in range(1, rec.num_frames()):
        state = rec.get_state(turn_idx)
        frame = frames[turn_idx]

        # Parse which ships moved and how
        move_map   = _parse_moves(frame, pid)
        pid_str    = str(pid)
        entities_t = state['entities'].get(pid_str, {})

        for eid_str, edata in entities_t.items():
            eid    = int(eid_str)
            action = move_map.get(eid, ACTION_STAY)

            # Skip ships that are being converted (construct command)
            construct_ids = {m['id'] for m in frame.get('moves', {}).get(pid_str, [])
                             if m.get('type') == 'c'}
            if eid in construct_ids:
                continue

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

def process_replay(src_path: str, dst_path: str, verbose: bool = True):
    if verbose:
        print(f"  Loading {src_path} … ", end='', flush=True)
    try:
        replay = load_replay(src_path)
    except Exception as e:
        print(f"SKIP ({e})")
        return False

    pid  = best_player(replay)
    data = extract_episode(replay, pid)

    if data is None:
        print("SKIP (no usable steps)")
        return False

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    np.savez_compressed(dst_path, **data)

    if verbose:
        n = len(data['actions'])
        print(f"{n} steps from player {pid} → {dst_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Extract imitation learning dataset from .hlt replays')
    parser.add_argument('source', help='path to a .hlt file or directory of .hlt files')
    parser.add_argument('--output', '-o', default='dataset/', help='output .npz file or directory')
    args = parser.parse_args()

    src    = args.source
    out    = args.output

    if os.path.isfile(src):
        dst = out if out.endswith('.npz') else os.path.join(out, 'ep_001.npz')
        process_replay(src, dst)
    elif os.path.isdir(src):
        replays = sorted(f for f in os.listdir(src) if f.endswith('.hlt'))
        if not replays:
            print(f"No .hlt files found in {src}")
            return
        print(f"Processing {len(replays)} replay(s) from {src}")
        ok = 0
        for i, name in enumerate(replays, 1):
            ep_name = f'ep_{i:06d}.npz'
            dst     = os.path.join(out, ep_name)
            if process_replay(os.path.join(src, name), dst):
                ok += 1
        print(f"\nDone: {ok}/{len(replays)} replays extracted to {out}")
    else:
        print(f"Source not found: {src}")


if __name__ == '__main__':
    main()
