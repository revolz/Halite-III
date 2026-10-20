#!/usr/bin/env python3
"""Replay real-engine .hlt games through BatchSim and assert identical state every turn.

Usage: python rl_v10/verify_sim.py replay1.hlt [replay2.hlt ...]
"""
import json
import sys

import numpy as np

from sim import BatchSim

DIR_TO_ACT = {'n': 1, 'e': 2, 's': 3, 'w': 4}


def load(path):
    raw = open(path, 'rb').read()
    if raw[:4] == b'\x28\xb5\x2f\xfd':
        import zstd
        raw = zstd.decompress(raw)
    return json.loads(raw)


def check(path):
    r = load(path)
    W, H = r['production_map']['width'], r['production_map']['height']
    sim = BatchSim(1, W, H)
    sim.reset([r['map_generator_seed']])
    grid = np.array([[c['energy'] for c in row] for row in r['production_map']['grid']])
    assert (sim.halite[0] == grid).all(), 'initial map differs'
    for p, pl in enumerate(r['players']):
        assert tuple(sim.fac[0, p]) == (pl['factory_location']['x'], pl['factory_location']['y'])
    frames = r['full_frames'][1:]
    for k, f in enumerate(frames, start=1):
        # state at start of turn k
        for p in (0, 1):
            ents = f['entities'].get(str(p), {})
            got = {int(sim.sid[0, p, y, x]): (x, y, int(sim.cargo[0, p, y, x]))
                   for y, x in zip(*np.nonzero(sim.occ[0, p]))}
            exp = {int(i): (e['x'], e['y'], e['energy']) for i, e in ents.items()}
            if got != exp:
                diff = {i: (got.get(i), exp.get(i)) for i in set(got) | set(exp) if got.get(i) != exp.get(i)}
                raise AssertionError(f'turn {k} player {p} ships differ: {list(diff.items())[:6]}')
            ins = {int(i): e['is_inspired'] for i, e in ents.items()}
            for y, x in zip(*np.nonzero(sim.occ[0, p])):
                assert bool(sim.insp[0, p, y, x]) == ins[int(sim.sid[0, p, y, x])], f'turn {k} inspiration'
        act = np.zeros((1, 2, H, W), np.int64)
        spawn = np.zeros((1, 2), bool)
        pos = {}
        for p in (0, 1):
            for y, x in zip(*np.nonzero(sim.occ[0, p])):
                pos[int(sim.sid[0, p, y, x])] = (p, y, x)
        for p_str, mv in f['moves'].items():
            p = int(p_str)
            for m in mv:
                if m['type'] == 'g':
                    spawn[0, p] = True
                elif m['type'] == 'm' and m['id'] in pos:
                    pp, y, x = pos[m['id']]
                    if pp == p:
                        act[0, p, y, x] = DIR_TO_ACT[m['direction']]
                elif m['type'] == 'c' and m['id'] in pos:
                    pp, y, x = pos[m['id']]
                    if pp == p:
                        act[0, p, y, x] = 5
        sim.step(act, spawn)
        exp_bank = [f['energy'][str(p)] for p in (0, 1)]
        assert list(sim.bank[0]) == exp_bank, f'turn {k} bank {list(sim.bank[0])} vs {exp_bank}'
        for c in f['cells']:
            grid[c['y'], c['x']] = c['production']
        bad = np.argwhere(sim.halite[0] != grid)
        assert len(bad) == 0, f'turn {k} halite differs at {bad[:5].tolist()}'
    final = [ps['final_production'] for ps in r['game_statistics']['player_statistics']]
    assert list(sim.bank[0]) == final
    print(f'OK {path}: {len(frames)} turns, final {final}')


if __name__ == '__main__':
    for p in sys.argv[1:]:
        check(p)
