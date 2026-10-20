"""
rl_v10 / obs.py -- observation encoding shared by the trainer (BatchSim state)
and the live bot (parsed engine state), so both see byte-identical inputs.

Every seat is encoded in a canonical frame: if the seat's shipyard is on the
right half of the map, the board is mirrored left-right (and E/W actions are
swapped), so the network always plays "from the left".  Only raw game facts
are encoded; there are no strategy features.
"""
import numpy as np

C = 20
N_ACT = 6
MAX_TURNS = 400


def _torus_dist0(H, W):
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    return (np.minimum(y, H - y) + np.minimum(x, W - x)).astype(np.float32)


_D0 = {}
DIAMOND4 = [(dy, dx) for dy in range(-4, 5) for dx in range(-4, 5)
            if abs(dy) + abs(dx) <= 4 and (dy, dx) != (0, 0)]


def _near(occ):
    out = np.zeros(occ.shape, np.float32)
    o = occ.astype(np.float32)
    for dy, dx in DIAMOND4:
        out += np.roll(o, (dy, dx), axis=(-2, -1))
    return out


def _dist_maps(owner, p):
    """(B,H,W) distance to nearest structure of player p."""
    B, H, W = owner.shape
    if (H, W) not in _D0:
        _D0[(H, W)] = _torus_dist0(H, W)
    d0 = _D0[(H, W)]
    out = np.full((B, H, W), 64.0, np.float32)
    for b in range(B):
        ys, xs = np.nonzero(owner[b] == p)
        for y, x in zip(ys, xs):
            np.minimum(out[b], np.roll(d0, (y, x), axis=(0, 1)), out=out[b])
    return out


def mirror_flags(fac, W):
    """fac: (B,2,2) (x,y).  True where that seat should be mirrored."""
    return fac[:, :, 0] > (W - 1) / 2.0


def encode(halite, owner, occ, cargo, bank, turn, initial_halite, fac, mirror, seats=(0, 1)):
    """
    halite (B,H,W) int, owner (B,H,W) int8 (-1 none), occ/cargo (B,2,H,W),
    bank (B,2), turn int (1-based), initial_halite (B,), fac (B,2,2) (x,y), mirror (B,2) bool.
    Returns obs (B,S,C,H,W) float32, ship_mask (B,S,6,H,W) bool, spawn_ok (B,S) bool
    for the requested seats, all in each seat's canonical frame.
    """
    B, H, W = halite.shape
    hal = halite.astype(np.float32)
    move_cost = (halite // 10)
    near = [_near(occ[:, 0]), _near(occ[:, 1])]
    dist = [_dist_maps(owner, 0), _dist_maps(owner, 1)]
    nships = occ.sum((2, 3))
    turns_left = (MAX_TURNS - turn + 1) / MAX_TURNS
    remain = hal.sum((1, 2)) / np.maximum(initial_halite, 1)
    obs_l, mask_l, spawn_l = [], [], []
    for p in seats:
        q = 1 - p
        o = np.empty((B, C, H, W), np.float32)
        o[:, 0] = hal / 1000.0
        o[:, 1] = occ[:, p]
        o[:, 2] = cargo[:, p] / 1000.0
        o[:, 3] = occ[:, q]
        o[:, 4] = cargo[:, q] / 1000.0
        o[:, 5] = owner == p
        o[:, 6] = owner == q
        o[:, 7] = 0.0
        o[np.arange(B), 7, fac[:, p, 1], fac[:, p, 0]] = 1.0
        o[:, 8] = dist[p] / 16.0
        o[:, 9] = dist[q] / 16.0
        o[:, 10] = near[q] / 4.0
        o[:, 11] = near[p] / 4.0
        cant_move = occ[:, p] & (cargo[:, p] < move_cost)
        o[:, 12] = cant_move
        o[:, 13] = move_cost / 100.0
        o[:, 14] = turns_left
        o[:, 15] = (np.minimum(bank[:, p], 30000) / 10000.0)[:, None, None]
        o[:, 16] = (np.minimum(bank[:, q], 30000) / 10000.0)[:, None, None]
        o[:, 17] = (nships[:, p] / 50.0)[:, None, None]
        o[:, 18] = (nships[:, q] / 50.0)[:, None, None]
        o[:, 19] = remain[:, None, None]
        m = np.zeros((B, N_ACT, H, W), bool)
        mine = occ[:, p]
        m[:, 0] = mine
        for a in (1, 2, 3, 4):
            m[:, a] = mine & ~cant_move
        build_cost = np.maximum(0, 4000 - halite - cargo[:, p])
        m[:, 5] = mine & (owner < 0) & (build_cost <= bank[:, p][:, None, None])
        # spawning onto our own ship destroys both ships for nothing: masked as dominated
        own_on_yard = occ[np.arange(B), p, fac[:, p, 1], fac[:, p, 0]]
        sp = (bank[:, p] >= 1000) & ~own_on_yard
        fl = mirror[:, p]
        if fl.any():
            o[fl] = o[fl][..., ::-1]
            mm = m[fl][..., ::-1].copy()
            mm[:, [2, 4]] = mm[:, [4, 2]]
            m[fl] = mm
        obs_l.append(o)
        mask_l.append(m)
        spawn_l.append(sp)
    return np.stack(obs_l, 1), np.stack(mask_l, 1), np.stack(spawn_l, 1)


def factory_cell(fac_xy, W, mirrored):
    x, y = fac_xy
    return (int(y), int(W - 1 - x if mirrored else x))


def uncanon_actions(act, mirrored):
    """act (H,W) int in canonical frame -> engine frame."""
    if not mirrored:
        return act
    a = act[..., ::-1].copy()
    e, w = a == 2, a == 4
    a[e], a[w] = 4, 2
    return a
