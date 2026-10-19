#!/usr/bin/env python3
"""
rl_v11 / widen.py -- function-preserving widening of rl_v10 (Net2Net-style).

Builds Net(new_ch, blocks) and copies rl_v10's learned weights into it so the
wide network computes exactly what rl_v10 computes.  Weights into the new
channels are small random values; every weight from a new channel into an old
channel or a head is zero, so the new channels don't change the output at
first, yet their outgoing weights receive gradients and start learning.

Usage: python rl_v11/widen.py [--src rl_v10/checkpoints/best.pt] [--ch 64]
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'rl_v10'))
from model import Net, load, save  # noqa: E402


def seg_map(old, new, n_seg):
    """Indices in the wide tensor of the old entries for an input made of n_seg concatenated channel groups."""
    return np.concatenate([np.arange(old) + k * new for k in range(n_seg)])


def widen(src, ch, scale=0.1, seed=0):
    torch.manual_seed(seed)
    old, _ = load(src)
    oc = old.cfg['ch']
    wide = Net(ch, old.cfg['blocks'])
    osd, wsd = old.state_dict(), wide.state_dict()
    # (out_segments, in_segments) for each parameter; None = dimension not widened
    for name, w in wsd.items():
        o = osd[name]
        if name.startswith('spawn.0.'):
            out_map, in_map = None, seg_map(oc, ch, 3)
        elif '.g.' in name:
            out_map, in_map = seg_map(oc, ch, 1), seg_map(oc, ch, 2)
        elif name.startswith('inp.'):
            out_map, in_map = seg_map(oc, ch, 1), None
        elif name.startswith(('pi.', 'v.2.')):
            out_map, in_map = None, seg_map(oc, ch, 1)
        elif name.startswith('spawn.2.'):
            out_map, in_map = None, None
        else:                                           # block convs, v.0
            out_map, in_map = seg_map(oc, ch, 1), seg_map(oc, ch, 1)
        new = w.clone() * scale
        if new.dim() == 1:                              # bias
            if out_map is None:
                new = o.clone()
            else:
                new.zero_()
                new[out_map] = o
        else:
            rows = torch.arange(new.shape[0]) if out_map is None else torch.as_tensor(out_map)
            new[rows] = 0.0                             # old outputs: only old->old weights below
            cols = torch.arange(new.shape[1]) if in_map is None else torch.as_tensor(in_map)
            new[rows[:, None], cols[None, :]] = o
        wsd[name] = new
    wide.load_state_dict(wsd)
    return old, wide


def check(old, wide, n=64):
    from obs import C
    torch.manual_seed(1)
    x = torch.rand(n, C, 32, 32)
    fy = torch.full((n,), 16)
    fx = torch.full((n,), 8)
    with torch.no_grad():
        a, b = old(x, fy, fx), wide(x, fy, fx)
    return max(float((p - q).abs().max()) for p, q in zip(a, b))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(os.path.dirname(HERE), 'rl_v10', 'checkpoints', 'best.pt'))
    ap.add_argument('--ch', type=int, default=64)
    ap.add_argument('--out', default=os.path.join(HERE, 'checkpoints', 'init.pt'))
    a = ap.parse_args()
    old, wide = widen(a.src, a.ch)
    d = check(old, wide)
    print(f'widened {old.cfg} -> {wide.cfg}: {sum(p.numel() for p in wide.parameters()):,} params, '
          f'max output difference vs source {d:.2e}')
    assert d < 1e-4
    save(wide, a.out, {'iter': 0, 'widened_from': os.path.relpath(a.src, os.path.dirname(HERE))})
    print('saved', a.out)
