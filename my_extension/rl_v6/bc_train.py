#!/usr/bin/env python3
"""
bc_train.py — behavioral cloning of rl_v5 for rl_v6.

Trains, from the .npz shards produced by rl_collect.py:
  * the per-ship policy  — ActorCritic actor head, cross-entropy on the resolved
    move (0-4) / DROPOFF (5).  Class-weighted because STAY dominates Halite move
    logs (without weighting the policy collapses to all-STAY).
  * the spawn head       — SpawnHead, BCE with pos_weight on the per-turn 0/1
    spawn label (spawning is rare, ~3 %).

Only the actor (conv + scalar MLP + trunk + actor head) is trained here; the
critic is left for PPO to fit.  Weights save in ActorCritic.load() format so the
bot and the PPO warm-start can load them directly.

This script GLOBS the whole dataset dir, so DAgger can simply drop new shards in
and re-run it (that is the DAgger outer loop).

Usage
-----
    python bc_train.py --data dataset/ --epochs 15 \
        --out checkpoints/model_weights.pt --spawn-out checkpoints/spawn_weights.pt
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from rl_model    import ActorCritic
from spawn_model import SpawnHead
from rl_config   import N_SCALARS_V6, N_SHIP_ACTIONS_V6, SPAWN_FEATURE_DIM


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(data_dir: str, max_samples: int = 0):
    """Concatenate all shards into RAM tensors.

    Returns (spatial, scalars, actions, spawn_feats, spawn_label).  With
    max_samples > 0 the ship-step arrays are uniformly subsampled to that size
    (spawn rows are small — always kept whole).
    """
    shards = sorted(glob.glob(os.path.join(data_dir, '**', '*.npz'), recursive=True))
    if not shards:
        raise SystemExit(f"No .npz shards in {data_dir}")
    sp, sc, ac, sf, sl = [], [], [], [], []
    for f in shards:
        d = np.load(f)
        sp.append(d['obs_spatial']); sc.append(d['obs_scalars']); ac.append(d['actions'])
        if 'spawn_feats' in d:
            sf.append(d['spawn_feats']); sl.append(d['spawn_label'])
    sp = np.concatenate(sp); sc = np.concatenate(sc); ac = np.concatenate(ac)
    sf = np.concatenate(sf) if sf else np.zeros((0, SPAWN_FEATURE_DIM), np.float32)
    sl = np.concatenate(sl) if sl else np.zeros((0,), np.int8)
    print(f"Loaded {len(shards)} shards: {len(ac):,} ship-steps, {len(sl):,} spawn-turns")

    if max_samples and len(ac) > max_samples:
        idx = np.random.choice(len(ac), max_samples, replace=False)
        sp, sc, ac = sp[idx], sc[idx], ac[idx]
        print(f"  subsampled ship-steps to {max_samples:,}")
    return sp, sc, ac, sf, sl


def class_weights(actions: np.ndarray, n_classes: int, cap: float = 10.0):
    """Capped inverse-frequency class weights.

    Raw inverse frequency `total / (n_classes * count)` is ~1 for a balanced
    class, <1 for common classes (STAY) and >1 for rare ones; we cap the top so a
    single ultra-rare class (DROPOFF) can't dominate.  No mean-renormalisation —
    that previously let one rare class crush every other weight toward 0.
    """
    counts = np.bincount(actions, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (n_classes * counts)
    w = np.minimum(w, cap)
    return torch.tensor(w, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_policy(model, sp, sc, ac, device, epochs, batch, lr, val_frac=0.1):
    n = len(ac)
    perm = np.random.permutation(n)
    n_val = int(n * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    def to_ds(idx):
        return TensorDataset(torch.from_numpy(sp[idx]), torch.from_numpy(sc[idx]),
                             torch.from_numpy(ac[idx].astype(np.int64)))
    tr_dl = DataLoader(to_ds(tr_idx), batch_size=batch, shuffle=True)
    va_dl = DataLoader(to_ds(val_idx), batch_size=batch)

    w = class_weights(ac[tr_idx], N_SHIP_ACTIONS_V6).to(device)
    print(f"  class weights: {[round(x,2) for x in w.tolist()]}")
    crit = nn.CrossEntropyLoss(weight=w)
    opt  = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for bsp, bsc, bac in tr_dl:
            bsp, bsc, bac = bsp.to(device), bsc.to(device), bac.to(device)
            logits, _ = model(bsp, bsc)
            loss = crit(logits, bac)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(bac)
        tr_loss = tot / len(tr_idx)

        model.eval()
        correct = np.zeros(N_SHIP_ACTIONS_V6); seen = np.zeros(N_SHIP_ACTIONS_V6)
        match = 0
        with torch.no_grad():
            for bsp, bsc, bac in va_dl:
                bsp, bsc = bsp.to(device), bsc.to(device)
                pred = model(bsp, bsc)[0].argmax(1).cpu().numpy()
                lab  = bac.numpy()
                match += (pred == lab).sum()
                for c in range(N_SHIP_ACTIONS_V6):
                    m = lab == c
                    seen[c] += m.sum(); correct[c] += (pred[m] == c).sum()
        acc = match / max(1, len(val_idx))
        per = [f"{int(correct[c])}/{int(seen[c])}" for c in range(N_SHIP_ACTIONS_V6)]
        print(f"  ep{ep:02d} loss={tr_loss:.4f}  val_match={acc:.3f}  per-class(correct/seen)={per}")
    return acc


def train_spawn(head, sf, sl, device, epochs, batch, lr):
    if len(sl) == 0:
        print("  (no spawn data — skipping spawn head)")
        return 0.0
    pos = max(1, int(sl.sum())); neg = max(1, len(sl) - pos)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    print(f"  spawn pos={pos} neg={neg} pos_weight={neg/pos:.2f}")
    ds = TensorDataset(torch.from_numpy(sf), torch.from_numpy(sl.astype(np.float32)))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt  = torch.optim.Adam(head.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        head.train(); tot = 0.0
        for bf, bl in dl:
            bf, bl = bf.to(device), bl.to(device)
            loss = crit(head(bf), bl)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(bl)
        # train-set accuracy at 0.5
        head.eval()
        with torch.no_grad():
            pred = (torch.sigmoid(head(torch.from_numpy(sf).to(device))) >= 0.5).cpu().numpy()
        acc = (pred == sl.astype(bool)).mean()
        print(f"  spawn ep{ep:02d} loss={tot/len(sl):.4f}  acc={acc:.3f}")
    return float(acc)


def run_bc(data, out, spawn_out, epochs=15, spawn_epochs=30, batch=512, lr=3e-4,
           max_samples=0, device=None, resume=None):
    """Train the BC policy + spawn head; return metrics dict.  Importable entry
    point shared by main() and run_pipeline.py."""
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")
    sp, sc, ac, sf, sl = load_dataset(data, max_samples)

    model = ActorCritic(n_scalars=N_SCALARS_V6, n_actions=N_SHIP_ACTIONS_V6).to(device)
    if resume and os.path.exists(resume):
        model.load_state_dict(torch.load(resume, map_location=device))
        print(f"Resumed policy from {resume}")
    print("Training policy (behavioral cloning)…")
    val_match = train_policy(model, sp, sc, ac, device, epochs, batch, lr)
    model.save(out)
    print(f"Saved policy -> {out}")

    head = SpawnHead().to(device)
    print("Training spawn head…")
    spawn_acc = train_spawn(head, sf, sl, device, spawn_epochs, batch, lr)
    head.save(spawn_out)
    print(f"Saved spawn head -> {spawn_out}")
    return {'val_match': float(val_match), 'spawn_acc': float(spawn_acc),
            'n_steps': int(len(ac))}


def main():
    ap = argparse.ArgumentParser(description='Behavioral cloning of rl_v5 for rl_v6')
    ap.add_argument('--data', default=os.path.join(_HERE, 'dataset'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'checkpoints', 'model_weights.pt'))
    ap.add_argument('--spawn-out', dest='spawn_out',
                    default=os.path.join(_HERE, 'checkpoints', 'spawn_weights.pt'))
    ap.add_argument('--resume', default=None, help='warm-start policy weights')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--spawn-epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--max-samples', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    run_bc(args.data, args.out, args.spawn_out, args.epochs, args.spawn_epochs,
           args.batch, args.lr, args.max_samples, args.device, args.resume)


if __name__ == '__main__':
    main()
