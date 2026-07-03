#!/usr/bin/env python3
"""
rl_v9 / bc_train.py  --  behavioral cloning of V71 for BOTH heads:
the per-ship move/dropoff policy AND the per-turn spawn policy.

Class-weighted cross-entropy (STAY dominates ship data; NO dominates spawn
data).  Reports held-out match rates.  Saves a single bundle checkpoint
(bc.pt) containing the best ship policy and the best spawn policy.

Usage:
    python rl_v9/bc_train.py --epochs 30 --device cuda
"""

import argparse
import copy
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config                                  # noqa: E402
from config import (FEATURE_NAMES, SPAWN_FEATURE_NAMES, ACTION_NAMES,
                    N_ACTIONS, N_SPAWN_ACTIONS)  # noqa: E402
from net import ShipPolicy, SpawnPolicy, save_bundle   # noqa: E402


def class_weights(y, n_classes, cap=10.0):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    inv = np.zeros_like(counts)
    nonzero = counts > 0
    inv[nonzero] = counts[nonzero].sum() / (counts[nonzero] * nonzero.sum())
    inv = np.clip(inv, 0.0, cap)
    return torch.tensor(inv, dtype=torch.float32)


def split(n, val_frac, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    return perm[:n_val], perm[n_val:]


def train_ship(args, device):
    df = pd.read_csv(config.FEATURES_CSV)
    patches = np.load(config.PATCHES_NPY)          # [N, 9, 9, 6] f16 HWC
    gmaps = np.load(config.GLOBALS_NPY)            # [N, 4, 8, 8] f16 CHW
    assert len(df) == len(patches) == len(gmaps)
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = df['action'].to_numpy(dtype=np.int64)
    P = np.transpose(patches, (0, 3, 1, 2)).astype(np.float32)
    G = gmaps.astype(np.float32)

    n = len(y)
    print(f"[ship] {n} rows, {X.shape[1]} scalars, patch {P.shape[1:]}, global {G.shape[1:]}")
    val_idx, tr_idx = split(n, args.val_frac, args.seed)

    Xt = torch.tensor(X[tr_idx], device=device)
    Pt = torch.tensor(P[tr_idx], device=device)
    Gt = torch.tensor(G[tr_idx], device=device)
    yt = torch.tensor(y[tr_idx], device=device)
    Xv = torch.tensor(X[val_idx], device=device)
    Pv = torch.tensor(P[val_idx], device=device)
    Gv = torch.tensor(G[val_idx], device=device)
    yv = torch.tensor(y[val_idx], device=device)

    w = class_weights(y[tr_idx], N_ACTIONS, cap=args.weight_cap).to(device)
    print("[ship] class weights:",
          {ACTION_NAMES[i]: round(float(w[i]), 2) for i in range(N_ACTIONS)})

    model = ShipPolicy().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    n_tr = len(tr_idx)
    best_match, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_perm = torch.randperm(n_tr, device=device)
        total_loss = 0.0
        for i in range(0, n_tr, args.batch_size):
            b = ep_perm[i:i + args.batch_size]
            logits = model(Xt[b], Pt[b], Gt[b])
            loss = loss_fn(logits, yt[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item()) * len(b)

        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(val_idx), 4096):
                logits = model(Xv[i:i + 4096], Pv[i:i + 4096], Gv[i:i + 4096])
                preds.append(torch.argmax(logits, dim=1))
            pred = torch.cat(preds)
            match = float((pred == yv).float().mean().item())
        tag = ""
        if match > best_match:
            best_match, best_state = match, copy.deepcopy(model.state_dict())
            tag = "  *best*"
        print(f"[ship] epoch {epoch:3d}  loss {total_loss/n_tr:.4f}  "
              f"val_match {match*100:5.2f}%{tag}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(val_idx), 4096):
            logits = model(Xv[i:i + 4096], Pv[i:i + 4096], Gv[i:i + 4096])
            preds.append(torch.argmax(logits, dim=1))
        pred = torch.cat(preds).cpu().numpy()
    yv_np = yv.cpu().numpy()
    print(f"\n[ship] best val match-rate: {best_match*100:.2f}%")
    print("[ship] per-action recall (val):")
    for a in range(N_ACTIONS):
        sel = yv_np == a
        if sel.sum() == 0:
            continue
        rec = (pred[sel] == a).mean()
        print(f"  {ACTION_NAMES[a]:8s}: {rec*100:5.1f}%  (n={int(sel.sum())})")
    return model, best_match


def train_spawn(args, device):
    df = pd.read_csv(config.SPAWN_CSV)
    gmaps = np.load(config.SPAWN_GLOBALS_NPY)      # [M, 4, 8, 8]
    assert len(df) == len(gmaps)
    X = df[SPAWN_FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = df['action'].to_numpy(dtype=np.int64)
    G = gmaps.astype(np.float32)

    n = len(y)
    print(f"\n[spawn] {n} rows ({int(y.sum())} spawns)")
    val_idx, tr_idx = split(n, args.val_frac, args.seed + 1)

    Xt = torch.tensor(X[tr_idx], device=device)
    Gt = torch.tensor(G[tr_idx], device=device)
    yt = torch.tensor(y[tr_idx], device=device)
    Xv = torch.tensor(X[val_idx], device=device)
    Gv = torch.tensor(G[val_idx], device=device)
    yv = torch.tensor(y[val_idx], device=device)

    w = class_weights(y[tr_idx], N_SPAWN_ACTIONS, cap=args.weight_cap).to(device)
    print(f"[spawn] class weights: NO={float(w[0]):.2f} YES={float(w[1]):.2f}")

    model = SpawnPolicy().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    n_tr = len(tr_idx)
    best_match, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_perm = torch.randperm(n_tr, device=device)
        total_loss = 0.0
        for i in range(0, n_tr, args.batch_size):
            b = ep_perm[i:i + args.batch_size]
            logits = model(Xt[b], Gt[b])
            loss = loss_fn(logits, yt[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item()) * len(b)

        model.eval()
        with torch.no_grad():
            pred = torch.argmax(model(Xv, Gv), dim=1)
            match = float((pred == yv).float().mean().item())
            # balanced accuracy matters more (NO dominates)
            accs = []
            for c in range(N_SPAWN_ACTIONS):
                sel = yv == c
                if sel.sum() > 0:
                    accs.append(float((pred[sel] == c).float().mean().item()))
            bal = sum(accs) / len(accs)
        tag = ""
        if bal > best_match:
            best_match, best_state = bal, copy.deepcopy(model.state_dict())
            tag = "  *best*"
        print(f"[spawn] epoch {epoch:3d}  loss {total_loss/n_tr:.4f}  "
              f"val_match {match*100:5.2f}%  balanced {bal*100:5.2f}%{tag}")

    model.load_state_dict(best_state)
    print(f"[spawn] best balanced accuracy: {best_match*100:.2f}%")
    return model, best_match


def main():
    ap = argparse.ArgumentParser(description='Behavioral cloning for rl_v9 (ship + spawn).')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--spawn-epochs', type=int, default=None,
                    help='default: same as --epochs')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--weight-cap', type=float, default=10.0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(config.CHECKPOINT_DIR, 'bc.pt'))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    ship_model, ship_match = train_ship(args, device)

    spawn_args = copy.copy(args)
    if args.spawn_epochs is not None:
        spawn_args.epochs = args.spawn_epochs
    spawn_model, spawn_match = train_spawn(spawn_args, device)

    save_bundle(args.out, ship_model, spawn_model,
                extra={'bc_ship_match': ship_match,
                       'bc_spawn_balanced': spawn_match})
    print(f"\nsaved bundle -> {args.out}")
    print(f"ship val match {ship_match*100:.2f}%   "
          f"spawn balanced {spawn_match*100:.2f}%")


if __name__ == '__main__':
    main()
