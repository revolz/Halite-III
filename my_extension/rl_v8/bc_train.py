#!/usr/bin/env python3
"""
rl_v8 / bc_train.py  --  behavioral cloning: train the network to imitate V71
(the 2019 hand-coded bot).

Loads features.csv + patches.npy, splits train/val, and minimises a
class-weighted cross-entropy between the network's action and V71's action.
STAY dominates the data, so per-class inverse-frequency weights (capped) keep
the rarer moves from being ignored.

Reports overall and per-action match-rate vs V71 on a held-out validation
split. V71 is a per-ship FSM with hidden turn-to-turn state (see rl_v8's
README), so unlike rl_v7's original ~90% target vs a stateless imitation
target, match-rate here may plateau lower -- report the number plainly either
way; that's the hypothesis this bot is testing.

Usage:
    python rl_v8/bc_train.py --epochs 30
    python rl_v8/bc_train.py --epochs 30 --device cuda
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config                                  # noqa: E402
from config import FEATURE_NAMES, ACTION_NAMES, N_ACTIONS  # noqa: E402
from net import ActorCritic                    # noqa: E402


def load_dataset():
    df = pd.read_csv(config.FEATURES_CSV)
    patches = np.load(config.PATCHES_NPY)            # [N, S, S, C] float16
    assert len(df) == len(patches), \
        f"CSV rows {len(df)} != patches {len(patches)} -- regenerate dataset"
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = df['action'].to_numpy(dtype=np.int64)
    # patches -> [N, C, S, S] float32
    P = np.transpose(patches, (0, 3, 1, 2)).astype(np.float32)
    return X, P, y


def class_weights(y, cap=10.0):
    counts = np.bincount(y, minlength=N_ACTIONS).astype(np.float64)
    inv = np.zeros_like(counts)
    nonzero = counts > 0
    inv[nonzero] = counts[nonzero].sum() / (counts[nonzero] * nonzero.sum())
    inv = np.clip(inv, 0.0, cap)
    return torch.tensor(inv, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser(description='Behavioral cloning for rl_v8.')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--weight-cap', type=float, default=10.0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(config.CHECKPOINT_DIR, 'bc.pt'))
    ap.add_argument('--resume', default=None,
                    help='path to checkpoint to warm-start from (e.g. bc.pt from a previous run)')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print(f"Loading dataset from {config.DATASET_DIR} ...")
    X, P, y = load_dataset()
    n = len(y)
    print(f"  {n} rows, {X.shape[1]} scalar features, patch {P.shape[1:]}")

    # train/val split
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = int(n * args.val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    Xt = torch.tensor(X[tr_idx], device=device)
    Pt = torch.tensor(P[tr_idx], device=device)
    yt = torch.tensor(y[tr_idx], device=device)
    Xv = torch.tensor(X[val_idx], device=device)
    Pv = torch.tensor(P[val_idx], device=device)
    yv = torch.tensor(y[val_idx], device=device)

    w = class_weights(y[tr_idx], cap=args.weight_cap).to(device)
    print("  class weights:", {ACTION_NAMES[i]: round(float(w[i]), 2) for i in range(N_ACTIONS)})

    if args.resume:
        print(f"  resuming from {args.resume}")
        model = ActorCritic.load(args.resume, device=device)
    else:
        model = ActorCritic().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    n_tr = len(tr_idx)
    best_match = -1.0
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_perm = torch.randperm(n_tr, device=device)
        total_loss = 0.0
        for i in range(0, n_tr, args.batch_size):
            b = ep_perm[i:i + args.batch_size]
            logits, _ = model(Xt[b], Pt[b])
            loss = loss_fn(logits, yt[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item()) * len(b)

        # validation
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(val_idx), 4096):
                logits, _ = model(Xv[i:i + 4096], Pv[i:i + 4096])
                preds.append(torch.argmax(logits, dim=1))
            pred = torch.cat(preds)
            match = float((pred == yv).float().mean().item())

        if match > best_match:
            best_match = match
            model.save(args.out)
            tag = "  *saved*"
        else:
            tag = ""
        print(f"epoch {epoch:3d}  loss {total_loss/n_tr:.4f}  val_match {match*100:5.2f}%{tag}")

    # per-action match report on val
    model = ActorCritic.load(args.out, device=device)
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(val_idx), 4096):
            logits, _ = model(Xv[i:i + 4096], Pv[i:i + 4096])
            preds.append(torch.argmax(logits, dim=1))
        pred = torch.cat(preds).cpu().numpy()
    yv_np = yv.cpu().numpy()
    print(f"\nbest val match-rate: {best_match*100:.2f}%  ->  {args.out}")
    print("per-action recall (val):")
    for a in range(N_ACTIONS):
        sel = yv_np == a
        if sel.sum() == 0:
            continue
        rec = (pred[sel] == a).mean()
        print(f"  {ACTION_NAMES[a]:8s}: {rec*100:5.1f}%  (n={int(sel.sum())})")


if __name__ == '__main__':
    main()
