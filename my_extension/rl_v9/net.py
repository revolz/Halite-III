#!/usr/bin/env python3
"""
rl_v9 / net.py  --  policy and value networks.

Four small modules (policy and value are deliberately SEPARATE networks --
in rl_v8 they shared a trunk, so the untrained critic's early PPO gradients
flowed straight into the BC-trained policy and wrecked it):

  ShipPolicy : scalars(46) + 9x9x6 local patch + 8x8x4 global map -> 6 logits
  ShipValue  : same inputs -> scalar V(s)
  SpawnPolicy: spawn scalars(18) + factory-centred 8x8x4 global map -> 2 logits
  SpawnValue : same inputs -> scalar V(s)

Checkpoints are a single file bundling whichever modules exist:
    {'ship': sd, 'spawn': sd, 'ship_value': sd|None, 'spawn_value': sd|None,
     'meta': {...dims...}}
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    N_SCALARS, N_ACTIONS, PATCH_SIZE, PATCH_CHANNELS,
    GLOBAL_SIZE, GLOBAL_CHANNELS, N_SPAWN_SCALARS, N_SPAWN_ACTIONS,
)

NEG_INF = -1e9


def _init_weights(module, actor_head=None):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    if actor_head is not None:
        nn.init.orthogonal_(actor_head.weight, gain=0.01)
        nn.init.zeros_(actor_head.bias)


class _ShipTrunk(nn.Module):
    """Shared architecture for ShipPolicy / ShipValue (separate instances)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(PATCH_CHANNELS, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.conv_fc = nn.Sequential(
            nn.Linear(32 * PATCH_SIZE * PATCH_SIZE, 64), nn.ReLU())
        self.gconv = nn.Sequential(
            nn.Conv2d(GLOBAL_CHANNELS, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.gconv_fc = nn.Sequential(
            nn.Linear(16 * GLOBAL_SIZE * GLOBAL_SIZE, 32), nn.ReLU())
        self.scalar_mlp = nn.Sequential(
            nn.Linear(N_SCALARS, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(128 + 64 + 32, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )

    def forward(self, scalars, patch, gmap):
        c = self.conv(patch)
        c = self.conv_fc(c.reshape(c.size(0), -1))
        g = self.gconv(gmap)
        g = self.gconv_fc(g.reshape(g.size(0), -1))
        s = self.scalar_mlp(scalars)
        return self.trunk(torch.cat([s, c, g], dim=1))


class ShipPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = _ShipTrunk()
        self.actor = nn.Linear(128, N_ACTIONS)
        _init_weights(self, actor_head=self.actor)

    def forward(self, scalars, patch, gmap):
        return self.actor(self.body(scalars, patch, gmap))


class ShipValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = _ShipTrunk()
        self.critic = nn.Linear(128, 1)
        _init_weights(self)

    def forward(self, scalars, patch, gmap):
        return self.critic(self.body(scalars, patch, gmap)).squeeze(-1)


class _SpawnTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.gconv = nn.Sequential(
            nn.Conv2d(GLOBAL_CHANNELS, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.gconv_fc = nn.Sequential(
            nn.Linear(16 * GLOBAL_SIZE * GLOBAL_SIZE, 32), nn.ReLU())
        self.scalar_mlp = nn.Sequential(
            nn.Linear(N_SPAWN_SCALARS, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.trunk = nn.Sequential(nn.Linear(64 + 32, 64), nn.ReLU())

    def forward(self, scalars, gmap):
        g = self.gconv(gmap)
        g = self.gconv_fc(g.reshape(g.size(0), -1))
        s = self.scalar_mlp(scalars)
        return self.trunk(torch.cat([s, g], dim=1))


class SpawnPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = _SpawnTrunk()
        self.actor = nn.Linear(64, N_SPAWN_ACTIONS)
        _init_weights(self, actor_head=self.actor)

    def forward(self, scalars, gmap):
        return self.actor(self.body(scalars, gmap))


class SpawnValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = _SpawnTrunk()
        self.critic = nn.Linear(64, 1)
        _init_weights(self)

    def forward(self, scalars, gmap):
        return self.critic(self.body(scalars, gmap)).squeeze(-1)


# ---------------------------------------------------------------------------
# masking / sampling helpers (shared)
# ---------------------------------------------------------------------------

def mask_logits(logits, mask=None):
    if mask is not None:
        m = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
        logits = torch.where(m, logits, torch.full_like(logits, NEG_INF))
    return logits


@torch.no_grad()
def ship_act(policy: ShipPolicy, scalars_np, patch_np, gmap_np,
             mask=None, deterministic=False, device='cpu',
             value: Optional[ShipValue] = None):
    """Single-ship inference.  patch_np is HWC; gmap_np is CHW.
    Returns (action, log_prob, value)."""
    s = torch.as_tensor(scalars_np[None, ...], dtype=torch.float32, device=device)
    p = torch.as_tensor(np.transpose(patch_np, (2, 0, 1))[None, ...],
                        dtype=torch.float32, device=device)
    g = torch.as_tensor(gmap_np[None, ...], dtype=torch.float32, device=device)
    logits = mask_logits(policy(s, p, g)[0], mask)
    v = float(value(s, p, g)[0].item()) if value is not None else 0.0
    if deterministic:
        return int(torch.argmax(logits).item()), 0.0, v
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample()
    return int(a.item()), float(dist.log_prob(a).item()), v


@torch.no_grad()
def spawn_act(policy: SpawnPolicy, scalars_np, gmap_np,
              mask=None, deterministic=False, device='cpu',
              value: Optional[SpawnValue] = None):
    s = torch.as_tensor(scalars_np[None, ...], dtype=torch.float32, device=device)
    g = torch.as_tensor(gmap_np[None, ...], dtype=torch.float32, device=device)
    logits = mask_logits(policy(s, g)[0], mask)
    v = float(value(s, g)[0].item()) if value is not None else 0.0
    if deterministic:
        return int(torch.argmax(logits).item()), 0.0, v
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample()
    return int(a.item()), float(dist.log_prob(a).item()), v


# ---------------------------------------------------------------------------
# checkpoint bundle
# ---------------------------------------------------------------------------

def save_bundle(path, ship: ShipPolicy, spawn: SpawnPolicy,
                ship_value: Optional[ShipValue] = None,
                spawn_value: Optional[SpawnValue] = None,
                extra: Optional[dict] = None):
    torch.save({
        'ship': ship.state_dict(),
        'spawn': spawn.state_dict(),
        'ship_value': ship_value.state_dict() if ship_value is not None else None,
        'spawn_value': spawn_value.state_dict() if spawn_value is not None else None,
        'meta': {
            'n_scalars': N_SCALARS, 'n_actions': N_ACTIONS,
            'patch_size': PATCH_SIZE, 'patch_channels': PATCH_CHANNELS,
            'global_size': GLOBAL_SIZE, 'global_channels': GLOBAL_CHANNELS,
            'n_spawn_scalars': N_SPAWN_SCALARS,
            **(extra or {}),
        },
    }, path)


def load_bundle(path, device='cpu', need_values=False):
    """Returns (ship_policy, spawn_policy, ship_value|None, spawn_value|None).
    With need_values=True, missing value nets are freshly initialised."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ship = ShipPolicy()
    ship.load_state_dict(ckpt['ship'])
    ship.to(device)
    spawn = SpawnPolicy()
    spawn.load_state_dict(ckpt['spawn'])
    spawn.to(device)
    ship_value = spawn_value = None
    if ckpt.get('ship_value') is not None:
        ship_value = ShipValue()
        ship_value.load_state_dict(ckpt['ship_value'])
        ship_value.to(device)
    elif need_values:
        ship_value = ShipValue().to(device)
    if ckpt.get('spawn_value') is not None:
        spawn_value = SpawnValue()
        spawn_value.load_state_dict(ckpt['spawn_value'])
        spawn_value.to(device)
    elif need_values:
        spawn_value = SpawnValue().to(device)
    return ship, spawn, ship_value, spawn_value
