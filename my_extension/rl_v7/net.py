#!/usr/bin/env python3
"""
rl_v7 / net.py  --  the policy/value network.

A hybrid net matching the feature design:
  * scalar branch : MLP over the ~45 engineered features
  * patch branch  : small CNN over the 9x9x6 local map tensor
  * shared trunk  : concat -> MLP
  * actor head    : N_ACTIONS logits (the ship's *intent*)
  * critic head   : scalar state value (used by PPO in phase 2)

The network only ever produces an *intent* per ship; collision-freedom is
enforced afterwards by the deterministic resolver (resolver.py), so the net is
never asked to learn fleet coordination.

Action masking and an optional additive logit prior are supported so the same
model class serves both behavioral cloning and PPO fine-tuning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    N_SCALARS, N_ACTIONS, PATCH_SIZE, PATCH_CHANNELS,
)

NEG_INF = -1e9


class ActorCritic(nn.Module):
    def __init__(self, n_scalars=N_SCALARS, n_actions=N_ACTIONS,
                 patch_channels=PATCH_CHANNELS, patch_size=PATCH_SIZE,
                 hidden=256):
        super().__init__()
        self.n_scalars = n_scalars
        self.n_actions = n_actions
        self.patch_channels = patch_channels
        self.patch_size = patch_size

        # patch CNN branch
        self.conv = nn.Sequential(
            nn.Conv2d(patch_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        conv_out = 32 * patch_size * patch_size
        self.conv_fc = nn.Sequential(nn.Linear(conv_out, 64), nn.ReLU())

        # scalar MLP branch
        self.scalar_mlp = nn.Sequential(
            nn.Linear(n_scalars, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )

        # shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(128 + 64, hidden), nn.ReLU(),
            nn.Linear(hidden, 128), nn.ReLU(),
        )

        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # small actor gain -> near-uniform initial policy
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

    # ------------------------------------------------------------------
    def forward(self, scalars, patch):
        """scalars: [B, n_scalars]   patch: [B, C, S, S]  ->  (logits, value)."""
        c = self.conv(patch)
        c = c.reshape(c.size(0), -1)
        c = self.conv_fc(c)
        s = self.scalar_mlp(scalars)
        h = self.trunk(torch.cat([s, c], dim=1))
        return self.actor(h), self.critic(h).squeeze(-1)

    # ------------------------------------------------------------------
    # inference helpers (single sample, numpy in)
    # ------------------------------------------------------------------
    @staticmethod
    def _to_chw(patch_np):
        # [S, S, C] -> [1, C, S, S]
        return np.transpose(patch_np, (2, 0, 1))[None, ...]

    def _logits_value(self, scalars_np, patch_np, device):
        s = torch.as_tensor(scalars_np[None, ...], dtype=torch.float32, device=device)
        p = torch.as_tensor(self._to_chw(patch_np), dtype=torch.float32, device=device)
        logits, value = self.forward(s, p)
        return logits[0], value[0]

    @staticmethod
    def _mask_and_prior(logits, mask=None, prior=None):
        if prior is not None:
            logits = logits + torch.as_tensor(prior, dtype=torch.float32,
                                              device=logits.device)
        if mask is not None:
            m = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
            logits = torch.where(m, logits, torch.full_like(logits, NEG_INF))
        return logits

    @torch.no_grad()
    def greedy_action(self, scalars_np, patch_np, mask=None, prior=None, device='cpu'):
        logits, _ = self._logits_value(scalars_np, patch_np, device)
        logits = self._mask_and_prior(logits, mask, prior)
        return int(torch.argmax(logits).item())

    @torch.no_grad()
    def select_action(self, scalars_np, patch_np, mask=None, prior=None, device='cpu'):
        """Sample an action; returns (action, log_prob, value)."""
        logits, value = self._logits_value(scalars_np, patch_np, device)
        logits = self._mask_and_prior(logits, mask, prior)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(value.item())

    def evaluate_batch(self, scalars, patch, actions, masks=None, priors=None):
        """Batched: returns (log_probs, entropy, values) for PPO updates."""
        logits, values = self.forward(scalars, patch)
        if priors is not None:
            logits = logits + priors
        if masks is not None:
            logits = torch.where(masks, logits, torch.full_like(logits, NEG_INF))
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values

    # ------------------------------------------------------------------
    def save(self, path):
        torch.save({'state_dict': self.state_dict(),
                    'n_scalars': self.n_scalars,
                    'n_actions': self.n_actions,
                    'patch_channels': self.patch_channels,
                    'patch_size': self.patch_size}, path)

    @classmethod
    def load(cls, path, device='cpu'):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(n_scalars=ckpt['n_scalars'], n_actions=ckpt['n_actions'],
                    patch_channels=ckpt['patch_channels'],
                    patch_size=ckpt['patch_size'])
        model.load_state_dict(ckpt['state_dict'])
        model.to(device)
        return model
