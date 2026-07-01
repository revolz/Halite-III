"""
Learned spawn decision for rl_v6.

A tiny MLP mapping the compact per-turn global feature vector
(`rl_config.spawn_global_features`, dim SPAWN_FEATURE_DIM) to a single logit:
spawn a ship this turn or not.  This replaces rl_v5's hand-coded `spawn_econ_ok`
economic gate so that rl_v6 has *no* rule-based logic.
"""

import os

import torch
import torch.nn as nn

from rl_config import SPAWN_FEATURE_DIM


class SpawnHead(nn.Module):
    """Per-turn spawn policy (Bernoulli) + value head, so it can be trained with
    PPO alongside the ship policy (2026-06-28: the shipyard now LEARNS to build a
    fleet instead of using a frozen BC head)."""

    def __init__(self, in_dim: int = SPAWN_FEATURE_DIM, hidden: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, 1)   # spawn logit
        self.value  = nn.Linear(hidden, 1)   # state value (for PPO)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """Return (logit[B], value[B]) for input [B, in_dim]."""
        h = self.body(x)
        return self.policy(h).squeeze(-1), self.value(h).squeeze(-1)

    @torch.no_grad()
    def spawn_prob(self, feats: torch.Tensor) -> float:
        """Probability of spawning for a single feature vector [in_dim]."""
        logit, _ = self.forward(feats.unsqueeze(0))
        return torch.sigmoid(logit).item()

    @torch.no_grad()
    def select_spawn(self, feats: torch.Tensor):
        """Sample a spawn action for one turn -> (action 0/1, log_prob, value)."""
        logit, value = self.forward(feats.unsqueeze(0))
        dist = torch.distributions.Bernoulli(logits=logit)
        a = dist.sample()
        return int(a.item()), dist.log_prob(a).item(), value.item()

    def evaluate(self, feats: torch.Tensor, actions: torch.Tensor):
        """Batch (log_probs[B], values[B], entropy[B]) for a PPO update."""
        logit, value = self.forward(feats)
        dist = torch.distributions.Bernoulli(logits=logit)
        return dist.log_prob(actions), value, dist.entropy()

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu', **kwargs) -> 'SpawnHead':
        """Load tolerantly: a NEW-format checkpoint loads directly; an OLD BC
        checkpoint (Sequential `net.0/2/4`) is remapped onto body+policy so the
        learned spawn behaviour is preserved, with a fresh value head."""
        model = cls(**kwargs)
        old = torch.load(path, map_location=device)
        new = model.state_dict()
        remap = {'net.0.weight': 'body.0.weight', 'net.0.bias': 'body.0.bias',
                 'net.2.weight': 'body.2.weight', 'net.2.bias': 'body.2.bias',
                 'net.4.weight': 'policy.weight', 'net.4.bias': 'policy.bias'}
        for ok, nk in remap.items():
            if ok in old and old[ok].shape == new[nk].shape:
                new[nk] = old[ok]
        for k, v in old.items():               # already-new-format keys
            if k in new and new[k].shape == v.shape:
                new[k] = v
        model.load_state_dict(new)
        model.eval()
        return model
