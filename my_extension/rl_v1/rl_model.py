"""
CNN Actor-Critic model for Halite III RL.

Architecture
------------
Spatial input  [B, H, W, C] → Conv2D stack → Flatten
Scalar input   [B, S]       → Linear → ReLU
Combined       → Shared trunk → Actor head  (action logits)
                              → Critic head (state value)

Shared weights across all ships (ship-centric observations handle identity).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from rl_features import WINDOW_SIZE, N_SPATIAL_CHANNELS, N_SCALAR_FEATURES, N_SHIP_ACTIONS


class ActorCritic(nn.Module):
    """
    Shared actor-critic network.

    Parameters
    ----------
    window_size : int    – spatial observation window (must match WINDOW_SIZE)
    n_spatial   : int    – number of spatial channels
    n_scalars   : int    – number of scalar features
    n_actions   : int    – number of discrete actions per ship
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        n_spatial:   int = N_SPATIAL_CHANNELS,
        n_scalars:   int = N_SCALAR_FEATURES,
        n_actions:   int = N_SHIP_ACTIONS,
    ):
        super().__init__()

        # CNN: operates on [B, n_spatial, H, W]
        self.conv = nn.Sequential(
            nn.Conv2d(n_spatial, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        conv_flat = 64 * window_size * window_size  # 64 × 11 × 11 = 7744

        # MLP for scalar features
        self.scalar_mlp = nn.Sequential(
            nn.Linear(n_scalars, 64),
            nn.ReLU(),
        )

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(conv_flat + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Policy head (actor)
        self.actor  = nn.Linear(128, n_actions)

        # Value head (critic)
        self.critic = nn.Linear(128, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Actor head uses smaller init for more uniform initial policy
        nn.init.orthogonal_(self.actor.weight, gain=0.01)

    def forward(self, spatial: torch.Tensor, scalars: torch.Tensor):
        """
        Parameters
        ----------
        spatial : [B, H, W, C]   float32
        scalars : [B, S]         float32

        Returns
        -------
        logits : [B, n_actions]
        values : [B, 1]
        """
        # Reorder to [B, C, H, W] for Conv2d
        x       = spatial.permute(0, 3, 1, 2)
        conv_out = self.conv(x).flatten(1)
        scal_out = self.scalar_mlp(scalars)

        trunk_in = torch.cat([conv_out, scal_out], dim=1)
        trunk    = self.trunk(trunk_in)

        return self.actor(trunk), self.critic(trunk)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(
        self,
        spatial: torch.Tensor,
        scalars: torch.Tensor,
    ):
        """
        Sample an action for a single ship (no batch dimension).

        Returns
        -------
        action   : int
        log_prob : float
        value    : float
        """
        logits, value = self.forward(spatial.unsqueeze(0), scalars.unsqueeze(0))
        dist          = Categorical(logits=logits)
        action        = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    @torch.no_grad()
    def greedy_action(self, spatial: torch.Tensor, scalars: torch.Tensor) -> int:
        """Select the highest-probability action (no exploration)."""
        logits, _ = self.forward(spatial.unsqueeze(0), scalars.unsqueeze(0))
        return int(logits.argmax(dim=-1).item())

    def evaluate(
        self,
        spatial: torch.Tensor,
        scalars: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Evaluate a batch of (obs, action) pairs for PPO update.

        Returns
        -------
        log_probs : [B]
        values    : [B]
        entropy   : [B]
        """
        logits, values = self.forward(spatial, scalars)
        dist           = Categorical(logits=logits)
        log_probs      = dist.log_prob(actions)
        entropy        = dist.entropy()
        return log_probs, values.squeeze(-1), entropy

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu', **kwargs) -> 'ActorCritic':
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
