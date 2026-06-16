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
    # Action masking
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_mask(logits: torch.Tensor, mask) -> torch.Tensor:
        """Set logits of illegal actions to a large negative value.

        `mask` is a bool array/tensor broadcastable to `logits` ([..., n_actions]);
        True = legal.  Returns masked logits (illegal → −1e9 so softmax ≈ 0).
        Must be applied identically at action-selection and at PPO evaluate time
        so the policy distribution (and thus log-prob ratios) stays consistent.
        """
        if mask is None:
            return logits
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, device=logits.device)
        mask = mask.bool().to(logits.device)
        return logits.masked_fill(~mask, -1e9)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_prior(logits: torch.Tensor, prior_bonus) -> torch.Tensor:
        """Add a per-action logit prior (rl_v5 FSM-hybrid).

        `prior_bonus` is broadcastable to `logits` ([..., n_actions]); it nudges
        the policy toward the FSM-suggested action so the FSM is the default and
        the network only overrides when it has learned a clearly better move.
        Applied identically at action-selection and PPO-evaluate time so the
        distribution (and log-prob ratios) stay consistent.
        """
        if prior_bonus is None:
            return logits
        if not isinstance(prior_bonus, torch.Tensor):
            prior_bonus = torch.as_tensor(prior_bonus, dtype=logits.dtype)
        return logits + prior_bonus.to(logits.device)

    @torch.no_grad()
    def select_action(
        self,
        spatial: torch.Tensor,
        scalars: torch.Tensor,
        mask=None,
        prior_bonus=None,
    ):
        """
        Sample an action for a single ship (no batch dimension).

        `mask` (optional) is a bool[n_actions] of legal actions.
        `prior_bonus` (optional) is a float[n_actions] logit prior (FSM hybrid).

        Returns
        -------
        action   : int
        log_prob : float
        value    : float
        """
        logits, value = self.forward(spatial.unsqueeze(0), scalars.unsqueeze(0))
        if prior_bonus is not None:
            logits = self._apply_prior(logits, torch.as_tensor(prior_bonus).unsqueeze(0))
        if mask is not None:
            logits = self._apply_mask(logits, torch.as_tensor(mask).unsqueeze(0))
        dist          = Categorical(logits=logits)
        action        = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    @torch.no_grad()
    def greedy_action(self, spatial: torch.Tensor, scalars: torch.Tensor,
                      mask=None, prior_bonus=None) -> int:
        """Select the highest-probability legal action (no exploration)."""
        logits, _ = self.forward(spatial.unsqueeze(0), scalars.unsqueeze(0))
        if prior_bonus is not None:
            logits = self._apply_prior(logits, torch.as_tensor(prior_bonus).unsqueeze(0))
        if mask is not None:
            logits = self._apply_mask(logits, torch.as_tensor(mask).unsqueeze(0))
        return int(logits.argmax(dim=-1).item())

    def evaluate(
        self,
        spatial: torch.Tensor,
        scalars: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor = None,
        prior_bonus: torch.Tensor = None,
    ):
        """
        Evaluate a batch of (obs, action) pairs for PPO update.

        `masks` (optional) is a bool[B, n_actions] applied to the logits so the
        evaluated distribution matches the one actions were sampled from.
        `prior_bonus` (optional) is a float[B, n_actions] FSM logit prior, applied
        before masking — must match what select_action used at sampling time.

        Returns
        -------
        log_probs : [B]
        values    : [B]
        entropy   : [B]
        """
        logits, values = self.forward(spatial, scalars)
        if prior_bonus is not None:
            logits = self._apply_prior(logits, prior_bonus)
        if masks is not None:
            logits = self._apply_mask(logits, masks)
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
