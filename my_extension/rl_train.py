"""
PPO training loop for Halite III with self-play opponent pool.

Usage
-----
    python rl_train.py --episodes 2000 --checkpoint-dir checkpoints/

Each episode:
  1. A fresh game is initialised via HaliteEnv.
  2. All player-0 ships are stepped; each produces a (obs, action, reward, …) tuple.
  3. Team reward (halite deposited) is shared across all ships that turn.
  4. After the episode, GAE advantages are computed per-ship trajectory.
  5. The policy is updated with PPO for n_epochs mini-batch passes.
  6. Every checkpoint_interval episodes the current weights are saved and
     added to the self-play opponent pool.

Opponent pool
  Opponents load a random past checkpoint when one is available.
  Before any checkpoints exist they use the built-in greedy scripted policy.
"""

import argparse
import os
import random
import sys
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from rl_env   import HaliteEnv
from rl_model import ActorCritic
from rl_features import N_SHIP_ACTIONS

# ---------------------------------------------------------------------------
# Hyper-parameters (can be overridden by CLI)
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    width               = 32,
    height              = 32,
    num_players         = 2,
    episodes            = 2000,
    gamma               = 0.999,
    lam                 = 0.95,    # GAE lambda
    clip_eps            = 0.2,
    vf_coef             = 0.5,
    ent_coef            = 0.05,    # entropy bonus weight — higher = more exploration (was 0.01)
    ent_floor           = 0.5,     # nats — hard minimum entropy; prevents total policy collapse
    lr                  = 3e-4,
    n_epochs            = 4,
    minibatch_size      = 64,
    max_grad_norm       = 0.5,
    checkpoint_interval = 50,
    pool_size           = 10,      # max old checkpoints to keep
    checkpoint_dir      = 'checkpoints',
    device              = 'cpu',
    seed                = None,
    resume              = None,    # path to .pt file to resume from
    start_episode       = 1,       # episode number to start counting from (use with --resume)
    opponent_policy     = 'idle',  # 'idle' for early training, 'greedy' for harder challenge
    death_penalty_scale = 0.5,     # weight on cargo lost when a ship dies (proportional, not flat)
    cargo_reward_scale  = 0.3,     # weight on per-turn cargo-gained reward (dense signal)
)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class ShipTrajectory:
    """Collects one ship's experience for one episode."""
    __slots__ = ('spatials', 'scalars', 'actions', 'rewards',
                 'values', 'log_probs', 'dones')

    def __init__(self):
        self.spatials  = []
        self.scalars   = []
        self.actions   = []
        self.rewards   = []
        self.values    = []
        self.log_probs = []
        self.dones     = []

    def add(self, spatial, scalar, action, reward, value, log_prob, done):
        self.spatials.append(spatial)
        self.scalars.append(scalar)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)


def compute_gae(trajectory: ShipTrajectory, gamma: float, lam: float):
    """Compute GAE advantages and discounted returns for a single trajectory."""
    rewards    = trajectory.rewards
    values     = trajectory.values
    dones      = trajectory.dones
    T          = len(rewards)
    advantages = [0.0] * T
    gae        = 0.0
    next_val   = 0.0  # bootstrap value = 0 (ship dead or episode ended)

    for t in reversed(range(T)):
        mask  = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_val * mask - values[t]
        gae   = delta + gamma * lam * mask * gae
        advantages[t] = gae
        next_val      = values[t]

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


# ---------------------------------------------------------------------------
# Self-play opponent pool
# ---------------------------------------------------------------------------

class OpponentPool:
    """Stores past model checkpoints for self-play."""

    def __init__(self, directory: str, max_size: int = 10):
        self.directory = directory
        self.max_size  = max_size
        self._paths    = deque()

    def add(self, model: ActorCritic, episode: int, device: str = 'cpu'):
        path = os.path.join(self.directory, f'opponent_ep{episode}.pt')
        os.makedirs(self.directory, exist_ok=True)
        torch.save(model.state_dict(), path)
        self._paths.append(path)
        if len(self._paths) > self.max_size:
            old = self._paths.popleft()
            try:
                os.remove(old)
            except OSError:
                pass

    def sample(self, device: str = 'cpu') -> Optional[ActorCritic]:
        if not self._paths:
            return None
        path  = random.choice(list(self._paths))
        model = ActorCritic()
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:

    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.device = torch.device(cfg['device'])

        self.model  = ActorCritic().to(self.device)
        self.optim  = torch.optim.Adam(self.model.parameters(), lr=cfg['lr'])
        self.pool   = OpponentPool(cfg['checkpoint_dir'], cfg['pool_size'])

        self._episode_rewards = deque(maxlen=100)

        # Resume from checkpoint if requested
        if cfg.get('resume'):
            self._load_checkpoint(cfg['resume'])

    # ------------------------------------------------------------------
    # Checkpoint load
    # ------------------------------------------------------------------

    def _load_checkpoint(self, path: str):
        if not os.path.isfile(path):
            print(f"[WARNING] Resume file not found: {path}  — starting fresh.")
            return
        ckpt = torch.load(path, map_location=self.device)
        # Support both plain state_dict and full checkpoint dict
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            self.model.load_state_dict(ckpt['model_state'])
            self.optim.load_state_dict(ckpt['optim_state'])
            print(f"Resumed model + optimizer from {path}")
        else:
            # Plain model weights only (saved by ActorCritic.save())
            self.model.load_state_dict(ckpt)
            print(f"Resumed model weights from {path}  (optimizer state reset)")

    # ------------------------------------------------------------------
    # Episode collection
    # ------------------------------------------------------------------

    def _collect_episode(self, env: HaliteEnv) -> Tuple[List[ShipTrajectory], float]:
        """
        Run one complete game and collect per-ship trajectories.

        Returns
        -------
        trajectories : list of ShipTrajectory (one per ship that lived ≥1 step)
        ep_reward    : total reward accumulated by player 0 this episode
        """
        obs, _      = env.reset()
        done        = False
        ep_reward   = 0.0

        # Active trajectories keyed by ship_id
        active: Dict[int, ShipTrajectory] = {}
        # Completed trajectories (ships that died mid-episode)
        finished: List[ShipTrajectory]    = []

        while not done:
            # Decide actions for all player-0 ships
            ship_actions: Dict[int, int] = {}

            for ship_id, (spatial, scalars) in obs.items():
                if ship_id not in active:
                    active[ship_id] = ShipTrajectory()

                sp_t  = torch.from_numpy(spatial).to(self.device)
                sc_t  = torch.from_numpy(scalars).to(self.device)
                action, log_prob, value = self.model.select_action(sp_t, sc_t)
                ship_actions[ship_id]  = action

                # Store obs + selected action temporarily
                active[ship_id].add(spatial, scalars, action, 0.0, value, log_prob, False)

            # Heuristic spawn: spawn when fleet is small and we can afford it
            spawn = (len(obs) < 8 and
                     env.engine.players[0]['energy'] >= 1000 and
                     env.engine.turn < env.engine.max_turns * 0.7)

            obs, reward, done, _ = env.step(ship_actions, spawn=spawn)
            ep_reward += reward

            # Distribute team reward to all ships that acted this step
            current_ids = set(ship_actions.keys())
            for ship_id in current_ids:
                if ship_id in active and active[ship_id].rewards:
                    active[ship_id].rewards[-1] = reward / max(1, len(current_ids))

            # Mark done for ships that died (missing from next obs)
            next_ids    = set(obs.keys())
            dead_ids    = current_ids - next_ids
            for ship_id in dead_ids:
                if ship_id in active and active[ship_id].dones:
                    active[ship_id].dones[-1] = True
                    finished.append(active.pop(ship_id))

        # End of episode: close all remaining trajectories
        for traj in active.values():
            if traj.rewards:
                traj.dones[-1] = True
            finished.append(traj)

        return [t for t in finished if t.rewards], ep_reward

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self, trajectories: List[ShipTrajectory]) -> float:
        """Run PPO update and return mean policy entropy (nats) for logging."""
        cfg = self.cfg

        # Flatten all trajectories into arrays
        all_sp, all_sc, all_ac, all_ret, all_adv, all_lp = [], [], [], [], [], []

        for traj in trajectories:
            adv, ret = compute_gae(traj, cfg['gamma'], cfg['lam'])
            all_sp .extend(traj.spatials)
            all_sc .extend(traj.scalars)
            all_ac .extend(traj.actions)
            all_ret.extend(ret)
            all_adv.extend(adv)
            all_lp .extend(traj.log_probs)

        if not all_sp:
            return 0.0

        sp_t  = torch.tensor(np.array(all_sp),  dtype=torch.float32, device=self.device)
        sc_t  = torch.tensor(np.array(all_sc),  dtype=torch.float32, device=self.device)
        ac_t  = torch.tensor(all_ac,            dtype=torch.long,    device=self.device)
        ret_t = torch.tensor(all_ret,           dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(all_adv,           dtype=torch.float32, device=self.device)
        lp_t  = torch.tensor(all_lp,            dtype=torch.float32, device=self.device)

        # Normalize advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        N  = len(all_sp)
        bs = cfg['minibatch_size']

        ent_floor   = cfg.get('ent_floor', 0.5)
        total_ent   = 0.0
        total_steps = 0

        for _ in range(cfg['n_epochs']):
            idx = torch.randperm(N, device=self.device)
            for start in range(0, N, bs):
                mb    = idx[start: start + bs]
                lp_new, val, ent = self.model.evaluate(sp_t[mb], sc_t[mb], ac_t[mb])

                ratio     = torch.exp(lp_new - lp_t[mb])
                adv_mb    = adv_t[mb]
                pg_loss1  = -adv_mb * ratio
                pg_loss2  = -adv_mb * ratio.clamp(1 - cfg['clip_eps'], 1 + cfg['clip_eps'])
                pg_loss   = torch.max(pg_loss1, pg_loss2).mean()

                vf_loss   = F.mse_loss(val, ret_t[mb])

                # Entropy floor: if mean entropy is below the floor, the penalty
                # is proportionally larger, pushing the policy back toward exploration.
                mean_ent  = ent.mean()
                ent_loss  = -torch.clamp(mean_ent, min=torch.tensor(ent_floor, device=self.device))

                loss = pg_loss + cfg['vf_coef'] * vf_loss + cfg['ent_coef'] * ent_loss

                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg['max_grad_norm'])
                self.optim.step()

                total_ent   += mean_ent.item() * mb.shape[0]
                total_steps += mb.shape[0]

        return total_ent / max(total_steps, 1)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        cfg = self.cfg
        os.makedirs(cfg['checkpoint_dir'], exist_ok=True)

        env = HaliteEnv(
            width               = cfg['width'],
            height              = cfg['height'],
            num_players         = cfg['num_players'],
            seed                = cfg['seed'],
            opponent_policy     = cfg.get('opponent_policy', 'idle'),
            death_penalty_scale = cfg.get('death_penalty_scale', 0.5),
            cargo_reward_scale  = cfg.get('cargo_reward_scale', 0.3),
        )

        start_ep   = cfg.get('start_episode', 1)
        total_eps  = start_ep + cfg['episodes'] - 1
        start_time = time.time()

        # CSV log (append so resuming keeps history)
        import csv
        log_path   = os.path.join(cfg['checkpoint_dir'], 'training_log.csv')
        log_exists = os.path.isfile(log_path)
        log_file   = open(log_path, 'a', newline='')
        log_writer = csv.writer(log_file)
        if not log_exists:
            log_writer.writerow(['episode', 'reward', 'avg100_reward', 'mean_entropy', 'elapsed_sec'])

        for ep in range(start_ep, total_eps + 1):
            trajectories, ep_reward = self._collect_episode(env)
            mean_ent = self._ppo_update(trajectories)
            self._episode_rewards.append(ep_reward)
            mean_r  = sum(self._episode_rewards) / len(self._episode_rewards)
            elapsed = time.time() - start_time

            log_writer.writerow([ep, f'{ep_reward:.2f}', f'{mean_r:.2f}', f'{mean_ent:.3f}', f'{elapsed:.1f}'])
            log_file.flush()

            if ep % 10 == 0:
                print(f"Episode {ep:5d} | reward {ep_reward:8.1f} | "
                      f"avg100 {mean_r:8.1f} | entropy {mean_ent:.3f} | {elapsed:.0f}s")

            if ep % cfg['checkpoint_interval'] == 0:
                path = os.path.join(cfg['checkpoint_dir'], f'model_ep{ep}.pt')
                # Save full checkpoint (model + optimizer) for resuming
                torch.save({
                    'model_state': self.model.state_dict(),
                    'optim_state': self.optim.state_dict(),
                    'episode':     ep,
                }, path)
                # Also save plain weights for rl_bot.py / rl_eval.py
                self.model.save(path.replace('.pt', '_weights.pt'))
                self.pool.add(self.model, ep)
                print(f"Checkpoint saved → {path}")

        log_file.close()

        # Final save
        final_path = os.path.join(cfg['checkpoint_dir'], 'model_final.pt')
        torch.save({
            'model_state': self.model.state_dict(),
            'optim_state': self.optim.state_dict(),
            'episode':     total_eps,
        }, final_path)
        self.model.save(os.path.join(cfg['checkpoint_dir'], 'model_final_weights.pt'))
        print(f"\nTraining complete. Final model: {final_path}")
        print(f"Training log:    {log_path}")
        return self.model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train Halite III PPO agent')
    for key, default in DEFAULTS.items():
        if key in ('resume',):
            parser.add_argument(f'--{key}', default=None, type=str)
        elif key in ('seed',):
            parser.add_argument(f'--{key}', default=None, type=int)
        else:
            t = type(default) if default is not None else str
            parser.add_argument(f'--{key.replace("_", "-")}', default=default, type=t)

    args = parser.parse_args()
    cfg  = {k: getattr(args, k.replace('-', '_')) for k in DEFAULTS}

    trainer = PPOTrainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()
