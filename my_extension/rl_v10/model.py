"""
rl_v10 / model.py -- full-map convolutional actor-critic.

Input  (N, 20, H, W) canonical-frame observation (obs.py).
Trunk  circular-padded residual conv blocks, each with a global-context
       path (mean+max pool -> linear -> broadcast add), so every cell sees
       the whole board.
Heads  ship policy: 6 logits per cell (read at cells holding our ships)
       ship value:  1 scalar per cell (per-ship return estimate)
       spawn:       2 logits + value from the shipyard cell + global pool.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from obs import C, N_ACT


class Block(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode='circular')
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode='circular')
        self.g = nn.Linear(2 * ch, ch)

    def forward(self, x):
        g = torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)
        h = F.relu(self.c1(x) + self.g(g)[:, :, None, None])
        return F.relu(x + self.c2(h))


class Net(nn.Module):
    def __init__(self, ch=48, blocks=6):
        super().__init__()
        self.cfg = {'ch': ch, 'blocks': blocks}
        self.inp = nn.Conv2d(C, ch, 3, padding=1, padding_mode='circular')
        self.blocks = nn.ModuleList([Block(ch) for _ in range(blocks)])
        self.pi = nn.Conv2d(ch, N_ACT, 1)
        self.v = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.ReLU(), nn.Conv2d(ch, 1, 1))
        self.spawn = nn.Sequential(nn.Linear(3 * ch, 128), nn.ReLU(), nn.Linear(128, 3))
        nn.init.zeros_(self.pi.weight)
        nn.init.zeros_(self.pi.bias)

    def trunk(self, x):
        h = F.relu(self.inp(x))
        for b in self.blocks:
            h = b(h)
        return h

    def forward(self, x, fy, fx):
        """x (N,C,H,W); fy, fx (N,) shipyard cell in canonical frame."""
        h = self.trunk(x)
        logits = self.pi(h)
        value = self.v(h)[:, 0]
        n = torch.arange(h.shape[0], device=h.device)
        fcell = h[n, :, fy, fx]
        g = torch.cat([fcell, h.mean((2, 3)), h.amax((2, 3))], 1)
        s = self.spawn(g)
        return logits, value, s[:, :2], s[:, 2]


def save(net, path, extra=None):
    torch.save({'cfg': net.cfg, 'state': net.state_dict(), 'extra': extra or {}}, path)


def load(path, device='cpu'):
    d = torch.load(path, map_location=device, weights_only=False)
    net = Net(**d['cfg'])
    net.load_state_dict(d['state'])
    return net, d.get('extra', {})
