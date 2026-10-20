"""
rl_v10 / ext.py -- run an external engine-protocol bot (V71, rl_v9, ...) as the
opponent inside a BatchSim game.  Messages are byte-for-byte the engine's
format (halite_engine._init_message / _turn_message), including its constants.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MY_EXT = os.path.dirname(HERE)
sys.path.insert(0, MY_EXT)
from halite_engine import CONSTANTS_JSON  # noqa: E402

PY = sys.executable
OPP_CMDS = {
    'v71': f'"{PY}" -u "{MY_EXT}/Year 2019/MyBot - V71/MyBot.py"',
    'rl_v9': f'"{PY}" -u "{MY_EXT}/rl_v9/rl_bot.py" --model "{MY_EXT}/rl_v9/checkpoints/best.pt" --deterministic',
}
DIR = {'n': 1, 'e': 2, 's': 3, 'w': 4}


class ExtBot:
    def __init__(self, name, sim, b, pid, seed):
        self.sim, self.b, self.pid = sim, b, pid
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        self.wd = tempfile.mkdtemp(prefix='v10ext_')
        self.proc = subprocess.Popen(OPP_CMDS[name], shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1, cwd=self.wd, env=env)
        H, W = sim.H, sim.W
        c = dict(CONSTANTS_JSON, map_width=W, map_height=H, game_seed=int(seed))
        lines = [json.dumps(c), f'2 {pid}']
        for p in (0, 1):
            fx, fy = sim.fac[b, p]
            lines.append(f'{p} {fx} {fy}')
        lines.append(f'{W} {H}')
        for y in range(H):
            lines.append(' '.join(str(int(v)) for v in sim.halite[b, y]))
        self._write('\n'.join(lines) + '\n')
        self.alive = True
        self.proc.stdout.readline()                       # bot name
        self.last = sim.halite[b].copy()
        self.pending = False

    def _write(self, s):
        try:
            self.proc.stdin.write(s)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self.alive = False

    def send_turn(self):
        sim, b = self.sim, self.b
        lines = [str(sim.turn)]
        for p in (0, 1):
            ys, xs = np.nonzero(sim.occ[b, p])
            drops = [d for d in sim.drop_list[b] if d[1] == p]
            lines.append(f'{p} {len(ys)} {len(drops)} {int(sim.bank[b, p])}')
            for y, x in zip(ys, xs):
                lines.append(f'{int(sim.sid[b, p, y, x])} {x} {y} {int(sim.cargo[b, p, y, x])}')
            for did, _, x, y in drops:
                lines.append(f'{did} {x} {y}')
        ch = np.argwhere(sim.halite[b] != self.last)
        lines.append(str(len(ch)))
        for y, x in ch:
            lines.append(f'{x} {y} {int(sim.halite[b, y, x])}')
        self.last = sim.halite[b].copy()
        self._write('\n'.join(lines) + '\n')
        self.pending = self.alive

    def read_actions(self):
        """Returns (act (H,W) engine frame, spawn bool)."""
        sim, b, p = self.sim, self.b, self.pid
        act = np.zeros((sim.H, sim.W), np.int64)
        spawn = False
        if not self.pending:
            return act, spawn
        self.pending = False
        line = self.proc.stdout.readline()
        if not line:
            self.alive = False
            return act, spawn
        pos = {}
        for y, x in zip(*np.nonzero(sim.occ[b, p])):
            pos[int(sim.sid[b, p, y, x])] = (y, x)
        tok = line.split()
        i = 0
        seen = set()
        while i < len(tok):
            t = tok[i]
            if t == 'g':
                spawn = True
                i += 1
            elif t == 'm' and i + 2 < len(tok):
                try:
                    s = int(tok[i + 1])
                except ValueError:
                    s = -1
                if s in pos and s not in seen and tok[i + 2] in DIR:
                    act[pos[s]] = DIR[tok[i + 2]]
                    seen.add(s)
                elif s in pos:
                    seen.add(s)
                i += 3
            elif t == 'c' and i + 1 < len(tok):
                try:
                    s = int(tok[i + 1])
                except ValueError:
                    s = -1
                if s in pos and s not in seen:
                    act[pos[s]] = 5
                    seen.add(s)
                i += 2
            else:
                i += 1
        return act, spawn

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
