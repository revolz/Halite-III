"""
Frozen-bot drivers for rl_v6 training (DAgger expert + PPO opponents).

A FrozenBotDriver loads an archived bot (rl_v5 or rl_v4) — its ActorCritic, its
feature code AND its HaliteEnv command builder — in module isolation, so it runs
with its OWN rules (FSM, logit prior, collision avoidance, spawn economy).  Given
a shared live engine it returns that bot's faithful command string for a player.

Two uses:
  * DAgger expert  — `expert_actions(engine, pid)` runs rl_v5's full pipeline and
    parses the result into per-ship PURE primitive labels (0-4, 5=DROPOFF) plus
    the spawn decision: "what would rl_v5 do in this exact (rl_v6-visited) state".
  * PPO opponent   — `command(engine, pid)` returns the raw command string to feed
    HaliteEnvV6.opponent_command_fn so the opponent plays at full strength.

The isolation pattern mirrors rl_v5/rl_train.py's `_load_rl_v4_modules`.
"""

import importlib
import os
import sys

_HERE   = os.path.dirname(os.path.abspath(__file__))
_MY_EXT = os.path.dirname(_HERE)

from rl_features import DIR_TO_ACTION, ACTION_STAY
from rl_config   import ACTION_DROPOFF_V6

_ISOLATED = ('rl_features', 'rl_model', 'rl_env')


def parse_command(cmd_str: str):
    """Parse an engine command string -> ({ship_id: action 0-5}, spawned_bool)."""
    acts, spawned = {}, False
    toks = cmd_str.split()
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == 'm' and i + 2 < len(toks):
            acts[int(toks[i + 1])] = DIR_TO_ACTION.get(toks[i + 2], ACTION_STAY)
            i += 3
        elif t == 'c' and i + 1 < len(toks):
            acts[int(toks[i + 1])] = ACTION_DROPOFF_V6
            i += 2
        elif t == 'g':
            spawned = True
            i += 1
        else:
            i += 1
    return acts, spawned


class FrozenBotDriver:
    def __init__(self, version: str, weights: str = None, device: str = 'cpu'):
        self.version = version
        vdir = os.path.join(_MY_EXT, version)
        if weights is None:
            weights = os.path.join(vdir, 'checkpoints', 'model_final_weights.pt')
        saved = {n: sys.modules.pop(n) for n in _ISOLATED if n in sys.modules}
        sys.path.insert(0, vdir)
        try:
            self.feats = importlib.import_module('rl_features')
            modl       = importlib.import_module('rl_model')
            renv       = importlib.import_module('rl_env')
            self.model = modl.ActorCritic.load(weights, device=device)
            self.model.eval()
            # A throwaway env instance just to reuse its command builder + FSM state.
            self.env = renv.HaliteEnv(allow_dropoff=True)
            self.env.reset()
            self.env.opponent_model = self.model   # drives ALL players via its policy
        finally:
            sys.path.remove(vdir)
            for n in _ISOLATED:
                sys.modules.pop(n, None)
            sys.modules.update(saved)              # restore rl_v6's modules

    def command(self, engine, pid: int) -> str:
        """Faithful command string for `pid` on the shared `engine`.

        reset() already created the bot's per-player state (_opp_homing always;
        _fsm only for FSM bots like rl_v5) for pids 0..num_players-1, so we just
        point it at the shared engine.  We defensively ensure the dicts cover
        `pid` without assuming the bot HAS an FSM (rl_v4 does not)."""
        self.env.engine = engine
        self.env._opp_homing.setdefault(pid, set())
        fsm = getattr(self.env, '_fsm', None)
        if fsm is not None and pid not in fsm:
            fsm[pid] = self.feats.FSMController()
        return self.env._model_opponent_command(pid)

    def expert_actions(self, engine, pid: int = 0):
        """rl_v5's per-ship PURE primitive labels + spawn decision for `pid`.

        Ships with no explicit command are labelled ACTION_STAY.  Returns
        ({ship_id: action 0-5}, spawned_bool).
        """
        acts, spawned = parse_command(self.command(engine, pid))
        for sid in engine.player_entities[pid]:
            acts.setdefault(sid, ACTION_STAY)
        return acts, spawned
