#!/usr/bin/env python3
"""
rl_v7 / v5_opponent.py  --  run the *pristine* rl_v5 bot but report its name as
"rl_v5" in the replay.

rl_v5/rl_bot.py hard-codes ``game.ready("RLBot")`` (that is the "bot names were
not well stated" problem the user hit before).  We must not modify rl_v5, so
this thin launcher imports rl_v5's bot module unchanged and monkeypatches
``hlt.Game.ready`` at runtime so the engine records the player name as "rl_v5".
No file in rl_v5/ is touched.

Usage (as a bot command for the engine / run_game.py):
    python rl_v7/v5_opponent.py [--model <path>] [--deterministic]
"""

import argparse
import os
import sys

HERE      = os.path.dirname(os.path.abspath(__file__))
MY_EXT    = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(MY_EXT)
RL_V5_DIR = os.path.join(MY_EXT, 'rl_v5')

# Put rl_v5 and the starter kit on the path so its imports resolve unchanged.
sys.path.insert(0, RL_V5_DIR)
sys.path.insert(0, MY_EXT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'starter_kits', 'Python3'))

import hlt  # noqa: E402

# --- monkeypatch the reported name to "rl_v5" without touching rl_v5 ---
_REPORT_NAME = "rl_v5"
_orig_ready = hlt.Game.ready


def _patched_ready(self, name):           # noqa: ARG001  (ignore caller's name)
    return _orig_ready(self, _REPORT_NAME)


hlt.Game.ready = _patched_ready

# Import rl_v5's bot module unchanged (its argparse only runs under __main__).
import rl_bot as v5_bot  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='rl_v5 opponent (named correctly)')
    default_model = os.path.join(RL_V5_DIR, 'checkpoints', 'model_final_weights.pt')
    parser.add_argument('--model', default=default_model)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()
    v5_bot.main(args.model, args.device, args.deterministic)


if __name__ == '__main__':
    main()
