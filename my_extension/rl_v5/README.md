# rl_v5 — Halite III FSM-hybrid PPO bot

The current active RL bot (created 2026-06-15 from rl_v4). Optimizes the true game
objective — **bank more halite than the opponent** (halite *deposited*). rl_v1–v4 are
frozen archives kept for benchmarking; do not modify them.

## Why rl_v5 was redesigned (the root cause)
rl_v4's learned policy never specialised: `mean_entropy ≈ 2.0` out of a max
`ln(9)=2.197`, with a near-uniform action distribution. The bot's "intelligence" lives
in 9 *macro-actions* (go-to-rich-cell, mine, home …); the policy only had to *select*
the right one per state and never learned to, because in `rl_train.py` advantages are
normalised to unit-std (policy gradient ≈ O(1)) while `ent_coef = 0.20` (~20× normal)
pinned entropy near maximum. That is exactly why the bot didn't reliably go to rich
cells or stay to harvest — those fired ~1/9 of the time at random.

## Design: a per-ship state machine the NN refines
Instead of hoping a near-random policy picks good actions, rl_v5 makes the strategy an
explicit **finite state machine** (`FSMController` in `rl_features.py`) that is the
**default**, with the NN **refining** it.

**States** (per ship): `PROSPECT → HARVEST → HOME`, plus `ESCAPE`.
- **PROSPECT** — lock onto the nearest *richest* cell (deconflicted against other ships
  so the fleet spreads out) and walk toward it.
- **HARVEST** — mine until cargo ≈ full (`HOME_CARGO=900`), then HOME; if the cell is
  exhausted first (`< EXHAUSTED_FLOOR`), re-PROSPECT.
- **HOME** — return along the least-cost path (`least_cost_home_step`); on arrival,
  PROSPECT again.
- **ESCAPE** — after `STUCK_TURNS=5` turns stuck next to friendlies, wander randomly for
  `ESCAPE_TURNS=3` turns to break a jam, then resume. Applies to stuck **homing** ships
  too (endgame-exempt) to clear shipyard gridlock.

**How the NN refines it.** Each turn the FSM emits a *suggested* macro-action per ship.
That suggestion is fed to the network two ways that stay in lock-step:
1. as **observation features** — a one-hot of the FSM state + a one-hot of the suggested
   action, appended to the scalar vector (`fsm_feature_vector`); and
2. as a **logit prior** — `FSM_PRIOR_LOGIT = 3.0` added to the suggested action's logit
   in `rl_model` (`_apply_prior`), applied identically at sampling, PPO-evaluate, and
   inference. So the FSM is ~71 % of the stochastic policy and always the greedy choice,
   while the net keeps ~29 % exploration and can learn to override it where it pays.

**Training fix (mandatory).** `ent_coef 0.20 → 0.01`, floor relaxed
(`ent_floor 0.3`, `ent_floor_coef 0.05`), so the policy can actually sharpen. Watch that
`mean_entropy` settles well below 2.0 (a trained run sits ≈ 0.8–1.1) and `deposited`
climbs past rl_v4's ~5000.

**Endgame "home sacrifice".** In the final `ENDGAME_COLLAPSE_TURNS = 15` turns, loaded
ships pile onto the shipyard/dropoff (friendly-collision avoidance is relaxed *only* for
deposit-cell entry, in `rl_env._build_commands` + `rl_bot`). This required a shared-engine
fix: `halite_engine.py` resolves collisions *before* deposits, so a wreck on your **own**
structure now banks the cargo instead of dumping it (matching official Halite III). The
collision reward term counts only cargo *actually* dumped, so a sacrifice isn't penalised.

### Dimensions / checkpoint compatibility
Observation is **14 spatial channels / 42 scalars** (29 base + 13 FSM) / **9 actions**.
The 42-scalar input differs from rl_v4's 29, so rl_v5 **cold-starts** (cannot warm-start
from rl_v4). The endgame engine change does *not* alter dims, so it is checkpoint-safe.

## Train
Run from `my_extension/rl_v5/`. On Windows set `PYTHONIOENCODING=utf-8` so the
checkpoint-save prints don't crash the console.
```bash
# Cold start vs rl_v4 (the champ to beat; rl_v4 is the default opponent)
PYTHONIOENCODING=utf-8 python rl_train.py --opponent rl_v4 --episodes 8000 --checkpoint-dir checkpoints/

# Resume rl_v5's own run (new-format checkpoint only)
PYTHONIOENCODING=utf-8 python rl_train.py --opponent rl_v4 --resume checkpoints/model_final.pt --start-episode <N+1>
```
Watch `checkpoints/training_log.csv`: want `deposited` rising, `mean_entropy` below 2.0
and non-collapsing, and a **non-uniform** per-action distribution (`dropoff` becomes
non-zero as the net learns to build dropoffs — the one behaviour the FSM never suggests).

## Evaluate & replay
```bash
# Head-to-head vs the champ (each bot uses its own features, via the engine)
PYTHONIOENCODING=utf-8 python rl_eval.py --model checkpoints/model_final_weights.pt --opponent rl_v4 --games 50 --deterministic

# Replay vs rl_v4 — run from my_extension/ ; writes .hlt to replays/
python run_game.py --replay --width 32 --height 32 \
  --bot "python -u rl_v5/rl_bot.py --model rl_v5/checkpoints/model_final_weights.pt --deterministic" \
  --bot "python -u rl_v4/rl_bot.py --model rl_v4/checkpoints/model_final_weights.pt"
```
Benchmark with `--deterministic` to see clean FSM behaviour. Note: every game (training,
eval, head-to-head, replays) runs on the Python `halite_engine.py` — there is no C++
binary in the loop.

## Notes
- The FSM is the single source of truth in `rl_features.py`, driven identically by
  `rl_env.py` (training) and `rl_bot.py` (inference) so there is no train/inference skew.
- Checkpoint weights are committed for replication; bot logs and `__pycache__` are
  gitignored.
- The endgame engine change lives in `my_extension/halite_engine.py` (shared), so it also
  affects rl_v1–v4 play and exact replay comparability — an intentional, endgame-only fix.
- `PLAN.md` captures the original four-change design; the FSM-hybrid above supersedes it.
