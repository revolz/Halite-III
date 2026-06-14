# rl_v4 — Halite III PPO bot

The current active RL bot. Optimizes the true game objective: **bank more halite
than the opponent**, by maximizing halite *deposited*. rl_v1/v2/v3 are frozen
archives kept for benchmarking — do not modify them.

## What's different from rl_v3
- **Learned dropoff** action (9 actions; `DROPOFF` is mask-gated by `dropoff_legal`).
- **Features**: 14 spatial channels, 29 scalars (see `rl_features.py` docstring).
- **Production-aligned reward** (`rl_env.py`): deposited halite + potential shaping
  (cargo valued only if returnable) + small terminal bonus on the *deposited*
  margin vs the opponent. Not raw bank — so hoarding starting capital is worthless.
- **End-game aware**: `game_max_turns()` gives the true length (400 on 32×32, not
  the engine's advertised 500); spawning stops with payback room
  (`SPAWN_MIN_TURNS_LEFT`).

## Train
Run from `my_extension/rl_v4/`:
```bash
# Train against the fixed rl_v3 opponent (recommended)
python rl_train.py --opponent rl_v3

# Default curriculum (idle → greedy → self-play)
python rl_train.py

# Resume
python rl_train.py --opponent rl_v3 --resume checkpoints/model_final.pt
```
Watch `checkpoints/training_log.csv`: want `deposited` rising, `win_flag`/
`bank_margin` positive, and `mean_entropy` **not** collapsing to 0 (collapse =
retrain / raise `--ent-coef`). Don't judge by win rate alone — check `deposited`.

## Evaluate & replay
```bash
# Head-to-head vs an archived bot (each uses its own features, via hlt protocol)
python rl_eval.py --model checkpoints/model_final_weights.pt --opponent rl_v3 --games 50

# Replay vs rl_v3 — run from my_extension/ ; writes .hlt to replays/,
# view with replay_viewer.py
python run_game.py --replay --width 32 --height 32 \
  --bot "python -u rl_v4/rl_bot.py --model rl_v4/checkpoints/model_final_weights.pt" \
  --bot "python -u rl_v3/rl_bot.py --model rl_v3/checkpoints/model_final_weights.pt"
```

## Notes
- Checkpoint weights are committed for replication. Archived degenerate runs
  (`checkpoints_v1_hoarding/`, `checkpoints_v2_collapse/`) and bot logs are
  gitignored.
- All changes keep the network's input/output dimensions stable, so existing
  checkpoints stay loadable.
