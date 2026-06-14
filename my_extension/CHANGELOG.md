# Change Log

## 2026-06-14 — rl_v4 bot + visualization fixes

### rl_v4 (new active bot)
- New PPO bot in `rl_v4/`, copied from rl_v3 and extended. Earlier bots
  (rl_v1/v2/v3) are frozen archives for benchmarking.
- **Learned dropoff action**: action space 8→9 (adds mask-gated `DROPOFF`);
  dropoffs are now a policy decision, not a hardcoded heuristic.
- **Features**: spatial channels 11→14 (dropoff-suitability, inspiration-potential,
  friendly-congestion); scalars 24→29 (bank-margin, opp-bank, dropoff-affordable,
  dropoff-slack, halite-remaining).
- **Production-aligned reward**: anchored on halite *deposited* + potential-based
  shaping + small terminal bonus on the *deposited* margin vs the opponent.
  Replaces the bank-margin reward that was being gamed by capital-hoarding.
- **Real self-play** wired up (`env.opponent_model`), plus a fixed-opponent mode
  `--opponent rl_v3` to train directly against the archived rl_v3 bot.
- **Training stability**: reward scaling (value targets stay O(tens)), stronger
  entropy regularization — fixes an entropy→0 / all-HOME collapse.
- **End-game fix**: `rl_bot.py` now computes the true game length via
  `game_max_turns()` instead of the engine's advertised `MAX_TURNS=500`
  (a 32×32 game ends at 400). This removed a train/inference skew and the
  wasteful late-game spawning; spawn is also payback-aware (`SPAWN_MIN_TURNS_LEFT`
  75→100). Checkpoint-compatible (no network-dimension changes).

### Visualization (`replay_viewer.py`, `libhaliteviz` assets)
- Player 1 now renders with the **purple** turtle scheme (swapped with red).
- Restored the original purple sprite artwork.

### Repo
- `rl_v3/` and `rl_v4/` added to version control, **including checkpoint weights**
  for experiment replication. Transient bot logs and archived degenerate runs are
  gitignored.
