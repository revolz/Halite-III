# rl_v5 — design plan (beat rl_v4 on halite deposited)

> **SUPERSEDED (2026-06-16).** This is the original four-change plan. The four reward/
> homing tweaks below did not help on their own — the real blocker was that the policy
> never specialised (entropy pinned near max by `ent_coef=0.20`). rl_v5 was redesigned as
> an explicit per-ship **state machine the NN refines**, plus the entropy fix and an
> endgame home-sacrifice. See `README.md` for the current design; kept here for history.

rl_v5 is a fresh copy of rl_v4 (2026-06-15). Goal: collect **and deposit** more
halite than rl_v4. rl_v1–v4 are frozen archives (benchmark only — do NOT modify).

**Feature/action dims are UNCHANGED from rl_v4** (14 spatial channels, 29 scalars,
9 actions) — the changes below are to the reward signal and to the homing/spawn
*algorithms* (which affect train and inference identically). So rl_v5's network is
architecturally identical to rl_v4 and can **warm-start from rl_v4's weights**
(`--resume ../rl_v4/checkpoints/model_final_weights.pt`, fresh optimizer) — a
stronger, faster start than cold training. Cold start also remains possible.

The four owner-requested improvements, mapped to concrete, engine-correct changes,
plus the guardrails carried over from rl_v4's hard-won lessons.

---

## 1. Spawn / friendly-collision is too conservative

**Today.** Spawn is blocked whenever the factory cell *or any of its 4 neighbours*
holds a friendly ship (`rl_env._build_commands` `factory_safe`, `rl_bot._should_spawn`).
This is the conservatism the owner flagged: it stalls early fleet growth whenever
ships sit near the shipyard.

**Why it's wrong.** A spawn lands a ship *on the factory cell only*. The single
real constraint is: the factory cell must be empty at end of turn. Adjacent
friendlies that move away (or just sit) are harmless — we control them.
(Note: head-on *swaps* already work — the engine resolves collisions by final
cell only, so A→B + B→A never collide and the Phase 4 cascade already permits them.
No swap change is needed.)

**Change (engine-correct, less conservative):**
- Replace the adjacency guard with a precise one: spawn is safe iff **no friendly
  ENDS on the factory cell** this turn (resolve moves first, then decide spawn).
- When we intend to spawn, **reserve the factory cell** *before* the Phase 4
  cascade: any friendly whose destination is the factory is forced to STAY off it,
  and the cascade then resolves any induced conflict consistently.
- Apply identically in `rl_env.py` (training) and `rl_bot.py` (inference) to avoid
  train/inference skew.

## 2. Ships clump at the shipyard early (jamming, poor harvest)

**Change — pull the fleet outward, mostly via economics, lightly via shaping:**
- **No new feature needed**: the policy can already perceive local own-ship density
  via scalar 12 ("other friendly ships within 2 steps / 10") and spatial channels 10
  (friendly danger zone ≤1 step) and 13 (friendly cargo congestion). Keeping the
  feature/action dims unchanged also lets rl_v5 warm-start from rl_v4 (see below).
- Added a small, time-limited **early-game congestion penalty** in reward
  (`rl_env.py`): for turns ≤ `CONGEST_TURNS`, `−W_CONGEST · (#own ships with another
  own ship ≤1 step away)`, raw-halite units. Small so production stays dominant.
- The move-cost penalty (§3) + relaxed spawn (§1) do most of the work: rich cells
  sit in an outer ring, so a production-seeking fleet naturally spreads to them.

## 3. Ships don't stand still long enough to harvest

**Change — make leaving a rich cell cost what it really costs ("two birds"):**
- Add an explicit **movement-cost penalty** = the halite actually burned to move =
  `0.1 × halite(current cell)` (the engine's real move cost), as a dense reward term
  in raw-halite units. Leaving a rich cell is expensive, so the ship mines it down
  first; once mined, the cell is cheap to leave — exactly the requested behaviour,
  and it's the *true* objective (burned halite never gets deposited).
- This composes with the existing Φ shaping (cargo valued only if returnable), which
  already rewards cargo growth. Net effect: stay and mine until marginal extract <
  marginal move cost — optimal Halite mining.
- Keep the anti-camp idle penalty, but refine so it only bites genuinely
  unproductive stays (empty/near-empty cargo loitering), not a ship productively
  mining a depleting cell.

## 4. Homing must be least-COST, not just shortest

**Today.** `_home_dir` / `_home_dir_hlt` pick the larger-delta axis — pure Manhattan,
cost-blind.

**Change.** Replace with a **bounded least-cost path** to the nearest deposit:
weighted shortest path on the torus where stepping off a cell costs `0.1 ×
halite(cell)`, allowing at most a small detour budget over the Manhattan distance
(so it routes through low-halite corridors but never wanders). Return the first
step. Implement once as a **shared helper in `rl_features.py`** used by both env and
bot (no skew). Cache the per-deposit cost field per turn for speed.

---

## Benchmark / training
- **Primary opponent = rl_v4** (the champ to beat); add `make_rl_v4_opponent`
  mirroring the existing `make_rl_v3_opponent` (module-isolation load). Keep rl_v3
  as a secondary/regression opponent.
- `rl_eval.py` / `gen_replay` updated to target rl_v4 by default.

## Guardrails carried from rl_v4 (do not regress)
- **Production-anchored reward**: deposited dominates; all new dense terms in raw
  halite and folded into the same `reward_scale=0.01` learning scale. Terminal bonus
  on *deposited* margin only (hoarding starting capital stays worthless).
- **Entropy-collapse defences**: keep ent_coef floor / lr / decay from rl_v4; watch
  `mean_entropy` and the action distribution, not win% alone. Want `deposited` rising.
- **MAX_TURNS**: keep `game_max_turns()` everywhere (true 400 on 32×32).
- Keep all four changes mirrored in env (train) and bot (inference).

## Suggested implementation order
1. §1 spawn/swap (mechanical, low-risk, immediate anti-jam).
2. §4 shared least-cost homing helper.
3. §3 move-cost penalty + idle refinement.
4. §2 density feature + early congestion penalty.
5. rl_v4 opponent wiring; smoke-test; then a full training run.
