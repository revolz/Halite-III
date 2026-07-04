# my_extension — teaching a neural network to beat a hand-coded Halite III bot

A series of reinforcement-learning bots (`rl_v1` … `rl_v9`) built in 2026 on a
Python re-implementation of the Halite III engine (`halite_engine.py`). The
arc of the project: start from scratch with pure RL, discover why that fails,
and end with **rl_v9 — a bot whose decisions are all learned — beating V71,
the author's strongest hand-written 2019 competition bot, 57.3% over 450
games.**

Each `rl_v*/` folder is a frozen, self-contained bot (own features, model,
training scripts); later bots benchmark against earlier ones. Detailed docs
live in each folder's README; this file is the high-level lineage.

**Published checkpoints:** every bot ships exactly one weight file,
`checkpoints/best.pt` — its strongest checkpoint (for rl_v9, the model behind
the 57.3%-vs-V71 verdict). All other training artifacts (intermediate
snapshots, BC checkpoints, full-pickle saves) are gitignored and reproducible
via each bot's documented pipeline. Exception: rl_v6 publishes no weights —
it was abandoned with no result to verify.

## Version lineage: objective → outcome

*Terms: **BC** = behavioral cloning (train a network to imitate a bot from
its replays). **PPO** = reinforcement learning by playing games. **FSM** =
hand-coded finite state machine (per-ship states like MINING/RETURNING).*

| Bot | Objective | Outcome & main lesson |
|---|---|---|
| **rl_v1–v2** | First from-scratch RL experiments: play the game at all | Superseded; kept as frozen benchmarks |
| **rl_v3** | From-scratch PPO with macro-actions: besides the 5 basic moves, the policy could pick HOME (auto-walk to a deposit) or PROSPECT (auto-walk to rich halite) | Became the baseline. Flaws found later: its "self-play" never actually ran, and it never built dropoffs |
| **rl_v4** | Beat rl_v3; make dropoff building a *learned* decision | **Achieved.** Lesson: rewards get gamed — a bank-based reward gave 99% "wins" with zero halite deposited; fixed by rewarding deposits directly |
| **rl_v5** | Beat rl_v4 | **Achieved decisively** (≈23.6k vs 4.5k halite). But the strength came from its hand-coded FSM, not the network — rl_v5 is really a **rule-based bot with an RL veneer**. That made it the perfect benchmark: could a *truly* learned bot beat it? |
| **rl_v6** | Beat rl_v5 with *pure* learning — zero hand-coded rules, spawn included | **Failed** (0/15). Lesson: ships deciding independently can't avoid crashing into each other — some hand-coded collision resolution is necessary |
| **rl_v7** | Beat rl_v5 by imitating its replays (BC), then improving with PPO; hand-coded collision resolver allowed | **Failed** (≈25% wins). Discovered the **≈58% imitation ceiling** (explained below) and proposed the memory-feature fix that rl_v9 later used |
| **rl_v8** | Beat rl_v5 and rl_v7 by imitating a stronger teacher: **V71**, the hand-coded 2019 bot, which had just crushed both (9–1, 10–0) | **Failed.** PPO made the bot *worse*, not better (early 4-of-6 vs V71 → ≈20%), and the same ≈58% imitation ceiling reappeared. Diagnosing *why* PPO failed (six bugs, listed in `rl_v9/README.md`) became rl_v9's design document |
| **rl_v9** | Beat V71 >50%, with moves, spawn, dropoffs and combat **all learned** | **ACHIEVED: 57.3% over 450 games** (95% CI 52.8–61.9%). FleetMemory features broke the imitation ceiling (58% → 86.9% match); fixed PPO improved on BC for the first time in the project. Also beats rl_v8 head-to-head (75% over 100 games) |

**The ≈58% imitation ceiling (the "FSM ceiling"), explained:** behavioral
cloning trains a network to predict, from the board, the action the teacher
took. But rl_v5 and V71 are finite state machines: each ship carries hidden
state (e.g. "I'm RETURNING", a queued path), so two identical-looking boards
can get different actions. A board-only network cannot predict those turns —
its match rate stalls at ≈58% regardless of training, a missing-information
limit, not a tuning problem. rl_v7 hit it first (vs rl_v5); rl_v8 hit it
again (vs V71), proving the cause was statelessness, not the teacher. rl_v9
broke it by feeding the network FleetMemory — a small per-ship memory
(homing flag, last action, stuck counter) approximating the teacher's hidden
state — lifting match rate to 86.9%.

## rl_v9 vs V71 — learned bot vs hand-coded bot

The final match-up is the whole experiment in miniature: **2019 human
expertise versus 2026 machine learning, on equal terms.**

| Aspect | V71 (2019) | rl_v9 (2026) |
|---|---|---|
| Nature | 739 lines of hand-written strategy, refined across 71 versions in competition | Neural networks (policy + spawn heads); no strategy code written by a human |
| Strategy origin | A human thought of every behaviour and wrote it down | Imitation of V71's replays, then self-improvement playing against V71 |
| Core mechanism | Explicit per-ship FSM (SEARCHING → NAVIGATING → MINING → RETURNING) | Learned policy over hand-engineered observations (incl. FleetMemory) |
| Targeting | Hand-derived ROI formula scanned over a local window | Implicit — learned end-to-end, no formula anyone wrote |
| Pathing | Precomputed minimum-cost paths, popped one step per turn | One action per turn, re-decided fresh each turn |
| Can it improve? | Frozen since 2019, forever | More training keeps changing it |

Decision by decision:

| Decision | V71 (hand-coded) | rl_v9 (learned) |
|---|---|---|
| Where each ship moves | FSM + ROI formula + path solver | `ShipPolicy` net — imitation of V71's replays, then PPO self-improvement |
| When to spawn ships | `ship count ≤ richest enemy + 5` rule | `SpawnPolicy` net — learned from V71's own spawns, then PPO-tuned |
| When/where to build dropoffs | bank/turn threshold formula | Learned (same nets; physical legality is the only mask) |
| Whether to risk enemy collision | hard-coded avoidance | Learned — the reward prices each potential trade; the net decides |
| *Not* learned (fleet hygiene only) | — | friendly-collision deconfliction, endgame recall |

The "self-learning how to defeat V71" arc, stage by stage:

| Stage | What happened | Result |
|---|---|---|
| 1. Study | Behavioral cloning on 200 games of V71's own replays | 86.9% action match — the student mimics the teacher |
| 2. Parity | Cloned policy evaluated against V71 | 50% wins — as good as the teacher, no better |
| 3. Self-improvement | 300 PPO episodes *playing against* V71, keeping only checkpoints the scoreboard proved better | Eval margin −1712 → **+5973** (episode-160 champion) |
| 4. Verdict | 450 deterministic games vs V71 | **57.3% wins** — the student surpasses the teacher |

The result is a policy that no longer plays like V71 — and its winning
strategy was never designed by anyone; it emerged from the reward:

| Emergent behaviour (from replay analysis) | Evidence |
|---|---|
| **Positional suppression** — squat the halite V71's ROI formula wants; V71's own hard-coded enemy avoidance freezes it | 13 of 34 wins in one batch held V71 under 5k halite (extremes: 148, 376) |
| Surgical, not suicidal, combat | ≈0.7 enemy collisions per game, every one a clean 1-for-1 trade (35 lost / 35 killed over 50 games) |
| Disciplined fleet | 0 open-field self-collisions in 50 games; ≈5.7 intentional endgame cargo-banking wrecks per game |
| V71's remaining crown | On an uncontested rich board V71 still out-mines rl_v9 (blowouts up to 74k) — the open problem for any rl_v10 |

The honest asterisk, documented throughout — what is and isn't learned:

| Component | Learned? | Note |
|---|---|---|
| Ship moves, spawn, dropoff, enemy-combat decisions | **Yes** | The goal; all network outputs |
| Observations / features (incl. FleetMemory) | No | Hand-engineered inputs, like every obs space |
| Friendly-collision deconfliction | No | Fleet hygiene — accepted carve-out |
| Endgame recall | No | Fleet hygiene — accepted carve-out |

That line — hand-designed inputs, learned policy — is the compromise that
finally worked after v6's purist attempt and v7/v8's stateless attempts both
failed.

## Independent verification: full 9-bot round robin (2026-07-04)

To sanity-check the repo end-to-end on a fresh clone, all runnable bots
(`rl_v1`–`rl_v5`, `rl_v7`–`rl_v9`, and `V71`) were pitted against each other
in a round robin: every pair played 10 games (seeds 1–10, deterministic
policies, 32×32 map, `run_game.py` / `halite_engine.py`) — 36 pairings × 10
games = **360 games total, 0 crashes/errors**. `rl_v6` was excluded: its
`checkpoints/` folder ships no `best.pt`, matching its documented
"abandoned, no result to verify" status above.

### Final standings

| Rank | Bot | Objective | Record (W-L-T) | Win % | Avg Halite | Avg Opp Halite |
|---|---|---|---|---|---|---|
| 1 | **rl_v9** | Beat V71, everything learned | 72-8-0 | 90.0% | 28,253 | 6,087 |
| 2 | **V71 (2019)** | Hand-coded 2019 benchmark bot | 70-10-0 | 87.5% | 22,495 | 5,707 |
| 3 | **rl_v8** | Beat rl_v5/rl_v7 by imitating V71 | 60-19-1 | 75.0% | 16,951 | 9,537 |
| 4 | **rl_v5** | Beat rl_v4 (FSM hybrid) | 54-25-1 | 67.5% | 12,430 | 11,072 |
| 5 | **rl_v4** | Beat rl_v3, learned dropoffs | 36-44-0 | 45.0% | 3,252 | 11,926 |
| 6 | **rl_v7** | Beat rl_v5 via BC + PPO | 31-48-1 | 38.8% | 4,211 | 11,670 |
| 7 | **rl_v3** | From-scratch PPO, macro-actions | 24-56-0 | 30.0% | 1,640 | 12,857 |
| 8 | **rl_v2** | Early from-scratch RL experiment | 8-70-2 | 10.0% | 384 | 13,912 |
| 9 | **rl_v1** | Early from-scratch RL experiment | 0-75-5 | 0.0% | 193 | 7,041 |

This independently corroborates the version-lineage table above: rl_v9 is
the strongest bot in the repo and the only one to out-rank V71 overall, the
FSM-hybrid/imitation bots (rl_v8, rl_v5) form the next tier, and the earlier
from-scratch/stateless experiments (rl_v1–v4, v7) trail behind in roughly
their documented order.

### Head-to-head cross table

Row bot's record vs column bot, out of 10 games each:

| Bot | rl_v9 | V71 | rl_v8 | rl_v5 | rl_v4 | rl_v7 | rl_v3 | rl_v2 | rl_v1 |
|---|---|---|---|---|---|---|---|---|---|
| **rl_v9** | — | 8-2 | 7-3 | 7-3 | 10-0 | 10-0 | 10-0 | 10-0 | 10-0 |
| **V71** | 2-8 | — | 9-1 | 10-0 | 10-0 | 10-0 | 9-1 | 10-0 | 10-0 |
| **rl_v8** | 3-7 | 1-9 | — | 7-3 | 10-0 | 10-0 | 10-0 | 10-0 | 9-1 |
| **rl_v5** | 3-7 | 0-10 | 3-7 | — | 10-0 | 9-1 | 10-0 | 10-0 | 9-1 |
| **rl_v4** | 0-10 | 0-10 | 0-10 | 0-10 | — | 7-3 | 9-1 | 10-0 | 10-0 |
| **rl_v7** | 0-10 | 0-10 | 0-10 | 1-9 | 3-7 | — | 8-2 | 10-0 | 9-1 |
| **rl_v3** | 0-10 | 1-9 | 0-10 | 0-10 | 1-9 | 2-8 | — | 10-0 | 10-0 |
| **rl_v2** | 0-10 | 0-10 | 0-10 | 0-10 | 0-10 | 0-10 | 0-10 | — | 8-2 |
| **rl_v1** | 0-10 | 0-10 | 0-10 | 0-10 | 0-10 | 0-10 | 0-10 | 0-10 | — |

The table is almost perfectly transitive by rank — every bot beats everyone
ranked below it and loses to everyone above it — except for the top: rl_v9
takes the rl_v9-vs-V71 series 8-2 despite finishing only narrowly ahead of
V71 in overall win rate, and rl_v7 (rank 6) still loses the majority of its
games against rl_v4 (rank 5, 3-7), consistent with rl_v7's own documented
conclusion that it was tuned only to beat rl_v5 and never specifically
tested against rl_v4.

Raw per-game data: `round_robin_results.csv`; full log: `round_robin_log.txt`;
driver script: `round_robin.py` (all in this directory).

## Repo pointers

* `rl_v9/README.md` — full design, the rl_v8 failure autopsy it fixes, run
  log with all numbers, and how to read the training logs.
* `halite_engine.py` — the local Python engine every bot trains and
  evaluates on; `replay_viewer.py` / `halite_web_viewer.html` — replay
  viewers (tkinter / browser).
* `Year 2019/` — V71, the original 2019 competition bot (plus the official
  starter kit it was built on).
