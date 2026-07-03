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

## Version lineage: objective → outcome

| Bot | Objective | Outcome & main lesson |
|---|---|---|
| **rl_v1–v2** | Earliest from-scratch PPO experiments: get a learning bot to play the game at all | Superseded; kept as frozen benchmarks (pre-documentation era) |
| **rl_v3** | From-scratch PPO with macro-actions | Worked as a baseline, with skeletons in the closet found later: its "self-play" phase never actually engaged, and it never built dropoffs (spawn drained the bank below the dropoff threshold) |
| **rl_v4** | Beat rl_v3; make dropoff a *learned* action | Achieved. Paid the classic RL tuition: a bank-margin reward was hacked by capital-hoarding (99% "wins" while depositing 0), and an outlier game caused entropy collapse — fixed by a deposited-anchored reward, reward scaling, and entropy guards |
| **rl_v5** | Beat rl_v4 | Achieved decisively (≈23.6k vs 4.5k halite) — but the post-mortem mattered more: the net had never specialised (entropy ≈ max), so the strength was really the hand-coded FSM (PROSPECT/HARVEST/HOME/ESCAPE) it was given as a prior. rl_v5 is in truth a **rule-based bot with an RL veneer** — which made it the perfect foil for everything after |
| **rl_v6** | *Pure* RL: imitate then beat rl_v5 with zero hand-coded rules at inference (learned spawn too) | **Failed** (0/15 vs rl_v5). Key insight: independent per-ship networks cannot represent joint collision resolution — fleets self-destruct; and PPO consistently eroded the BC policy instead of improving it |
| **rl_v7** | Beat rl_v5 via imitation (BC on rl_v5 replays) + PPO, with a hand-coded collision resolver | **Failed** (~25% best). Discovered the **FSM ceiling**: a stateless per-turn policy can only match ~58% of a bot whose decisions depend on hidden per-ship state. Proposed the fix (shadow-FSM features) that rl_v9 later used |
| **rl_v8** | Pivot: imitate **V71**, the 2019 hand-coded bot, after V71 swept a pitch vs rl_v5 (9–1) and rl_v7 (10–0) | **Failed, informatively.** BC alone already won 67% vs V71 — then PPO destroyed it (→ ~20%), and the ~58% BC match ceiling reappeared against a second FSM target. Its autopsy (GAE across interleaved ships, untrained shared-trunk critic, no done-flags on death, no KL guard, no eval gating, stateless-vs-FSM) became rl_v9's design document |
| **rl_v9** | Beat V71 >50% with spawn, dropoff, and enemy-combat behaviour **all learned** | **ACHIEVED: 258/450 = 57.3%** (95% CI 52.8–61.9%). FleetMemory features broke the FSM ceiling (BC match 58% → 86.9%); gated PPO improved on BC for the first time in the project (eval margin −1712 → +5973) |

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
| Surgical, not suicidal, combat | ~0.7 enemy collisions per game, every one a clean 1-for-1 trade (35 lost / 35 killed over 50 games) |
| Disciplined fleet | 0 open-field self-collisions in 50 games; ~5.7 intentional endgame cargo-banking wrecks per game |
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

## Repo pointers

* `rl_v9/README.md` — full design, the rl_v8 failure autopsy it fixes, run
  log with all numbers, and how to read the training logs.
* `halite_engine.py` — the local Python engine every bot trains and
  evaluates on; `replay_viewer.py` / `halite_web_viewer.html` — replay
  viewers (tkinter / browser).
* `Year 2019/` — V71, the original 2019 competition bot (plus the official
  starter kit it was built on).
