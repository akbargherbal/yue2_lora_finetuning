# Session 8 — track-4 audition verdict, and the plan turns to the head

> Reconstructed in session 9 from `context.md` and `PLAN.md`, which were
> rewritten during this session. No dedicated narrative was written at the
> time (that omission is why this file exists).

## The listening verdict (resolves KI-21)

The user listened to the track-4 A/B: base vs planner checkpoints 30/60/100 on
`04-...` (Maqam Nahawand), at the same seed.

- **Pronunciation held at every checkpoint** — the acceptance floor is intact.
- **Step 30 sounded better than 60 and 100.** More steps made it worse, not
  better.
- Even at 30, the لحن (melodic line) reads as **"foreign"** against the
  training corpus, and musicality is simpler than the reference tracks.

## Diagnosis

KL divergence was *climbing*, not leveling off (0 → 0.25 by step 100, KI-03),
so later steps follow tokenizer noise rather than converging on style. The
noise has a specific source: the community semantic-token head
(`tokenizer_head_joint_v4.pt`, KI-04, ~16% exact top-1 on YuE2's own songs) was
**never adapted to this corpus** — used frozen, as shipped. Fine, imprecise
melodic detail is exactly what an out-of-distribution teacher washes out first.
This reframed KI-04 from "accepted ceiling" to "untried fix available": the
model card's own workflow has an *adapt the head to your audio* step
(`joint.py`) this project had never run.

## Decision

Before any more planner training: run Mothersuperior's head calibration on our
corpus and get a listenable answer to *does a calibrated head sound less
"foreign" on the same round-trip test*. `PLAN.md` was rewritten from scratch
(superseding the sessions 6–7 version) around that; the sessions 6–7 planner
work is kept as a labelled comparison point, not discarded.

## Not done

No code ran this session. The calibration execution (and the blocker it would
hit — see session 9) was left to the next.
