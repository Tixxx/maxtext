---
name: fdo-collective-overlap-playbook
description: Profile-guided heuristics for hiding async collective latency on GPU, distilled from gpu_collective_overlap_pass.py (exposed-time metric, earliest legal start, blocker chasing, filling windows with heavy compute, comm-stream and SM-gating simulation, done placement, host-offload hoisting). Use when analysing a scheduled HLO plus a PGLE/FDO profile to decide which collectives to move and where.
---

# FDO collective-overlap playbook

The goal: for every async collective, the compute scheduled between its start
and done should cover its profiled latency. Where it does not, move the start
earlier, pull heavy compute into the window, and place the done late enough to
hide latency but early enough not to delay consumers. Detail, constants and
function map: `references/pass-heuristics.md`.

Legality is not optional: read `xla-hlo-schedule-legality` before moving
anything, and `xla-gpu-scheduling-heuristics` to know what LHS already did.

## Inputs

- The **scheduled** HLO module (post-LHS), per-computation sequences.
- The **FDO profile** (`ProfiledInstructionsProto`, `cost_us` by instruction
  name; PGLE averages per kernel launch). Delivered in the module config
  (`fdo_profile`) to a registered transform; see `jax/_src/xla_transform.py`
  (`module_config` argument).
- Corrections for ops that run several kernels per execution (te_ep dispatch/
  combine, grouped GEMMs, NCCL kernels): per-invocation spans re-derived from
  the raw XSpace, since a per-launch average understates them.

## Core metric

For collective `c` with start `s`, done `d` and profiled latency `L(c)`:

```
overlap(c) = sum cost(inst) for inst strictly between s and d
deficit(c) = max(0, L(c) - overlap(c))
exposed    = sum over c of deficit(c)      # minimise; accept a move only if it
                                           # lowers the total, not just c's
```

Rules for costs:
- a missing profile entry means "free"; log coverage gaps instead of trusting
  them (`_log_profile_coverage_gaps`);
- a custom fusion with no entry of its own is the sum of its nested
  instructions' entries;
- host-offload copies are DMA, not SM compute: cost 0, and they hide nothing.

## Moves, in the order the pass tries them

1. **Hoist the start to its earliest legal position** (floor). Walk the
   operand chain through trivially movable ops; the first real producer or a
   control predecessor pins the floor. Always go all the way to the floor:
   going earlier only adds headroom.
2. **Chase the blocker.** If a non-trivial producer sits right before the start,
   move that producer earlier too, up to a bounded number of hops, stopping at
   a heavy anchor (GEMM / cuDNN / cuBLAS). A move is accepted if it lowers total
   exposed time, or (margin-safe fallback) if the producer's cost fits inside the
   slack of every window it leaves.
3. **Fill exposed windows with heavy compute.** For each still-exposed window
   (earliest start first), scan forward from the done for a heavy instruction
   (cost >= 5 us, not trivial) whose dependencies already sit before the done,
   and move it into the window. Only candidates whose cost fits within the slack
   of every other collective covering their current position are taken, so a
   fill can never push another collective below its latency. Repeat to a fixed
   point; recompute deficits from real prefix sums each time.
4. **Place the done.** For an already-hidden collective, move the done to just
   before the first large async call (te_ep) or its first consumer, so consumers
   are freed to fill the biggest windows. Never later than the first consumer.
5. **Hoist starts in while bodies with a comm-stream model.** Choose the start
   (and done) position that minimises *simulated* stalls on one FIFO comm
   stream: collectives and custom comm calls share it; a start waits for the
   previous queued op; a done stalls compute until its start finishes; a
   running large GEMM (>= ~1 ms) blocks a collective from beginning (SM gate).
   Candidates within a tolerance (default 250 us) of the best are equal; take
   the earliest start, then the latest done.
6. **Hoist host-offload copy starts last** to their earliest legal position,
   so DMAs get the longest window. Leave a D2H start alone if its host buffer
   has other readers.
7. **Repeat to a fixed point** (<= 4 iterations): `schedule.update()` can
   relocate instructions to repair a broken dependency, changing windows.

Process computations innermost-first (a while body before the computation that
contains the while); entry is largest and dominates compile time.

## Heuristics worth stating to an agent

- Prefer moving the *collective start* over moving compute; compute moves must
  be justified by slack accounting.
- Do not chase a done into another exposed collective's blocker unless the
  chased start's own floor allows it (`_chase_exposed_collective_blocker_earlier`).
- Do not hoist a control predecessor that is the done of a collective (it drags
  the done next to its start).
- No single give-up strategy wins: "streak" suits small while bodies, "floor
  aware" suits the large entry computation; try each from the same baseline and
  keep the lowest exposed total.
- Keep per-iteration changes small and re-measure: PGLE data is noisy and the
  static window sum ignores comm-stream serialization.

## What to report back

Per collective: name, computation, profile key, latency, overlap before/after,
deficit before/after, positions moved, and the blocker that pinned the floor.
Total exposed time before/after per computation.
