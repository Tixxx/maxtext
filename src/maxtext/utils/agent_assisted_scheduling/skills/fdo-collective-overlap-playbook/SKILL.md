---
name: fdo-collective-overlap-playbook
description: Profile-guided method for hiding async collective latency on GPU with the agent-assisted scheduling CLI - the exposed-time metric, priority collectives that must never regress, the ordered moves (hoist the start, chase blockers, fill windows with heavy compute without capping another done, place the done, comm-stream and SM-gate effects, host-offload hoisting) and how to read analyze/check-plan output. Use when analysing a scheduled HLO plus a PGLE/FDO profile to decide which collectives to move and where.
---

# FDO collective-overlap playbook

The goal: for every async collective, the compute scheduled between its start
and done should cover its profiled latency. Where it does not, move the start
earlier, pull compute into the window, and place the done late enough to hide
latency but early enough not to delay consumers. Parameters, thresholds and the
reasoning behind each move: `references/move-catalog.md`.

Everything here is done with the CLI (`python -m
maxtext.utils.agent_assisted_scheduling.cli {analyze,check-plan,stamp-plan}`)
and the workflow in `agent-assisted-scheduling-workflow`. Legality is not
optional: read `xla-hlo-schedule-legality` before proposing an order, and
`xla-gpu-scheduling-heuristics` to know what LHS already did.

## Inputs

- The **scheduled** HLO module (post-LHS): `*_gpu_after_optimizations.hlo.pb`.
- The **FDO profile** (`cost_us` by `<module fingerprint>::<scheduling name>`; PGLE averages per kernel
  launch), as the text XLA dumps with `--xla_gpu_experimental_dump_fdo_profiles`.
- A profile entry per collective. A collective without one shows
  `[no profile entry]` in `analyze` and its exposure is **unknown, not zero**:
  do not move it on a guess and do not count it as hidden.

## Reading `analyze`

Per computation, a ranked table; the columns that matter:

| Column | Meaning |
|---|---|
| `P` | priority collective (see below) |
| `latency_us` / `overlap_us` / `deficit_us` | profiled latency, compute between start and done, and `max(0, latency - overlap)` |
| `sim_stall` | stall of the done on one FIFO comm stream with SM gating; the static deficit ignores queueing |
| `pos` | start->done positions |
| `hoist_to` | earliest legal start position, or `pinned by <instruction>` (an operand producer or a control predecessor) |

JSON adds `reachable_deficit_us` (deficit if the done sank to its first real
consumer) and per-computation `priority_exposed_us`. Coverage gaps are listed
at the end: ops with no cost count as free, so a low total next to many gaps
cannot be trusted.

```
overlap(c) = sum cost(inst) strictly between start and done
deficit(c) = max(0, latency(c) - overlap(c))
exposed    = sum of deficits       # a move should lower this
```

## Priority collectives

All-gather, reduce-scatter and all-reduce are priority by default: in sharded
training they carry the weight gathers and gradient reductions that bound the
step. `--priority-prefixes` changes the set (for example add `all-to-all` for a
workload where expert dispatch is the bottleneck). `check-plan` rejects a
plan, even one that lowers the total, if for any priority collective

- its deficit rises (`PRIORITY_REGRESSION ... exposed`),
- its `reachable` deficit rises: a consumer of its done moved earlier and caps
  how far the done can sink, which shrinks its overlap even though the current
  overlap is unchanged, or
- the simulated stall summed over priority collectives rises.

Every other async op (all-to-all, collective-permute, async calls wrapping
custom kernels, ...) shares the comm stream but is not priority: it yields.
Trading a non-priority op's overlap for a priority collective's is the intended
direction; the reverse is not.
`--allow-priority-regression` turns the rejections into warnings; use it only
when you can say why the regression is worth it.

## Moves, in the order to try them

1. **Hoist the start to its earliest legal position** (`hoist_to`). Always go
   all the way: earlier only adds headroom. Moving trivially movable producers
   (bitcast, convert, gte, ...) along is free.
2. **Chase the blocker.** When a real producer pins the start, move that
   producer earlier too, a bounded number of hops, stopping at a heavy anchor
   (GEMM, convolution, attention kernels). Accept if the total falls, or if the producer's cost
   fits inside the slack of every window it leaves, and no priority collective
   is worse.
3. **Fill exposed windows with heavy compute.** For each exposed window
   (earliest start first), move into it a heavy instruction whose dependencies
   already sit before the done. Only take candidates whose cost fits within the
   slack of every other collective covering their current position. Skip a
   candidate that consumes another collective's done when placing it earlier
   would cap that done's reachable overlap. Repeat until nothing changes,
   recomputing deficits from real sums each time.
4. **Place the done.** For an already-hidden collective, put the done just
   before the first large async call or its first consumer, never later than the
   first consumer, so consumers are free to fill the biggest windows.
5. **Choose start/done positions on the comm stream.** Collectives and custom
   comm calls queue on one stream: a start waits for the previous queued op, a
   done stalls compute until its start finishes, and a running large GEMM
   (>= ~1 ms) keeps a collective from beginning. Pick the placement with the
   lowest simulated stall (`check-plan` prints it before and after); among
   placements within the tolerance, take the earliest start, then the latest
   done. Never accept a placement that raises a priority collective's deficit
   or the summed priority stall.
6. **Hoist host-offload copy starts last**, to their earliest legal position.
   Leave a D2H start alone if its host buffer has other readers.
7. **Iterate.** After each batch re-run `analyze` on the proposed order via
   `check-plan`; windows shift when anything moves.

### Start with `optimize`

`cli optimize MODULE --profile P --out plan.json --time-budget-s 120` runs a
bounded local search window by window (hoist a start, pull heavy compute into the
window, sink a done; each move relocates an instruction with its dependency
closure) and keeps only moves that stay legal, add no synchronous conversion,
regress no priority collective and lower `static + 0.3 * stall`. It writes a
stamped plan. It converges in seconds on a loop body and in under a minute on a
14,000-instruction entry computation, usually well before the budget. Take its
plan as the base, then use the moves above on what it leaves, and check every
change with `check-plan`. If one collective holds nearly all the exposure,
`analyze` may print a warning about its latency (see below): fix the input before
trusting the search.

### Which number to trust: static or simulated stall

With one communication stream, collectives queue, so **simulated stall is the
closer model** and the static deficit is an optimistic lower bound (a collective
can show zero static deficit and still wait behind the previous one). Optimize
the two together (default weight 0.3 on stall), never static alone: a search
that took only the static total gained ~70 us of it and raised stall by ~2.7 ms.
Keep the per-collective priority checks regardless. Neither is a measured step
time; if they disagree with a real run, trust the run and say so.

Process computations innermost first (a while body before the computation that
contains the while); the entry computation is the largest.

## Heuristics

- Prefer moving the *collective start* over moving compute; every compute move
  needs slack accounting.
- A better total is not enough: read the per-collective before/after for every
  priority collective.
- Do not chase a done into another exposed collective's blocker unless that
  start's own floor allows it.
- Do not hoist a control predecessor that is the done of a collective: it drags
  the done next to its start.
- Keep changes per batch small. PGLE is noisy and the static window sum ignores
  comm-stream serialization; a better simulated total is a hypothesis until a
  real run confirms it.

## When a collective stays exposed

Not every deficit can be fixed by reordering one computation. Check, in order:

1. **Not enough compute.** If `latency_us` exceeds all the compute that is
   independent of the collective in the same loop body (look at the deficit
   after `hoist_to` is reached and the heaviest window), no order inside the
   body hides it. It needs cross-iteration prefetch or a different decomposition,
   which a plan cannot express (instructions never leave their computation).
2. **Pinned start.** `hoist_to` reads `pinned by X`: a producer or control
   predecessor holds the floor. Try the chase in move 2; if X is itself pinned,
   say so in the report.
3. **Start pinned and done needed immediately.** `analyze` shows `pinned by X`
   with overlap near zero: the op's input is produced just before it and its
   result is consumed right after (typical of all-to-all around an expert or
   attention block). Only independent compute can fill it; do not expect it to
   hide behind its own producers or consumers. This applies to any async op,
   priority or not.
4. **No profile entry.** Entries are keyed by the name the scheduler saw
   (`scheduling_name`, falling back to the wrapped root's) and carry a
   `<fingerprint>::` prefix that the tooling strips for the module being
   analyzed. A sync collective renamed after scheduling is still found through
   its `scheduling_name`. Cloned instructions (`*.double_buffer_clone`) often
   have no entry. These are ignored by the metrics: list them, do not guess.
   If *every* collective lacks an entry, suspect keying first (the module's
   `fingerprint_before_lhs` not matching the profile's prefix, or a profile from
   a different compile), not the profile run.
5. **Suspicious latency.** A latency far above its neighbours for the same bytes
   may include cross-rank skew or contention. `analyze` warns when a collective
   is more than 10x the median of its peers and above 5 ms. Say so before building
   a plan on it: reordering cannot hide a delay that is not the collective's own.

6. **Chains of async calls.** Async calls often come as chains (start, done,
   next start, ...). Each start is pinned behind the previous done
   (`pinned by call-done.N`), so moves 1 and 2 do nothing for them. They are not
   priority, so they stay exposed unless independent compute is left over after
   the priority collectives are covered; report them as the residual rather than
   trading priority overlap for them.

## What to report back

Per collective: name, computation, profile key, latency, overlap before/after,
deficit before/after, positions moved, and what pinned the floor. Total exposed
and summed priority exposure before/after per computation, and any collective
that got worse.
