# Move catalog: rules, thresholds, rationale

These are tool-independent and model-independent. Thresholds are starting
values, not laws: tune them for the workload with evidence from `analyze` and
the measured step time.
The CLI is the authority on numbers (`analyze`, `check-plan`); this file says
what to try and why.

## Quantities

| Quantity | Definition | Where to read it |
|---|---|---|
| overlap | summed profiled cost of instructions strictly between start and done | `analyze` `overlap_us` |
| deficit | `max(0, latency - overlap)` | `deficit_us` |
| reachable deficit | deficit if the done sank to its first real consumer (looking through get-tuple-element/bitcast), overlap capped at latency | JSON `reachable_deficit_us` |
| sim stall | stall of the done on one FIFO comm stream, SM gating on | `sim_stall` |
| floor | earliest legal start position | `hoist_to`, or `pinned by <instruction>` |
| slack of a window | `overlap - latency` when positive: compute it can lose without becoming exposed | compute from `analyze` rows |

Missing costs count as zero. Host-offload copies are DMA, not SM compute: cost 0,
they hide nothing.

## Cost corrections

PGLE averages per kernel launch. Ops that run several kernels per execution
(grouped GEMMs, fused communication kernels, NCCL collectives) are understated
by the average; their true cost is the span of one invocation, i.e. kernels
separated by less than ~100 us belong together. If the profile you were given is
an average, expect `deficit_us` of those ops to be too small and say so in the
report.

## Moves

### 1. Hoist the start to its floor
- Walk operands through trivially movable ops (bitcast, convert, transpose,
  reshape, broadcast, get-tuple-element, tuple, copy, slice, parameter, constant,
  and fusions whose body is only trivial elementwise/layout ops). The first real
  producer pins the floor just after it. A control predecessor outside the moved
  set pins it the same way.
- Go all the way to the floor. Starting earlier only adds headroom, and a start
  that is not pinned costs nothing to launch early. Exception: the comm-stream
  placement of move 5, which may prefer a later start for stream reasons; the
  priority rule still applies.

### 2. Chase the blocker
- When a real producer pins the floor, try to move it (and its own trivial
  chain) earlier. Bound the hops (64 is generous), and stop at a heavy anchor
  (GEMM, convolution, attention kernels): moving those usually costs more overlap elsewhere than
  it gains.
- Accept when the total deficit falls, or when the producer's cost fits inside the
  slack of every window it leaves ("margin-safe"). Never accept if a priority
  collective's deficit rises.
- Moving *another exposed collective's start* earlier so it stops sitting
  between the window and a candidate is allowed when that start's own floor
  permits.

### 3. Fill an exposed window with heavy compute
- Heavy = cost >= 5 us and not trivially movable. Candidates are instructions
  after the done whose operands already sit before the done (or whose blocker
  chain can be moved).
- Order windows earliest start first; recompute every deficit from real sums
  after each placement.
- A candidate must fit within the slack of every other collective covering its
  current position, so a fill cannot push another collective below its latency.
- Skip a candidate that consumes another collective's done when placing it
  earlier would lower that done's reachable overlap.
- Give up on a candidate after ~4 unresolved restarts, and stop the whole fill
  after ~3 passes with no change. Try the same fill both with and without the
  give-up rule and keep the lower total: small loop bodies like the patient
  strategy, the large entry computation the impatient one.

### 4. Place the done
- A hidden collective's done goes just before the first large async op that is
  not a priority collective (an all-to-all, say) or its first consumer, whichever
  is first. Later than the
  first consumer is illegal; earlier wastes overlap the consumer could have
  filled.

### 5. Comm-stream placement (while bodies)
- All collectives and custom comm calls queue on one stream in this model: a start
  waits for the previous queued op; a done stalls compute until its start has
  finished; a large SM-filling kernel (cost >= ~1000 us: GEMM, convolution,
  attention) holds the SMs so a collective cannot begin while it runs. Which ops
  count is `--sm-gate-ops` (`gemm` default, `all`, `none`).
- Candidate placements are (start position, done position) pairs. Drop any
  that raise a priority collective's simulated stall or its own static deficit.
  Among the rest, prefer those with the lowest summed priority stall, then the
  lowest simulated total; placements within ~250 us of the best are equal, so take
  the earliest start, then the latest done.
- The one-queue model is pessimistic if collectives of different kinds really run
  on separate streams. If `sim_stall` and the measured step disagree, say so and
  prefer the static deficit.

### 6. Host-offload copies
- Hoist D2H/H2D copy starts to their floor last, so the DMA gets the longest
  window. Leave a D2H start whose host buffer has other readers.

### 7. Iterate
- After a batch, evaluate the proposed order with `check-plan`. Instructions can
  be relocated when XLA re-verifies a schedule whose dependency a move invalidated,
  so a window scored as hidden may not be.

## Search objective (for scripted search)

The give-up rules above are for step-by-step filling. A scripted search should
instead minimise `static_exposed + 0.3 * sim_stall` over single relocations
(an instruction plus its dependency closure: hoist pulls in operands and control
predecessors at or after the target, sink pushes users at or before it), keep a
candidate only if it is legal, adds no synchronous-conversion warning and regresses
no priority collective, and stop at a wall-clock budget. `cli optimize` does this, scoring candidates with array arithmetic and sending only the few best by static score through the full simulation and legality checks.
Static alone is a poor objective: it ignores queueing (see SKILL.md).

## Priority collectives

Default priority set: wrapped op `all-gather`, `reduce-scatter`, `all-reduce`.
Reasons: in sharded training they are the weight and gradient traffic that bounds
step time, and a reorder that frees compute for another async op by exposing one
of them is almost never a win. Change the set with `--priority-prefixes` when the
workload says otherwise. The rule is per collective, not per total:

1. static deficit must not rise;
2. reachable deficit must not rise;
3. summed simulated stall over the priority set must not rise (stall moves
   between neighbouring dones on one queue, so a single done may rise if others
   fall by at least as much).

## Known limits

- No memory model: a moved start lengthens buffer lifetimes. Check peak memory
  yourself when moving large all-gathers earlier.
- A cost of zero for ops without a profile entry hides real compute; treat gaps as
  unknown.
- Instructions never leave their computation, so no plan can prefetch across
  loop iterations.
