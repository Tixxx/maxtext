# Legality checklist with sources

Facts below come from reading the sources named; each is a place to re-check
before relying on it.

## Dependencies that act as pins

| Pin | How the library reads it | Source of truth |
|---|---|---|
| Operand edges | `Inst.operands`; `earliest_legal_pos` (placement.py) walks them | `HloSchedule::Verify` |
| Control predecessors | `control-predecessors={...}` parsed from the instruction text (`hlo_text.control_predecessor_names`) because the Python binding exposes little | `HloSchedule::Verify`; set by copy insertion, `LoopScheduleLinearizer`, `RunP2PSchedulePreparation`, `SchedulerMemoryFencing` |
| Control successors | users' control preds | same |
| Consumers of a done | users of the done | a done must stay before its first consumer |

`earliest_legal_pos` walks an async start's operand chain through
*trivially movable* ops only (bitcast, convert, transpose, reshape, broadcast,
gte, tuple, copy, slice, parameter, constant, and fusions whose body is all
trivial elementwise/layout ops). The first non-trivial producer pins the floor
just after it; a control predecessor outside the moving set pins it the same
way. `analyze` reports the result per collective as `hoist_to`, or "pinned by <instruction>" when the start cannot move earlier. That conservative rule is the safe default for an agent as well.

## Async pairing details

- Start has exactly one user (its done) in the passes' working assumption; a
  start with other users is skipped (`_reorder_collectives`).
- Generic `async-start`/`async-done` wraps reduce-scatter, all-to-all, custom
  calls and offload fusions; dedicated `all-gather-start`, `all-reduce-start`,
  `collective-permute-start` exist for those collectives. The scheduled module
  may print host-offload copies as `fusion-start`.
- `async-update` chains: a done may trace back through updates to one start.

## Overlap and ordering constraints from XLA

- `GpuAsyncTracker::GetNumAvailableResources`: collectives resource limit from
  the flag (default 1); async compute default 2; P2P effectively unlimited.
- `GpuScheduleCrossesOverlapLimit`: more than one shared device between a
  candidate collective and an in-flight one is rejected; unknown groups reject.
- `RunP2PSchedulePreparation` ordering for send/recv chains (see header comment
  in `xla/service/p2p_schedule_preparation.h`).
- `ConvertAsyncCollectivesToSync` (`xla/hlo/transforms/collectives/`):
  walks the schedule and converts a start/done pair with only no-op
  instructions in between.

## Aliasing and in-place

- Copy insertion runs before scheduling and again after
  (`RunPostSchedulingCopyInsertion`) and encodes buffer-sharing hazards as
  edges. Do not remove or reorder across a control edge.
- Host D2H copy = async dynamic-update-slice into an `S(5)` buffer, in place.
  Leave a D2H start alone if its host-buffer operand has other users
  (`Inst.writes_operand` makes the validator enforce it).
- Large-buffer lifetime: `SchedulerMemoryFencing` adds control edges from users
  of buffers >= threshold to a later async start so LHS cannot defer them more
  than `slack_windows` windows. Moving across those edges is impossible, which
  is what keeps peak memory bounded.

## Cross-rank safety

- Whatever applies a plan computes it once (one rank or offline) and every rank
  applies identical bytes: the stamped plan is canonical JSON and `Plan.digest()`
  is what ranks compare. Content hashes of the HLO proto are not stable across
  ranks (a unique-id field differs per compile), which is why `input_checksum`
  hashes instruction names and order instead. Nothing may run a collective inside
  the compile callback.

## Things the verifier does NOT check

`HloSchedule::Verify` does not check memory, overlap limits, device-group
overlap, aliasing hazards that are not already edges, or that a collective
still overlaps anything. Those are the agent-side validator's job.
