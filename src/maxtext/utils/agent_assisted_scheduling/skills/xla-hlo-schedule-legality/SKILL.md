---
name: xla-hlo-schedule-legality
description: Rules that make an XLA:GPU HLO schedule legal or unsafe, covering what HloSchedule verify checks, async start/done pairing, control dependencies, device-group overlap, aliasing, host offload, rank consistency and post-schedule passes. Use before proposing or applying any reordering of a scheduled HLO module, and when validating an agent-produced schedule.
---

# HLO schedule legality

A schedule is the per-computation instruction order stored in `HloSchedule`.
Anything an agent emits must pass three layers. Detail and sources:
`references/legality-checklist.md`.

## Layer 1: hard errors (XLA rejects or crashes)

`HloSchedule::Verify` (`xla/hlo/ir/hlo_schedule.cc`) requires, for every live
non-fusion computation:

- every instruction appears **exactly once**, none missing, none extra;
- every **operand** is scheduled before its user;
- every **control predecessor** is scheduled before its user;
- **no instruction leaves its computation** (strict, enforced by the validator
  as `CROSS_COMPUTATION`): a plan for computation `c` may only contain
  instructions of `c`. XLA's `Verify` would also reject the extra/missing
  instruction, but the agent must never even propose a cross-computation move.
  A moved instruction would otherwise be both MISSING in its own computation
  and foreign in the other.

`HloSchedule::Update` (what `schedule.update()` does) replays your order and
postpones any instruction whose operand or control predecessor is not placed
yet, emitting it right after that predecessor; it never moves the operand
earlier, and everything depending on a postponed instruction is postponed with
it. One early misplacement can therefore drag a whole downstream cluster away
from where you put it, and a window you scored as hidden ends up exposed. A
fully valid order is reproduced exactly, so never leave a violation in. The
complete list of rules (membership, dependencies, async pairs, in-place writes,
cross-rank order, side effects): `references/order-invariants.md`.

HLO verifier async rules (`VerifyAsynchronousInstructionPairs`,
`xla/service/hlo_verifier.cc`): an `async-done` traces back to exactly one
`async-start`; each start has exactly one done (except a start carried across a
while loop); async values may only be consumed by their done/update (or
tuple/gte/while/call/conditional/opt-barrier plumbing).

## Layer 2: unsafe but "valid" (deadlock, wrong result, OOM)

- **Rank-consistent order.** Every rank must run the same collectives in a
  compatible order. All ranks must receive byte-identical modules; compute once
  (rank 0) and share. Per-rank different schedules can deadlock NCCL.
- **Device-group overlap (only with multi-streaming).** XLA rejects overlapping
  collectives whose replica groups share more than one device
  (`GpuScheduleCrossesOverlapLimit`, "cyclic launch dependency"), but installs
  that rule only when `xla_gpu_experimental_enable_collective_multi_streaming`
  is on (or the module uses multiple collective domains). Without multi-streaming
  there is no such XLA constraint and the check should be off. The number of
  collectives in flight is *not* a constraint: there is a single communication
  stream.
- **Send/recv chains.** Keep the control-edge order from
  `RunP2PSchedulePreparation` (`p2p_schedule_preparation.h`), e.g. recv => send
  => recv-done => send-done; pipelined chains have loop-carried order.
- **Aliasing / in-place ops.** `dynamic-update-slice`, donated buffers, fused
  in-place ops and while-loop carried buffers share storage. A reader of the old
  value must stay ordered before the in-place write. Copy insertion already
  encoded this as data/control edges; preserve all control edges, and check
  buffer liveness if you move a writer earlier or a reader later.
  `LoopScheduleLinearizer` adds loop control edges for this reason.
- **Host offload (S(5)).** D2H copy starts update a host buffer in place: do not
  hoist one past other users of that buffer. H2D copies only read it.
- **Memory.** Hoisting a collective start earlier extends the lifetime of its
  operand and output buffers; deferring a done extends its consumers' inputs.
  LHS enforces `memory_limit`; a post-pass must estimate peak memory itself, and
  `SchedulerMemoryFencing` control edges bound large-buffer deferral.
- **Scheduling annotations.** Instructions with `_scheduling_group_id`
  (`LegalizeSchedulingAnnotations`) form a unit; do not split a group or insert
  non-group ops inside it unless you re-run the legalizer.
- **Cross-loop and called computations.** A schedule is per computation and
  instructions never move between computations (Layer 1, strict). Treat a
  `while`, `call` or `conditional` as one instruction in its parent; hoisting
  work out of a loop body is not a legal move for this flow.

## Layer 3: what runs after your plan is applied

`RunPostSchedulingPipelines` (`xla/service/gpu/gpu_compiler.cc`) runs
post-scheduling copy insertion, then registered `kPostScheduler` transforms
(where your plan is applied), then:

- **async -> sync conversion** (`ConvertAsyncCollectivesToSync`): a start whose
  done has only no-op instructions between them is turned into a synchronous
  collective. Placing a start next to its done throws the overlap away.
- **rematerialization / host offload** (`HloRematerialization`): may change
  the schedule again if the memory limit is exceeded.
- thunk emission consumes the final order.

## Checklist before accepting a proposed schedule

0. Read `references/order-invariants.md` once; it is the full rule list.
1. Same instruction set per computation; no duplicates; no instruction placed
   in a computation other than its own.
2. All data and control edges respected (run the verifier, do not eyeball).
3. Every start precedes its done; the pair stays in one computation.
4. If multi-streaming is on: no concurrent collectives with overlapping device groups. No cap on how many are in flight.
5. Host-offload and in-place ordering preserved; side-effecting ops keep their
   original relative order; at least one real op between every start and done.
6. Memory estimate within the scheduler limit, or compared against XLA's.
7. Same result on every rank.
8. On any failure, discard the whole proposal and keep XLA's schedule; never
   partially apply.
