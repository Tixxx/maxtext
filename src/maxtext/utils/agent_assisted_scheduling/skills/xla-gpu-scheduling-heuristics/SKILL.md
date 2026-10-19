---
name: xla-gpu-scheduling-heuristics
description: How XLA:GPU's latency hiding scheduler (LHS) orders instructions, including gpu_early_scheduling_rule, GPU async resources and the LHS overlap limits, force-early/delay flags, and PGLE cost lookup. Use when reasoning about why XLA placed a collective where it did, or before proposing a schedule that should complement or replace XLA's.
---

# XLA:GPU scheduling heuristics

Read this to predict what LHS did and which of its decisions a profile-guided
pass can improve. Details: `references/candidate-rule-order.md` (rule order),
`references/resources-and-flags.md` (resources, limits, flags).

## Pipeline position

`ScheduleGpuModule` (`xla/service/gpu/gpu_hlo_schedule.cc`):

1. `RunP2PSchedulePreparation` adds control edges for send/recv chains.
2. A memory-minimizing schedule (`ScheduleGpuModuleWithMemoryScheduler`) is
   computed first; LHS starts from it and uses its order as the final
   tie-break (`kOriginalOrder`).
3. If `IsLHSEnabled` (flag `xla_gpu_enable_latency_hiding_scheduler`, else true
   when an optimization effort, the SOL estimator or a valid PGLE profile is
   present), `RunLatencyHidingSchedulerPasses` runs: legalize scheduling
   annotations, optional `SchedulerMemoryFencing`, then `LatencyHidingScheduler`
   with `DefaultSchedulerCore`.
4. Later, `RunPostSchedulingPipelines` does post-schedule copy insertion, then
   the registered `kPostScheduler` transforms (our pass), then
   async->sync conversion and rematerialization. See `xla-hlo-schedule-legality`.

## The scheduler in one paragraph

LHS builds the schedule **bottom-up** (from the last instruction to the first).
Seen in that direction, an async *done* occupies a resource and is scheduled
early in the walk (late in program order); the matching *start* releases it
and is delayed in the walk so compute lands between the pair. The ready set is
ranked pairwise by `ReadySetLt` (`latency_hiding_scheduler.cc`); the first rule
that distinguishes two candidates wins. A start can only be hoisted up to what
the ready-set order and resource limits allow, which is why a profile-guided
post-pass can still find slack.

## `gpu_early_scheduling_rule` (gpu_hlo_schedule.cc)

Installed as `early_target_scheduling_rule`, so it runs before generic
async-window heuristics. In order:

1. `GpuD2DOverlapSchedulingRule`: when selective memcpy overlap is on and the
   memcpy window is open, prefer candidates "valuable for selective overlap"
   (non-memory-bound kernels): `kPreferComputeForD2DMemcpyOverlap`.
2. If `aggressive_scheduling_policies` and `prioritize_compute_over_async_start`
   (needs `parallel_collective_overlap_limit > 1` or the scale-up limit > 1
   **and** `xla_gpu_experimental_collective_start_as_early_as_possible`; the
   defaults in `debug_options_flags.cc` are limit 1 and the flag false, so this
   branch is **off by default**):
   - if either candidate occupies a resource (is a done), defer to later rules;
   - a candidate that releases a resource (is a start) and is not resource
     constrained is delayed in favour of plain compute:
     `kDelayAsyncStartForCompute` (bottom-up delay = earlier in program order);
   - two such starts: the one with greater graph depth is delayed:
     `kDelayAsyncStartForDepth`.
3. For bottom-up scheduling, `DelayMoveToHostAsyncStartCandidateCondition`
   delays an async start whose wrapped fusion has a dynamic-update-slice into
   host memory (D2H offload copy): `kDelayMoveToHostAsyncStart`.

`force_delay_over_memory_pressure` decides whether this rule runs before or
after the memory-pressure policy. GPU config leaves it off, so memory pressure
(when usage exceeds half the limit) is evaluated first.

Defaults from `MakeGPUSchedulerConfig`: `aggressive_scheduling_policies = true`,
`schedule_send_recvs = true`, `use_real_cost_model = false`, per-kind overlap
limits of 1 for all-reduce / collective-broadcast / collective-permute, and the
`parallel_*` limits from debug options.

## What marks an op async to the GPU scheduler

`IsGpuAsyncStart` / `IsGpuAsyncDone` (`gpu_latency_hiding_scheduler.cc`):
async collective start/done (incl. send/recv, collective groups), async compute
start/done, and memcpy async ops. A collective whose backend config says it is
a GPU *sync collective* is not treated as async. A reduce-scatter, all-to-all
or custom call uses the generic `async-start`/`async-done` wrapper, so opcode
alone does not identify the kind; look at the wrapped instruction.

## GPU cost model: what PGLE provides

`ProfileGuidedLatencyEstimator` (`xla/service/profile_guided_latency_estimator.cc`):

- `NodeCost(instr)`: profiled `cost_us` by **instruction name**; async
  collective start/done nodes cost 1.0 (their time is latency, not node cost).
  A miss falls back to the analytical GPU estimator.
- `GetLatencyBetween(start, done)`: profiled start->done latency by name; for
  an async start/done with no direct entry it retries with the wrapped
  instruction's name; a miss falls back to the analytical estimate.
- Instruction names must match the profiled module, so any renaming pass
  between profiling and scheduling silently drops entries.

`xla_gpu_pgle_accuracy_checker` (WARN/ERROR) reports profile misses.

## Practical consequences for an agent

- Do not rely on LHS logs: users normally do not enable them. To judge a
  placement, infer which rules could have decided it from the scheduled module
  and the profile alone; `references/candidate-rule-order.md` ("Inferring the
  deciding rule without logs") lists what is observable.
- LHS limits in-flight collectives by resource (`parallel_collective_overlap_limit`
  defaults to 1): with limit 1 it emits `s0,e0,s1,e1`, never `s0,s1,e0,e1`.
  A proposed schedule may exceed this limit; it is a heuristic, not a
  correctness rule. The one XLA overlap constraint that is about safety, the
  device-group overlap rule, is only installed when
  `xla_gpu_experimental_enable_collective_multi_streaming` is on (or the module
  uses multiple collective domains); see `references/resources-and-flags.md`.
- LHS cost is static per node; real GPUs also serialize NCCL kernels on one comm
  stream and a large GEMM can hold the SMs. The profile-guided pass models that
  (`fdo-collective-overlap-playbook`).
