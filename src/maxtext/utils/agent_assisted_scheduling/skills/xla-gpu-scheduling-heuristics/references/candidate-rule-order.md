# LHS candidate comparison order

Source: `ReadySetLt::operator()` in `xla/service/latency_hiding_scheduler.cc`.
Two ready candidates `a`, `b` are compared by the first rule that differs.
The reason names below are what XLA's scheduler logs internally, but agents
must not depend on those logs (see "Inferring the deciding rule without logs").
"Direction" below is for bottom-up scheduling
(the GPU default); top-down variants swap start/done and depth/height.

Candidate `resource_constrained` and per-node flags are updated first
(`UpdateCandidateResourceConstrained`).

| # | Rule (reason) | Effect |
|---|---|---|
| 0a | `early_target_scheduling_rule` when `force_delay_over_memory_pressure` (off on GPU) | see `gpu_early_scheduling_rule` in SKILL.md |
| 1 | `kForceEarly` | nodes with ForceEarly are chosen first in the walk (placed late in program order). GPU sets it for large independent fills (loop fusion, no operands, >= 1 MiB, no explicit stream/group), partially-pipelined send/recv-done, pipelined recv. |
| 2 | `kForceDelay`, `kForceDelayPriority` | nodes with ForceDelay are chosen last in the walk (placed early in program order). GPU sets it from `GpuBackendConfig.force_earliest_schedule` and for partially-pipelined send/recv. |
| 3 | `kPreference` | only if both nodes carry a preference. |
| 4 | Memory pressure (only when usage > limit/2): `kNoIncreaseMemoryOverLimit`, `kDepthOverLimit`, `kDecreaseMemoryOverLimit`, `kMemoryPeakOverLimit` | protects against OOM; wins over overlap heuristics. |
| 0b | `early_target_scheduling_rule` when not `force_delay_over_memory_pressure` (GPU default): `gpu_early_scheduling_rule` | D2D overlap, delay async start for compute / depth, delay host-offload start. |
| 5 | `kForceDelayAfterTarget` | set by target hooks. |
| 6 | `kDelayAsyncStartOfSamePreference` (only with `enable_schedule_by_structure`) | |
| 7 | `kScheduleDone` | prefer scheduling async dones first in the walk, so their windows can open. |
| 8 | `kDelayDoneOfForceDelayedAsyncStart` | preserve overlap windows of force-delayed starts. |
| 9 | `kReleaseNonextendable` | do not extend past-due nonextendable resources. |
| 10 | `kScheduleStart` (only with `enable_release_start_policy`, collective-permute) | |
| 11 | `kStartAtZeroDepth` | with `aggressive_scheduling_policies`: a start with async depth 0 and not resource constrained is delayed, to avoid cost-model-driven hoisting in loops. Position relative to `kLessStall` depends on `prioritize_async_depth_over_stall`. |
| 12 | `kLessStall` | prefer the candidate whose ready time causes less stall. |
| 13 | `kLessSerialResourceConflict` | only with `resource_serializing`. |
| 14 | `kFreeBackedupResource` | prefer a start that frees a resource other work is waiting on. |
| 15 | `kAsyncDepth` (aggressive) | prefer paths behind long async chains. |
| 16 | `kAvoidWaste` (aggressive) | pick the node whose cost best matches the latency still to hide; do not burn big nodes on small windows. |
| 17 | `kUnlockDone` | prefer an op that has an async-done operand. |
| 18 | `target_scheduling_rule` | none on GPU. |
| 19 | `kNotValuableForSelectiveOverlap` | selective resources only. |
| 20 | `kCreatesMoreReadyNodes` (aggressive) | favour ops that unlock more candidates. |
| 21 | `kDecreaseMemory` | when memory increases were computed. |
| 22 | `kOriginalOrder` | final tie-break: position in the memory-scheduler schedule. |

Takeaways:
- Memory rules sit **above** the GPU early rule, so under memory pressure LHS
  will not hoist a start even when the profile says it is exposed. A post-pass
  hoist lengthens a collective's buffer lifetime; check memory (legality skill).
- Everything after the early rule uses the PGLE costs only through ready time
  (`kLessStall`) and `kAvoidWaste`; there is no global search, which is the gap
  the profile-guided pass fills.

## Inferring the deciding rule without logs

LHS logs are normally off, so work from what the compiled module and profile
show. A hypothesis, not a proof: the actual comparison also depended on ready-set
state that is not recorded.

| Rule family | Observable from the module / profile |
|---|---|
| `kForceEarly` / `kForceDelay` | `force_earliest_schedule` in an instruction's backend config; large independent fills (operand-less loop fusion >= 1 MiB, no stream or scheduling-group attribute); pipelined send/recv attributes |
| Early GPU rule: delay start for compute / depth | whether the run used limit > 1 with `collective_start_as_early_as_possible` (from the compile flags, not the HLO); async start depth is recomputable from the dependency graph |
| Host-offload start delay | async start wrapping a fusion with a dynamic-update-slice into an `S(5)` buffer |
| `kScheduleDone`, `kUnlockDone` | done / operand-is-done structure |
| `kLessStall`, `kAvoidWaste` | recompute ready times and window costs from the FDO profile; a placement that contradicts them points at a different rule |
| `kAsyncDepth`, `kCreatesMoreReadyNodes` | dependency-graph shape |
| Memory-pressure rules | not recorded; estimate buffer liveness (peak vs the scheduler limit) if it matters |
| `kOriginalOrder` | the memory scheduler's earlier order is not in the output; a start sitting where nothing above explains it is a candidate for this tie-break |

Practical use: a collective that no profile-driven or flag-driven rule explains
is more likely to be a structural or tie-break outcome, and so more likely to
improve under a profile-guided move.

