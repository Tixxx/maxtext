# GPU async resources, overlap limits and flags

Sources: `xla/service/gpu/gpu_latency_hiding_scheduler.cc` (`GpuAsyncTracker`,
`GpuScheduleCrossesOverlapLimit`), `xla/service/gpu/gpu_hlo_schedule.cc`
(`MakeGPUSchedulerConfig`), `xla/xla.proto` (flag comments).

## Resources (`GpuResourceType`)

An async start *releases* and its done *occupies* the resource in bottom-up
scheduling. A resource hazard type of `kUnshareable` means two ops cannot hold
it at once beyond the limit.

| Resource | Used by | Limit |
|---|---|---|
| `kGpuAsyncStreamCollectives` | NCCL collectives, custom collective calls, collective groups | the scheduler's per-resource limit (not a constraint on a plan) |
| `kGpuAsyncStreamScaleUpCollectives` | collectives in the scale-up fabric domain | the scheduler's per-resource limit (not a constraint on a plan) |
| `kGpuAsyncStreamComputes` | async compute (e.g. offload copy fusions, async wrapped compute) | `xla_gpu_experimental_parallel_async_compute_limit` (default 2) |
| `kGpuAsyncStreamCollectivesP2P`, `Send0/1`, `Recv0/1` | send/recv and P2P-annotated collectives | effectively unlimited (`kNumAsyncCollectivesP2P`) |
| `kGpuAsyncStreamMemcpy` | dynamic-slice D2D memcpy fusions | unlimited; `kSelective` hazard if selective memcpy overlap is on |
| generic `kAllReduce`, `kCollectivePermute`, ... | `SchedulerConfig` per-kind limits | 1 each in `MakeGPUSchedulerConfig` |

## Device-overlap constraint (`GpuScheduleCrossesOverlapLimit`)

Installed only when `xla_gpu_experimental_enable_collective_multi_streaming` is
on or the module uses multiple collective domains (`overlap_limit_rule` in
`gpu_hlo_schedule.cc`); otherwise none of this applies. A collective-done candidate is rejected if any in-flight collective shares **more
than one device** with it (`CountOverlappingDevices > 1`): "More than one shared
device can create a cyclic launch dependency". Unknown or empty participating
device groups are treated as unsafe. Custom collectives are opaque and skipped.
XLA's stated reason is a cyclic launch dependency; treat that as the rule's
rationale when it is enabled, not as a general guarantee.

## Flags worth knowing

| Flag (`DebugOptions`) | Meaning |
|---|---|
| `xla_gpu_enable_latency_hiding_scheduler` | explicit on/off for LHS |
| `xla_gpu_experimental_parallel_async_compute_limit` | async compute (offload) overlap |
| `xla_gpu_experimental_enable_selective_memcpy_overlap` | D2D memcpy overlap rule |
| `xla_gpu_experimental_enable_collective_multi_streaming` | enables the device-overlap rule |
| `xla_gpu_experimental_scheduler_memory_fencing_threshold_bytes`, `..._slack_windows` | `SchedulerMemoryFencing`: control deps bounding how long LHS can defer a large buffer's users; negative threshold disables |
| `xla_gpu_pgle_accuracy_checker` | WARN/ERROR when profile entries are missing |
| `xla_gpu_enable_analytical_sol_latency_estimator` | analytical latency estimator when no profile |
| `xla_gpu_disable_async_collectives` | forces listed collective kinds synchronous |
| `xla_gpu_enable_pipelined_p2p`, `..._pipeline_parallelism_opt_level` | pipelined send/recv force-early/delay |

JAX side: `jax_enable_pgle`, `jax_pgle_profiling_runs`,
`jax_pgle_aggregation_percentile` (`/opt/jax/jax/_src/config.py`) control when
a profile is collected and fed back as `fdo_profile` in the module config.
