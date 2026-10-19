# collective_overlap_pass.py: function map and constants

File: `src/maxtext/utils/collective_overlap_pass.py` on branch
`tixxx/experimental_custom_scheduling_pass` (2708 lines when this was written).

## Entry points

| Function | Role |
|---|---|
| `register()` | registers `_collective_overlap_pass` as a `POST_SCHEDULER` transform via `jax.extend.xla.register_hlo_module_transformation`; disabled unless `COLLECTIVE_OVERLAP_DISABLE=0`; needs `COLLECTIVE_OVERLAP_XLA_PATH` to compile profile protos |
| `_patch_pgle_profiler()` | hooks PGLE to capture the FDO bytes and the raw XSpace |
| `_collective_overlap_pass(serialized_hlo)` | early-out if no async op; rank 0 computes, shares bytes via JAX KV store with barriers; returns `None` for "no change" |
| `_compute_collective_overlap` | rank-0 body: parse module, take schedule, run `_reorder_to_fixed_point`, serialize |
| `_reorder_to_fixed_point` | loops `_reorder_collectives` (max `_MAX_REORDER_ITERS`=4) with `schedule.update()/verify()`, then `_hoist_host_offload_copy_starts` |

## Cost and profile helpers

| Function | Role |
|---|---|
| `_update_profile`, `_load_costs_from_fdo` | FDO bytes -> `{name: cost_us}` |
| `_load_costs_from_xspace_bytes`, `_average_invocation_costs` | per-invocation spans from XSpace, clustering kernels separated by < `_INVOCATION_GAP_US` (100 us) |
| `_te_ep_overrides`, `_gemm_overrides`, `_collective_overrides` | corrections; te_ep replaces PGLE values, GEMM/collective ones feed only the stream model |
| `_resolve_profile_key` / `_unwrap_to_leaf_name` | name under which a start's latency is keyed: `async_wrapped_root()`, else `to_apply=`, then unwrap nested call/async wrappers |
| `_resolve_inst_cost` | profiled cost; custom fusion = sum of nested entries; host-offload copies 0 |
| `_total_exposed_us`, `_prefix_costs_excluding` | exposure metric on current schedule |
| `_log_profile_coverage_gaps` | warns about fusions/custom calls without cost |

## Structure helpers

| Function | Role |
|---|---|
| `_is_async_start` | permissive: dedicated start opcodes or generic `async-start`; excludes host-offload copies |
| `_is_trivially_movable_inst`, `_is_trivial_fusion_body` | what may be relocated for free (see legality skill) |
| `_control_predecessor_names` | parses `control-predecessors={...}` from text |
| `_computation_root` | `root_instruction()` or parse `ROOT %name =` |
| `_host_offload_copy_kind`, `_register_host_offload_copies` | classify D2H (async DUS into `:S(5)`) / H2D (async DS from it) |
| `_is_te_ep_call_start`, `_is_te_gemm_custom_call`, `_is_heavy_anchor_custom_call` | MaxText/Transformer-Engine specific recognizers; replace for other models |
| `_innermost_first_computations` | order: while body before its parent, entry last |

## Move logic

| Function | Role |
|---|---|
| `_earliest_legal_pos` | floor + movable chain + blocker for a start |
| `_find_best_blocker_position`, `_try_relocate_blocker_earlier` | relocate a producer if it lowers total exposed time or fits every covering window's slack |
| `_chase_async_start_blocker_toward_heavy_compute` | async-only producer chase, <= `_MAX_PRODUCER_RELOCATE_HOPS`=64, stops at heavy anchor |
| `_chase_exposed_collective_blocker_earlier` | move another exposed collective's start so it sits between window and candidate |
| `_find_relocatable_ancestor`, `_chase_heavy_compute_blocker_chain` | walk blocker chain for a candidate heavy op; <= `_MAX_HEAVY_COMPUTE_CHASE_HOPS` |
| `_compute_position_margins` | per position, min slack of covering windows |
| `_fill_exposed_collectives_with_heavy_compute` | worklist fill to a fixed point (`_MAX_FILL_FIXED_POINT_ITERS`=3), `_HEAVY_COMPUTE_MIN_US`=5 |
| `_StuckCandidateTracker`, `_fill_exposed_collectives_best_of` | give-up strategies `none` / `streak` (`_STUCK_CANDIDATE_GIVE_UP_STREAK`=4) / `floor_aware`; keep best total |
| `_relocate_async_done_before_te_ep` | pull a hidden collective's done before the first te_ep start |
| `_may_hoist_control_pred` | anything may move earlier except a collective's done |
| `_simulate_comm_stream`, `_sim_inst_cost` | one-FIFO-comm-stream simulation with optional SM gating |
| `_best_async_placement`, `_hoist_async_starts_to_floor` | while-body start/done placement by simulated stall |
| `_hoist_host_offload_copy_starts` | final DMA hoist |
| `_commit_sequence`, `_reinsert` | splice moved instructions into a sequence |

## Environment knobs

| Variable | Default | Effect |
|---|---|---|
| `COLLECTIVE_OVERLAP_DISABLE` | `1` | pass not registered unless `0` |
| `COLLECTIVE_OVERLAP_XLA_PATH` | unset | XLA checkout for proto compilation |
| `COLLECTIVE_OVERLAP_WHILE_BODY_ONLY` | off | skip entry computation (faster compile) |
| `COLLECTIVE_OVERLAP_HOIST_ASYNC_STARTS` | on | final while-body hoist |
| `COLLECTIVE_OVERLAP_ASYNC_STREAM_MODEL` | on | comm-stream placement in while bodies |
| `COLLECTIVE_OVERLAP_ASYNC_STREAM_TOL_US` | 250 | tie tolerance for placement |
| `COLLECTIVE_OVERLAP_SM_GATE`, `..._SM_GATE_MIN_US` | on, 1000 | large-GEMM gating of collectives |
| `COLLECTIVE_OVERLAP_HOIST_OFFLOAD_COPIES` | on | host-offload copy hoist |
| `COLLECTIVE_OVERLAP_DUMP_MODULE` | `1` | log per-rank input hash / dump modules |
| `COLLECTIVE_OVERLAP_SHARE_TIMEOUT_MS` | 20 min | KV-store wait |

## Known limitations (from the code comments)

- The Python HLO binding exposes little beyond name/opcode/operands/users/
  `to_string()`; several facts are regex-parsed from text.
- Missing profile entries are treated as free cost.
- A moved start can lengthen buffer lifetimes; the pass has no memory model.
- The chase for async-start producers is ungated by exposure; if validation
  shows another collective regressing, the gate must come back.
- Rank consistency relies on every rank reaching invocations in the same order
  (call-count keyed barriers).
