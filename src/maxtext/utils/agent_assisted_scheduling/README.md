# Agent-assisted scheduling

Skills that let a coding agent propose, check and apply better XLA:GPU
schedules from an FDO (PGLE) profile. They distill two sources:

- the profile-guided `POST_SCHEDULER` pass in
  `src/maxtext/utils/collective_overlap_pass.py` (branch
  `tixxx/experimental_custom_scheduling_pass`), and
- XLA's own scheduler: `xla/service/gpu/gpu_hlo_schedule.cc`
  (`gpu_early_scheduling_rule`), `xla/service/gpu/gpu_latency_hiding_scheduler.cc`,
  `xla/service/latency_hiding_scheduler.cc`, and the schedule/verifier rules in
  `xla/hlo/ir/hlo_schedule.cc` and `xla/service/hlo_verifier.cc`.

Source paths: XLA in `/opt/xla`, JAX in `/opt/jax`. Skills cite function and
file names rather than line numbers, which drift; grep for them.

## Skills

| Skill | Use it to |
|---|---|
| `skills/agent-assisted-scheduling-workflow` | Entry point: the analyze -> propose -> validate -> apply loop and what the agent must never do. |
| `skills/xla-gpu-scheduling-heuristics` | Understand what XLA's latency hiding scheduler (LHS) does and why, so proposals complement it. |
| `skills/xla-hlo-schedule-legality` | Know which schedules are illegal or unsafe before proposing one. |
| `skills/fdo-collective-overlap-playbook` | Apply the profile-guided heuristics of `collective_overlap_pass.py`. |

Each `SKILL.md` is short and links to `references/` for the detail.

## Intended architecture

```
agent (offline, or rank 0 only)           registered POST_SCHEDULER pass (all ranks)
  analyze exposed collectives  ----+
  propose moves                    |      load plan -> apply -> validate
  validate + simulate (tools)      +--->  (HloSchedule.update()/verify(), dependency
  emit schedule plan                       and async-pair checks) -> on any failure
                                           return None and keep XLA's schedule
```

The agent never decides legality alone, and every rank must compile a
byte-identical module (rank 0 computes, the others read it through the JAX
distributed KV store, as `_collective_overlap_pass` does today).

## Library

Jax-free Python the agent (or a CLI) calls instead of reasoning about legality:

| Module | Provides |
|---|---|
| `schedule_model.py` | `Inst`, `Computation`, `AsyncRole`: a name-keyed model of one computation; a schedule is a `list[str]` |
| `legality.py` | `validate_order` / `is_legal`: exactly-once membership, operand and control order, async pairing, device-group overlap (opt-in, only relevant with collective multi-streaming; no cap on in-flight collectives), in-place reorder vs a baseline, and a warning when a start/done pair would become synchronous |
| `placement.py` | `earliest_legal_pos`, `hoist_start_to_floor`, `reinsert` (ports of `_earliest_legal_pos` / `_reinsert`) |
| `cost.py` | `collective_exposures`, `total_exposed_us`, `coverage_gaps` (ports of `_total_exposed_us`) |
| `simulate.py` | `simulate_comm_stream` with SM gating (port of `_simulate_comm_stream`) |

Tests: `tests/unit/agent_assisted_scheduling_test.py`. The repo `conftest.py`
skips unmarked tests on accelerator hosts, so on a GPU machine run it directly:
`PYTHONPATH=src:. python3 tests/unit/agent_assisted_scheduling_test.py`.

## Status

Skills and the library are written. The adapter from the HLO binding and FDO
profile, the plan format and the plan-applying pass are not; the workflow skill
marks where they plug in. Facts in these skills are taken from reading the sources, not
from running a compile; confirm a rule against the source before relying on it
for a change that ships.
