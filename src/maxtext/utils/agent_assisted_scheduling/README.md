# Agent-assisted scheduling

Skills that let a coding agent propose, check and apply better XLA:GPU
schedules from an FDO (PGLE) profile. The skills and the CLI are self-contained:
an agent needs nothing else. They distill two sources:

- the heuristics of the temporary profile-guided `POST_SCHEDULER` reorderer
  (`src/maxtext/utils/gpu_collective_overlap_pass.py`, branch
  `tixxx/experimental_custom_scheduling_pass_fix`). Nothing in `skills/` refers
  to it; `placement.py`, `simulate.py`, `hlo_adapter.py` and `schedule_model.py`
  are ports of parts of it, and
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
| `skills/fdo-collective-overlap-playbook` | Choose and order the moves that hide collective latency, read `analyze`/`check-plan` output, and know the priority-collective rule. |

Each `SKILL.md` is short and links to `references/` for the detail. Nothing
parses `references/`: an agent reads a file only when a `SKILL.md` names it
(`agent_runner.py` tells it to start from the workflow skill and read the others
as needed), so every reference must be linked from its skill.

## Environment variables

Nothing is patched unless these are set. `skills/agent-assisted-scheduling-workflow` has the full procedure; this is
what each run needs.

| Variable | Set in | Meaning |
|---|---|---|
| `COLLECTIVE_AOT_MODE` | run 1: `dump`; run 2: `load` | `aot_executable.py`: `dump` writes the executable metadata header, `load` loads a bundle instead of compiling. `off` (default) does nothing. |
| `COLLECTIVE_AOT_PATH` | with the above | Directory to write the header to (run 1), or the bundle directory made by `cli aot-build` (run 2). |
| `COLLECTIVE_AOT_MODULE` | optional | Module the AOT hook handles. Default `jit_train_step`. |
| `COLLECTIVE_HLO_MODE` | run 1: `dump`; run 2 without a bundle: `load` | `module_exchange.py`: `dump` writes the scheduled module from the post-scheduler hook, `load` swaps in an optimized module's schedule while compiling. |
| `COLLECTIVE_HLO_PATH` | with the above | Directory to dump into, or the optimized module file to load. |
| `COLLECTIVE_DUMP_MODULE_RE`, `COLLECTIVE_DUMP_MIN_BYTES` | optional, run 1 | Which modules `COLLECTIVE_HLO_MODE=dump` writes (default all with at least 10,000 bytes). |
| `JAX_ENABLE_PGLE` | run 1 only | `1` for the profile-guided recompile that gives the FDO profile; leave it unset in run 2 and the baseline. |
| `XLA_FLAGS` | run 1 | Keep your usual flags and add `--xla_dump_to=<dump dir> --xla_dump_hlo_pass_re=NO_PASS_DUMPS --xla_gpu_experimental_dump_fdo_profiles=true`. |
| `XLA_COMPILE_BIN` | `cli aot-build` | Path of the `xla_compile` binary (`build-jax.sh --no-install --extra-targets=@xla//xla/service:xla_compile --extra-target-dest=<dir>` writes `<dir>/bin/xla_compile`). `--xla-compile` overrides it. |
| `PYTHONPATH` | every `cli` command and training | `<maxtext>/src` |

Typical use, with `DUMP`, `OPT` and `BUNDLE` directories:

```
# run 1: compile normally and dump
JAX_ENABLE_PGLE=1 COLLECTIVE_HLO_MODE=dump COLLECTIVE_HLO_PATH=$DUMP COLLECTIVE_AOT_MODE=dump COLLECTIVE_AOT_PATH=$DUMP \
  XLA_FLAGS="$XLA_FLAGS --xla_dump_to=$DUMP --xla_dump_hlo_pass_re=NO_PASS_DUMPS --xla_gpu_experimental_dump_fdo_profiles=true" \
  python -m maxtext.trainers.pre_train.train ...
# offline: optimize, then build the bundle
python -m maxtext.utils.agent_assisted_scheduling.cli optimize-dump $DUMP --out-module $OPT/optimized.hlo_module.pb
python -m maxtext.utils.agent_assisted_scheduling.cli aot-build $DUMP --optimized-module $OPT/optimized.hlo_module.pb --out-dir $BUNDLE
# run 2: load the bundle, no PGLE
COLLECTIVE_AOT_MODE=load COLLECTIVE_AOT_PATH=$BUNDLE python -m maxtext.trainers.pre_train.train ...
```

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
distributed KV store; see `agent_pass.py` and `kv_share.py`).

## Library

Jax-free Python the agent (or a CLI) calls instead of reasoning about legality:

| Module | Provides |
|---|---|
| `schedule_model.py` | `Inst`, `Computation`, `Module`, `AsyncRole`: a name-keyed model; a schedule is a `list[str]`; `Module.owner` maps every instruction to its one computation |
| `legality.py` | `validate_plan` (whole module; **strictly no cross-computation moves**, `CROSS_COMPUTATION`), `validate_order` / `is_legal` (one computation): exactly-once membership, operand and control order, async pairing, device-group overlap (opt-in, only relevant with collective multi-streaming; no cap on in-flight collectives), in-place reorder vs a baseline, and a warning when a start/done pair would become synchronous |
| `placement.py` | `earliest_legal_pos`, `hoist_start_to_floor`, `reinsert` (ports of `_earliest_legal_pos` / `_reinsert`) |
| `priority.py` | `is_priority`, `priority_regressions`, `collective_measures`: per-collective exposed / reachable / summed-stall comparison between two orders; `check_plan` turns a regression of an all-gather, reduce-scatter or all-reduce into `PRIORITY_REGRESSION` (`--priority-prefixes`, `--allow-priority-regression`) |
| `optimize.py` | bounded local search (window by window, array-scored; seconds per loop body): relocate an instruction with its dependency closure, keep only moves that are legal, add no synchronous conversion, regress no priority collective and lower `static + 0.3 * stall`; `cli optimize` wraps it |
| `cost.py` | `collective_exposures`, `total_exposed_us`, `coverage_gaps` (ports of `_total_exposed_us`) |
| `simulate.py` | `simulate_comm_stream` with SM gating (port of `_simulate_comm_stream`) |
| `profile.py` | `Profile`: FDO costs/latencies decoded from the text dump XLA writes (`<module>.fdo_profile`, `--xla_gpu_experimental_dump_fdo_profiles` + `--xla_dump_to`), the binary proto, or `module_config` field 31; no protobuf/protoc/tsl needed |
| `hlo_text.py` | pure-string parsers for what the binding only exposes via `to_string()`: control predecessors, callees, custom-call targets, replica groups (explicit, iota and mesh-axis forms) |
| `plan.py` | `Plan`: the document an agent hands to the pass. Canonical (deterministic) JSON with the full order of each computation to change, plus identity: module name, `input_checksum` (SHA-256 of XLA's schedule by names and order, not proto bytes, which differ per compile) and the profile digest. `verify_against` rejects a plan made for another module or a changed schedule; a profile mismatch is a warning (error with `--strict-profile`) |
| `analysis.py` | `analyze_module` (ranked exposed collectives: latency, overlap, deficit, stream-sim stall, whether the start can be hoisted and what pins it) and `check_plan` (validate a plan, check priority regressions, then before/after exposure per computation); rows carry `priority` and `reachable_deficit_us` |
| `cli.py` | `python -m maxtext.utils.agent_assisted_scheduling.cli analyze MODULE --profile P` and `check-plan MODULE --profile P --plan plan.json [--require-identity]` and `stamp-plan ... --out stamped.json` (checks the plan, then adds the identity fields); text or `--json`; exit 0 ok, 1 plan rejected, 2 bad input. MODULE is the `.hlo.pb` from `--xla_dump_hlo_as_proto` (text dumps of real scheduled modules do not re-parse) |
| `hlo_adapter.py` | `load_module(hlo_module, profile)` -> `LoadedModule` (model + baseline orders); `apply_plan(loaded, plan)` validates, applies to a copy of the schedule, runs XLA `update()`/`verify()`, and commits with `set_schedule` only if everything holds |

Tests: `tests/unit/agent_assisted_scheduling_test.py`. The repo `conftest.py`
skips unmarked tests on accelerator hosts, so on a GPU machine run it directly:
`PYTHONPATH=src:. python3 tests/unit/agent_assisted_scheduling_test.py`.

## Status

Skills and the library are written. The adapter from the HLO binding and FDO
profile, the plan format and the plan-applying pass are not; the workflow skill
marks where they plug in. Facts in these skills are taken from reading the sources, not
from running a compile; confirm a rule against the source before relying on it
for a change that ships.
