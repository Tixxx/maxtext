---
name: agent-assisted-scheduling-workflow
description: End-to-end loop for agent-assisted XLA:GPU scheduling - analyze exposed collectives from an FDO profile, propose reorderings, validate legality, simulate, and emit a schedule plan that a registered POST_SCHEDULER pass applies with fallback to XLA's schedule. Start here for any task about improving collective/compute overlap with an agent.
---

# Agent-assisted scheduling workflow

Related skills: `xla-gpu-scheduling-heuristics` (what XLA did),
`xla-hlo-schedule-legality` (what is allowed), `fdo-collective-overlap-playbook`
(how to improve it).

## Principles

1. **The agent proposes; deterministic code disposes.** Legality and cost come
   from a validator/simulator, never from the model's reasoning alone.
2. **Fail closed.** Any validation failure discards the entire plan and keeps
   XLA's own schedule.
3. **Compute once.** One rank (or an offline run) produces the plan; every rank
   applies identical bytes. No LLM call and no collectives inside the compiler
   callback.
4. **Prefer smaller moves.** Re-measure after each batch; PGLE data is noisy.

## Loop

1. **Collect.** Dump the scheduled module (`module.to_string()` of the module
   passed to the transform, post-LHS) and the FDO profile from
   `module_config` (`fdo_profile`). Some PJRT plugins pass empty `module_config`
   bytes; then read the profile XLA dumps with `--xla_dump_to=<dir>
   --xla_gpu_experimental_dump_fdo_profiles`, which writes
   `<module>.fdo_profile` as text. Record the module name and a stable key
   (instruction names are unique per module; do not hash the proto, a unique-id
   field differs per compile).
2. **Analyze.** Run `python -m maxtext.utils.agent_assisted_scheduling.cli analyze
   <module>.hlo.pb --profile <module>.fdo_profile --json` (dump with
   `--xla_dump_to`, `--xla_dump_hlo_as_proto=true`,
   `--xla_gpu_experimental_dump_fdo_profiles=true`; take the
   `*_gpu_after_optimizations.hlo.pb`). Per computation (innermost first): for each async start/done
   pair, compute latency, overlap, deficit; list blockers, heavy compute
   candidates, async-resource contention (one comm stream), and host-offload
   copies. Output a ranked exposed-collective table.
3. **Propose.** Pick moves from the playbook. Express each as data, not prose,
   e.g. `{computation, move: <instruction>, to_before: <instruction>}`, with the
   playbook step and expected exposed-time change.
4. **Validate.** Check the full proposed order against the legality checklist.
   Reject, do not repair. `cli check-plan <module>.hlo.pb --profile ... --plan
   plan.json` does validation and step 5 in one dry run; a rejected plan exits 1.
5. **Simulate.** Estimate total exposed time with the comm-stream model (SM
   gating on) for before and after. Keep a move only if the total improves
   by more than noise (tolerance ~250 us in the existing pass).
6. **Emit plan.** Write `{"computations": {"<computation>": [full order]}}` for
   only the computations that change (optional `moves` notes), then run `cli
   stamp-plan <module>.hlo.pb --profile ... --plan plan.json --out
   stamped.json`. It refuses an illegal plan and otherwise adds module name,
   `input_checksum` and the profile digest (`plan.py`). The stamped file is
   canonical JSON: every rank reading it sees identical bytes, and
   `Plan.digest()` is what ranks compare.
7. **Apply (the pass).** `hlo_adapter.apply_plan(loaded, Plan.load(path),
   identity=IdentityOptions(require_identity=True))`: checks name and checksum
   against the module being compiled, validates, sets the sequences,
   `schedule.update()`, `schedule.verify()`, re-checks device-group overlap if
   multi-streaming is on; on any mismatch it leaves the module untouched and the
   pass returns `None`.
8. **Evaluate.** Run the real benchmark and compare step time with and without
   the plan, keeping XLA's schedule as the baseline. A better simulated total is
   a hypothesis, not a result.

## Status of the building blocks

| Piece | State |
|---|---|
| Heuristic + legality knowledge | these skills |
| Profile-guided reorderer | `gpu_collective_overlap_pass.py` (works as a `POST_SCHEDULER` pass) |
| Jax-free validator / cost / simulator library | **written**: `schedule_model.py`, `legality.py`, `placement.py`, `cost.py`, `simulate.py` in this directory, tested by `tests/unit/agent_assisted_scheduling_test.py` |
| Adapter from `jax._src.lib.hlo.HloModule` + FDO profile to `schedule_model.Module` (`hlo_adapter.load_module`), and writing a validated plan back (`hlo_adapter.apply_plan`) | **written**, tested on a handwritten module and a real 8-GPU compiled module; the profile loads from the text dump XLA writes (`profile.py`) |
| CLI for the agent | **written**: `cli.py` (`analyze`, `check-plan`; dry run, never modifies a module) |
| Plan format (`plan.py`: stamped canonical JSON, identity checks) | **written**, enforced by `check-plan`/`stamp-plan` and `apply_plan` |
| Plan-applying `POST_SCHEDULER` pass | **not written** (`apply_plan` is its core) |
| Replacing LHS | not possible with the current hooks: transforms run only at `kPreScheduler` / `kPostScheduler` (`xla/service/gpu/gpu_compiler.cc`), after LHS has run. Options: disable LHS with `xla_gpu_enable_latency_hiding_scheduler=false` and let the plan own the schedule (loses LHS memory/resource handling, which the validator must then provide), or add a scheduler-replacement hook in XLA. |

## Things the agent must not do

- Edit or reorder across control edges, or move an instruction to a different
  computation. Strictly never: `validate_plan` rejects it as `CROSS_COMPUTATION`.
- Put a start immediately before its done (it becomes synchronous).
- With collective multi-streaming on, overlap collectives whose device groups share more than one device.
- Move an in-place writer relative to readers of the old buffer, or a D2H start
  past other users of its host buffer.
- Apply a plan to a module whose instruction names or order differ from the one
  it was made for.
- Treat a missing profile entry as evidence that an op is cheap.
- Run anything that issues a collective from inside the compile callback.
