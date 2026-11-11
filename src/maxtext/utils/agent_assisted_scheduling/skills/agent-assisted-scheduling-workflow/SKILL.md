---
name: agent-assisted-scheduling-workflow
description: End-to-end procedure for agent-assisted XLA:GPU scheduling of a MaxText model, from scratch - run 1 (normal compile that dumps the scheduled HLO, FDO profile and executable header), the offline agent loop (analyze exposed collectives, propose reorderings, validate legality and priority-collective regressions, emit a stamped plan, build an executable bundle), and run 2 (load the optimized executable instead of compiling, with fallback to XLA's schedule). Start here for any task about improving collective/compute overlap with an agent.
---

# Agent-assisted scheduling workflow

Related skills: `xla-gpu-scheduling-heuristics` (what XLA did),
`xla-hlo-schedule-legality` (what is allowed), `fdo-collective-overlap-playbook`
(how to improve it).

## Run it from scratch

Three steps. Run 1 and run 2 need the GPUs; step 2 is offline.
Use the same model, MaxText config, device count and GPU type in both runs. `CLI` below is
`python -m maxtext.utils.agent_assisted_scheduling.cli`; every command here needs `export PYTHONPATH=<maxtext>/src`
in the same shell. Typical times for llama3-8b on 8 GPUs: run 1, `aot-build` and run 2 about 1 to 2 minutes each,
`optimize-dump` under a minute.

**Prerequisites (once)**
- The `xla_compile` binary (only `aot-build` needs it). Build it with `build-jax.sh --no-install --sm local
  --extra-targets=@xla//xla/service:xla_compile --extra-target-dest=<dir>`; it needs no `--src-path-*` flags in the
  container. It writes `<dir>/bin/xla_compile`. A warm bazel cache takes about 5 minutes, a cold one much longer.
  Run it as a plain background command (not with `nohup`, `sleep` or `&` in one line). Then
  `export XLA_COMPILE_BIN=<dir>/bin/xla_compile` or pass `--xla-compile` to `aot-build`. It must come from the same XLA
  build as the installed jaxlib.
- Pick a dump directory `DUMP`, an optimized-module directory `OPT` and a bundle directory `BUNDLE`; all must be
  readable by every later step (and by every rank, for `BUNDLE`).

**The training command.** Run from `<maxtext>` with `PYTHONPATH=<maxtext>/src`. Define it once and reuse it for every run:

```
python -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml model_name=llama3-8b hardware=gpu \
  attention=cudnn_flash_jax dataset_type=synthetic enable_checkpointing=false steps=20 \
  per_device_batch_size=2 max_target_length=4096 base_output_directory=<out> run_name=<name>
```
- `hardware=gpu` is required (the config defaults to TPU). `steps` and `per_device_batch_size` default to very large
  values, so set both for a short run.
- `attention=cudnn_flash_jax` is for machines where the TE attention (`cudnn_flash_te`) is unavailable.
- One process drives all GPUs of a node, so each log line below appears once per process, not once per GPU.
- The container's `XLA_FLAGS` may already hold flags (for example `--xla_gpu_enable_latency_hiding_scheduler=true`).
  Keep them: `BASE_XLA="$XLA_FLAGS"` and append to it. Use identical `BASE_XLA` in run 1, run 2 and the baseline.

**Step 1: normal compile (run 1)**
Run the training command with PGLE on (the profile is what the agent optimizes against) and these added:

```
JAX_ENABLE_PGLE=1
COLLECTIVE_HLO_MODE=dump   COLLECTIVE_HLO_PATH=$DUMP
COLLECTIVE_AOT_MODE=dump   COLLECTIVE_AOT_PATH=$DUMP
XLA_FLAGS="$BASE_XLA --xla_dump_to=$DUMP --xla_dump_hlo_pass_re=NO_PASS_DUMPS --xla_gpu_experimental_dump_fdo_profiles=true"
```
- Dump on every rank (XLA flags are identical on all ranks). Dumping on rank 0 only hangs PGLE recompiles.
- Use at least 10 steps: PGLE profiles a few steps and then recompiles with the profile; the recompile is what the agent
  optimizes. 20 steps worked.
- Both hooks are registered in `train.py` and do nothing unless these variables are set (`module_exchange.py`, `aot_executable.py`).
- The run logs contain binary bytes: use `grep -a` on them.

Done when `$DUMP` holds, for the train step (`jit_train_step`):
`jit_train_step.post_scheduler.<n>.hlo_module.pb` (the highest `n` is the profiled recompile),
`module_<id>.jit_train_step.config.pbtxt`, `module_<id>.jit_train_step.autotune_results.pbtxt`, `jit_train_step.aot_header`
and the FDO profile. The dump also holds roughly a hundred `.fdo_profile` files for many modules, and many are empty.
Two ids exist for `jit_train_step`: the first compile (empty profile) and the profiled recompile. The profile to use is
the non-empty `module_<id>.jit_train_step.sm_*_gpu_after_optimizations.fdo_profile` with the highest id.
If the header is missing, `COLLECTIVE_AOT_MODE=dump` was not set in run 1: rerun it.

**Step 2: agent flow on the dumped HLO (offline)**
1. Analyze and optimize with the loop below (the next section). Quick path: `CLI optimize-dump $DUMP --out-module
   $OPT/optimized.hlo_module.pb --out $OPT/optimized.hlo_module.plan.json --time-budget-s 120` picks the profiled
   module and its profile itself, runs the bounded local search, checks the result like `check-plan`, and writes the
   optimized module and the stamped plan. A non-zero exit, or no output module, means the plan was rejected; keep XLA's
   schedule. It prints the exposed time before and after for each computation it changed. An agent can start from that
   plan or write its own with the playbook.
2. To check a plan yourself (an agent-written or edited one), run `CLI check-plan
   $DUMP/jit_train_step.post_scheduler.<n>.hlo_module.pb --profile <the profile above> --plan <plan.json>` with the
   highest `n`, then `stamp-plan`. Exit 0 means accepted. Never build a bundle from a rejected plan.
3. Build the bundle: `CLI aot-build $DUMP --optimized-module
   $OPT/optimized.hlo_module.pb --out-dir $BUNDLE`. It runs `xla_compile` on the optimized module with run 1's
   compile config and autotune results, adds run 1's header and writes writes `manifest.json`. Only this command needs
   `XLA_COMPILE_BIN`. It needs no GPU: it compiles from run 1's `module_<id>.jit_train_step.gpu_target_config.pbtxt`
   (written by `--xla_dump_to`) and sets `xla_gpu_cudnn_deviceless_compilation_mode` to `AUTO` in the compile config.
4. `CLI aot-check $BUNDLE` exits 0 when the bundle is usable and prints the reason when it is not.

**What `aot-build` depends on, and how it fails**
- cuDNN 9.23 or newer: the deviceless cuDNN graph build requires it.
- An `xla_compile` that takes `--num_devices_per_host` (a local change to `xla/tools/xla_compile_lib.cc` and
  `xla/service/xla_compile_main.cc`, not upstream when written). The GPU topology decides which collectives count as
  local to a node; a stock `xla_compile` assumes one device per host, so no multi-device collective is local. Set
  `COLLECTIVE_AOT_DEVICES_PER_HOST` to the GPUs per node (`8`) for a multi-node job. Unset, all partitions are assumed
  to be on one node, which is right only for a single node. Use an `xla_compile` built from the same XLA tree as the
  jaxlib.
- If run 2 fails at its first step with `No serialized cuDNN graph with fingerprint ... __cudnn$fmhaSoftmax` (any model
  with cuDNN fused attention), the bundle was compiled with the cuDNN mode `DISABLED`: run 1's config says so, and a
  compile with no device then skips building the graphs. `aot-build` overrides it; a hand-run `xla_compile` must too.
- `aot-check` rejects a bundle built with a different jaxlib version. The version string is date-based, so rebuild after
  a jaxlib rebuild.

**Step 3: load the optimized module and run (run 2)**
Run the training command with the same arguments as run 1 (same `BASE_XLA`), except:

```
COLLECTIVE_AOT_MODE=load   COLLECTIVE_AOT_PATH=$BUNDLE       # no COLLECTIVE_*_MODE=dump; PGLE off (JAX_ENABLE_PGLE unset)
```
- The `--xla_dump_*` flags are optional in run 2: keep them or drop them, the bundle loads either way. Dropping them
  keeps `$DUMP` from filling up again.
- Success: the log shows `aot_executable [jit_train_step]: loaded the precompiled executable from $BUNDLE` (once per
  process). No HLO passes, scheduling or autotuning run for that module; the other jitted modules compile as usual.
- Fallback: any mismatch (manifest, hashes, jaxlib version, partition count, argument/result counts) logs
  `not using the bundle ... compiling normally` with the reason, and XLA compiles with its own schedule. The run still
  works, so check the log before trusting the step time.
- Without a bundle (no `xla_compile`): `COLLECTIVE_HLO_MODE=load COLLECTIVE_HLO_PATH=$OPT/optimized.hlo_module.pb`
  recompiles and swaps the schedule in the post-scheduler hook instead. It matches the module by name and structure
  and falls back to XLA's schedule if they differ.

**Evaluate.** Run a baseline with the same arguments and `XLA_FLAGS="$BASE_XLA"`, no `COLLECTIVE_*` variables and PGLE
off. In each log take the `completed step: N, seconds: X` lines of `metric_logger.py` (`grep -a`), use steps 10 and up
for the median of `seconds`, and the `loss:` of the last step. Loss differences within run-to-run spread are expected;
a larger one is a bug. If the step is compute-bound the gain is small even when exposed communication drops.

**Environment variables** (all unset by default, in which case nothing is patched):

| Variable | Used in | Meaning |
|---|---|---|
| `COLLECTIVE_HLO_MODE` | run 1 (`dump`), fallback run 2 (`load`) | `module_exchange.py` hook: `dump`, `load` or `off` |
| `COLLECTIVE_HLO_PATH` | with the above | directory to dump into, or the optimized module file to load |
| `COLLECTIVE_AOT_MODE` | run 1 (`dump`), run 2 (`load`) | `aot_executable.py` hook: `dump`, `load` or `off` |
| `COLLECTIVE_AOT_PATH` | with the above | directory to write the header to, or the bundle to load |
| `COLLECTIVE_AOT_MODULE` | optional | module the AOT hook handles, default `jit_train_step` |
| `JAX_ENABLE_PGLE` | run 1 only | profile-guided recompile that yields the profile |
| `XLA_COMPILE_BIN` | `aot-build` | path of the `xla_compile` binary |
| `COLLECTIVE_AOT_DEVICES_PER_HOST` | `aot-build` | GPUs per node (e.g. `8`); set it for a multi-node job, unset assumes one node |
| `PYTHONPATH` | every command | `<maxtext>/src` |

**Redo step 1 when** the model, config, sharding, device count, GPU type, jaxlib or XLA build changes. A bundle from the
old run is rejected or stale. Redo step 2 when only the optimizer, profile or plan changes.

## Principles

1. **The agent proposes; deterministic code disposes.** Legality and cost come
   from a validator/simulator, never from the model's reasoning alone.
2. **Fail closed.** Any validation failure discards the entire plan and keeps
   XLA's own schedule.
3. **Compute once.** One rank (or an offline run) produces the plan; every rank
   applies identical bytes. No LLM call and no collectives inside the compiler
   callback.
4. **Prefer smaller moves.** Re-measure after each batch; PGLE data is noisy.
5. **Priority collectives never regress.** All-gather, reduce-scatter and
   all-reduce (the weight and gradient collectives of sharded training; `--priority-prefixes` changes the set) must not get more exposed from any move,
   even if total exposed time improves. Check them individually.

## Loop (step 2 in detail)

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
   by more than noise (tolerance ~250 us) and no priority collective
   (principle 5) is worse. `check-plan` enforces this and rejects the plan with
   `PRIORITY_REGRESSION` errors naming the collective and the measure (exposed,
   reachable, or summed stall). A move that shrinks a collective's reachable
   window (for example moving a consumer of its done earlier) counts as a
   regression even if its current overlap is unchanged.
6. **Emit plan.** Write `{"computations": {"<computation>": [full order]}}` for
   only the computations that change (optional `moves` notes), then run `cli
   stamp-plan <module>.hlo.pb --profile ... --plan plan.json --out
   stamped.json`. It refuses an illegal or priority-regressing plan and otherwise adds module name,
   `input_checksum` and the profile digest (`plan.py`). The stamped file is
   canonical JSON: every rank reading it sees identical bytes, and
   `Plan.digest()` is what ranks compare.
7. **Apply.** `hlo_adapter.apply_plan(loaded, Plan.load(path),
   identity=IdentityOptions(require_identity=True))` checks name and checksum
   against the module being compiled, validates, sets the sequences, runs
   `schedule.update()` and `schedule.verify()`, and re-checks device-group
   overlap if multi-streaming is on. On any mismatch it leaves the module
   untouched and the caller keeps XLA's schedule.
8. **Evaluate.** Run the real benchmark and compare step time with and without
   the plan, keeping XLA's schedule as the baseline. A better simulated total is
   a hypothesis, not a result.

## Status of the building blocks

| Piece | State |
|---|---|
| Heuristic + legality knowledge | these skills |
| Jax-free validator / cost / simulator library | **written**: `schedule_model.py`, `legality.py`, `placement.py`, `cost.py`, `simulate.py` in this directory, `priority.py` (priority-collective regression rules), tested by `tests/unit/agent_assisted_scheduling_test.py` and, on a recorded loop body, `tests/unit/agent_assisted_scheduling_replay_test.py` |
| Adapter from `jax._src.lib.hlo.HloModule` + FDO profile to `schedule_model.Module` (`hlo_adapter.load_module`), and writing a validated plan back (`hlo_adapter.apply_plan`) | **written**, tested on a handwritten module and a real 8-GPU compiled module; the profile loads from the text dump XLA writes (`profile.py`) |
| CLI for the agent | **written**: `cli.py` (`analyze`, `check-plan`, `stamp-plan`, `optimize`, `optimize-dump`, `aot-build`, `aot-check`; the analysis commands never modify a module and reject priority-collective regressions) |
| Run 1 dump and run 2 load hooks | **written**: `module_exchange.py` (`COLLECTIVE_HLO_MODE`) and `aot_executable.py` / `aot_bundle.py` (`COLLECTIVE_AOT_MODE`, the executable bundle), registered in `train.py`; checked on 8 GPUs with llama3-8b |
| Plan format (`plan.py`: stamped canonical JSON, identity checks) | **written**, enforced by `check-plan`/`stamp-plan` and `apply_plan` |
| Plan applier hooked into the compiler (a `POST_SCHEDULER` transform) | `agent_pass.py` / `agent_runner.py` run the agent and apply the plan through `apply_plan` |
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
