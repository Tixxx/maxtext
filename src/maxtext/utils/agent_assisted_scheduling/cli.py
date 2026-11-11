"""Command line for agent-assisted scheduling: analyze a dumped module, check a plan.

  python -m maxtext.utils.agent_assisted_scheduling.cli analyze MODULE --profile P
  python -m maxtext.utils.agent_assisted_scheduling.cli check-plan MODULE --profile P --plan plan.json
  python -m maxtext.utils.agent_assisted_scheduling.cli optimize MODULE --profile P --out plan.json --time-budget-s 120
  python -m maxtext.utils.agent_assisted_scheduling.cli optimize-dump DUMP_DIR --out-module optimized.hlo_module.pb
  python -m maxtext.utils.agent_assisted_scheduling.cli aot-build DUMP_DIR --optimized-module optimized.hlo_module.pb --out-dir BUNDLE
  python -m maxtext.utils.agent_assisted_scheduling.cli aot-check BUNDLE

MODULE is a scheduled module: a serialized HloModuleProto or the HloProto XLA
writes with --xla_dump_hlo_as_proto (`*_gpu_after_optimizations.hlo.pb`), or HLO
text that the parser accepts (a text dump of a real scheduled module usually does
not re-parse, because of forward-referenced control predecessors; use the proto).
P is an FDO profile, text or binary; with --xla_dump_to and
--xla_gpu_experimental_dump_fdo_profiles XLA writes it as text next to the dumps.

PLAN is JSON (see plan.py): {"computations": {"<computation>": ["<instruction>",
...]}} with the full proposed order of each computation to change, optionally
stamped with the module name, a checksum of XLA's input schedule and the profile
identity (`stamp-plan` adds them). Instructions may not leave their computation.
Nothing here modifies a module: `check-plan` is a dry run.

A legal plan is still rejected (PRIORITY_REGRESSION) if it makes a priority
collective (all-gather, reduce-scatter, all-reduce by default; --priority-prefixes)
more exposed, raises its simulated stall, or shrinks the overlap it could reach
by sinking its done to its first consumer, even when the total improves. --allow-priority-regression downgrades that to a warning.

`optimize` runs the bounded local search of optimize.py on each computation with
exposed collectives (largest first, the budget split between them), checks the
result like `check-plan` and writes a stamped plan (--out) and/or applies it and
writes the optimized module (--out-module). `optimize-dump` does the same for the
profiled module that the COLLECTIVE_HLO_MODE=dump transform (module_exchange.py)
wrote into a dump directory; load the result in a later run with
COLLECTIVE_HLO_MODE=load. Use it as the first batch of
moves or as a baseline for an agent to improve.

`aot-build` goes one step further, so the later run does not even compile: it runs `xla_compile` on the optimized
module (no GPU needed) and writes a bundle of the executable, the metadata header run 1 saved with
COLLECTIVE_AOT_MODE=dump, and a manifest (aot_bundle.py). Load it with COLLECTIVE_AOT_MODE=load
COLLECTIVE_AOT_PATH=BUNDLE (aot_executable.py); `aot-check` says whether a bundle is usable.

Exit status: 0 on success, 1 if `check-plan` rejects the plan, 2 on bad input.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from typing import Optional, Sequence

from maxtext.utils.agent_assisted_scheduling.analysis import CollectiveRow, ModuleReport, PlanReport, analyze_module, check_plan
from maxtext.utils.agent_assisted_scheduling.plan import (
    IdentityOptions,
    Plan,
    PlanFormatError,
    input_checksum,
    profile_identity,
)
from maxtext.utils.agent_assisted_scheduling.priority import DEFAULT_PRIORITY_PREFIXES
from maxtext.utils.agent_assisted_scheduling.profile import Profile, iter_fields


class InputError(Exception):
  """Bad file or argument; reported to the user without a traceback."""


def load_hlo_module(path: str):
  """Read a module file into a jaxlib HloModule (proto, HloProto, or HLO text)."""
  try:
    from jax._src.lib import hlo  # pylint: disable=import-outside-toplevel
  except ImportError as exc:
    raise InputError(f"jaxlib is required to read modules: {exc}") from exc
  try:
    with open(path, "rb") as f:
      data = f.read()
  except OSError as exc:
    raise InputError(f"cannot read {path}: {exc}") from exc
  if path.endswith((".txt", ".hlo")):
    try:
      return hlo.hlo_module_from_text(data.decode("utf-8"))
    except Exception as exc:  # pylint: disable=broad-exception-caught
      raise InputError(
          f"{path} is not parseable HLO text ({str(exc).splitlines()[0]}); "
          "dump with --xla_dump_hlo_as_proto=true and pass the .hlo.pb instead"
      ) from exc
  try:
    # An HloProto wraps the module as field 1; a bare HloModuleProto starts with the
    # module name (a short string), so an inner message starting with a field tag
    # (0x0a) means "wrapped".
    first = next(((n, v) for n, w, v in iter_fields(data) if n == 1 and w == 2), None)
    if first is not None and first[1][:1] == b"\x0a":
      data = first[1]
    return hlo.HloModule.from_serialized_hlo_module_proto(data)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise InputError(f"{path} is not a serialized HloModuleProto/HloProto: {exc}") from exc


def load_profile(path: Optional[str]) -> Profile:
  """FDO profile from a file, or an empty profile (nothing is profiled) if none given."""
  if not path:
    return Profile()
  try:
    return Profile.from_file(path)
  except (OSError, ValueError) as exc:
    raise InputError(f"cannot read profile {path}: {exc}") from exc


def load_plan(path: str) -> Plan:
  """Plan JSON (`plan.Plan`): bare {"computations": ...} or stamped with identity."""
  try:
    return Plan.load(path)
  except (OSError, PlanFormatError) as exc:
    raise InputError(f"cannot read plan {path}: {exc}") from exc


# ---- rendering ------------------------------------------------------------


def _us(v: Optional[float]) -> str:
  return "-" if v is None else f"{v:,.1f}"


def _row_line(rank: int, comp: str, r: CollectiveRow, with_comp: bool) -> str:
  """One table line for a ranked collective."""
  hoist = f"{r.hoist_to}" if r.can_hoist else "pinned"
  why = ""
  if not r.can_hoist and (r.blocker or r.control_pins):
    why = f" by {r.blocker or r.control_pins[0]}"
  tag = "" if r.profiled else "  [no profile entry]"
  cols = [
      f"{rank:>3}",
      "P" if r.priority else " ",
      f"{r.start:<28}",
      f"{r.key or '-':<24}",
      f"{_us(r.latency_us):>12}",
      f"{r.overlap_us:>12,.1f}",
      f"{r.deficit_us:>11,.1f}",
      f"{r.sim_stall_us:>11,.1f}",
      f"{r.start_pos:>5}->{r.done_pos:<5}",
      f"{hoist:>6}{why}{tag}",
  ]
  if with_comp:
    cols.insert(2, f"{comp:<22}")
  return " ".join(cols)


def render_report(report: ModuleReport, top: int, min_deficit_us: float) -> str:
  """Human-readable ranked table."""
  lines = [
      f"module {report.module}: {len(report.computations)} computation(s) with async collectives",
      f"static exposed {report.static_exposed_us:,.1f} us (priority {report.priority_exposed_us:,.1f} us) | "
      f"stream-sim stall {report.sim_stall_us:,.1f} us",
      "",
  ]
  for c in report.computations:
    kind = "while body" if c.while_body else "computation"
    unprofiled = sum(1 for r in c.rows if not r.profiled)
    lines.append(
        f"{kind} {c.name}: {len(c.rows)} collective(s), {unprofiled} unprofiled, "
        f"static exposed {c.static_exposed_us:,.1f} us (priority {c.priority_exposed_us:,.1f} us), "
        f"stream-sim {c.sim_stall_us:,.1f} us"
    )
  ranked = report.ranked(min_deficit_us)
  lines += ["", f"ranked collectives (top {min(top, len(ranked))} of {len(ranked)}; unprofiled last):"]
  head = [
      "  #",
      "P",
      "computation",
      "start",
      "profile key",
      "latency_us",
      "overlap_us",
      "deficit_us",
      "sim_stall",
      "pos",
      "hoist_to",
  ]
  lines.append(
      f"{head[0]} {head[1]} {head[2]:<22} {head[3]:<28} {head[4]:<24} {head[5]:>12} {head[6]:>12} {head[7]:>11} "
      f"{head[8]:>11} {head[9]:<11} {head[10]}"
  )
  lines.append("(P = priority collective: must not get worse in any plan)")
  for i, (comp, r) in enumerate(ranked[:top], 1):
    lines.append(_row_line(i, comp, r, with_comp=True))
  for comp, start, latency, median in report.suspicious_latencies():
    lines += [
        "",
        f"warning: {comp}/{start} latency {latency:,.0f} us is {latency / median:,.0f}x the median of its peers "
        f"({median:,.0f} us): likely cross-rank skew in the profile, not exposure a reorder can fix",
    ]
  gaps = [(c.name, g) for c in report.computations for g in c.gaps]
  if gaps:
    lines += ["", "coverage gaps (their cost is treated as free; do not trust a low exposed number):"]
    lines += [f"  {n}: {g}" for n, g in gaps[:20]]
    if len(gaps) > 20:
      lines.append(f"  ... {len(gaps) - 20} more")
  return "\n".join(lines)


def render_plan_report(rep: PlanReport) -> str:
  """Human-readable result of a plan check."""
  lines = ["plan ACCEPTED" if rep.accepted else "plan REJECTED"]
  for v in rep.violations:
    where = f" [{v.instruction}]" if v.instruction else ""
    lines.append(f"  {v.severity.value.upper()} {v.code}{where}: {v.message}")
  if rep.after is not None:
    lines.append("")
    for d in rep.deltas:
      lines.append(
          f"  {d.name}: static exposed {d.static_before_us:,.1f} -> {d.static_after_us:,.1f} us, "
          f"stream-sim {d.sim_before_us:,.1f} -> {d.sim_after_us:,.1f} us"
      )
    lines.append(
        f"  module: static exposed {rep.before.static_exposed_us:,.1f} -> {rep.after.static_exposed_us:,.1f} us, "
        f"stream-sim {rep.before.sim_stall_us:,.1f} -> {rep.after.sim_stall_us:,.1f} us"
    )
  return "\n".join(lines)


def _to_jsonable(obj):
  """Dataclasses (with their derived properties), enums and sequences as JSON-able values."""
  if dataclasses.is_dataclass(obj):
    return {f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)} | {
        k: getattr(obj, k)
        for k in ("static_exposed_us", "sim_stall_us", "accepted", "can_hoist", "priority_exposed_us")
        if hasattr(type(obj), k)
    }
  if isinstance(obj, (list, tuple)):
    return [_to_jsonable(x) for x in obj]
  if hasattr(obj, "value") and hasattr(obj, "name") and not isinstance(obj, str):  # enum
    return obj.value
  return obj


# ---- commands -------------------------------------------------------------


def _load(args):
  """Module and profile named by the command line, loaded into the library model."""
  # Imported here so `--help` and the pure helpers work without jaxlib.
  from maxtext.utils.agent_assisted_scheduling.hlo_adapter import load_module  # pylint: disable=import-outside-toplevel

  profile = load_profile(args.profile)
  hlo_module = load_hlo_module(args.module)
  try:
    loaded = load_module(hlo_module, profile, sm_gate_ops=args.sm_gate_ops)
  except ValueError as exc:
    raise InputError(str(exc)) from exc
  return loaded, profile


def _prefixes(args) -> tuple[str, ...]:
  """Priority prefixes from --priority-prefixes (comma-separated), else the default."""
  raw = getattr(args, "priority_prefixes", None)
  return tuple(p.strip() for p in raw.split(",") if p.strip()) if raw else DEFAULT_PRIORITY_PREFIXES


def _cmd_analyze(args, out) -> int:
  """`analyze`: ranked exposed-collective report."""
  loaded, profile = _load(args)
  only = set(args.computation) if args.computation else None
  if only:
    unknown = only - set(loaded.module.computations)
    if unknown:
      raise InputError(f"unknown computation(s): {', '.join(sorted(unknown))}")
  report = analyze_module(
      loaded.module,
      loaded.baseline,
      while_bodies=loaded.while_bodies,
      only=only,
      sm_gate=not args.no_sm_gate,
      priority_prefixes=_prefixes(args),
  )
  if args.json:
    ranked = [{"computation": c, **_to_jsonable(r)} for c, r in report.ranked(args.min_deficit_us)[: args.top]]
    doc = {
        "module": report.module,
        "profile_entries": len(profile.costs),
        "profile": profile_identity(profile),
        "input_checksum": input_checksum(loaded.baseline),
        "report": _to_jsonable(report),
        "suspicious_latencies": [
            {"computation": c, "start": st, "latency_us": lat, "median_us": med}
            for c, st, lat, med in report.suspicious_latencies()
        ],
        "ranked": ranked,
    }
    json.dump(doc, out, indent=1)
    out.write("\n")
  else:
    if not profile.costs:
      out.write("warning: no profile given or it is empty; every latency is unknown, nothing can be ranked\n\n")
    out.write(render_report(report, args.top, args.min_deficit_us) + "\n")
  return 0


def _cmd_check_plan(args, out) -> int:
  """`check-plan`: dry-run validation and before/after exposure of a plan."""
  loaded, profile = _load(args)
  plan = load_plan(args.plan)
  rep = check_plan(
      loaded.module,
      loaded.baseline,
      plan,
      profile=profile if args.profile else None,
      identity=IdentityOptions(require_identity=args.require_identity, strict_profile=args.strict_profile),
      while_bodies=loaded.while_bodies,
      check_device_groups=args.device_groups,
      sm_gate=not args.no_sm_gate,
      priority_prefixes=_prefixes(args),
      allow_priority_regression=args.allow_priority_regression,
  )
  if args.json:
    doc = _to_jsonable(rep)
    doc["plan_digest"] = plan.digest()
    json.dump(doc, out, indent=1)
    out.write("\n")
  else:
    out.write(render_plan_report(rep) + f"\nplan digest {plan.digest()}\n")
  return 0 if rep.accepted else 1


def _cmd_stamp_plan(args, out) -> int:
  """`stamp-plan`: check a plan, then write it with module/checksum/profile identity."""
  loaded, profile = _load(args)
  plan = load_plan(args.plan)
  rep = check_plan(
      loaded.module,
      loaded.baseline,
      Plan(computations=plan.computations, moves=plan.moves),
      while_bodies=loaded.while_bodies,
      check_device_groups=args.device_groups,
      sm_gate=not args.no_sm_gate,
      priority_prefixes=_prefixes(args),
      allow_priority_regression=args.allow_priority_regression,
  )
  if not rep.accepted:
    out.write(render_plan_report(rep) + "\nnot stamped\n")
    return 1
  stamped = Plan.for_module(
      loaded.module, loaded.baseline, plan.computations, profile=profile if args.profile else None, moves=plan.moves
  )
  stamped.save(args.out)
  out.write(
      f"wrote {args.out}: module {stamped.module}, input_checksum {stamped.input_checksum}, digest {stamped.digest()}\n"
  )
  return 0


def _cmd_optimize(args, out) -> int:
  """`optimize`: local search per computation, then check and write a stamped plan."""
  from maxtext.utils.agent_assisted_scheduling.optimize import optimize_computation  # pylint: disable=import-outside-toplevel

  if not args.out and not args.out_module:
    raise InputError("optimize needs --out (plan) and/or --out-module (optimized module)")
  loaded, profile = _load(args)
  prefixes = _prefixes(args)
  report = analyze_module(
      loaded.module, loaded.baseline, while_bodies=loaded.while_bodies, sm_gate=not args.no_sm_gate, priority_prefixes=prefixes
  )
  todo = sorted((c for c in report.computations if c.static_exposed_us > 0), key=lambda c: -c.static_exposed_us)
  chosen: dict[str, list[str]] = {}
  for i, c in enumerate(todo):
    budget = args.time_budget_s / (len(todo) - i)  # unused time flows to the later ones
    res = optimize_computation(
        loaded.module.computations[c.name],
        loaded.baseline[c.name],
        prefixes=prefixes,
        sm_gate=not args.no_sm_gate,
        sim_weight=args.sim_weight,
        time_budget_s=budget,
        seed=args.seed,
    )
    args.time_budget_s -= res.seconds
    out.write(
        f"{c.name}: exposed {res.static_before_us:,.0f} -> {res.static_after_us:,.0f} us, "
        f"stall {res.stall_before_us:,.0f} -> {res.stall_after_us:,.0f} us, "
        f"{res.moves} moves, {res.seconds:.1f}s ({res.stopped})\n"
    )
    if res.order != loaded.baseline[c.name]:
      chosen[c.name] = res.order
  plan = Plan.for_module(loaded.module, loaded.baseline, chosen, profile=profile if args.profile else None)
  rep = check_plan(
      loaded.module,
      loaded.baseline,
      plan,
      while_bodies=loaded.while_bodies,
      sm_gate=not args.no_sm_gate,
      priority_prefixes=prefixes,
  )
  if not rep.accepted:
    out.write(render_plan_report(rep) + "\nnot written\n")
    return 1
  if args.out:
    plan.save(args.out)
    out.write(f"wrote {args.out}: {len(chosen)} computation(s), digest {plan.digest()}\n")
  if args.out_module:
    if not chosen:
      out.write("nothing to change: the search found no better order, no module written\n")
      return 0
    from maxtext.utils.agent_assisted_scheduling.hlo_adapter import apply_plan  # pylint: disable=import-outside-toplevel

    result = apply_plan(loaded, plan, identity=IdentityOptions(require_identity=True))
    if not result.applied or result.reason != "applied":
      out.write(f"plan not applied to the module: {result.reason}\n")
      return 1
    with open(args.out_module, "wb") as f:
      f.write(loaded.hlo_module.as_serialized_hlo_module_proto())
    out.write(f"wrote optimized module {args.out_module}\n")
  return 0


def _find_dumped_module(directory: str, name_re: str) -> tuple[str, str]:
  """(module file, profile file) of the latest compile of a module in a COLLECTIVE_HLO_MODE=dump directory.

  The profile is the compile's own `.fdo_profile.txt` if the transform wrote one,
  else the newest non-empty `<module name>.*.fdo_profile` that XLA itself dumps with
  --xla_gpu_experimental_dump_fdo_profiles (the transform is not handed the profile).
  """
  import glob  # pylint: disable=import-outside-toplevel
  import os  # pylint: disable=import-outside-toplevel
  import re  # pylint: disable=import-outside-toplevel

  latest: dict[str, tuple[int, str]] = {}
  for path in glob.glob(os.path.join(directory, "*.post_scheduler.*.hlo_module.pb")):
    m = re.match(r"(.+)\.post_scheduler\.(\d+)\.hlo_module\.pb$", os.path.basename(path))
    if m and re.search(name_re, m.group(1)) and int(m.group(2)) >= latest.get(m.group(1), (-1, ""))[0]:
      latest[m.group(1)] = (int(m.group(2)), path)
  if not latest:
    raise InputError(f"no dumped module matching {name_re!r} in {directory} (was COLLECTIVE_HLO_MODE=dump set?)")
  # The biggest matching module: the train step, not a helper jit that happens to match.
  name, (_, path) = max(latest.items(), key=lambda kv: os.path.getsize(kv[1][1]))
  sibling = path.replace(".hlo_module.pb", ".fdo_profile.txt")
  if os.path.exists(sibling) and os.path.getsize(sibling) > 0:
    return path, sibling
  xla_profiles = [
      f
      for f in glob.glob(os.path.join(directory, f"module_*.{name}.*fdo_profile"))
      if os.path.getsize(f) > 0
  ]
  if not xla_profiles:
    raise InputError(
        f"no FDO profile for {name} in {directory}: run with --pgle and "
        "--xla_gpu_experimental_dump_fdo_profiles=true (the launcher's --agent-dump sets it)"
    )
  return path, max(xla_profiles, key=os.path.getmtime)


def _cmd_optimize_dump(args, out) -> int:
  """`optimize-dump`: optimize the profiled module a dump directory holds."""
  args.module, args.profile = _find_dumped_module(args.dump_dir, args.module_name)
  out.write(f"module {args.module}\nprofile {args.profile}\n")
  if not args.out_module:
    raise InputError("optimize-dump needs --out-module")
  return _cmd_optimize(args, out)


def _cmd_aot_build(args, out) -> int:
  """`aot-build`: compile the optimized module of a dump directory into a bundle a later run can load."""
  import os  # pylint: disable=import-outside-toplevel
  import re  # pylint: disable=import-outside-toplevel

  from maxtext.utils.agent_assisted_scheduling import aot_bundle  # pylint: disable=import-outside-toplevel

  module_file, profile = _find_dumped_module(args.dump_dir, args.module_name)
  stem = re.match(r"(.+)\.post_scheduler\.\d+\.hlo_module\.pb$", os.path.basename(module_file))
  if not stem:
    raise InputError(f"cannot read the module name from {module_file}")
  try:
    manifest = aot_bundle.build(
        args.dump_dir, stem.group(1), args.optimized_module, args.out_dir, profile=profile, xla_compile=args.xla_compile
    )
  except aot_bundle.BundleError as exc:
    raise InputError(str(exc)) from exc
  out.write(f"bundle {args.out_dir}\n{manifest.to_json()}")
  return 0


def _cmd_aot_check(args, out) -> int:
  """`aot-check`: whether a bundle is safe to load; exit 0 if it is, 1 with the reason if not."""
  from maxtext.utils.agent_assisted_scheduling import aot_bundle  # pylint: disable=import-outside-toplevel

  try:
    manifest = aot_bundle.check(args.bundle_dir, args.module)
  except aot_bundle.BundleError as exc:
    out.write(f"not usable: {exc}\n")
    return 1
  out.write(f"ok: {manifest.module} ({manifest.num_parameters} parameters, {manifest.num_outputs} outputs)\n")
  return 0


def build_parser() -> argparse.ArgumentParser:
  """The argument parser (exposed for tests)."""
  p = argparse.ArgumentParser(prog="agent_assisted_scheduling", description=__doc__.split("\n\n", maxsplit=1)[0])
  sub = p.add_subparsers(dest="command", required=True)

  def common(sp):
    sp.add_argument("module", help="scheduled module: .hlo.pb / .pb / parseable HLO text")
    sp.add_argument("--profile", help="FDO profile file (text or binary)")
    sp.add_argument("--no-sm-gate", action="store_true", help="simulate without SM-saturating GEMM gating")
    sp.add_argument(
        "--sm-gate-ops",
        choices=("gemm", "all", "none"),
        default="gemm",
        help="which ops hold the SMs and delay a collective: GEMM/conv/attention kernels (default), any compute op, or none",
    )
    sp.add_argument("--json", action="store_true", help="machine-readable output")
    sp.add_argument(
        "--priority-prefixes",
        help="comma-separated profile-key prefixes of priority collectives (default: all-gather,reduce-scatter,all-reduce)",
    )

  a = sub.add_parser("analyze", help="rank exposed collectives")
  common(a)
  a.add_argument("--computation", action="append", help="restrict to this computation (repeatable)")
  a.add_argument("--top", type=int, default=25, help="rows to show (default 25)")
  a.add_argument("--min-deficit-us", type=float, default=0.0, help="hide profiled rows below this deficit")
  a.set_defaults(func=_cmd_analyze)

  c = sub.add_parser("check-plan", help="validate a plan and compare exposure before/after (dry run)")
  common(c)
  c.add_argument("--plan", required=True, help='JSON: {"computations": {name: [instruction, ...]}}')
  c.add_argument("--device-groups", action="store_true", help="also check device-group overlap (multi-streaming only)")
  c.add_argument("--require-identity", action="store_true", help="reject a plan without module name and input checksum")
  c.add_argument("--strict-profile", action="store_true", help="treat a profile mismatch as an error, not a warning")
  c.add_argument(
      "--allow-priority-regression", action="store_true", help="report priority/reachable regressions as warnings"
  )
  c.set_defaults(func=_cmd_check_plan)

  st = sub.add_parser("stamp-plan", help="validate a plan and write it with module/checksum/profile identity")
  common(st)
  st.add_argument("--plan", required=True, help="plan JSON to stamp")
  st.add_argument("--out", required=True, help="where to write the stamped plan")
  st.add_argument("--device-groups", action="store_true", help="also check device-group overlap (multi-streaming only)")
  st.add_argument(
      "--allow-priority-regression", action="store_true", help="report priority/reachable regressions as warnings"
  )
  st.set_defaults(func=_cmd_stamp_plan)

  o = sub.add_parser("optimize", help="local-search a better order per computation and write a stamped plan")
  common(o)
  o.add_argument("--out", help="where to write the stamped plan")
  o.add_argument("--out-module", help="apply the plan and write the optimized serialized HloModuleProto here")
  o.add_argument("--time-budget-s", type=float, default=120.0, help="wall clock for the whole search (default 120)")
  o.add_argument(
      "--sim-weight",
      type=float,
      default=0.3,
      help="weight of the simulated comm-stream stall next to the static exposed total (default 0.3)",
  )
  o.add_argument("--seed", type=int, default=0, help="search order seed (default 0)")
  o.set_defaults(func=_cmd_optimize)

  od = sub.add_parser("optimize-dump", help="optimize the profiled module in a COLLECTIVE_HLO_MODE=dump directory")
  od.add_argument("dump_dir", help="directory written by the module_exchange transform")
  od.add_argument("--module-name", default="train_step", help="regex for the module name (default: train_step)")
  od.add_argument("--out-module", required=True, help="where to write the optimized serialized HloModuleProto")
  od.add_argument("--out", help="also write the stamped plan here")
  od.add_argument("--no-sm-gate", action="store_true", help="simulate without SM-saturating GEMM gating")
  od.add_argument("--sm-gate-ops", choices=("gemm", "all", "none"), default="gemm")
  od.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
  od.add_argument("--priority-prefixes", help="comma-separated priority collective prefixes")
  od.add_argument("--time-budget-s", type=float, default=120.0, help="wall clock for the search (default 120)")
  od.add_argument("--sim-weight", type=float, default=0.3, help="weight of simulated stall (default 0.3)")
  od.add_argument("--seed", type=int, default=0, help="search order seed (default 0)")
  od.set_defaults(func=_cmd_optimize_dump)

  ab = sub.add_parser("aot-build", help="compile the optimized module into a bundle a later run loads without compiling")
  ab.add_argument("dump_dir", help="directory of run 1 (COLLECTIVE_HLO_MODE=dump and COLLECTIVE_AOT_MODE=dump)")
  ab.add_argument("--optimized-module", required=True, help="the file `optimize-dump --out-module` wrote")
  ab.add_argument("--out-dir", required=True, help="bundle directory to write")
  ab.add_argument("--module-name", default="train_step", help="regex for the module name (default: train_step)")
  ab.add_argument("--xla-compile", help="path of the xla_compile binary (default: $XLA_COMPILE_BIN, then PATH)")
  ab.set_defaults(func=_cmd_aot_build)

  ac = sub.add_parser("aot-check", help="whether a bundle is safe to load (exit 0 if so)")
  ac.add_argument("bundle_dir", help="bundle directory written by aot-build")
  ac.add_argument("--module", default="jit_train_step", help="module the bundle must be for (default: jit_train_step)")
  ac.set_defaults(func=_cmd_aot_check)
  return p


def main(argv: Optional[Sequence[str]] = None, out=None) -> int:
  """Entry point; returns the exit status."""
  out = out or sys.stdout
  args = build_parser().parse_args(argv)
  try:
    return args.func(args, out)
  except InputError as exc:
    sys.stderr.write(f"error: {exc}\n")
    return 2


if __name__ == "__main__":
  sys.exit(main())
