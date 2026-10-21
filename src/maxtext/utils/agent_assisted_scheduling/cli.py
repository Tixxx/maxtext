"""Command line for agent-assisted scheduling: analyze a dumped module, check a plan.

  python -m maxtext.utils.agent_assisted_scheduling.cli analyze MODULE --profile P
  python -m maxtext.utils.agent_assisted_scheduling.cli check-plan MODULE --profile P --plan plan.json

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
    cols.insert(1, f"{comp:<22}")
  return " ".join(cols)


def render_report(report: ModuleReport, top: int, min_deficit_us: float) -> str:
  """Human-readable ranked table."""
  lines = [
      f"module {report.module}: {len(report.computations)} computation(s) with async collectives",
      f"static exposed {report.static_exposed_us:,.1f} us | stream-sim stall {report.sim_stall_us:,.1f} us",
      "",
  ]
  for c in report.computations:
    kind = "while body" if c.while_body else "computation"
    unprofiled = sum(1 for r in c.rows if not r.profiled)
    lines.append(
        f"{kind} {c.name}: {len(c.rows)} collective(s), {unprofiled} unprofiled, "
        f"static exposed {c.static_exposed_us:,.1f} us, stream-sim {c.sim_stall_us:,.1f} us"
    )
  ranked = report.ranked(min_deficit_us)
  lines += ["", f"ranked collectives (top {min(top, len(ranked))} of {len(ranked)}; unprofiled last):"]
  head = [
      "  #",
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
      f"{head[0]} {head[1]:<22} {head[2]:<28} {head[3]:<24} {head[4]:>12} {head[5]:>12} {head[6]:>11} {head[7]:>11} "
      f"{head[8]:<11} {head[9]}"
  )
  for i, (comp, r) in enumerate(ranked[:top], 1):
    lines.append(_row_line(i, comp, r, with_comp=True))
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
        for k in ("static_exposed_us", "sim_stall_us", "accepted", "can_hoist")
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
    loaded = load_module(hlo_module, profile)
  except ValueError as exc:
    raise InputError(str(exc)) from exc
  return loaded, profile


def _cmd_analyze(args, out) -> int:
  """`analyze`: ranked exposed-collective report."""
  loaded, profile = _load(args)
  only = set(args.computation) if args.computation else None
  if only:
    unknown = only - set(loaded.module.computations)
    if unknown:
      raise InputError(f"unknown computation(s): {', '.join(sorted(unknown))}")
  report = analyze_module(
      loaded.module, loaded.baseline, while_bodies=loaded.while_bodies, only=only, sm_gate=not args.no_sm_gate
  )
  if args.json:
    ranked = [{"computation": c, **_to_jsonable(r)} for c, r in report.ranked(args.min_deficit_us)[: args.top]]
    doc = {
        "module": report.module,
        "profile_entries": len(profile.costs),
        "profile": profile_identity(profile),
        "input_checksum": input_checksum(loaded.baseline),
        "report": _to_jsonable(report),
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


def build_parser() -> argparse.ArgumentParser:
  """The argument parser (exposed for tests)."""
  p = argparse.ArgumentParser(prog="agent_assisted_scheduling", description=__doc__.split("\n\n", maxsplit=1)[0])
  sub = p.add_subparsers(dest="command", required=True)

  def common(sp):
    sp.add_argument("module", help="scheduled module: .hlo.pb / .pb / parseable HLO text")
    sp.add_argument("--profile", help="FDO profile file (text or binary)")
    sp.add_argument("--no-sm-gate", action="store_true", help="simulate without SM-saturating GEMM gating")
    sp.add_argument("--json", action="store_true", help="machine-readable output")

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
  c.set_defaults(func=_cmd_check_plan)

  st = sub.add_parser("stamp-plan", help="validate a plan and write it with module/checksum/profile identity")
  common(st)
  st.add_argument("--plan", required=True, help="plan JSON to stamp")
  st.add_argument("--out", required=True, help="where to write the stamped plan")
  st.add_argument("--device-groups", action="store_true", help="also check device-group overlap (multi-streaming only)")
  st.set_defaults(func=_cmd_stamp_plan)
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
