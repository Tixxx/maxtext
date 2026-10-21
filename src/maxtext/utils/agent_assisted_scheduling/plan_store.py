"""Where plans are cached and recorded.

If XLA is dumping (`--xla_dump_to=<dir>`), plans are cached in and recorded to that
same directory, next to the HLO dumps they were made from:

  <dir>/<module>.agent_plan.<input_checksum[:16]>.<profile_sha[:16]>.json
  <dir>/<module>.agent_plan.<input_checksum[:16]>.<profile_sha[:16]>.agent_log.json

A plan is a cache hit only if its stamped identity (module name, input checksum,
profile digest) matches the module being compiled; the file name is just an
index. Without `xla_dump_to` plans are cached in memory for the life of the
process only (a module is compiled more than once per job) and nothing is
written. The plan JSON is canonical, so a recorded plan replays byte for byte.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from typing import Any, Optional

from maxtext.utils.agent_assisted_scheduling.plan import Plan, PlanFormatError
from maxtext.utils.agent_assisted_scheduling.profile import iter_fields

_DEBUG_OPTIONS_FIELD_IN_MODULE_CONFIG = 14  # HloModuleConfigProto.debug_options
_XLA_DUMP_TO_FIELD = 109  # DebugOptions.xla_dump_to
_DUMP_TO_FLAG_RE = re.compile(r"""--xla_dump_to(?:=|\s+)("[^"]*"|'[^']*'|\S+)""")


def _from_module_config(module_config: bytes) -> Optional[str]:
  try:
    for num, wire, value in iter_fields(module_config):
      if num == _DEBUG_OPTIONS_FIELD_IN_MODULE_CONFIG and wire == 2:
        for n, w, v in iter_fields(value):
          if n == _XLA_DUMP_TO_FIELD and w == 2:
            return v.decode("utf-8")
  except (ValueError, UnicodeDecodeError):
    return None
  return None


def _from_xla_flags(xla_flags: Optional[str]) -> Optional[str]:
  if not xla_flags:
    return None
  m = _DUMP_TO_FLAG_RE.search(xla_flags)
  if not m:
    return None
  parts = shlex.split(m.group(1))
  return parts[0] if parts else None


def dump_dir(module_config: bytes = b"", xla_flags: Optional[str] = None) -> Optional[str]:
  """The `xla_dump_to` directory in effect, or None if XLA is not dumping to a directory.

  Taken from the module's compile options when the plugin passes them, else from
  the `XLA_FLAGS` string. "-" and "stdout" (dump to stdout) mean no directory.
  """
  path = _from_module_config(module_config) or _from_xla_flags(xla_flags)
  if not path or path in ("-", "stdout", "stderr"):
    return None
  return path


def _safe(name: str) -> str:
  return re.sub(r"[^A-Za-z0-9_.\-]", "_", name)


def plan_stem(module: str, input_checksum: str, profile_sha: Optional[str]) -> str:
  """File-name stem shared by a plan and its agent log."""
  return f"{_safe(module)}.agent_plan.{input_checksum[:16]}.{(profile_sha or 'noprofile')[:16]}"


class PlanStore:
  """Memory cache of plans, mirrored to `directory` (the xla_dump_to dir) when set."""

  def __init__(self, directory: Optional[str] = None):
    self.directory = directory
    self._memory: dict[str, Plan] = {}

  def _path(self, stem: str, suffix: str = ".json") -> Optional[str]:
    return os.path.join(self.directory, stem + suffix) if self.directory else None

  def load(self, module: str, input_checksum: str, profile_sha: Optional[str]) -> Optional[Plan]:
    """A cached plan whose stamped identity matches, else None."""
    stem = plan_stem(module, input_checksum, profile_sha)
    plan = self._memory.get(stem)
    if plan is None:
      path = self._path(stem)
      if path and os.path.exists(path):
        try:
          plan = Plan.load(path)
        except (OSError, PlanFormatError):
          return None
    if plan is None:
      return None
    plan_sha = (plan.profile or {}).get("sha256")
    if plan.module != module or plan.input_checksum != input_checksum or plan_sha != profile_sha:
      return None
    self._memory[stem] = plan
    return plan

  def record(self, plan: Plan) -> Optional[str]:
    """Remember `plan` and, if a directory is set, write it there atomically. Returns the path."""
    if not plan.has_identity:
      raise ValueError("only a stamped plan (module + input_checksum) can be recorded")
    stem = plan_stem(plan.module, plan.input_checksum, (plan.profile or {}).get("sha256"))
    self._memory[stem] = plan
    path = self._path(stem)
    if path is None:
      return None
    os.makedirs(self.directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    plan.save(tmp)
    os.replace(tmp, path)
    return path

  def record_log(
      self, module: str, input_checksum: str, profile_sha: Optional[str], log: dict[str, Any]
  ) -> Optional[str]:
    """Write the agent run's log next to the plan (or where the plan would be)."""
    path = self._path(plan_stem(module, input_checksum, profile_sha), ".agent_log.json")
    if path is None:
      return None
    os.makedirs(self.directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
      json.dump(log, f, indent=1, sort_keys=True)
      f.write("\n")
    return path
