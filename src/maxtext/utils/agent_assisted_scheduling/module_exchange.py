"""Offline hand-off of a scheduled module, as a `POST_SCHEDULER` XLA transform.

Two environment variables control it, and it either dumps or loads, never both:

  COLLECTIVE_HLO_MODE   dump | load | off (default: off, nothing is registered)
  COLLECTIVE_HLO_PATH   the directory to dump into, or the file to load

  COLLECTIVE_HLO_MODE=dump  COLLECTIVE_HLO_PATH=<dir>
      Write what XLA hands the transform (the module right after scheduling and
      post-scheduling copy insertion, before later passes rewrite it) to `<dir>`,
      from process 0 only:
        <module>.post_scheduler.<n>.hlo_module.pb    serialized HloModuleProto
        <module>.post_scheduler.<n>.fdo_profile.txt  only if the transform is handed a profile
      `n` counts compiles of that module name in this process: with PGLE the first
      compile has no profile and the recompile that follows the profiling runs does,
      so the highest `n` is the one to optimize. The transform is normally not handed
      the profile; run with --xla_gpu_experimental_dump_fdo_profiles=true and XLA
      dumps it as text (`module_*.<module>.*.fdo_profile`) into the same directory.
      COLLECTIVE_DUMP_MODULE_RE (default ".*") and COLLECTIVE_DUMP_MIN_BYTES
      (default 10000) restrict which modules are written.

  COLLECTIVE_HLO_MODE=load  COLLECTIVE_HLO_PATH=<file>
      The file is a serialized HloModuleProto (what `cli optimize ... --out-module` writes).
      Every module XLA hands the transform is compared with it: the module name
      must match, the computations must be the same, and the instructions may
      differ by at most a handful (compiles are not bit-for-bit repeatable: a copy
      can appear or vanish). The file's order is kept for the instructions both
      have, new ones are slotted in beside their neighbours, and the result must be
      legal for the module's graph and survive XLA's own update()/verify(). Then
      the module is returned to XLA (the file's bytes as they are when nothing
      differs); otherwise the transform returns None and XLA keeps its own
      schedule.

      With several processes only rank 0 does that work (the merge and legality checks take
      seconds on a real module): every rank waits at an entry barrier, rank 0 publishes
      the bytes through the JAX coordination-service key-value store, the others read
      them, and an exit barrier follows, so every rank returns the same bytes. If the
      hand-off fails a rank falls back to the same deterministic computation itself.

The transform never raises into the compiler: any problem is logged and XLA's
schedule is kept.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Callable, Optional

from maxtext.utils.agent_assisted_scheduling import kv_share
from maxtext.utils.agent_assisted_scheduling.profile import Profile, iter_fields

_logger = logging.getLogger(__name__)
_NAME = "collective_module_exchange"
MODE_ENV = "COLLECTIVE_HLO_MODE"
PATH_ENV = "COLLECTIVE_HLO_PATH"
MODES = ("off", "dump", "load")
MIN_TOLERATED = 8  # added or dropped instructions always tolerated
MAX_DIFFERENT_FRACTION = 0.001  # ... and up to this fraction of the module
_registered = False
_compiles: dict[str, int] = {}
SHARE_TIMEOUT_MS = 10 * 60 * 1000


def module_name(serialized_hlo: bytes) -> Optional[str]:
  """The HloModuleProto's name (field 1) without parsing the whole module."""
  for number, wire, value in iter_fields(serialized_hlo):
    if number == 1 and wire == 2:
      return bytes(value).decode("utf-8", "replace")
    break
  return None


def _process_index() -> int:
  try:
    import jax  # pylint: disable=import-outside-toplevel

    return jax.process_index()
  except Exception:  # pylint: disable=broad-exception-caught
    return 0


def dump_module(
    serialized_hlo: bytes, module_config: bytes, directory: str, *, name_re: str = ".*", min_bytes: int = 10_000
) -> Optional[str]:
  """Write the module and its profile into `directory`; the module file's path, or None if skipped."""
  name = module_name(serialized_hlo)
  if name is None or len(serialized_hlo) < min_bytes or not re.search(name_re, name):
    return None
  n = _compiles.get(name, 0)
  _compiles[name] = n + 1
  os.makedirs(directory, exist_ok=True)
  stem = os.path.join(directory, f"{name}.post_scheduler.{n}")
  profile = Profile.from_module_config(module_config) if module_config else Profile()
  with open(stem + ".hlo_module.pb", "wb") as f:
    f.write(serialized_hlo)
  if profile.costs or profile.latencies:
    with open(stem + ".fdo_profile.txt", "w", encoding="utf-8") as f:
      f.write(profile.to_text())
  _logger.info(
      "%s: dumped %s (%d bytes, %d profile entries) to %s.*", _NAME, name, len(serialized_hlo), len(profile.costs), stem
  )
  return stem + ".hlo_module.pb"


class OptimizedModule:
  """The module loaded in COLLECTIVE_HLO_MODE=load, matched against what XLA hands over."""

  def __init__(self, path: str):
    self.path = path
    with open(path, "rb") as f:
      self.data = f.read()
    self.name = module_name(self.data)
    if self.name is None:
      raise ValueError(f"{path} is not a serialized HloModuleProto")
    self._orders: Optional[dict[str, list[str]]] = None

  def _file_orders(self) -> dict[str, list[str]]:
    if self._orders is None:
      from jax._src.lib import hlo  # pylint: disable=import-outside-toplevel
      from maxtext.utils.agent_assisted_scheduling.hlo_adapter import load_module  # pylint: disable=import-outside-toplevel

      self._orders = load_module(hlo.HloModule.from_serialized_hlo_module_proto(self.data), Profile()).baseline
    return self._orders

  def returns_for(self, serialized_hlo: bytes) -> Optional[bytes]:
    """The bytes to give XLA for this input module, or None to keep XLA's schedule."""
    if module_name(serialized_hlo) != self.name:
      return None
    from jax._src.lib import hlo  # pylint: disable=import-outside-toplevel
    from maxtext.utils.agent_assisted_scheduling.hlo_adapter import apply_plan, load_module  # pylint: disable=import-outside-toplevel
    from maxtext.utils.agent_assisted_scheduling.legality import errors, validate_plan  # pylint: disable=import-outside-toplevel

    loaded = load_module(hlo.HloModule.from_serialized_hlo_module_proto(serialized_hlo), Profile())
    wanted = self._file_orders()
    if set(wanted) != set(loaded.baseline):
      return self._mismatch(loaded, "different computations")
    added = {c: [n for n in o if n not in set(wanted[c])] for c, o in loaded.baseline.items()}
    dropped = {c: [n for n in o if n not in set(loaded.baseline[c])] for c, o in wanted.items()}
    differing = sum(len(v) for v in added.values()) + sum(len(v) for v in dropped.values())
    total = sum(len(o) for o in loaded.baseline.values())
    if differing > max(MIN_TOLERATED, int(total * MAX_DIFFERENT_FRACTION)):
      return self._mismatch(loaded, f"{differing} of {total} instructions differ")
    if differing:
      # Compiles are not bit-for-bit repeatable (a copy may appear or vanish): keep the optimized
      # order for everything both modules have and slot the new instructions in next to their
      # neighbours. The legality check below and XLA's update()/verify() decide whether it holds.
      wanted = {c: _merge_order(loaded, c, loaded.baseline[c], o) for c, o in wanted.items()}
    problems = errors(validate_plan(loaded.module, wanted, baseline=loaded.baseline))
    if problems:
      return self._mismatch(
          loaded, "the loaded schedule is not legal for this module (" + "; ".join(f"{v.code}: {v.message}" for v in problems[:3]) + ")"
      )
    if wanted == loaded.baseline:
      _logger.info("%s [%s]: the loaded schedule equals XLA's; nothing to change.", _NAME, self.name)
      return None
    moved = sum(1 for c, o in wanted.items() if o != loaded.baseline[c])
    if not differing:
      _logger.info(
          "%s [%s]: loaded the optimized module from %s (%d computation(s) reordered).", _NAME, self.name, self.path, moved
      )
      return self.data
    result = apply_plan(loaded, wanted)
    if not result.applied:
      return self._mismatch(loaded, f"the merged schedule was rejected ({result.reason})")
    _logger.info(
        "%s [%s]: loaded the optimized schedule from %s with %d added and %d dropped instruction(s) merged "
        "(%d computation(s) reordered).",
        _NAME,
        self.name,
        self.path,
        sum(len(v) for v in added.values()),
        sum(len(v) for v in dropped.values()),
        moved,
    )
    return loaded.hlo_module.as_serialized_hlo_module_proto()

  def _mismatch(self, loaded, why: str) -> None:
    _logger.warning("%s [%s]: not using %s: %s; keeping XLA's schedule.", _NAME, self.name, self.path, why)
    return None


def _merge_order(loaded, computation: str, base_order: list[str], file_order: list[str]) -> list[str]:
  """`file_order` restricted to instructions `base_order` has, plus the instructions only `base_order` has.

  Each new instruction goes after its operands and control predecessors and after the instruction that
  preceded it in `base_order`, in `base_order` order, so the result is deterministic.
  """
  present = set(base_order)
  merged = [n for n in file_order if n in present]
  known = set(merged)
  comp = loaded.module.computations[computation]
  index = {n: i for i, n in enumerate(base_order)}
  for name in base_order:
    if name in known:
      continue
    pos = {n: i for i, n in enumerate(merged)}
    inst = comp[name]
    after = max((pos[d] for d in inst.operands + inst.control_predecessors if d in pos), default=-1) + 1
    prev = next((base_order[j] for j in range(index[name] - 1, -1, -1) if base_order[j] in pos), None)
    merged.insert(max(after, pos[prev] + 1 if prev is not None else 0), name)
    known.add(name)
  return merged


def _load_on_rank0(
    optimized: OptimizedModule, hlo_module: bytes, client, rank: int, loads: dict[str, int]
) -> Optional[bytes]:
  """`optimized.returns_for(hlo_module)` computed on rank 0 only and shared with every rank."""
  if client is None:
    return optimized.returns_for(hlo_module)
  count = loads.get(optimized.name, 0) + 1  # every rank sees the same modules in the same order
  loads[optimized.name] = count
  tag = f"{_NAME}/{optimized.name}/{count}"

  def barrier(name: str) -> None:
    try:
      client.wait_at_barrier(f"{tag}/{name}", SHARE_TIMEOUT_MS)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      _logger.warning("%s: %s barrier failed (%s); proceeding without it.", _NAME, name, exc)

  barrier("entry")
  try:
    payload = kv_share.share(
        client,
        f"{tag}/module",
        rank=rank,
        produce=lambda: optimized.returns_for(hlo_module) or b"",
        timeout_ms=SHARE_TIMEOUT_MS,
    )
  except kv_share.ShareTimeout as exc:
    _logger.warning("%s: %s; computing it on this rank.", _NAME, exc)
    payload = optimized.returns_for(hlo_module) or b""
  barrier("exit")
  if rank != 0 and payload:
    _logger.info("%s [%s]: using the optimized module from rank 0 (rank %d).", _NAME, optimized.name, rank)
  return payload or None


def make_callback(
    dump_dir: Optional[str],
    load_path: Optional[str],
    *,
    name_re: str = ".*",
    min_bytes: int = 10_000,
    distributed=kv_share.distributed_client,
) -> Callable[..., Optional[bytes]]:
  """The transform body for the given settings (empty strings and None mean "off").

  `distributed()` returns (kv client, rank); a client means several processes share the load.
  """
  optimized = OptimizedModule(load_path) if load_path else None
  loads: dict[str, int] = {}

  def callback(hlo_module: bytes, *, module_config: bytes = b"") -> Optional[bytes]:
    try:
      if dump_dir and _process_index() == 0:
        dump_module(hlo_module, module_config, dump_dir, name_re=name_re, min_bytes=min_bytes)
      if not optimized or module_name(hlo_module) != optimized.name:
        return None
      client, rank = distributed()
      return _load_on_rank0(optimized, hlo_module, client, rank, loads)
    except Exception:  # pylint: disable=broad-exception-caught
      _logger.exception("%s: failed; keeping XLA's schedule.", _NAME)
      return None

  return callback


def settings(env=None) -> tuple[str, str]:
  """(mode, path) from the environment; a mode other than off needs a path."""
  env = os.environ if env is None else env
  mode = env.get(MODE_ENV, "").strip().lower() or "off"
  path = env.get(PATH_ENV, "").strip()
  if mode not in MODES:
    raise ValueError(f"{MODE_ENV}={mode!r} is not one of {', '.join(MODES)}")
  if mode != "off" and not path:
    raise ValueError(f"{MODE_ENV}={mode} needs {PATH_ENV}")
  if mode == "load" and not os.path.isfile(path):
    raise FileNotFoundError(f"{PATH_ENV}={path} is not a file")
  return mode, path


def register() -> bool:
  """Register the transform once unless COLLECTIVE_HLO_MODE is off. True if registered."""
  global _registered
  mode, path = settings()
  if _registered or mode == "off":
    return _registered
  import jax.extend.xla as jex_xla  # pylint: disable=import-outside-toplevel

  callback = make_callback(
      path if mode == "dump" else None,
      path if mode == "load" else None,
      name_re=os.environ.get("COLLECTIVE_DUMP_MODULE_RE", ".*"),
      min_bytes=int(os.environ.get("COLLECTIVE_DUMP_MIN_BYTES", "10000")),
  )
  jex_xla.register_hlo_module_transformation(callback, name=_NAME, stage=jex_xla.PipelineStage.POST_SCHEDULER)
  _registered = True
  _logger.info("%s: registered POST_SCHEDULER transform (mode=%s, path=%s).", _NAME, mode, path)
  return True
