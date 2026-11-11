"""FDO (PGLE) profile as plain data, decoded without protobuf, tsl or protoc.

Two encodings are read: the binary wire format and the text format XLA writes
next to the HLO dumps (`<module>.fdo_profile` under --xla_dump_to, with
--xla_gpu_experimental_dump_fdo_profiles). Prefer the text dump.

`ProfiledInstructionsProto` (tsl/profiler/protobuf/profiled_instructions.proto):

  repeated InstructionCost costs = 1;   // {string name = 1; double cost_us = 2;}
  repeated Latency latencies = 2;       // {string source = 1; string target = 2;
                                        //  double latency_us = 3;}

Costs are keyed by *instruction name* (collectives under the name of their
wrapped root, e.g. "all-gather.3.1", not the "all-gather-start" wrapper); a
missing name means "not profiled", never "free". Newer XLA changed two things:
  * Keys carry the module fingerprint: "<fingerprint>::<name>", where the
    fingerprint is the module's `fingerprint_before_lhs` frontend attribute. A
    cost applies only to the module with that fingerprint; an unprefixed cost
    applies to any module. Use `Profile.for_module` before looking names up.
  * The name is the one the scheduler saw (`OpMetadata.scheduling_name`), which
    differs from the final name for instructions renamed after scheduling (e.g.
    a synchronous `all-gather.579` is keyed `all-gather-start.33`). `module_config` handed to a
transform carries the profile as field 31 (`bytes fdo_profile`) of
`HloModuleConfigProto` (xla/xla.proto); some PJRT plugins pass empty bytes there.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from typing import Iterator, Mapping

_MODULE_CONFIG_FDO_PROFILE_FIELD = 31
FINGERPRINT_SEP = "::"  # xla/service/hlo_profiling_utils.cc kCostNameSep
_FINGERPRINT_RE = re.compile(r'fingerprint_before_lhs="([^"]*)"')


def module_fingerprint(module_text: str) -> str | None:
  """`fingerprint_before_lhs` of a printed HloModule (its frontend attribute), or None."""
  m = _FINGERPRINT_RE.search(module_text.split("\n", 1)[0])  # the header line can be ~100 KB
  return m.group(1) if m else None


def _varint(buf: bytes, i: int) -> tuple[int, int]:
  """Decode a base-128 varint at `i`; returns (value, next index)."""
  shift = value = 0
  while True:
    if i >= len(buf):
      raise ValueError("truncated varint")
    b = buf[i]
    i += 1
    value |= (b & 0x7F) << shift
    if not b & 0x80:
      return value, i
    shift += 7
    if shift > 63:
      raise ValueError("varint too long")


def iter_fields(buf: bytes) -> Iterator[tuple[int, int, object]]:
  """Yields (field_number, wire_type, value) for one serialized message."""
  i = 0
  while i < len(buf):
    key, i = _varint(buf, i)
    num, wire = key >> 3, key & 7
    if wire == 0:
      value, i = _varint(buf, i)
    elif wire == 1:
      value, i = buf[i : i + 8], i + 8
    elif wire == 2:
      n, i = _varint(buf, i)
      value, i = buf[i : i + n], i + n
    elif wire == 5:
      value, i = buf[i : i + 4], i + 4
    else:
      raise ValueError(f"unsupported wire type {wire}")
    if isinstance(value, bytes) and len(value) < (8 if wire == 1 else 4 if wire == 5 else 0):
      raise ValueError("truncated field")
    yield num, wire, value


_BLOCK_RE = re.compile(r"(costs|latencies)\s*\{([^}]*)\}")
_ATTR_RE = re.compile(r'(\w+)\s*:\s*("(?:[^"\\]|\\.)*"|[^\s}]+)')


def _escape(value: str) -> str:
  return value.replace("\\", "\\\\").replace('"', '\\"')


def _unquote(value: str) -> str:
  if value.startswith('"'):
    return re.sub(r"\\(.)", r"\1", value[1:-1])
  return value


@dataclass(frozen=True)
class Profile:
  """Per-instruction costs (us) and source->target latencies (us)."""

  costs: Mapping[str, float] = field(default_factory=dict)
  latencies: Mapping[tuple[str, str], float] = field(default_factory=dict)

  @classmethod
  def from_costs(cls, costs: Mapping[str, float]) -> "Profile":
    return cls(dict(costs))

  @classmethod
  def from_fdo_bytes(cls, data: bytes) -> "Profile":
    """Decode a serialized ProfiledInstructionsProto. Empty bytes give an empty profile."""
    costs: dict[str, float] = {}
    latencies: dict[tuple[str, str], float] = {}
    for num, wire, value in iter_fields(data):
      if wire != 2:
        continue
      if num == 1:
        name, cost = "", 0.0
        for n, w, v in iter_fields(value):
          if n == 1 and w == 2:
            name = v.decode("utf-8")
          elif n == 2 and w == 1:
            cost = struct.unpack("<d", v)[0]
        if name:
          costs[name] = cost
      elif num == 2:
        src, tgt, lat = "", "", 0.0
        for n, w, v in iter_fields(value):
          if n == 1 and w == 2:
            src = v.decode("utf-8")
          elif n == 2 and w == 2:
            tgt = v.decode("utf-8")
          elif n == 3 and w == 1:
            lat = struct.unpack("<d", v)[0]
        latencies[(src, tgt)] = lat
    return cls(costs, latencies)

  @classmethod
  def from_text(cls, text: str) -> "Profile":
    """Parse the text-format ProfiledInstructionsProto XLA dumps."""
    costs: dict[str, float] = {}
    latencies: dict[tuple[str, str], float] = {}
    for kind, body in _BLOCK_RE.findall(text):
      attrs = {k: _unquote(v) for k, v in _ATTR_RE.findall(body)}
      if kind == "costs":
        if attrs.get("name"):
          costs[attrs["name"]] = float(attrs.get("cost_us", 0.0))
      else:
        latencies[(attrs.get("source", ""), attrs.get("target", ""))] = float(attrs.get("latency_us", 0.0))
    return cls(costs, latencies)

  @classmethod
  def from_file(cls, path: str) -> "Profile":
    """Load a profile file, text or binary, as written by --xla_dump_to or a profiler."""
    with open(path, "rb") as f:
      data = f.read()
    head = data.lstrip()[:9]
    if head.startswith((b"costs", b"latencies", b"#")) or not data.strip():
      return cls.from_text(data.decode("utf-8"))
    return cls.from_fdo_bytes(data)

  @classmethod
  def from_module_config(cls, module_config: bytes) -> "Profile":
    """The profile embedded in a serialized HloModuleConfigProto (empty if none)."""
    for num, wire, value in iter_fields(module_config):
      if num == _MODULE_CONFIG_FDO_PROFILE_FIELD and wire == 2:
        return cls.from_fdo_bytes(value)
    return cls()

  def to_text(self) -> str:
    """Text-format ProfiledInstructionsProto (what `from_text` and XLA's dump read)."""
    lines = []
    for name in sorted(self.costs):
      lines.append(f'costs {{ name: "{_escape(name)}" cost_us: {float(self.costs[name])!r} }}')
    for (src, tgt), lat in sorted(self.latencies.items()):
      lines.append(f'latencies {{ source: "{_escape(src)}" target: "{_escape(tgt)}" latency_us: {float(lat)!r} }}')
    return "\n".join(lines) + ("\n" if lines else "")

  def fingerprints(self) -> set[str]:
    """Module fingerprints that prefix cost names (empty for an unprefixed profile)."""
    return {k.split(FINGERPRINT_SEP, 1)[0] for k in self.costs if FINGERPRINT_SEP in k}

  def for_module(self, fingerprint: str | None) -> "Profile":
    """The profile with names unprefixed, keeping only costs that apply to the module.

    Mirrors XLA's `FilterWithFingerprint`: a cost prefixed with another module's
    fingerprint is dropped, an unprefixed cost is kept. With `fingerprint=None`
    (unknown) the prefix is stripped only if the profile holds a single
    fingerprint; with several it is ambiguous, so raise rather than guess.
    """
    prints = self.fingerprints()
    if not prints:
      return self
    if fingerprint is None:
      if len(prints) > 1:
        raise ValueError(
            f"profile holds costs of {len(prints)} modules ({sorted(prints)}) and the module's "
            "fingerprint_before_lhs is unknown; cannot choose which costs apply"
        )
      fingerprint = next(iter(prints))

    def keep(table, names):
      out = {}
      for key, val in table.items():
        parts = [k.split(FINGERPRINT_SEP, 1) for k in names(key)]
        if any(len(p) == 2 and p[0] != fingerprint for p in parts):
          continue
        out[key] = val
      return out

    def strip(name: str) -> str:
      head, sep, tail = name.partition(FINGERPRINT_SEP)
      return tail if sep and head == fingerprint else name

    costs = {strip(k): v for k, v in keep(self.costs, lambda k: (k,)).items()}
    lats = {(strip(s), strip(t)): v for (s, t), v in keep(self.latencies, lambda k: k).items()}
    return Profile(costs, lats)

  def cost(self, name: str) -> float:
    """Profiled cost, or 0.0 when the name is not profiled."""
    return float(self.costs.get(name, 0.0) or 0.0)

  def has(self, name: str) -> bool:
    return bool(self.costs.get(name))
