"""Parsers for the HLO text the jaxlib binding exposes.

The `HloInstruction` binding offers name, opcode, operands, users, to_string and
async_wrapped_root; everything else (control predecessors, callees, replica
groups, custom-call targets) is read from `to_string()`. These helpers are pure
string functions so they can be unit-tested without jaxlib.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

CALLS_RE = re.compile(r"calls=%([A-Za-z0-9_.\-]+)")
TO_APPLY_RE = re.compile(r"to_apply=%([A-Za-z0-9_.\-]+)")
WHILE_BODY_RE = re.compile(r"\bbody=%([A-Za-z0-9_.\-]+)")
CONTROL_PRED_RE = re.compile(r"control-predecessors=\{([^}]*)\}")
CUSTOM_CALL_TARGET_RE = re.compile(r'custom_call_target="([^"]*)"')
TE_GEMM_TARGET_RE = re.compile(r'custom_call_target="(te_grouped_gemm[^"]*|te_gemm[^"]*)"')

# Memory space S(5) is pinned host memory; D2H/H2D offload copies touch it.
HOST_MEMORY_SPACE_TAG = ":S(5)"


def opcode_str(opcode_name: str) -> str:
  """Hyphenated opcode, e.g. "kGetTupleElement" -> "get-tuple-element"."""
  return re.sub(r"(?<!^)(?=[A-Z])", "-", opcode_name[1:]).lower()


def control_predecessor_names(text: str) -> tuple[str, ...]:
  m = CONTROL_PRED_RE.search(text)
  if not m:
    return ()
  return tuple(n.strip().lstrip("%") for n in m.group(1).split(",") if n.strip())


def while_body_name(text: str) -> Optional[str]:
  """Body computation of a `while` instruction."""
  m = WHILE_BODY_RE.search(text)
  return m.group(1) if m else None


def callee_name(text: str) -> Optional[str]:
  """Name of the computation an instruction calls (calls= or to_apply=)."""
  m = CALLS_RE.search(text) or TO_APPLY_RE.search(text)
  return m.group(1) if m else None


def custom_call_target(text: str) -> Optional[str]:
  m = CUSTOM_CALL_TARGET_RE.search(text)
  return m.group(1) if m else None


def touches_host_memory(text: str) -> bool:
  return HOST_MEMORY_SPACE_TAG in text


# ---- replica groups -------------------------------------------------------

_EXPLICIT_RE = re.compile(r"replica_groups=(\{(?:\{[0-9,\s]*\},?\s*)*\})")
_IOTA_RE = re.compile(r"replica_groups=\[([0-9,\s]+)\]<=\[([0-9,\s]+)\](?:T\(([0-9,\s]+)\))?")
_MESH_RE = re.compile(r"replica_groups=mesh\[([^\]]*)\]\s*\{([^}]*)\}")
_PAIRS_RE = re.compile(r"source_target_pairs=(\{(?:\{[0-9,\s]*\},?\s*)*\})")


def _ints(s: str) -> list[int]:
  return [int(x) for x in s.replace(" ", "").split(",") if x]


def _brace_groups(s: str) -> tuple[frozenset, ...]:
  return tuple(frozenset(_ints(g)) for g in re.findall(r"\{([0-9,\s]*)\}", s[1:-1]))


def parse_device_groups(text: str) -> Optional[tuple[frozenset, ...]]:
  """Participating device groups of a collective, from its printed attributes.

  Understands the three replica-group spellings XLA prints (explicit, iota and
  mesh-axis form) and collective-permute `source_target_pairs`. The mesh form is
  resolved assuming mesh devices are numbered in row-major order, which holds for
  the default mesh; anything not understood returns None (unknown), never a
  guess. An empty `replica_groups={}` also returns None: it means "all devices"
  and the count is not in the text.
  """
  m = _EXPLICIT_RE.search(text)
  if m:
    groups = _brace_groups(m.group(1))
    return groups or None
  m = _IOTA_RE.search(text)
  if m:
    shape, base = _ints(m.group(1)), _ints(m.group(2))
    ids = np.arange(int(np.prod(base))).reshape(base)
    if m.group(3):
      ids = ids.transpose(_ints(m.group(3)))
    if len(shape) != 2 or ids.size != shape[0] * shape[1]:
      return None
    ids = ids.reshape(shape)
    return tuple(frozenset(int(d) for d in row) for row in ids)
  m = _MESH_RE.search(text)
  if m:
    axes: list[tuple[str, int]] = []
    for part in m.group(1).split(","):
      am = re.match(r"\s*'([^']+)'\s*=\s*(\d+)\s*$", part)
      if not am:
        return None
      axes.append((am.group(1), int(am.group(2))))
    chosen = [a.strip().strip("'") for a in m.group(2).split(",") if a.strip()]
    names = [a for a, _ in axes]
    if not chosen or any(c not in names for c in chosen):
      return None
    ids = np.arange(int(np.prod([n for _, n in axes]))).reshape([n for _, n in axes])
    keep = [i for i, a in enumerate(names) if a not in chosen]
    over = [i for i, a in enumerate(names) if a in chosen]
    ids = ids.transpose(keep + over).reshape(-1, int(np.prod([axes[i][1] for i in over])))
    return tuple(frozenset(int(d) for d in row) for row in ids)
  m = _PAIRS_RE.search(text)
  if m:
    groups = _brace_groups(m.group(1))
    return groups or None
  return None
