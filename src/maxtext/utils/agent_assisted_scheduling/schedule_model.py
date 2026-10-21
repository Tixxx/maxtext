"""Minimal, jax-free model of one scheduled HLO computation.

The validator, cost and simulator modules operate on this model so they can be
unit-tested without jaxlib or a GPU. `hlo_adapter` builds it from
`jax._src.lib.hlo.HloModule`, filling in costs from the FDO profile.

Everything is keyed by instruction *name* (unique per module). A schedule for a
computation is a plain `list[str]` of instruction names.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


class AsyncRole(enum.Enum):
  NONE = "none"
  START = "start"
  DONE = "done"


# Ops that can be relocated without moving real compute. Mirrors
# _TRIVIAL_SINGLE_OPCODES in gpu_collective_overlap_pass.py; fusions whose body is
# entirely trivial are flagged by the adapter via `Inst.trivially_movable`.
TRIVIAL_OPCODES = frozenset(
    {
        "bitcast",
        "convert",
        "transpose",
        "reshape",
        "broadcast",
        "get-tuple-element",
        "tuple",
        "copy",
        "slice",
        "parameter",
        "constant",
    }
)

# Zero-cost plumbing: a start/done pair with only these between is converted to
# a synchronous collective after scheduling (ConvertAsyncCollectivesToSync).
NOP_OPCODES = frozenset({"bitcast", "get-tuple-element", "tuple", "parameter", "constant"})


@dataclass(frozen=True)
class Inst:
  """One instruction.

  Attributes:
    name: unique instruction name.
    opcode: hyphenated opcode string, e.g. "all-gather-start", "fusion".
    operands: operand instruction names, in order.
    control_predecessors: names that must be scheduled before this one.
    cost_us: profiled compute cost; 0 when unknown (treated as free) and 0 for
      DMA-only ops that hide nothing (host-offload copies).
    async_role: START for the first half of an async pair, DONE for the second.
      A done's operand 0 is its start.
    latency_us: profiled start->done latency of a START; None if not profiled.
    uses_comm_stream: a START that is queued on the (single) comm stream.
    sm_gate: a large GEMM that holds the SMs; a collective cannot begin while it
      runs (see simulate.simulate_comm_stream).
    trivially_movable: override for the opcode-based default.
    writes_operand: name of an operand buffer this instruction updates in place
      (e.g. a D2H dynamic-update-slice into a host buffer). Every other user of
      that operand must keep its relative order with this instruction.
    device_groups: for collective STARTs, the participating device groups, as a
      tuple of frozensets of global device ids. None means unknown.
    profile_key: for STARTs, the instruction name the latency is looked up under
      in the FDO profile (the wrapped root, e.g. "all-gather.3.1").
  """

  name: str
  opcode: str
  operands: tuple[str, ...] = ()
  control_predecessors: tuple[str, ...] = ()
  cost_us: float = 0.0
  async_role: AsyncRole = AsyncRole.NONE
  latency_us: Optional[float] = None
  uses_comm_stream: bool = False
  sm_gate: bool = False
  trivially_movable: Optional[bool] = None
  writes_operand: Optional[str] = None
  device_groups: Optional[tuple[frozenset, ...]] = None
  profile_key: Optional[str] = None

  @property
  def is_trivially_movable(self) -> bool:
    if self.trivially_movable is not None:
      return self.trivially_movable
    return self.opcode in TRIVIAL_OPCODES


@dataclass
class Computation:
  """A computation's instructions, independent of any order."""

  name: str
  instructions: dict[str, Inst] = field(default_factory=dict)

  @classmethod
  def from_instructions(cls, name: str, insts: list[Inst]) -> "Computation":
    by_name: dict[str, Inst] = {}
    for inst in insts:
      if inst.name in by_name:
        raise ValueError(f"duplicate instruction name {inst.name!r} in {name!r}")
      by_name[inst.name] = inst
    return cls(name, by_name)

  def __getitem__(self, name: str) -> Inst:
    return self.instructions[name]

  def __contains__(self, name: str) -> bool:
    return name in self.instructions

  def users(self) -> dict[str, list[str]]:
    """name -> names of instructions that use it as an operand."""
    out: dict[str, list[str]] = {n: [] for n in self.instructions}
    for inst in self.instructions.values():
      for op in inst.operands:
        if op in out:
          out[op].append(inst.name)
    return out

  def done_of(self, start_name: str) -> Optional[str]:
    """The DONE whose operand 0 is `start_name`, if exactly one exists."""
    dones = [
        i.name
        for i in self.instructions.values()
        if i.async_role is AsyncRole.DONE and i.operands and i.operands[0] == start_name
    ]
    return dones[0] if len(dones) == 1 else None

  def async_pairs(self) -> list[tuple[str, str]]:
    """(start, done) name pairs, in no particular order."""
    pairs = []
    for inst in self.instructions.values():
      if inst.async_role is AsyncRole.START:
        done = self.done_of(inst.name)
        if done is not None:
          pairs.append((inst.name, done))
    return pairs


@dataclass
class Module:
  """The scheduled (non-fusion) computations of one HLO module.

  XLA instruction names are unique per module, so `owner` maps every instruction
  to the single computation it lives in. That is what makes "no instruction
  leaves its computation" checkable from names alone.
  """

  name: str
  computations: dict[str, Computation] = field(default_factory=dict)
  owner: dict[str, str] = field(init=False, default_factory=dict)

  def __post_init__(self):
    self.owner = {}
    for comp in self.computations.values():
      for inst_name in comp.instructions:
        if inst_name in self.owner:
          raise ValueError(
              f"instruction name {inst_name!r} appears in both {self.owner[inst_name]!r} "
              f"and {comp.name!r}; names must be unique per module"
          )
        self.owner[inst_name] = comp.name

  @classmethod
  def from_computations(cls, name: str, comps: list[Computation]) -> "Module":
    return cls(name, {c.name: c for c in comps})


def positions(order: list[str]) -> dict[str, int]:
  """name -> index in `order` (last occurrence wins; validate first)."""
  return {name: i for i, name in enumerate(order)}
