"""Where an async start may legally move to.

Port of `_earliest_legal_pos` / `_reinsert` from collective_overlap_pass.py onto
the jax-free model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from maxtext.utils.agent_assisted_scheduling.schedule_model import Computation, positions

MAX_RELOCATE_CHAIN = 64


@dataclass(frozen=True)
class EarliestLegalPos:
  """Result of `earliest_legal_pos`.

  Attributes:
    floor: earliest legal index for the start (insert the chain, then the start,
      at this index of the order with those instructions removed).
    movable: trivial operand-chain instructions at or after `floor`, in schedule
      order, excluding the start. They must move with it. Chain members already
      before `floor` stay put (moving them would drag the start later).
    blocker: non-trivial data producer that pins the floor, or None if the floor
      is 0 or pinned by a control predecessor.
    control_pins: (position, name) of control predecessors outside the moving set.
  """

  floor: int
  movable: tuple[str, ...]
  blocker: Optional[str]
  control_pins: tuple[tuple[int, str], ...]


def earliest_legal_pos(
    comp: Computation, order: list[str], start: str,
    max_chain: int = MAX_RELOCATE_CHAIN,
) -> EarliestLegalPos:
  """Earliest position `start` can legally move to in `order`.

  Walks the start's operands transitively through trivially movable
  instructions. The walk stops at any non-trivial instruction, which pins the
  floor to just after it. Control predecessors outside the moving set pin it the
  same way.
  """
  pos = positions(order)
  candidates: set[str] = set()
  seen: set[str] = set()
  floor = 0
  blocker: Optional[str] = None
  queue = list(comp[start].operands)
  while queue:
    name = queue.pop()
    if name in seen:
      continue
    seen.add(name)
    if name not in pos:
      continue
    inst = comp[name]
    if len(candidates) < max_chain and inst.is_trivially_movable:
      candidates.add(name)
      queue.extend(inst.operands)
    elif pos[name] + 1 > floor:
      floor = pos[name] + 1
      blocker = name

  moving = candidates | {start}
  control_pins: list[tuple[int, str]] = []
  for name in list(candidates) + [start]:
    for pred in comp[name].control_predecessors:
      if pred in moving or pred not in pos:
        continue
      control_pins.append((pos[pred], pred))
      if pos[pred] + 1 > floor:
        floor = pos[pred] + 1
        blocker = None

  # The floor only grows during the walk, so early candidates may already sit
  # before the final one.
  movable = sorted((n for n in candidates if pos[n] >= floor), key=lambda n: pos[n])
  return EarliestLegalPos(floor, tuple(movable), blocker, tuple(control_pins))


def reinsert(order: list[str], moving: list[str] | tuple[str, ...], at: int, last: str) -> list[str]:
  """`order` without `moving`/`last`, with `moving` then `last` inserted at `at`."""
  drop = set(moving) | {last}
  new_order = [n for n in order if n not in drop]
  new_order[at:at] = list(moving) + [last]
  return new_order


def hoist_start_to_floor(comp: Computation, order: list[str], start: str) -> list[str]:
  """`order` with `start` (and its movable trivial chain) moved to the floor.

  Returns `order` unchanged if there is no room (floor + chain length is not
  before the start's current position).
  """
  result = earliest_legal_pos(comp, order, start)
  if result.floor + len(result.movable) >= positions(order)[start]:
    return list(order)
  return reinsert(order, result.movable, result.floor, start)
