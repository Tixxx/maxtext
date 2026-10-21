"""The schedule plan: what an agent hands to the pass, and how it is checked.

A plan is canonical JSON:

  {
    "version": 1,
    "module": "jit_train_step",
    "input_checksum": "<sha256 of the schedule the plan was made for>",
    "profile": {"entries": 4380, "sha256": "<digest of the FDO costs>"},
    "computations": {"<computation>": ["<instruction>", ...]},
    "moves": [{"computation": "...", "instruction": "...", "to_before": "...",
               "playbook_step": "...", "expected_delta_us": -120.0, "note": "..."}]
  }

* `computations` holds the FULL proposed order of every computation to change;
  computations not listed keep XLA's order. Instructions never leave their
  computation (see `legality.validate_plan`).
* `input_checksum` identifies the module *as scheduled by XLA* (instruction names
  and order of every computation). It deliberately ignores the serialized proto
  bytes, which differ per compile (unique ids). A plan whose checksum does not
  match the module being compiled must not be applied.
* `profile` identifies the FDO costs the plan was scored with. A mismatch is a
  warning (the plan is still legal, just possibly stale).
* `moves` is optional annotation for reviewers and the agent's own bookkeeping;
  it never changes behaviour beyond a consistency check.

Serialization is deterministic (sorted keys, fixed separators, UTF-8), so every
rank that reads the same plan sees identical bytes and `Plan.digest()` agrees.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional

from maxtext.utils.agent_assisted_scheduling.legality import Severity, Violation
from maxtext.utils.agent_assisted_scheduling.profile import Profile
from maxtext.utils.agent_assisted_scheduling.schedule_model import Module

PLAN_VERSION = 1


class PlanFormatError(ValueError):
  """The plan file is not a valid plan document."""


def _canonical(obj: Any) -> str:
  return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def input_checksum(baseline: dict[str, list[str]]) -> str:
  """SHA-256 of the schedule (names and order of every computation) XLA produced."""
  return hashlib.sha256(_canonical(baseline).encode("utf-8")).hexdigest()


def profile_identity(profile: Profile) -> dict[str, Any]:
  """Entry count and a digest of the profile's costs (exact floats, name-sorted)."""
  h = hashlib.sha256()
  for name in sorted(profile.costs):
    h.update(f"{name}\0{float(profile.costs[name]).hex()}\n".encode("utf-8"))
  for (src, tgt), lat in sorted(profile.latencies.items()):
    h.update(f"{src}\0{tgt}\0{float(lat).hex()}\n".encode("utf-8"))
  return {"entries": len(profile.costs), "sha256": h.hexdigest()}


@dataclass(frozen=True)
class Move:
  """Annotation of one intended move; informational."""

  computation: str
  instruction: str
  to_before: Optional[str] = None
  playbook_step: Optional[str] = None
  expected_delta_us: Optional[float] = None
  note: str = ""


@dataclass(frozen=True)
class Plan:
  """A validated-shape plan document (shape only; legality is `validate_plan`)."""

  computations: dict[str, list[str]]
  module: Optional[str] = None
  input_checksum: Optional[str] = None
  profile: Optional[dict[str, Any]] = None
  moves: tuple[Move, ...] = ()
  version: int = PLAN_VERSION

  # ---- construction -----------------------------------------------------

  @classmethod
  def for_module(
      cls,
      module: Module,
      baseline: dict[str, list[str]],
      computations: dict[str, list[str]],
      *,
      profile: Optional[Profile] = None,
      moves: tuple[Move, ...] = (),
  ) -> "Plan":
    """A plan stamped with the identity of `module` as scheduled by XLA (`baseline`)."""
    return cls(
        computations={k: list(v) for k, v in computations.items()},
        module=module.name,
        input_checksum=input_checksum(baseline),
        profile=profile_identity(profile) if profile is not None else None,
        moves=tuple(moves),
    )

  @classmethod
  def from_dict(cls, doc: Any) -> "Plan":
    """Parse and shape-check a decoded JSON document."""
    if not isinstance(doc, dict):
      raise PlanFormatError("plan must be a JSON object")
    version = doc.get("version", PLAN_VERSION)
    if version != PLAN_VERSION:
      raise PlanFormatError(f"unsupported plan version {version!r}; this code reads version {PLAN_VERSION}")
    comps = doc.get("computations")
    if not isinstance(comps, dict) or not comps:
      raise PlanFormatError('plan needs a non-empty "computations" object')
    for name, order in comps.items():
      if not isinstance(order, list) or not all(isinstance(n, str) for n in order):
        raise PlanFormatError(f"computation {name!r} must map to a list of instruction names")
    for key in ("module", "input_checksum"):
      if key in doc and doc[key] is not None and not isinstance(doc[key], str):
        raise PlanFormatError(f"{key!r} must be a string")
    prof = doc.get("profile")
    if prof is not None and not (
        isinstance(prof, dict) and isinstance(prof.get("sha256"), str) and isinstance(prof.get("entries"), int)
    ):
      raise PlanFormatError('"profile" must be {"entries": int, "sha256": str}')
    moves = []
    for m in doc.get("moves", []):
      if (
          not isinstance(m, dict)
          or not isinstance(m.get("computation"), str)
          or not isinstance(m.get("instruction"), str)
      ):
        raise PlanFormatError('each move needs string "computation" and "instruction"')
      moves.append(
          Move(
              m["computation"],
              m["instruction"],
              m.get("to_before"),
              m.get("playbook_step"),
              m.get("expected_delta_us"),
              m.get("note", ""),
          )
      )
    return cls(
        computations={k: list(v) for k, v in comps.items()},
        module=doc.get("module"),
        input_checksum=doc.get("input_checksum"),
        profile=dict(prof) if prof else None,
        moves=tuple(moves),
    )

  @classmethod
  def loads(cls, text: str) -> "Plan":
    try:
      doc = json.loads(text)
    except ValueError as exc:
      raise PlanFormatError(f"plan is not valid JSON: {exc}") from exc
    return cls.from_dict(doc)

  @classmethod
  def load(cls, path: str) -> "Plan":
    """Read a plan file (raises OSError or PlanFormatError)."""
    with open(path, encoding="utf-8") as f:
      return cls.loads(f.read())

  # ---- serialization ----------------------------------------------------

  def to_dict(self) -> dict[str, Any]:
    """The document as plain data; optional fields are omitted when unset."""
    doc: dict[str, Any] = {"version": self.version, "computations": self.computations}
    if self.module is not None:
      doc["module"] = self.module
    if self.input_checksum is not None:
      doc["input_checksum"] = self.input_checksum
    if self.profile is not None:
      doc["profile"] = self.profile
    if self.moves:
      doc["moves"] = [{k: v for k, v in m.__dict__.items() if v not in (None, "")} for m in self.moves]
    return doc

  def dumps(self) -> str:
    """Deterministic JSON text; equal plans give equal bytes on every rank."""
    return _canonical(self.to_dict()) + "\n"

  def save(self, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
      f.write(self.dumps())

  def digest(self) -> str:
    """SHA-256 of the canonical text: what ranks compare to know they hold the same plan."""
    return hashlib.sha256(self.dumps().encode("utf-8")).hexdigest()

  @property
  def has_identity(self) -> bool:
    return self.module is not None and self.input_checksum is not None


@dataclass(frozen=True)
class IdentityOptions:
  """How strictly to match a plan's identity against the module being compiled."""

  require_identity: bool = False
  strict_profile: bool = False


def verify_against(
    plan: Plan,
    module: Module,
    baseline: dict[str, list[str]],
    profile: Optional[Profile] = None,
    options: IdentityOptions = IdentityOptions(),
) -> list[Violation]:
  """Does this plan belong to this module (and profile)? Errors mean do not apply.

  Checks identity only; legality of the orders is `legality.validate_plan`.
  """
  out: list[Violation] = []
  if plan.module is not None and plan.module != module.name:
    out.append(Violation("PLAN_MODULE_MISMATCH", f"plan is for module {plan.module!r}, this is {module.name!r}"))
  if plan.input_checksum is None:
    if options.require_identity:
      out.append(Violation("PLAN_MISSING_IDENTITY", "plan has no input_checksum; it cannot be tied to this module"))
  elif plan.input_checksum != input_checksum(baseline):
    out.append(
        Violation(
            "PLAN_INPUT_MISMATCH",
            "the module's XLA schedule differs from the one the plan was made for "
            "(instruction names or order changed); regenerate the plan",
        )
    )
  if options.require_identity and plan.module is None:
    out.append(Violation("PLAN_MISSING_IDENTITY", "plan has no module name"))
  if plan.profile is not None:
    sev = Severity.ERROR if options.strict_profile else Severity.WARNING
    if profile is None:
      out.append(
          Violation(
              "PLAN_PROFILE_UNCHECKED",
              "plan records a profile but none was provided to compare",
              severity=Severity.WARNING,
          )
      )
    elif profile_identity(profile) != plan.profile:
      out.append(
          Violation("PLAN_PROFILE_MISMATCH", "the profile differs from the one the plan was scored with", severity=sev)
      )
  for m in plan.moves:
    order = plan.computations.get(m.computation)
    if order is None or m.instruction not in order or (m.to_before is not None and m.to_before not in order):
      out.append(
          Violation(
              "PLAN_BAD_MOVE",
              f"move of {m.instruction!r} in {m.computation!r} refers to a computation or instruction not in the plan",
              m.instruction,
          )
      )
  return out
