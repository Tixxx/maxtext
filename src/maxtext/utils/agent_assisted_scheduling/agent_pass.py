"""Online agent-assisted scheduling as a `POST_SCHEDULER` XLA transform.

For each scheduled module that has async collectives and an FDO profile:

  1. Rank 0 looks for a cached plan (memory, and `<xla_dump_to>/` if XLA is dumping).
  2. On a miss it writes the module and profile to a temp dir and runs the agent
     (`claude -p`, 3 minute wall clock, only Read/Write/the package CLI), then
     takes the plan the agent wrote, stamps it with the module's identity, checks
     it, and records plan + agent log in `xla_dump_to` if set.
  3. Rank 0 applies the plan with `apply_plan` (identity required, legality
     re-checked, XLA's own `update()`/`verify()`) and publishes the resulting
     serialized module through the coordination-service KV store (`kv_share`).
  4. Every rank returns exactly those bytes (after checking the payload was made
     from the same input schedule as its own), or None to keep XLA's schedule.
     The module is shared, not the plan: serializing a parsed module is not
     byte-stable (the module id differs per parse), so ranks that each applied the
     plan locally would hand XLA different bytes.

Nothing the agent writes is trusted: it is re-stamped, re-validated and re-applied
here. Any error anywhere means "keep XLA's schedule". The agent is never called
more than once per (module, input schedule, profile) per process, and not at all
when a recorded plan exists or `AGENT_SCHEDULING_REPLAY_ONLY=1`.

Environment (all optional; the pass is off unless ENABLE=1):
  AGENT_SCHEDULING_ENABLE=1         register the pass
  AGENT_SCHEDULING_TIMEOUT_S=180    agent wall-clock budget per module
  AGENT_SCHEDULING_REPLAY_ONLY=1    use recorded plans only, never call the agent
  AGENT_SCHEDULING_FDO_PROFILE=path text/binary FDO profile used if the compile
                                    options carry none (some plugins pass none)
  AGENT_SCHEDULING_MIN_EXPOSED_US=100  skip the agent if the module's static
                                    exposed collective time is below this
  AGENT_SCHEDULING_CLAUDE_BIN=claude, AGENT_SCHEDULING_MODEL,
  AGENT_SCHEDULING_MAX_BUDGET_USD   passed to the `claude` CLI

Register this pass or `gpu_collective_overlap_pass`, not both: they would reorder the
same schedule.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Optional

from maxtext.utils.agent_assisted_scheduling import agent_runner, kv_share, plan_store
from maxtext.utils.agent_assisted_scheduling.analysis import analyze_module, check_plan
from maxtext.utils.agent_assisted_scheduling.hlo_adapter import apply_plan, load_module
from maxtext.utils.agent_assisted_scheduling.plan import IdentityOptions, Plan, PlanFormatError, input_checksum, profile_identity
from maxtext.utils.agent_assisted_scheduling.profile import Profile

_logger = logging.getLogger(__name__)
_NAME = "agent_assisted_scheduling"
_STRICT = IdentityOptions(require_identity=True)


@dataclass(frozen=True)
class PassConfig:
  """Settings for one process; `from_env` reads the AGENT_SCHEDULING_* variables."""

  timeout_s: float = agent_runner.DEFAULT_TIMEOUT_S
  share_margin_s: float = 120.0  # other ranks wait for the agent budget plus this
  replay_only: bool = False
  profile_path: Optional[str] = None
  min_exposed_us: float = 100.0
  claude_bin: str = "claude"
  model: Optional[str] = None
  max_budget_usd: Optional[float] = None

  @classmethod
  def from_env(cls, env: Optional[dict[str, str]] = None) -> "PassConfig":
    """Settings from `AGENT_SCHEDULING_*` environment variables."""
    env = os.environ if env is None else env

    def get(key, default=None):
      return env.get(f"AGENT_SCHEDULING_{key}", default)

    budget = get("MAX_BUDGET_USD")
    return cls(
        timeout_s=float(get("TIMEOUT_S", agent_runner.DEFAULT_TIMEOUT_S)),
        replay_only=get("REPLAY_ONLY") == "1",
        profile_path=get("FDO_PROFILE"),
        min_exposed_us=float(get("MIN_EXPOSED_US", 100.0)),
        claude_bin=get("CLAUDE_BIN", "claude"),
        model=get("MODEL"),
        max_budget_usd=float(budget) if budget else None,
    )


# Process-wide state: one store per dump directory, and per-key invocation counters
# so the same module compiled twice uses distinct (but rank-consistent) KV keys.
_stores: dict[Optional[str], plan_store.PlanStore] = {}
_invocations: dict[str, int] = {}


def _store_for(directory: Optional[str]) -> plan_store.PlanStore:
  if directory not in _stores:
    _stores[directory] = plan_store.PlanStore(directory)
  return _stores[directory]


def _resolve_profile(module_config: bytes, config: PassConfig) -> Profile:
  profile = Profile.from_module_config(module_config) if module_config else Profile()
  if not profile.costs and config.profile_path:
    profile = Profile.from_file(config.profile_path)
  return profile


_HEADER_SEP = b"\n"


def _payload(checksum: str, module_bytes: bytes = b"") -> bytes:
  """`<input checksum>\\n<module bytes>`; empty module bytes mean "no plan"."""
  return checksum.encode("ascii") + _HEADER_SEP + module_bytes


def _apply_to_payload(loaded, plan: Plan, checksum: str) -> bytes:
  """Rank 0: apply `plan` (identity required) and return the payload carrying the new module."""
  result = apply_plan(loaded, plan, identity=_STRICT)
  if not result.applied:
    _logger.warning("%s [%s]: plan not applied: %s", _NAME, loaded.module.name, result.reason)
    return _payload(checksum)
  if result.reason != "applied":  # the plan equals XLA's schedule: nothing to change
    return _payload(checksum)
  return _payload(checksum, loaded.hlo_module.as_serialized_hlo_module_proto())


def _produce(
    *,
    serialized_hlo: bytes,
    loaded,
    profile: Profile,
    store: plan_store.PlanStore,
    config: PassConfig,
    runner: agent_runner.AgentRunner,
    checksum: str,
    profile_sha: str,
) -> bytes:
  """Rank 0: the payload (see `_payload`), with the scheduled module if a plan applied. Never raises."""
  module = loaded.module
  none = _payload(checksum)
  log: dict[str, Any] = {"module": module.name, "input_checksum": checksum, "profile_sha256": profile_sha}
  started = time.monotonic()
  try:
    cached = store.load(module.name, checksum, profile_sha)
    if cached is not None:
      _logger.info("%s [%s]: using recorded plan (digest %s).", _NAME, module.name, cached.digest()[:12])
      log["source"] = "recorded plan"
      return _apply_to_payload(loaded, cached, checksum)
    if config.replay_only:
      _logger.info("%s [%s]: replay-only and no recorded plan; keeping XLA's schedule.", _NAME, module.name)
      return none
    report = analyze_module(module, loaded.baseline, while_bodies=loaded.while_bodies)
    log["static_exposed_us"] = report.static_exposed_us
    if report.static_exposed_us < config.min_exposed_us:
      _logger.info(
          "%s [%s]: static exposed %.1f us < %.1f us; not calling the agent.",
          _NAME,
          module.name,
          report.static_exposed_us,
          config.min_exposed_us,
      )
      return none
    with tempfile.TemporaryDirectory(prefix="agent_scheduling_") as workdir:
      module_path = os.path.join(workdir, f"{module.name}.hlo.pb")
      profile_path = os.path.join(workdir, f"{module.name}.fdo_profile")
      plan_path = os.path.join(workdir, "plan.json")
      with open(module_path, "wb") as f:
        f.write(serialized_hlo)
      with open(profile_path, "w", encoding="utf-8") as f:
        f.write(profile.to_text())
      request = agent_runner.AgentRequest(
          module=module.name,
          workdir=workdir,
          module_path=module_path,
          profile_path=profile_path,
          plan_path=plan_path,
          timeout_s=config.timeout_s,
          summary=(
              f"It has {sum(len(c.rows) for c in report.computations)} async collectives; "
              f"static exposed time is {report.static_exposed_us:.0f} us."
          ),
      )
      result = runner.run(request)
      log["agent"] = {
          k: getattr(result, k)
          for k in (
              "plan_written",
              "duration_s",
              "returncode",
              "timed_out",
              "error",
              "stdout_tail",
              "stderr_tail",
              "command",
          )
      }
      if not result.plan_written:
        _logger.info("%s [%s]: agent proposed no plan (%s).", _NAME, module.name, result.error or "no move")
        return none
      proposed = Plan.load(plan_path)
    # Stamp with OUR view of the module: the agent cannot assert identity.
    plan = Plan.for_module(module, loaded.baseline, proposed.computations, profile=profile, moves=proposed.moves)
    verdict = check_plan(
        module, loaded.baseline, plan, profile=profile, identity=_STRICT, while_bodies=loaded.while_bodies
    )
    log["violations"] = [(v.severity.value, v.code, v.message) for v in verdict.violations]
    if not verdict.accepted:
      _logger.warning("%s [%s]: agent plan rejected: %s", _NAME, module.name, log["violations"][:3])
      return none
    before, after = verdict.before, verdict.after
    log.update(
        static_after_us=after.static_exposed_us, sim_before_us=before.sim_stall_us, sim_after_us=after.sim_stall_us
    )
    if after.static_exposed_us >= before.static_exposed_us and after.sim_stall_us >= before.sim_stall_us:
      _logger.info("%s [%s]: agent plan does not reduce exposed time; discarding.", _NAME, module.name)
      return none
    payload = _apply_to_payload(loaded, plan, checksum)
    if payload == none:
      return none
    path = store.record(plan)  # only a plan that applied is recorded
    log["source"] = "agent"
    _logger.info(
        "%s [%s]: plan accepted in %.1fs: static exposed %.1f -> %.1f us, stream-sim %.1f -> %.1f us%s.",
        _NAME,
        module.name,
        time.monotonic() - started,
        before.static_exposed_us,
        after.static_exposed_us,
        before.sim_stall_us,
        after.sim_stall_us,
        f", recorded to {path}" if path else "",
    )
    return payload
  except (OSError, PlanFormatError, ValueError, RuntimeError) as exc:
    log["error"] = f"{type(exc).__name__}: {exc}"
    _logger.warning("%s [%s]: no plan: %s", _NAME, module.name, log["error"])
    return none
  except Exception as exc:  # pylint: disable=broad-exception-caught
    log["error"] = f"unexpected {type(exc).__name__}: {exc}"
    _logger.exception("%s [%s]: unexpected failure; keeping XLA's schedule.", _NAME, module.name)
    return none
  finally:
    log["total_s"] = time.monotonic() - started
    try:
      store.record_log(module.name, checksum, profile_sha, log)
    except OSError:
      pass


def run_pass(
    serialized_hlo: bytes,
    module_config: bytes = b"",
    *,
    config: PassConfig,
    runner: Optional[agent_runner.AgentRunner] = None,
    kv_client: Optional[kv_share.KvClient] = None,
    rank: int = 0,
    xla_flags: Optional[str] = None,
) -> Optional[bytes]:
  """The transform body: the new serialized module, or None to keep XLA's schedule."""
  from jax._src.lib import hlo  # pylint: disable=import-outside-toplevel

  hlo_module = hlo.HloModule.from_serialized_hlo_module_proto(serialized_hlo)
  profile = _resolve_profile(module_config, config)
  if not profile.costs:
    _logger.info("%s [%s]: no FDO profile yet; keeping XLA's schedule.", _NAME, hlo_module.name)
    return None
  try:
    loaded = load_module(hlo_module, profile)
  except ValueError as exc:
    _logger.info("%s [%s]: %s", _NAME, hlo_module.name, exc)
    return None
  if not any(c.async_pairs() for c in loaded.module.computations.values()):
    return None

  checksum = input_checksum(loaded.baseline)
  psha = profile_identity(profile)["sha256"]
  store = _store_for(
      plan_store.dump_dir(module_config, xla_flags if xla_flags is not None else os.environ.get("XLA_FLAGS"))
  )
  stem = plan_store.plan_stem(loaded.module.name, checksum, psha)
  _invocations[stem] = _invocations.get(stem, 0) + 1
  key = f"{_NAME}/{stem}/{_invocations[stem]}"
  runner = runner or agent_runner.ClaudeCliRunner(config.claude_bin, config.model, config.max_budget_usd)

  try:
    payload = kv_share.share(
        kv_client,
        key,
        rank=rank,
        produce=lambda: _produce(
            serialized_hlo=serialized_hlo,
            loaded=loaded,
            profile=profile,
            store=store,
            config=config,
            runner=runner,
            checksum=checksum,
            profile_sha=psha,
        ),
        timeout_ms=int((config.timeout_s + config.share_margin_s) * 1000),
    )
  except kv_share.ShareTimeout as exc:
    _logger.warning("%s [%s]: %s; keeping XLA's schedule on this rank.", _NAME, loaded.module.name, exc)
    return None
  header, sep, module_bytes = payload.partition(_HEADER_SEP)
  if not sep or header.decode("ascii", "replace") != checksum:
    _logger.warning(
        "%s [%s]: the shared plan was made from a different input schedule; keeping XLA's schedule on this rank.",
        _NAME,
        loaded.module.name,
    )
    return None
  if not module_bytes:
    return None
  _logger.info("%s [%s]: using the scheduled module from rank 0 (rank %d).", _NAME, loaded.module.name, rank)
  return module_bytes


# ---- registration ---------------------------------------------------------

_registered = False


_distributed_client = kv_share.distributed_client


def register(config: Optional[PassConfig] = None, runner: Optional[agent_runner.AgentRunner] = None) -> bool:
  """Register the pass once if AGENT_SCHEDULING_ENABLE=1 (or `config` is given). True if registered."""
  global _registered
  if _registered or (config is None and os.environ.get("AGENT_SCHEDULING_ENABLE") != "1"):
    return _registered
  import jax.extend.xla as jex_xla  # pylint: disable=import-outside-toplevel

  config = config or PassConfig.from_env()

  def callback(hlo_module: bytes, *, module_config: bytes = b"") -> Optional[bytes]:
    try:
      client, rank = _distributed_client()
      return run_pass(hlo_module, module_config, config=config, runner=runner, kv_client=client, rank=rank)
    except Exception:  # pylint: disable=broad-exception-caught
      _logger.exception("%s: failed; keeping XLA's schedule.", _NAME)
      return None

  jex_xla.register_hlo_module_transformation(callback, name=_NAME, stage=jex_xla.PipelineStage.POST_SCHEDULER)
  _registered = True
  _logger.info(
      "%s: registered POST_SCHEDULER pass (timeout %.0fs, replay_only=%s).", _NAME, config.timeout_s, config.replay_only
  )
  return True
