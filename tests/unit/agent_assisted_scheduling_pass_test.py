"""Tests for the online agent-assisted scheduling pass (fake agent, fake KV store)."""

import json
import os
import stat
import sys

from absl.testing import absltest

from maxtext.utils.agent_assisted_scheduling import agent_runner, kv_share, plan_store
from maxtext.utils.agent_assisted_scheduling.agent_runner import AgentRequest, AgentResult, ClaudeCliRunner
from maxtext.utils.agent_assisted_scheduling.plan import Plan
from maxtext.utils.agent_assisted_scheduling.profile import Profile

try:
  from jax._src.lib import hlo as _hlo
  from maxtext.utils.agent_assisted_scheduling import agent_pass
except ImportError:  # jaxlib not installed: the pass tests are skipped
  _hlo = None

# The start sits behind `mm`; hoisting it ahead of `mm` hides more of its latency.
_HLO = """
HloModule tiny, is_scheduled=true

%async_comp (p: bf16[16,64]) -> bf16[128,64] {
  %p = bf16[16,64]{1,0} parameter(0)
  ROOT %ag = bf16[128,64]{1,0} all-gather(%p), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, use_global_device_ids=true
}

ENTRY %main (x: bf16[16,64], w: bf16[64,64]) -> (bf16[128,64], bf16[16,64]) {
  %x = bf16[16,64]{1,0} parameter(0)
  %w = bf16[64,64]{1,0} parameter(1)
  %mm = bf16[16,64]{1,0} dot(%x, %w), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %ag-start = ((bf16[16,64]{1,0}), bf16[128,64]{1,0}) async-start(%x), calls=%async_comp
  %mm2 = bf16[16,64]{1,0} dot(%mm, %w), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %ag-done = bf16[128,64]{1,0} async-done(%ag-start), calls=%async_comp
  ROOT %out = (bf16[128,64], bf16[16,64]) tuple(%ag-done, %mm2)
}
"""
_BASE = ["x", "w", "mm", "ag-start", "mm2", "ag-done", "out"]
_GOOD = ["x", "w", "ag-start", "mm", "mm2", "ag-done", "out"]
_PROFILE = Profile.from_costs({"mm": 100.0, "mm2": 250.0, "ag": 900.0})


class _FakeRunner:
  """Stands in for the agent: writes a canned plan (or nothing) to the requested path."""

  def __init__(self, plan=None, error=None, raises=None):
    self.plan, self.error, self.raises, self.calls = plan, error, raises, []

  def run(self, request: AgentRequest) -> AgentResult:
    self.calls.append(request)
    if self.raises:
      raise self.raises
    if self.plan is not None:
      with open(request.plan_path, "w", encoding="utf-8") as f:
        json.dump({"computations": self.plan}, f)
    return AgentResult(self.plan is not None, 0.1, 0 if self.plan is not None else 1, error=self.error)


class _FakeKv:
  """Coordination-service stand-in shared by two simulated ranks."""

  def __init__(self):
    self.data = {}

  def key_value_set_bytes(self, key, value):
    assert key not in self.data, f"key {key} set twice"
    self.data[key] = value

  def blocking_key_value_get_bytes(self, key, timeout_ms):
    if key not in self.data:
      raise TimeoutError(key)
    return self.data[key]


def _module_bytes():
  return _hlo.hlo_module_from_text(_HLO).as_serialized_hlo_module_proto()


def _order(module_bytes):
  m = _hlo.HloModule.from_serialized_hlo_module_proto(module_bytes)
  comp = next(c for c in m.make_nonfusion_computations() if c.name == "main")
  return [i.name for i in m.schedule().sequence(comp)]


def _config(**kw):
  return agent_pass.PassConfig(**{"min_exposed_us": 1.0, **kw})


@absltest.skipIf(_hlo is None, "jaxlib not available")
class AgentPassTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    agent_pass._stores.clear()  # pylint: disable=protected-access
    agent_pass._invocations.clear()  # pylint: disable=protected-access
    self.profile_file = os.path.join(self.create_tempdir().full_path, "p.fdo_profile")
    with open(self.profile_file, "w", encoding="utf-8") as f:
      f.write(_PROFILE.to_text())

  def _run(self, runner, *, config=None, rank=0, kv=None, flags="", module_config=b""):
    cfg = config or _config(profile_path=self.profile_file)
    return agent_pass.run_pass(
        _module_bytes(), module_config, config=cfg, runner=runner, kv_client=kv, rank=rank, xla_flags=flags
    )

  def test_applies_good_plan_and_caches_in_memory(self):
    runner = _FakeRunner(_GOOD_PLAN := {"main": _GOOD})
    out = self._run(runner)
    self.assertEqual(_order(out), _GOOD)
    self.assertEqual(len(runner.calls), 1)
    # The same module compiled again is served from the cache; the agent is not called twice.
    self.assertEqual(_order(self._run(runner)), _GOOD)
    self.assertEqual(len(runner.calls), 1)
    self.assertEqual(_GOOD_PLAN, {"main": _GOOD})

  def test_agent_request_contents(self):
    runner = _FakeRunner({"main": _GOOD})
    self._run(runner, config=_config(profile_path=self.profile_file, timeout_s=180.0))
    req = runner.calls[0]
    self.assertEqual(req.timeout_s, 180.0)
    self.assertTrue(req.module_path.endswith("tiny.hlo.pb"))
    self.assertIn("exposed", req.summary)

  def test_records_and_replays_from_xla_dump_to(self):
    dump = self.create_tempdir().full_path
    flags = f"--xla_dump_to={dump} --xla_gpu_foo=1"
    runner = _FakeRunner({"main": _GOOD})
    self.assertEqual(_order(self._run(runner, flags=flags)), _GOOD)
    files = sorted(os.listdir(dump))
    plans = [f for f in files if f.endswith(".json") and ".agent_plan." in f and "agent_log" not in f]
    logs = [f for f in files if f.endswith(".agent_log.json")]
    self.assertLen(plans, 1)
    self.assertLen(logs, 1)
    self.assertTrue(plans[0].startswith("tiny.agent_plan."))
    plan = Plan.load(os.path.join(dump, plans[0]))
    self.assertEqual((plan.module, plan.computations), ("tiny", {"main": _GOOD}))
    with open(os.path.join(dump, logs[0]), encoding="utf-8") as f:
      log = json.load(f)
    self.assertTrue(log["agent"]["plan_written"])
    self.assertLess(log["static_after_us"], log["static_exposed_us"])
    # A new process (empty memory cache) replays the recorded plan without the agent.
    agent_pass._stores.clear()  # pylint: disable=protected-access
    replay = _FakeRunner(None)
    out = self._run(replay, config=_config(profile_path=self.profile_file, replay_only=True), flags=flags)
    self.assertEqual(_order(out), _GOOD)
    self.assertEqual(replay.calls, [])

  def test_without_xla_dump_to_nothing_is_written(self):
    self._run(_FakeRunner({"main": _GOOD}), flags="--xla_gpu_foo=1")
    store = next(iter(agent_pass._stores.values()))  # pylint: disable=protected-access
    self.assertIsNone(store.directory)
    self.assertLen(store._memory, 1)  # pylint: disable=protected-access  # cached in memory only

  def test_replay_only_without_recorded_plan_keeps_xla_schedule(self):
    runner = _FakeRunner({"main": _GOOD})
    self.assertIsNone(self._run(runner, config=_config(profile_path=self.profile_file, replay_only=True)))
    self.assertEqual(runner.calls, [])

  def test_dump_dir_sources(self):
    self.assertEqual(plan_store.dump_dir(xla_flags="--a=1 --xla_dump_to=/tmp/d --b"), "/tmp/d")
    self.assertEqual(plan_store.dump_dir(xla_flags="--xla_dump_to /tmp/d2"), "/tmp/d2")
    self.assertEqual(plan_store.dump_dir(xla_flags='--xla_dump_to="/tmp/with space"'), "/tmp/with space")
    self.assertIsNone(plan_store.dump_dir(xla_flags="--xla_dump_to=-"))
    self.assertIsNone(plan_store.dump_dir(xla_flags="--other=1"))
    self.assertIsNone(plan_store.dump_dir())
    # module_config: HloModuleConfigProto.debug_options (14) -> DebugOptions.xla_dump_to (109)
    inner = b"\xea\x06" + bytes([len(b"/cfg/dir")]) + b"/cfg/dir"  # field 109, wire type 2
    cfg = b"\x72" + bytes([len(inner)]) + inner  # field 14, wire type 2
    self.assertEqual(plan_store.dump_dir(cfg, "--xla_dump_to=/flags/dir"), "/cfg/dir")

  def test_no_profile_or_no_async_keeps_xla_schedule(self):
    runner = _FakeRunner({"main": _GOOD})
    out = agent_pass.run_pass(_module_bytes(), b"", config=_config(), runner=runner, xla_flags="")
    self.assertIsNone(out)
    self.assertEqual(runner.calls, [])

  def test_small_exposed_time_skips_agent(self):
    runner = _FakeRunner({"main": _GOOD})
    cfg = _config(profile_path=self.profile_file, min_exposed_us=10_000.0)
    self.assertIsNone(self._run(runner, config=cfg))
    self.assertEqual(runner.calls, [])

  def test_rejected_plans_keep_xla_schedule(self):
    bad_plans = {
        "control order / operand order": {"main": ["x", "w", "ag-done", "ag-start", "mm", "mm2", "out"]},
        "cross computation": {"main": _BASE + ["ag"]},
        "missing instruction": {"main": _BASE[:-1]},
        "unknown computation": {"nope": _BASE},
        "no improvement": {"main": list(_BASE)},
    }
    for name, plan in bad_plans.items():
      agent_pass._stores.clear()  # pylint: disable=protected-access
      runner = _FakeRunner(plan)
      self.assertIsNone(self._run(runner), msg=name)
      self.assertLen(runner.calls, 1, msg=name)

  def test_agent_failures_keep_xla_schedule(self):
    for runner in (_FakeRunner(None, error="timeout"), _FakeRunner(raises=RuntimeError("boom"))):
      agent_pass._stores.clear()  # pylint: disable=protected-access
      self.assertIsNone(self._run(runner))

  def test_agent_cannot_forge_identity(self):
    class Forger(_FakeRunner):

      def run(self, request):
        self.calls.append(request)
        with open(request.plan_path, "w", encoding="utf-8") as f:
          json.dump({"module": "other", "input_checksum": "0" * 64, "computations": {"main": _GOOD}}, f)
        return AgentResult(True, 0.1, 0)

    out = self._run(Forger())
    self.assertEqual(_order(out), _GOOD)  # the pass re-stamped it with the real identity

  def test_two_ranks_apply_identical_bytes_and_call_agent_once(self):
    kv = _FakeKv()
    runner = _FakeRunner({"main": _GOOD})
    out0 = self._run(runner, kv=kv, rank=0)
    agent_pass._invocations.clear()  # pylint: disable=protected-access  # rank 1 is another process
    out1 = self._run(_FakeRunner(None), kv=kv, rank=1)
    self.assertEqual(out0, out1)
    self.assertEqual(_order(out1), _GOOD)
    self.assertLen(runner.calls, 1)

  def test_rank_zero_failure_publishes_no_plan_and_others_keep_xla(self):
    kv = _FakeKv()
    self.assertIsNone(self._run(_FakeRunner(raises=RuntimeError("boom")), kv=kv, rank=0))
    agent_pass._invocations.clear()  # pylint: disable=protected-access
    self.assertIsNone(self._run(_FakeRunner(None), kv=kv, rank=1))
    payload = next(iter(kv.data.values()))
    self.assertTrue(payload.startswith(b"AAS1"))
    self.assertTrue(payload.endswith(b"\n"))  # checksum header only: no module attached

  def test_rank_refuses_payload_made_from_a_different_input(self):
    kv = _FakeKv()
    self._run(_FakeRunner({"main": _GOOD}), kv=kv, rank=0)
    key = next(iter(kv.data))
    kv.data[key] = b"AAS1" + b"0" * 64 + b"\n" + b"module-bytes"
    agent_pass._invocations.clear()  # pylint: disable=protected-access
    self.assertIsNone(self._run(_FakeRunner(None), kv=kv, rank=1))

  def test_other_rank_times_out_to_xla_schedule(self):
    self.assertIsNone(self._run(_FakeRunner(None), kv=_FakeKv(), rank=1))


class KvShareTest(absltest.TestCase):

  def test_rank_zero_publishes_even_when_produce_raises(self):
    kv = _FakeKv()

    def produce():
      raise RuntimeError("x")

    self.assertEqual(kv_share.share(kv, "k", rank=0, produce=produce, timeout_ms=1), b"")
    self.assertEqual(kv_share.share(kv, "k", rank=1, produce=produce, timeout_ms=1), b"")

  def test_malformed_payload_and_missing_client(self):
    kv = _FakeKv()
    kv.data["k"] = b"junk"
    with self.assertRaises(kv_share.ShareTimeout):
      kv_share.share(kv, "k", rank=1, produce=bytes, timeout_ms=1)
    with self.assertRaises(kv_share.ShareTimeout):
      kv_share.share(None, "k", rank=1, produce=bytes, timeout_ms=1)
    self.assertEqual(kv_share.share(None, "k", rank=0, produce=lambda: b"p", timeout_ms=1), b"p")


class ClaudeCliRunnerTest(absltest.TestCase):

  def _request(self, timeout_s=30.0):
    d = self.create_tempdir().full_path
    return AgentRequest(
        module="m",
        workdir=d,
        module_path=f"{d}/m.pb",
        profile_path=f"{d}/m.fdo_profile",
        plan_path=f"{d}/plan.json",
        timeout_s=timeout_s,
    )

  def _fake_claude(self, body):
    path = os.path.join(self.create_tempdir().full_path, "claude")
    with open(path, "w", encoding="utf-8") as f:
      f.write(f"#!{sys.executable}\n{body}\n")
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path

  def test_command_line(self):
    req = self._request()
    cmd = ClaudeCliRunner("claude", model="some-model", max_budget_usd=2.5).command(req)
    self.assertEqual(cmd[:2], ["claude", "-p"])
    prompt = cmd[2]
    for text in (
        req.module_path,
        req.profile_path,
        req.plan_path,
        "180",
        "agent-assisted-scheduling-workflow",
        "cross-computation",
    ):
      self.assertIn(text if text != "180" else str(int(req.timeout_s)), prompt)
    self.assertIn("--no-session-persistence", cmd)
    self.assertEqual(cmd[cmd.index("--permission-mode") + 1], "dontAsk")
    self.assertEqual(cmd[cmd.index("--tools") + 1], "Read,Write,Bash")
    allowed = cmd[cmd.index("--allowedTools") + 1 : cmd.index("--add-dir")]
    self.assertEqual(allowed[:2], ["Read", "Write"])
    self.assertEqual(allowed[2], f"Bash({agent_runner.cli_prefix(req.python)} *)")
    self.assertEqual(cmd[cmd.index("--model") + 1], "some-model")
    self.assertEqual(cmd[cmd.index("--max-budget-usd") + 1], "2.5")
    self.assertNotIn("--dangerously-skip-permissions", cmd)

  def test_default_budget_is_three_minutes(self):
    self.assertEqual(agent_runner.DEFAULT_TIMEOUT_S, 180.0)

  def test_runs_agent_and_reports_plan(self):
    req = self._request()
    fake = self._fake_claude(
        "import json,sys\nopen(%r,'w').write(json.dumps({'computations': {'c': ['a']}}))\nprint('DONE 1')" % req.plan_path
    )
    res = ClaudeCliRunner(fake).run(req)
    self.assertTrue(res.plan_written)
    self.assertEqual(res.returncode, 0)
    self.assertIn("DONE 1", res.stdout_tail)
    self.assertIsNone(res.error)

  def test_agent_that_writes_nothing(self):
    res = ClaudeCliRunner(self._fake_claude("print('no safe move')")).run(self._request())
    self.assertFalse(res.plan_written)

  def test_timeout_kills_the_process_group(self):
    req = self._request(timeout_s=1.0)
    marker = os.path.join(req.workdir, "survived")
    fake = self._fake_claude(
        """import subprocess, sys, time
marker = 'MARKER'
subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(2);open(' + repr(marker) + ',"w").write("x")'])
time.sleep(60)""".replace(
            "MARKER", marker
        )
    )
    res = ClaudeCliRunner(fake).run(req)
    self.assertTrue(res.timed_out)
    self.assertEqual(res.error, "timeout")
    self.assertLess(res.duration_s, 10.0)
    import time  # pylint: disable=import-outside-toplevel

    time.sleep(3)
    self.assertFalse(os.path.exists(marker))  # the child was killed with the group

  def test_missing_binary_is_reported_not_raised(self):
    res = ClaudeCliRunner("/nonexistent/claude").run(self._request())
    self.assertFalse(res.plan_written)
    self.assertIn("cannot start", res.error)


if __name__ == "__main__":
  absltest.main()
