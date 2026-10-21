"""Tests for the jax-free agent-assisted scheduling building blocks."""

import io
import json
import os
import struct

from absl.testing import absltest

from maxtext.utils.agent_assisted_scheduling import (
    AsyncRole,
    Computation,
    Inst,
    Severity,
    collective_exposures,
    coverage_gaps,
    earliest_legal_pos,
    hoist_start_to_floor,
    is_legal,
    simulate_comm_stream,
    total_exposed_us,
    validate_order,
)
from maxtext.utils.agent_assisted_scheduling import hlo_text
from maxtext.utils.agent_assisted_scheduling.analysis import analyze_module, check_plan
from maxtext.utils.agent_assisted_scheduling.legality import Severity as _Sev
from maxtext.utils.agent_assisted_scheduling.legality import validate_plan
from maxtext.utils.agent_assisted_scheduling.plan import (
    IdentityOptions,
    Move,
    Plan,
    PlanFormatError,
    input_checksum,
    profile_identity,
    verify_against,
)
from maxtext.utils.agent_assisted_scheduling.profile import Profile
from maxtext.utils.agent_assisted_scheduling.schedule_model import Module

try:
  from jax._src.lib import hlo as _hlo  # pylint: disable=g-import-not-at-top
  from maxtext.utils.agent_assisted_scheduling import cli
  from maxtext.utils.agent_assisted_scheduling.hlo_adapter import apply_plan, load_module
except ImportError:  # jaxlib not installed: adapter tests are skipped
  _hlo = None


def _start(name, operands=(), latency=500.0, **kw):
  return Inst(
      name,
      "all-gather-start",
      tuple(operands),
      async_role=AsyncRole.START,
      latency_us=latency,
      uses_comm_stream=True,
      **kw,
  )


def _done(name, start, **kw):
  return Inst(name, "all-gather-done", (start,), async_role=AsyncRole.DONE, **kw)


def _fus(name, operands=(), cost=0.0, **kw):
  return Inst(name, "fusion", tuple(operands), cost_us=cost, **kw)


def _codes(violations):
  return sorted(v.code for v in violations)


def _basic():
  """p0, a(100), b(200), c(300), w=convert(p0), ag-start(w), ag-done, use."""
  comp = Computation.from_instructions(
      "main",
      [
          Inst("p0", "parameter"),
          _fus("a", ["p0"], 100),
          _fus("b", ["a"], 200),
          _fus("c", ["b"], 300),
          Inst("w", "convert", ("p0",)),
          _start("s", ["w"]),
          _done("d", "s"),
          _fus("use", ["d", "c"], 50),
      ],
  )
  order = ["p0", "a", "b", "c", "w", "s", "d", "use"]
  return comp, order


class LegalityTest(absltest.TestCase):

  def test_valid_order_has_no_errors(self):
    comp, order = _basic()
    self.assertTrue(is_legal(comp, order))

  def test_duplicate_missing_unknown(self):
    comp, order = _basic()
    bad = [n for n in order if n != "c"] + ["a", "zzz"]
    self.assertEqual(_codes(validate_order(comp, bad)), ["DUPLICATE", "MISSING", "UNKNOWN"])

  def test_operand_order(self):
    comp, _ = _basic()
    bad = ["p0", "b", "a", "c", "w", "s", "d", "use"]
    self.assertIn("OPERAND_ORDER", _codes(validate_order(comp, bad)))

  def test_start_after_done_is_operand_violation(self):
    comp, _ = _basic()
    bad = ["p0", "a", "b", "c", "w", "d", "s", "use"]
    self.assertIn("OPERAND_ORDER", _codes(validate_order(comp, bad)))

  def test_control_order(self):
    comp = Computation.from_instructions(
        "c",
        [
            Inst("x", "fusion"),
            Inst("y", "fusion", control_predecessors=("x",)),
        ],
    )
    self.assertTrue(is_legal(comp, ["x", "y"]))
    self.assertEqual(_codes(validate_order(comp, ["y", "x"])), ["CONTROL_ORDER"])

  def test_async_pairing(self):
    comp = Computation.from_instructions("c", [_start("s"), _start("s2")])
    codes = _codes(validate_order(comp, ["s", "s2"]))
    self.assertEqual(codes, ["ASYNC_PAIR", "ASYNC_PAIR"])
    comp = Computation.from_instructions(
        "c",
        [
            _start("s"),
            _done("d1", "s"),
            _done("d2", "s"),
        ],
    )
    self.assertIn("ASYNC_PAIR", _codes(validate_order(comp, ["s", "d1", "d2"])))

  def test_start_used_by_non_done(self):
    comp = Computation.from_instructions(
        "c",
        [
            _start("s"),
            _done("d", "s"),
            _fus("bad", ["s"]),
        ],
    )
    self.assertIn("ASYNC_PAIR", _codes(validate_order(comp, ["s", "bad", "d"])))

  def test_sync_conversion_warning_only(self):
    comp, _ = _basic()
    order = ["p0", "a", "b", "c", "w", "s", "d", "use"]
    violations = validate_order(comp, order)
    warn = [v for v in violations if v.code == "SYNC_CONVERSION"]
    self.assertLen(warn, 1)
    self.assertEqual(warn[0].severity, Severity.WARNING)
    self.assertTrue(is_legal(comp, order))
    hoisted = ["p0", "w", "s", "a", "b", "c", "d", "use"]
    self.assertNotIn("SYNC_CONVERSION", _codes(validate_order(comp, hoisted)))

  def test_no_cap_on_in_flight_collectives(self):
    comp = Computation.from_instructions(
        "c",
        [
            _start("s1"),
            _start("s2"),
            _start("s3"),
            _fus("f", [], 10),
            _done("d1", "s1"),
            _done("d2", "s2"),
            _done("d3", "s3"),
        ],
    )
    self.assertEqual(validate_order(comp, ["s1", "s2", "s3", "f", "d1", "d2", "d3"]), [])

  def test_device_groups(self):
    def g(*ids):
      return (frozenset(ids),)

    comp = Computation.from_instructions(
        "c",
        [
            _start("s1", device_groups=g(0, 1, 2, 3)),
            _start("s2", device_groups=g(2, 3, 4, 5)),
            _start("s3", device_groups=g(4, 5, 6, 7)),
            _start("s4"),
            _fus("f", [], 10),
            _done("d1", "s1"),
            _done("d2", "s2"),
            _done("d3", "s3"),
            _done("d4", "s4"),
        ],
    )
    kw = {"check_device_groups": True}
    # s1 & s2 share devices {2,3}
    self.assertIn(
        "DEVICE_GROUP_OVERLAP", _codes(validate_order(comp, ["s1", "s2", "d1", "d2", "s3", "d3", "s4", "d4", "f"], **kw))
    )
    # s1 & s3 share none
    self.assertTrue(is_legal(comp, ["s1", "s3", "f", "d1", "d3", "s2", "d2", "s4", "d4"], **kw))
    # unknown groups while overlapping
    self.assertIn(
        "DEVICE_GROUPS_UNKNOWN", _codes(validate_order(comp, ["s1", "s4", "d1", "d4", "s2", "d2", "s3", "d3", "f"], **kw))
    )

  def test_in_place_reorder(self):
    comp = Computation.from_instructions(
        "c",
        [
            Inst("host", "parameter"),
            Inst("reader", "fusion", ("host",)),
            Inst("writer", "dynamic-update-slice", ("host",), writes_operand="host"),
        ],
    )
    baseline = ["host", "reader", "writer"]
    self.assertTrue(is_legal(comp, baseline, baseline=baseline))
    swapped = ["host", "writer", "reader"]
    self.assertEqual(_codes(validate_order(comp, swapped, baseline=baseline)), ["IN_PLACE_REORDER"])
    self.assertTrue(is_legal(comp, swapped))  # no baseline, no check


class PlacementTest(absltest.TestCase):

  def test_floor_through_trivial_chain(self):
    comp, order = _basic()
    r = earliest_legal_pos(comp, order, "s")
    self.assertEqual(r.floor, 0)
    self.assertEqual(r.movable, ("p0", "w"))
    self.assertIsNone(r.blocker)
    self.assertEqual(hoist_start_to_floor(comp, order, "s"), ["p0", "w", "s", "a", "b", "c", "d", "use"])

  def test_non_trivial_producer_pins_floor(self):
    comp = Computation.from_instructions(
        "c",
        [
            Inst("p0", "parameter"),
            _fus("a", ["p0"], 100),
            _fus("b", ["a"], 100),
            Inst("w", "convert", ("b",)),
            _start("s", ["w"]),
            _done("d", "s"),
        ],
    )
    order = ["p0", "a", "b", "w", "s", "d"]
    r = earliest_legal_pos(comp, order, "s")
    self.assertEqual(r.floor, 3)
    self.assertEqual(r.blocker, "b")
    self.assertEqual(r.movable, ("w",))
    self.assertEqual(hoist_start_to_floor(comp, order, "s"), order)  # no room

  def test_control_predecessor_pins_floor(self):
    comp = Computation.from_instructions(
        "c",
        [
            Inst("p0", "parameter"),
            _fus("a", ["p0"], 100),
            _fus("b", ["p0"], 100),
            _start("s", ["p0"], control_predecessors=("a",)),
            _done("d", "s"),
        ],
    )
    order = ["p0", "a", "b", "s", "d"]
    r = earliest_legal_pos(comp, order, "s")
    self.assertEqual(r.floor, 2)
    self.assertIsNone(r.blocker)
    self.assertEqual(r.control_pins, ((1, "a"),))
    new = hoist_start_to_floor(comp, order, "s")
    self.assertEqual(new, ["p0", "a", "s", "b", "d"])
    self.assertTrue(is_legal(comp, new))

  def test_control_pred_inside_moving_set_is_ignored(self):
    comp = Computation.from_instructions(
        "c",
        [
            Inst("p0", "parameter"),
            Inst("w", "convert", ("p0",)),
            _fus("a", ["p0"], 100),
            _start("s", ["w"], control_predecessors=("w",)),
            _done("d", "s"),
        ],
    )
    r = earliest_legal_pos(comp, ["p0", "w", "a", "s", "d"], "s")
    self.assertEqual(r.floor, 0)

  def test_hoisted_order_is_legal(self):
    comp, order = _basic()
    self.assertTrue(is_legal(comp, hoist_start_to_floor(comp, order, "s"), baseline=order))


class CostTest(absltest.TestCase):

  def test_exposure_before_and_after_hoist(self):
    comp, order = _basic()
    e = collective_exposures(comp, order)[0]
    self.assertEqual((e.overlap_us, e.deficit_us), (0.0, 500.0))
    self.assertEqual(total_exposed_us(comp, order), 500.0)
    hoisted = hoist_start_to_floor(comp, order, "s")
    e = collective_exposures(comp, hoisted)[0]
    self.assertEqual(e.overlap_us, 600.0)
    self.assertEqual(total_exposed_us(comp, hoisted), 0.0)

  def test_partial_overlap_and_unprofiled_pairs_ignored(self):
    comp = Computation.from_instructions(
        "c",
        [
            _start("s", latency=500),
            _fus("f", [], 120),
            _done("d", "s"),
            _start("s2", latency=None),
            _done("d2", "s2"),
        ],
    )
    order = ["s", "f", "d", "s2", "d2"]
    self.assertEqual(total_exposed_us(comp, order), 380.0)
    self.assertLen(collective_exposures(comp, order), 1)

  def test_coverage_gaps(self):
    comp = Computation.from_instructions(
        "c",
        [
            _start("s", latency=500),
            _fus("unprofiled", []),
            _done("d", "s"),
            _start("s2", latency=None),
            _done("d2", "s2"),
        ],
    )
    gaps = coverage_gaps(comp, ["s", "unprofiled", "d", "s2", "d2"])
    self.assertIn("s2: no profiled latency", gaps)
    self.assertIn("unprofiled: zero cost inside an async window", gaps)


class SimulateTest(absltest.TestCase):

  def test_hidden_vs_exposed(self):
    comp, order = _basic()
    self.assertEqual(simulate_comm_stream(comp, order).total_stall_us, 500.0)
    hoisted = hoist_start_to_floor(comp, order, "s")
    self.assertEqual(simulate_comm_stream(comp, hoisted).total_stall_us, 0.0)

  def test_collectives_serialize_on_one_stream(self):
    comp = Computation.from_instructions(
        "c",
        [
            _start("s1", latency=400),
            _start("s2", latency=400),
            _fus("f", [], 500),
            _done("d1", "s1"),
            _done("d2", "s2"),
        ],
    )
    res = simulate_comm_stream(comp, ["s1", "s2", "f", "d1", "d2"])
    # s1 ends at 400, s2 queues behind it and ends at 800; compute ends at 500.
    self.assertEqual(res.stalls, {"d2": 300.0})
    self.assertEqual(res.total_stall_us, 300.0)

  def test_sm_gate_delays_queued_collective(self):
    comp = Computation.from_instructions(
        "c",
        [
            _start("s1", latency=1000),
            _start("s2", latency=100),
            _fus("gemm", [], 2000, sm_gate=True),
            _done("d1", "s1"),
            _done("d2", "s2"),
        ],
    )
    order = ["s1", "s2", "gemm", "d1", "d2"]
    self.assertEqual(simulate_comm_stream(comp, order, sm_gate=False).total_stall_us, 0.0)
    gated = simulate_comm_stream(comp, order)
    # s2 would begin at 1000 inside the GEMM [0, 2000), so it waits to 2000.
    self.assertEqual(gated.stalls, {"d2": 100.0})

  def test_small_gemm_does_not_gate(self):
    comp = Computation.from_instructions(
        "c",
        [
            _start("s1", latency=1000),
            _start("s2", latency=100),
            _fus("gemm", [], 2000, sm_gate=True),
            _done("d1", "s1"),
            _done("d2", "s2"),
        ],
    )
    order = ["s1", "s2", "gemm", "d1", "d2"]
    res = simulate_comm_stream(comp, order, sm_gate_min_us=5000)
    self.assertEqual(res.total_stall_us, 0.0)

  def test_unprofiled_start_costs_nothing(self):
    comp = Computation.from_instructions(
        "c",
        [
            _start("s", latency=None),
            _fus("f", [], 50),
            _done("d", "s"),
        ],
    )
    self.assertEqual(simulate_comm_stream(comp, ["s", "f", "d"]).total_stall_us, 0.0)


class CrossComputationTest(absltest.TestCase):
  """Instructions never move between computations."""

  def _module(self):
    a = Computation.from_instructions("a", [Inst("a0", "parameter"), _fus("a1", ["a0"], 10)])
    b = Computation.from_instructions("b", [Inst("b0", "parameter"), _fus("b1", ["b0"], 10)])
    return Module.from_computations("m", [a, b])

  def test_same_computation_reorder_is_legal(self):
    self.assertEqual(validate_plan(self._module(), {"a": ["a0", "a1"], "b": ["b0", "b1"]}), [])

  def test_instruction_placed_in_other_computation(self):
    v = validate_plan(self._module(), {"a": ["a0", "a1", "b1"]})
    self.assertIn("CROSS_COMPUTATION", _codes(v))
    # Reported once as a cross move, not additionally as an unknown name.
    self.assertNotIn("UNKNOWN", _codes(v))

  def test_move_out_is_missing_in_source_and_cross_in_target(self):
    v = validate_plan(self._module(), {"a": ["a0"], "b": ["b0", "b1", "a1"]})
    self.assertEqual(_codes(v), ["CROSS_COMPUTATION", "MISSING"])

  def test_unknown_computation(self):
    v = validate_plan(self._module(), {"zzz": ["a0"]})
    self.assertEqual(_codes(v), ["PLAN_UNKNOWN_COMPUTATION"])

  def test_duplicate_names_across_computations_rejected(self):
    a = Computation.from_instructions("a", [Inst("x", "parameter")])
    b = Computation.from_instructions("b", [Inst("x", "parameter")])
    with self.assertRaises(ValueError):
      Module.from_computations("m", [a, b])

  def test_computation_absent_from_plan_is_not_checked(self):
    self.assertEqual(validate_plan(self._module(), {"a": ["a0", "a1"]}), [])


def _wire_cost(name, cost):
  body = b"\x0a" + bytes([len(name)]) + name.encode() + b"\x11" + struct.pack("<d", cost)
  return b"\x0a" + bytes([len(body)]) + body


class ProfileTest(absltest.TestCase):

  def test_binary(self):
    p = Profile.from_fdo_bytes(_wire_cost("all-gather.1", 12.5) + _wire_cost("f", 3.0))
    self.assertEqual(p.cost("all-gather.1"), 12.5)
    self.assertEqual(p.cost("f"), 3.0)
    self.assertEqual(p.cost("missing"), 0.0)
    self.assertFalse(p.has("missing"))

  def test_text_matches_binary(self):
    text = 'costs {\n  name: "all-gather.1"\n  cost_us: 12.5\n}\ncosts { name: "f" cost_us: 3 }\n'
    self.assertEqual(
        Profile.from_text(text).costs,
        Profile.from_fdo_bytes(_wire_cost("all-gather.1", 12.5) + _wire_cost("f", 3.0)).costs,
    )

  def test_latencies_text(self):
    p = Profile.from_text('latencies {\n source: "a"\n target: "b"\n latency_us: 7\n}\n')
    self.assertEqual(p.latencies[("a", "b")], 7.0)

  def test_module_config_field_31(self):
    fdo = _wire_cost("f", 4.0)
    cfg = b"\x08\x01" + b"\xfa\x01" + bytes([len(fdo)]) + fdo  # field 1 varint, field 31 bytes
    self.assertEqual(Profile.from_module_config(cfg).cost("f"), 4.0)
    self.assertEqual(Profile.from_module_config(b"").costs, {})

  def test_truncated_binary_raises(self):
    with self.assertRaises(ValueError):
      Profile.from_fdo_bytes(_wire_cost("f", 4.0)[:-3])


class HloTextTest(absltest.TestCase):

  def test_opcode_str(self):
    self.assertEqual(hlo_text.opcode_str("kGetTupleElement"), "get-tuple-element")
    self.assertEqual(hlo_text.opcode_str("kAllGatherStart"), "all-gather-start")

  def test_control_predecessors(self):
    t = "%x = f32[] add(%a, %b), control-predecessors={%p, %q.1}, metadata={}"
    self.assertEqual(hlo_text.control_predecessor_names(t), ("p", "q.1"))
    self.assertEqual(hlo_text.control_predecessor_names("%x = f32[] add(%a, %b)"), ())

  def test_explicit_groups(self):
    g = hlo_text.parse_device_groups("all-gather-start(%x), replica_groups={{0,1},{2,3}}, dimensions={0}")
    self.assertEqual(g, (frozenset({0, 1}), frozenset({2, 3})))

  def test_iota_groups(self):
    g = hlo_text.parse_device_groups("replica_groups=[2,4]<=[8]")
    self.assertEqual(g, (frozenset({0, 1, 2, 3}), frozenset({4, 5, 6, 7})))
    g = hlo_text.parse_device_groups("replica_groups=[4,2]<=[2,4]T(1,0)")
    self.assertEqual(g, (frozenset({0, 4}), frozenset({1, 5}), frozenset({2, 6}), frozenset({3, 7})))

  def test_mesh_groups(self):
    g = hlo_text.parse_device_groups("replica_groups=mesh['axis_0'=8] {'axis_0'}")
    self.assertEqual(g, (frozenset(range(8)),))
    g = hlo_text.parse_device_groups("replica_groups=mesh['dp'=2,'fsdp'=4] {'fsdp'}")
    self.assertEqual(g, (frozenset({0, 1, 2, 3}), frozenset({4, 5, 6, 7})))
    g = hlo_text.parse_device_groups("replica_groups=mesh['dp'=2,'fsdp'=4] {'dp'}")
    self.assertEqual(g, (frozenset({0, 4}), frozenset({1, 5}), frozenset({2, 6}), frozenset({3, 7})))

  def test_unknown_forms_are_none(self):
    self.assertIsNone(hlo_text.parse_device_groups("replica_groups={}"))
    self.assertIsNone(hlo_text.parse_device_groups("%x = f32[] add(%a, %b)"))
    self.assertIsNone(hlo_text.parse_device_groups("replica_groups=mesh['a'=2] {'nope'}"))

  def test_collective_permute_pairs(self):
    g = hlo_text.parse_device_groups("source_target_pairs={{0,1},{1,2}}")
    self.assertEqual(g, (frozenset({0, 1}), frozenset({1, 2})))


_TINY_HLO = """
HloModule tiny, is_scheduled=true

%async_comp (p: bf16[16,64]) -> bf16[128,64] {
  %p = bf16[16,64]{1,0} parameter(0)
  ROOT %ag = bf16[128,64]{1,0} all-gather(%p), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, use_global_device_ids=true
}

ENTRY %main (x: bf16[16,64], w: bf16[64,64]) -> (bf16[128,64], bf16[16,64]) {
  %x = bf16[16,64]{1,0} parameter(0)
  %w = bf16[64,64]{1,0} parameter(1)
  %ag-start = ((bf16[16,64]{1,0}), bf16[128,64]{1,0}) async-start(%x), calls=%async_comp
  %mm = bf16[16,64]{1,0} dot(%x, %w), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %mm2 = bf16[16,64]{1,0} dot(%mm, %w), lhs_contracting_dims={1}, rhs_contracting_dims={0}, control-predecessors={%mm}
  %ag-done = bf16[128,64]{1,0} async-done(%ag-start), calls=%async_comp
  ROOT %out = (bf16[128,64], bf16[16,64]) tuple(%ag-done, %mm2)
}
"""


@absltest.skipIf(_hlo is None, "jaxlib not available")
class HloAdapterTest(absltest.TestCase):

  def _load(self, **profile_costs):
    hm = _hlo.hlo_module_from_text(_TINY_HLO)
    return hm, load_module(hm, Profile.from_costs(profile_costs))

  def test_loads_structure(self):
    _, loaded = self._load()
    main = loaded.module.computations["main"]
    self.assertEqual(loaded.baseline["main"], ["x", "w", "ag-start", "mm", "mm2", "ag-done", "out"])
    self.assertEqual(main["mm2"].control_predecessors, ("mm",))
    self.assertEqual(main["ag-start"].async_role, AsyncRole.START)
    self.assertEqual(main["ag-done"].async_role, AsyncRole.DONE)
    self.assertEqual(main["ag-done"].operands, ("ag-start",))
    self.assertEqual(main["ag-start"].device_groups, (frozenset(range(8)),))
    self.assertTrue(main["ag-start"].uses_comm_stream)
    # The wrapped computation is its own scheduled computation, so names stay owned.
    self.assertIn("async_comp", loaded.module.computations)
    self.assertEqual(loaded.module.owner["ag"], "async_comp")

  def test_costs_and_latency_from_profile(self):
    _, loaded = self._load(mm=100.0, mm2=250.0, **{"ag": 900.0})
    main = loaded.module.computations["main"]
    self.assertEqual(main["mm"].cost_us, 100.0)
    self.assertEqual(main["mm2"].cost_us, 250.0)
    self.assertEqual(main["ag-start"].latency_us, 900.0)
    self.assertEqual(main["ag-start"].cost_us, 0.0)

  def test_unprofiled_latency_is_none(self):
    _, loaded = self._load()
    self.assertIsNone(loaded.module.computations["main"]["ag-start"].latency_us)

  def test_apply_legal_plan(self):
    hm, loaded = self._load()
    plan = {"main": ["x", "w", "mm", "ag-start", "mm2", "ag-done", "out"]}
    res = apply_plan(loaded, plan)
    self.assertTrue(res.applied, res)
    sched = hm.schedule()
    comp = next(c for c in hm.make_nonfusion_computations() if c.name == "main")
    self.assertEqual([i.name for i in sched.sequence(comp)], plan["main"])

  def test_apply_rejects_control_order_violation_and_leaves_module_untouched(self):
    hm, loaded = self._load()
    plan = {"main": ["x", "w", "ag-start", "mm2", "mm", "ag-done", "out"]}
    res = apply_plan(loaded, plan)
    self.assertFalse(res.applied)
    self.assertIn("CONTROL_ORDER", {v.code for v in res.violations})
    comp = next(c for c in hm.make_nonfusion_computations() if c.name == "main")
    self.assertEqual([i.name for i in hm.schedule().sequence(comp)], loaded.baseline["main"])

  def test_apply_rejects_cross_computation_move(self):
    hm, loaded = self._load()
    plan = {"main": ["x", "w", "ag-start", "mm", "mm2", "ag", "ag-done", "out"]}
    res = apply_plan(loaded, plan)
    self.assertFalse(res.applied)
    self.assertIn("CROSS_COMPUTATION", {v.code for v in res.violations})
    comp = next(c for c in hm.make_nonfusion_computations() if c.name == "main")
    self.assertEqual([i.name for i in hm.schedule().sequence(comp)], loaded.baseline["main"])

  def test_apply_stamped_plan_and_identity_mismatch(self):
    hm, loaded = self._load(mm=100.0)
    good = {"main": ["x", "w", "mm", "ag-start", "mm2", "ag-done", "out"]}
    stamped = Plan.for_module(loaded.module, loaded.baseline, good, profile=loaded.profile)
    strict = IdentityOptions(require_identity=True)
    stale = Plan(computations=good, module="tiny", input_checksum="0" * 64)
    res = apply_plan(loaded, stale, identity=strict)
    self.assertFalse(res.applied)
    self.assertIn("PLAN_INPUT_MISMATCH", {v.code for v in res.violations})
    comp = next(c for c in hm.make_nonfusion_computations() if c.name == "main")
    self.assertEqual([i.name for i in hm.schedule().sequence(comp)], loaded.baseline["main"])
    self.assertFalse(apply_plan(loaded, good, identity=strict).applied)  # bare plan refused when identity required
    self.assertTrue(apply_plan(loaded, stamped, identity=strict).applied)

  def test_unchanged_plan_applies_nothing(self):
    _, loaded = self._load()
    res = apply_plan(loaded, {"main": list(loaded.baseline["main"])})
    self.assertTrue(res.applied)
    self.assertIn("nothing to apply", res.reason)

  def test_unscheduled_module_is_an_error(self):
    hm = _hlo.hlo_module_from_text(_TINY_HLO.replace(", is_scheduled=true", ""))
    with self.assertRaises(ValueError):
      load_module(hm)


class AnalysisTest(absltest.TestCase):

  def _module(self):
    comp, order = _basic()
    return Module.from_computations("m", [comp]), {"main": order}

  def test_row_fields_and_ranking(self):
    module, orders = self._module()
    report = analyze_module(module, orders)
    (cr,) = report.computations
    (row,) = cr.rows
    self.assertEqual((row.start, row.done, row.deficit_us, row.overlap_us), ("s", "d", 500.0, 0.0))
    self.assertTrue(row.profiled)
    self.assertEqual(row.hoist_to, 2)  # p0, w move ahead of the long compute: p0 w s a b c d use
    self.assertTrue(row.can_hoist)
    self.assertEqual(report.ranked()[0][1].start, "s")
    self.assertEqual(report.static_exposed_us, 500.0)

  def test_already_hoisted_start_is_not_hoistable(self):
    comp, _ = _basic()
    order = ["p0", "w", "s", "a", "b", "c", "d", "use"]
    (row,) = analyze_module(Module.from_computations("m", [comp]), {"main": order}).computations[0].rows
    self.assertFalse(row.can_hoist)
    self.assertIsNone(row.hoist_to)
    self.assertEqual(row.deficit_us, 0.0)

  def test_unprofiled_rows_rank_last_and_flag_unknown(self):
    comp = Computation.from_instructions(
        "c",
        [
            _start("s1", latency=800.0),
            _done("d1", "s1"),
            Inst("s2", "all-gather-start", async_role=AsyncRole.START, uses_comm_stream=True),
            _done("d2", "s2"),
        ],
    )
    report = analyze_module(Module.from_computations("m", [comp]), {"c": ["s1", "s2", "d1", "d2"]})
    ranked = report.ranked()
    self.assertEqual([r.start for _, r in ranked], ["s1", "s2"])
    self.assertFalse(ranked[1][1].profiled)
    self.assertIsNone(ranked[1][1].latency_us)

  def test_computation_filter_and_while_flag(self):
    module, orders = self._module()
    self.assertEqual(analyze_module(module, orders, only={"nope"}).computations, ())
    report = analyze_module(module, orders, while_bodies=frozenset({"main"}))
    self.assertTrue(report.computations[0].while_body)

  def test_check_plan_improvement_and_rejection(self):
    module, orders = self._module()
    good = ["p0", "w", "s", "a", "b", "c", "d", "use"]
    rep = check_plan(module, orders, {"main": good})
    self.assertTrue(rep.accepted)
    self.assertEqual((rep.deltas[0].static_before_us, rep.deltas[0].static_after_us), (500.0, 0.0))
    bad = check_plan(module, orders, {"main": ["p0", "s", "w", "a", "b", "c", "d", "use"]})
    self.assertFalse(bad.accepted)
    self.assertIsNone(bad.after)

  def test_check_plan_unchanged_computation_has_no_delta(self):
    module, orders = self._module()
    rep = check_plan(module, orders, {"main": list(orders["main"])})
    self.assertTrue(rep.accepted)
    self.assertEqual(rep.deltas, ())


class PlanTest(absltest.TestCase):

  def _setup(self):
    comp, order = _basic()
    module = Module.from_computations("m", [comp])
    return module, {"main": order}

  def test_roundtrip_and_deterministic_bytes(self):
    module, base = self._setup()
    good = {"main": ["p0", "w", "s", "a", "b", "c", "d", "use"]}
    prof = Profile.from_costs({"a": 1.0, "b": 2.0})
    p1 = Plan.for_module(
        module, base, good, profile=prof, moves=(Move("main", "s", to_before="a", expected_delta_us=-500.0),)
    )
    p2 = Plan.loads(p1.dumps())
    self.assertEqual(p2, p1)
    self.assertEqual(p2.dumps(), p1.dumps())
    self.assertEqual(p2.digest(), p1.digest())
    # Key order in the source document does not change the canonical bytes.
    shuffled = json.dumps(dict(reversed(list(p1.to_dict().items()))))
    self.assertEqual(Plan.loads(shuffled).dumps(), p1.dumps())

  def test_input_checksum_tracks_names_and_order(self):
    _, base = self._setup()
    c0 = input_checksum(base)
    self.assertEqual(c0, input_checksum({"main": list(base["main"])}))
    swapped = {"main": ["p0", "b", "a", *base["main"][3:]]}
    self.assertNotEqual(c0, input_checksum(swapped))
    self.assertNotEqual(c0, input_checksum({"main": base["main"][:-1]}))
    self.assertNotEqual(c0, input_checksum({"other": base["main"]}))

  def test_profile_identity_is_exact(self):
    a = profile_identity(Profile.from_costs({"x": 1.0, "y": 2.0}))
    self.assertEqual(a, profile_identity(Profile.from_costs({"y": 2.0, "x": 1.0})))
    self.assertNotEqual(a, profile_identity(Profile.from_costs({"x": 1.0, "y": 2.0000001})))
    self.assertEqual(a["entries"], 2)

  def test_shape_errors(self):
    for doc in (
        [],
        {},
        {"computations": {}},
        {"computations": {"c": "nope"}},
        {"computations": {"c": [1]}},
        {"version": 2, "computations": {"c": ["a"]}},
        {"computations": {"c": ["a"]}, "profile": {"sha256": 1}},
        {"computations": {"c": ["a"]}, "moves": [{"computation": "c"}]},
    ):
      with self.assertRaises(PlanFormatError, msg=str(doc)):
        Plan.from_dict(doc)
    with self.assertRaises(PlanFormatError):
      Plan.loads("not json")

  def test_bare_plan_has_no_identity(self):
    plan = Plan.from_dict({"computations": {"main": ["a"]}})
    self.assertFalse(plan.has_identity)

  def test_verify_matching_plan_is_clean(self):
    module, base = self._setup()
    plan = Plan.for_module(module, base, {"main": base["main"]}, profile=Profile.from_costs({"a": 1.0}))
    self.assertEqual(verify_against(plan, module, base, Profile.from_costs({"a": 1.0})), [])

  def test_verify_rejects_other_module_and_stale_input(self):
    module, base = self._setup()
    plan = Plan.for_module(module, base, {"main": base["main"]})
    other = Module.from_computations("other", list(module.computations.values()))
    self.assertEqual(_codes(verify_against(plan, other, base)), ["PLAN_MODULE_MISMATCH"])
    changed = {"main": ["p0", "b", "a", *base["main"][3:]]}
    self.assertEqual(_codes(verify_against(plan, module, changed)), ["PLAN_INPUT_MISMATCH"])

  def test_verify_require_identity(self):
    module, base = self._setup()
    bare = Plan(computations={"main": base["main"]})
    self.assertEqual(verify_against(bare, module, base), [])
    strict = verify_against(bare, module, base, options=IdentityOptions(require_identity=True))
    self.assertIn("PLAN_MISSING_IDENTITY", _codes(strict))

  def test_profile_mismatch_is_warning_unless_strict(self):
    module, base = self._setup()
    plan = Plan.for_module(module, base, {"main": base["main"]}, profile=Profile.from_costs({"a": 1.0}))
    other = Profile.from_costs({"a": 2.0})
    (v,) = verify_against(plan, module, base, other)
    self.assertEqual((v.code, v.severity), ("PLAN_PROFILE_MISMATCH", _Sev.WARNING))
    (v,) = verify_against(plan, module, base, other, IdentityOptions(strict_profile=True))
    self.assertEqual(v.severity, _Sev.ERROR)
    (v,) = verify_against(plan, module, base, None)
    self.assertEqual(v.code, "PLAN_PROFILE_UNCHECKED")

  def test_bad_move_annotation(self):
    module, base = self._setup()
    plan = Plan.for_module(module, base, {"main": base["main"]}, moves=(Move("main", "ghost"),))
    self.assertEqual(_codes(verify_against(plan, module, base)), ["PLAN_BAD_MOVE"])

  def test_check_plan_applies_identity_before_scoring(self):
    module, base = self._setup()
    good = {"main": ["p0", "w", "s", "a", "b", "c", "d", "use"]}
    stamped = Plan.for_module(module, base, good)
    self.assertTrue(check_plan(module, base, stamped).accepted)
    stale = Plan(computations=good, module="m", input_checksum="0" * 64)
    rep = check_plan(module, base, stale)
    self.assertFalse(rep.accepted)
    self.assertIsNone(rep.after)
    self.assertFalse(check_plan(module, base, good, identity=IdentityOptions(require_identity=True)).accepted)


def _write(path, data):
  """Write str (UTF-8) or bytes to `path`."""
  mode, kwargs = ("wb", {}) if isinstance(data, bytes) else ("w", {"encoding": "utf-8"})
  with open(path, mode, **kwargs) as f:
    f.write(data)


def _as_hlo_proto(module_bytes: bytes) -> bytes:
  """Wrap a serialized HloModuleProto as field 1 of an HloProto, like --xla_dump_hlo_as_proto."""
  n, varint = len(module_bytes), b""
  while True:
    b = n & 0x7F
    n >>= 7
    varint += bytes([b | (0x80 if n else 0)])
    if not n:
      break
  return b"\x0a" + varint + module_bytes


@absltest.skipIf(_hlo is None, "jaxlib not available")
class CliTest(absltest.TestCase):

  def setUp(self):
    """Write the tiny module (three encodings), a profile and a plan path to a temp dir."""
    super().setUp()
    d = self.create_tempdir().full_path
    module_bytes = _hlo.hlo_module_from_text(_TINY_HLO).as_serialized_hlo_module_proto()
    self.proto = os.path.join(d, "tiny.pb")
    self.wrapped = os.path.join(d, "tiny.hlo.pb")
    self.text = os.path.join(d, "tiny.txt")
    self.profile = os.path.join(d, "tiny.fdo_profile")
    self.plan = os.path.join(d, "plan.json")
    _write(self.proto, module_bytes)
    _write(self.wrapped, _as_hlo_proto(module_bytes))
    _write(self.text, _TINY_HLO)
    _write(
        self.profile,
        'costs { name: "mm" cost_us: 100 }\ncosts { name: "mm2" cost_us: 250 }\ncosts { name: "ag" cost_us: 900 }\n',
    )

  def _run(self, *argv):
    out = io.StringIO()
    status = cli.main(list(argv), out=out)
    return status, out.getvalue()

  def test_analyze_text_proto_and_hlo_proto_agree(self):
    results = []
    for path in (self.text, self.proto, self.wrapped):
      status, text = self._run("analyze", path, "--profile", self.profile, "--json")
      self.assertEqual(status, 0)
      doc = json.loads(text)
      row = doc["ranked"][0]
      results.append((row["start"], row["key"], row["latency_us"], row["overlap_us"], row["deficit_us"]))
    self.assertEqual(results[0], ("ag-start", "ag", 900.0, 350.0, 550.0))
    self.assertEqual(results[0], results[1])
    self.assertEqual(results[0], results[2])

  def test_analyze_table_mentions_deficit(self):
    status, text = self._run("analyze", self.proto, "--profile", self.profile)
    self.assertEqual(status, 0)
    self.assertIn("ag-start", text)
    self.assertIn("550.0", text)

  def test_analyze_without_profile_warns(self):
    status, text = self._run("analyze", self.proto)
    self.assertEqual(status, 0)
    self.assertIn("no profile", text)
    self.assertIn("[no profile entry]", text)

  def test_check_plan_accepts_and_scores(self):
    _write(self.plan, json.dumps({"computations": {"main": ["x", "w", "mm", "ag-start", "mm2", "ag-done", "out"]}}))
    status, text = self._run("check-plan", self.proto, "--profile", self.profile, "--plan", self.plan, "--json")
    self.assertEqual(status, 0)
    doc = json.loads(text)
    self.assertTrue(doc["accepted"])
    delta = doc["deltas"][0]
    self.assertEqual((delta["static_before_us"], delta["static_after_us"]), (550.0, 650.0))

  def test_check_plan_rejects_with_exit_1(self):
    _write(self.plan, json.dumps({"computations": {"main": ["x", "w", "ag-start", "mm2", "mm", "ag-done", "out"]}}))
    status, text = self._run("check-plan", self.proto, "--profile", self.profile, "--plan", self.plan)
    self.assertEqual(status, 1)
    self.assertIn("REJECTED", text)
    self.assertIn("CONTROL_ORDER", text)

  def test_check_plan_rejects_cross_computation_move(self):
    _write(self.plan, json.dumps({"computations": {"main": ["x", "w", "ag-start", "mm", "mm2", "ag", "ag-done", "out"]}}))
    status, text = self._run("check-plan", self.proto, "--plan", self.plan)
    self.assertEqual(status, 1)
    self.assertIn("CROSS_COMPUTATION", text)

  def test_stamp_then_check_with_identity(self):
    _write(self.plan, json.dumps({"computations": {"main": ["x", "w", "mm", "ag-start", "mm2", "ag-done", "out"]}}))
    stamped = os.path.join(self.create_tempdir().full_path, "stamped.json")
    status, text = self._run("stamp-plan", self.proto, "--profile", self.profile, "--plan", self.plan, "--out", stamped)
    self.assertEqual(status, 0, text)
    with open(stamped, encoding="utf-8") as f:
      doc = json.load(f)
    self.assertEqual(doc["module"], "tiny")
    self.assertEqual(len(doc["input_checksum"]), 64)
    self.assertEqual(doc["profile"]["entries"], 3)
    status, text = self._run(
        "check-plan", self.proto, "--profile", self.profile, "--plan", stamped, "--require-identity", "--strict-profile"
    )
    self.assertEqual(status, 0, text)
    self.assertIn("plan digest", text)
    # Identity survives all three module encodings (the checksum ignores proto bytes).
    for path in (self.text, self.wrapped):
      self.assertEqual(
          self._run("check-plan", path, "--profile", self.profile, "--plan", stamped, "--require-identity")[0], 0
      )

  def test_check_plan_rejects_unstamped_when_required_and_stale_checksum(self):
    _write(self.plan, json.dumps({"computations": {"main": ["x", "w", "mm", "ag-start", "mm2", "ag-done", "out"]}}))
    status, text = self._run("check-plan", self.proto, "--plan", self.plan, "--require-identity")
    self.assertEqual(status, 1)
    self.assertIn("PLAN_MISSING_IDENTITY", text)
    _write(
        self.plan,
        json.dumps(
            {
                "module": "tiny",
                "input_checksum": "0" * 64,
                "computations": {"main": ["x", "w", "mm", "ag-start", "mm2", "ag-done", "out"]},
            }
        ),
    )
    status, text = self._run("check-plan", self.proto, "--plan", self.plan)
    self.assertEqual(status, 1)
    self.assertIn("PLAN_INPUT_MISMATCH", text)

  def test_stamp_refuses_illegal_plan(self):
    _write(self.plan, json.dumps({"computations": {"main": ["x", "w", "ag-start", "mm2", "mm", "ag-done", "out"]}}))
    out = os.path.join(self.create_tempdir().full_path, "o.json")
    status, _ = self._run("stamp-plan", self.proto, "--plan", self.plan, "--out", out)
    self.assertEqual(status, 1)
    self.assertFalse(os.path.exists(out))

  def test_analyze_json_exposes_identity_for_stamping(self):
    doc = json.loads(self._run("analyze", self.proto, "--profile", self.profile, "--json")[1])
    self.assertEqual(len(doc["input_checksum"]), 64)
    self.assertEqual(doc["profile"]["entries"], 3)

  def test_bad_inputs_exit_2(self):
    self.assertEqual(self._run("analyze", "/nonexistent.pb")[0], 2)
    bad_plan = os.path.join(self.create_tempdir().full_path, "p.json")
    _write(bad_plan, "[1, 2]")
    self.assertEqual(self._run("check-plan", self.proto, "--plan", bad_plan)[0], 2)
    self.assertEqual(self._run("analyze", self.proto, "--computation", "nope")[0], 2)


if __name__ == "__main__":
  absltest.main()
