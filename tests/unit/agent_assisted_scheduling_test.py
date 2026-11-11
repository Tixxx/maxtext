"""Tests for the jax-free agent-assisted scheduling building blocks."""

import io
import json
import os
import struct
from unittest import mock

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
from maxtext.utils.agent_assisted_scheduling import aot_bundle, aot_executable, hlo_text, module_exchange
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
from maxtext.utils.agent_assisted_scheduling.priority import (
    DEFAULT_PRIORITY_PREFIXES,
    is_priority,
    priority_regressions,
    reachable_overlap_us,
    real_consumers,
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

  def test_for_module_strips_fingerprint_and_drops_other_modules(self):
    p = Profile({"aa::ag.1": 5.0, "bb::ag.1": 9.0, "ag.2": 2.0}, {("aa::s", "aa::d"): 4.0, ("bb::s", "bb::d"): 8.0})
    q = p.for_module("aa")
    self.assertEqual(q.costs, {"ag.1": 5.0, "ag.2": 2.0})
    self.assertEqual(q.latencies, {("s", "d"): 4.0})

  def test_for_module_unknown_fingerprint(self):
    self.assertEqual(Profile({"aa::ag.1": 5.0}).for_module(None).costs, {"ag.1": 5.0})
    with self.assertRaises(ValueError):
      Profile({"aa::x": 1.0, "bb::x": 2.0}).for_module(None)
    plain = Profile({"x": 1.0})
    self.assertIs(plain.for_module("aa"), plain)

  def test_module_fingerprint(self):
    from maxtext.utils.agent_assisted_scheduling.profile import module_fingerprint

    text = 'HloModule m, frontend_attributes={fingerprint_before_lhs="abc123"}\n'
    self.assertEqual(module_fingerprint(text), "abc123")
    self.assertIsNone(module_fingerprint("HloModule m\n"))
    long_header = "HloModule m, entry_computation_layout={" + "f32[1]," * 20000 + '}, frontend_attributes={fingerprint_before_lhs="zz"}\n'
    self.assertEqual(module_fingerprint(long_header), "zz")

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

  def test_start_records_wrapped_opcode_so_priority_does_not_depend_on_names(self):
    _, loaded = self._load()
    start = loaded.module.computations["main"]["ag-start"]
    self.assertEqual((start.profile_key, start.wrapped_opcode), ("ag", "all-gather"))
    self.assertTrue(is_priority(start))

  def test_sm_gate_ops_modes(self):
    hm = _hlo.hlo_module_from_text(_TINY_HLO)
    gate = lambda mode: {n: i.sm_gate for n, i in load_module(hm, Profile(), sm_gate_ops=mode).module.computations["main"].instructions.items()}
    default = gate("gemm")
    self.assertTrue(default["mm"] and default["mm2"])  # dot is SM-filling in any model, TE or not
    self.assertFalse(default["ag-start"] or default["x"])
    self.assertFalse(any(gate("none").values()))
    every = gate("all")
    self.assertTrue(every["mm"])
    self.assertFalse(every["x"] or every["out"])  # parameters / tuple are trivially movable
    with self.assertRaises(ValueError):
      load_module(hm, Profile(), sm_gate_ops="bogus")
    self.assertTrue(hlo_text.GEMM_LIKE_TARGET_RE.search('custom_call_target="__cublas$gemm"'))
    self.assertTrue(hlo_text.GEMM_LIKE_TARGET_RE.search('custom_call_target="te_grouped_gemm_ffi"'))
    self.assertTrue(hlo_text.GEMM_LIKE_TARGET_RE.search('custom_call_target="__cudnn$fmhaForward"'))
    self.assertFalse(hlo_text.GEMM_LIKE_TARGET_RE.search('custom_call_target="te_dbias_quantize_ffi"'))

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


class PriorityTest(absltest.TestCase):
  """Per-collective regression rules (priority.py)."""

  def _comp(self, key="all-gather.1"):
    """s(500 us) -> d -> g(get-tuple-element) -> c; a and b are 300 us of compute."""
    return Computation.from_instructions(
        "c",
        [
            _start("s", latency=500.0, profile_key=key),
            _done("d", "s"),
            Inst("g", "get-tuple-element", ("d",)),
            _fus("c", ["g"], 10),
            _fus("a", [], 300),
            _fus("b", [], 300),
        ],
    )

  def test_prefix_match_uses_profile_key_then_opcode(self):
    comp = self._comp()
    self.assertTrue(is_priority(comp["s"]))
    self.assertFalse(is_priority(self._comp("te_ep_combine.2")["s"]))
    self.assertTrue(is_priority(_start("x"), DEFAULT_PRIORITY_PREFIXES))  # no key: opcode all-gather-start
    self.assertFalse(is_priority(comp["a"]))  # not a start
    self.assertEqual(DEFAULT_PRIORITY_PREFIXES, ("all-gather", "reduce-scatter", "all-reduce"))

  def test_real_consumers_look_through_trivial_users(self):
    comp = self._comp()
    self.assertEqual(real_consumers(comp, "d"), ["c"])
    self.assertEqual(reachable_overlap_us(comp, ["s", "a", "b", "d", "g", "c"], "s", "d"), 500.0)  # capped at latency

  def test_moving_a_consumer_earlier_is_a_reachable_regression_even_if_overlap_is_unchanged(self):
    comp = self._comp()
    before = ["s", "a", "d", "b", "g", "c"]  # overlap 300; b could still sink into the window
    after = ["s", "a", "d", "g", "c", "b"]  # overlap 300; the consumer now caps the done
    regs = priority_regressions(comp, before, after)
    self.assertEqual([(r.kind, r.before_us, r.after_us) for r in regs], [("reachable", 0.0, 200.0)])

  def test_exposed_regression_and_improvement(self):
    comp = self._comp()
    wide = ["s", "a", "b", "d", "g", "c"]
    narrow = ["s", "a", "d", "b", "g", "c"]
    self.assertEqual({r.kind for r in priority_regressions(comp, wide, narrow)}, {"exposed", "stall"})
    self.assertEqual(priority_regressions(comp, narrow, wide), [])

  def test_non_priority_collective_may_regress(self):
    comp = self._comp("te_ep_combine.2")
    self.assertEqual(priority_regressions(comp, ["s", "a", "b", "d", "g", "c"], ["s", "a", "d", "b", "g", "c"]), [])

  def test_stall_is_summed_over_priority_collectives(self):
    """One done may stall more if another stalls less by at least as much."""
    comp = Computation.from_instructions(
        "c",
        [
            _start("s1", latency=400.0, profile_key="all-gather.1"),
            _done("d1", "s1"),
            _start("s2", latency=400.0, profile_key="all-gather.2"),
            _done("d2", "s2"),
            _fus("a", [], 100),
        ],
    )
    before = ["s1", "s2", "a", "d1", "d2"]
    after = ["s1", "a", "s2", "d2", "d1"]
    kinds = {r.kind for r in priority_regressions(comp, before, after)}
    self.assertNotIn("stall", kinds)

  def test_check_plan_rejects_and_allows(self):
    comp = self._comp()
    module = Module.from_computations("m", [comp])
    baseline = {"c": ["s", "a", "b", "d", "g", "c"]}
    plan = {"c": ["s", "a", "d", "b", "g", "c"]}
    rep = check_plan(module, baseline, plan)
    self.assertFalse(rep.accepted)
    self.assertIsNotNone(rep.after)  # numbers are still reported
    self.assertEqual({v.code for v in rep.violations if v.severity is _Sev.ERROR}, {"PRIORITY_REGRESSION"})
    self.assertTrue(check_plan(module, baseline, plan, allow_priority_regression=True).accepted)
    self.assertTrue(check_plan(module, baseline, plan, priority_prefixes=("collective-permute",)).accepted)

  def test_analysis_rows_carry_priority_and_reachable_deficit(self):
    comp = self._comp()
    report = analyze_module(Module.from_computations("m", [comp]), {"c": ["s", "a", "d", "b", "g", "c"]})
    (row,) = report.computations[0].rows
    self.assertTrue(row.priority)
    self.assertEqual((row.deficit_us, row.reachable_deficit_us), (200.0, 0.0))
    self.assertEqual(report.priority_exposed_us, 200.0)


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
    _write(self.plan, json.dumps({"computations": {"main": ["w", "x", "ag-start", "mm", "mm2", "ag-done", "out"]}}))
    status, text = self._run("check-plan", self.proto, "--profile", self.profile, "--plan", self.plan, "--json")
    self.assertEqual(status, 0)
    doc = json.loads(text)
    self.assertTrue(doc["accepted"])
    delta = doc["deltas"][0]
    self.assertEqual((delta["static_before_us"], delta["static_after_us"]), (550.0, 550.0))

  def test_check_plan_rejects_priority_regression_unless_allowed(self):
    # Legal, but mm moves in front of ag-start, so the all-gather loses 100 us of overlap.
    _write(self.plan, json.dumps({"computations": {"main": ["x", "w", "mm", "ag-start", "mm2", "ag-done", "out"]}}))
    args = ("check-plan", self.proto, "--profile", self.profile, "--plan", self.plan)
    status, text = self._run(*args)
    self.assertEqual(status, 1)
    self.assertIn("PRIORITY_REGRESSION", text)
    self.assertIn("550.0 -> 650.0", text)
    out = os.path.join(self.create_tempdir().full_path, "o.json")
    stamp = ("stamp-plan", self.proto, "--profile", self.profile, "--plan", self.plan, "--out", out)
    self.assertEqual(self._run(*stamp)[0], 1)
    status, text = self._run(*args, "--allow-priority-regression")
    self.assertEqual(status, 0, text)
    self.assertIn("WARNING PRIORITY_REGRESSION", text)

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
    _write(self.plan, json.dumps({"computations": {"main": ["w", "x", "ag-start", "mm", "mm2", "ag-done", "out"]}}))
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
    _write(self.plan, json.dumps({"computations": {"main": ["w", "x", "ag-start", "mm", "mm2", "ag-done", "out"]}}))
    status, text = self._run("check-plan", self.proto, "--plan", self.plan, "--require-identity")
    self.assertEqual(status, 1)
    self.assertIn("PLAN_MISSING_IDENTITY", text)
    _write(
        self.plan,
        json.dumps(
            {
                "module": "tiny",
                "input_checksum": "0" * 64,
                "computations": {"main": ["w", "x", "ag-start", "mm", "mm2", "ag-done", "out"]},
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


# The all-gather starts after the compute that could have covered it.
_LATE_START_HLO = _TINY_HLO.replace(
    "  %ag-start = ((bf16[16,64]{1,0}), bf16[128,64]{1,0}) async-start(%x), calls=%async_comp\n", ""
).replace(
    "  %ag-done =",
    "  %ag-start = ((bf16[16,64]{1,0}), bf16[128,64]{1,0}) async-start(%x), calls=%async_comp\n"
    "  %mm3 = bf16[16,64]{1,0} dot(%mm2, %w), lhs_contracting_dims={1}, rhs_contracting_dims={0}\n"
    "  %ag-done =",
).replace("tuple(%ag-done, %mm2)", "tuple(%ag-done, %mm3)")
_LATE_START_PROFILE = (
    'costs { name: "mm" cost_us: 100 }\ncosts { name: "mm2" cost_us: 250 }\n'
    'costs { name: "mm3" cost_us: 100 }\ncosts { name: "ag" cost_us: 900 }\n'
)


@absltest.skipIf(_hlo is None, "jaxlib not available")
class OptimizeAndExchangeTest(absltest.TestCase):
  """`cli optimize --out-module` and the module_exchange dump / load transform."""

  def setUp(self):
    super().setUp()
    self.dir = self.create_tempdir().full_path
    self.module_bytes = _hlo.hlo_module_from_text(_LATE_START_HLO).as_serialized_hlo_module_proto()
    self.module = os.path.join(self.dir, "late.pb")
    self.profile = os.path.join(self.dir, "late.fdo_profile")
    _write(self.module, self.module_bytes)
    _write(self.profile, _LATE_START_PROFILE)

  def _run(self, *argv):
    out = io.StringIO()
    return cli.main(list(argv), out=out), out.getvalue()

  def _order(self, data: bytes) -> list[str]:
    return load_module(_hlo.HloModule.from_serialized_hlo_module_proto(data), Profile()).baseline["main"]

  def _optimized(self) -> str:
    path = os.path.join(self.dir, "optimized.pb")
    status, text = self._run(
        "optimize", self.module, "--profile", self.profile, "--out-module", path, "--time-budget-s", "5"
    )
    self.assertEqual(status, 0, text)
    return path

  def test_optimize_writes_a_module_with_the_start_hoisted(self):
    before = self._order(self.module_bytes)
    self.assertLess(before.index("mm2"), before.index("ag-start"))
    with open(self._optimized(), "rb") as f:
      after = self._order(f.read())
    self.assertLess(after.index("ag-start"), after.index("mm"))
    self.assertLess(after.index("ag-start"), after.index("ag-done"))

  def test_analyze_flags_a_latency_far_above_its_peers(self):
    # Five all-gathers; one profiled at 50 ms against 1 ms for the rest.
    comps = []
    insts = [Inst("p", "parameter")]
    for k in range(5):
      lat = 50_000.0 if k == 0 else 1_000.0
      insts += [
          Inst(f"ag-start.{k}", "all-gather-start", ("p",), async_role=AsyncRole.START, latency_us=lat, uses_comm_stream=True, profile_key=f"ag{k}"),
          Inst(f"ag-done.{k}", "all-gather-done", (f"ag-start.{k}",), async_role=AsyncRole.DONE),
      ]
    comp = Computation.from_instructions("main", insts)
    report = analyze_module(Module.from_computations("m", [comp]), {"main": [i.name for i in insts]})
    flagged = report.suspicious_latencies()
    self.assertEqual([(c, s) for c, s, _, _ in flagged], [("main", "ag-start.0")])

  def test_optimize_needs_an_output(self):
    self.assertEqual(self._run("optimize", self.module, "--profile", self.profile)[0], 2)

  def test_dump_then_optimize_dump_roundtrip(self):
    dump = os.path.join(self.dir, "dump")
    config = b""  # no profile in the config: the first compile of PGLE
    stem = module_exchange.dump_module(self.module_bytes, config, dump, min_bytes=0)
    self.assertTrue(stem.endswith("tiny.post_scheduler.0.hlo_module.pb"), stem)
    out_module = os.path.join(self.dir, "from_dump.pb")
    # Only a compile that had a profile is picked.
    self.assertEqual(self._run("optimize-dump", dump, "--module-name", ".*", "--out-module", out_module)[0], 2)
    _write(stem.replace(".hlo_module.pb", ".fdo_profile.txt"), _LATE_START_PROFILE)
    status, text = self._run(
        "optimize-dump", dump, "--module-name", ".*", "--out-module", out_module, "--time-budget-s", "5"
    )
    self.assertEqual(status, 0, text)
    with open(out_module, "rb") as f:
      self.assertLess(self._order(f.read()).index("ag-start"), self._order(self.module_bytes).index("ag-start"))

  def test_dump_skips_small_and_unmatched_modules(self):
    dump = os.path.join(self.dir, "dump")
    self.assertIsNone(module_exchange.dump_module(self.module_bytes, b"", dump, min_bytes=10**9))
    self.assertIsNone(module_exchange.dump_module(self.module_bytes, b"", dump, name_re="^nope$", min_bytes=0))
    self.assertFalse(os.path.exists(dump))

  def test_load_returns_the_file_only_for_the_matching_module(self):
    optimized = self._optimized()
    callback = module_exchange.make_callback(None, optimized)
    with open(optimized, "rb") as f:
      expected = f.read()
    self.assertEqual(callback(self.module_bytes), expected)
    # another module name: XLA keeps its schedule
    other = _hlo.hlo_module_from_text(_LATE_START_HLO.replace("HloModule tiny", "HloModule other")).as_serialized_hlo_module_proto()
    self.assertIsNone(callback(other))
    # the same name, one instruction fewer: merged, and the result equals XLA's, so nothing changes
    changed = _hlo.hlo_module_from_text(_TINY_HLO).as_serialized_hlo_module_proto()
    self.assertIsNone(callback(changed))

  def test_load_merges_an_instruction_the_optimized_file_does_not_have(self):
    """A copy that appears in a later compile is slotted in; the optimized order is kept for the rest."""
    optimized = self._optimized()
    with_copy = _LATE_START_HLO.replace(
        "  %ag-done =", "  %cp = bf16[16,64]{1,0} copy(%mm3)\n  %ag-done =", 1
    ).replace("tuple(%ag-done, %mm3)", "tuple(%ag-done, %cp)")
    data = _hlo.hlo_module_from_text(with_copy).as_serialized_hlo_module_proto()
    out = module_exchange.make_callback(None, optimized)(data)
    self.assertIsNotNone(out)
    order = self._order(out)
    self.assertIn("cp", order)
    self.assertLess(order.index("ag-start"), order.index("mm"))  # the optimized order survived
    self.assertLess(order.index("mm3"), order.index("cp"))

  def test_load_refuses_when_too_many_instructions_differ(self):
    optimized = self._optimized()
    extra = "".join(f"  %c{k} = bf16[16,64]{{1,0}} copy(%mm3)\n" for k in range(20))
    bigger = _LATE_START_HLO.replace("  %ag-done =", extra + "  %ag-done =", 1)
    data = _hlo.hlo_module_from_text(bigger).as_serialized_hlo_module_proto()
    self.assertIsNone(module_exchange.make_callback(None, optimized)(data))

  def test_only_rank_zero_computes_and_every_rank_gets_the_same_bytes(self):
    import threading  # pylint: disable=import-outside-toplevel

    optimized = self._optimized()
    store, cond = {}, threading.Condition()
    barriers = {}

    class FakeClient:

      def key_value_set_bytes(self, key, value):
        with cond:
          store[key] = value
          cond.notify_all()

      def blocking_key_value_get_bytes(self, key, timeout_ms):
        with cond:
          assert cond.wait_for(lambda: key in store, timeout_ms / 1000), key
          return store[key]

      def wait_at_barrier(self, name, timeout_ms):
        with cond:
          barriers[name] = barriers.get(name, 0) + 1
          cond.notify_all()
          assert cond.wait_for(lambda: barriers[name] == 2, 10), name

    client = FakeClient()
    computed, results = [], {}
    real = module_exchange.OptimizedModule.returns_for

    def counting(self, data):
      computed.append(threading.current_thread().name)
      return real(self, data)

    def rank(r):
      callback = module_exchange.make_callback(None, optimized, distributed=lambda: (client, r))
      results[r] = callback(self.module_bytes)

    module_exchange.OptimizedModule.returns_for = counting
    try:
      threads = [threading.Thread(target=rank, args=(r,), name=f"rank{r}") for r in (1, 0)]
      for t in threads:
        t.start()
      for t in threads:
        t.join(30)
    finally:
      module_exchange.OptimizedModule.returns_for = real
    self.assertEqual(computed, ["rank0"])
    self.assertIsNotNone(results[0])
    self.assertEqual(results[0], results[1])

  def test_load_ignores_an_equal_schedule_and_survives_garbage(self):
    callback = module_exchange.make_callback(None, self.module)  # the file equals XLA's input
    self.assertIsNone(callback(self.module_bytes))
    self.assertIsNone(callback(b"\x00garbage"))

  def test_settings_pick_one_mode_and_check_its_path(self):
    settings = module_exchange.settings
    self.assertEqual(settings({}), ("off", ""))
    self.assertEqual(settings({"COLLECTIVE_HLO_MODE": "off", "COLLECTIVE_HLO_PATH": "/x"}), ("off", "/x"))
    self.assertEqual(settings({"COLLECTIVE_HLO_MODE": "DUMP", "COLLECTIVE_HLO_PATH": "/d"}), ("dump", "/d"))
    self.assertEqual(settings({"COLLECTIVE_HLO_MODE": "load", "COLLECTIVE_HLO_PATH": self.module}), ("load", self.module))
    with self.assertRaises(ValueError):
      settings({"COLLECTIVE_HLO_MODE": "both", "COLLECTIVE_HLO_PATH": "/d"})
    with self.assertRaises(ValueError):
      settings({"COLLECTIVE_HLO_MODE": "dump"})
    with self.assertRaises(FileNotFoundError):
      settings({"COLLECTIVE_HLO_MODE": "load", "COLLECTIVE_HLO_PATH": os.path.join(self.dir, "missing.pb")})

  def test_register_is_a_noop_when_the_mode_is_off(self):
    old = {k: os.environ.pop(k, None) for k in (module_exchange.MODE_ENV, module_exchange.PATH_ENV)}
    try:
      self.assertFalse(module_exchange.register())
      os.environ[module_exchange.MODE_ENV] = "off"
      os.environ[module_exchange.PATH_ENV] = self.dir
      self.assertFalse(module_exchange.register())
    finally:
      for k in old:
        os.environ.pop(k, None)
        if old[k] is not None:
          os.environ[k] = old[k]


def _varint(n: int) -> bytes:
  out = bytearray()
  while True:
    byte = n & 0x7F
    n >>= 7
    out.append(byte | (0x80 if n else 0))
    if not n:
      return bytes(out)


def _field(number: int, payload: bytes) -> bytes:
  return _varint(number << 3 | 2) + _varint(len(payload)) + payload


def _fake_header(params: int, outputs: int) -> bytes:
  """A size-delimited metadata message with `params` parameter specs (field 6) and `outputs` output specs (field 5)."""
  body = _field(2, b"pjrt_ifrt") + b"".join(_field(6, b"p") for _ in range(params)) + b"".join(_field(5, b"o") for _ in range(outputs))
  return _varint(len(body)) + body


_FAKE_XLA_COMPILE = """#!/bin/sh
for a in "$@"; do
  case "$a" in
    --output_file=*) out="${a#--output_file=}";;
    --module_config_file=*) cfg="${a#--module_config_file=}";;
  esac
done
printf '%s\\n' "$@" > "$out.args"
cp "$cfg" "$out.config"
printf 'executable' > "$out"
"""

_DUMPED_CONFIG = """entry_computation_layout {
  parameters {
    element_type: F32
  }
}
replica_count: 1
num_partitions: 8
debug_options {
  xla_dump_to: "/original/dump"
  xla_dump_hlo_as_text: true
}
"""


class AotBundleTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.dir = self.create_tempdir().full_path
    self.dump = os.path.join(self.dir, "dump")
    os.makedirs(self.dump)
    _write(os.path.join(self.dump, "jit_step.aot_header"), _fake_header(3, 2))
    _write(os.path.join(self.dump, "module_5.jit_step.config.pbtxt"), _DUMPED_CONFIG)
    _write(os.path.join(self.dump, "module_5.jit_step.autotune_results.pbtxt"), "")
    _write(os.path.join(self.dump, "module_5.jit_step.gpu_target_config.pbtxt"), "gpu_device_info {}")
    _write(os.path.join(self.dump, "jit_step.post_scheduler.1.hlo_module.pb"), b"module")
    _write(os.path.join(self.dump, "module_5.jit_step.after_spmd_partitioner.fdo_profile"), "profile")
    self.optimized = os.path.join(self.dir, "optimized.pb")
    _write(self.optimized, b"optimized")
    self.compiler = os.path.join(self.dir, "xla_compile")
    _write(self.compiler, _FAKE_XLA_COMPILE)
    os.chmod(self.compiler, 0o755)
    self.out = os.path.join(self.dir, "bundle")

  def _build(self):
    profile = os.path.join(self.dump, "module_5.jit_step.after_spmd_partitioner.fdo_profile")
    return aot_bundle.build(self.dump, "jit_step", self.optimized, self.out, profile=profile, xla_compile=self.compiler)

  def test_header_counts(self):
    header = _fake_header(886, 893)
    self.assertEqual(aot_bundle.header_spec_counts(header), (886, 893))
    self.assertEqual(aot_bundle.header_of(header + b"payload"), header)
    self.assertIsNone(aot_bundle.header_of(header[:-1]))

  def test_build_compiles_the_scheduled_module_without_a_gpu(self):
    manifest = self._build()
    executable = aot_bundle.executable_path(self.out, "jit_step")
    args = open(executable + ".args", encoding="utf-8").read().split("\n")
    self.assertIn(f"--module_file={self.optimized}", args)
    self.assertNotIn("--use_attached_device", args)
    self.assertTrue(any(a.startswith("--gpu_target_config=") and a.endswith("module_5.jit_step.gpu_target_config.pbtxt") for a in args))
    self.assertIn("--platform=gpu", args)
    self.assertTrue(any(a.startswith("--autotune_results=") and a.endswith("module_5.jit_step.autotune_results.pbtxt") for a in args))
    config = open(executable + ".config", encoding="utf-8").read()
    self.assertNotIn("entry_computation_layout", config)  # the dumped one is the global, pre-SPMD layout
    self.assertNotIn("xla_dump_to", config)  # must not write into the original run's dump
    self.assertIn("num_partitions: 8", config)
    self.assertIn("xla_dump_hlo_as_text", config)
    self.assertIn("xla_gpu_cudnn_deviceless_compilation_mode: CUDNN_DEVICELESS_COMPILATION_AUTO", config)
    self.assertEqual((manifest.num_parameters, manifest.num_outputs, manifest.num_partitions), (3, 2, 8))
    self.assertEqual(aot_bundle.check(self.out, "jit_step"), manifest)

  def test_deviceless_cudnn_replaces_or_adds_the_mode(self):
    auto = "xla_gpu_cudnn_deviceless_compilation_mode: CUDNN_DEVICELESS_COMPILATION_AUTO"
    disabled = "debug_options {\n  xla_gpu_cudnn_deviceless_compilation_mode: CUDNN_DEVICELESS_COMPILATION_DISABLED\n  other: 1\n}\n"
    self.assertEqual(aot_bundle._enable_deviceless_cudnn(disabled), f"debug_options {{\n  {auto}\n  other: 1\n}}\n")
    self.assertEqual(aot_bundle._enable_deviceless_cudnn("debug_options {\n  other: 1\n}\n"), f"debug_options {{\n  {auto}\n  other: 1\n}}\n")
    self.assertEqual(aot_bundle._enable_deviceless_cudnn("num_partitions: 8\n"), f"num_partitions: 8\ndebug_options {{\n  {auto}\n}}\n")

  def _compile_args(self):
    return open(aot_bundle.executable_path(self.out, "jit_step") + ".args", encoding="utf-8").read().split("\n")

  def test_devices_per_host_comes_from_the_environment(self):
    with mock.patch.dict(os.environ, {aot_bundle.DEVICES_PER_HOST_ENV: "8"}):
      self._build()
    self.assertIn("--num_devices_per_host=8", self._compile_args())

  def test_devices_per_host_is_left_to_xla_compile_when_unset(self):
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop(aot_bundle.DEVICES_PER_HOST_ENV, None)
      self._build()
    self.assertFalse(any(a.startswith("--num_devices_per_host") for a in self._compile_args()))

  def test_devices_per_host_must_be_a_positive_integer(self):
    for bad in ("eight", "0", "-4"):
      with mock.patch.dict(os.environ, {aot_bundle.DEVICES_PER_HOST_ENV: bad}):
        with self.assertRaisesRegex(aot_bundle.BundleError, aot_bundle.DEVICES_PER_HOST_ENV):
          self._build()

  def test_build_needs_the_target_config_of_run_1(self):
    os.remove(os.path.join(self.dump, "module_5.jit_step.gpu_target_config.pbtxt"))
    with self.assertRaisesRegex(aot_bundle.BundleError, "gpu_target_config.pbtxt"):
      self._build()

  def test_build_needs_the_header_and_the_binary(self):
    os.remove(os.path.join(self.dump, "jit_step.aot_header"))
    with self.assertRaisesRegex(aot_bundle.BundleError, "COLLECTIVE_AOT_MODE=dump"):
      self._build()
    _write(os.path.join(self.dump, "jit_step.aot_header"), _fake_header(1, 1))
    with self.assertRaisesRegex(aot_bundle.BundleError, "xla_compile not found"):
      aot_bundle.build(self.dump, "jit_step", self.optimized, self.out, xla_compile=os.path.join(self.dir, "missing"))

  def test_check_rejects_a_bundle_that_is_not_what_it_says(self):
    self._build()
    with self.assertRaisesRegex(aot_bundle.BundleError, "not 'other'"):
      aot_bundle.check(self.out, "other")
    with self.assertRaisesRegex(aot_bundle.BundleError, "jaxlib"):
      aot_bundle.check(self.out, "jit_step", current_jaxlib="0.0.0")
    with open(aot_bundle.executable_path(self.out, "jit_step"), "ab") as f:
      f.write(b"x")
    with self.assertRaisesRegex(aot_bundle.BundleError, "does not match the manifest"):
      aot_bundle.check(self.out, "jit_step")
    os.remove(aot_bundle.header_path(self.out, "jit_step"))
    with self.assertRaisesRegex(aot_bundle.BundleError, "missing"):
      aot_bundle.check(self.out, "jit_step")
    with self.assertRaisesRegex(aot_bundle.BundleError, "cannot read"):
      aot_bundle.check(self.dir, "jit_step")

  @absltest.skipIf(_hlo is None, "cli needs jaxlib")
  def test_cli_build_and_check(self):
    out = io.StringIO()
    code = cli.main(
        ["aot-build", self.dump, "--optimized-module", self.optimized, "--out-dir", self.out, "--module-name", "jit_step", "--xla-compile", self.compiler],
        out=out,
    )
    self.assertEqual(code, 0, out.getvalue())
    out = io.StringIO()
    self.assertEqual(cli.main(["aot-check", self.out, "--module", "jit_step"], out=out), 0)
    self.assertIn("ok: jit_step (3 parameters, 2 outputs)", out.getvalue())
    out = io.StringIO()
    self.assertEqual(cli.main(["aot-check", self.out, "--module", "other"], out=out), 1)
    self.assertIn("not usable", out.getvalue())

  def test_hook_settings(self):
    self.assertEqual(aot_executable.settings({}), ("off", "", "jit_train_step"))
    self.assertEqual(aot_executable.settings({"COLLECTIVE_AOT_MODE": "LOAD", "COLLECTIVE_AOT_PATH": "/b"}), ("load", "/b", "jit_train_step"))
    self.assertEqual(aot_executable.settings({"COLLECTIVE_AOT_MODE": "dump"})[0], "off")  # no path
    self.assertEqual(aot_executable.settings({"COLLECTIVE_AOT_MODE": "bogus", "COLLECTIVE_AOT_PATH": "/b"})[0], "off")
    self.assertEqual(aot_executable.settings({"COLLECTIVE_AOT_MODE": "dump", "COLLECTIVE_AOT_PATH": "/d", "COLLECTIVE_AOT_MODULE": "m"}), ("dump", "/d", "m"))


if __name__ == "__main__":
  absltest.main()
