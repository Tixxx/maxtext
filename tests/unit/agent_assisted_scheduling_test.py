"""Tests for the jax-free agent-assisted scheduling building blocks."""

from absl.testing import absltest

from maxtext.utils.agent_assisted_scheduling import (
    AsyncRole, Computation, Inst, Severity, collective_exposures, coverage_gaps,
    earliest_legal_pos, hoist_start_to_floor, is_legal, simulate_comm_stream,
    total_exposed_us, validate_order,
)


def _start(name, operands=(), latency=500.0, **kw):
  return Inst(name, "all-gather-start", tuple(operands), async_role=AsyncRole.START,
              latency_us=latency, uses_comm_stream=True, **kw)


def _done(name, start, **kw):
  return Inst(name, "all-gather-done", (start,), async_role=AsyncRole.DONE, **kw)


def _fus(name, operands=(), cost=0.0, **kw):
  return Inst(name, "fusion", tuple(operands), cost_us=cost, **kw)


def _codes(violations):
  return sorted(v.code for v in violations)


def _basic():
  """p0, a(100), b(200), c(300), w=convert(p0), ag-start(w), ag-done, use."""
  comp = Computation.from_instructions("main", [
      Inst("p0", "parameter"),
      _fus("a", ["p0"], 100), _fus("b", ["a"], 200), _fus("c", ["b"], 300),
      Inst("w", "convert", ("p0",)),
      _start("s", ["w"]), _done("d", "s"),
      _fus("use", ["d", "c"], 50),
  ])
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
    comp, order = _basic()
    bad = ["p0", "b", "a", "c", "w", "s", "d", "use"]
    self.assertIn("OPERAND_ORDER", _codes(validate_order(comp, bad)))

  def test_start_after_done_is_operand_violation(self):
    comp, order = _basic()
    bad = ["p0", "a", "b", "c", "w", "d", "s", "use"]
    self.assertIn("OPERAND_ORDER", _codes(validate_order(comp, bad)))

  def test_control_order(self):
    comp = Computation.from_instructions("c", [
        Inst("x", "fusion"), Inst("y", "fusion", control_predecessors=("x",)),
    ])
    self.assertTrue(is_legal(comp, ["x", "y"]))
    self.assertEqual(_codes(validate_order(comp, ["y", "x"])), ["CONTROL_ORDER"])

  def test_async_pairing(self):
    comp = Computation.from_instructions("c", [_start("s"), _start("s2")])
    codes = _codes(validate_order(comp, ["s", "s2"]))
    self.assertEqual(codes, ["ASYNC_PAIR", "ASYNC_PAIR"])
    comp = Computation.from_instructions("c", [
        _start("s"), _done("d1", "s"), _done("d2", "s"),
    ])
    self.assertIn("ASYNC_PAIR", _codes(validate_order(comp, ["s", "d1", "d2"])))

  def test_start_used_by_non_done(self):
    comp = Computation.from_instructions("c", [
        _start("s"), _done("d", "s"), _fus("bad", ["s"]),
    ])
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
    comp = Computation.from_instructions("c", [
        _start("s1"), _start("s2"), _start("s3"), _fus("f", [], 10),
        _done("d1", "s1"), _done("d2", "s2"), _done("d3", "s3"),
    ])
    self.assertEqual(validate_order(comp, ["s1", "s2", "s3", "f", "d1", "d2", "d3"]), [])

  def test_device_groups(self):
    g = lambda *ids: (frozenset(ids),)
    comp = Computation.from_instructions("c", [
        _start("s1", device_groups=g(0, 1, 2, 3)),
        _start("s2", device_groups=g(2, 3, 4, 5)),
        _start("s3", device_groups=g(4, 5, 6, 7)),
        _start("s4"),
        _fus("f", [], 10),
        _done("d1", "s1"), _done("d2", "s2"), _done("d3", "s3"), _done("d4", "s4"),
    ])
    kw = dict(check_device_groups=True)
    # s1 & s2 share devices {2,3}
    self.assertIn("DEVICE_GROUP_OVERLAP",
                  _codes(validate_order(comp, ["s1", "s2", "d1", "d2", "s3", "d3", "s4", "d4", "f"], **kw)))
    # s1 & s3 share none
    self.assertTrue(is_legal(comp, ["s1", "s3", "f", "d1", "d3", "s2", "d2", "s4", "d4"], **kw))
    # unknown groups while overlapping
    self.assertIn("DEVICE_GROUPS_UNKNOWN",
                  _codes(validate_order(comp, ["s1", "s4", "d1", "d4", "s2", "d2", "s3", "d3", "f"], **kw)))

  def test_in_place_reorder(self):
    comp = Computation.from_instructions("c", [
        Inst("host", "parameter"),
        Inst("reader", "fusion", ("host",)),
        Inst("writer", "dynamic-update-slice", ("host",), writes_operand="host"),
    ])
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
    self.assertEqual(hoist_start_to_floor(comp, order, "s"),
                     ["p0", "w", "s", "a", "b", "c", "d", "use"])

  def test_non_trivial_producer_pins_floor(self):
    comp = Computation.from_instructions("c", [
        Inst("p0", "parameter"), _fus("a", ["p0"], 100), _fus("b", ["a"], 100),
        Inst("w", "convert", ("b",)), _start("s", ["w"]), _done("d", "s"),
    ])
    order = ["p0", "a", "b", "w", "s", "d"]
    r = earliest_legal_pos(comp, order, "s")
    self.assertEqual(r.floor, 3)
    self.assertEqual(r.blocker, "b")
    self.assertEqual(r.movable, ("w",))
    self.assertEqual(hoist_start_to_floor(comp, order, "s"), order)  # no room

  def test_control_predecessor_pins_floor(self):
    comp = Computation.from_instructions("c", [
        Inst("p0", "parameter"), _fus("a", ["p0"], 100), _fus("b", ["p0"], 100),
        _start("s", ["p0"], control_predecessors=("a",)), _done("d", "s"),
    ])
    order = ["p0", "a", "b", "s", "d"]
    r = earliest_legal_pos(comp, order, "s")
    self.assertEqual(r.floor, 2)
    self.assertIsNone(r.blocker)
    self.assertEqual(r.control_pins, ((1, "a"),))
    new = hoist_start_to_floor(comp, order, "s")
    self.assertEqual(new, ["p0", "a", "s", "b", "d"])
    self.assertTrue(is_legal(comp, new))

  def test_control_pred_inside_moving_set_is_ignored(self):
    comp = Computation.from_instructions("c", [
        Inst("p0", "parameter"), Inst("w", "convert", ("p0",)),
        _fus("a", ["p0"], 100),
        _start("s", ["w"], control_predecessors=("w",)), _done("d", "s"),
    ])
    r = earliest_legal_pos(comp, ["p0", "w", "a", "s", "d"], "s")
    self.assertEqual(r.floor, 0)

  def test_hoisted_order_is_legal(self):
    comp, order = _basic()
    self.assertTrue(is_legal(comp, hoist_start_to_floor(comp, order, "s"), baseline=order))


class CostTest(absltest.TestCase):

  def test_exposure_before_and_after_hoist(self):
    comp, order = _basic()
    (e,) = collective_exposures(comp, order)
    self.assertEqual((e.overlap_us, e.deficit_us), (0.0, 500.0))
    self.assertEqual(total_exposed_us(comp, order), 500.0)
    hoisted = hoist_start_to_floor(comp, order, "s")
    (e,) = collective_exposures(comp, hoisted)
    self.assertEqual(e.overlap_us, 600.0)
    self.assertEqual(total_exposed_us(comp, hoisted), 0.0)

  def test_partial_overlap_and_unprofiled_pairs_ignored(self):
    comp = Computation.from_instructions("c", [
        _start("s", latency=500), _fus("f", [], 120), _done("d", "s"),
        _start("s2", latency=None), _done("d2", "s2"),
    ])
    order = ["s", "f", "d", "s2", "d2"]
    self.assertEqual(total_exposed_us(comp, order), 380.0)
    self.assertLen(collective_exposures(comp, order), 1)

  def test_coverage_gaps(self):
    comp = Computation.from_instructions("c", [
        _start("s", latency=500), _fus("unprofiled", []), _done("d", "s"),
        _start("s2", latency=None), _done("d2", "s2"),
    ])
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
    comp = Computation.from_instructions("c", [
        _start("s1", latency=400), _start("s2", latency=400), _fus("f", [], 500),
        _done("d1", "s1"), _done("d2", "s2"),
    ])
    res = simulate_comm_stream(comp, ["s1", "s2", "f", "d1", "d2"])
    # s1 ends at 400, s2 queues behind it and ends at 800; compute ends at 500.
    self.assertEqual(res.stalls, {"d2": 300.0})
    self.assertEqual(res.total_stall_us, 300.0)

  def test_sm_gate_delays_queued_collective(self):
    comp = Computation.from_instructions("c", [
        _start("s1", latency=1000), _start("s2", latency=100),
        _fus("gemm", [], 2000, sm_gate=True),
        _done("d1", "s1"), _done("d2", "s2"),
    ])
    order = ["s1", "s2", "gemm", "d1", "d2"]
    self.assertEqual(simulate_comm_stream(comp, order, sm_gate=False).total_stall_us, 0.0)
    gated = simulate_comm_stream(comp, order)
    # s2 would begin at 1000 inside the GEMM [0, 2000), so it waits to 2000.
    self.assertEqual(gated.stalls, {"d2": 100.0})

  def test_small_gemm_does_not_gate(self):
    comp = Computation.from_instructions("c", [
        _start("s1", latency=1000), _start("s2", latency=100),
        _fus("gemm", [], 2000, sm_gate=True),
        _done("d1", "s1"), _done("d2", "s2"),
    ])
    order = ["s1", "s2", "gemm", "d1", "d2"]
    res = simulate_comm_stream(comp, order, sm_gate_min_us=5000)
    self.assertEqual(res.total_stall_us, 0.0)

  def test_unprofiled_start_costs_nothing(self):
    comp = Computation.from_instructions("c", [
        _start("s", latency=None), _fus("f", [], 50), _done("d", "s"),
    ])
    self.assertEqual(simulate_comm_stream(comp, ["s", "f", "d"]).total_stall_us, 0.0)


if __name__ == "__main__":
  absltest.main()
