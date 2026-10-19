"""One-FIFO-comm-stream simulation of a schedule.

Port of `_simulate_comm_stream` from collective_overlap_pass.py.

Compute ops run back to back at their cost. Every comm-stream start is queued
and begins at the later of its issue time and the end of the previously queued
comm op; a done stalls compute until its start has finished. NCCL collectives
and custom comm calls share one stream, which the static window sum in cost.py
does not see.

With SM gating, a large GEMM (`Inst.sm_gate`, cost >= `sm_gate_min_us`) holds the
SMs, so a collective cannot begin while one is running. The GEMM timeline depends
on the stalls, which depend on the collective timeline, so it is iterated from
the previous pass's GEMM intervals (three passes converge in practice).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from maxtext.utils.agent_assisted_scheduling.schedule_model import AsyncRole, Computation

DEFAULT_SM_GATE_MIN_US = 1000.0
_GATE_PASSES = 3


@dataclass(frozen=True)
class SimResult:
  total_stall_us: float
  stalls: dict[str, float] = field(default_factory=dict)  # done name -> stall us


def simulate_comm_stream(
    comp: Computation, order: list[str], *,
    sm_gate: bool = True, sm_gate_min_us: float = DEFAULT_SM_GATE_MIN_US,
) -> SimResult:
  """Total compute stall (us) waiting on collectives, and per-done stalls."""
  lat = {n: comp[n].latency_us for n in order
         if comp[n].async_role is AsyncRole.START and comp[n].uses_comm_stream
         and comp[n].latency_us is not None}
  done_start = {n: comp[n].operands[0] for n in order
                if comp[n].async_role is AsyncRole.DONE and comp[n].operands
                and comp[n].operands[0] in lat}
  gate = {n for n in order if comp[n].sm_gate and comp[n].cost_us >= sm_gate_min_us} if sm_gate else set()

  intervals: list[tuple[float, float]] = []
  total = 0.0
  stalls: dict[str, float] = {}
  for _ in range(_GATE_PASSES if gate else 1):
    t = 0.0
    comm = 0.0
    end: dict[str, float] = {}
    total = 0.0
    stalls = {}
    found: list[tuple[float, float]] = []
    for name in order:
      start = done_start.get(name)
      if start is not None:
        e = end.get(start)
        if e is not None and e > t:
          stalls[name] = e - t
          total += e - t
          t = e
      elif name in lat:
        begin = max(t, comm)
        for gs, ge in intervals:
          if gs < begin < ge:
            begin = ge
        comm = begin + lat[name]
        end[name] = comm
      else:
        cost = comp[name].cost_us
        if name in gate:
          found.append((t, t + cost))
        t += cost
    if not gate or found == intervals:
      break
    intervals = found
  return SimResult(total, stalls)
