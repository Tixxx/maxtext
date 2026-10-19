"""Deterministic building blocks for agent-assisted XLA:GPU scheduling.

jax-free: operates on `schedule_model.Computation` and name-list schedules.
"""

from maxtext.utils.agent_assisted_scheduling.cost import (
    CollectiveExposure, collective_exposures, coverage_gaps, total_exposed_us,
)
from maxtext.utils.agent_assisted_scheduling.legality import (
    Severity, Violation, errors, is_legal, validate_order,
)
from maxtext.utils.agent_assisted_scheduling.placement import (
    EarliestLegalPos, earliest_legal_pos, hoist_start_to_floor, reinsert,
)
from maxtext.utils.agent_assisted_scheduling.schedule_model import (
    AsyncRole, Computation, Inst,
)
from maxtext.utils.agent_assisted_scheduling.simulate import SimResult, simulate_comm_stream
