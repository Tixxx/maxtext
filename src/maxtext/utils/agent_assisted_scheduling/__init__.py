"""Deterministic building blocks for agent-assisted XLA:GPU scheduling.

jax-free: operates on `schedule_model.Module` / `Computation` and name-list
schedules. `hlo_adapter` (needs jaxlib) loads a real HloModule into that model and
applies a validated plan; import it directly.
"""

from maxtext.utils.agent_assisted_scheduling.cost import (
    CollectiveExposure,
    collective_exposures,
    coverage_gaps,
    total_exposed_us,
)
from maxtext.utils.agent_assisted_scheduling.legality import (
    Severity,
    Violation,
    errors,
    is_legal,
    validate_order,
    validate_plan,
)
from maxtext.utils.agent_assisted_scheduling.placement import (
    EarliestLegalPos,
    earliest_legal_pos,
    hoist_start_to_floor,
    reinsert,
)
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
    PriorityRegression,
    is_priority,
    priority_regressions,
)
from maxtext.utils.agent_assisted_scheduling.profile import Profile
from maxtext.utils.agent_assisted_scheduling.schedule_model import (
    AsyncRole,
    Computation,
    Inst,
    Module,
)
from maxtext.utils.agent_assisted_scheduling.simulate import SimResult, simulate_comm_stream
