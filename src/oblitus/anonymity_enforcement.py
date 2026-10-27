"""The table-level steps a plan can run on the released table after its column actions.

In order:
1. Joint microaggregation of sets of numeric columns, when the plan lists any.
2. k-anonymity, by local suppression (the default) or by Mondrian generalization.
3. l-diversity enforcement, when the plan asks for it.

Each step works on the whole table in its engine. This module only decides which steps run
and collects what they changed, so the facade stays free of the rules.
"""

from dataclasses import dataclass

from oblitus.diversity_enforcement import enforce_l_diversity
from oblitus.engines import WorkingFrame
from oblitus.joint_microaggregation import microaggregate_jointly
from oblitus.local_suppression import enforce_k_anonymity
from oblitus.mondrian import enforce_k_anonymity_by_mondrian
from oblitus.plan import AnonymizationPlan, KAnonymityMethod
from oblitus.table import ROW_KEY_COLUMN


@dataclass(frozen=True, slots=True)
class EnforcementOutcome:
    """The released table after the plan's table-level steps.

    `changed_columns` are the columns these steps rewrote, so a pandas table can hand back
    the rewritten values and not the user's originals.
    """

    data: WorkingFrame
    suppressed_cells: int
    changed_columns: list[str]


def enforce_plan_targets(released: WorkingFrame, plan: AnonymizationPlan) -> EnforcementOutcome:
    """Run the plan's microaggregation, k-anonymity and l-diversity steps on a released table.

    `released` carries `ROW_KEY_COLUMN`, which identifies its rows for the steps that need it.
    """
    changed: list[str] = []
    for step in plan.joint_microaggregation:
        outcome = microaggregate_jointly(
            released, step.columns, step.group_size, row_key=ROW_KEY_COLUMN
        )
        released = outcome.data
        changed += step.columns

    suppressed = 0
    if plan.enforce_k_anonymity:
        released, cells, columns = _enforce_k(released, plan)
        suppressed += cells
        changed += columns

    if plan.enforce_l_diversity:
        diverse = enforce_l_diversity(
            released, plan.quasi_identifiers, plan.sensitive_attributes, plan.l_target
        )
        released = diverse.data
        suppressed += diverse.suppressed_cells
        if diverse.suppressed_cells:
            changed += plan.quasi_identifiers
    return EnforcementOutcome(released, suppressed, list(dict.fromkeys(changed)))


def _enforce_k(
    released: WorkingFrame, plan: AnonymizationPlan
) -> tuple[WorkingFrame, int, list[str]]:
    if plan.k_anonymity_method is KAnonymityMethod.MONDRIAN:
        generalized = enforce_k_anonymity_by_mondrian(
            released,
            plan.quasi_identifiers,
            plan.k_target,
            hierarchies=plan.hierarchies,
            style=plan.range_style,
        )
        return generalized.data, 0, generalized.changed_columns
    suppressed = enforce_k_anonymity(released, plan.quasi_identifiers, plan.k_target)
    columns = list(plan.quasi_identifiers) if suppressed.suppressed_cells else []
    return suppressed.data, suppressed.suppressed_cells, columns
