"""Plain-text summaries of a plan and of a release, for the terminal and for notebooks.

The command line prints these, and `AnonymizationPlan.describe()` and
`AnonymizationResult.describe()` return them, so both show the same words.
"""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oblitus.anonymizer import AnonymizationResult
    from oblitus.chunked_apply import ChunkedResult
    from oblitus.plan import AnonymizationPlan, ColumnAction


_COLUMN_WIDTH = 22
# Columns whose added blank share is below this are not worth a line of their own.
_NOTABLE_BLANK_SHARE = 0.05


def describe_plan(plan: "AnonymizationPlan") -> str:
    """What the plan does to each column, and what it protects."""
    lines = [f"{plan.protection.value} plan, {plan.fidelity.value} fidelity", ""]
    for action in plan.actions:
        lines.extend(_action_lines(action))
    lines.append("")
    if plan.quasi_identifiers:
        lines.append(f"quasi-identifiers: {', '.join(plan.quasi_identifiers)}")
    if plan.enforce_k_anonymity:
        method = plan.k_anonymity_method.value.replace("_", " ")
        lines.append(f"k-anonymity: at least {plan.k_target} rows alike, by {method}")
    return "\n".join(lines)


def _action_lines(action: "ColumnAction") -> list[str]:
    """The column and its technique on one line, the reason indented below."""
    technique = action.technique
    if action.params:
        # JSON spelling (false, "ID") is what the reader will edit in the plan file.
        settings = ", ".join(f"{name}={json.dumps(value)}" for name, value in action.params.items())
        technique = f"{technique} ({settings})"
    lines = [f"  {action.column:<{_COLUMN_WIDTH}} {technique}"]
    if action.reason:
        lines.append(f"  {'':<{_COLUMN_WIDTH}} {action.reason}")
    return lines


def describe_result(result: "AnonymizationResult | ChunkedResult") -> str:
    """The protection reached, why it fell short if it did, and what the release cost."""
    before = result.privacy_before.k_anonymity
    after = result.privacy_after.k_anonymity
    lines = [
        f"result is {result.protection.level.value} "
        f"(target {result.protection.target.value}); k-anonymity {before} -> {after}"
    ]
    if result.protection.problems:
        lines.append("why the target was not reached:")
        lines.extend(f"  - {problem}" for problem in result.protection.problems)
    lines.extend(_cost_lines(result))
    return "\n".join(lines)


def _cost_lines(result: "AnonymizationResult | ChunkedResult") -> list[str]:
    blanked = [
        (item.column, item.added_blank_share)
        for item in result.utility.columns
        if item.added_blank_share is not None and item.added_blank_share >= _NOTABLE_BLANK_SHARE
    ]
    lines = []
    if blanked:
        listed = ", ".join(f"{column} {share:.0%}" for column, share in blanked)
        lines.append(f"cells blanked to protect rare rows: {listed}")
    change = result.utility.max_correlation_change
    if change is not None:
        lines.append(f"largest change in a correlation between numeric columns: {change:.2f}")
    return lines
