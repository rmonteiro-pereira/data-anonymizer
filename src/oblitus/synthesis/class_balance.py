"""Class imbalance for classification challenges.

A question's `positive_share` is met by shifting the log-odds of its `truth_column` (the
probability the target is drawn from) by one constant, so the rows keep their order and the
planted dependence on the other columns, and only the rate of the positive class moves. The
answer key then reports scores that suit a rare positive class: the precision-recall AUC of the
true probabilities, next to the baseline a model with no skill reaches.
"""

import numpy as np

from oblitus.synthesis.schema import DatasetSchema


def with_positive_share(schema: DatasetSchema) -> DatasetSchema:
    """Return the schema with the question's positive share set on its truth column.

    A schema whose question has no `positive_share` is returned as it is.
    """
    question = schema.question
    if question is None or question.positive_share is None:
        return schema
    assert question.truth_column is not None
    columns = [
        column.model_copy(update={"probability_share": question.positive_share})
        if column.name == question.truth_column
        else column
        for column in schema.columns
    ]
    return schema.with_changes(columns=columns)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Return the area under the precision-recall curve, as the average precision.

    Rows with the same score count as one threshold, so ties do not favour either class.
    """
    positives = int((labels == 1).sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels == 1)
    predicted = np.arange(1, len(labels) + 1)
    last_of_tie = np.append(sorted_scores[1:] != sorted_scores[:-1], True)
    precision = true_positives[last_of_tie] / predicted[last_of_tie]
    recall = true_positives[last_of_tie] / positives
    previous_recall = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - previous_recall) * precision))


def describe_class_balance(
    requested_share: float,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    test_probabilities: np.ndarray | None,
) -> dict[str, object]:
    """Return the answer key's class-balance section.

    The labels are the published ones as 0/1 numbers with NaN where missing. The probabilities
    are the noise-free truth of the test rows, or None when the key cannot compute it.
    """
    train_rate = _positive_rate(train_labels)
    test_rate = _positive_rate(test_labels)
    section: dict[str, object] = {
        "requested_positive_share": requested_share,
        "positive_rate": {"train": train_rate, "test": test_rate},
        "majority_class_accuracy": max(test_rate, 1 - test_rate),
        "pr_auc_baseline": test_rate,
    }
    if test_probabilities is None:
        section["best_achievable_pr_auc"] = "not computable: the truth is not available"
        return section
    scored = ~np.isnan(test_labels) & ~np.isnan(test_probabilities)
    section["best_achievable_pr_auc"] = average_precision(
        test_labels[scored], test_probabilities[scored]
    )
    return section


def _positive_rate(labels: np.ndarray) -> float:
    present = labels[~np.isnan(labels)]
    return float(present.mean()) if len(present) else float("nan")
