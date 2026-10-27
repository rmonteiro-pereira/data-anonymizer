"""The plain-language task description participants get with a challenge.

It is written from the schema's question and from what is published: the files, the columns,
the score and the rules. It never reads the answer key's findings about how the data was built,
so it does not name drivers, formulas, missing-value mechanisms, label noise, drift or
aggregates. Anything it states, such as the share of the positive class in the training rows,
a participant could measure from the published files.
"""

from collections.abc import Sequence

import pandas as pd

from oblitus.synthesis.challenge import Challenge
from oblitus.synthesis.relational import RelatedTable, RelationalSchema
from oblitus.synthesis.schema import DatasetSchema, Metric, Question, Task

# A positive class rarer than this is called rare in the text, and scored by default with PR AUC.
RARE_CLASS_SHARE = 0.2

_METRIC_TEXT = {
    Metric.RMSE: "root mean squared error (lower is better)",
    Metric.MAE: "mean absolute error (lower is better)",
    Metric.R2: "R squared (higher is better)",
    Metric.ACCURACY: "accuracy (higher is better)",
    Metric.ROC_AUC: "area under the ROC curve (higher is better)",
    Metric.PR_AUC: "area under the precision-recall curve (higher is better)",
    Metric.LOG_LOSS: "log loss (lower is better)",
}


def default_metric(question: Question, positive_rate: float | None) -> Metric:
    """Return the metric a question uses when it names none."""
    if question.task is Task.REGRESSION:
        return Metric.RMSE
    is_rare = question.positive_share is not None or (
        positive_rate is not None and positive_rate < RARE_CLASS_SHARE
    )
    return Metric.PR_AUC if is_rare else Metric.ROC_AUC


def describe_task(schema: DatasetSchema | RelationalSchema, challenge: Challenge) -> str:
    """Return the Markdown task description for the published files of a challenge."""
    question = schema.question
    assert question is not None
    key = challenge.answer_key
    sections = [
        f"# {question.title}\n\n{question.prompt}",
        _what_to_predict(question, challenge),
        _files(question, challenge, schema),
        _scoring(question, challenge),
        _data_notes(question, challenge, key),
        _rules(),
        _columns(challenge),
    ]
    return "\n\n".join(section for section in sections if section) + "\n"


def _what_to_predict(question: Question, challenge: Challenge) -> str:
    kind = "a number" if question.task is Task.REGRESSION else "which class a row belongs to"
    text = (
        f"## What to predict\n\nPredict `{question.target}` ({kind}) for every row of `test.csv`."
    )
    if question.task is Task.CLASSIFICATION:
        positive = "1 (True)" if question.positive_label is None else question.positive_label
        text += f" The class to detect is `{positive}`."
        rate = _positive_rate(question, challenge.train)
        if rate is not None and rate < RARE_CLASS_SHARE:
            text += f" It is rare: about {rate:.0%} of the training rows."
    return text


def _positive_rate(question: Question, train: pd.DataFrame) -> float | None:
    labels = train[question.target].dropna()
    if labels.empty:
        return None
    if question.positive_label is not None:
        return float((labels.astype(str) == question.positive_label).mean())
    try:
        return float(labels.astype(float).mean())
    except (TypeError, ValueError):
        return None


def _files(
    question: Question, challenge: Challenge, schema: DatasetSchema | RelationalSchema
) -> str:
    lines = [
        f"- `train.csv`: {len(challenge.train)} rows, with `{question.target}`.",
        f"- `test.csv`: {len(challenge.test_features)} rows, without `{question.target}`.",
    ]
    text = "## Files\n\n" + "\n".join(lines)
    if isinstance(schema, RelationalSchema):
        text += _related_files(schema, challenge)
    return text


def _related_files(schema: RelationalSchema, challenge: Challenge) -> str:
    lines = []
    root = schema.tables[0]
    for name, frame in challenge.related_tables.items():
        table = schema.table(name)
        lines.append(f"- `{name}.csv`: {len(frame)} rows. {_link_sentence(table)}")
    note = (
        f"The `{root.name}` rows in `train.csv` and `test.csv` are identified by `{root.key}`. "
        "Information about their related rows is in the other files."
    )
    return "\n" + "\n".join(lines) + "\n\n" + note


def _link_sentence(table: RelatedTable) -> str:
    return (
        f"Each row belongs to one row of `{table.parent}` through `{table.foreign_key}`, "
        f"and a `{table.parent}` row can have many rows here, or none."
    )


def _scoring(question: Question, challenge: Challenge) -> str:
    rate = _positive_rate(question, challenge.train)
    metric = question.metric or default_metric(question, rate)
    text = f"## How you are scored\n\nThe score is the {_METRIC_TEXT[metric]}, on `test.csv`."
    if question.task is Task.CLASSIFICATION and metric in (Metric.ROC_AUC, Metric.PR_AUC):
        text += " Predict the probability of the class to detect, not just a label."
    if metric is Metric.PR_AUC:
        text += (
            " Accuracy would be misleading here, because always answering the common class "
            "is right most of the time."
        )
    return text


def _data_notes(question: Question, challenge: Challenge, key: dict) -> str:
    notes = []
    blanks = _columns_with_blank_cells(challenge)
    if blanks:
        names = ", ".join(f"`{name}`" for name in blanks)
        notes.append(f"These columns have blank cells: {names}. Decide how to treat them.")
    causal = key.get("causal")
    if causal is not None:
        values = causal["treatment_values"]
        notes.append(
            f"Treatment column: `{causal['treatment']}` (control {values['control']!r}, "
            f"treated {values['treated']!r})."
        )
    series = key.get("time_series")
    if series is not None:
        notes.append(_panel_sentence(series))
    return "## About the data\n\n" + "\n\n".join(notes) if notes else ""


def _columns_with_blank_cells(challenge: Challenge) -> list[str]:
    tables: Sequence[pd.DataFrame] = [challenge.train, challenge.test_features]
    names = []
    for name in challenge.train.columns:
        if any(name in frame.columns and frame[name].isna().any() for frame in tables):
            names.append(str(name))
    return names


def _panel_sentence(series: dict) -> str:
    note = f"Each row is one `{series['entity']}` in one period, dated by `{series['time']}`."
    first_test = series["first_test_period"]
    if first_test is not None:
        periods = series["periods_per_split"]["test"]
        note += f" The test rows are the last {periods} periods, from {first_test['date']}."
    return note


def _rules() -> str:
    return (
        "## Rules\n\n"
        "- Fit your model on the training rows only; the test answers are not available.\n"
        "- Give one prediction per row of `test.csv`, in the same order."
    )


def _columns(challenge: Challenge) -> str:
    lines = "\n".join(f"- `{name}`" for name in challenge.train.columns)
    return f"## Columns of `train.csv`\n\n{lines}"
