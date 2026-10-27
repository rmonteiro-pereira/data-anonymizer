import numpy as np
import pandas as pd
import pytest

from data_anonymizer import Anonymizer, ColumnSchema, DatasetSchema
from data_anonymizer.synthesis.expressions import ExpressionError, evaluate, referenced_columns
from data_anonymizer.synthesis.schema import ColumnOverride, Correlation, Distribution
from data_anonymizer.taxonomy import ColumnKind

ROWS = 4000


def _generate(schema: DatasetSchema, rows: int = ROWS, seed: int = 11) -> pd.DataFrame:
    return Anonymizer(seed=seed).synthesize(rows, schema=schema)


def test_expressions_compute_over_columns_and_reject_code():
    frame = pd.DataFrame({"a": [1.0, 2.0], "plan": ["pro", "free"]})

    assert evaluate("a * 2 + 1", frame).tolist() == [3.0, 5.0]
    assert evaluate("where(plan == 'pro', a, 0)", frame).tolist() == [1.0, 0.0]
    assert evaluate("plan in ['pro', 'team']", frame).tolist() == [True, False]
    assert referenced_columns("log(a) + where(plan == 'x', 1, 2)") == {"a", "plan"}
    for unsafe in ("__import__('os')", "a.__class__", "[x for x in a]", "open('f')"):
        with pytest.raises(ExpressionError):
            evaluate(unsafe, frame)


@pytest.mark.parametrize(
    ("distribution", "mean", "std"),
    [
        (Distribution.NORMAL, 50.0, 10.0),
        (Distribution.LOGNORMAL, 3000.0, 1500.0),
        (Distribution.GAMMA, 20.0, 8.0),
    ],
)
def test_distributions_hit_the_requested_mean_and_std(distribution, mean, std):
    schema = DatasetSchema(
        columns=[
            ColumnSchema(
                name="x", kind=ColumnKind.FLOAT, distribution=distribution, mean=mean, std=std
            )
        ]
    )

    values = _generate(schema)["x"]

    assert values.mean() == pytest.approx(mean, rel=0.05)
    assert values.std() == pytest.approx(std, rel=0.1)


def test_a_discrete_distribution_keeps_its_minimum_when_truncated():
    schema = DatasetSchema(
        columns=[
            ColumnSchema(
                name="visits",
                kind=ColumnKind.INTEGER,
                distribution=Distribution.POISSON,
                mean=3,
                minimum=2,
                maximum=5,
            ),
            ColumnSchema(
                name="zeros",
                kind=ColumnKind.INTEGER,
                distribution=Distribution.POISSON,
                mean=1,
                minimum=0,
            ),
        ]
    )

    frame = _generate(schema, rows=40_000)
    shares = frame["visits"].value_counts(normalize=True)

    assert shares[2] == pytest.approx(0.312, abs=0.015)
    assert shares[5] == pytest.approx(0.141, abs=0.015)
    assert (frame["zeros"] == 0).mean() == pytest.approx(0.368, abs=0.015)


@pytest.mark.parametrize(
    "settings",
    [
        {"distribution": Distribution.UNIFORM, "minimum": 1, "maximum": 6},
        {"mean": 3.5, "std": 10, "minimum": 1, "maximum": 6},
        {"quantiles": {0.5: 3.5}, "minimum": 1, "maximum": 6},
    ],
)
def test_integer_bounds_are_as_likely_as_the_values_between_them(settings):
    schema = DatasetSchema(columns=[ColumnSchema(name="roll", kind=ColumnKind.INTEGER, **settings)])

    shares = _generate(schema, rows=60_000)["roll"].value_counts(normalize=True)

    assert shares[1] > 0.8 * shares[3]
    assert shares[6] > 0.8 * shares[4]


def test_exact_statistics_are_matched_exactly():
    schema = DatasetSchema(
        columns=[
            ColumnSchema(name="score", kind=ColumnKind.FLOAT, mean=600, std=50, exact=True),
            ColumnSchema(
                name="plan",
                kind=ColumnKind.CATEGORY,
                choices=["free", "pro"],
                weights=[7, 3],
                exact=True,
            ),
        ]
    )

    frame = _generate(schema, rows=1000)

    assert frame["score"].mean() == pytest.approx(600, abs=1e-6)
    assert frame["score"].std(ddof=0) == pytest.approx(50, abs=1e-6)
    assert (frame["plan"] == "free").sum() == 700


def test_quantiles_shape_the_distribution():
    schema = DatasetSchema(
        columns=[
            ColumnSchema(
                name="income",
                kind=ColumnKind.FLOAT,
                quantiles={0: 1000, 0.5: 3000, 0.9: 9000, 1: 50000},
            )
        ]
    )

    income = _generate(schema)["income"]

    assert income.median() == pytest.approx(3000, rel=0.05)
    assert income.quantile(0.9) == pytest.approx(9000, rel=0.05)
    assert income.between(1000, 50000).all()


def test_rank_correlations_are_imposed_across_kinds():
    schema = DatasetSchema(
        columns=[
            ColumnSchema(name="age", kind=ColumnKind.INTEGER, mean=40, std=12, minimum=18),
            ColumnSchema(
                name="income", kind=ColumnKind.FLOAT, distribution="lognormal", mean=4000, std=2000
            ),
            ColumnSchema(name="tier", kind=ColumnKind.CATEGORY, choices=["low", "mid", "high"]),
        ],
        correlations=[
            Correlation(columns=("age", "income"), value=0.6),
            Correlation(columns=("income", "tier"), value=-0.4),
        ],
    )

    frame = _generate(schema)
    tier_rank = frame["tier"].map({"low": 0, "mid": 1, "high": 2})

    assert frame["age"].corr(frame["income"], method="spearman") == pytest.approx(0.6, abs=0.05)
    assert frame["income"].corr(tier_rank, method="spearman") == pytest.approx(-0.4, abs=0.07)


def test_conditions_change_a_segment_and_keep_its_correlation():
    schema = DatasetSchema(
        columns=[
            ColumnSchema(name="plan", kind=ColumnKind.CATEGORY, choices=["free", "pro"]),
            ColumnSchema(name="age", kind=ColumnKind.FLOAT, mean=40, std=10),
            ColumnSchema(
                name="spend",
                kind=ColumnKind.FLOAT,
                mean=100,
                std=20,
                conditions=[
                    ColumnOverride(when="plan == 'pro'", mean=300, std=40, null_fraction=0.5)
                ],
            ),
        ],
        correlations=[Correlation(columns=("age", "spend"), value=0.7)],
    )

    frame = _generate(schema)
    pro = frame[frame["plan"] == "pro"]
    free = frame[frame["plan"] == "free"]

    assert pro["spend"].mean() == pytest.approx(300, rel=0.05)
    assert free["spend"].mean() == pytest.approx(100, rel=0.05)
    assert pro["spend"].isna().mean() == pytest.approx(0.5, abs=0.05)
    assert free["spend"].isna().sum() == 0
    assert pro["age"].corr(pro["spend"], method="spearman") > 0.6


def test_formulas_plant_trends_and_seasonality():
    schema = DatasetSchema(
        columns=[
            ColumnSchema(
                name="day", kind=ColumnKind.DATETIME, start="2024-01-01", end="2025-12-31"
            ),
            ColumnSchema(
                name="sales",
                kind=ColumnKind.FLOAT,
                formula="100 + 0.05 * (days(day) - days('2024-01-01'))"
                " + 20 * sin(2 * pi * dayofyear(day) / 365)",
                noise_std=2,
            ),
        ]
    )

    frame = _generate(schema)
    elapsed = (frame["day"] - pd.Timestamp("2024-01-01")).dt.days
    seasonal = np.sin(2 * np.pi * frame["day"].dt.day_of_year / 365)
    design = np.column_stack([np.ones(len(frame)), elapsed, seasonal])
    coefficients, *_ = np.linalg.lstsq(design, frame["sales"], rcond=None)

    assert coefficients == pytest.approx([100, 0.05, 20], rel=0.05)


def test_constraints_hold_on_every_row_and_unique_keys_stay_unique():
    schema = DatasetSchema(
        columns=[
            ColumnSchema(name="id", kind=ColumnKind.INTEGER, minimum=1, unique=True),
            ColumnSchema(name="start", kind=ColumnKind.FLOAT, minimum=0, maximum=100),
            ColumnSchema(name="end", kind=ColumnKind.FLOAT, minimum=0, maximum=100),
        ],
        constraints=["end >= start + 10"],
    )

    frame = _generate(schema, rows=500)

    assert len(frame) == 500
    assert (frame["end"] >= frame["start"] + 10).all()
    assert frame["id"].is_unique


def test_outliers_are_planted_far_from_the_mean():
    schema = DatasetSchema(
        columns=[
            ColumnSchema(
                name="amount",
                kind=ColumnKind.FLOAT,
                mean=100,
                std=10,
                outlier_fraction=0.01,
                outlier_scale=8,
            )
        ]
    )

    amount = _generate(schema, rows=2000)["amount"]

    far = ((amount - 100).abs() > 60).sum()
    assert far == pytest.approx(20, abs=2)


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        (
            {
                "columns": [
                    {"name": "a", "kind": "float", "formula": "b + 1"},
                    {"name": "b", "kind": "float", "formula": "a + 1"},
                ]
            },
            "cycle",
        ),
        ({"columns": [{"name": "a", "kind": "float", "formula": "missing * 2"}]}, "unknown"),
        (
            {
                "columns": [{"name": "a", "kind": "text", "category": "email"}],
                "correlations": [{"columns": ["a", "a"], "value": 0.5}],
            },
            "correlation column",
        ),
    ],
)
def test_inconsistent_pattern_schemas_are_rejected(schema, message):
    with pytest.raises(ValueError, match=message):
        DatasetSchema.model_validate(schema)


@pytest.mark.parametrize(
    "expression",
    [
        "a + 9 ** 9 ** 9",
        "10 ** 10 ** 8",
        "'a' * 1000000000",
        "[1, 2] * 1000000000",
        "('9' * 4000) ** 5000",
        "'%1000000000d' % 1",
        "-" * 1500 + "1",
        "+".join(["1"] * 3000),
        "1 // 0",
        "1 / 0",
        "1 % 0",
        "sin(1, 2, 3, 4)",
        "round(a, 10 ** 12)",
        "normal(0, -1)",
        "clip(a, 'x', 'y')",
        "days('notadate')",
    ],
)
def test_hostile_expressions_fail_with_an_expression_error(expression):
    frame = pd.DataFrame({"a": [1.0, 2.0]})

    with pytest.raises(ExpressionError):
        evaluate(expression, frame, np.random.default_rng(0))


def test_deeply_nested_schema_expressions_are_rejected_when_the_schema_loads():
    deep = "-" * 150 + "a"

    with pytest.raises(ValueError, match="nested"):
        ColumnSchema(name="b", kind=ColumnKind.FLOAT, formula=deep)


def test_ordinary_arithmetic_still_works_in_the_sandbox():
    frame = pd.DataFrame({"a": [1.0, 2.0], "n": [2, 3]})

    assert evaluate("2 ** 10 + a ** 2", frame).tolist() == [1025.0, 1028.0]
    assert evaluate("n ** 2", frame).tolist() == [4, 9]
    assert evaluate("round(a, 1) * 3 % 2", frame).tolist() == [1.0, 0.0]
