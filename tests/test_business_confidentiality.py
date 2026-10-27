import numpy as np
import pandas as pd
import pytest

from data_anonymizer import (
    AnonymizationPlan,
    Anonymizer,
    ColumnAction,
    Fidelity,
    ProtectionLevel,
)
from data_anonymizer.techniques.pseudonymization import scramble_categories
from data_anonymizer.techniques.rescaling import normalize

SEGMENTS = ["Moda", "Alimentação", "Casa", "Lazer"]


@pytest.fixture
def mall() -> pd.DataFrame:
    """40 stores over 24 months, with one fashion anchor far bigger than the rest."""
    rng = np.random.default_rng(11)
    stores = pd.DataFrame(
        {
            "store_name": [f"Loja {index:02d}" for index in range(40)],
            "segmento": rng.choice(SEGMENTS, 40),
            "base_sales": rng.lognormal(11, 0.4, 40),
        }
    )
    stores.loc[0, ["store_name", "segmento", "base_sales"]] = ["Anchor Fashion", "Moda", 4e6]
    months = pd.date_range("2024-01-01", periods=24, freq="MS")
    seasonality = 1 + 0.3 * np.sin(2 * np.pi * (months.month - 3) / 12)
    rows = stores.merge(pd.DataFrame({"month": months, "season": seasonality}), how="cross")
    noise = rng.normal(1, 0.05, len(rows))
    rows["faturamento"] = (rows["base_sales"] * rows["season"] * noise).round(2)
    rows["aluguel"] = (rows["base_sales"] * 0.08 + 5_000).round(2)
    return rows.drop(columns=["base_sales", "season"])


def test_raw_revenue_next_to_its_segment_is_flagged(mall):
    plan = AnonymizationPlan(
        fidelity=Fidelity.STATISTICAL,
        actions=[
            ColumnAction(
                column="store_name", technique="scramble_categories", params={"prefix": "ORG"}
            )
        ],
        direct_identifiers=["store_name"],
        quasi_identifiers=["segmento"],
        sensitive_attributes=["faturamento"],
        entity_column="store_name",
    )

    protection = Anonymizer().apply(mall, plan).protection

    assert protection.level is ProtectionLevel.AT_RISK
    assert any("segmento = 'Moda'" in problem for problem in protection.problems)


def test_the_default_plan_hides_revenue_and_rent_but_keeps_the_pattern(mall):
    anonymizer = Anonymizer(seed=2)
    plan = anonymizer.propose(mall)

    result = anonymizer.apply(mall, plan)
    released = result.data

    assert plan.action_for("store_name").technique == "scramble_categories"
    assert plan.action_for("segmento").technique == "scramble_categories"
    assert plan.action_for("faturamento").technique == "normalize"
    assert plan.action_for("aluguel").technique == "normalize"
    assert not released["store_name"].isin(mall["store_name"]).any()
    assert not released["segmento"].isin(SEGMENTS).any()
    assert released["faturamento"].abs().max() < 1_000
    assert not any("original units" in problem for problem in result.protection.problems)
    assert any("two records" in caveat for caveat in result.protection.caveats)

    # The monthly pattern of every store survives: same rank order over time.
    original_monthly = mall.groupby("month")["faturamento"].sum()
    released_monthly = released.groupby("month")["faturamento"].sum()
    assert original_monthly.corr(released_monthly, method="spearman") > 0.95


def test_robust_normalization_keeps_the_distribution_shape(run, context):
    values = pd.Series(np.random.default_rng(0).lognormal(10, 1, 2000))

    rescaled = run(normalize, values, context)

    assert rescaled.skew() == pytest.approx(values.skew(), rel=1e-9)
    assert (rescaled.rank() == values.rank()).all()
    assert rescaled.median() == pytest.approx(0)


@pytest.mark.parametrize(
    "params",
    [{"method": "index"}, {"method": "rank"}, {"method": "zscore", "by": "store"}],
)
def test_shape_changing_normalization_needs_the_flag(run, context, params):
    with pytest.raises(ValueError, match="allow_shape_change"):
        run(normalize, pd.Series([1.0, 2.0, 3.0]), context, **params)


def test_index_by_store_removes_every_store_size(mall):
    plan = AnonymizationPlan(
        fidelity=Fidelity.STATISTICAL,
        actions=[
            ColumnAction(
                column="faturamento",
                technique="normalize",
                params={"method": "index", "by": "store_name", "allow_shape_change": True},
            )
        ],
        shuffle_rows=False,
    )

    released = Anonymizer().apply(mall, plan).data

    store_means = released.groupby(mall["store_name"])["faturamento"].mean()
    assert np.allclose(store_means, 100)


def test_growth_follows_each_store_over_time(mall):
    plan = AnonymizationPlan(
        fidelity=Fidelity.STATISTICAL,
        actions=[
            ColumnAction(
                column="faturamento",
                technique="normalize",
                params={
                    "method": "growth",
                    "by": "store_name",
                    "order_by": "month",
                    "allow_shape_change": True,
                },
            )
        ],
        shuffle_rows=False,
    )

    released = Anonymizer().apply(mall, plan).data

    first_months = released[mall["month"] == mall["month"].min()]["faturamento"]
    assert first_months.isna().all()
    assert released["faturamento"].abs().median() < 1


def test_scramble_categories_hides_order_and_merges_small_segments(run, context):
    labels = pd.Series(["Moda"] * 10 + ["Alimentação"] * 8 + ["Casa"] * 6 + ["Joalheria"])

    codes = run(scramble_categories, labels, context, prefix="SEG", min_count=3)

    # Joalheria alone has 1 row, so the smallest segment, Casa, joins it in OTHER.
    assert set(codes) == {"SEG-01", "SEG-02", "SEG-OTHER"}
    assert (codes[labels.isin(["Casa", "Joalheria"])] == "SEG-OTHER").all()
    assert codes.groupby(labels).nunique().max() == 1


def test_keyed_scramble_gives_the_same_codes_in_every_release(run, context):
    labels = pd.Series(["Moda", "Casa", "Lazer", "Moda"])

    first = run(scramble_categories, labels, context, keyed=True)
    second = run(scramble_categories, labels.iloc[::-1].reset_index(drop=True), context, keyed=True)

    assert dict(zip(labels, first, strict=True)) == dict(
        zip(labels.iloc[::-1], second, strict=True)
    )
