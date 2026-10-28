import json
import sys

import numpy as np
import pandas as pd
import pytest

from oblitus import Anonymizer, PrivacyBudget, PrivacyBudgetError, SchemaError, Statistic
from oblitus.cli import main

pytest.importorskip("opendp")

GROUPS = {"city": ["north", "south", "east"]}
STATISTICS = [
    Statistic(name="customers", kind="count"),
    Statistic(name="spend_total", kind="sum", column="spend", lower=0, upper=100),
    Statistic(name="spend_mean", kind="mean", column="spend", lower=0, upper=100),
]


@pytest.fixture
def orders() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = 3000
    return pd.DataFrame(
        {"city": rng.choice(["north", "south"], rows), "spend": rng.uniform(0, 100, rows)}
    )


def test_with_a_huge_budget_the_release_matches_the_exact_statistics(orders):
    result = Anonymizer().aggregate_private(orders, GROUPS, STATISTICS, PrivacyBudget.pure(1e7))

    table = result.table.set_index("city")
    exact = orders.groupby("city")["spend"].agg(["count", "sum", "mean"])
    assert table.loc["north", "customers"] == pytest.approx(exact.loc["north", "count"], abs=0.5)
    assert table.loc["south", "spend_total"] == pytest.approx(exact.loc["south", "sum"], abs=1)
    assert table.loc["south", "spend_mean"] == pytest.approx(exact.loc["south", "mean"], abs=0.01)


def test_every_declared_group_is_released_even_when_it_is_empty(orders):
    result = Anonymizer().aggregate_private(orders, GROUPS, STATISTICS, PrivacyBudget.pure(1.0))

    assert set(result.table["city"]) == {"north", "south", "east"}
    assert result.accuracy.shape == result.table.shape


def test_a_pure_budget_is_split_over_queries_and_spent_exactly(orders):
    result = Anonymizer().aggregate_private(orders, GROUPS, STATISTICS, PrivacyBudget.pure(2.0))

    shares = {spend.label: spend.pure_epsilon for spend in result.budget.spends}
    assert shares == pytest.approx({"customers": 0.5, "spend_total": 0.5, "spend_mean": 1.0})
    assert result.budget.spent_epsilon == pytest.approx(2.0)


def test_a_zcdp_budget_uses_gaussian_noise_and_spends_all_of_rho(orders):
    budget = PrivacyBudget.from_epsilon_delta(1.0, 1e-6)

    result = Anonymizer().aggregate_private(orders, GROUPS, STATISTICS, budget)

    assert result.budget.spent_rho == pytest.approx(budget.rho, rel=1e-9)
    assert result.budget.spent_epsilon == pytest.approx(1.0, rel=1e-6)


def test_the_reported_half_width_covers_the_exact_count_at_the_stated_confidence(orders):
    orders = orders.head(100)
    groups = {"city": ["north", "south"]}
    exact_north = int((orders["city"] == "north").sum())
    budget = PrivacyBudget.pure(0.5)
    count_only = [Statistic(name="n", kind="count")]
    covered = 0
    trials = 120

    for _ in range(trials):
        result = Anonymizer().aggregate_private(orders, groups, count_only, budget)
        row = result.table.index[result.table["city"] == "north"][0]
        error = abs(result.table.loc[row, "n"] - exact_north)
        covered += error <= result.accuracy.loc[row, "n"]

    assert covered / trials >= 0.86


def test_the_half_width_of_a_sum_grows_with_its_declared_bound(orders):
    wide = [Statistic(name="s", kind="sum", column="spend", lower=0, upper=1000)]
    narrow = [Statistic(name="s", kind="sum", column="spend", lower=0, upper=10)]
    budget = PrivacyBudget.from_rho(0.5)

    wide_result = Anonymizer().aggregate_private(orders, GROUPS, wide, budget)
    narrow_result = Anonymizer().aggregate_private(orders, GROUPS, narrow, budget)

    ratio = wide_result.accuracy["s"].iloc[0] / narrow_result.accuracy["s"].iloc[0]
    assert ratio == pytest.approx(100, rel=1e-6)


def test_a_sum_without_declared_bounds_is_refused():
    with pytest.raises(ValueError, match="never taken from the data"):
        Statistic(name="s", kind="sum", column="spend")


def test_text_in_a_summed_column_is_refused_without_naming_it(orders):
    odd = orders.assign(spend=orders["spend"].astype(object))
    odd.loc[7, "spend"] = "CONFIDENTIAL_DIAGNOSIS"
    budget = PrivacyBudget.from_epsilon_delta(1.0, 1e-6)

    with pytest.raises(SchemaError, match="not numbers") as refusal:
        Anonymizer(seed=1).aggregate_private(odd, GROUPS, STATISTICS, budget)

    assert "CONFIDENTIAL" not in str(refusal.value)
    assert refusal.value.__suppress_context__


def test_a_group_value_outside_the_declared_list_is_refused(orders):
    with pytest.raises(SchemaError, match="outside its declared domain"):
        Anonymizer().aggregate_private(
            orders, {"city": ["north"]}, STATISTICS, PrivacyBudget.pure(1.0)
        )


def test_statistic_names_must_be_unique(orders):
    twice = [Statistic(name="n", kind="count"), Statistic(name="n", kind="count")]

    with pytest.raises(SchemaError, match="unique"):
        Anonymizer().aggregate_private(orders, GROUPS, twice, PrivacyBudget.pure(1.0))


def test_values_are_clamped_to_the_declared_bounds_so_one_outlier_cannot_dominate(orders):
    huge = orders.assign(spend=orders["spend"].mask(orders.index == 0, 1e9))
    budget = PrivacyBudget.pure(1e7)
    total_only = [Statistic(name="s", kind="sum", column="spend", lower=0, upper=100)]

    result = Anonymizer().aggregate_private(huge, GROUPS, total_only, budget)

    assert result.table["s"].sum() < huge["spend"].clip(upper=100).sum() + 5


def test_without_the_opendp_extra_the_error_names_it(orders, monkeypatch):
    monkeypatch.setitem(sys.modules, "opendp.prelude", None)

    with pytest.raises(ImportError, match=r"oblitus\[opendp\]"):
        Anonymizer().aggregate_private(orders, GROUPS, STATISTICS, PrivacyBudget.pure(1.0))


def test_a_budget_that_is_too_small_to_declare_is_refused(orders):
    with pytest.raises(PrivacyBudgetError):
        Anonymizer().aggregate_private(orders, GROUPS, STATISTICS, PrivacyBudget.pure(0.0))


def test_the_cli_writes_the_noisy_table_and_its_accuracy(orders, tmp_path, capsys):
    source = tmp_path / "orders.csv"
    orders.to_csv(source, index=False)
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "group_by": GROUPS,
                "statistics": [statistic.model_dump(exclude_none=True) for statistic in STATISTICS],
            }
        )
    )
    output = tmp_path / "noisy.csv"
    accuracy = tmp_path / "accuracy.csv"

    exit_code = main(
        ["dp-aggregate", str(source), "--spec", str(spec), "-o", str(output)]
        + ["--epsilon", "1", "--accuracy-output", str(accuracy)]
    )

    assert exit_code == 0
    assert list(pd.read_csv(output).columns) == ["city", "customers", "spend_total", "spend_mean"]
    assert len(pd.read_csv(accuracy)) == 3
    assert "privacy spent: epsilon 1 of 1" in capsys.readouterr().out
