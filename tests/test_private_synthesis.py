import json

import numpy as np
import pandas as pd
import pytest

from oblitus import Anonymizer, PrivacyBudget, PrivacyBudgetError, SchemaError
from oblitus.cli import main
from oblitus.differential_privacy import CategoricalDomain, NumericDomain, synthesize_mst

DOMAINS = {
    "age": {"kind": "numeric", "lower": 0, "upper": 100, "bins": 10, "integer": True},
    "plan": ["free", "pro", "team"],
    "region": ["north", "south"],
}
LARGE_BUDGET = PrivacyBudget.from_epsilon_delta(5000.0, 1e-6)
SMALL_BUDGET = PrivacyBudget.from_epsilon_delta(1.0, 1e-6)


@pytest.fixture
def customers() -> pd.DataFrame:
    """A chain of dependence: age drives the plan, and the plan drives the region."""
    rng = np.random.default_rng(11)
    rows = 6000
    age = rng.integers(0, 101, rows)
    plan = np.where(age < 35, "free", np.where(age < 70, "pro", "team"))
    flip = rng.random(rows) < 0.1
    plan = np.where(flip, rng.choice(["free", "pro", "team"], rows), plan)
    region = np.where((plan == "team") ^ (rng.random(rows) < 0.1), "north", "south")
    return pd.DataFrame({"age": age, "plan": plan, "region": region})


def _joint(frame: pd.DataFrame, first: str, second: str) -> pd.Series:
    return frame.groupby([first, second]).size().div(len(frame))


def _total_variation(first: pd.Series, second: pd.Series) -> float:
    return float(first.sub(second, fill_value=0).abs().sum() / 2)


def _binned_age(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(age=(frame["age"] // 10).clip(upper=9))


def test_with_a_large_budget_the_measured_pairs_are_close_to_the_original(customers):
    result = Anonymizer(seed=1).synthesize_private(customers, DOMAINS, LARGE_BUDGET, rows=6000)

    original = _binned_age(customers)
    synthetic = _binned_age(result.data)
    for first, second in result.tree_edges:
        original_joint = _joint(original, first, second)
        synthetic_joint = _joint(synthetic, first, second)
        assert _total_variation(original_joint, synthetic_joint) < 0.05, (first, second)


def test_the_tree_keeps_the_strong_dependencies_with_a_large_budget(customers):
    result = Anonymizer(seed=2).synthesize_private(customers, DOMAINS, LARGE_BUDGET)

    chosen = {frozenset(edge) for edge in result.tree_edges}
    assert frozenset({"age", "plan"}) in chosen
    assert len(chosen) == 2


def test_the_tree_spans_every_column(customers):
    result = Anonymizer(seed=3).synthesize_private(customers, DOMAINS, SMALL_BUDGET)

    touched = {column for edge in result.tree_edges for column in edge}
    assert touched == set(DOMAINS)
    assert len(result.tree_edges) == len(DOMAINS) - 1


def test_the_whole_declared_budget_is_spent_exactly_and_itemized(customers):
    report = Anonymizer(seed=4).synthesize_private(customers, DOMAINS, SMALL_BUDGET).budget

    columns = len(DOMAINS)
    assert len(report.spends) == columns + 2 * (columns - 1)
    assert report.spent_rho == pytest.approx(SMALL_BUDGET.rho, rel=1e-9)
    assert report.spent_epsilon == pytest.approx(1.0, rel=1e-6)
    assert report.spent_epsilon <= 1.0
    assert report.delta == 1e-6


def test_the_three_phases_each_cost_a_third_of_the_budget(customers):
    budget = PrivacyBudget.from_rho(0.3)

    report = Anonymizer(seed=5).synthesize_private(customers, DOMAINS, budget).budget

    by_phase = {"one-way": 0.0, "choose": 0.0, "two-way": 0.0}
    for spend in report.spends:
        phase = next(name for name in by_phase if spend.label.startswith(name))
        by_phase[phase] += spend.rho
    assert list(by_phase.values()) == pytest.approx([0.1, 0.1, 0.1])


class _NoiseRecorder:
    """A random generator that remembers the standard deviation of every normal draw."""

    def __init__(self, seed: int) -> None:
        self._generator = np.random.default_rng(seed)
        self.scales: list[float] = []

    def normal(self, loc, scale, size):
        self.scales.append(scale)
        return self._generator.normal(loc, scale, size)

    def __getattr__(self, name):
        return getattr(self._generator, name)


def test_the_gaussian_noise_scale_matches_the_rho_the_report_charges(customers):
    budget = PrivacyBudget.from_rho(0.3)
    recorder = _NoiseRecorder(seed=14)

    synthesize_mst(customers, DOMAINS, budget, recorder)  # type: ignore[arg-type]

    columns = len(DOMAINS)
    one_way_scale = (columns / (2 * 0.1)) ** 0.5
    two_way_scale = ((columns - 1) / (2 * 0.1)) ** 0.5
    expected = [one_way_scale] * columns + [two_way_scale] * (columns - 1)
    assert recorder.scales == pytest.approx(expected)


def test_a_single_column_spends_everything_on_its_counts(customers):
    budget = PrivacyBudget.from_rho(0.2)

    result = Anonymizer(seed=6).synthesize_private(customers, {"plan": DOMAINS["plan"]}, budget)

    assert result.tree_edges == ()
    assert result.budget.spent_rho == pytest.approx(0.2)
    assert set(result.data["plan"]) <= {"free", "pro", "team"}


def test_a_pure_epsilon_budget_is_refused_because_mst_needs_gaussian_noise(customers):
    with pytest.raises(PrivacyBudgetError, match="delta"):
        Anonymizer(seed=7).synthesize_private(customers, DOMAINS, PrivacyBudget.pure(1.0))


def test_a_value_outside_a_declared_domain_is_refused_not_dropped(customers):
    odd = customers.assign(plan=customers["plan"].replace({"team": "enterprise"}))

    with pytest.raises(SchemaError, match="outside its declared domain"):
        Anonymizer(seed=8).synthesize_private(odd, DOMAINS, SMALL_BUDGET)


@pytest.mark.parametrize("method", ["mst", "aim"])
def test_a_text_value_in_a_numeric_column_is_refused_without_naming_it(customers, method):
    if method == "aim":
        pytest.importorskip("mbi")
    odd = customers.assign(age=customers["age"].astype(object))
    odd.loc[3, "age"] = "CONFIDENTIAL_DIAGNOSIS"

    with pytest.raises(SchemaError, match="not numbers") as refusal:
        Anonymizer(seed=9).synthesize_private(odd, DOMAINS, SMALL_BUDGET, method=method)

    assert "CONFIDENTIAL" not in str(refusal.value)
    assert refusal.value.__cause__ is None
    assert refusal.value.__suppress_context__


def test_missing_values_are_refused(customers):
    gappy = customers.assign(age=customers["age"].astype(float).mask(customers.index == 0))

    with pytest.raises(SchemaError, match="missing values"):
        Anonymizer(seed=9).synthesize_private(gappy, DOMAINS, SMALL_BUDGET)


def test_a_domain_for_a_column_the_table_lacks_is_refused(customers):
    with pytest.raises(SchemaError, match="not in the table"):
        Anonymizer(seed=10).synthesize_private(customers, {"income": ["low"]}, SMALL_BUDGET)


def test_synthetic_numbers_stay_inside_the_declared_bounds_as_integers(customers):
    budget = PrivacyBudget.from_epsilon_delta(2.0, 1e-6)

    synthetic = Anonymizer(seed=12).synthesize_private(customers, DOMAINS, budget, rows=3000).data

    assert len(synthetic) == 3000
    assert synthetic["age"].between(0, 100).all()
    assert pd.api.types.is_integer_dtype(synthetic["age"])


def test_numbers_outside_the_declared_bounds_are_clipped_to_them():
    domain = NumericDomain(lower=0, upper=10, bins=5)

    codes = domain.encode("x", pd.Series([-50.0, 0.0, 9.99, 10.0, 500.0]))

    assert codes.tolist() == [0, 0, 4, 4, 4]


def test_a_domain_must_have_ordered_finite_bounds_and_unique_values():
    with pytest.raises(ValueError):
        NumericDomain(lower=5, upper=1)
    with pytest.raises(ValueError, match="finite"):
        NumericDomain(lower=-1e308, upper=1e308)
    with pytest.raises(ValueError):
        CategoricalDomain(values=["a", "a"])


def test_the_same_seed_gives_the_same_table_and_the_input_is_not_changed(customers):
    before = customers.copy()

    first = Anonymizer(seed=13).synthesize_private(customers, DOMAINS, SMALL_BUDGET).data
    second = Anonymizer(seed=13).synthesize_private(customers, DOMAINS, SMALL_BUDGET).data

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(customers, before)


def test_the_cli_writes_a_synthetic_table_and_prints_the_budget(customers, tmp_path, capsys):
    source = tmp_path / "customers.csv"
    customers.to_csv(source, index=False)
    domains = tmp_path / "domains.json"
    domains.write_text(json.dumps(DOMAINS))
    output = tmp_path / "private.csv"

    exit_code = main(
        ["--seed", "1", "dp-synthesize", str(source), "--domains", str(domains)]
        + ["-o", str(output), "--epsilon", "2", "--delta", "1e-6", "--rows", "500"]
    )

    assert exit_code == 0
    assert len(pd.read_csv(output)) == 500
    assert "privacy spent: epsilon 2" in capsys.readouterr().out


def test_the_cli_refuses_a_run_without_a_declared_budget(customers, tmp_path, capsys):
    source = tmp_path / "customers.csv"
    customers.to_csv(source, index=False)
    domains = tmp_path / "domains.json"
    domains.write_text(json.dumps(DOMAINS))
    output = tmp_path / "o.csv"

    exit_code = main(["dp-synthesize", str(source), "--domains", str(domains), "-o", str(output)])

    assert exit_code == 2
    assert "privacy budget" in capsys.readouterr().err
