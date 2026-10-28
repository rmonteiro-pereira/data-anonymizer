import sys

import numpy as np
import pandas as pd
import pytest

from oblitus import Anonymizer, PrivacyBudget, PrivacyBudgetError, SchemaError
from oblitus.cli import main
from oblitus.differential_privacy import synthesize_aim
from oblitus.errors import MissingExtraError

pytest.importorskip("mbi")

BINARY = [0, 1]
DOMAINS = {"first": BINARY, "second": BINARY, "parity": BINARY, "noise": ["x", "y", "z"]}
LARGE_BUDGET = PrivacyBudget.from_epsilon_delta(200.0, 1e-6)
SMALL_BUDGET = PrivacyBudget.from_epsilon_delta(1.0, 1e-6)
PARITY_GROUP = [("first", "second", "parity")]


@pytest.fixture
def parity_table() -> pd.DataFrame:
    """`parity` is the XOR of two coin flips: every pair looks independent, the triple does not."""
    rng = np.random.default_rng(21)
    rows = 4000
    first = rng.integers(0, 2, rows)
    second = rng.integers(0, 2, rows)
    flip = rng.random(rows) < 0.05
    parity = np.where(flip, 1 - (first ^ second), first ^ second)
    noise = rng.choice(["x", "y", "z"], rows)
    return pd.DataFrame({"first": first, "second": second, "parity": parity, "noise": noise})


def _parity_holds(frame: pd.DataFrame) -> float:
    return float(((frame["first"] ^ frame["second"]) == frame["parity"]).mean())


def test_aim_keeps_a_three_column_relation_that_no_pair_shows(parity_table):
    anonymizer = Anonymizer(seed=1)

    aim = anonymizer.synthesize_private(
        parity_table, DOMAINS, LARGE_BUDGET, method="aim", workload=PARITY_GROUP
    )
    mst = anonymizer.synthesize_private(parity_table, DOMAINS, LARGE_BUDGET)

    assert _parity_holds(parity_table) > 0.9
    assert _parity_holds(aim.data) > 0.85
    assert _parity_holds(mst.data) < 0.6
    assert ("first", "second", "parity") in aim.measured_marginals


def test_aim_spends_exactly_the_declared_budget(parity_table):
    budget = PrivacyBudget.from_rho(0.4)

    report = (
        Anonymizer(seed=2).synthesize_private(parity_table, DOMAINS, budget, method="aim").budget
    )

    assert report.spent_rho == pytest.approx(0.4)
    assert report.spent_rho <= 0.4 * (1 + 1e-9)
    assert any(spend.label.startswith("choose group") for spend in report.spends)


class _NoiseRecorder:
    """A generator that records the scale of every Gaussian draw."""

    def __init__(self, seed: int) -> None:
        self._generator = np.random.default_rng(seed)
        self.scales: list[float] = []

    def normal(self, loc, scale, size):
        self.scales.append(scale)
        return self._generator.normal(loc, scale, size)

    def __getattr__(self, name):
        return getattr(self._generator, name)


def test_every_gaussian_draw_is_charged_at_its_own_scale(parity_table):
    recorder = _NoiseRecorder(seed=3)

    report = synthesize_aim(
        parity_table,
        DOMAINS,
        SMALL_BUDGET,
        recorder,  # type: ignore[arg-type]
    ).budget

    charged = [spend.rho for spend in report.spends if spend.label.startswith("counts of")]
    drawn = [1 / (2 * scale**2) for scale in recorder.scales]
    assert charged == pytest.approx(drawn)


def test_without_a_workload_aim_measures_pairs_of_columns(parity_table):
    result = Anonymizer(seed=4).synthesize_private(
        parity_table, DOMAINS, SMALL_BUDGET, method="aim"
    )

    joint = [group for group in result.measured_marginals if len(group) > 1]
    assert joint
    assert all(len(group) == 2 for group in joint)
    assert result.tree_edges == ()


def test_a_single_column_spends_everything_on_its_own_counts(parity_table):
    budget = PrivacyBudget.from_rho(0.2)

    result = Anonymizer(seed=5).synthesize_private(
        parity_table, {"noise": DOMAINS["noise"]}, budget, method="aim"
    )

    assert result.budget.spent_rho == pytest.approx(0.2)
    assert set(result.data["noise"]) <= {"x", "y", "z"}


def test_the_same_seed_gives_the_same_table(parity_table):
    first = Anonymizer(seed=6).synthesize_private(parity_table, DOMAINS, SMALL_BUDGET, method="aim")
    second = Anonymizer(seed=6).synthesize_private(
        parity_table, DOMAINS, SMALL_BUDGET, method="aim"
    )

    pd.testing.assert_frame_equal(first.data, second.data)


@pytest.mark.parametrize(
    ("workload", "message"),
    [([("first", "salary")], "unknown"), ([("first", "first")], "twice")],
)
def test_a_workload_must_name_released_columns_once(parity_table, workload, message):
    with pytest.raises(SchemaError, match=message):
        Anonymizer(seed=7).synthesize_private(
            parity_table, DOMAINS, SMALL_BUDGET, method="aim", workload=workload
        )


def test_mst_refuses_a_workload_instead_of_ignoring_it(parity_table):
    with pytest.raises(SchemaError, match="only AIM"):
        Anonymizer(seed=8).synthesize_private(
            parity_table, DOMAINS, SMALL_BUDGET, workload=PARITY_GROUP
        )


def test_a_pure_epsilon_budget_is_refused(parity_table):
    with pytest.raises(PrivacyBudgetError, match="delta"):
        Anonymizer(seed=9).synthesize_private(
            parity_table, DOMAINS, PrivacyBudget.pure(1.0), method="aim"
        )


def test_a_value_outside_a_declared_domain_is_refused(parity_table):
    odd = parity_table.assign(noise=parity_table["noise"].replace({"z": "w"}))

    with pytest.raises(SchemaError, match="outside its declared domain"):
        Anonymizer(seed=10).synthesize_private(odd, DOMAINS, SMALL_BUDGET, method="aim")


def test_without_the_extra_the_error_names_it(parity_table, monkeypatch):
    monkeypatch.setitem(sys.modules, "jax", None)

    with pytest.raises(MissingExtraError, match=r"oblitus\[aim\]"):
        Anonymizer(seed=11).synthesize_private(parity_table, DOMAINS, SMALL_BUDGET, method="aim")


def test_the_cli_runs_aim_with_a_workload(parity_table, tmp_path, capsys):
    source = tmp_path / "table.csv"
    parity_table.to_csv(source, index=False)
    domains = tmp_path / "domains.json"
    domains.write_text(pd.Series(DOMAINS).to_json())
    output = tmp_path / "private.csv"

    exit_code = main(
        ["--seed", "1", "dp-synthesize", str(source), "--domains", str(domains)]
        + ["-o", str(output), "--epsilon", "2", "--delta", "1e-6", "--rows", "300"]
        + ["--method", "aim", "--workload", "first,second,parity"]
    )

    printed = capsys.readouterr().out
    assert exit_code == 0
    assert len(pd.read_csv(output)) == 300
    assert "privacy spent: epsilon 2" in printed
    assert "column groups measured:" in printed
