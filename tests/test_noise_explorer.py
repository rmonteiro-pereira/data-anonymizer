import json

import numpy as np
import pandas as pd
import pytest

from data_anonymizer import AnonymizationPlan, Anonymizer, ColumnAction, Fidelity, Protection
from data_anonymizer.cli import main
from data_anonymizer.evaluation.disclosure import deflection_metrics, linkage_rate
from data_anonymizer.noise_explorer import explore_noise, render_page


def test_linkage_rate_is_one_for_a_copy_and_near_chance_for_a_shuffle():
    values = np.random.default_rng(0).normal(50, 10, 2000)

    assert linkage_rate(values, values) == pytest.approx(1.0)
    shuffled = np.random.default_rng(1).permutation(values)
    assert linkage_rate(values, shuffled) < 0.01


def test_linkage_rate_shares_credit_among_tied_values():
    values = np.repeat(np.arange(100.0), 5)

    assert linkage_rate(values, values) == pytest.approx(0.2)


def test_more_noise_moves_records_further_and_links_fewer():
    values = np.random.default_rng(0).normal(50, 10, 3000)
    noise = np.random.default_rng(1).normal(0, 1, 3000)

    weak = deflection_metrics(values, values + 0.5 * noise)
    strong = deflection_metrics(values, values + 10 * noise)

    assert weak.median_deviation_sd < strong.median_deviation_sd
    assert weak.linkage_rate > strong.linkage_rate
    assert weak.interval_disclosure > strong.interval_disclosure


def test_explore_noise_sweeps_every_technique_from_weak_to_strong():
    frame = pd.DataFrame({"salary": np.random.default_rng(2).lognormal(8, 0.5, 1000)})

    exploration = explore_noise(frame, "salary", seed=3)
    metrics = exploration.metrics
    gaussian = metrics[metrics["technique"] == "gaussian_noise"]

    assert set(metrics["technique"]) == {
        "gaussian_noise",
        "multiplicative_noise",
        "laplace_noise",
        "rank_swap",
        "microaggregate",
    }
    assert gaussian["median_deviation_sd"].is_monotonic_increasing
    assert all(sum(sweep.original_histogram) == 1000 for sweep in exploration.sweeps)


def test_the_page_embeds_the_data_safely():
    frame = pd.DataFrame({"</script><b>x": np.arange(100.0)})

    page = render_page(explore_noise(frame, "</script><b>x"))

    assert "</script><b>" not in page
    payload = page.split('<script id="data" type="application/json">')[1].split("</script>")[0]
    assert json.loads(payload)["column"] == "</script><b>x"


def test_explore_noise_command_writes_a_page(tmp_path):
    source = tmp_path / "data.csv"
    pd.DataFrame({"amount": np.arange(200.0)}).to_csv(source, index=False)
    page = tmp_path / "noise.html"

    assert main(["explore-noise", str(source), "--column", "amount", "-o", str(page)]) == 0
    assert "Noise explorer" in page.read_text()
    assert main(["explore-noise", str(source), "--column", "missing", "-o", str(page)]) == 2


def test_weak_noise_fails_the_anonymization_linkage_check():
    frame = pd.DataFrame({"income": np.random.default_rng(0).lognormal(8, 0.6, 1000)})
    plan = AnonymizationPlan(
        fidelity=Fidelity.STATISTICAL,
        protection=Protection.ANONYMIZATION,
        actions=[
            ColumnAction(
                column="income", technique="gaussian_noise", params={"relative_scale": 0.001}
            )
        ],
    )

    protection = Anonymizer(seed=1).apply(frame, plan).protection

    assert not protection.reached
    assert any("lead back to their record" in problem for problem in protection.problems)


def test_rows_unique_across_discrete_columns_are_flagged():
    frame = pd.DataFrame(
        {
            "region": ["north"] * 10 + ["south"] * 10,
            "plan": ["free"] * 19 + ["vip"],
        }
    )
    plan = AnonymizationPlan(
        fidelity=Fidelity.PRIVACY,
        protection=Protection.ANONYMIZATION,
        actions=[],
        quasi_identifiers=["region"],
    )

    protection = Anonymizer().apply(frame, plan).protection

    assert any("1 rows are unique" in problem for problem in protection.problems)
