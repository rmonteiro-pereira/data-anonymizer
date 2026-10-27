import numpy as np
import pytest

from benchmarks.detection_benchmark import LabelledColumn, build_benchmark
from oblitus.detection import scan_column
from oblitus.detection.calibration import (
    NO_PERSONAL_DATA_KNOTS,
    PERSONAL_DATA_KNOTS,
    REVIEW_CONFIDENCE,
    calibrated_confidence,
)

BINS = 10
TRAINING_ERROR_BOUND = 0.05
HELD_OUT_ERROR_BOUND = 0.12


def _calibrated_scores_and_hits(columns: list[LabelledColumn]) -> tuple[np.ndarray, np.ndarray]:
    findings = [scan_column(column.name, column.values) for column in columns]
    scores = np.array([finding.confidence for finding in findings])
    hits = np.array(
        [
            float(finding.category is column.truth)
            for finding, column in zip(findings, columns, strict=True)
        ]
    )
    # Below the review line a finding is a guess for a person to check and keeps its tier value.
    reviewed = scores >= REVIEW_CONFIDENCE
    return scores[reviewed], hits[reviewed]


def expected_calibration_error(scores: np.ndarray, hits: np.ndarray) -> float:
    """The average gap between a confidence and how often it is right, weighted by bin size."""
    edges = np.linspace(0.0, 1.0, BINS + 1)
    bin_of = np.minimum(np.digitize(scores, edges[1:-1]), BINS - 1)
    error = 0.0
    for bin_index in range(BINS):
        in_bin = bin_of == bin_index
        if in_bin.any():
            gap = abs(scores[in_bin].mean() - hits[in_bin].mean())
            error += gap * in_bin.sum() / len(scores)
    return float(error)


def test_the_benchmark_is_deterministic():
    first = build_benchmark(seed=3)
    second = build_benchmark(seed=3)

    assert [column.name for column in first] == [column.name for column in second]
    assert all(a.values.equals(b.values) for a, b in zip(first, second, strict=True))


def test_calibrated_confidence_matches_how_often_findings_are_right():
    scores, hits = _calibrated_scores_and_hits(build_benchmark(seed=0))

    assert expected_calibration_error(scores, hits) < TRAINING_ERROR_BOUND


def test_calibration_holds_on_columns_from_another_seed():
    scores, hits = _calibrated_scores_and_hits(build_benchmark(seed=1))

    assert expected_calibration_error(scores, hits) < HELD_OUT_ERROR_BOUND


def test_the_high_confidence_findings_are_mostly_right():
    scores, hits = _calibrated_scores_and_hits(build_benchmark(seed=1))
    confident = scores >= 0.9

    assert confident.any()
    assert hits[confident].mean() >= 0.8


@pytest.mark.parametrize("knots", [PERSONAL_DATA_KNOTS, NO_PERSONAL_DATA_KNOTS])
def test_the_curves_never_rank_a_higher_score_lower(knots):
    raw = [knot[0] for knot in knots]
    calibrated = [knot[1] for knot in knots]

    assert raw == sorted(raw)
    assert calibrated == sorted(calibrated)
    assert calibrated[0] >= REVIEW_CONFIDENCE
    assert calibrated[-1] <= 1.0


def test_guesses_below_the_review_line_keep_their_tier_value():
    assert calibrated_confidence(0.55, found_personal_data=True) == 0.55
    assert calibrated_confidence(0.4, found_personal_data=False) == 0.4
    assert calibrated_confidence(0.6, found_personal_data=True) >= REVIEW_CONFIDENCE


def test_the_raw_score_is_available():
    column = build_benchmark(seed=0)[0]

    raw = scan_column(column.name, column.values, calibrated=False)
    calibrated = scan_column(column.name, column.values)

    assert raw.category is calibrated.category
    expected = calibrated_confidence(raw.confidence, found_personal_data=True)
    assert calibrated.confidence == pytest.approx(expected, abs=1e-3)
