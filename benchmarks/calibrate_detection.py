"""Fit the detector's confidence calibration on the labelled benchmark and write it to the code.

Run `uv run python -m benchmarks.calibrate_detection` after changing a detection rule. The
script scans the benchmark without calibration and, separately for findings of personal data
and findings of "nothing sensitive", fits an isotonic regression of "the finding was right"
on the raw confidence. It rewrites the knots in `src/oblitus/detection/calibration.py`.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmarks.detection_benchmark import LabelledColumn, build_benchmark
from oblitus.detection import scan_column
from oblitus.detection.calibration import REVIEW_CONFIDENCE
from oblitus.taxonomy import Sensitivity

CALIBRATION_FILE = (
    Path(__file__).resolve().parent.parent / "src" / "oblitus" / "detection" / "calibration.py"
)
TRAINING_SEED = 0
_KNOT_BLOCK = re.compile(r"# BEGIN GENERATED KNOTS\n.*?# END GENERATED KNOTS", flags=re.DOTALL)

Knots = tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class ScoredFindings:
    """The raw confidence of each finding, whether it was right, and whether it found PII."""

    scores: np.ndarray
    hits: np.ndarray
    found_personal_data: np.ndarray

    def select(self, mask: np.ndarray) -> "ScoredFindings":
        return ScoredFindings(self.scores[mask], self.hits[mask], self.found_personal_data[mask])


def scored_findings(columns: list[LabelledColumn]) -> ScoredFindings:
    """Scan each column without calibration and record how each finding fared."""
    scores = []
    hits = []
    found_personal_data = []
    for column in columns:
        finding = scan_column(column.name, column.values, calibrated=False)
        scores.append(finding.confidence)
        hits.append(float(finding.category is column.truth))
        found_personal_data.append(finding.sensitivity is not Sensitivity.NON_SENSITIVE)
    return ScoredFindings(np.array(scores), np.array(hits), np.array(found_personal_data))


def isotonic_knots(scores: np.ndarray, hits: np.ndarray) -> Knots:
    """Fit a non-decreasing curve of hit rate against score by pooling adjacent violators.

    Each pooled block becomes one knot at the block's mean score. The curve is anchored at
    the review line and at 1, so every score above the line maps to a probability.
    """
    order = np.argsort(scores, kind="stable")
    blocks = [
        [float(score), float(hit), 1] for score, hit in zip(scores[order], hits[order], strict=True)
    ]
    # Each block holds [mean score, mean hit, size]; merge neighbours that break the order.
    merged: list[list[float]] = []
    for block in blocks:
        merged.append(block)
        while len(merged) > 1 and merged[-2][1] > merged[-1][1]:
            last = merged.pop()
            previous = merged.pop()
            size = previous[2] + last[2]
            merged.append(
                [
                    (previous[0] * previous[2] + last[0] * last[2]) / size,
                    (previous[1] * previous[2] + last[1] * last[2]) / size,
                    size,
                ]
            )
    knots = _distinct_knots([(round(block[0], 3), round(block[1], 3)) for block in merged])
    return _anchored(knots)


def _distinct_knots(knots: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Keep the last knot for each score, so the curve is a function of the score."""
    by_score = dict(knots)
    return sorted(by_score.items())


def _anchored(knots: list[tuple[float, float]]) -> Knots:
    anchored = list(knots)
    if anchored[0][0] > REVIEW_CONFIDENCE:
        anchored.insert(0, (REVIEW_CONFIDENCE, anchored[0][1]))
    if anchored[-1][0] < 1.0:
        anchored.append((1.0, anchored[-1][1]))
    return tuple(anchored)


def render_knots(personal_data_knots: Knots, clean_knots: Knots) -> str:
    return (
        "# BEGIN GENERATED KNOTS\n"
        f"{_render_tuple('PERSONAL_DATA_KNOTS', personal_data_knots)}\n"
        f"{_render_tuple('NO_PERSONAL_DATA_KNOTS', clean_knots)}\n"
        "# END GENERATED KNOTS"
    )


def _render_tuple(name: str, knots: Knots) -> str:
    rows = "".join(f"    ({raw}, {calibrated}),\n" for raw, calibrated in knots)
    return f"{name}: tuple[tuple[float, float], ...] = (\n{rows})"


def main() -> None:
    findings = scored_findings(build_benchmark(TRAINING_SEED))
    # Findings below the review line keep their tier values and are not fitted.
    above_line = findings.select(findings.scores >= REVIEW_CONFIDENCE)
    personal = above_line.select(above_line.found_personal_data)
    clean = above_line.select(~above_line.found_personal_data)
    block = render_knots(
        isotonic_knots(personal.scores, personal.hits), isotonic_knots(clean.scores, clean.hits)
    )
    source = CALIBRATION_FILE.read_text(encoding="utf-8")
    CALIBRATION_FILE.write_text(_KNOT_BLOCK.sub(lambda _match: block, source), encoding="utf-8")
    print(f"fitted {len(personal.scores)} personal-data and {len(clean.scores)} clean findings")


if __name__ == "__main__":
    main()
