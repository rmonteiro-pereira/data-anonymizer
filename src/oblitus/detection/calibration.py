"""Turn the detector's raw confidence into the chance that the finding is right.

The raw score is a rule of thumb: the share of values that pass a check, plus a bonus when the
column name agrees. It is not a probability. Above the review line this module maps it to one,
so a confidence of 0.9 means that about nine in ten findings with that score were right on the
labelled benchmark in `benchmarks/detection_benchmark.py`.

Below the review line a finding is a guess that the plan hands to a person (a name hint alone,
values that contradict the name, a weak share of valid values). Those keep their fixed tier
values, which are documented in `docs/detection.md`, and calibration never moves a finding
across the line in either direction.

The map is a monotone piecewise-linear curve fitted with isotonic regression. Regenerate the
knots with `uv run python -m benchmarks.calibrate_detection` after changing detection rules.
"""

import numpy as np

# Findings below this confidence are proposed with a "please review" note.
REVIEW_CONFIDENCE = 0.6

# BEGIN GENERATED KNOTS
PERSONAL_DATA_KNOTS: tuple[tuple[float, float], ...] = (
    (0.6, 0.981),
    (0.881, 0.981),
    (1.0, 1.0),
)
NO_PERSONAL_DATA_KNOTS: tuple[tuple[float, float], ...] = (
    (0.6, 0.846),
    (1.0, 0.846),
)
# END GENERATED KNOTS


def calibrated_confidence(raw: float, *, found_personal_data: bool) -> float:
    """Return the calibrated confidence for a raw score between 0 and 1.

    Findings of personal data and findings of "nothing sensitive" have separate curves, because
    a column with no sign of personal data scores 1.0 whatever the evidence, and that score
    is wrong more often than a high score for personal data.
    Scores below the review line are returned unchanged.
    """
    if raw < REVIEW_CONFIDENCE:
        return raw
    knots = PERSONAL_DATA_KNOTS if found_personal_data else NO_PERSONAL_DATA_KNOTS
    raw_scores = [knot[0] for knot in knots]
    calibrated = [knot[1] for knot in knots]
    return max(REVIEW_CONFIDENCE, float(np.interp(raw, raw_scores, calibrated)))
