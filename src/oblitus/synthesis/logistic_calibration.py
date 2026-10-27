"""Shift probabilities on the log-odds scale until their mean is a chosen share.

Adding one constant to every row's log-odds keeps the order of the rows and the shape of the
dependence on the other columns, and only moves how often the event happens. Class imbalance
and missing-value shares both use it.
"""

import numpy as np
from scipy import special

# Probabilities are kept away from 0 and 1 so their log-odds stay finite.
_PROBABILITY_EDGE = 1e-9
_BISECTION_STEPS = 80
# The search covers the spread of the log-odds plus this much on either side.
_SEARCH_MARGIN = 40.0


def log_odds(probabilities: np.ndarray) -> np.ndarray:
    """Return the log-odds of each probability, kept finite at 0 and 1."""
    clipped = np.clip(probabilities, _PROBABILITY_EDGE, 1 - _PROBABILITY_EDGE)
    return special.logit(clipped)


def intercept_for_mean_probability(logits: np.ndarray, share: float) -> float:
    """Return the constant that makes the mean of `sigmoid(logits + constant)` equal `share`."""
    if len(logits) == 0:
        return 0.0
    limit = _SEARCH_MARGIN + float(np.max(np.abs(logits)))
    low = -limit
    high = limit
    for _ in range(_BISECTION_STEPS):
        middle = (low + high) / 2
        mean_probability = float(np.mean(special.expit(logits + middle)))
        if mean_probability < share:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def calibrate_probabilities(probabilities: np.ndarray, share: float) -> np.ndarray:
    """Return the probabilities with their log-odds shifted so that they average `share`."""
    logits = log_odds(probabilities)
    return special.expit(logits + intercept_for_mean_probability(logits, share))
