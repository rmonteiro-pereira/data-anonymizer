"""Privacy budget accounting for differential privacy.

Every mechanism is accounted in zero-concentrated differential privacy (zCDP), where costs
simply add up. A budget declared as (epsilon, delta) is converted to the equivalent rho, and
the epsilon actually spent is converted back at the same delta, so the report never claims
more than was declared. A budget declared as a pure epsilon (delta = 0) only accepts
mechanisms that are pure epsilon-DP, such as the Laplace mechanism, and adds their epsilons.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from oblitus.errors import PrivacyBudgetError

# Tolerance for float rounding when many small spends add up to exactly the whole budget.
_RELATIVE_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class PrivacyBudget:
    """The total privacy loss a release may cost.

    Build it with `pure(epsilon)`, `from_epsilon_delta(epsilon, delta)` or `from_rho(rho)`.
    `delta` is zero for a pure budget; `rho` is the zCDP parameter, or None for a pure budget.
    """

    epsilon: float
    delta: float
    rho: float | None

    @classmethod
    def pure(cls, epsilon: float) -> "PrivacyBudget":
        """A budget of epsilon-DP with no failure probability."""
        _require_positive(epsilon, "epsilon")
        return cls(epsilon=epsilon, delta=0.0, rho=None)

    @classmethod
    def from_epsilon_delta(cls, epsilon: float, delta: float) -> "PrivacyBudget":
        """A budget of (epsilon, delta)-DP, accounted in zCDP."""
        _require_positive(epsilon, "epsilon")
        _require_delta(delta)
        return cls(epsilon=epsilon, delta=delta, rho=rho_from_epsilon_delta(epsilon, delta))

    @classmethod
    def from_rho(cls, rho: float, delta: float = 1e-6) -> "PrivacyBudget":
        """A budget of rho-zCDP; `delta` only chooses how epsilon is reported."""
        _require_positive(rho, "rho")
        _require_delta(delta)
        return cls(epsilon=epsilon_from_rho(rho, delta), delta=delta, rho=rho)

    @property
    def is_pure(self) -> bool:
        """True when only pure epsilon-DP mechanisms are allowed."""
        return self.rho is None


@dataclass(frozen=True, slots=True)
class Spend:
    """The cost of one mechanism: its zCDP rho, and its epsilon when it is pure epsilon-DP."""

    label: str
    rho: float
    pure_epsilon: float | None = None


def pure_epsilon_spend(label: str, epsilon: float) -> Spend:
    """A pure epsilon-DP mechanism, which is also (epsilon^2 / 2)-zCDP."""
    _require_positive(epsilon, "epsilon")
    return Spend(label, rho=epsilon**2 / 2, pure_epsilon=epsilon)


def gaussian_spend(label: str, sensitivity: float, noise_scale: float) -> Spend:
    """A Gaussian mechanism with L2 sensitivity and standard deviation `noise_scale`."""
    _require_positive(noise_scale, "noise_scale")
    return Spend(label, rho=sensitivity**2 / (2 * noise_scale**2))


def exponential_mechanism_spend(label: str, epsilon: float) -> Spend:
    """The exponential mechanism at epsilon, which is (epsilon^2 / 8)-zCDP by bounded range."""
    _require_positive(epsilon, "epsilon")
    return Spend(label, rho=epsilon**2 / 8, pure_epsilon=epsilon)


@dataclass(frozen=True, slots=True)
class BudgetReport:
    """What a release cost against what was declared."""

    declared_epsilon: float
    delta: float
    declared_rho: float | None
    spent_rho: float
    spent_epsilon: float
    spends: tuple[Spend, ...]

    @property
    def remaining_epsilon(self) -> float:
        """Declared epsilon minus the epsilon spent, never below zero."""
        return max(0.0, self.declared_epsilon - self.spent_epsilon)


def compose_budget(budget: PrivacyBudget, spends: Sequence[Spend]) -> BudgetReport:
    """Add up what the mechanisms cost and refuse any total above the declared budget.

    Returns the rho and the epsilon spent. Raises `PrivacyBudgetError` when the spends exceed
    the budget, or when a pure budget includes a mechanism that is not pure epsilon-DP.
    """
    if budget.is_pure:
        return _compose_pure(budget, spends)
    return _compose_zcdp(budget, spends)


def _compose_pure(budget: PrivacyBudget, spends: Sequence[Spend]) -> BudgetReport:
    not_pure = [spend.label for spend in spends if spend.pure_epsilon is None]
    if not_pure:
        raise PrivacyBudgetError(
            f"{not_pure} are not pure epsilon-DP mechanisms; declare delta or rho "
            "to use Gaussian noise"
        )
    spent_epsilon = math.fsum(spend.pure_epsilon or 0.0 for spend in spends)
    if spent_epsilon > budget.epsilon * (1 + _RELATIVE_TOLERANCE):
        raise PrivacyBudgetError(
            f"the mechanisms spend epsilon {spent_epsilon:.6g}, above the declared "
            f"{budget.epsilon:.6g}"
        )
    spent_rho = math.fsum(spend.rho for spend in spends)
    return BudgetReport(budget.epsilon, 0.0, None, spent_rho, spent_epsilon, tuple(spends))


def _compose_zcdp(budget: PrivacyBudget, spends: Sequence[Spend]) -> BudgetReport:
    declared_rho = budget.rho or 0.0
    spent_rho = math.fsum(spend.rho for spend in spends)
    if spent_rho > declared_rho * (1 + _RELATIVE_TOLERANCE):
        raise PrivacyBudgetError(
            f"the mechanisms spend rho {spent_rho:.6g}, above the declared {declared_rho:.6g} "
            f"(epsilon {budget.epsilon:.6g} at delta {budget.delta:g})"
        )
    spent_epsilon = min(budget.epsilon, epsilon_from_rho(spent_rho, budget.delta))
    return BudgetReport(
        budget.epsilon, budget.delta, declared_rho, spent_rho, spent_epsilon, tuple(spends)
    )


def rho_from_epsilon_delta(epsilon: float, delta: float) -> float:
    """A rho whose rho-zCDP implies (epsilon, delta)-DP.

    It inverts epsilon = rho + 2 * sqrt(rho * ln(1 / delta)), the conversion of Bun and
    Steinke (2016). It is safe but not the tightest: the conversion of Canonne, Kamath and
    Steinke (2020) allows a larger rho, so this one adds 15 to 45% more noise than needed.
    """
    log_inverse_delta = math.log(1 / delta)
    return (math.sqrt(log_inverse_delta + epsilon) - math.sqrt(log_inverse_delta)) ** 2


def epsilon_from_rho(rho: float, delta: float) -> float:
    """The epsilon such that rho-zCDP implies (epsilon, delta)-DP."""
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


def _require_positive(value: float, name: str) -> None:
    if not value > 0 or math.isinf(value):
        raise PrivacyBudgetError(f"{name} must be a positive number, got {value}")


def _require_delta(delta: float) -> None:
    if not 0 < delta < 1:
        raise PrivacyBudgetError(f"delta must be between 0 and 1 (exclusive), got {delta}")
