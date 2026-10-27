import pytest

from oblitus import PrivacyBudget, PrivacyBudgetError
from oblitus.differential_privacy import compose_budget, epsilon_from_rho, rho_from_epsilon_delta
from oblitus.differential_privacy.budget import (
    Spend,
    exponential_mechanism_spend,
    gaussian_spend,
    pure_epsilon_spend,
)


def test_converting_epsilon_to_rho_and_back_gives_the_same_epsilon():
    rho = rho_from_epsilon_delta(2.0, 1e-6)

    assert epsilon_from_rho(rho, 1e-6) == pytest.approx(2.0, rel=1e-12)


def test_a_known_rho_converts_to_the_standard_epsilon():
    expected = 0.5 + 2 * (0.5 * 11.512925465) ** 0.5

    assert epsilon_from_rho(0.5, 1e-5) == pytest.approx(expected)


def test_each_mechanism_is_priced_by_its_own_formula():
    assert pure_epsilon_spend("laplace", 2.0).rho == 2.0
    assert gaussian_spend("gaussian", sensitivity=3.0, noise_scale=6.0).rho == pytest.approx(0.125)
    assert exponential_mechanism_spend("choice", 4.0).rho == 2.0


def test_composition_adds_rho_and_reports_the_epsilon_it_implies():
    budget = PrivacyBudget.from_epsilon_delta(1.0, 1e-5)
    third = (budget.rho or 0) / 3

    report = compose_budget(budget, [Spend("a", third), Spend("b", third), Spend("c", third)])

    assert report.spent_rho == pytest.approx(budget.rho, rel=1e-12)
    assert report.spent_epsilon == pytest.approx(1.0, rel=1e-9)
    assert report.remaining_epsilon == pytest.approx(0.0, abs=1e-9)


def test_composition_refuses_to_exceed_the_declared_rho():
    budget = PrivacyBudget.from_rho(0.1)

    with pytest.raises(PrivacyBudgetError, match="above the declared"):
        compose_budget(budget, [Spend("a", 0.06), Spend("b", 0.06)])


def test_spending_less_than_declared_reports_a_smaller_epsilon():
    budget = PrivacyBudget.from_epsilon_delta(2.0, 1e-6)

    report = compose_budget(budget, [Spend("half", (budget.rho or 0) / 2)])

    assert report.spent_epsilon < 2.0
    assert report.remaining_epsilon > 0


def test_a_pure_budget_adds_epsilons_and_refuses_the_total_above_it():
    budget = PrivacyBudget.pure(1.0)

    report = compose_budget(budget, [pure_epsilon_spend("a", 0.4), pure_epsilon_spend("b", 0.6)])
    assert report.spent_epsilon == pytest.approx(1.0)

    with pytest.raises(PrivacyBudgetError, match="above the declared"):
        compose_budget(budget, [pure_epsilon_spend("a", 0.4), pure_epsilon_spend("b", 0.7)])


def test_a_pure_budget_refuses_gaussian_noise():
    with pytest.raises(PrivacyBudgetError, match="not pure"):
        compose_budget(PrivacyBudget.pure(1.0), [gaussian_spend("noisy", 1.0, 5.0)])


@pytest.mark.parametrize("epsilon", [0, -1, float("inf"), float("nan")])
def test_a_budget_needs_a_positive_finite_epsilon(epsilon):
    with pytest.raises(PrivacyBudgetError):
        PrivacyBudget.pure(epsilon)


@pytest.mark.parametrize("delta", [0, 1, 1.5, -0.1])
def test_delta_must_be_a_probability_strictly_between_zero_and_one(delta):
    with pytest.raises(PrivacyBudgetError):
        PrivacyBudget.from_epsilon_delta(1.0, delta)
