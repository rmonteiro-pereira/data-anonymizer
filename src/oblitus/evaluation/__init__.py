"""Measure the privacy a table offers and the statistical usefulness it keeps."""

from oblitus.evaluation.attacks import AttackRisk, AttackRiskReport, assess_attack_risks
from oblitus.evaluation.privacy import PrivacyReport, assess_privacy, t_closeness
from oblitus.evaluation.synthetic_privacy import (
    MembershipInference,
    SyntheticPrivacyReport,
    SyntheticPrivacyThresholds,
    assess_synthetic_privacy,
)
from oblitus.evaluation.utility import ColumnUtility, UtilityReport, compare_tables

__all__ = [
    "AttackRisk",
    "AttackRiskReport",
    "ColumnUtility",
    "MembershipInference",
    "PrivacyReport",
    "SyntheticPrivacyReport",
    "SyntheticPrivacyThresholds",
    "UtilityReport",
    "assess_attack_risks",
    "assess_privacy",
    "assess_synthetic_privacy",
    "compare_tables",
    "t_closeness",
]
