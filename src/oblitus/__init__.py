"""Detect sensitive data, anonymize it with a reviewed plan, and generate synthetic data."""

import logging

from oblitus.anonymizer import AnonymizationResult, Anonymizer
from oblitus.detection import ColumnFinding
from oblitus.differential_privacy import PrivacyBudget, Statistic
from oblitus.errors import (
    AnonymizerError,
    PlanError,
    PrivacyBudgetError,
    SchemaError,
    TableFormatError,
)
from oblitus.evaluation import (
    PrivacyReport,
    SyntheticPrivacyReport,
    SyntheticPrivacyThresholds,
    UtilityReport,
)
from oblitus.generalization_hierarchies import GeneralizationHierarchy
from oblitus.plan import (
    AnonymizationPlan,
    ColumnAction,
    Fidelity,
    JointMicroaggregation,
    KAnonymityMethod,
    Protection,
    RangeStyle,
)
from oblitus.protection import ProtectionAssessment, ProtectionLevel
from oblitus.synthesis import ColumnSchema, DatasetSchema, RelationalSchema, infer_schema
from oblitus.taxonomy import ColumnKind, DataCategory, Sensitivity
from oblitus.techniques import (
    BUILTIN_TECHNIQUES,
    StatisticalEffect,
    Technique,
    TechniqueContext,
    TechniqueFamily,
)
from oblitus.token_vault import MemoryVault, SqliteVault, TokenVault

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "BUILTIN_TECHNIQUES",
    "AnonymizationPlan",
    "AnonymizationResult",
    "Anonymizer",
    "AnonymizerError",
    "ColumnAction",
    "ColumnFinding",
    "ColumnKind",
    "ColumnSchema",
    "DataCategory",
    "DatasetSchema",
    "Fidelity",
    "MemoryVault",
    "GeneralizationHierarchy",
    "JointMicroaggregation",
    "KAnonymityMethod",
    "PlanError",
    "PrivacyBudget",
    "PrivacyBudgetError",
    "PrivacyReport",
    "Protection",
    "ProtectionAssessment",
    "ProtectionLevel",
    "RelationalSchema",
    "RangeStyle",
    "SchemaError",
    "Sensitivity",
    "SqliteVault",
    "Statistic",
    "StatisticalEffect",
    "SyntheticPrivacyReport",
    "SyntheticPrivacyThresholds",
    "TableFormatError",
    "Technique",
    "TechniqueContext",
    "TechniqueFamily",
    "TokenVault",
    "UtilityReport",
    "infer_schema",
]
