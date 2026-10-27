"""Detect sensitive data, anonymize it with a reviewed plan, and generate synthetic data."""

import logging

from oblitus.anonymizer import AnonymizationResult, Anonymizer
from oblitus.detection import ColumnFinding
from oblitus.errors import AnonymizerError, PlanError, SchemaError, TableFormatError
from oblitus.evaluation import PrivacyReport, UtilityReport
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
from oblitus.synthesis import ColumnSchema, DatasetSchema, infer_schema
from oblitus.taxonomy import ColumnKind, DataCategory, Sensitivity
from oblitus.techniques import (
    BUILTIN_TECHNIQUES,
    StatisticalEffect,
    Technique,
    TechniqueContext,
    TechniqueFamily,
)

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
    "GeneralizationHierarchy",
    "JointMicroaggregation",
    "KAnonymityMethod",
    "PlanError",
    "PrivacyReport",
    "Protection",
    "ProtectionAssessment",
    "ProtectionLevel",
    "RangeStyle",
    "SchemaError",
    "Sensitivity",
    "StatisticalEffect",
    "TableFormatError",
    "Technique",
    "TechniqueContext",
    "TechniqueFamily",
    "UtilityReport",
    "infer_schema",
]
