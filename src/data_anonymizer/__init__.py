"""Detect sensitive data, anonymize it with a reviewed plan, and generate synthetic data."""

import logging

from data_anonymizer.anonymizer import AnonymizationResult, Anonymizer
from data_anonymizer.detection import ColumnFinding
from data_anonymizer.errors import AnonymizerError, PlanError, SchemaError, TableFormatError
from data_anonymizer.evaluation import PrivacyReport, UtilityReport
from data_anonymizer.plan import AnonymizationPlan, ColumnAction, Fidelity, Protection
from data_anonymizer.protection import ProtectionAssessment, ProtectionLevel
from data_anonymizer.synthesis import ColumnSchema, DatasetSchema, infer_schema
from data_anonymizer.taxonomy import ColumnKind, DataCategory, Sensitivity
from data_anonymizer.techniques import (
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
    "PlanError",
    "PrivacyReport",
    "Protection",
    "ProtectionAssessment",
    "ProtectionLevel",
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
