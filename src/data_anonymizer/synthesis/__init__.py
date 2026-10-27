"""Generate synthetic tables from a schema, or from a schema learned from a sample."""

from data_anonymizer.synthesis.challenge import Challenge, build_challenge
from data_anonymizer.synthesis.generator import generate_from_reference, generate_from_schema
from data_anonymizer.synthesis.inference import infer_schema
from data_anonymizer.synthesis.schema import ColumnSchema, DatasetSchema

__all__ = [
    "Challenge",
    "build_challenge",
    "ColumnSchema",
    "DatasetSchema",
    "generate_from_reference",
    "generate_from_schema",
    "infer_schema",
]
