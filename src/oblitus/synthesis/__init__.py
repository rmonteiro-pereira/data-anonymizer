"""Generate synthetic tables from a schema, or from a schema learned from a sample."""

from oblitus.synthesis.challenge import Challenge, build_challenge
from oblitus.synthesis.generator import generate_from_reference, generate_from_schema
from oblitus.synthesis.inference import infer_schema
from oblitus.synthesis.schema import ColumnSchema, DatasetSchema

__all__ = [
    "Challenge",
    "build_challenge",
    "ColumnSchema",
    "DatasetSchema",
    "generate_from_reference",
    "generate_from_schema",
    "infer_schema",
]
