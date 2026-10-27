"""Generate synthetic tables from a schema, or from a schema learned from a sample."""

from oblitus.synthesis.challenge import Challenge, build_challenge
from oblitus.synthesis.generator import generate_from_reference, generate_from_schema
from oblitus.synthesis.inference import infer_schema
from oblitus.synthesis.relational import RelationalSchema, load_schema
from oblitus.synthesis.relational_challenge import build_relational_challenge
from oblitus.synthesis.relational_generation import GeneratedTables, generate_tables
from oblitus.synthesis.schema import ColumnSchema, DatasetSchema

__all__ = [
    "Challenge",
    "build_challenge",
    "ColumnSchema",
    "DatasetSchema",
    "GeneratedTables",
    "RelationalSchema",
    "build_relational_challenge",
    "generate_from_reference",
    "generate_from_schema",
    "generate_tables",
    "infer_schema",
    "load_schema",
]
