"""What every technique receives besides the column: randomness, the secret key and fakers."""

import hmac
from dataclasses import dataclass

import numpy as np
import pandas as pd
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from data_anonymizer.surrogates import SurrogateFactory


@dataclass(frozen=True, slots=True)
class TechniqueContext:
    """Shared inputs for one anonymization run.

    The secret key makes pseudonyms reproducible across runs while keeping them
    impossible to reverse for anyone without the key. `randomness_key` drives every
    random draw: noise per row comes from keyed hashes of the row key, so a run is
    reproducible on any engine and nobody without the key can recompute the noise.

    `rng` and `table` serve techniques that still work on pandas columns: a random
    generator, and the original table for grouping by another column.
    """

    rng: np.random.Generator
    secret_key: bytes
    surrogates: SurrogateFactory
    table: pd.DataFrame | None = None
    randomness_key: bytes = b""

    def stream_key(self, column: str, purpose: str) -> bytes:
        """Key the random numbers one technique draws for one column.

        Each column and purpose gets independent numbers, so the noise of one column
        says nothing about another's.
        """
        return _hkdf(self.randomness_key, f"stream:{purpose}:{column}", 32)

    def generator(self, column: str, purpose: str) -> np.random.Generator:
        """A seeded generator for random choices made in Python, such as shuffling codes."""
        seed = int.from_bytes(self.stream_key(column, purpose)[:8], "big")
        return np.random.default_rng(seed)

    def group_keys(self, column: str) -> pd.Series:
        """Return another column of the original table, for techniques that group rows by it."""
        if self.table is None:
            raise ValueError(f"grouping by {column!r} needs the original table")
        if column not in self.table.columns:
            raise ValueError(f"grouping column {column!r} is not in the table")
        return self.table[column]

    def keyed_digest(self, value: str, algorithm: str = "sha256") -> bytes:
        """Return the HMAC of a value under the run's secret key, SHA-256 by default."""
        return hmac.new(self.secret_key, value.encode("utf-8"), algorithm).digest()

    def derived_key(self, purpose: str, length: int = 32) -> bytes:
        """Derive a key for one purpose from the secret key with HKDF-SHA256.

        Each purpose gets an independent key, so the encryption key cannot be
        recovered from pseudonyms made with the secret key, or the other way round.
        """
        return _hkdf(self.secret_key, purpose, length)

    def keyed_seed(self, value: str) -> int:
        """Return a stable 63-bit seed derived from a value and the secret key."""
        return int.from_bytes(self.keyed_digest(value)[:8], "big") >> 1


def _hkdf(material: bytes, purpose: str, length: int) -> bytes:
    derivation = HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=None,
        info=f"data-anonymizer:{purpose}".encode(),
    )
    return derivation.derive(material)
