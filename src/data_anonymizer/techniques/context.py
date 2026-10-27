"""What every technique receives besides the column: randomness, the secret key and fakers."""

import hmac
from dataclasses import dataclass

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from data_anonymizer.surrogates import SurrogateFactory


@dataclass(frozen=True, slots=True)
class TechniqueContext:
    """Shared inputs for one anonymization run.

    The random generator makes noise reproducible for a given seed. The secret key
    makes pseudonyms reproducible across runs while keeping them impossible to
    reverse for anyone without the key.
    """

    rng: np.random.Generator
    secret_key: bytes
    surrogates: SurrogateFactory

    def keyed_digest(self, value: str, algorithm: str = "sha256") -> bytes:
        """Return the HMAC of a value under the run's secret key, SHA-256 by default."""
        return hmac.new(self.secret_key, value.encode("utf-8"), algorithm).digest()

    def derived_key(self, purpose: str, length: int = 32) -> bytes:
        """Derive a key for one purpose from the secret key with HKDF-SHA256.

        Each purpose gets an independent key, so the encryption key cannot be
        recovered from pseudonyms made with the secret key, or the other way round.
        """
        derivation = HKDF(
            algorithm=hashes.SHA256(),
            length=length,
            salt=None,
            info=f"data-anonymizer:{purpose}".encode(),
        )
        return derivation.derive(self.secret_key)

    def keyed_seed(self, value: str) -> int:
        """Return a stable 63-bit seed derived from a value and the secret key."""
        return int.from_bytes(self.keyed_digest(value)[:8], "big") >> 1
