"""What every technique receives besides the column: randomness, the secret key and fakers."""

import hmac
from dataclasses import dataclass

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from oblitus.surrogates import SurrogateFactory
from oblitus.token_vault import TokenVault


@dataclass(frozen=True, slots=True)
class TechniqueContext:
    """Shared inputs for one anonymization run.

    The secret key makes pseudonyms reproducible across runs while keeping them
    impossible to reverse for anyone without the key. `randomness_key` drives every
    random draw: noise per row comes from keyed hashes of the row key, so a run is
    reproducible on any engine and nobody without the key can recompute the noise.
    `vault` is where tokenization keeps its token and value pairs; it is None unless the
    caller passes one.
    """

    secret_key: bytes
    surrogates: SurrogateFactory
    randomness_key: bytes
    vault: TokenVault | None = None

    def stream_key(self, column: str, purpose: str) -> bytes:
        """Key the random numbers one technique draws for one column.

        Each column and purpose gets independent numbers, so the noise of one column
        says nothing about another's.
        """
        return hkdf_sha256(self.randomness_key, f"stream:{purpose}:{column}", 32)

    def choice_key(self, column: str, purpose: str) -> bytes:
        """Key random choices that stand in for values, such as category codes or fakes.

        Unlike noise, these choices could link two releases: the same code or fake in both
        would join them. So they also depend on the secret key, which an anonymization run
        replaces with a throwaway one. Codes and fakes are then fresh in every anonymized
        release, and reproducible for the key holder under pseudonymization.
        """
        return hkdf_sha256(self.secret_key + self.randomness_key, f"choice:{purpose}:{column}", 32)

    def generator(self, column: str, purpose: str) -> np.random.Generator:
        """A seeded generator for random choices made in Python, such as shuffling codes.

        It follows `choice_key`, so its choices are fresh in every anonymized release.
        """
        seed = int.from_bytes(self.choice_key(column, purpose)[:8], "big")
        return np.random.default_rng(seed)

    def keyed_digest(self, value: str, algorithm: str = "sha256") -> bytes:
        """Return the HMAC of a value under the run's secret key, SHA-256 by default."""
        return hmac.new(self.secret_key, value.encode("utf-8"), algorithm).digest()

    def derived_key(self, purpose: str, length: int = 32) -> bytes:
        """Derive a key for one purpose from the secret key with HKDF-SHA256.

        Each purpose gets an independent key, so the encryption key cannot be
        recovered from pseudonyms made with the secret key, or the other way round.
        """
        return hkdf_sha256(self.secret_key, purpose, length)

    def keyed_seed(self, value: str) -> int:
        """Return a stable 63-bit seed derived from a value and the secret key."""
        return int.from_bytes(self.keyed_digest(value)[:8], "big") >> 1


def hkdf_sha256(material: bytes, purpose: str, length: int) -> bytes:
    derivation = HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=None,
        info=f"oblitus:{purpose}".encode(),
    )
    return derivation.derive(material)
