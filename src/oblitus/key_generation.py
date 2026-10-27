"""Secret keys mixed from several entropy sources.

`os.urandom` is always part of the mix, so a key stays secret even when every other source
is weak, broken or public. Other sources add independence: if the operating system's
generator were ever compromised, the key would still hold as long as one physical source
is sound. Each physical source passes a health check first, and the report says what each
source contributed.
"""

import math
import os
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from oblitus.errors import EntropySourceUnavailableError, KeyGenerationError
from oblitus.techniques.context import hkdf_sha256

OS_SOURCE_NAME = "os"
DEFAULT_KEY_BYTES = 32
MIN_KEY_BYTES = 16
MAX_KEY_BYTES = 255 * 32

# A physical sample must hold at least this many bytes and this much min-entropy per byte.
# Raw sensor and timing bytes are far from uniform, so the bar only catches broken sources.
MIN_SAMPLE_BYTES = 64
MIN_ENTROPY_BITS_PER_BYTE = 2.0


@dataclass(frozen=True, slots=True)
class EntropySource:
    """A named producer of random bytes.

    `public` marks output that anyone can read, such as a randomness beacon. It adds
    unpredictability at a past moment but no secrecy, and it skips the health check.
    """

    name: str
    collect: Callable[[], bytes]
    public: bool = False


class SourceStatus(StrEnum):
    USED = "used"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class SourceReport:
    """What one source contributed: bytes, estimated min-entropy and why it was skipped."""

    name: str
    status: SourceStatus
    byte_count: int
    min_entropy_bits: float
    note: str = ""


@dataclass(frozen=True, slots=True)
class GeneratedKey:
    """A key and the report of the sources that went into it."""

    key: bytes
    sources: tuple[SourceReport, ...]


def estimate_min_entropy_bits(sample: bytes) -> float:
    """Estimate the sample's total min-entropy with the most-common-value method."""
    if not sample:
        return 0.0
    most_common = Counter(sample).most_common(1)[0][1]
    return -math.log2(most_common / len(sample)) * len(sample)


def check_sample_health(sample: bytes) -> str | None:
    """Return why a physical sample is unfit, or None when it passes."""
    if len(sample) < MIN_SAMPLE_BYTES:
        return f"only {len(sample)} bytes, need at least {MIN_SAMPLE_BYTES}"
    bits_per_byte = estimate_min_entropy_bits(sample) / len(sample)
    if bits_per_byte < MIN_ENTROPY_BITS_PER_BYTE:
        return (
            f"too repetitive: {bits_per_byte:.2f} bits of min-entropy per byte, "
            f"need {MIN_ENTROPY_BITS_PER_BYTE}"
        )
    return None


def generate_key(
    extra_sources: Sequence[EntropySource] = (),
    *,
    length: int = DEFAULT_KEY_BYTES,
    random_bytes: Callable[[int], bytes] = os.urandom,
) -> GeneratedKey:
    """Mix `os.urandom` and the extra sources into a key of `length` bytes.

    The operating system source cannot be left out or replaced: a source named "os" among
    the extras is refused. `random_bytes` exists so tests can fix the operating system's
    output; callers leave it alone.
    """
    _check_length(length)
    _check_names(extra_sources)
    os_source = EntropySource(OS_SOURCE_NAME, lambda: random_bytes(max(length, DEFAULT_KEY_BYTES)))
    reports: list[SourceReport] = []
    mixed_input = b""
    for source in (os_source, *extra_sources):
        report, sample = _collect(source)
        reports.append(report)
        mixed_input += _tagged(source.name, sample)
    key = hkdf_sha256(mixed_input, "key generation", length)
    return GeneratedKey(key=key, sources=tuple(reports))


def _check_length(length: int) -> None:
    if not MIN_KEY_BYTES <= length <= MAX_KEY_BYTES:
        raise KeyGenerationError(
            f"key length must be between {MIN_KEY_BYTES} and {MAX_KEY_BYTES} bytes, got {length}"
        )


def _check_names(extra_sources: Sequence[EntropySource]) -> None:
    names = [source.name for source in extra_sources]
    if OS_SOURCE_NAME in names:
        raise KeyGenerationError(
            f'"{OS_SOURCE_NAME}" is always included and cannot be given as an extra source'
        )
    if len(set(names)) != len(names):
        raise KeyGenerationError(f"entropy sources must have distinct names, got {names}")


def _collect(source: EntropySource) -> tuple[SourceReport, bytes]:
    """Read one source; an unavailable or unhealthy one is reported and contributes nothing."""
    try:
        sample = source.collect()
    except EntropySourceUnavailableError as error:
        return _skipped(source.name, str(error)), b""
    if source.public:
        note = "public output: adds unpredictability, no secrecy"
        return _used(source.name, sample, note), sample
    problem = check_sample_health(sample) if source.name != OS_SOURCE_NAME else None
    if problem is not None:
        return _skipped(source.name, problem, len(sample)), b""
    return _used(source.name, sample), sample


def _used(name: str, sample: bytes, note: str = "") -> SourceReport:
    entropy = estimate_min_entropy_bits(sample)
    return SourceReport(name, SourceStatus.USED, len(sample), entropy, note)


def _skipped(name: str, reason: str, byte_count: int = 0) -> SourceReport:
    return SourceReport(name, SourceStatus.SKIPPED, byte_count, 0.0, reason)


def _tagged(name: str, sample: bytes) -> bytes:
    """Prefix the name and the sample with their lengths so no two mixes can collide."""
    encoded_name = name.encode("utf-8")
    return (
        len(encoded_name).to_bytes(2, "big")
        + encoded_name
        + len(sample).to_bytes(4, "big")
        + sample
    )
