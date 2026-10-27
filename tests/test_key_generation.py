import base64

import numpy as np
import pytest

from oblitus.anonymizer import Anonymizer
from oblitus.cli import main
from oblitus.entropy_sources import (
    low_bits_to_bytes,
    read_beacons,
    read_timing_jitter,
    sources_by_name,
)
from oblitus.errors import EntropySourceUnavailableError, KeyGenerationError
from oblitus.key_generation import (
    EntropySource,
    SourceStatus,
    check_sample_health,
    estimate_min_entropy_bits,
    generate_key,
)
from oblitus.plan import Protection

NOISE = bytes(np.random.default_rng(3).integers(0, 256, 256, dtype=np.uint8))


def fixed_os(byte: int):
    return lambda size: bytes([byte]) * size


def report_for(generated, name):
    return next(report for report in generated.sources if report.name == name)


def test_key_has_the_requested_length():
    assert len(generate_key().key) == 32
    assert len(generate_key(length=48).key) == 48


def test_key_is_deterministic_given_the_same_inputs():
    sources = [EntropySource("noise", lambda: NOISE)]
    first = generate_key(sources, random_bytes=fixed_os(1))
    second = generate_key(sources, random_bytes=fixed_os(1))
    assert first.key == second.key


def test_every_source_changes_the_key():
    noise = EntropySource("noise", lambda: NOISE)
    reversed_noise = EntropySource("noise", lambda: NOISE[::-1])
    base = generate_key([noise], random_bytes=fixed_os(1))
    other_noise = generate_key([reversed_noise], random_bytes=fixed_os(1))
    other_os = generate_key([noise], random_bytes=fixed_os(2))
    assert len({base.key, other_noise.key, other_os.key}) == 3


def test_the_operating_system_source_cannot_be_removed_or_replaced():
    fake_os = EntropySource("os", lambda: b"\x00" * 64)
    with pytest.raises(KeyGenerationError):
        generate_key([fake_os])
    assert report_for(generate_key(), "os").status is SourceStatus.USED


def test_a_constant_source_alone_cannot_fix_the_key():
    constant = EntropySource("constant", lambda: b"\x07" * 256)
    first = generate_key([constant])
    second = generate_key([constant])
    assert first.key != second.key
    assert report_for(first, "constant").status is SourceStatus.SKIPPED


def test_a_public_source_adds_no_secrecy():
    public = EntropySource("beacon", lambda: NOISE, public=True)
    first = generate_key([public])
    second = generate_key([public])
    assert first.key != second.key
    assert "public" in report_for(first, "beacon").note


def test_health_check_rejects_repetitive_and_short_samples():
    assert check_sample_health(b"\x00" * 256) is not None
    assert check_sample_health(b"\x00\x01" * 128) is not None
    assert check_sample_health(NOISE[:10]) is not None
    assert check_sample_health(NOISE) is None


def test_min_entropy_of_constant_and_uniform_data():
    assert estimate_min_entropy_bits(b"\x01" * 100) == 0.0
    assert estimate_min_entropy_bits(bytes(range(256))) == pytest.approx(2048)


def test_an_unavailable_source_is_reported_not_fatal():
    def missing() -> bytes:
        raise EntropySourceUnavailableError("no device")

    generated = generate_key([EntropySource("microphone", missing)])
    skipped = report_for(generated, "microphone")
    assert skipped.status is SourceStatus.SKIPPED
    assert skipped.note == "no device"
    assert len(generated.key) == 32


def test_invalid_lengths_and_duplicate_names_are_refused():
    with pytest.raises(KeyGenerationError):
        generate_key(length=8)
    twin = EntropySource("noise", lambda: NOISE)
    with pytest.raises(KeyGenerationError):
        generate_key([twin, twin])


def test_low_bits_are_packed_four_samples_per_byte():
    samples = np.array([0b01, 0b10, 0b11, 0b00, 0b111], dtype=np.int16)
    assert low_bits_to_bytes(samples) == bytes([0b00111001])


def test_timing_jitter_reads_an_injected_clock():
    ticks = iter(range(0, 10**9, 3))
    assert len(read_timing_jitter(clock=lambda: next(ticks))) == 2048


def test_beacons_are_read_through_the_injected_fetcher():
    answers = {
        "https://api.drand.sh/public/latest": {"randomness": "ab" * 32},
        "https://beacon.nist.gov/beacon/2.0/pulse/last": {"pulse": {"outputValue": "cd" * 64}},
    }
    assert read_beacons(lambda url: answers[url]) == bytes.fromhex("ab" * 32 + "cd" * 64)
    with pytest.raises(EntropySourceUnavailableError):
        read_beacons(lambda url: {})


def test_unknown_source_names_are_refused_and_os_is_implicit():
    assert sources_by_name(["os"]) == []
    assert [source.name for source in sources_by_name(["timing", "timing"])] == ["timing"]
    with pytest.raises(KeyGenerationError):
        sources_by_name(["radio"])


def test_cli_keygen_prints_the_key_and_reports_sources(capsys):
    assert main(["keygen", "--source", "timing"]) == 0
    captured = capsys.readouterr()
    assert len(bytes.fromhex(captured.out.strip())) == 32
    assert "os" in captured.err
    assert "timing" in captured.err


def test_cli_keygen_base64_and_unknown_source(capsys):
    assert main(["keygen", "--format", "base64"]) == 0
    assert len(base64.b64decode(capsys.readouterr().out.strip())) == 32
    assert main(["keygen", "--source", "radio"]) == 2


def test_anonymization_can_draw_its_throwaway_key_from_the_mixed_pool(people):
    draws = []

    def mixed_pool_key() -> bytes:
        draws.append(generate_key().key)
        return draws[-1]

    anonymizer = Anonymizer(secret_key="k" * 32, seed=1, throwaway_key_source=mixed_pool_key)
    plan = anonymizer.propose(people, protection=Protection.ANONYMIZATION)
    anonymizer.apply(people, plan)
    assert draws
