"""The built-in entropy sources: hardware generator, microphone, camera, CPU jitter, beacons.

Each source reads a device or the network and returns raw bytes. A source that cannot
deliver raises `EntropySourceUnavailableError` with the reason, and the key report shows it
as skipped. Sensor sources keep only the low-order bits of their samples, where thermal
noise lives and the signal (speech, scene) does not.
"""

import json
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from oblitus.errors import EntropySourceUnavailableError, KeyGenerationError
from oblitus.key_generation import OS_SOURCE_NAME, EntropySource

HARDWARE_RNG_PATH = Path("/dev/hwrng")
HARDWARE_BYTES = 64
AUDIO_SECONDS = 1.0
AUDIO_SAMPLE_RATE = 44_100
CAMERA_FRAMES = 3
TIMING_SAMPLES = 8192
LOW_BITS_KEPT = 2
BEACON_TIMEOUT_SECONDS = 5.0
DRAND_URL = "https://api.drand.sh/public/latest"
NIST_BEACON_URL = "https://beacon.nist.gov/beacon/2.0/pulse/last"

JsonFetcher = Callable[[str], dict[str, Any]]


def low_bits_to_bytes(samples: np.ndarray, bits: int = LOW_BITS_KEPT) -> bytes:
    """Keep the lowest `bits` bits of each integer sample and pack them into bytes."""
    if 8 % bits != 0:
        raise ValueError("bits must divide 8")
    kept = (samples.astype(np.int64).ravel() & ((1 << bits) - 1)).astype(np.uint8)
    per_byte = 8 // bits
    usable = len(kept) - len(kept) % per_byte
    grouped = kept[:usable].reshape(-1, per_byte)
    packed = np.zeros(len(grouped), dtype=np.uint8)
    for position in range(per_byte):
        packed |= grouped[:, position] << (bits * position)
    return packed.tobytes()


def read_hardware_rng(path: Path = HARDWARE_RNG_PATH) -> bytes:
    """Read bytes from the Linux hardware random number generator."""
    try:
        with path.open("rb", buffering=0) as device:
            return device.read(HARDWARE_BYTES)
    except OSError as error:
        raise EntropySourceUnavailableError(f"{path} is not readable: {error}") from error


def read_microphone_noise() -> bytes:
    """Record a second of ambient noise and keep the low bits of the samples."""
    try:
        import sounddevice  # type: ignore[import-untyped]
    except (ImportError, OSError) as error:
        raise EntropySourceUnavailableError(
            'sounddevice is not usable; install it with pip install "oblitus[audio]" '
            f"(it needs the PortAudio library): {error}"
        ) from error
    frames = int(AUDIO_SECONDS * AUDIO_SAMPLE_RATE)
    try:
        recording = sounddevice.rec(frames, samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype="int16")
        sounddevice.wait()
    except sounddevice.PortAudioError as error:
        raise EntropySourceUnavailableError(f"no usable microphone: {error}") from error
    return low_bits_to_bytes(recording)


def read_camera_noise() -> bytes:
    """Grab a few frames and keep the low bits of the pixels, where sensor noise lives."""
    try:
        import cv2
    except ImportError as error:
        raise EntropySourceUnavailableError(
            'opencv is not installed; install it with pip install "oblitus[camera]"'
        ) from error
    capture = cv2.VideoCapture(0)
    try:
        if not capture.isOpened():
            raise EntropySourceUnavailableError("no camera could be opened")
        frames = [capture.read() for _ in range(CAMERA_FRAMES)]
    finally:
        capture.release()
    images = [image for succeeded, image in frames if succeeded]
    if not images:
        raise EntropySourceUnavailableError("the camera returned no frames")
    return low_bits_to_bytes(np.concatenate([image.ravel() for image in images]))


def read_timing_jitter(clock: Callable[[], int] = time.perf_counter_ns) -> bytes:
    """Time a short busy loop many times and keep the low bits of the differences."""
    deltas = np.empty(TIMING_SAMPLES, dtype=np.int64)
    for index in range(TIMING_SAMPLES):
        start = clock()
        sum(range(50))
        deltas[index] = clock() - start
    return low_bits_to_bytes(deltas)


def fetch_json(url: str) -> dict[str, Any]:
    """Fetch a JSON document over HTTPS with a short timeout."""
    if not url.startswith("https://"):
        raise EntropySourceUnavailableError(f"refusing a non-HTTPS url: {url}")
    try:
        with urllib.request.urlopen(url, timeout=BEACON_TIMEOUT_SECONDS) as response:
            return json.loads(response.read())
    except (OSError, ValueError) as error:
        raise EntropySourceUnavailableError(f"{url} did not answer: {error}") from error


def read_beacons(fetcher: JsonFetcher = fetch_json) -> bytes:
    """Read the latest drand and NIST beacon outputs. Both are public."""
    try:
        drand_output = bytes.fromhex(fetcher(DRAND_URL)["randomness"])
        nist_output = bytes.fromhex(fetcher(NIST_BEACON_URL)["pulse"]["outputValue"])
    except (KeyError, TypeError, ValueError) as error:
        raise EntropySourceUnavailableError(f"unexpected beacon answer: {error}") from error
    return drand_output + nist_output


BUILTIN_SOURCES: dict[str, EntropySource] = {
    "hardware": EntropySource("hardware", read_hardware_rng),
    "microphone": EntropySource("microphone", read_microphone_noise),
    "camera": EntropySource("camera", read_camera_noise),
    "timing": EntropySource("timing", read_timing_jitter),
    "beacon": EntropySource("beacon", read_beacons, public=True),
}


def sources_by_name(names: list[str]) -> list[EntropySource]:
    """Resolve source names. "os" is accepted and ignored because it is always included."""
    resolved: list[EntropySource] = []
    for name in names:
        if name == OS_SOURCE_NAME or any(source.name == name for source in resolved):
            continue
        if name not in BUILTIN_SOURCES:
            known = ", ".join([OS_SOURCE_NAME, *BUILTIN_SOURCES])
            raise KeyGenerationError(f'unknown entropy source "{name}"; choose from {known}')
        resolved.append(BUILTIN_SOURCES[name])
    return resolved
