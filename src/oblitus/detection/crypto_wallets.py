"""Checksums of crypto wallet addresses: Bitcoin (base58check and bech32) and Ethereum.

A wallet address is a pseudonym, but it links every transaction of its owner, so it is
treated as an identifier. The checksums keep random strings of the same alphabet out.
"""

import hashlib

_BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_BITCOIN_VERSIONS = frozenset({0x00, 0x05})
_BASE58_PAYLOAD_BYTES = 25
_CHECKSUM_BYTES = 4

_BECH32_ALPHABET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
_BECH32_GENERATORS = (0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3)
_BECH32_CONSTANT = 1
_BECH32M_CONSTANT = 0x2BC830A3
_BECH32_HRP = "bc"
_BECH32_CHECKSUM_CHARACTERS = 6

_ETHEREUM_HEX_DIGITS = 40

_KECCAK_ROUNDS = 24
_KECCAK_RATE_BYTES = 136
_KECCAK_MASK = (1 << 64) - 1
_KECCAK_ROTATIONS = (
    (0, 36, 3, 41, 18),
    (1, 44, 10, 45, 2),
    (62, 6, 43, 15, 61),
    (28, 55, 25, 21, 56),
    (27, 20, 39, 8, 14),
)
_KECCAK_ROUND_CONSTANTS = (
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
)


def is_valid_bitcoin_address(value: str) -> bool:
    """A Bitcoin address: legacy or script-hash base58check, or a bech32 SegWit address."""
    if value[:1] in "13":
        return _is_valid_base58check(value)
    return _is_valid_bech32(value)


def is_valid_ethereum_address(value: str) -> bool:
    """An Ethereum address: `0x` and 40 hex digits, with the EIP-55 check when mixed case."""
    digits = value[2:]
    if not value.startswith("0x") or len(digits) != _ETHEREUM_HEX_DIGITS:
        return False
    if not _is_hexadecimal(digits):
        return False
    if digits == digits.lower() or digits == digits.upper():
        return True
    return digits == eip55_checksum_case(digits)


def eip55_checksum_case(hex_digits: str) -> str:
    """Return 40 hex digits cased as EIP-55 prescribes."""
    lowered = hex_digits.lower()
    hashed = keccak256(lowered.encode("ascii")).hex()
    return "".join(
        character.upper() if int(hashed[index], 16) >= 8 else character
        for index, character in enumerate(lowered)
    )


def base58check_encode(payload: bytes) -> str:
    """Return the base58check text of a payload, with its four checksum bytes appended."""
    data = payload + _double_sha256(payload)[:_CHECKSUM_BYTES]
    number = int.from_bytes(data, "big")
    characters = []
    while number:
        number, remainder = divmod(number, 58)
        characters.append(_BASE58_ALPHABET[remainder])
    leading_zeros = len(data) - len(data.lstrip(b"\x00"))
    return "1" * leading_zeros + "".join(reversed(characters))


def _is_valid_base58check(value: str) -> bool:
    if any(character not in _BASE58_ALPHABET for character in value):
        return False
    number = 0
    for character in value:
        number = number * 58 + _BASE58_ALPHABET.index(character)
    leading_ones = len(value) - len(value.lstrip("1"))
    data = b"\x00" * leading_ones + number.to_bytes((number.bit_length() + 7) // 8, "big")
    if len(data) != _BASE58_PAYLOAD_BYTES or data[0] not in _BITCOIN_VERSIONS:
        return False
    payload, checksum = data[:-_CHECKSUM_BYTES], data[-_CHECKSUM_BYTES:]
    return _double_sha256(payload)[:_CHECKSUM_BYTES] == checksum


def _double_sha256(payload: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(payload).digest()).digest()


def _is_valid_bech32(value: str) -> bool:
    if value != value.lower() and value != value.upper():
        return False
    text = value.lower()
    separator = text.rfind("1")
    human_part, data_part = text[:separator], text[separator + 1 :]
    if human_part != _BECH32_HRP or len(data_part) < _BECH32_CHECKSUM_CHARACTERS + 1:
        return False
    if any(character not in _BECH32_ALPHABET for character in data_part):
        return False
    data = [_BECH32_ALPHABET.index(character) for character in data_part]
    residue = _bech32_polymod(_expand_human_part(human_part) + data)
    return residue in (_BECH32_CONSTANT, _BECH32M_CONSTANT)


def _expand_human_part(human_part: str) -> list[int]:
    codes = [ord(character) for character in human_part]
    return [code >> 5 for code in codes] + [0] + [code & 31 for code in codes]


def _bech32_polymod(values: list[int]) -> int:
    checksum = 1
    for value in values:
        top = checksum >> 25
        checksum = (checksum & 0x1FFFFFF) << 5 ^ value
        for bit, generator in enumerate(_BECH32_GENERATORS):
            if (top >> bit) & 1:
                checksum ^= generator
    return checksum


def _is_hexadecimal(text: str) -> bool:
    return all(character in "0123456789abcdefABCDEF" for character in text)


def keccak256(data: bytes) -> bytes:
    """Keccak-256 as Ethereum uses it, which pads differently from the standard SHA3-256."""
    padded = bytearray(data)
    padded.append(0x01)
    while len(padded) % _KECCAK_RATE_BYTES:
        padded.append(0x00)
    padded[-1] |= 0x80
    state = [0] * 25
    for offset in range(0, len(padded), _KECCAK_RATE_BYTES):
        block = padded[offset : offset + _KECCAK_RATE_BYTES]
        for lane in range(_KECCAK_RATE_BYTES // 8):
            state[lane] ^= int.from_bytes(block[lane * 8 : lane * 8 + 8], "little")
        _keccak_permutation(state)
    return b"".join(lane.to_bytes(8, "little") for lane in state)[:32]


def _keccak_permutation(state: list[int]) -> None:
    for round_constant in _KECCAK_ROUND_CONSTANTS[:_KECCAK_ROUNDS]:
        column_parity = [
            state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20]
            for x in range(5)
        ]
        for x in range(5):
            mixed = column_parity[(x + 4) % 5] ^ _rotate_left(column_parity[(x + 1) % 5], 1)
            for y in range(5):
                state[x + 5 * y] ^= mixed
        moved = [0] * 25
        for x in range(5):
            for y in range(5):
                rotated = _rotate_left(state[x + 5 * y], _KECCAK_ROTATIONS[x][y])
                moved[y + 5 * ((2 * x + 3 * y) % 5)] = rotated
        for x in range(5):
            for y in range(5):
                state[x + 5 * y] = moved[x + 5 * y] ^ (
                    ~moved[(x + 1) % 5 + 5 * y] & moved[(x + 2) % 5 + 5 * y]
                )
        state[0] ^= round_constant


def _rotate_left(value: int, shift: int) -> int:
    shift %= 64
    return ((value << shift) | (value >> (64 - shift))) & _KECCAK_MASK
