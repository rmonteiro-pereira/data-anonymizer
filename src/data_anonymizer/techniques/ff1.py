"""FF1 format-preserving encryption from NIST SP 800-38G.

FF1 encrypts a string of numerals in a given radix into another string of the same
length and radix: ten digits stay ten digits. It is the format-preserving mode NIST
still recommends; FF3-1 was withdrawn after published attacks.
"""

import math

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

_ROUNDS = 10
_BLOCK = 16

# NIST requires radix ** length >= one million so the domain is not trivially small.
_MIN_DOMAIN = 1_000_000


class FF1:
    """An FF1 cipher for one AES key, one radix and one tweak."""

    def __init__(self, key: bytes, radix: int, tweak: bytes = b"") -> None:
        if len(key) not in (16, 24, 32):
            raise ValueError("FF1 needs a 128, 192 or 256-bit AES key")
        if not 2 <= radix <= 2**16:
            raise ValueError("FF1 radix must be between 2 and 65536")
        self._cipher = Cipher(algorithms.AES(key), modes.ECB())
        self._radix = radix
        self._tweak = tweak
        self.min_length = math.ceil(math.log(_MIN_DOMAIN, radix))

    def encrypt(self, numerals: list[int]) -> list[int]:
        """Encrypt a list of numerals, each between 0 and radix - 1."""
        return self._run(numerals, decrypt=False)

    def decrypt(self, numerals: list[int]) -> list[int]:
        """Reverse `encrypt`."""
        return self._run(numerals, decrypt=True)

    def _run(self, numerals: list[int], *, decrypt: bool) -> list[int]:
        length = len(numerals)
        if length < self.min_length:
            raise ValueError(f"FF1 needs at least {self.min_length} numerals, got {length}")

        left_length = length // 2
        right_length = length - left_length
        left = numerals[:left_length]
        right = numerals[left_length:]
        byte_count = math.ceil(math.ceil(right_length * math.log2(self._radix)) / 8)
        output_bytes = 4 * math.ceil(byte_count / 4) + 4
        header = self._header(length, left_length)

        rounds = reversed(range(_ROUNDS)) if decrypt else range(_ROUNDS)
        for round_index in rounds:
            half_length = left_length if round_index % 2 == 0 else right_length
            modulus = self._radix**half_length
            if decrypt:
                round_value = self._round_value(header, round_index, left, byte_count, output_bytes)
                shifted = (_number(right, self._radix) - round_value) % modulus
                left, right = _numerals(shifted, self._radix, half_length), left
            else:
                round_value = self._round_value(
                    header, round_index, right, byte_count, output_bytes
                )
                shifted = (_number(left, self._radix) + round_value) % modulus
                left, right = right, _numerals(shifted, self._radix, half_length)
        return left + right

    def _header(self, length: int, left_length: int) -> bytes:
        return (
            bytes([1, 2, 1])
            + self._radix.to_bytes(3, "big")
            + bytes([10, left_length % 256])
            + length.to_bytes(4, "big")
            + len(self._tweak).to_bytes(4, "big")
        )

    def _round_value(
        self,
        header: bytes,
        round_index: int,
        half: list[int],
        byte_count: int,
        output_bytes: int,
    ) -> int:
        padding = bytes((-len(self._tweak) - byte_count - 1) % _BLOCK)
        block_input = (
            self._tweak
            + padding
            + bytes([round_index])
            + _number(half, self._radix).to_bytes(byte_count, "big")
        )
        mac = self._cbc_mac(header + block_input)
        stream = mac
        counter = 1
        while len(stream) < output_bytes:
            stream += self._encrypt_block(_xor(mac, counter.to_bytes(_BLOCK, "big")))
            counter += 1
        return int.from_bytes(stream[:output_bytes], "big")

    def _cbc_mac(self, data: bytes) -> bytes:
        state = bytes(_BLOCK)
        for start in range(0, len(data), _BLOCK):
            state = self._encrypt_block(_xor(state, data[start : start + _BLOCK]))
        return state

    def _encrypt_block(self, block: bytes) -> bytes:
        encryptor = self._cipher.encryptor()
        return encryptor.update(block) + encryptor.finalize()


def _xor(first: bytes, second: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(first, second, strict=True))


def _number(numerals: list[int], radix: int) -> int:
    value = 0
    for numeral in numerals:
        value = value * radix + numeral
    return value


def _numerals(value: int, radix: int, length: int) -> list[int]:
    digits = [0] * length
    for position in reversed(range(length)):
        value, digits[position] = divmod(value, radix)
    return digits
