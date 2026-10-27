"""Tokenization: replace values with random tokens, and keep the mapping in a vault.

A token is drawn from `os.urandom`, never from a seeded generator, so it carries no
information about the value and cannot be recomputed from a key. The mapping lives in a
`TokenVault` the caller passes in. The same value always gets the same token from one vault,
so joins survive, and `detokenize` reads the vault to bring the values back.
"""

import secrets
import string
from collections.abc import Callable

import narwhals as nw

from oblitus.table import Table
from oblitus.techniques.context import TechniqueContext
from oblitus.techniques.value_mapping import lookup_distinct
from oblitus.token_vault import TokenVault

_TOKEN_CHARACTERS = string.ascii_letters + string.digits

# Short values leave few tokens to pick from, so a draw may hit a token already taken.
_MAX_DRAWS_PER_VALUE = 200


def tokenize(
    table: Table,
    column: str,
    context: TechniqueContext,
    *,
    prefix: str = "",
    length: int = 16,
    preserve_format: bool = False,
    namespace: str = "",
) -> nw.Expr:
    """Replace each value with a random token, remembered in the vault.

    By default a token is `prefix` followed by `length` random letters and digits. With
    `preserve_format`, a token keeps the value's length and character classes (digits stay
    digits, letters keep their case, separators stay in place), and `length` is ignored.
    Equal values get equal tokens from the same vault, so joins and counts survive. A
    `namespace` gives a column its own tokens, so the same value in two columns does not
    share one. Tokens link back to a person through the vault, so this is pseudonymization.
    """
    vault = _vault_of(context)
    scheme = _scheme(prefix, length, preserve_format, namespace)
    if not preserve_format and length < 1:
        raise ValueError("length must be at least 1")

    def token_of(text: str) -> str:
        def draw() -> str:
            return _new_token(text, prefix, length, preserve_format)

        return _token_for(vault, scheme, text, draw)

    return lookup_distinct(table, column, token_of)


def detokenize(
    table: Table,
    column: str,
    context: TechniqueContext,
    *,
    prefix: str = "",
    length: int = 16,
    preserve_format: bool = False,
    namespace: str = "",
) -> nw.Expr:
    """Reverse `tokenize` with the vault; the parameters are accepted and ignored.

    A token the vault does not know raises an error rather than staying in the column.
    """
    vault = _vault_of(context)

    def value_of(token: str) -> str:
        value = vault.value_for(token)
        if value is None:
            raise ValueError("a value is not a token in this vault")
        return value

    return lookup_distinct(table, column, value_of)


def _vault_of(context: TechniqueContext) -> TokenVault:
    if context.vault is None:
        raise ValueError("tokenization needs a vault: pass vault= to apply and reveal")
    return context.vault


def _scheme(prefix: str, length: int, preserve_format: bool, namespace: str) -> str:
    # Two columns tokenized the same way share tokens, and that is what keeps joins working.
    shape = "format" if preserve_format else f"random{length}"
    return f"{namespace}|{shape}|{prefix}"


def _token_for(vault: TokenVault, scheme: str, text: str, draw_token: Callable[[], str]) -> str:
    for _attempt in range(_MAX_DRAWS_PER_VALUE):
        existing = vault.token_for(scheme, text)
        if existing is not None:
            return existing
        candidate = draw_token()
        if vault.add(scheme, text, candidate):
            return candidate
    raise ValueError("could not find an unused token; use longer tokens or a prefix")


def _new_token(text: str, prefix: str, length: int, preserve_format: bool) -> str:
    if preserve_format:
        return prefix + _same_format_token(text)
    return prefix + "".join(secrets.choice(_TOKEN_CHARACTERS) for _ in range(length))


def _same_format_token(text: str) -> str:
    if not any(character.isalnum() for character in text):
        raise ValueError("a value with no letters or digits cannot get a format-preserving token")
    return "".join(_random_like(character) for character in text)


def _random_like(character: str) -> str:
    if character.isdigit():
        return secrets.choice(string.digits)
    if character.isalpha():
        letters = string.ascii_uppercase if character.isupper() else string.ascii_lowercase
        return secrets.choice(letters)
    return character
