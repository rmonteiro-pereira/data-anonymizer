"""Boundaries for identifier patterns that also work next to underscores.

In a regular expression `\b` treats the underscore as part of a word, so it finds no edge
in `wallet_0x...` or `cpf_52998224725`. Exports often tag identifiers that way, so an
identifier counts as separate from its surroundings unless a letter or digit touches it.
"""

# A character that is a letter or digit in any script, and not an underscore.
_ALPHANUMERIC = r"[^\W_]"

NOT_AFTER_ALPHANUMERIC = rf"(?<!{_ALPHANUMERIC})"
NOT_BEFORE_ALPHANUMERIC = rf"(?!{_ALPHANUMERIC})"


def bounded(body: str, *, ignore_case: bool = False) -> str:
    """Wrap an identifier pattern so that letters and digits cannot touch either end."""
    if ignore_case:
        body = f"(?i:{body})"
    return f"{NOT_AFTER_ALPHANUMERIC}{body}{NOT_BEFORE_ALPHANUMERIC}"
