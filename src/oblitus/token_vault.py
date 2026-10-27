"""The vault behind tokenization: where each token and the value it stands for are kept.

A token is random, so nothing in it can be turned back into the value. The vault is the only
place that holds the mapping, which makes it exactly as sensitive as the original data:
whoever can read it can reveal every token. Keep it apart from the released table, and
restrict who may read it.

The caller passes a vault in at apply and reveal time, so a plan never stores one and stays
plain JSON. `MemoryVault` lives and dies with the process. `SqliteVault` keeps the mapping in
a file, so tokens stay consistent between runs and can be revealed later.
"""

import os
import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self


class TokenVault(Protocol):
    """What tokenization needs from a store of token and value pairs.

    A `scheme` names how a token was made (its format and namespace). The same value gets one
    token per scheme, and every token is unique across the whole vault, so `value_for` is
    never ambiguous.
    """

    def token_for(self, scheme: str, value: str) -> str | None:
        """Return the token already stored for this value, or None."""
        ...

    def add(self, scheme: str, value: str, token: str) -> bool:
        """Store a pair. Return False, storing nothing, when the token or the value is taken."""
        ...

    def value_for(self, token: str) -> str | None:
        """Return the value a token stands for, or None when the vault does not know it."""
        ...


class MemoryVault:
    """A vault held in memory, for one process: tests, notebooks and a single run."""

    def __init__(self) -> None:
        self._tokens: dict[tuple[str, str], str] = {}
        self._values: dict[str, str] = {}

    def token_for(self, scheme: str, value: str) -> str | None:
        """Return the token already stored for this value, or None."""
        return self._tokens.get((scheme, value))

    def add(self, scheme: str, value: str, token: str) -> bool:
        """Store a pair. Return False, storing nothing, when the token or the value is taken."""
        if token in self._values or (scheme, value) in self._tokens:
            return False
        self._tokens[(scheme, value)] = token
        self._values[token] = value
        return True

    def value_for(self, token: str) -> str | None:
        """Return the value a token stands for, or None when the vault does not know it."""
        return self._values.get(token)

    def __len__(self) -> int:
        return len(self._values)


class SqliteVault:
    """A vault kept in a SQLite file, created readable by its owner only.

    Use it as a context manager, or call `close`. Every stored pair is committed at once, so
    a run that stops half way never loses tokens that were already handed out.
    """

    def __init__(self, path: Path) -> None:
        _create_private_file(path)
        self._connection = sqlite3.connect(path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS tokens ("
            "token TEXT PRIMARY KEY, scheme TEXT NOT NULL, value TEXT NOT NULL, "
            "UNIQUE (scheme, value))"
        )
        self._connection.commit()

    def token_for(self, scheme: str, value: str) -> str | None:
        """Return the token already stored for this value, or None."""
        row = self._connection.execute(
            "SELECT token FROM tokens WHERE scheme = ? AND value = ?", (scheme, value)
        ).fetchone()
        return None if row is None else str(row[0])

    def add(self, scheme: str, value: str, token: str) -> bool:
        """Store a pair. Return False, storing nothing, when the token or the value is taken."""
        try:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO tokens (token, scheme, value) VALUES (?, ?, ?)",
                    (token, scheme, value),
                )
        except sqlite3.IntegrityError:
            return False
        return True

    def value_for(self, token: str) -> str | None:
        """Return the value a token stands for, or None when the vault does not know it."""
        row = self._connection.execute(
            "SELECT value FROM tokens WHERE token = ?", (token,)
        ).fetchone()
        return None if row is None else str(row[0])

    def close(self) -> None:
        """Close the file."""
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _create_private_file(path: Path) -> None:
    # Owner-only permissions from the first byte: the file maps tokens to real values.
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.close(descriptor)
