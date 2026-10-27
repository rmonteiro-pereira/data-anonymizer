"""The manifest `oblitus apply-many` reads: which tables to protect together, and with what.

A manifest is a JSON file that lists related tables by name. Relative paths are read from the
manifest's own folder, so a folder of tables can be moved as a unit.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ManifestTable(BaseModel):
    """One table of a manifest: where it is read, where it is written, and its reviewed plan.

    Without a `plan`, one is proposed for the table.
    """

    model_config = ConfigDict(extra="forbid")

    input: Path
    output: Path
    plan: Path | None = None


class TableManifest(BaseModel):
    """The related tables protected in one call, keyed by table name."""

    model_config = ConfigDict(extra="forbid")

    tables: dict[str, ManifestTable] = Field(min_length=1)

    @classmethod
    def load(cls, path: Path) -> "TableManifest":
        """Read a manifest and resolve its relative paths against the manifest's folder."""
        manifest = cls.model_validate_json(path.read_text(encoding="utf-8"))
        return manifest._resolved_against(path.parent)

    def _resolved_against(self, folder: Path) -> "TableManifest":
        return TableManifest(
            tables={
                name: ManifestTable(
                    input=folder / table.input,
                    output=folder / table.output,
                    plan=None if table.plan is None else folder / table.plan,
                )
                for name, table in self.tables.items()
            }
        )
