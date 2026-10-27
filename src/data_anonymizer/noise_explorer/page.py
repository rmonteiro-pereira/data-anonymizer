"""Render a noise exploration as one self-contained, interactive HTML page."""

import json
from importlib import resources
from pathlib import Path

from data_anonymizer.noise_explorer.sweep import NoiseExploration

_PLACEHOLDER = "/*__DATA__*/"


def render_page(exploration: NoiseExploration) -> str:
    """Return the page as HTML text; it needs no network and no server."""
    template = resources.files(__package__).joinpath("template.html").read_text("utf-8")
    payload = json.dumps(exploration.to_dict(), allow_nan=False)
    # Inside a <script> element, "</" could close the element early, so it is escaped.
    safe_payload = payload.replace("</", "<\\/")
    return template.replace(_PLACEHOLDER, safe_payload)


def save_page(exploration: NoiseExploration, path: Path) -> None:
    """Write the page to a file."""
    path.write_text(render_page(exploration), encoding="utf-8")
