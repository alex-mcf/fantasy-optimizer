"""Replace archive files only once their replacement is complete on disk.

Importers download from the network and overwrite a clean historical archive. A
crash or a truncated download partway through a direct write leaves a corrupt
CSV in place of good data, so every import writes a temporary file in the
destination directory and renames it over the target, which is atomic on the
same filesystem.
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd


def _replace_with_temp(destination: Path, write) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with temporary as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary.name, destination)
    except BaseException:
        Path(temporary.name).unlink(missing_ok=True)
        raise
    return destination


def write_csv(frame: pd.DataFrame, destination: Path, **kwargs) -> Path:
    """Write a CSV that either fully replaces the old file or leaves it intact."""
    return _replace_with_temp(
        destination, lambda handle: frame.to_csv(handle, **kwargs)
    )


def write_text(text: str, destination: Path) -> Path:
    """Write a text file (metadata sidecars) with the same replace guarantee."""
    return _replace_with_temp(destination, lambda handle: handle.write(text))
