# dataset_manager.py
#
# Utilities to create/append labeled datasets as CSV
# (used by Spectral Analysis and Vegetation Indices tabs)

from __future__ import annotations

import csv
import os
from typing import List, Sequence, Tuple


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _read_csv_header(path: str) -> List[str]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    return header or []


def save_dataset_rows(
    csv_path: str,
    header: Sequence[str],
    rows: Sequence[Sequence[object]],
    mode: str,
) -> None:
    """
    Save rows into a dataset CSV.

    Parameters
    ----------
    csv_path : str
        Output CSV path.
    header : Sequence[str]
        Column names (must include 'label' as last column).
    rows : Sequence[Sequence[object]]
        Data rows (must match header length).
    mode : str
        'create' or 'append'.
    """
    if mode not in ("create", "append"):
        raise ValueError("mode must be 'create' or 'append'")

    _ensure_parent_dir(csv_path)

    if mode == "create":
        # Overwrite/create new dataset
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(list(header))
            writer.writerows(rows)
        return

    # append mode
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    existing_header = _read_csv_header(csv_path)
    if list(existing_header) != list(header):
        raise ValueError(
            "This dataset has a different structure (columns do not match)."
        )

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)



