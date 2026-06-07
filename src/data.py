"""Load and clean the Polish companies bankruptcy ARFF data."""

from pathlib import Path
from typing import Any

import arff
import numpy as np
import pandas as pd

from src.config import FEATURE_NAMES, RAW_DATA_PATH, TARGET_COLUMN


TARGET_NAME_CANDIDATES = {
    "class",
    "target",
    "label",
    "bankrupt",
    "bankruptcy",
    "bankruptcy_label",
}


def _decode_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _find_target_column(columns: list[str]) -> str:
    for column in columns:
        if column.strip().lower() in TARGET_NAME_CANDIDATES:
            return column
    return columns[-1]


def _parse_target(value: Any) -> int:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = value.strip()
    if value in (None, "", "?"):
        raise ValueError("The target column contains a missing value.")

    numeric_value = float(value)
    if not numeric_value.is_integer() or int(numeric_value) not in (0, 1):
        raise ValueError(f"Unexpected target label: {value!r}")
    return int(numeric_value)


def load_bankruptcy_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load an ARFF file and return X1-X64 plus a binary target column."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"ARFF dataset not found at {path}. "
            "Place 5year.arff in data/raw/ before training."
        )

    with path.open("r", encoding="utf-8") as file:
        dataset = arff.load(file)

    attributes = dataset.get("attributes", [])
    rows = dataset.get("data", [])
    if not attributes or not rows:
        raise ValueError(f"The ARFF file at {path} has no attributes or data rows.")

    original_columns = [_decode_name(attribute[0]) for attribute in attributes]
    frame = pd.DataFrame(rows, columns=original_columns)
    frame = frame.replace({"?": np.nan, b"?": np.nan})

    target_column = _find_target_column(original_columns)
    feature_columns = [
        column for column in original_columns if column != target_column
    ]
    if len(feature_columns) != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected 64 feature columns, found {len(feature_columns)} in {path}."
        )

    features = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan).astype(float)
    features.columns = FEATURE_NAMES

    target = frame[target_column].map(_parse_target).astype(int)
    cleaned = features.copy()
    cleaned[TARGET_COLUMN] = target
    return cleaned
