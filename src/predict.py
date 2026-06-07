"""Artifact loading and bankruptcy-risk prediction helpers."""

from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CALIBRATED_MODEL_PATH,
    FEATURE_NAMES_PATH,
    IMPUTER_PATH,
    MODEL_PATH,
    SCALER_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TEST_PREDICTIONS_PATH,
)
from src.features import transform_features


def load_artifacts() -> dict[str, Any]:
    """Load trained model artifacts and fail with a useful training hint."""
    required_paths = {
        "model": MODEL_PATH,
        "calibrated_model": CALIBRATED_MODEL_PATH,
        "imputer": IMPUTER_PATH,
        "scaler": SCALER_PATH,
        "feature_names": FEATURE_NAMES_PATH,
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts: "
            + ", ".join(missing)
            + ". Run `python -m src.train` first."
        )

    artifacts = {
        name: joblib.load(path) for name, path in required_paths.items()
    }
    if TEST_PREDICTIONS_PATH.exists():
        artifacts["test_predictions"] = pd.read_csv(TEST_PREDICTIONS_PATH)
    else:
        artifacts["test_predictions"] = None
    return artifacts


def risk_bucket(probability: float) -> str:
    """Map a probability to a business-readable risk bucket."""
    if probability < 0.05:
        return "Low risk"
    if probability < 0.15:
        return "Medium risk"
    return "High risk"


def _to_feature_frame(
    input_features: Mapping[str, Any] | pd.Series | pd.DataFrame | Sequence[float],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    if isinstance(input_features, pd.DataFrame):
        frame = input_features.copy()
    elif isinstance(input_features, (pd.Series, Mapping)):
        frame = pd.DataFrame([dict(input_features)])
    else:
        values = np.asarray(input_features, dtype=float)
        if values.ndim != 1 or len(values) != len(feature_names):
            raise ValueError(
                f"Expected a one-dimensional sequence of {len(feature_names)} values."
            )
        frame = pd.DataFrame([values], columns=feature_names)

    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing input features: {', '.join(missing)}")
    return frame[list(feature_names)].apply(pd.to_numeric, errors="coerce")


def _risk_percentile(
    probability: float,
    test_predictions: pd.DataFrame | None,
) -> float | None:
    if test_predictions is None or test_predictions.empty:
        return None
    reference = test_predictions["risk_probability"].to_numpy(dtype=float)
    return float((reference <= probability).mean() * 100)


def predict_risk(
    input_features: Mapping[str, Any] | pd.Series | pd.DataFrame | Sequence[float],
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict calibrated bankruptcy risk for one company snapshot."""
    artifacts = artifacts or load_artifacts()
    feature_names = artifacts["feature_names"]
    features = _to_feature_frame(input_features, feature_names)
    transformed = transform_features(features, artifacts["imputer"])

    scaler = artifacts.get("scaler")
    if scaler is not None:
        transformed_values = scaler.transform(transformed)
    else:
        transformed_values = transformed

    probability = float(
        artifacts["calibrated_model"].predict_proba(transformed_values)[0, 1]
    )
    return {
        "probability": probability,
        "risk_percentile": _risk_percentile(
            probability, artifacts.get("test_predictions")
        ),
        "risk_bucket": risk_bucket(probability),
    }


def predict_for_existing_row(
    row_index: int,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict a saved test row, accepting either row_id or CSV row position."""
    if not Path(TEST_DATA_PATH).exists():
        raise FileNotFoundError(
            f"Processed test data not found at {TEST_DATA_PATH}. "
            "Run `python -m src.train` first."
        )

    test_data = pd.read_csv(TEST_DATA_PATH)
    if "row_id" in test_data.columns and row_index in set(test_data["row_id"]):
        row = test_data.loc[test_data["row_id"] == row_index].iloc[0]
    elif 0 <= row_index < len(test_data):
        row = test_data.iloc[row_index]
    else:
        raise IndexError(f"No test company found for row index {row_index}.")

    artifacts = artifacts or load_artifacts()
    result = predict_risk(row[artifacts["feature_names"]], artifacts)
    result["row_id"] = int(row.get("row_id", row_index))
    result["true_label"] = int(row[TARGET_COLUMN])
    return result
