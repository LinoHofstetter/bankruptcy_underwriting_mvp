"""Model evaluation and report generation."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    CALIBRATION_CURVE_PATH,
    METRICS_PATH,
    TEST_PREDICTIONS_PATH,
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def evaluate_model(
    y_true: pd.Series | np.ndarray,
    risk_probability: np.ndarray,
    row_ids: pd.Series | np.ndarray,
    metrics_path: Path = METRICS_PATH,
    predictions_path: Path = TEST_PREDICTIONS_PATH,
    calibration_path: Path = CALIBRATION_CURVE_PATH,
) -> dict[str, Any]:
    """Calculate standard and top-decile underwriting triage metrics."""
    y_true_array = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(risk_probability, dtype=float)
    row_id_array = np.asarray(row_ids, dtype=int)

    if not (
        len(y_true_array) == len(probabilities) == len(row_id_array)
        and len(y_true_array) > 0
    ):
        raise ValueError("Evaluation arrays must have the same non-zero length.")

    predicted_at_half = (probabilities >= 0.5).astype(int)

    top_count = max(1, math.ceil(0.10 * len(probabilities)))
    top_indices = np.argsort(-probabilities, kind="stable")[:top_count]
    predicted_top_10 = np.zeros(len(probabilities), dtype=int)
    predicted_top_10[top_indices] = 1
    top_10_threshold = float(probabilities[top_indices].min())

    positive_count = int(y_true_array.sum())
    positives_in_top_10 = int(y_true_array[top_indices].sum())
    recall_top_10 = (
        positives_in_top_10 / positive_count if positive_count else 0.0
    )
    bankruptcy_rate_top_10 = positives_in_top_10 / top_count

    risk_percentiles = (
        pd.Series(probabilities).rank(method="average", pct=True).to_numpy() * 100
    )

    metrics = {
        "roc_auc": roc_auc_score(y_true_array, probabilities),
        "pr_auc": average_precision_score(y_true_array, probabilities),
        "accuracy_at_0_5": accuracy_score(y_true_array, predicted_at_half),
        "precision_at_0_5": precision_score(
            y_true_array, predicted_at_half, zero_division=0
        ),
        "recall_at_0_5": recall_score(
            y_true_array, predicted_at_half, zero_division=0
        ),
        "f1_at_0_5": f1_score(y_true_array, predicted_at_half, zero_division=0),
        "brier_score": brier_score_loss(y_true_array, probabilities),
        "confusion_matrix_at_0_5": confusion_matrix(
            y_true_array, predicted_at_half, labels=[0, 1]
        ),
        "top_10_percent_threshold": top_10_threshold,
        "top_10_percent_company_count": top_count,
        "confusion_matrix_top_10_percent": confusion_matrix(
            y_true_array, predicted_top_10, labels=[0, 1]
        ),
        "recall_at_top_10_percent_risk": recall_top_10,
        "bankruptcy_rate_top_10_percent": bankruptcy_rate_top_10,
        "baseline_bankruptcy_rate": float(y_true_array.mean()),
        "test_company_count": len(y_true_array),
        "test_bankruptcy_count": positive_count,
    }
    metrics = _json_ready(metrics)

    metrics_path = Path(metrics_path)
    predictions_path = Path(predictions_path)
    calibration_path = Path(calibration_path)
    for path in (metrics_path, predictions_path, calibration_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    predictions = pd.DataFrame(
        {
            "row_id": row_id_array,
            "y_true": y_true_array,
            "risk_probability": probabilities,
            "risk_percentile": risk_percentiles,
            "predicted_label_0_5": predicted_at_half,
            "predicted_label_top_10_percent": predicted_top_10,
        }
    )
    predictions.to_csv(predictions_path, index=False)

    fraction_positive, mean_predicted = calibration_curve(
        y_true_array,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )
    calibration = pd.DataFrame(
        {
            "mean_predicted_probability": mean_predicted,
            "fraction_of_positives": fraction_positive,
        }
    )
    calibration.to_csv(calibration_path, index=False)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics
