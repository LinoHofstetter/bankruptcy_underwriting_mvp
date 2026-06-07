"""Simple, robust model explanations."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_DESCRIPTIONS, FEATURE_IMPORTANCE_PATH


def get_feature_importance(
    model: object,
    feature_names: Sequence[str],
    feature_descriptions: Mapping[str, str] = FEATURE_DESCRIPTIONS,
) -> pd.DataFrame:
    """Return CatBoost global feature importance with readable descriptions."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise ValueError("The supplied model does not expose feature_importances_.")
    if len(importances) != len(feature_names):
        raise ValueError("Feature names do not match the model importance vector.")

    importance = pd.DataFrame(
        {
            "feature": list(feature_names),
            "description": [
                feature_descriptions.get(name, name) for name in feature_names
            ],
            "importance": np.asarray(importances, dtype=float),
        }
    )
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def save_feature_importance(
    model: object,
    feature_names: Sequence[str],
    output_path: Path = FEATURE_IMPORTANCE_PATH,
) -> pd.DataFrame:
    """Save global feature importance to a CSV report."""
    importance = get_feature_importance(model, feature_names)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_path, index=False)
    return importance


def explain_single_prediction(
    x_row: pd.Series | Mapping[str, float],
    model: object,
    feature_descriptions: Mapping[str, str],
    top_k: int = 8,
    training_medians: pd.Series | Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Compare important values for one company with training-set medians."""
    row = pd.Series(x_row, dtype=float)
    feature_names = list(row.index)
    importance = get_feature_importance(
        model,
        feature_names,
        feature_descriptions,
    ).head(top_k)

    if training_medians is None:
        training_medians = getattr(model, "training_medians_", None)
    if training_medians is None:
        raise ValueError(
            "Training medians are required for per-company explanations."
        )
    medians = pd.Series(training_medians, dtype=float)

    records = []
    for item in importance.itertuples(index=False):
        company_value = float(row[item.feature])
        training_median = float(medians[item.feature])
        if np.isnan(company_value):
            company_value = training_median
        tolerance = max(abs(training_median) * 0.05, 1e-8)
        difference = company_value - training_median
        if abs(difference) <= tolerance:
            direction = "near median"
        elif difference > 0:
            direction = "above median"
        else:
            direction = "below median"

        records.append(
            {
                "feature": item.feature,
                "description": item.description,
                "company_value": company_value,
                "training_median": training_median,
                "direction": direction,
                "global_importance": float(item.importance),
            }
        )

    return pd.DataFrame(records)
