"""Numerical preprocessing for the bankruptcy model."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def create_imputer() -> SimpleImputer:
    """Create the train-fitted median imputer used by the model."""
    return SimpleImputer(strategy="median", keep_empty_features=True)


def fit_transform_features(
    training_features: pd.DataFrame,
) -> tuple[pd.DataFrame, SimpleImputer]:
    """Fit median imputation on training data and return transformed features."""
    clean_features = training_features.replace([np.inf, -np.inf], np.nan)
    imputer = create_imputer()
    transformed = imputer.fit_transform(clean_features)
    transformed_frame = pd.DataFrame(
        transformed,
        columns=training_features.columns,
        index=training_features.index,
    )
    return transformed_frame, imputer


def transform_features(
    features: pd.DataFrame,
    imputer: SimpleImputer,
) -> pd.DataFrame:
    """Apply a fitted imputer while preserving DataFrame labels."""
    clean_features = features.replace([np.inf, -np.inf], np.nan)
    transformed = imputer.transform(clean_features)
    return pd.DataFrame(
        transformed,
        columns=features.columns,
        index=features.index,
    )
