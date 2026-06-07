"""Train, calibrate, evaluate, and save the bankruptcy-risk model."""

import json

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from src.config import (
    CALIBRATED_MODEL_PATH,
    FEATURE_NAMES,
    FEATURE_NAMES_PATH,
    IMPUTER_PATH,
    METRICS_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    ROW_ID_COLUMN,
    SCALER_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    ensure_output_directories,
)
from src.data import load_bankruptcy_data
from src.evaluate import evaluate_model
from src.explain import save_feature_importance
from src.features import fit_transform_features, transform_features


def create_model() -> CatBoostClassifier:
    """Create the main imbalance-aware CatBoost model."""
    return CatBoostClassifier(
        iterations=600,
        learning_rate=0.03,
        depth=5,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        auto_class_weights="Balanced",
        allow_writing_files=False,
    )


def calibrate_model(
    model: CatBoostClassifier,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[CalibratedClassifierCV, str]:
    """Fit isotonic calibration, falling back to sigmoid if necessary."""
    try:
        calibrated = CalibratedClassifierCV(
            estimator=model,
            method="isotonic",
            cv=3,
        )
        calibrated.fit(x_train, y_train)
        return calibrated, "isotonic"
    except Exception as isotonic_error:
        print(
            "Isotonic calibration failed; falling back to sigmoid. "
            f"Reason: {isotonic_error}"
        )
        calibrated = CalibratedClassifierCV(
            estimator=model,
            method="sigmoid",
            cv=3,
        )
        calibrated.fit(x_train, y_train)
        return calibrated, "sigmoid"


def main() -> None:
    """Run the complete single-dataset training workflow."""
    ensure_output_directories()
    data = load_bankruptcy_data()
    data.insert(0, ROW_ID_COLUMN, data.index.astype(int))

    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        stratify=data[TARGET_COLUMN],
        random_state=42,
    )
    train_data = train_data.sort_values(ROW_ID_COLUMN).reset_index(drop=True)
    test_data = test_data.sort_values(ROW_ID_COLUMN).reset_index(drop=True)
    train_data.to_csv(TRAIN_DATA_PATH, index=False)
    test_data.to_csv(TEST_DATA_PATH, index=False)

    x_train_raw = train_data[FEATURE_NAMES]
    y_train = train_data[TARGET_COLUMN]
    x_test_raw = test_data[FEATURE_NAMES]
    y_test = test_data[TARGET_COLUMN]

    x_train, imputer = fit_transform_features(x_train_raw)
    x_test = transform_features(x_test_raw, imputer)

    model = create_model()
    model.fit(x_train, y_train)
    model.training_medians_ = pd.Series(
        imputer.statistics_,
        index=FEATURE_NAMES,
        dtype=float,
    )

    calibrated_model, calibration_method = calibrate_model(
        model,
        x_train,
        y_train,
    )

    joblib.dump(model, MODEL_PATH)
    joblib.dump(calibrated_model, CALIBRATED_MODEL_PATH)
    joblib.dump(imputer, IMPUTER_PATH)
    joblib.dump(None, SCALER_PATH)
    joblib.dump(FEATURE_NAMES, FEATURE_NAMES_PATH)

    probabilities = calibrated_model.predict_proba(x_test)[:, 1]
    metrics = evaluate_model(
        y_true=y_test,
        risk_probability=probabilities,
        row_ids=test_data[ROW_ID_COLUMN],
    )
    metrics["calibration_method"] = calibration_method

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    save_feature_importance(model, FEATURE_NAMES)

    print(f"Training complete using {calibration_method} calibration.")
    print(f"Saved model artifacts to {MODEL_PATH.parent}")
    print(f"Saved evaluation reports to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
