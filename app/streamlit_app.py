"""Interactive Streamlit app for the bankruptcy-risk MVP."""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    CALIBRATION_CURVE_PATH,
    FEATURE_DESCRIPTIONS,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    TEST_DATA_PATH,
    TEST_PREDICTIONS_PATH,
)
from src.explain import explain_single_prediction  # noqa: E402
from src.features import transform_features  # noqa: E402
from src.predict import load_artifacts, predict_risk, risk_bucket  # noqa: E402


st.set_page_config(
    page_title="Historical Company Risk Underwriting MVP",
    page_icon=None,
    layout="wide",
)


@st.cache_resource
def cached_artifacts():
    return load_artifacts()


@st.cache_data
def load_app_data():
    required = [
        TEST_DATA_PATH,
        TEST_PREDICTIONS_PATH,
        METRICS_PATH,
        FEATURE_IMPORTANCE_PATH,
        CALIBRATION_CURVE_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing generated files: "
            + ", ".join(missing)
            + ". Run `python -m src.train` first."
        )

    test_data = pd.read_csv(TEST_DATA_PATH)
    predictions = pd.read_csv(TEST_PREDICTIONS_PATH)
    companies = test_data.merge(
        predictions,
        on="row_id",
        how="inner",
        validate="one_to_one",
    )
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    feature_importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    calibration = pd.read_csv(CALIBRATION_CURVE_PATH)
    return companies, metrics, feature_importance, calibration


def probability_card(probability: float, bucket: str, percentile: float | None):
    col1, col2, col3 = st.columns(3)
    col1.metric("Bankruptcy probability", f"{probability:.1%}")
    col2.metric("Risk bucket", bucket)
    col3.metric(
        "Risk percentile",
        "Unavailable" if percentile is None else f"{percentile:.1f}th",
    )


def explanation_for_row(row: pd.Series, artifacts, top_k: int = 8):
    feature_names = artifacts["feature_names"]
    raw_features = pd.DataFrame([row[feature_names]], columns=feature_names)
    imputed = transform_features(raw_features, artifacts["imputer"]).iloc[0]
    medians = pd.Series(
        artifacts["imputer"].statistics_,
        index=feature_names,
        dtype=float,
    )
    return explain_single_prediction(
        imputed,
        artifacts["model"],
        FEATURE_DESCRIPTIONS,
        top_k=top_k,
        training_medians=medians,
    )


def render_inspect_company(companies: pd.DataFrame, artifacts):
    st.subheader("Inspect test company")
    row_ids = companies["row_id"].astype(int).tolist()
    selected_id = st.selectbox("Test company row_id", row_ids)
    row = companies.loc[companies["row_id"] == selected_id].iloc[0]

    st.metric(
        "Actual outcome",
        "Bankrupt" if int(row["y_true"]) == 1 else "Did not go bankrupt",
    )
    result = {
        "probability": float(row["risk_probability"]),
        "risk_percentile": float(row["risk_percentile"]),
    }
    result["risk_bucket"] = risk_bucket(result["probability"])
    probability_card(
        result["probability"],
        result["risk_bucket"],
        result["risk_percentile"],
    )

    st.markdown("#### Important feature comparison")
    explanation = explanation_for_row(row, artifacts)
    st.dataframe(
        explanation,
        width="stretch",
        hide_index=True,
        column_config={
            "company_value": st.column_config.NumberColumn(format="%.4g"),
            "training_median": st.column_config.NumberColumn(format="%.4g"),
            "global_importance": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    with st.expander("All financial ratio values"):
        feature_names = artifacts["feature_names"]
        all_values = pd.DataFrame(
            {
                "feature": feature_names,
                "description": [
                    FEATURE_DESCRIPTIONS[name] for name in feature_names
                ],
                "value": [row[name] for name in feature_names],
            }
        )
        st.dataframe(all_values, width="stretch", hide_index=True)


def render_manual_input(
    companies: pd.DataFrame,
    feature_importance: pd.DataFrame,
    artifacts,
):
    st.subheader("Manual feature input")
    st.caption(
        "Choose a test company as a complete 64-feature template, then edit "
        "the eight most important ratios."
    )
    row_ids = companies["row_id"].astype(int).tolist()
    selected_id = st.selectbox("Template company row_id", row_ids)
    template = companies.loc[companies["row_id"] == selected_id].iloc[0]
    feature_names = artifacts["feature_names"]
    modified = template[feature_names].copy()
    medians = pd.Series(
        artifacts["imputer"].statistics_,
        index=feature_names,
        dtype=float,
    )

    top_features = feature_importance.head(8)["feature"].tolist()
    columns = st.columns(2)
    for index, feature in enumerate(top_features):
        template_value = template[feature]
        initial_value = (
            float(medians[feature])
            if pd.isna(template_value)
            else float(template_value)
        )
        with columns[index % 2]:
            modified[feature] = st.number_input(
                f"{feature}: {FEATURE_DESCRIPTIONS[feature]}",
                value=initial_value,
                format="%.6g",
                key=f"manual_{selected_id}_{feature}",
            )

    result = predict_risk(modified, artifacts)
    st.markdown("#### Modified-company prediction")
    probability_card(
        result["probability"],
        result["risk_bucket"],
        result["risk_percentile"],
    )

    st.markdown("#### Important feature comparison")
    explanation = explanation_for_row(modified, artifacts)
    st.dataframe(explanation, width="stretch", hide_index=True)


def render_performance(
    metrics: dict,
    feature_importance: pd.DataFrame,
    calibration: pd.DataFrame,
):
    st.subheader("Model performance")
    columns = st.columns(3)
    columns[0].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    columns[1].metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
    columns[2].metric("Brier score", f"{metrics['brier_score']:.4f}")

    columns = st.columns(3)
    columns[0].metric(
        "Recall in top 10% risk",
        f"{metrics['recall_at_top_10_percent_risk']:.1%}",
    )
    columns[1].metric(
        "Bankruptcy rate in top 10%",
        f"{metrics['bankruptcy_rate_top_10_percent']:.1%}",
    )
    columns[2].metric(
        "Baseline bankruptcy rate",
        f"{metrics['baseline_bankruptcy_rate']:.1%}",
    )

    left, right = st.columns(2)
    with left:
        matrix = pd.DataFrame(
            metrics["confusion_matrix_at_0_5"],
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )
        figure = px.imshow(
            matrix,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Confusion matrix at threshold 0.5",
        )
        st.plotly_chart(figure, width="stretch")

    with right:
        top_importance = feature_importance.head(20).sort_values("importance")
        figure = px.bar(
            top_importance,
            x="importance",
            y="feature",
            orientation="h",
            hover_data=["description"],
            title="Top 20 global feature importances",
        )
        st.plotly_chart(figure, width="stretch")

    calibration_figure = go.Figure()
    calibration_figure.add_trace(
        go.Scatter(
            x=calibration["mean_predicted_probability"],
            y=calibration["fraction_of_positives"],
            mode="lines+markers",
            name="Calibrated model",
        )
    )
    calibration_figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash"},
            name="Perfect calibration",
        )
    )
    calibration_figure.update_layout(
        title="Calibration curve",
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed bankruptcy rate",
    )
    st.plotly_chart(calibration_figure, width="stretch")


def main():
    st.title("Historical Company Risk Underwriting MVP")
    st.write(
        "This prototype predicts one-year bankruptcy risk from historical "
        "company financial ratios. It is a proxy for commercial-insurance "
        "underwriting risk. It is not a production underwriting system."
    )

    try:
        artifacts = cached_artifacts()
        companies, metrics, feature_importance, calibration = load_app_data()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.code("python -m src.train")
        st.stop()

    mode = st.sidebar.radio(
        "Select mode",
        [
            "Inspect test company",
            "Manual feature input",
            "Model performance",
        ],
    )
    if mode == "Inspect test company":
        render_inspect_company(companies, artifacts)
    elif mode == "Manual feature input":
        render_manual_input(companies, feature_importance, artifacts)
    else:
        render_performance(metrics, feature_importance, calibration)


if __name__ == "__main__":
    main()
