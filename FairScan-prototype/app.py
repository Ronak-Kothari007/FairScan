from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from fpdf import FPDF
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="FairScan",
    page_icon="FS",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
    :root {
        color-scheme: dark;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(71, 180, 180, 0.15), transparent 30rem),
            linear-gradient(135deg, #0d1117 0%, #10151f 45%, #111827 100%);
        color: #e5edf7;
    }

    [data-testid="stSidebar"] {
        background: #0a0f18;
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: 0 !important;
    }

    .subtle {
        color: #9fb0c5;
        font-size: 0.98rem;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin: 0.75rem 0 1rem;
    }

    .metric-card {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 8px;
        padding: 1rem;
        background: rgba(15, 23, 42, 0.76);
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.22);
        min-height: 128px;
    }

    .metric-label {
        color: #aebed1;
        font-size: 0.86rem;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 760;
        line-height: 1.05;
        margin-bottom: 0.55rem;
    }

    .metric-band {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.22rem 0.62rem;
        font-size: 0.8rem;
        font-weight: 700;
        color: #061018;
    }

    .band-green { background: #47d18c; }
    .band-yellow { background: #facc15; }
    .band-red { background: #fb7185; }

    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.45rem;
    }

    .finding-pill {
        border-radius: 999px;
        padding: 0.42rem 0.7rem;
        font-size: 0.86rem;
        font-weight: 750;
        color: #07111f;
    }

    .panel {
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 8px;
        padding: 1rem;
        background: rgba(15, 23, 42, 0.58);
    }

    @media (max-width: 900px) {
        .metric-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
"""


def make_adult_income_sample(rows: int = 900, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic Adult Income-style demo dataset."""
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Female", "Male"], rows, p=[0.47, 0.53])
    race = rng.choice(
        ["White", "Black", "Asian-Pac-Islander", "Other"],
        rows,
        p=[0.67, 0.16, 0.10, 0.07],
    )
    age = rng.integers(19, 68, rows)
    education = rng.choice(
        ["HS-grad", "Some-college", "Bachelors", "Masters", "Doctorate"],
        rows,
        p=[0.32, 0.28, 0.25, 0.12, 0.03],
    )
    workclass = rng.choice(
        ["Private", "Self-emp", "Government"],
        rows,
        p=[0.72, 0.12, 0.16],
    )
    marital_status = rng.choice(
        ["Never-married", "Married", "Separated", "Widowed"],
        rows,
        p=[0.34, 0.48, 0.13, 0.05],
    )
    occupation = rng.choice(
        ["Admin", "Craft-repair", "Exec-managerial", "Sales", "Tech-support", "Service"],
        rows,
        p=[0.18, 0.17, 0.18, 0.20, 0.12, 0.15],
    )
    hours_per_week = np.clip(rng.normal(39, 9, rows).round(), 18, 70).astype(int)

    education_boost = pd.Series(education).map(
        {
            "HS-grad": -0.45,
            "Some-college": -0.18,
            "Bachelors": 0.45,
            "Masters": 0.75,
            "Doctorate": 1.05,
        }
    ).to_numpy()
    occupation_boost = pd.Series(occupation).map(
        {
            "Admin": -0.08,
            "Craft-repair": 0.05,
            "Exec-managerial": 0.65,
            "Sales": 0.15,
            "Tech-support": 0.35,
            "Service": -0.35,
        }
    ).to_numpy()

    historical_bias = (
        np.where(gender == "Male", 0.38, -0.18)
        + np.where(race == "White", 0.22, -0.14)
    )
    logit = (
        -2.15
        + 0.038 * (age - 35)
        + 0.035 * (hours_per_week - 38)
        + education_boost
        + occupation_boost
        + np.where(marital_status == "Married", 0.35, -0.08)
        + historical_bias
        + rng.normal(0, 0.65, rows)
    )
    probability = 1 / (1 + np.exp(-logit))
    income = np.where(rng.random(rows) < probability, ">50K", "<=50K")

    return pd.DataFrame(
        {
            "age": age,
            "workclass": workclass,
            "education": education,
            "marital_status": marital_status,
            "occupation": occupation,
            "race": race,
            "gender": gender,
            "hours_per_week": hours_per_week,
            "income": income,
        }
    )


def normalize_binary_target(series: pd.Series, positive_value: Any) -> pd.Series:
    return (series.astype(str) == str(positive_value)).astype(int)


def default_positive_value(values: list[Any]) -> Any:
    preferred = [">50K", "1", "yes", "true", "approved", "approve", "positive"]
    value_lookup = {str(value).strip().lower(): value for value in values}
    for candidate in preferred:
        if candidate in value_lookup:
            return value_lookup[candidate]
    return values[-1]


def make_model(X: pd.DataFrame) -> Pipeline:
    categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_columns = [column for column in X.columns if column not in categorical_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1_000, class_weight="balanced")),
        ]
    )


def calibration_gap(y_true: pd.Series | np.ndarray, y_score: pd.Series | np.ndarray) -> float:
    """Absolute difference between observed and predicted positive rate."""
    true_mean = np.asarray(y_true, dtype=float).mean()
    score_mean = np.asarray(y_score, dtype=float).mean()
    return float(abs(score_mean - true_mean))


def group_calibration_difference(
    y_true: pd.Series,
    probabilities: np.ndarray,
    sensitive_features: pd.Series,
) -> tuple[float, pd.Series]:
    frame = MetricFrame(
        metrics={"calibration_gap": calibration_gap},
        y_true=y_true,
        y_pred=probabilities,
        sensitive_features=sensitive_features,
    )
    by_group = frame.by_group["calibration_gap"].fillna(0)
    if by_group.empty:
        return 0.0, by_group
    return float(by_group.max() - by_group.min()), by_group


def score_band(value: float) -> tuple[str, str, str]:
    if value > 0.10:
        return "High Bias", "band-red", "#fb7185"
    if value > 0.05:
        return "Moderate", "band-yellow", "#facc15"
    return "Fair", "band-green", "#47d18c"


def verdict_from_metrics(metrics: dict[str, float]) -> str:
    worst = max(metrics.values()) if metrics else 0
    if worst > 0.10:
        return "High Risk"
    if worst > 0.05:
        return "Needs Review"
    return "Fair"


def recommendation(metric_name: str, sensitive_attribute: str) -> str:
    recommendations = {
        "Demographic Parity Difference": (
            f"Review whether positive outcomes are distributed consistently across "
            f"{sensitive_attribute} groups. Consider threshold mitigation, sampling checks, "
            "and feature review before deployment."
        ),
        "Equalized Odds Difference": (
            f"Audit model error rates by {sensitive_attribute}. Retrain with better subgroup "
            "coverage or use post-processing to reduce uneven false positive and false "
            "negative rates."
        ),
        "Calibration Difference": (
            f"Check whether predicted probabilities mean the same thing across "
            f"{sensitive_attribute} groups. Consider calibration curves and group-aware "
            "probability calibration."
        ),
    }
    return recommendations.get(metric_name, "Review subgroup performance before deployment.")


def metric_card(title: str, value: float) -> str:
    label, band_class, _ = score_band(value)
    return f"""
    <div class="metric-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value:.3f}</div>
        <span class="metric-band {band_class}">{label}</span>
    </div>
    """


def build_report_pdf(
    dataset_name: str,
    sensitive_attribute: str,
    target_column: str,
    metrics: dict[str, float],
    verdict: str,
    flagged_metrics: list[str],
) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "FairScan Bias Audit Report", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 7, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Dataset", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, f"Name: {dataset_name}\nSensitive attribute: {sensitive_attribute}\nTarget: {target_column}")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Fairness Metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    for name, value in metrics.items():
        status, _, _ = score_band(value)
        pdf.cell(0, 7, f"{name}: {value:.3f} ({status})", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Verdict", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 9, verdict, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Recommendations", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)

    if flagged_metrics:
        for metric_name in flagged_metrics:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, metric_name, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 7, recommendation(metric_name, sensitive_attribute))
            pdf.ln(1)
    else:
        pdf.multi_cell(
            0,
            7,
            "No high-risk fairness gaps were detected. Continue monitoring subgroup "
            "performance as new data arrives.",
        )

    output = pdf.output(dest="S")
    if isinstance(output, str):
        return output.encode("latin-1")
    return bytes(output)


def read_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def positive_rate_frame(
    sensitive_features: pd.Series,
    before_predictions: np.ndarray,
    after_predictions: np.ndarray,
) -> pd.DataFrame:
    before = MetricFrame(
        metrics={"approval_rate": selection_rate},
        y_true=before_predictions,
        y_pred=before_predictions,
        sensitive_features=sensitive_features,
    ).by_group["approval_rate"]
    after = MetricFrame(
        metrics={"approval_rate": selection_rate},
        y_true=after_predictions,
        y_pred=after_predictions,
        sensitive_features=sensitive_features,
    ).by_group["approval_rate"]

    return (
        pd.concat(
            [
                before.rename("Before mitigation"),
                after.rename("After mitigation"),
            ],
            axis=1,
        )
        .fillna(0)
        .reset_index()
        .rename(columns={sensitive_features.name: "Sensitive group"})
    )


def build_bias_chart(rate_df: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_bar(
        x=rate_df["Sensitive group"],
        y=rate_df["Before mitigation"],
        name="Before mitigation",
        marker_color="#fb7185",
    )
    figure.add_bar(
        x=rate_df["Sensitive group"],
        y=rate_df["After mitigation"],
        name="After mitigation",
        marker_color="#47d18c",
    )
    figure.update_layout(
        barmode="group",
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)",
        font={"color": "#dbe7f5"},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        margin={"l": 20, "r": 20, "t": 45, "b": 30},
        yaxis={
            "title": "Positive outcome rate",
            "tickformat": ".0%",
            "gridcolor": "rgba(148, 163, 184, 0.18)",
            "range": [0, max(1, float(rate_df[["Before mitigation", "After mitigation"]].max().max()) * 1.15)],
        },
        xaxis={"title": None},
    )
    return figure


def render_pills(flagged_metrics: list[str], sensitive_attribute: str) -> None:
    if not flagged_metrics:
        st.markdown(
            """
            <div class="pill-row">
                <span class="finding-pill" style="background:#47d18c;">No high-risk bias findings</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    pills = []
    for metric in flagged_metrics:
        label, _, color = score_band(0.11)
        pills.append(
            f'<span class="finding-pill" style="background:{color};">'
            f"{label} detected in: {sensitive_attribute} ({metric})"
            "</span>"
        )
    st.markdown(f'<div class="pill-row">{"".join(pills)}</div>', unsafe_allow_html=True)


def run_audit(
    df: pd.DataFrame,
    dataset_name: str,
    sensitive_attribute: str,
    target_column: str,
    positive_value: Any,
) -> None:
    working_df = df.copy()
    working_df = working_df.dropna(subset=[sensitive_attribute, target_column])

    y = normalize_binary_target(working_df[target_column], positive_value)
    X = working_df.drop(columns=[target_column])
    sensitive = working_df[sensitive_attribute].astype(str)

    if y.nunique() != 2:
        st.error("The selected target must contain exactly two classes after choosing the positive outcome.")
        return

    if sensitive.nunique() < 2:
        st.error("The selected sensitive attribute must contain at least two groups.")
        return

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X,
        y,
        sensitive,
        test_size=0.35,
        random_state=7,
        stratify=stratify,
    )

    model = make_model(X_train)
    model.fit(X_train, y_train)

    before_predictions = model.predict(X_test)
    before_probabilities = model.predict_proba(X_test)[:, 1]

    mitigator = ThresholdOptimizer(
        estimator=model,
        constraints="demographic_parity",
        objective="balanced_accuracy_score",
        predict_method="predict_proba",
        prefit=True,
    )
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
    after_predictions = mitigator.predict(X_test, sensitive_features=sensitive_test)

    metrics = {
        "Demographic Parity Difference": float(
            demographic_parity_difference(
                y_test,
                before_predictions,
                sensitive_features=sensitive_test,
            )
        ),
        "Equalized Odds Difference": float(
            equalized_odds_difference(
                y_test,
                before_predictions,
                sensitive_features=sensitive_test,
            )
        ),
    }
    calibration_difference, calibration_by_group = group_calibration_difference(
        y_test,
        before_probabilities,
        sensitive_test,
    )
    metrics["Calibration Difference"] = calibration_difference

    flagged_metrics = [name for name, value in metrics.items() if value > 0.10]
    review_metrics = [name for name, value in metrics.items() if 0.05 < value <= 0.10]
    verdict = verdict_from_metrics(metrics)
    rate_df = positive_rate_frame(sensitive_test, before_predictions, after_predictions)

    st.markdown(
        f"""
        <div>
            <h1 style="margin-bottom:0.2rem;">FairScan</h1>
            <div class="subtle">
                AI bias detection dashboard for dataset audits, mitigation previews, and PDF reporting.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    top_left, top_right = st.columns([1.35, 0.9])
    with top_left:
        st.markdown("### Dataset Audit")
        st.markdown(
            f"""
            <div class="metric-grid">
                {metric_card("Demographic Parity Difference", metrics["Demographic Parity Difference"])}
                {metric_card("Equalized Odds Difference", metrics["Equalized Odds Difference"])}
                {metric_card("Calibration Difference", metrics["Calibration Difference"])}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_right:
        st.markdown("### Model Snapshot")
        accuracy = accuracy_score(y_test, before_predictions)
        balanced_accuracy = balanced_accuracy_score(y_test, before_predictions)
        st.markdown(
            f"""
            <div class="panel">
                <div class="metric-label">Dataset</div>
                <div style="font-weight:760; color:#f8fafc;">{dataset_name}</div>
                <br>
                <div class="metric-label">Verdict</div>
                <div style="font-size:1.65rem; font-weight:800; color:#f8fafc;">{verdict}</div>
                <br>
                <div class="metric-label">Accuracy</div>
                <div style="color:#dbe7f5;">{accuracy:.1%} accuracy · {balanced_accuracy:.1%} balanced accuracy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Visual Bias Report")
    st.plotly_chart(build_bias_chart(rate_df), use_container_width=True)
    render_pills(flagged_metrics, sensitive_attribute)

    with st.expander("Group details", expanded=False):
        detail_left, detail_right = st.columns(2)
        with detail_left:
            st.caption("Positive outcome rate by group")
            st.dataframe(rate_df, use_container_width=True, hide_index=True)
        with detail_right:
            st.caption("Calibration gap by group")
            st.dataframe(
                calibration_by_group.rename("Calibration gap").reset_index(),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("### One-Click PDF Report")
    st.write(
        "Download a plain-English summary with the selected sensitive attribute, metric scores, verdict, and recommendations."
    )
    pdf_bytes = build_report_pdf(
        dataset_name,
        sensitive_attribute,
        target_column,
        metrics,
        verdict,
        flagged_metrics or review_metrics,
    )
    st.download_button(
        label="Download PDF report",
        data=pdf_bytes,
        file_name=f"fairscan_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=False,
    )


def main() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)

    sample_df = make_adult_income_sample()
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = read_uploaded_csv(uploaded_file)
            dataset_name = uploaded_file.name
        except Exception as exc:
            st.sidebar.error(f"Could not read CSV: {exc}")
            df = sample_df
            dataset_name = "UCI Adult Income demo sample"
    else:
        df = sample_df
        dataset_name = "UCI Adult Income demo sample"

    st.sidebar.title("FairScan")
    st.sidebar.caption("Upload a CSV or use the built-in Adult Income demo sample.")

    if df.empty or len(df.columns) < 2:
        st.error("The dataset needs at least two columns and one row.")
        return

    columns = df.columns.tolist()
    sensitive_default = columns.index("gender") if "gender" in columns else 0
    target_default = columns.index("income") if "income" in columns else len(columns) - 1

    sensitive_attribute = st.sidebar.selectbox(
        "Sensitive attribute",
        columns,
        index=sensitive_default,
    )
    target_options = [column for column in columns if column != sensitive_attribute]
    target_index = target_options.index("income") if "income" in target_options else min(target_default, len(target_options) - 1)
    target_column = st.sidebar.selectbox("Target/outcome column", target_options, index=target_index)

    unique_target_values = sorted(df[target_column].dropna().astype(str).unique().tolist())
    if len(unique_target_values) != 2:
        st.warning("Select a binary target/outcome column to run the fairness audit.")
        st.dataframe(df.head(25), use_container_width=True)
        return

    positive_default = default_positive_value(unique_target_values)
    positive_index = unique_target_values.index(str(positive_default))
    positive_value = st.sidebar.selectbox(
        "Positive outcome",
        unique_target_values,
        index=positive_index,
    )

    st.sidebar.divider()
    st.sidebar.metric("Rows", f"{len(df):,}")
    st.sidebar.metric("Columns", f"{len(df.columns):,}")
    st.sidebar.caption("Bias bands: green <= 0.05, yellow <= 0.10, red > 0.10.")

    try:
        run_audit(df, dataset_name, sensitive_attribute, target_column, positive_value)
    except ValueError as exc:
        st.error(f"Audit could not run: {exc}")
    except Exception as exc:
        st.exception(exc)


if __name__ == "__main__":
    main()
