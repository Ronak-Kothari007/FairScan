import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from fpdf import FPDF, XPos, YPos
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="FairScan — AI Bias Detection",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Sample dataset (UCI Adult Income)
# ---------------------------------------------------------------------------

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]

ADULT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)


@st.cache_data(show_spinner=False)
def load_sample_dataset() -> pd.DataFrame:
    """Load the UCI Adult Income dataset; fall back to a synthetic sample if offline."""
    try:
        df = pd.read_csv(
            ADULT_URL,
            names=ADULT_COLUMNS,
            sep=r",\s*",
            engine="python",
            na_values="?",
        )
        df = df.dropna().reset_index(drop=True)
        df["income"] = (df["income"].str.strip() == ">50K").astype(int)
        return df
    except Exception:
        rng = np.random.default_rng(42)
        n = 2000
        sex = rng.choice(["Male", "Female"], size=n, p=[0.55, 0.45])
        race = rng.choice(
            ["White", "Black", "Asian-Pac-Islander", "Other"],
            size=n,
            p=[0.7, 0.15, 0.1, 0.05],
        )
        age = rng.integers(18, 75, size=n)
        education_num = rng.integers(5, 16, size=n)
        hours = rng.integers(20, 70, size=n)
        # Inject bias: higher income probability for one group
        base = 0.05 + 0.02 * (education_num - 5) + 0.005 * (hours - 20)
        bias = np.where(sex == "Male", 0.18, 0.0)
        prob = np.clip(base + bias, 0.02, 0.95)
        income = (rng.random(n) < prob).astype(int)
        return pd.DataFrame(
            {
                "age": age,
                "sex": sex,
                "race": race,
                "education_num": education_num,
                "hours_per_week": hours,
                "workclass": rng.choice(
                    ["Private", "Self-emp", "Government"], size=n
                ),
                "income": income,
            }
        )


# ---------------------------------------------------------------------------
# Modeling + fairness
# ---------------------------------------------------------------------------

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )


def calibration_difference(y_true, y_pred, sensitive) -> float:
    """Max gap in positive predictive value (calibration) across groups."""
    mf = MetricFrame(
        metrics=lambda yt, yp: (
            float((yt[yp == 1]).mean()) if (yp == 1).any() else 0.0
        ),
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        sensitive_features=np.asarray(sensitive),
    )
    return float(mf.difference())


def severity(value: float) -> str:
    if value > 0.10:
        return "high"
    if value > 0.05:
        return "moderate"
    return "fair"


COLOR = {
    "high": "#FF4B6E",
    "moderate": "#F5C451",
    "fair": "#3DD68C",
}

LABEL = {
    "high": "HIGH BIAS",
    "moderate": "MODERATE BIAS",
    "fair": "FAIR",
}


def metric_card(col, title: str, value: float, description: str) -> str:
    sev = severity(value)
    color = COLOR[sev]
    col.markdown(
        f"""
        <div style="
            background: linear-gradient(180deg,#171a26 0%,#10131c 100%);
            border: 1px solid {color}55;
            border-left: 4px solid {color};
            padding: 18px 20px;
            border-radius: 12px;
            box-shadow: 0 1px 0 #ffffff08 inset;
        ">
            <div style="font-size: 0.78rem; letter-spacing: .08em;
                        text-transform: uppercase; color: #9aa3b2;">
                {title}
            </div>
            <div style="font-size: 2.1rem; font-weight: 700;
                        color: {color}; margin: 4px 0 2px;">
                {value:.3f}
            </div>
            <div style="font-size: 0.72rem; color: {color};
                        font-weight: 600; letter-spacing: .1em;">
                {LABEL[sev]}
            </div>
            <div style="font-size: 0.78rem; color: #b6bccb;
                        margin-top: 8px; line-height: 1.35;">
                {description}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return sev


def pill(text: str, color: str) -> str:
    return (
        f'<span style="display:inline-block; padding:6px 14px; '
        f'border-radius:999px; background:{color}22; color:{color}; '
        f'border:1px solid {color}55; font-size:.82rem; font-weight:600; '
        f'margin:4px 6px 4px 0;">{text}</span>'
    )


def verdict_for(metrics: dict) -> tuple[str, str]:
    sevs = [severity(v) for v in metrics.values()]
    if "high" in sevs:
        return "High Risk", COLOR["high"]
    if "moderate" in sevs:
        return "Needs Review", COLOR["moderate"]
    return "Fair", COLOR["fair"]


def recommendation_for(name: str, value: float) -> str:
    sev = severity(value)
    if sev == "fair":
        return (
            f"{name}: Within acceptable range ({value:.3f}). Continue monitoring "
            f"on new data and during model retraining."
        )
    if sev == "moderate":
        return (
            f"{name}: Moderate disparity detected ({value:.3f}). Review training "
            f"data composition, consider reweighting samples or applying a "
            f"post-processing fairness constraint such as ThresholdOptimizer."
        )
    return (
        f"{name}: HIGH disparity detected ({value:.3f}). Apply mitigation before "
        f"deployment. Recommended actions: rebalance the training set, audit "
        f"feature engineering for proxy variables, and apply a demographic-parity "
        f"or equalized-odds constraint via fairlearn ThresholdOptimizer or "
        f"ExponentiatedGradient. Re-audit after mitigation."
    )


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------

_NEXT_LINE = dict(new_x=XPos.LMARGIN, new_y=YPos.NEXT)
_SAME_LINE = dict(new_x=XPos.RIGHT, new_y=YPos.TOP)


class ReportPDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(40, 40, 60)
        self.cell(0, 10, "FairScan - Bias Audit Report", **_NEXT_LINE)
        self.set_draw_color(200, 200, 210)
        self.line(10, 22, 200, 22)
        self.ln(6)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140, 140, 150)
        self.cell(
            0,
            8,
            f"Generated by FairScan - {datetime.now():%Y-%m-%d %H:%M}",
            align="C",
        )


def build_pdf(
    dataset_name: str,
    sensitive_attr: str,
    target_col: str,
    metrics: dict,
    verdict: str,
    recs: list[str],
) -> bytes:
    pdf = ReportPDF()
    pdf.add_page()

    def kv(label: str, value: str) -> None:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(50, 7, label, **_SAME_LINE)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, value, **_NEXT_LINE)

    kv("Dataset:", dataset_name)
    kv("Sensitive attribute:", sensitive_attr)
    kv("Target column:", target_col)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Fairness Metrics", **_NEXT_LINE)
    pdf.set_font("Helvetica", "", 11)
    for name, value in metrics.items():
        sev = severity(value).upper()
        pdf.cell(0, 7, f"  - {name}: {value:.4f}  [{sev}]", **_NEXT_LINE)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, f"Overall Verdict: {verdict}", **_NEXT_LINE)

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Recommendations", **_NEXT_LINE)
    pdf.set_font("Helvetica", "", 10)
    for r in recs:
        pdf.multi_cell(0, 6, f"- {r}")
        pdf.ln(1)

    out = pdf.output()
    return bytes(out)


# ---------------------------------------------------------------------------
# Sidebar — data source & controls
# ---------------------------------------------------------------------------

st.sidebar.title("⚖  FairScan")
st.sidebar.caption("AI bias detection dashboard")

source = st.sidebar.radio(
    "Data source",
    ["Sample (UCI Adult Income)", "Upload CSV"],
    index=0,
)

if source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        dataset_name = uploaded.name
    else:
        st.sidebar.info("Awaiting upload — using sample dataset for now.")
        df = load_sample_dataset()
        dataset_name = "UCI Adult Income (sample)"
else:
    df = load_sample_dataset()
    dataset_name = "UCI Adult Income (sample)"

st.sidebar.markdown("---")

columns = list(df.columns)

default_sensitive = next(
    (c for c in ["sex", "gender", "race", "age"] if c in columns), columns[0]
)
sensitive_attr = st.sidebar.selectbox(
    "Sensitive attribute",
    columns,
    index=columns.index(default_sensitive),
)

target_choices = [c for c in columns if c != sensitive_attr]
default_target = next(
    (c for c in ["income", "target", "label", "outcome"] if c in target_choices),
    target_choices[-1],
)
target_col = st.sidebar.selectbox(
    "Target / outcome",
    target_choices,
    index=target_choices.index(default_target),
)

# Decide how to binarize the target. We support three shapes:
#   1. Already 0/1 numeric -> use as-is.
#   2. Numeric (or string-but-numeric) with > 2 unique values -> cutoff slider.
#   3. Categorical with any number of unique values -> pick a "positive class".
target_series = df[target_col].dropna()
unique_vals = sorted(target_series.unique().tolist(), key=lambda x: str(x))
positive_class = None

is_binary_numeric = (
    target_series.dtype.kind in "biu"
    and set(target_series.unique()).issubset({0, 1})
)

# Try a numeric coercion so columns like "5"/"10" stored as strings still
# get the slider treatment. We only treat the column as numeric if a clear
# majority of rows can be parsed as numbers AND there are more than 2
# distinct values (otherwise it's really a binary categorical).
numeric_coerced = pd.to_numeric(target_series, errors="coerce")
numeric_ratio = (
    numeric_coerced.notna().sum() / len(target_series)
    if len(target_series) > 0 else 0.0
)
treat_as_numeric = (
    not is_binary_numeric
    and len(unique_vals) > 2
    and numeric_ratio >= 0.95
)

if is_binary_numeric:
    st.sidebar.caption("Target is already binary (0/1).")

elif treat_as_numeric:
    numeric_target = numeric_coerced.dropna()
    lo = float(numeric_target.min())
    hi = float(numeric_target.max())
    median = float(numeric_target.median())
    if hi <= lo:
        st.sidebar.error(
            f"'{target_col}' has no variation — pick a different target."
        )
    else:
        st.sidebar.info(
            f"'{target_col}' is numeric with {len(unique_vals)} unique values. "
            "Pick a cutoff — values strictly above it count as the positive outcome."
        )
        span = hi - lo
        step = 1.0 if span >= 50 else max(span / 100.0, 1e-4)
        default = median if median > lo else (lo + step)
        threshold = st.sidebar.slider(
            f"Cutoff for '{target_col}'",
            min_value=lo,
            max_value=hi,
            value=float(default),
            step=float(step),
        )
        pos_count = int((numeric_target > threshold).sum())
        total = int(len(numeric_target))
        st.sidebar.caption(
            f"Positive class: **{target_col} > {threshold:g}** "
            f"({pos_count:,} / {total:,} rows = {pos_count / total:.1%})"
        )
        if pos_count == 0 or pos_count == total:
            st.sidebar.error(
                "That cutoff puts every row on one side — pick a value where "
                "both classes have samples."
            )
        else:
            positive_class = ("__threshold__", threshold)

elif len(unique_vals) == 2:
    positive_class = st.sidebar.selectbox(
        "Positive class",
        unique_vals,
        index=1,
        help="The value that counts as the 'positive' outcome (e.g. approved, >50K).",
    )

elif len(unique_vals) >= 3:
    # Categorical target with 3+ categories: collapse to "chosen value vs rest".
    # Show the value counts so the user can pick something with enough samples.
    counts = target_series.value_counts()
    options = [str(v) for v in counts.index.tolist()]
    if len(options) > 30:
        st.sidebar.warning(
            f"'{target_col}' has {len(options)} categories. Pick the one that "
            "counts as the positive outcome — every other category becomes negative."
        )
    else:
        st.sidebar.info(
            f"'{target_col}' has {len(options)} categories. Pick the one that "
            "counts as the positive outcome — every other category becomes negative."
        )
    label_map = {str(v): v for v in counts.index.tolist()}

    def _fmt(opt: str) -> str:
        n = int(counts.loc[label_map[opt]])
        share = n / counts.sum()
        return f"{opt}  ({n:,} rows, {share:.1%})"

    chosen = st.sidebar.selectbox(
        "Positive class",
        options,
        index=0,
        format_func=_fmt,
    )
    chosen_value = label_map[chosen]
    pos_count = int(counts.loc[chosen_value])
    total = int(counts.sum())
    if pos_count == 0 or pos_count == total:
        st.sidebar.error(
            "That class covers every row (or none) — pick a different value."
        )
    else:
        positive_class = chosen_value

else:
    st.sidebar.error(
        f"'{target_col}' has only one unique value — pick a different target."
    )

test_size = st.sidebar.slider("Test set size", 0.1, 0.5, 0.25, 0.05)
can_run = is_binary_numeric or positive_class is not None
run = st.sidebar.button(
    "Run audit",
    type="primary",
    width="stretch",
    disabled=not can_run,
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Rows: **{len(df):,}**  |  Columns: **{len(df.columns)}**")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style="display:flex; align-items:baseline; gap:14px;">
        <h1 style="margin:0; font-weight:800;">FairScan</h1>
        <span style="color:#9aa3b2; font-size:.95rem;">
            Detect, visualize, and mitigate AI bias in tabular models
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

with st.expander("Preview dataset", expanded=False):
    st.dataframe(df.head(50), width="stretch", height=260)


def _coerce_binary_target(series: pd.Series, positive_class) -> pd.Series:
    """Convert a target column into a 0/1 vector using the chosen positive class."""
    if positive_class is None:
        if series.dtype.kind in "biu" and set(series.unique()).issubset({0, 1}):
            return series.astype(int)
        uniq = sorted(series.dropna().unique().tolist(), key=lambda x: str(x))
        if len(uniq) != 2:
            raise ValueError(
                f"Target '{series.name}' has {len(uniq)} unique values. "
                "Pick a positive class in the sidebar."
            )
        # default: alphabetically-second value is the positive class
        return (series == uniq[1]).astype(int)

    if isinstance(positive_class, tuple) and positive_class[0] == "__threshold__":
        return (series.astype(float) > float(positive_class[1])).astype(int)

    return (series == positive_class).astype(int)


def _safe_train_test_split(X, y, s, test_size: float):
    """Stratify by (target, sensitive) when possible; fall back gracefully."""
    try:
        strat = y.astype(str) + "__" + s.astype(str)
        if strat.value_counts().min() >= 2:
            return train_test_split(
                X, y, s, test_size=test_size, random_state=42, stratify=strat
            )
    except Exception:
        pass
    try:
        return train_test_split(
            X, y, s, test_size=test_size, random_state=42, stratify=y
        )
    except Exception:
        return train_test_split(X, y, s, test_size=test_size, random_state=42)


@st.cache_data(show_spinner=False)
def run_audit(
    df: pd.DataFrame,
    sensitive_attr: str,
    target_col: str,
    test_size: float,
    positive_class,
):
    work = df.dropna(subset=[sensitive_attr, target_col]).copy()
    y = _coerce_binary_target(work[target_col], positive_class)

    pos_count = int(y.sum())
    if pos_count < 5 or pos_count > len(y) - 5:
        raise ValueError(
            f"Target is too imbalanced after binarization: only {pos_count} "
            f"positive out of {len(y)} rows. Pick a different positive class."
        )

    sensitive = work[sensitive_attr].astype(str)

    # Drop tiny groups that ThresholdOptimizer can't fit reliably
    group_sizes = sensitive.value_counts()
    keep_groups = group_sizes[group_sizes >= 30].index
    if len(keep_groups) < 2:
        raise ValueError(
            f"Sensitive attribute '{sensitive_attr}' needs at least two groups "
            f"with 30+ rows each. Largest groups: "
            f"{group_sizes.head(5).to_dict()}"
        )
    if len(keep_groups) < len(group_sizes):
        mask = sensitive.isin(keep_groups)
        work, y, sensitive = work[mask], y[mask], sensitive[mask]

    X = work.drop(columns=[target_col])

    X_tr, X_te, y_tr, y_te, s_tr, s_te = _safe_train_test_split(
        X, y, sensitive, test_size
    )

    pipe = build_pipeline(X_tr)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    dpd = demographic_parity_difference(
        y_te, y_pred, sensitive_features=s_te
    )
    eod = equalized_odds_difference(
        y_te, y_pred, sensitive_features=s_te
    )
    cal = calibration_difference(y_te, y_pred, s_te)

    grouped = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
            "accuracy": accuracy_score,
        },
        y_true=y_te,
        y_pred=y_pred,
        sensitive_features=s_te,
    ).by_group

    base_rate_by_group = (
        pd.Series(np.asarray(y_te))
        .groupby(np.asarray(s_te))
        .mean()
        .rename("base_rate")
    )
    n_by_group = (
        pd.Series(np.asarray(y_te))
        .groupby(np.asarray(s_te))
        .size()
        .rename("n")
    )

    rates_before = grouped["selection_rate"]
    accuracy_before = float(accuracy_score(y_te, y_pred))

    # Mitigation via ThresholdOptimizer (graceful fallback if it can't fit)
    mitigation_error: str | None = None
    try:
        mitigator = ThresholdOptimizer(
            estimator=pipe,
            constraints="demographic_parity",
            prefit=True,
            predict_method="predict_proba",
        )
        mitigator.fit(X_tr, y_tr, sensitive_features=s_tr)
        y_pred_mit = mitigator.predict(X_te, sensitive_features=s_te)
        rates_after = MetricFrame(
            metrics=selection_rate,
            y_true=y_te,
            y_pred=y_pred_mit,
            sensitive_features=s_te,
        ).by_group
        dpd_after = float(
            demographic_parity_difference(
                y_te, y_pred_mit, sensitive_features=s_te
            )
        )
        accuracy_after = float(accuracy_score(y_te, y_pred_mit))
    except Exception as exc:
        mitigation_error = str(exc)
        rates_after = rates_before * np.nan
        dpd_after = float("nan")
        accuracy_after = float("nan")

    breakdown = pd.concat(
        [
            n_by_group,
            base_rate_by_group,
            rates_before.rename("model_rate_before"),
            rates_after.rename("model_rate_after"),
            grouped["tpr"].rename("tpr"),
            grouped["fpr"].rename("fpr"),
        ],
        axis=1,
    )
    breakdown.index.name = sensitive_attr

    return {
        "metrics": {
            "Demographic Parity Difference": float(dpd),
            "Equalized Odds Difference": float(eod),
            "Calibration Difference": float(cal),
        },
        "rates_before": rates_before,
        "rates_after": rates_after,
        "dpd_after": dpd_after,
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "breakdown": breakdown,
        "n_test": int(len(y_te)),
        "n_train": int(len(y_tr)),
        "mitigation_error": mitigation_error,
    }


if "results" not in st.session_state:
    st.session_state["results"] = None
    st.session_state["meta"] = None
    st.session_state["auto_ran"] = False

# Auto-run once on first load with the sample dataset, so the demo is instant
should_auto_run = (
    not st.session_state["auto_ran"]
    and source == "Sample (UCI Adult Income)"
    and can_run
    and st.session_state["results"] is None
)

if run or should_auto_run:
    try:
        with st.spinner("Training model and computing fairness metrics..."):
            res = run_audit(
                df, sensitive_attr, target_col, test_size, positive_class
            )
        st.session_state["results"] = res
        st.session_state["meta"] = {
            "dataset_name": dataset_name,
            "sensitive_attr": sensitive_attr,
            "target_col": target_col,
            "positive_class": positive_class,
        }
        st.session_state["auto_ran"] = True
    except Exception as exc:
        st.error(f"Audit failed: {exc}")
        st.session_state["auto_ran"] = True

results = st.session_state["results"]
meta = st.session_state["meta"]

if results is None:
    st.info(
        "Pick a sensitive attribute and target column in the sidebar, then "
        "click **Run audit**. Defaults are pre-filled for the sample dataset."
    )
    st.stop()

metrics = results["metrics"]

# Metric cards
st.subheader("Fairness metrics")
c1, c2, c3 = st.columns(3)
metric_card(
    c1,
    "Demographic Parity Difference",
    metrics["Demographic Parity Difference"],
    "Gap in positive-outcome rate between the most and least favored groups.",
)
metric_card(
    c2,
    "Equalized Odds Difference",
    metrics["Equalized Odds Difference"],
    "Max gap in true-positive and false-positive rates across groups.",
)
metric_card(
    c3,
    "Calibration Difference",
    metrics["Calibration Difference"],
    "Gap in positive predictive value across groups (probability calibration).",
)

st.caption(
    f"Trained on **{results['n_train']:,}** rows  ·  evaluated on "
    f"**{results['n_test']:,}** rows  ·  thresholds: <0.05 fair · "
    f"0.05–0.10 moderate · >0.10 high"
)

st.markdown("---")

# Visual bias report
st.subheader("Visual bias report")
st.caption(
    "Approval rate per group, before vs after mitigation with "
    "ThresholdOptimizer (demographic parity constraint)."
)

groups = [str(g) for g in results["rates_before"].index.tolist()]
before = [float(v) for v in results["rates_before"].values]
after = [float(v) for v in results["rates_after"].values]

fig = go.Figure(
    data=[
        go.Bar(
            name="Before mitigation",
            x=groups,
            y=before,
            marker_color="#FF4B6E",
            text=[f"{v:.1%}" for v in before],
            textposition="outside",
        ),
        go.Bar(
            name="After mitigation",
            x=groups,
            y=after,
            marker_color="#3DD68C",
            text=[f"{v:.1%}" for v in after],
            textposition="outside",
        ),
    ]
)
fig.update_layout(
    barmode="group",
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(title="Approval / positive-outcome rate", tickformat=".0%"),
    xaxis=dict(title=meta["sensitive_attr"]),
    height=420,
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, width="stretch")

if results.get("mitigation_error"):
    st.warning(
        f"Mitigation step skipped — {results['mitigation_error']}. "
        "Showing pre-mitigation rates only."
    )
elif np.isnan(results["dpd_after"]):
    st.caption("Mitigation results unavailable.")
else:
    st.caption(
        f"After mitigation, demographic parity difference dropped to "
        f"**{results['dpd_after']:.3f}** "
        f"(from **{metrics['Demographic Parity Difference']:.3f}**)."
    )

# Accuracy vs fairness tradeoff
acc_b = results["accuracy_before"]
acc_a = results["accuracy_after"]
dpd_b = metrics["Demographic Parity Difference"]
dpd_a = results["dpd_after"]

acc_cols = st.columns(2)
acc_cols[0].metric(
    "Model accuracy",
    f"{acc_b:.1%}",
    delta=(f"{(acc_a - acc_b) * 100:+.1f} pp after mitigation"
           if not np.isnan(acc_a) else None),
    delta_color="inverse",
)
acc_cols[1].metric(
    "Demographic parity gap",
    f"{dpd_b:.3f}",
    delta=(f"{(dpd_a - dpd_b):+.3f} after mitigation"
           if not np.isnan(dpd_a) else None),
    delta_color="inverse",
)

# Per-group breakdown
st.markdown("**Per-group breakdown** (test set)")
bd = results["breakdown"].copy()
bd_display = pd.DataFrame(
    {
        "Group": bd.index.astype(str),
        "Samples": bd["n"].astype(int),
        "Base rate": bd["base_rate"],
        "Approval (before)": bd["model_rate_before"],
        "Approval (after)": bd["model_rate_after"],
        "True positive rate": bd["tpr"],
        "False positive rate": bd["fpr"],
    }
)
st.dataframe(
    bd_display,
    width="stretch",
    hide_index=True,
    column_config={
        "Base rate": st.column_config.ProgressColumn(
            "Base rate", format="%.1f%%", min_value=0.0, max_value=1.0
        ),
        "Approval (before)": st.column_config.ProgressColumn(
            "Approval (before)", format="%.1f%%", min_value=0.0, max_value=1.0
        ),
        "Approval (after)": st.column_config.ProgressColumn(
            "Approval (after)", format="%.1f%%", min_value=0.0, max_value=1.0
        ),
        "True positive rate": st.column_config.NumberColumn(
            "TPR", format="%.2f"
        ),
        "False positive rate": st.column_config.NumberColumn(
            "FPR", format="%.2f"
        ),
    },
)
st.caption(
    "Base rate = fraction of the group with the positive outcome in the data. "
    "Approval = fraction the model predicts as positive. "
    "Large gaps between base rate and approval, or between groups, indicate bias."
)

# Findings pills
flagged = [
    (name, value, severity(value))
    for name, value in metrics.items()
    if severity(value) != "fair"
]

st.markdown("**Flagged findings**")
if not flagged:
    st.markdown(
        pill(f"No bias detected in: {meta['sensitive_attr']}", COLOR["fair"]),
        unsafe_allow_html=True,
    )
else:
    pills_html = "".join(
        pill(
            f"{LABEL[sev].title()} in {meta['sensitive_attr']} "
            f"— {name} ({value:.3f})",
            COLOR[sev],
        )
        for name, value, sev in flagged
    )
    st.markdown(pills_html, unsafe_allow_html=True)

st.markdown("---")

# PDF report
st.subheader("Download report")
verdict, vcolor = verdict_for(metrics)
recs = [recommendation_for(name, value) for name, value in metrics.items()]

st.markdown(
    f"<div style='font-size:1.05rem;'>Overall verdict: "
    f"<b style='color:{vcolor}'>{verdict}</b></div>",
    unsafe_allow_html=True,
)

pdf_bytes = build_pdf(
    dataset_name=meta["dataset_name"],
    sensitive_attr=meta["sensitive_attr"],
    target_col=meta["target_col"],
    metrics=metrics,
    verdict=verdict,
    recs=recs,
)

st.download_button(
    label="⬇  Download PDF report",
    data=pdf_bytes,
    file_name=(
        f"fairscan_report_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    ),
    mime="application/pdf",
    type="primary",
    width="content",
)
