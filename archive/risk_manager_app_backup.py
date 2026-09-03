"""PreShip AI: defense-only return/RTO risk manager."""

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "Export_Product_Return_Data.csv"
MODEL_DIR = APP_DIR / "models"
MODEL_PATH = MODEL_DIR / "preship_risk_models.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
FEATURES = [
    "Age",
    "Gender",
    "State",
    "Category",
    "Brand",
    "Quantity",
    "Price",
    "Discount",
    "Product Rating",
]
NUMERIC_FEATURES = ["Age", "Quantity", "Price", "Discount", "Product Rating"]
CATEGORICAL_FEATURES = ["Gender", "State", "Category", "Brand"]


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data = data.drop(columns=[column for column in data.columns if column.startswith("Unnamed:")])
    data = data.rename(columns={"high_return_risk": "High_Return_Risk"})
    missing = sorted(set(FEATURES + ["High_Return_Risk"]) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")
    data["target"] = data["High_Return_Risk"].astype(str).str.lower().eq("yes").astype(int)
    return data


def build_model() -> Pipeline:
    numeric = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def anomaly_risk(values, reference_scores):
    """Convert Isolation Forest scores into an interpretable 0-1 novelty risk."""
    return 1 - np.array([np.mean(reference_scores <= value) for value in values])


def save_model_bundle(model, sentinel, reference_scores):
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "return_risk_model": model,
            "novelty_sentinel": sentinel,
            "reference_scores": reference_scores,
            "features": FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "model_version": "1.0",
        },
        MODEL_PATH,
    )
    METADATA_PATH.write_text(
        json.dumps(
            {
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                "training_data": str(DATA_PATH.relative_to(APP_DIR)),
                "model_path": str(MODEL_PATH.relative_to(APP_DIR)),
                "purpose": "Defense-only return/RTO risk scoring and human verification support",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@st.cache_resource(show_spinner="Training the held-out evaluation model...")
def train_model(data: pd.DataFrame):
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        data[FEATURES], data["target"], test_size=0.4, random_state=42, stratify=data["target"]
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_holdout, y_holdout, test_size=0.5, random_state=42, stratify=y_holdout
    )
    model = build_model()
    model.fit(x_train, y_train)
    validation_probabilities = model.predict_proba(x_validation)[:, 1]
    probabilities = model.predict_proba(x_test)[:, 1]
    sentinel = IsolationForest(n_estimators=150, contamination="auto", random_state=42)
    sentinel.fit(x_train[NUMERIC_FEATURES])
    save_model_bundle(model, sentinel, sentinel.decision_function(x_train[NUMERIC_FEATURES]))
    reference_scores = sentinel.decision_function(x_train[NUMERIC_FEATURES])
    validation_anomaly = anomaly_risk(
        sentinel.decision_function(x_validation[NUMERIC_FEATURES]), reference_scores
    )
    test_anomaly = anomaly_risk(
        sentinel.decision_function(x_test[NUMERIC_FEATURES]), reference_scores
    )
    validation_risk = 0.8 * validation_probabilities + 0.2 * validation_anomaly
    test_risk = 0.8 * probabilities + 0.2 * test_anomaly
    return (
        model, sentinel, reference_scores, y_validation, validation_risk,
        x_test, y_test, test_risk, test_anomaly,
    )


def threshold_for_cost(y_true, probabilities, false_positive_cost, false_negative_cost):
    thresholds = np.linspace(0.05, 0.95, 181)
    costs = []
    for threshold in thresholds:
        predicted = probabilities >= threshold
        matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
        false_positives, false_negatives = matrix[0, 1], matrix[1, 0]
        costs.append(false_positives * false_positive_cost + false_negatives * false_negative_cost)
    best = int(np.argmin(costs))
    return float(thresholds[best]), float(costs[best])


def metric_report(y_true, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    return {
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "auc_pr": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "true_negatives": int(matrix[0, 0]),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "true_positives": int(matrix[1, 1]),
    }


def threshold_tradeoff(y_true, probabilities, false_positive_cost, false_negative_cost):
    rows = []
    for threshold in np.linspace(0.05, 0.95, 91):
        report = metric_report(y_true, probabilities, threshold)
        costs = cost_report(report, false_positive_cost, false_negative_cost)
        rows.append(
            {
                "Threshold": threshold,
                "Precision": report["precision"],
                "Recall": report["recall"],
                "False positives": report["false_positives"],
                "False negatives": report["false_negatives"],
                "Estimated cost": costs["total"],
            }
        )
    return pd.DataFrame(rows)


def cost_report(report, false_positive_cost, false_negative_cost):
    false_positive_total = report["false_positives"] * false_positive_cost
    false_negative_total = report["false_negatives"] * false_negative_cost
    return {
        "false_positive_total": false_positive_total,
        "false_negative_total": false_negative_total,
        "total": false_positive_total + false_negative_total,
    }


def action_for_probability(probability, threshold):
    if probability >= threshold:
        return "Manual review / verify"
    return "Proceed, monitor outcome"


def risk_band(score, threshold):
    if score >= threshold:
        return "High: review / verify"
    if score >= threshold * 0.5:
        return "Watch: below cutoff, monitor"
    return "Low: proceed"


def explain_prediction(model, order):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    transformed = preprocessor.transform(order[FEATURES])
    contributions = np.asarray(transformed[0].todense()).ravel() * classifier.coef_[0]
    names = preprocessor.get_feature_names_out()
    explanation = pd.DataFrame({"signal": names, "contribution": contributions})
    explanation["signal"] = explanation["signal"].str.replace(
        r"^(numeric|categorical)__", "", regex=True
    )
    return explanation.sort_values("contribution", ascending=False)


st.set_page_config(page_title="PreShip AI Risk Manager", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL APP
    ===================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(99, 102, 241, 0.10),
                transparent 30%
            ),
            radial-gradient(
                circle at 85% 15%,
                rgba(34, 211, 238, 0.07),
                transparent 28%
            ),
            #080b12;

        color: #e5e7eb;

        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: transparent;
        color: #e5e7eb;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 11, 18, 0.85);
        backdrop-filter: blur(12px);
    }


    /* =====================================================
       SIDEBAR
    ===================================================== */

    section[data-testid="stSidebar"] {
        background: #0b0f18;
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    section[data-testid="stSidebar"] * {
        color: #d1d5db;
    }


    /* =====================================================
       TEXT
    ===================================================== */

    .stMarkdown,
    .stMarkdown p,
    .stCaption,
    label,
    [data-testid="stWidgetLabel"] p,
    [data-baseweb="tab"] {
        color: #a7b0c0 !important;
    }

    h1,
    h2,
    h3 {
        color: #f9fafb !important;
        font-weight: 750 !important;
        letter-spacing: -0.025em;
    }


    /* =====================================================
       HERO
    ===================================================== */

    .risk-hero {

        position: relative;

        overflow: hidden;

        width: 100%;
        box-sizing: border-box;

        padding: 2rem 2.2rem;

        margin-bottom: 1.2rem;

        border-radius: 20px;

        background:
            linear-gradient(
                135deg,
                #111827 0%,
                #0f172a 55%,
                #17153b 100%
            );

        border: 1px solid rgba(139,92,246,0.35);

        box-shadow:
            0 18px 50px rgba(0,0,0,0.35),
            inset 0 1px 0 rgba(255,255,255,0.04);
    }


    /* Purple glow */

    .risk-hero::before {

        content: "";

        position: absolute;

        width: 280px;
        height: 280px;

        right: -100px;
        top: -130px;

        background:
            radial-gradient(
                circle,
                rgba(139,92,246,0.35),
                transparent 68%
            );

        pointer-events: none;
    }


    /* Cyan glow */

    .risk-hero::after {

        content: "";

        position: absolute;

        width: 220px;
        height: 220px;

        left: 48%;
        bottom: -160px;

        background:
            radial-gradient(
                circle,
                rgba(34,211,238,0.14),
                transparent 68%
            );

        pointer-events: none;
    }


    .risk-hero h1 {

        position: relative;

        z-index: 2;

        color: #ffffff !important;

        margin: 0;

        font-size: 2.25rem !important;

        line-height: 1.15;

        font-weight: 800 !important;

        letter-spacing: -0.04em;
    }


    .risk-hero p {

        position: relative;

        z-index: 2;

        margin: .55rem 0 0;

        max-width: 720px;

        color: #9ca3af !important;

        font-size: .95rem;

        line-height: 1.6;
    }


    /* =====================================================
       DEFENSE / BAR
    ===================================================== */

    .bar {

        display: flex;

        align-items: center;

        width: 100%;

        box-sizing: border-box;

        padding: .85rem 1rem;

        margin: 1rem 0;

        background:
            rgba(17,24,39,0.85);

        border:
            1px solid rgba(255,255,255,0.07);

        border-left:
            3px solid #22d3ee;

        border-radius: 10px;

        color: #9ca3af;

        font-size: .82rem;

        font-weight: 600;

        letter-spacing: .01em;

        box-shadow:
            0 8px 25px rgba(0,0,0,0.18);
    }


    /* Highlight first part of THE BAR */

    .bar::first-line {
        color: #22d3ee;
    }


    /* =====================================================
       METRICS
    ===================================================== */

    [data-testid="stMetric"] {

        background:
            linear-gradient(
                145deg,
                rgba(17,24,39,0.95),
                rgba(15,23,42,0.82)
            );

        border:
            1px solid rgba(255,255,255,0.07);

        border-top:
            3px solid #6366f1;

        border-radius: 14px;

        padding: .9rem 1rem;

        box-shadow:
            0 10px 30px rgba(0,0,0,0.22);

        transition:
            transform .2s ease,
            border .2s ease,
            box-shadow .2s ease;
    }


    [data-testid="stMetric"]:hover {

        transform:
            translateY(-3px);

        border-color:
            rgba(34,211,238,0.3);

        box-shadow:
            0 15px 40px rgba(0,0,0,0.30);
    }


    [data-testid="stMetricLabel"] p {

        color: #6b7280 !important;

        font-size: .72rem !important;

        font-weight: 650 !important;

        text-transform: uppercase;

        letter-spacing: .06em;
    }


    [data-testid="stMetricValue"] {

        color: #f9fafb !important;

        font-weight: 800 !important;
    }


    /* =====================================================
       BUTTONS
    ===================================================== */

    div.stButton > button,
    div.stFormSubmitButton > button {

        min-height: 42px;

        padding: .55rem 1.2rem;

        border-radius: 9px;

        border:
            1px solid rgba(139,92,246,0.45);

        background:
            linear-gradient(
                135deg,
                #7c3aed,
                #6366f1
            );

        color: #ffffff;

        font-weight: 700;

        box-shadow:
            0 5px 20px rgba(99,102,241,0.20);

        transition:
            transform .15s ease,
            box-shadow .2s ease;
    }


    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {

        transform:
            translateY(-2px);

        background:
            linear-gradient(
                135deg,
                #8b5cf6,
                #6366f1
            );

        color: #ffffff;

        box-shadow:
            0 8px 28px rgba(99,102,241,0.35);
    }


    /* =====================================================
       INPUTS
    ===================================================== */

    input,
    textarea {

        background:
            #111827 !important;

        color:
            #f9fafb !important;

        border:
            1px solid rgba(255,255,255,0.08) !important;

        border-radius:
            9px !important;
    }


    input:focus,
    textarea:focus {

        border-color:
            #6366f1 !important;

        box-shadow:
            0 0 0 1px #6366f1 !important;
    }


    /* =====================================================
       SELECTBOX
    ===================================================== */

    div[data-baseweb="select"] > div {

        background:
            #111827 !important;

        color:
            #f9fafb !important;

        border:
            1px solid rgba(255,255,255,0.08) !important;

        border-radius:
            9px !important;
    }


    /* =====================================================
       TABS
    ===================================================== */

    [data-baseweb="tab"] {

        color:
            #6b7280 !important;

        font-weight:
            650;

        padding:
            .7rem 1rem;
    }


    [aria-selected="true"] {

        color:
            #22d3ee !important;
    }


    /* =====================================================
       ALERTS
    ===================================================== */

    [data-testid="stAlert"] {

        background:
            rgba(17,24,39,0.85);

        border-radius:
            10px;

        border:
            1px solid rgba(255,255,255,0.08);
    }


    /* =====================================================
       DATAFRAME
    ===================================================== */

    [data-testid="stDataFrame"] {

        border-radius:
            12px;

        overflow:
            hidden;

        border:
            1px solid rgba(255,255,255,0.07);

        box-shadow:
            0 10px 30px rgba(0,0,0,0.20);
    }


    </style>


    <!-- =====================================================
         HERO CONTENT
    ===================================================== -->

    <div class="risk-hero">
        <h1>PreShip AI Risk Manager</h1>
        <p>Known-pattern scoring plus a blind-spot sentinel for Indian returns and RTO.</p>
    </div>


    <!-- =====================================================
         DEFENSE BAR
    ===================================================== -->

    <div>
        <strong>🛡️ THE BAR:</strong>
        Honest metrics · Explicit false-positive cost · Defense-only decisions.
    </div>

    """,
    unsafe_allow_html=True,
)
st.caption("A risk signal supports verification. It never automatically rejects a customer or order.")

try:
    data = load_data()
    model, sentinel, reference_scores, y_validation, validation_risk, x_test, y_test, probabilities, test_anomaly = train_model(data)
except Exception as error:
    st.error(f"Could not load or train the model: {error}")
    st.stop()

if MODEL_PATH.exists():
    st.caption(f"Saved model bundle: `{MODEL_PATH.relative_to(APP_DIR)}`")

with st.sidebar:
    st.header("Cost assumptions")
    false_positive_cost = st.number_input("False-positive cost (₹)", min_value=0, value=150, step=25)
    false_negative_cost = st.number_input("False-negative cost (₹)", min_value=0, value=350, step=25)
    verification_cost = st.number_input("Verification cost per flagged order (₹)", min_value=0, value=8, step=1)
    missed_order_loss = st.number_input("Loss when a risky order is missed (₹)", min_value=0, value=350, step=25)
    planning_volume = st.number_input("Planning volume (orders)", min_value=100, value=1000, step=100)
    threshold, expected_cost = threshold_for_cost(
        y_validation, validation_risk, false_positive_cost, false_negative_cost
    )
    st.metric("Cost-tuned threshold", f"{threshold:.2f}")
    st.caption("This threshold is selected on validation data and then reported on the untouched test set.")
    st.subheader("How to read the score")
    st.markdown(
        f"**Low:** below `{threshold * 0.5:.2f}`  \n"
        f"**Watch:** `{threshold * 0.5:.2f}` to below `{threshold:.2f}`  \n"
        f"**High:** `{threshold:.2f}` or above"
    )
    st.caption("High means the score crosses the model's cost-based review cutoff. It is not proof of abuse and never auto-rejects an order.")

report = metric_report(y_test, probabilities, threshold)
costs = cost_report(report, false_positive_cost, false_negative_cost)
baseline_report = metric_report(y_test, np.ones(len(y_test)), 0.5)
tradeoff = threshold_tradeoff(y_test, probabilities, false_positive_cost, false_negative_cost)
flagged_rate = float(np.mean(probabilities >= threshold))
baseline_loss = float(np.mean(y_test) * missed_order_loss)
model_loss = flagged_rate * verification_cost + (report["false_negatives"] / len(y_test)) * missed_order_loss
estimated_savings = (baseline_loss - model_loss) * planning_volume
metrics_tab, predict_tab, data_tab = st.tabs(["Evaluation", "Score an order", "Dataset"])

with metrics_tab:
    st.subheader("Held-out test set results")
    st.write(f"Training: {len(data) - len(y_validation) - len(y_test):,} | Validation: {len(y_validation):,} | Final test: {len(y_test):,} orders")
    st.caption("The validation set chooses the cost threshold. The final test set is used once for the metrics below.")
    cards = st.columns(4)
    cards[0].metric("Precision", f"{report['precision']:.1%}")
    cards[1].metric("Recall", f"{report['recall']:.1%}")
    cards[2].metric("AUC-PR", f"{report['auc_pr']:.3f}")
    cards[3].metric("Test-set cost", f"₹{costs['total']:,.0f}")
    impact = st.columns(4)
    impact[0].metric("Flag rate", f"{flagged_rate:.1%}")
    impact[1].metric("Baseline loss / order", f"₹{baseline_loss:,.0f}")
    impact[2].metric("Model loss / order", f"₹{model_loss:,.0f}")
    impact[3].metric(f"Estimated savings / {planning_volume:,.0f}", f"₹{estimated_savings:,.0f}")
    st.caption("Business impact is a planning estimate: every flagged order incurs verification cost, while every missed risky order incurs the configured loss. Replace these assumptions with measured merchant costs.")
    st.write(f"ROC-AUC: **{report['roc_auc']:.3f}** | False positives: **{report['false_positives']:,}** | False negatives: **{report['false_negatives']:,}**")
    st.write(f"False-positive cost: **₹{costs['false_positive_total']:,.0f}** | False-negative cost: **₹{costs['false_negative_total']:,.0f}**")
    st.subheader("Confusion matrix")
    confusion_table = pd.DataFrame(
        [
            [report["true_negatives"], report["false_positives"]],
            [report["false_negatives"], report["true_positives"]],
        ],
        index=["Actual safe", "Actual high-risk"],
        columns=["Predicted safe", "Predicted high-risk"],
    )
    st.dataframe(confusion_table, width="stretch")
    st.caption("False positives are safe orders sent to verification. False negatives are high-risk orders that the model did not flag.")
    st.write(f"Blind-spot sentinel: **20%** of the hybrid score, flagging unusual numeric order profiles.")
    st.write(f"Always-review baseline: precision **{baseline_report['precision']:.1%}**, recall **{baseline_report['recall']:.1%}**, cost **₹{baseline_report['false_positives'] * false_positive_cost:,.0f}**")
    st.subheader("Choosing the review cutoff")
    st.line_chart(tradeoff.set_index("Threshold")[["Precision", "Recall"]], y_label="Rate", x_label="Review threshold")
    st.caption("Raising the cutoff usually reduces reviews but can miss more risky orders. The selected cutoff minimizes the configured validation cost; it is not universally correct.")
    with st.expander("What these numbers mean"):
        st.markdown(
            "- **Precision:** among flagged orders, the share that were high-return-risk in the test labels.\n"
            "- **Recall:** among high-return-risk orders, the share the model flagged.\n"
            "- **AUC-PR / ROC-AUC:** ranking quality across many possible cutoffs; they are not the order's risk score.\n"
            "- **False positive:** a safer order sent to verification. **False negative:** a risky order allowed through.\n"
            "- The cutoff is chosen by `false-positive cost × false positives + false-negative cost × false negatives`.\n"
            "- The hybrid score is a prioritization signal, not a calibrated probability that an individual order will be returned."
        )
    with st.expander("What the model looks at"):
        st.write("Customer: age and gender | Order: quantity, price, discount | Product: category, brand, rating | Context: state")
        st.caption("The novelty sentinel checks whether the numeric profile looks unusual compared with training data. It does not identify intent or prove fraud.")
    with st.expander("Evaluation limitations"):
        st.warning("The source file contains one repeated timestamp, so a future-based time split cannot be verified. Results use a stratified 60/20/20 train, validation, and test split and should be treated as a benchmark until timestamped production data is available.")
    st.info("A flag is a recommendation for verification, never an automatic denial or customer action.")

with predict_tab:
    st.subheader("Score one order")
    with st.form("order_form"):
        left, right = st.columns(2)
        with left:
            category = st.selectbox("Category", sorted(data["Category"].dropna().unique()))
            brand = st.selectbox("Brand", sorted(data["Brand"].dropna().unique()))
            gender = st.selectbox("Gender", sorted(data["Gender"].dropna().unique()))
            state = st.selectbox("State", sorted(data["State"].dropna().unique()))
            age = st.number_input("Customer age", 18, 100, 30)
        with right:
            quantity = st.number_input("Quantity", 1, 50, 1)
            price = st.number_input("Price (₹)", 1.0, 100000.0, 700.0)
            discount = st.number_input("Discount (%)", 0.0, 100.0, 20.0)
            rating = st.number_input("Product rating", 1.0, 5.0, 3.5, step=0.5)
        submitted = st.form_submit_button("Assess risk")

    if submitted:
        order = pd.DataFrame(
            [{
                "Age": age, "Gender": gender, "State": state, "Category": category,
                "Brand": brand, "Quantity": quantity, "Price": price,
                "Discount": discount, "Product Rating": rating,
            }]
        )
        probability = float(model.predict_proba(order[FEATURES])[:, 1][0])
        novelty = float(anomaly_risk(
            sentinel.decision_function(order[NUMERIC_FEATURES]), reference_scores
        )[0])
        risk_score = 0.8 * probability + 0.2 * novelty
        action = action_for_probability(risk_score, threshold)
        st.metric("Hybrid return/RTO risk score", f"{risk_score:.1%}")
        st.write(f"Risk band: **{risk_band(risk_score, threshold)}**")
        st.write(f"Known-pattern risk: **{probability:.1%}** (80% of score) | Blind-spot novelty: **{novelty:.1%}** (20% of score)")
        st.caption(f"Review cutoff: {threshold:.1%}. This order is {('above' if risk_score >= threshold else 'below')} the cutoff by {abs(risk_score - threshold):.1%}.")
        st.caption("The score ranks orders for attention; it is not a guaranteed return probability.")
        explanation = explain_prediction(model, order)
        positive_signals = explanation[explanation["contribution"] > 0].head(3)
        negative_signals = explanation[explanation["contribution"] < 0].sort_values("contribution").head(3)
        with st.expander("Why this order received this score", expanded=True):
            if not positive_signals.empty:
                st.write("Signals pushing risk higher:")
                st.dataframe(positive_signals.assign(contribution=positive_signals["contribution"].map(lambda value: f"+{value:.2f}")), hide_index=True, width="stretch")
            if not negative_signals.empty:
                st.write("Signals pushing risk lower:")
                st.dataframe(negative_signals.assign(contribution=negative_signals["contribution"].map(lambda value: f"{value:.2f}")), hide_index=True, width="stretch")
            st.caption("Contributions show how the known-pattern classifier moved this prediction relative to its baseline. They are directional signals, not causal proof.")
        st.warning(action)
        st.info("Suggested workflow: verify payment/address details or use OTP/manual review, then record the outcome. This score supports that workflow; it does not make an autonomous decision.")

with data_tab:
    st.subheader("Source data and label balance")
    data_summary = st.columns(4)
    data_summary[0].metric("Orders", f"{len(data):,}")
    data_summary[1].metric("High-risk orders", f"{int(data['target'].sum()):,}")
    data_summary[2].metric("High-risk rate", f"{data['target'].mean():.1%}")
    data_summary[3].metric("Missing values", f"{int(data[FEATURES + ['High_Return_Risk']].isna().sum().sum()):,}")
    st.caption("The supplied dataset has 5,000 rows, balanced labels, no missing values in model fields, and no usable time variation. The label should be documented as historical or synthetic before production use.")
    st.dataframe(data[FEATURES + ["High_Return_Risk"]].head(100), width="stretch")
    st.bar_chart(data["High_Return_Risk"].value_counts())
    st.download_button(
        "Download source sample",
        data=data[FEATURES + ["High_Return_Risk"]].head(1000).to_csv(index=False),
        file_name="preship_return_risk_sample.csv",
        mime="text/csv",
    )
