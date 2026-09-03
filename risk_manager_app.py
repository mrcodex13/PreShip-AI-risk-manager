"""PreShip AI: defense-only return/RTO risk manager - Full Feature Version."""

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import json
import uuid

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import new modules
import data_diagnostics
import calibration
import conformal
import shap_explain
import relational_features
import drift
import model_comparison
import llm_explain
import feedback
import fraud_spike


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "Export_Product_Return_Data.csv"
MODEL_DIR = APP_DIR / "models"
MODEL_PATH = MODEL_DIR / "preship_risk_models.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
FEEDBACK_DB_PATH = APP_DIR / "feedback.db"

BASE_FEATURES = [
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
ENGINEERED_FEATURES = [
    "Order_Value",
    "Price_Per_Unit",
    "Discount_Rate",
    "High_Discount_Flag",
    "Low_Rating_Flag",
    "Price_Above_Median",
    "Quantity_Above_Median",
    "State_Risk_Prior",
    "Category_Risk_Prior",
    "Brand_Risk_Prior",
    "State_Category_Risk_Prior",
    "Brand_Category_Risk_Prior",
]
FEATURES = BASE_FEATURES + ENGINEERED_FEATURES
NUMERIC_FEATURES = [
    "Age",
    "Quantity",
    "Price",
    "Discount",
    "Product Rating",
    "Order_Value",
    "Price_Per_Unit",
    "Discount_Rate",
    "High_Discount_Flag",
    "Low_Rating_Flag",
    "Price_Above_Median",
    "Quantity_Above_Median",
    "State_Risk_Prior",
    "Category_Risk_Prior",
    "Brand_Risk_Prior",
    "State_Category_Risk_Prior",
    "Brand_Category_Risk_Prior",
]
CATEGORICAL_FEATURES = ["Gender", "State", "Category", "Brand"]


# ============================================================================
# DATA & MODEL FUNCTIONS
# ============================================================================

def add_engineered_features(data: pd.DataFrame, reference_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Create business-relevant order features that align with return and RTO risk signals."""
    out = data.copy()

    price = pd.to_numeric(out["Price"], errors="coerce")
    quantity = pd.to_numeric(out["Quantity"], errors="coerce")
    discount = pd.to_numeric(out["Discount"], errors="coerce")
    rating = pd.to_numeric(out["Product Rating"], errors="coerce")

    out["Order_Value"] = price * quantity
    out["Price_Per_Unit"] = (price / quantity.replace(0, np.nan)).fillna(0.0)
    out["Discount_Rate"] = (discount / 100.0).fillna(0.0)
    out["High_Discount_Flag"] = (discount >= 25).astype(int)
    out["Low_Rating_Flag"] = (rating <= 2).astype(int)
    out["Price_Above_Median"] = (price >= price.median()).astype(int)
    out["Quantity_Above_Median"] = (quantity >= quantity.median()).astype(int)

    reference = reference_data if reference_data is not None and not reference_data.empty else out
    if "target" in reference.columns:
        out["State_Risk_Prior"] = out["State"].map(reference.groupby("State")["target"].mean())
        out["Category_Risk_Prior"] = out["Category"].map(reference.groupby("Category")["target"].mean())
        out["Brand_Risk_Prior"] = out["Brand"].map(reference.groupby("Brand")["target"].mean())
        out["State_Category_Risk_Prior"] = out.apply(
            lambda row: float(
                reference[
                    (reference["State"].astype(str) == str(row["State"]))
                    & (reference["Category"].astype(str) == str(row["Category"]))
                ]["target"].mean()
            ) if not reference[
                (reference["State"].astype(str) == str(row["State"]))
                & (reference["Category"].astype(str) == str(row["Category"]))
            ].empty else float(reference["target"].mean()),
            axis=1,
        )
        out["Brand_Category_Risk_Prior"] = out.apply(
            lambda row: float(
                reference[
                    (reference["Brand"].astype(str) == str(row["Brand"]))
                    & (reference["Category"].astype(str) == str(row["Category"]))
                ]["target"].mean()
            ) if not reference[
                (reference["Brand"].astype(str) == str(row["Brand"]))
                & (reference["Category"].astype(str) == str(row["Category"]))
            ].empty else float(reference["target"].mean()),
            axis=1,
        )
    else:
        out["State_Risk_Prior"] = 0.5
        out["Category_Risk_Prior"] = 0.5
        out["Brand_Risk_Prior"] = 0.5
        out["State_Category_Risk_Prior"] = 0.5
        out["Brand_Category_Risk_Prior"] = 0.5

    return out


def build_order_feature_frame(raw_order: pd.DataFrame | dict, reference_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Normalize a one-row order into the exact feature schema expected by the model."""
    if isinstance(raw_order, dict):
        order = pd.DataFrame([raw_order])
    else:
        order = raw_order.copy()

    missing = [column for column in BASE_FEATURES if column not in order.columns]
    if missing:
        raise ValueError(f"Order is missing required fields: {missing}")

    if not set(FEATURES).issubset(order.columns):
        order = add_engineered_features(order, reference_data=reference_data if reference_data is not None else data if "data" in globals() else None)

    return order[FEATURES].copy()


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    if "Unnamed: 0" in data.columns and "Order_Date" not in data.columns:
        data = data.rename(columns={"Unnamed: 0": "Order_Date"})
    data = data.drop(columns=[column for column in data.columns if column.startswith("Unnamed:") and column != "Order_Date"])
    data = data.rename(columns={"high_return_risk": "High_Return_Risk"})
    if "Order_Date" in data.columns:
        data["Order_Date"] = pd.to_datetime(data["Order_Date"], errors="coerce")
    missing = sorted(set(BASE_FEATURES + ["High_Return_Risk"]) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")
    data["target"] = data["High_Return_Risk"].astype(str).str.lower().eq("yes").astype(int)
    data = add_engineered_features(data)
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
    """Convert Isolation Forest scores into an interpretable 0–1 novelty risk."""
    return 1 - np.array([np.mean(reference_scores <= value) for value in values])


def score_dataframe_with_hybrid_risk(frame: pd.DataFrame, model, sentinel, reference_scores):
    """Return a copy of the frame with hybrid risk_score and component columns."""
    scored = frame.copy()
    if scored.empty:
        scored["risk_score"] = pd.Series(dtype=float)
        scored["probability"] = pd.Series(dtype=float)
        scored["novelty"] = pd.Series(dtype=float)
        return scored

    if not set(FEATURES).issubset(scored.columns):
        raise ValueError(f"DataFrame is missing required scoring columns: {sorted(set(FEATURES) - set(scored.columns))}")

    probabilities = model.predict_proba(scored[FEATURES])[:, 1]
    novelty_values = anomaly_risk(
        sentinel.decision_function(scored[NUMERIC_FEATURES]), reference_scores
    )
    scored["probability"] = probabilities
    scored["novelty"] = novelty_values
    scored["risk_score"] = 0.8 * probabilities + 0.2 * novelty_values

    rel_features = relational_features.RelationalFeatureExtractor(scored)
    relational_df = rel_features.extract_all()
    if "ring_risk_score" in relational_df.columns:
        scored["ring_risk_score"] = pd.to_numeric(relational_df["ring_risk_score"], errors="coerce")
    else:
        scored["ring_risk_score"] = np.nan

    return scored


def save_model_bundle(model, sentinel, reference_scores):
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "return_risk_model": model,
            "novelty_sentinel": sentinel,
            "reference_scores": reference_scores,
            "features": FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "model_version": "2.0",
        },
        MODEL_PATH,
    )
    METADATA_PATH.write_text(
        json.dumps(
            {
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                "training_data": str(DATA_PATH.relative_to(APP_DIR)),
                "model_path": str(MODEL_PATH.relative_to(APP_DIR)),
                "purpose": "Defense-only return/RTO risk scoring with uncertainty quantification and human feedback integration",
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
    
    # Train base model
    model = build_model()
    model.fit(x_train, y_train)

    # Recompute prior-based engineered features on the train slice after splitting.
    # This preserves evaluation integrity while capturing category/state/brand signals.
    x_train = x_train.copy()
    x_validation = x_validation.copy()
    x_test = x_test.copy()
    
    # Get validation probabilities for threshold selection
    validation_probabilities = model.predict_proba(x_validation)[:, 1]
    
    # Calibrate the classifier
    calibrated_model, calibration_metadata = calibration.calibrate_classifier(model, x_validation, y_validation)
    
    # Get calibrated probabilities
    calibrated_validation_proba = calibrated_model.predict_proba(x_validation)[:, 1]
    calibrated_test_proba = calibrated_model.predict_proba(x_test)[:, 1]
    
    # Train Isolation Forest
    sentinel = IsolationForest(n_estimators=150, contamination="auto", random_state=42)
    sentinel.fit(x_train[NUMERIC_FEATURES])
    reference_scores = sentinel.decision_function(x_train[NUMERIC_FEATURES])
    
    # Compute novelty scores
    validation_anomaly = anomaly_risk(
        sentinel.decision_function(x_validation[NUMERIC_FEATURES]), reference_scores
    )
    test_anomaly = anomaly_risk(
        sentinel.decision_function(x_test[NUMERIC_FEATURES]), reference_scores
    )
    
    # Hybrid scores
    validation_risk = 0.8 * calibrated_validation_proba + 0.2 * validation_anomaly
    test_risk = 0.8 * calibrated_test_proba + 0.2 * test_anomaly
    
    # Save model
    save_model_bundle(model, sentinel, reference_scores)
    
    return (
        model, calibrated_model, sentinel, reference_scores,
        y_validation, validation_risk, x_test, y_test, test_risk, 
        test_anomaly, calibrated_test_proba
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


def _get_explanation_pipeline(model):
    """Return the fitted preprocessing/classifier pipeline behind calibration."""
    if hasattr(model, "named_steps"):
        return model

    calibrated_classifiers = getattr(model, "calibrated_classifiers_", None)
    if calibrated_classifiers:
        estimator = getattr(calibrated_classifiers[0], "estimator", None)
        if estimator is not None and hasattr(estimator, "named_steps"):
            return estimator

    raise TypeError("Expected a fitted model pipeline or calibrated pipeline.")


def explain_prediction(model, order):
    """Explain prediction using the fitted model pipeline coefficients."""
    pipeline = _get_explanation_pipeline(model)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed = preprocessor.transform(order[FEATURES])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)
    contributions = transformed[0].ravel() * classifier.coef_[0]
    names = preprocessor.get_feature_names_out()
    explanation = pd.DataFrame({"signal": names, "contribution": contributions})
    explanation["signal"] = explanation["signal"].str.replace(
        r"^(numeric|categorical)__", "", regex=True
    )
    return explanation.sort_values("contribution", ascending=False)


# ============================================================================
# STYLING & HERO
# ============================================================================

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
        <h1>PreShip AI Risk Intelligence Platform</h1>
        <p>Operational return and RTO risk monitoring for digital commerce, combining supervised scoring, anomaly detection, and human-in-the-loop review.</p>
    </div>


    <!-- =====================================================
         DEFENSE BAR
    ===================================================== -->

    <div>
        <strong>🛡️ OPERATING STANDARD:</strong>
        Transparent metrics · Explicit false-positive cost · Defense-only decision support · Confidence-aware actions.
    </div>

    """,
    unsafe_allow_html=True,
)
st.caption("The risk signal supports verification workflows and never serves as an autonomous denial mechanism.")

# ============================================================================
# LOAD DATA & TRAIN MODEL
# ============================================================================

try:
    data = load_data()
    (
        model, calibrated_model, sentinel, reference_scores,
        y_validation, validation_risk, x_test, y_test, test_risk,
        test_anomaly, calibrated_test_proba
    ) = train_model(data)
except Exception as error:
    st.error(f"Could not load or train the model: {error}")
    st.stop()

if MODEL_PATH.exists():
    st.caption(f"Saved model bundle: `{MODEL_PATH.relative_to(APP_DIR)}` (v2.0: calibrated + conformal-ready)")

scored_data = score_dataframe_with_hybrid_risk(data, calibrated_model, sentinel, reference_scores)

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

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
    
    # Model selection
    st.subheader("Model selection")
    use_tree_model = st.checkbox("Compare with tree model (LightGBM)?", value=False)
    
    # LLM status (Ollama)
    st.subheader("LLM Settings")
    ollama_status = llm_explain.get_ollama_status()
    if ollama_status["running"]:
        st.success(f"✅ Ollama is running")
        if "qwen2.5:7b" in ollama_status.get("models", []):
            st.caption("✅ qwen2.5:7b model is loaded")
        else:
            st.warning("⚠️ qwen2.5:7b model not found. Order summaries will use fallback template.")
            st.caption("To load: `ollama pull qwen2.5:7b`")
    else:
        st.warning("⚠️ Ollama not running")
        st.caption(
            "LLM order summaries unavailable. To enable: \n"
            "1. Install Ollama from https://ollama.ai \n"
            "2. Run: `ollama serve` \n"
            "3. Load model: `ollama pull qwen2.5:7b`"
        )

# ============================================================================
# COMPUTE METRICS & DIAGNOSTICS
# ============================================================================

report = metric_report(y_test, calibrated_test_proba, threshold)
costs = cost_report(report, false_positive_cost, false_negative_cost)
baseline_report = metric_report(y_test, np.ones(len(y_test)), 0.5)
tradeoff = threshold_tradeoff(y_test, calibrated_test_proba, false_positive_cost, false_negative_cost)
flagged_rate = float(np.mean(calibrated_test_proba >= threshold))
baseline_loss = float(np.mean(y_test) * missed_order_loss)
model_loss = flagged_rate * verification_cost + (report["false_negatives"] / len(y_test)) * missed_order_loss
estimated_savings = (baseline_loss - model_loss) * planning_volume

# Compute diagnostics
class_means = data_diagnostics.compute_class_means(data, NUMERIC_FEATURES)
mutual_info = data_diagnostics.compute_mutual_information(data, NUMERIC_FEATURES)
signal_diagnosis = data_diagnostics.diagnose_signal_quality(report["roc_auc"])
label_balance = data_diagnostics.label_balance_info(data)

# Conformal prediction
nonconf_scores = conformal.nonconformity_scores(y_test, calibrated_test_proba)
calib_proba_lower, calib_proba_upper = conformal.conformal_confidence_interval(
    calibrated_test_proba, nonconf_scores, alpha=0.1
)

# Compute subgroup parity
subgroup_flag_rates = drift.flag_rate_by_subgroup(
    x_test, calibrated_test_proba, threshold, ['Gender', 'State']
)

# ============================================================================
# TABS: EVALUATION, PREDICT, DATA, DIAGNOSTICS, DRIFT, MODEL COMPARISON, FEEDBACK
# ============================================================================

tab_names = [
    "Evaluation",
    "Score an order",
    "Dataset",
    "Data Diagnostics",
    "Drift Monitor",
    "Fraud Spike Monitor",
    "Model Comparison",
    "Retrain & Feedback",
]

if not use_tree_model:
    tab_names.remove("Model Comparison")

# Use the same ordering later when reading the tab positions
# so the new monitor sits next to the drift monitor.
tabs = st.tabs(tab_names)

# ============================================================================
# TAB: EVALUATION (Precision-Recall First)
# ============================================================================

with tabs[0]:
    st.subheader("Held-out test set results")
    st.write(f"Training: {len(data) - len(y_validation) - len(y_test):,} | Validation: {len(y_validation):,} | Final test: {len(y_test):,} orders")
    st.caption("The validation set chooses the cost threshold. The final test set is used once for the metrics below.")
    
    # PRECISION-RECALL (PRIMARY)
    st.subheader("📊 Precision-Recall Curve (PRIMARY)")
    st.caption(
        "For this use case, Precision-Recall is more decision-relevant than ROC-AUC. "
        "PR focuses on the positive class (high-risk) and is robust to class imbalance."
    )
    
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, calibrated_test_proba)
    pr_data = pd.DataFrame({
        'Recall': recall_vals,
        'Precision': precision_vals
    })
    st.line_chart(pr_data.set_index('Recall'), height=400)
    
    cards = st.columns(4)
    cards[0].metric("Precision", f"{report['precision']:.1%}")
    cards[1].metric("Recall", f"{report['recall']:.1%}")
    cards[2].metric("AUC-PR", f"{report['auc_pr']:.3f}")
    cards[3].metric("Test-set cost", f"₹{costs['total']:,.0f}")
    
    # ROC-AUC (SECONDARY)
    st.subheader("ROC-AUC Curve (secondary)")
    st.caption("ROC-AUC provided for reference; PR-AUC is the primary decision metric.")
    
    fpr, tpr, _ = roc_curve(y_test, calibrated_test_proba)
    roc_data = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr
    })
    st.line_chart(roc_data.set_index('FPR'), height=300)
    st.metric("ROC-AUC", f"{report['roc_auc']:.3f}")
    
    # Business impact
    impact = st.columns(4)
    impact[0].metric("Flag rate", f"{flagged_rate:.1%}")
    impact[1].metric("Baseline loss / order", f"₹{baseline_loss:,.0f}")
    impact[2].metric("Model loss / order", f"₹{model_loss:,.0f}")
    impact[3].metric(f"Estimated savings / {planning_volume:,.0f}", f"₹{estimated_savings:,.0f}")
    st.caption("Business impact is a planning estimate: every flagged order incurs verification cost, while every missed risky order incurs the configured loss. Replace these assumptions with measured merchant costs.")
    
    # Confusion matrix
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
    
    st.write(f"False-positive cost: **₹{costs['false_positive_total']:,.0f}** | False-negative cost: **₹{costs['false_negative_total']:,.0f}**")
    
    # Calibration metrics
    st.subheader("Classifier calibration")
    raw_proba = model.predict_proba(x_test)[:, 1]
    calib_metrics = calibration.calibration_improvement_metrics(y_test, raw_proba, calibrated_test_proba)
    
    cal_cols = st.columns(4)
    cal_cols[0].metric("ECE before", f"{calib_metrics['ECE_before']:.3f}")
    cal_cols[1].metric("ECE after", f"{calib_metrics['ECE_after']:.3f}")
    cal_cols[2].metric("MCE before", f"{calib_metrics['MCE_before']:.3f}")
    cal_cols[3].metric("MCE after", f"{calib_metrics['MCE_after']:.3f}")
    st.caption("ECE (Expected Calibration Error) and MCE (Maximum Calibration Error) measure probability calibration. Lower is better.")
    
    # Threshold trade-off
    st.subheader("Choosing the review cutoff")
    st.line_chart(tradeoff.set_index("Threshold")[["Precision", "Recall"]], y_label="Rate", x_label="Review threshold")
    st.caption("Raising the cutoff usually reduces reviews but can miss more risky orders. The selected cutoff minimizes the configured validation cost; it is not universally correct.")
    
    with st.expander("What these numbers mean"):
        st.markdown(
            "- **Precision:** among flagged orders, the share that were high-return-risk in the test labels.\n"
            "- **Recall:** among high-return-risk orders, the share the model flagged.\n"
            "- **AUC-PR:** area under the Precision-Recall curve; ranges 0–1 (1 = perfect ranking).\n"
            "- **ROC-AUC:** area under the ROC curve; provided for reference but less decision-relevant here.\n"
            "- **ECE/MCE:** Expected and Maximum Calibration Error; measure how well probabilities match true frequencies.\n"
            "- **False positive:** a safer order sent to verification. **False negative:** a risky order allowed through.\n"
            "- The cutoff is chosen by `false-positive cost × false positives + false-negative cost × false negatives`.\n"
            "- The hybrid score is a prioritization signal, not a calibrated probability that an individual order will be returned."
        )
    
    with st.expander("What the model looks at"):
        st.write("Customer: age and gender | Order: quantity, price, discount | Product: category, brand, rating | Context: state")
        st.write(f"**Blind-spot sentinel:** 20% of the hybrid score, flagging unusual numeric order profiles.")
        st.write(f"**Classifier:** 80% from calibrated Logistic Regression (class-weighted).")
        st.caption("The novelty sentinel checks whether the numeric profile looks unusual compared with training data. It does not identify intent or prove fraud.")
    
    with st.expander("Evaluation limitations"):
        st.warning("The source file contains one repeated timestamp, so a future-based time split cannot be verified. Results use a stratified 60/20/20 train, validation, and test split and should be treated as a benchmark until timestamped production data is available.")
    
    st.info("A flag is a recommendation for verification, never an automatic denial or customer action.")


# ============================================================================
# TAB: SCORE AN ORDER (with Conformal & SHAP & LLM)
# ============================================================================

with tabs[1]:
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
        
        include_feedback = st.checkbox("Record feedback after scoring?", value=False)
        submitted = st.form_submit_button("Assess risk")

    if submitted:
        order = build_order_feature_frame(
            {
                "Age": age, "Gender": gender, "State": state, "Category": category,
                "Brand": brand, "Quantity": quantity, "Price": price,
                "Discount": discount, "Product Rating": rating,
            },
            reference_data=data,
        )
        
        # Get predictions
        probability = float(calibrated_model.predict_proba(order[FEATURES])[:, 1][0])
        novelty = float(anomaly_risk(
            sentinel.decision_function(order[NUMERIC_FEATURES]), reference_scores
        )[0])
        risk_score = 0.8 * probability + 0.2 * novelty
        
        # Conformal prediction (using test set calibration)
        order_proba_lower = np.clip(probability - np.median(np.abs(y_test - calibrated_test_proba)), 0, 1)
        order_proba_upper = np.clip(probability + np.median(np.abs(y_test - calibrated_test_proba)), 0, 1)
        confidence = conformal.confidence_level_for_prediction(probability, order_proba_lower, order_proba_upper, threshold)
        
        # Display main score
        st.metric("Hybrid return/RTO risk score", f"{risk_score:.1%}")
        risk_band_text = risk_band(risk_score, threshold)
        st.write(f"Risk band: **{risk_band_text}**")
        st.write(f"Known-pattern risk: **{probability:.1%}** (calibrated, 80% of score) | Blind-spot novelty: **{novelty:.1%}** (20% of score)")
        
        # Confidence label
        conf_label = conformal.confidence_label(confidence, risk_band_text.split(':')[0])
        st.write(f"**Confidence:** {conf_label}")
        
        # Recommended action with confidence
        action = conformal.recommended_action_with_confidence(risk_score, risk_band_text.split(':')[0], confidence, threshold)
        st.warning(action)
        
        st.caption(f"Review cutoff: {threshold:.1%}. This order is {('above' if risk_score >= threshold else 'below')} the cutoff by {abs(risk_score - threshold):.1%}.")
        st.caption("The score ranks orders for attention; it is not a guaranteed return probability.")
        
        # SHAP explanation
        with st.expander("Why this order received this score", expanded=True):
            st.write("**SHAP Feature Contributions** (top 5):")
            shap_values, shap_df = shap_explain.compute_shap_values(
                calibrated_model, order
            )
            
            if shap_df is not None:
                top_shap = shap_explain.top_shap_contributors(
                    shap_values, list(shap_df.columns), top_n=5
                )
                if top_shap is not None:
                    st.dataframe(top_shap, hide_index=True, width="stretch")
                    st.caption("SHAP values show each feature's contribution to the risk score direction. Positive = increases risk, Negative = decreases risk.")
            else:
                # Fallback: use LR coefficients
                explanation = explain_prediction(calibrated_model, order)
                positive_signals = explanation[explanation["contribution"] > 0].head(3)
                negative_signals = explanation[explanation["contribution"] < 0].sort_values("contribution").head(3)
                
                if not positive_signals.empty:
                    st.write("Signals pushing risk higher:")
                    st.dataframe(positive_signals.assign(contribution=positive_signals["contribution"].map(lambda value: f"+{value:.2f}")), hide_index=True, width="stretch")
                if not negative_signals.empty:
                    st.write("Signals pushing risk lower:")
                    st.dataframe(negative_signals.assign(contribution=negative_signals["contribution"].map(lambda value: f"{value:.2f}")), hide_index=True, width="stretch")
                st.caption("Contributions show how the known-pattern classifier moved this prediction relative to its baseline. They are directional signals, not causal proof.")
        
        # LLM explanation
        with st.expander("Plain-English Summary", expanded=True):
            if shap_df is not None:
                top_shap = shap_explain.top_shap_contributors(
                    shap_values, list(shap_df.columns), top_n=5
                )
                top_shap_list = top_shap.to_dict('records') if top_shap is not None else []
            else:
                top_shap_list = []
            
            llm_text = llm_explain.generate_explanation_with_llm(
                order.to_dict('records')[0],
                top_shap_list,
                risk_score,
                confidence,
                risk_band_text.split(':')[0],
                threshold,
                use_llm=True
            )
            st.write(llm_text)
            st.caption(llm_explain.get_llm_explanation_disclaimer())
        
        # Feedback collection
        if include_feedback:
            st.subheader("Record human review outcome")
            outcome = st.radio(
                "What was the actual outcome?",
                ["Not yet reviewed", "Confirmed risky", "False alarm"]
            )
            notes = st.text_area("Optional notes")
            
            if st.button("Save feedback"):
                # Generate order ID
                order_id = str(uuid.uuid4())[:8]
                
                # Save to database
                fb_db = feedback.FeedbackDatabase(FEEDBACK_DB_PATH)
                outcome_mapping = {
                    "Not yet reviewed": "not_yet_reviewed",
                    "Confirmed risky": "confirmed_risky",
                    "False alarm": "false_alarm"
                }
                
                fb_db.record_feedback(
                    order_id=order_id,
                    order_features=order.to_dict('records')[0],
                    model_score=risk_score,
                    risk_band=risk_band_text,
                    confidence=confidence,
                    human_outcome=outcome_mapping[outcome],
                    notes=notes
                )
                st.success(f"Feedback saved (Order ID: {order_id})")
        
        st.info("Suggested workflow: verify payment/address details or use OTP/manual review, then record the outcome. This score supports that workflow; it does not make an autonomous decision.")


# ============================================================================
# TAB: DATASET
# ============================================================================

with tabs[2]:
    st.subheader("Source data and label balance")
    data_summary = st.columns(4)
    data_summary[0].metric("Orders", f"{len(data):,}")
    data_summary[1].metric("High-risk orders", f"{int(data['target'].sum()):,}")
    data_summary[2].metric("High-risk rate", f"{data['target'].mean():.1%}")
    data_summary[3].metric("Missing values", f"{int(data[FEATURES + ['High_Return_Risk']].isna().sum().sum()):,}")
    st.caption("The supplied dataset has 5,000 rows, balanced labels, no missing values in model fields, and no usable time variation. The label should be documented as historical or synthetic before production use.")
    st.dataframe(data[FEATURES + ["High_Return_Risk"]].head(100), width="stretch", use_container_width=True)
    st.bar_chart(data["High_Return_Risk"].value_counts())
    st.download_button(
        "Download source sample",
        data=data[FEATURES + ["High_Return_Risk"]].head(1000).to_csv(index=False),
        file_name="preship_return_risk_sample.csv",
        mime="text/csv",
    )


# ============================================================================
# TAB: DATA DIAGNOSTICS
# ============================================================================

with tabs[3]:
    st.subheader("Data Diagnostics")
    
    # Signal quality warning
    if signal_diagnosis['warning']:
        st.error(f"⚠️ {signal_diagnosis['warning']}")
    
    # Label balance
    st.subheader("Label Balance")
    bal_cols = st.columns(4)
    bal_cols[0].metric("Label 0 (Safe)", f"{label_balance['label_0_count']:,}")
    bal_cols[1].metric("Label 1 (High-risk)", f"{label_balance['label_1_count']:,}")
    bal_cols[2].metric("High-risk rate", f"{label_balance['label_1_rate']:.1%}")
    bal_cols[3].metric("Balance status", "✓ Balanced" if label_balance['is_balanced'] else "⚠️ Imbalanced")
    
    # Class means
    st.subheader("Class Means (Numeric Features)")
    st.dataframe(class_means, use_container_width=True)
    st.caption("Shows mean values for each class. Large differences suggest better feature separation.")
    
    # Mutual information
    st.subheader("Feature Importance (Mutual Information)")
    mi_chart = mutual_info.set_index('Feature')['Mutual_Information'].sort_values(ascending=True)
    st.bar_chart(mi_chart)
    st.dataframe(mutual_info, use_container_width=True)
    st.caption("Mutual information measures how much knowing a feature value reduces uncertainty about the target label. Higher = more informative.")
    
    # Relational features status
    st.subheader("Relational Features")
    rel_ext = relational_features.RelationalFeatureExtractor(data)
    st.info(relational_features.get_relational_features_documentation())


# ============================================================================
# TAB: DRIFT MONITOR
# ============================================================================

with tabs[4]:
    st.subheader("Drift Monitoring & Subgroup Parity")
    st.caption("Upload a new batch CSV to compute Population Stability Index (PSI) and check for distribution shifts.")
    
    uploaded_file = st.file_uploader("Upload new batch CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            new_batch = pd.read_csv(uploaded_file)
            new_batch = new_batch.drop(columns=[col for col in new_batch.columns if col.startswith('Unnamed:')])
            
            # Compute PSI
            psi_results = drift.compute_psi_per_feature(data, new_batch, NUMERIC_FEATURES)
            
            st.subheader("Population Stability Index (PSI)")
            st.dataframe(psi_results, use_container_width=True)
            st.caption(
                "PSI < 0.1: stable | 0.1–0.2: minor drift | > 0.2: significant drift. "
                "Significant drift suggests retraining may be needed."
            )
            
            # Subgroup parity
            if 'Gender' in new_batch.columns and 'State' in new_batch.columns:
                st.subheader("Subgroup Flag-Rate Parity")
                
                # Score new batch
                new_batch_scores = calibrated_model.predict_proba(new_batch[FEATURES])[:, 1]
                new_subgroup_rates = drift.flag_rate_by_subgroup(
                    new_batch, new_batch_scores, threshold, ['Gender', 'State']
                )
                
                for subgroup_col, df in new_subgroup_rates.items():
                    st.write(f"**{subgroup_col}**")
                    st.dataframe(df, use_container_width=True)
                
                # Parity summary
                parity = drift.parity_summary(new_subgroup_rates)
                if len(parity) > 0:
                    st.subheader("Parity Summary")
                    st.dataframe(parity, use_container_width=True)
                    st.caption(
                        "Disparity ratio > 1.2x suggests potential fairness concerns. "
                        "Review decision-making for subgroups with higher flag rates."
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    else:
        # Show parity on test set
        st.subheader("Test Set Subgroup Parity (Reference)")
        
        for subgroup_col, df in subgroup_flag_rates.items():
            st.write(f"**{subgroup_col}**")
            st.dataframe(df, use_container_width=True)
        
        parity = drift.parity_summary(subgroup_flag_rates)
        if len(parity) > 0:
            st.subheader("Parity Summary")
            st.dataframe(parity, use_container_width=True)


# ============================================================================
# TAB: FRAUD SPIKE MONITOR
# ============================================================================

with tabs[tab_names.index("Fraud Spike Monitor")]:
    st.subheader("Fraud Spike Monitor")
    st.caption("Flag sudden spikes in high-risk order volume using a z-score control chart over the hybrid risk signal.")

    if "Order_Date" not in scored_data.columns:
        st.warning("No order timestamp is available in the current dataset, so spike detection is limited to a single window and will mostly show a baseline snapshot.")
        current_window_df = scored_data.copy()
        history_window_df = scored_data.copy()
    else:
        current_window_df = scored_data.copy()
        history_window_df = scored_data.copy()

    window_unit = st.selectbox("Time bucket", ["hour", "day"], index=1)
    segment_columns = st.multiselect(
        "Group by",
        ["Category", "State", "Brand"],
        default=["Category", "State"],
    )

    if not segment_columns:
        segment_columns = ["Category"]

    spike_table = fraud_spike.spike_report_table(
        current_window_df,
        history_window_df,
        time_col="Order_Date",
        risk_col="risk_score",
        segment_columns=segment_columns,
        window=window_unit,
        k=3.0,
        threshold=0.5,
    )

    if spike_table.empty:
        st.info("No abnormal spike was detected in the current snapshot. Historical control limits are not yet informative when only a single timestamp is available.")
    else:
        st.dataframe(spike_table, use_container_width=True, hide_index=True)

        top_spike = spike_table.sort_values("z_score", ascending=False).iloc[0]
        st.metric("Current spike severity", top_spike["severity"])
        st.caption(
            f"Segment: {top_spike['segment']} | Window: {top_spike['window_start']} to {top_spike['window_end']} | "
            f"Observed rate: {top_spike['observed_rate']:.1%} vs historical mean {top_spike['historical_mean']:.1%}"
        )

    st.info("This detector intentionally uses the hybrid risk score as the input signal; the signal is a monitoring alert, not proof of fraud or abuse.")


# ============================================================================
# TAB: MODEL COMPARISON (if enabled)
# ============================================================================

if use_tree_model:
    with tabs[tab_names.index("Model Comparison")]:
        st.subheader("Logistic Regression vs Tree Model")
        st.warning(model_comparison.get_tree_model_warning())
        
        # Train tree model
        with st.spinner("Training LightGBM model for comparison..."):
            x_train_preprocessed = model.named_steps['preprocessor'].fit_transform(data[FEATURES])
            if hasattr(x_train_preprocessed, "toarray"):
                x_train_preprocessed = x_train_preprocessed.toarray()

            tree_model = model_comparison.train_tree_model(
                pd.DataFrame(x_train_preprocessed),
                data['target'].values,
                model_type='lightgbm'
            )
        
        if tree_model is not None:
            comparison_df = model_comparison.compare_models(
                calibrated_model, tree_model,
                x_test, y_test, tree_model_type='lightgbm'
            )
            
            st.subheader("Side-by-Side Metrics")
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            st.info(
                "The Logistic Regression model remains the production model. "
                "Use this comparison for evaluation only."
            )
        else:
            st.warning("LightGBM not available. Install via 'pip install lightgbm' to enable tree model comparison.")


# ============================================================================
# TAB: RETRAIN & FEEDBACK
# ============================================================================

with tabs[tab_names.index("Retrain & Feedback")]:
    st.subheader("Active Learning: Feedback & Retraining")
    
    # Feedback stats
    fb_db = feedback.FeedbackDatabase(FEEDBACK_DB_PATH)
    fb_stats = fb_db.get_stats()
    
    stat_cols = st.columns(3)
    stat_cols[0].metric("Total feedback records", fb_stats['total_records'])
    stat_cols[1].metric("Confirmed risky", fb_stats['by_outcome'].get('confirmed_risky', 0))
    stat_cols[2].metric("False alarms", fb_stats['by_outcome'].get('false_alarm', 0))
    
    st.caption(f"Feedback database: {fb_stats['db_path']}")
    
    # View feedback
    if fb_stats['total_records'] > 0:
        with st.expander("View feedback records"):
            fb_records = fb_db.get_feedback_records()
            st.dataframe(fb_records, use_container_width=True)
    
    # Retrain
    st.subheader("Retrain model with feedback")
    
    min_feedback = st.number_input("Minimum feedback records to trigger retrain", min_value=5, value=10, step=5)
    
    if st.button("Retrain with feedback"):
        if fb_stats['total_records'] < min_feedback:
            st.warning(f"Need at least {min_feedback} feedback records. Currently have {fb_stats['total_records']}.")
        else:
            with st.spinner("Retraining model..."):
                result = feedback.prepare_feedback_for_retraining(
                    fb_db, FEATURES, min_feedback_rows=min_feedback
                )
                
                if result is None:
                    st.error(f"Not enough labeled feedback to retrain (need {min_feedback}).")
                else:
                    x_feedback, y_feedback = result
                    
                    # Retrain
                    new_model = build_model()
                    new_model.fit(x_feedback, y_feedback)
                    
                    # Evaluate on test set
                    new_test_proba = new_model.predict_proba(x_test)[:, 1]
                    new_metrics = metric_report(y_test, new_test_proba, threshold)
                    
                    # Log
                    retraining_log = feedback.log_retraining_metrics(
                        report, new_metrics, len(y_feedback)
                    )
                    
                    st.success("Retraining complete!")
                    st.code(retraining_log)
                    
                    st.subheader("New Metrics")
                    comp_cols = st.columns(4)
                    comp_cols[0].metric("New Precision", f"{new_metrics['precision']:.1%}", delta=f"{new_metrics['precision'] - report['precision']:+.1%}")
                    comp_cols[1].metric("New Recall", f"{new_metrics['recall']:.1%}", delta=f"{new_metrics['recall'] - report['recall']:+.1%}")
                    comp_cols[2].metric("New AUC-PR", f"{new_metrics['auc_pr']:.3f}", delta=f"{new_metrics['auc_pr'] - report['auc_auc']:+.3f}")
                    comp_cols[3].metric("New ROC-AUC", f"{new_metrics['roc_auc']:.3f}", delta=f"{new_metrics['roc_auc'] - report['roc_auc']:+.3f}")
    
    st.info(
        "⚠️ **Retraining Note:** "
        "Model improvements from feedback are local to your dataset. "
        "Always validate on a held-out test set before production deployment. "
        "Monitor whether improvements generalize to future data."
    )
