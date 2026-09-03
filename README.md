# PreShip AI Risk Intelligence Platform

PreShip AI is a defense-only return and RTO risk monitoring system for digital commerce operations. It helps merchants prioritize verification for high-risk orders while keeping the final decision within a human review workflow.

**Version:** 2.0 — production-oriented release featuring calibration, uncertainty quantification, explainability, active learning, and operational monitoring.

## Executive Summary

Returns, RTO losses, and chargeback exposure materially reduce margin in online commerce. This project provides a transparent, measurable decision-support layer for order review: it estimates risk, quantifies uncertainty, explains the drivers behind each score, and supports operational monitoring over time.

## Product Scope

- Scores an order using customer, product, price, discount, quantity, rating, and state signals.
- Combines a calibrated supervised return-risk classifier with an Isolation Forest novelty signal.
- Quantifies prediction confidence using split conformal prediction.
- Explains feature contributions using SHAP values (with LR coefficient fallback).
- Generates plain-English summaries of risk drivers (using local Ollama LLM qwen2.5:7b).
- Tunes the review cutoff using configurable false-positive and false-negative costs.
- Monitors data drift, detects subgroup parity issues, and compares alternative models.
- Collects human feedback and supports iterative retraining.
- Recommends a workflow: proceed, monitor, lightweight check, or manual review — with confidence-aware actions.
- Shows precision-recall (primary), ROC-AUC (secondary), confusion costs, calibration metrics, threshold trade-offs, and planning-level savings.
- Keeps the system defense-only: it scores and recommends verification; it never automatically rejects a customer or order.

## Dashboard

The Streamlit dashboard has seven working areas:

- **Evaluation:** Precision-Recall curve (primary), ROC-AUC (secondary), calibration metrics, business impact assumptions, cutoff trade-offs, limitations, and model inputs.
- **Score an order:** enter an order and inspect its risk band, score components, confidence level, SHAP contributions, plain-English summary, and record feedback.
- **Dataset:** inspect the source sample, label balance, data quality, and download a sample CSV.
- **Data Diagnostics:** class-means analysis, mutual information ranking, signal quality warning, label balance status, and relational feature readiness.
- **Drift Monitor:** upload new batch CSV to compute Population Stability Index (PSI) per feature; inspect subgroup flag-rate parity.
- **Model Comparison:** (optional) side-by-side evaluation of Logistic Regression vs. LightGBM (toggle via sidebar).
- **Retrain & Feedback:** collect human review outcomes, track feedback statistics, and retrain with labeled feedback.
- **Sidebar controls:** change false-positive cost, missed-risk loss, verification cost, and planning volume to see how the review policy changes; toggle model selection.

## Current Evaluation

The supplied dataset contains 5,000 rows and a 50.92% positive high-risk label rate. **⚠️ This dataset shows near-random separation (ROC-AUC ≈ 0.51), suggesting weak signal or label quality issues. Results should be treated as a baseline demonstration, not predictive evidence.** With the default assumptions of ₹150 false-positive cost and ₹350 false-negative cost:

| Metric | Held-out result |
|---|---:|
| Train / validation / test split | 60% / 20% / 20% |
| Cost-tuned threshold | 0.05 |
| Precision (at threshold) | 50.9% |
| Recall (at threshold) | 100.0% |
| AUC-PR | 0.514 |
| ROC-AUC | 0.513 |
| ECE (calibration) | 0.087 → 0.042 |
| False positives | 491 |
| False negatives | 0 |

### Confusion Matrix

The complete confusion matrix on the final held-out test set is:

|  | Predicted safe | Predicted high-risk |
|---|---:|---:|
| **Actual safe** | 0 (TN) | 491 (FP) |
| **Actual high-risk** | 0 (FN) | 509 (TP) |

This means the current cost-selected threshold flags all 1,000 test orders. It achieves 100% recall but sends every safe test order to review, producing 0 true negatives and 491 false positives. This is an honest baseline, not a production-ready operating point. The threshold and model should be recalibrated using real merchant costs and stronger timestamped outcome data.

These are benchmark results, not a production performance guarantee. The `0.05` cutoff means the configured costs strongly favor catching every labeled risky order, resulting in a very high review volume. Merchants should set costs from measured operational data.

### What Changed in v2.0

- ✅ Classifier probabilities are now calibrated (Expected Calibration Error reduced from 0.087 to 0.042).
- ✅ Prediction confidence is quantified using split conformal prediction.
- ✅ Feature contributions are explained via SHAP values (with LR coefficient fallback if SHAP unavailable).
- ✅ Plain-English summaries are generated via local Ollama LLM qwen2.5:7b (or deterministic template fallback if not running).
- ✅ Data diagnostics tab exposes signal quality and mutual information rankings.
- ✅ Drift monitoring tab allows PSI analysis and subgroup parity audits.
- ✅ Model comparison tab (optional) enables side-by-side LightGBM vs. LR evaluation.
- ✅ Active learning: feedback collection and iterative retraining with outcome tracking.
- ✅ Precision-Recall is now primary (Evaluation tab), ROC-AUC is secondary.
- ✅ Recommendations are now confidence-aware (high-confidence vs. borderline cases get different actions).

**None of these features change the fundamental signal limitation.** The near-zero AUC-PR and ROC-AUC indicate that the data or labels lack strong predictive separation. V2.0 makes this limitation visible and provides the infrastructure for iterative improvement once better data is available.

## Important Data Limitation

The CSV includes a timestamp column, but all rows have the same timestamp. A genuine future-based temporal split cannot therefore be verified. The app discloses this and uses a stratified 60/20/20 train, validation, and test split. Before production use, replace the demo data with timestamped historical orders and evaluate on a strictly later period.

The supplied file also appears to be a benchmark/demo dataset. Confirm and document how `high_return_risk` was labeled before using this system for real customer decisions.

## Model Design

### Pipeline (v2.0)

1. **Preprocessing**
   - Numeric features: `Age`, `Quantity`, `Price`, `Discount`, and `Product Rating` → median-imputed and standardized.
   - Categorical features: `Gender`, `State`, `Category`, and `Brand` → most-frequent-imputed and one-hot encoded.
   - Unknown categories at scoring time are ignored safely by the encoder.

2. **Calibrated Classifier**
   - Base model: class-weighted Logistic Regression.
   - Calibration: wrapped in `CalibratedClassifierCV` (isotonic method) using validation data.
   - Output: calibrated probability estimates instead of raw logits.

3. **Novelty Detection**
   - Isolation Forest fitted to numeric training fields.
   - Decision-function rank converted to 0–1 novelty signal.
   - Identifies unusual numeric profiles but does not identify intent or prove fraud.

4. **Hybrid Score**
   ```
   hybrid score = 0.8 × calibrated_classifier_risk + 0.2 × numeric_novelty_risk
   ```
   Prioritization signal; not a calibrated probability that an individual order will be returned.

5. **Uncertainty Quantification**
   - Split conformal prediction: computes prediction intervals for classifier probabilities.
   - Confidence level derived from distance to decision boundary within interval.
   - Recommendations are confidence-aware (high vs. low confidence).

6. **Explainability**
   - **SHAP values** (via `shap.LinearExplainer`): per-order feature contributions for classifier.
   - **Fallback:** LR coefficients if SHAP unavailable.
   - **LLM summaries** (via local Ollama qwen2.5:7b): plain-English risk driver explanations.
   - **Fallback:** deterministic template if Ollama is not running.

### Cutoff Selection

The review cutoff is selected on the validation set by minimizing:

```
false-positive cost × false positives
+ false-negative cost × false negatives
```

The final test set is used for reporting and is not used to choose the cutoff.

## Run Locally

For the local demo, this workspace includes a Windows Python environment in `PreShipAIpython`. That environment is excluded from GitHub by `.gitignore`; use `requirements.txt` to recreate it elsewhere.

### Prerequisites: Install Ollama (for LLM summaries)

The app uses a local Ollama LLM (qwen2.5:7b) for natural-language order summaries. This is optional; if Ollama is not running, the app will show a fallback template explanation.

**To enable local LLM summaries:**

1. Download and install Ollama from [ollama.ai](https://ollama.ai)
2. In a separate terminal, start the Ollama server:
   ```bash
   ollama serve
   ```
3. In another terminal, pull the qwen2.5:7b model:
   ```bash
   ollama pull qwen2.5:7b
   ```
4. The Streamlit app will auto-detect Ollama and show status in the sidebar.

**If Ollama is not running:**
- LLM summaries will use the deterministic template fallback (still useful, but less natural-sounding).
- The app continues to work normally.

### Run the Streamlit app

```powershell
.\PreShipAIpython\Scripts\streamlit.exe run risk_manager_app.py
```

Then open the local URL printed by Streamlit, normally `http://localhost:8501`.

To use the same interpreter in VS Code, select:

```text
PreShipAIpython\python.exe
```

## Repository Layout

```text
risk_manager_app.py                  Streamlit dashboard (v2.0)
data/Export_Product_Return_Data.csv  Input dataset
models/                               Saved model bundle and metadata

# v2.0 Support Modules
calibration.py                       Classifier calibration and reliability diagrams
conformal.py                         Split conformal prediction and confidence quantification
shap_explain.py                      SHAP value computation and interpretation
llm_explain.py                       LLM-based natural language explanations
data_diagnostics.py                  Signal quality analysis and mutual information
drift.py                             PSI drift monitoring and subgroup parity
model_comparison.py                  Tree model training and evaluation
relational_features.py               Extension point for graph-based features (not yet available in current data)
feedback.py                          SQLite feedback collection and active learning
feedback.db                          Feedback database (created at runtime)

GOALS.md                             Product goals and judging notes
FINAL_REPORT.md                      Detailed methodology and final assessment (updated for v2.0)
requirements.txt                     Python dependencies
```

## Responsible Use

This is a decision-support tool for loss prevention. A high score is not proof of fraud, abuse, or customer intent. Do not use the score as the sole basis for denial. Use proportionate verification, retain an appeal path, monitor subgroup performance, and record human review outcomes.

## Recommended Next Steps

### Immediate (Address Signal Limitation)

- **Verify label quality:** confirm that `high_return_risk` is a true outcome label (actual returns/RTO), not a heuristic proxy.
- **Improve data provenance:** collect timestamped historical orders with verified outcomes (not all same timestamp).
- **Add operational features:** payment method, account age, order velocity per customer, address/pincode mismatch, device fingerprints (if available).

### Short-term (Production Readiness)

- Replace current demo data with real merchant historical orders.
- Define and audit the return/RTO label using actual business data.
- Implement a time-based train/val/test split (older → newer).
- Retrain calibration and conformal intervals on production data.
- Set false-positive and false-negative costs using measured operational losses.
- Add subgroup fairness audits (gender, income, region parity).

### Medium-term (Monitoring & Continuous Improvement)

- Implement post-deployment precision/recall monitoring on new orders.
- Track feedback loop: monitor whether human review outcomes match predictions.
- Set up automated retraining triggers (drift > threshold, performance degradation, new feedback batch size).
- Add A/B testing for model updates.
- Monitor confidence calibration and reliability diagram drift.

### Long-term (Feature & Model Expansion)

- Add relational features (order velocity, address duplication, phone duplication, first-time buyer) — **highest priority for fraud signal**.
- Compare against stronger models (gradient boosting) once data quality is confirmed.
- Integrate device fingerprinting and IP-based signals where available.
- Implement custom SHAP background datasets for improved feature attribution.
- Explore hierarchical models (category-specific risk, brand risk profiles).

## License

The local bundled Python environment is excluded from the repository. If it is distributed separately, see `PreShipAIpython/LICENSE_PYTHON.txt` for its Python distribution license information.
