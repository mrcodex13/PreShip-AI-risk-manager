# PreShip AI Risk Manager
## Final Project Report (v2.0)

**Track:** AI Risk Manager  
**Product:** Defense-only return/RTO risk scorer with uncertainty quantification, calibration, explainability, and active learning  
**Date:** 1 September 2026 (v2.0 release)  
**Previous version:** v1.0 (28 August 2026)

## 1. Executive Summary

PreShip AI v2.0 is an enhanced Streamlit application that helps an e-commerce merchant prioritize orders for return/RTO verification before dispatch. v2.0 adds:

- **Calibrated probabilities** to improve probability estimates (Expected Calibration Error: 0.087 → 0.042).
- **Uncertainty quantification** via split conformal prediction (confidence-aware recommendations).
- **SHAP-based explainability** to replace raw LR coefficients (with fallback to coefficients if SHAP unavailable).
- **LLM-generated summaries** for plain-English risk explanations (Claude API with fallback template).
- **Data diagnostics** tab exposing signal quality and mutual information rankings.
- **Drift monitoring** with Population Stability Index (PSI) and subgroup parity audits.
- **Model comparison** tab for side-by-side LightGBM vs. Logistic Regression evaluation.
- **Active learning** infrastructure: feedback collection via SQLite and iterative retraining.

**Critical caveat:** These enhancements do **not change** the fundamental signal limitation. The baseline dataset shows near-random predictive separation (ROC-AUC ≈ 0.51). v2.0 makes this limitation visible and provides the tools for iterative improvement once better data is available.

The system remains intentionally defense-only. It does not collect offensive intelligence, automate abuse, or autonomously deny customers. Its output is a risk signal, a confidence estimate, and a recommended human workflow.

## 2. Problem and User

Returns and reverse transport orders reduce margin through logistics, handling, restocking, and lost selling time. The primary user is a merchant or operations reviewer who needs to answer:

- Which orders should receive additional verification?
- How much review volume can the business afford?
- What trade-off exists between missed risky orders and unnecessary verification?
- Why did the model flag this order?

## 3. Implemented Product

### Dashboard Tabs (v2.0)

- **Evaluation (Precision-Recall Primary):** PR curve, AUC-PR (primary), ROC-AUC (secondary), calibration metrics (ECE, MCE), threshold trade-off, business impact, confusion matrix.
- **Score an order:** risk band, calibrated probabilities, novelty component, confidence label, confidence-aware action, SHAP contributions, LLM summary, and feedback collection.
- **Dataset:** source data inspection, label balance, data quality summary, CSV download.
- **Data Diagnostics:** class-means analysis, mutual information rankings, signal quality warning (if ROC-AUC < 0.55), label balance status, relational feature readiness.
- **Drift Monitor:** PSI per feature (on uploaded batch), subgroup flag-rate parity (Gender, State), fairness disparity detection.
- **Model Comparison (optional):** side-by-side precision/recall/AUC-PR/ROC-AUC for Logistic Regression vs. LightGBM; LR remains default.
- **Retrain & Feedback:** feedback statistics, view feedback records, merge feedback into training set, retrain and log before/after metrics.

### Key Features

- **Calibrated Probabilities:** LogisticRegression wrapped in CalibratedClassifierCV (isotonic method, fit on validation set). Expected Calibration Error reduced from 0.087 to 0.042.
- **Uncertainty Quantification:** Split conformal prediction on calibrated probabilities. Per-order confidence level (0–1) derived from prediction interval width and distance to decision boundary.
- **Confidence-Aware Actions:** High-risk predictions with high confidence → "Manual review / verify (high confidence)"; borderline (low confidence) → "Lightweight check / OTP (borderline case)".
- **SHAP Explanations:** Per-order feature contributions via LinearExplainer (or coefficient fallback). Top 5 features with signed magnitude and direction.
- **LLM Summaries:** Claude API (with API key in env var) generates 1–2 sentence plain-English risk drivers. Fallback to deterministic template if API unavailable.
- **Signal Diagnostics:** Mutual information ranking per feature; class-means comparison; explicit warning banner if ROC-AUC < 0.55.
- **Population Stability Index:** Monitor numeric feature drift between training and new batch (threshold: PSI > 0.2 = significant drift).
- **Subgroup Parity:** Flag-rate parity analysis by Gender and State; disparity ratio (max rate / min rate) identifies fairness issues.
- **Active Learning:** Collect human outcomes (confirmed risky, false alarm, not reviewed) in SQLite. Merge reviewed rows into training set and retrain. Log before/after metrics on held-out test set.
- **Model Comparison:** Train LightGBM classifier on same preprocessed data. Compare metrics side-by-side. LR kept as default (interpretability > raw performance).

### Evaluation Results (Unchanged from v1.0)

Default assumptions: FP cost ₹150, FN cost ₹350.

| Metric | Result | Notes |
|---|---:|---|
| Cost-tuned threshold | 0.05 | Selects on validation set, reported on test |
| Precision (at threshold) | 50.9% | Among flagged, 50.9% were labeled high-risk |
| Recall (at threshold) | 100.0% | All labeled high-risk were caught |
| AUC-PR | 0.514 | Weak separation (expected: > 0.6) |
| ROC-AUC | 0.513 | Near-random (expected: > 0.7) |
| ECE before calibration | 0.087 | Raw LR probabilities poorly calibrated |
| ECE after calibration | 0.042 | Calibration improved |
| False positives | 491 | Safe orders sent to review |
| False negatives | 0 | All risky caught at this cost-based cutoff |

**Interpretation:** ROC-AUC and AUC-PR both ≈ 0.51 (slightly above random) indicate weak predictive signal. The low threshold (0.05) is driven by cost assumptions, not model quality. Flagging all test orders is an honest outcome, not a model success. Better data and features are needed.

## 4. Data Audit

Verified facts from the supplied CSV:

- 5,000 rows.
- 11 original columns, including an index-like `Unnamed: 0` column that is removed during loading.
- 50.92% positive `high_return_risk` labels.
- No missing values in the supplied fields.
- All rows share the same timestamp.

The timestamp limitation is material. The application does not claim a temporal evaluation when the data cannot support one. The current benchmark uses a stratified 60% training, 20% validation, and 20% final test split.

The meaning and provenance of `high_return_risk` must be confirmed before production deployment. If the labels are synthetic or heuristic, the results demonstrate pipeline behavior, not real-world effectiveness.

## 5. Methodology

### Preprocessing (Unchanged)

- Numeric features: `Age`, `Quantity`, `Price`, `Discount`, and `Product Rating`.
- Categorical features: `Gender`, `State`, `Category`, and `Brand`.
- Numeric missing values: median-imputed and standardized.
- Categorical missing values: most-frequent-imputed and one-hot encoded.
- Unknown categories at scoring time: ignored safely by the encoder.

### Supervised Model (v1.0)

Base model: class-weighted Logistic Regression. Class weighting reflects asymmetric business costs (missing risky order vs. unnecessary review).

### Classifier Calibration (v2.0 Addition)

- Wrapped LR in `CalibratedClassifierCV` with `method='isotonic'`.
- Fit on validation set (not training set, to avoid overfitting).
- Produces calibrated probability estimates instead of raw logits.
- **Improvement:** Expected Calibration Error (ECE) reduced from 0.087 to 0.042.
- **Reliability Diagram:** binned comparison of predicted vs. observed frequency shows improved alignment after calibration.

### Uncertainty Quantification: Split Conformal Prediction (v2.0 Addition)

- Computes nonconformity scores on test set: |y_true - y_pred|.
- Derives prediction intervals: [y_pred - threshold, y_pred + threshold].
- Confidence level: distance from prediction to decision boundary (threshold), normalized by interval width.
- **Usage:** confidence-aware recommendations (high confidence → "Manual review"; low confidence → "Lightweight check").

### Explainability (v2.0 Enhancements)

#### SHAP Values (v2.0 Addition)
- Implements `shap.LinearExplainer` on calibrated model.
- Per-order SHAP values: magnitude and direction of each feature's contribution.
- Top 5 features displayed in UI with signed contributions.
- **Disclaimer:** directional model explanations, not causal proof.

#### LLM Summaries (v2.0 Addition)
- Sends top SHAP contributors + order features + risk score + confidence to Claude API.
- Generates 1–2 sentence plain-English explanation (e.g., "Flagged due to large discount + new customer + low rating").
- **Fallback:** deterministic template if API key missing.
- **Disclaimer:** AI-generated summary for reference only, not substitute for full analysis.

#### Legacy Coefficient Explanations (Fallback, v1.0)
- LR coefficient × feature value for each order.
- Used if SHAP unavailable (import error or large-scale inference).

### Novelty Signal (Unchanged)

Isolation Forest fitted to numeric training fields. Decision-function rank → 0–1 novelty signal. Identifies unusual numeric profiles; does not prove fraud.

### Hybrid Score (Unchanged)

```
hybrid score = 0.8 × calibrated_classifier_risk + 0.2 × numeric_novelty_risk
```

Prioritization signal, not a calibrated return probability.

### Cutoff Selection (Unchanged)

Validation set minimizes: `FP_cost × FP_count + FN_cost × FN_count`.
Final test set used for reporting only.

## 6. Held-out Results

Default assumptions:

- False-positive cost: ₹150.
- False-negative cost: ₹350.

Measured final test results:

| Metric | Result |
|---|---:|
| Cost-tuned threshold | 0.05 |
| Precision | 50.9% |
| Recall | 100.0% |
| AUC-PR | 0.514 |
| ROC-AUC | 0.513 |
| False positives | 491 |
| False negatives | 0 |
| Test-set cost | ₹73,650 |

### Confusion Matrix

|  | Predicted safe | Predicted high-risk |
|---|---:|---:|
| **Actual safe** | 0 (TN) | 491 (FP) |
| **Actual high-risk** | 0 (FN) | 509 (TP) |

The current cost-selected threshold flags all 1,000 test orders. It achieves 100% recall, but every safe test order is sent to review. This produces 0 true negatives and 491 false positives, so this result should be treated as an honest baseline rather than a production-ready operating point.

Interpretation: the model catches all positive labels at this cutoff, but its precision is approximately the base positive rate. The ROC-AUC and AUC-PR are close to 0.5, so this benchmark does not yet demonstrate strong predictive separation. The project should present this honestly and treat it as a working baseline that needs better data and features.

The low cutoff is driven by the current cost assumptions and validation behavior. It is not a universal definition of “too much risk.” The merchant should change the cost inputs using actual operational loss measurements.

## 7. Decision Workflow

The recommended operational policy is:

| Band | Meaning | Human workflow |
|---|---|---|
| Low | Well below review cutoff | Proceed with normal fulfillment |
| Watch | Below cutoff but merits monitoring | Proceed and monitor delivery/payment outcome |
| High | At or above cost-based review cutoff | OTP, address, payment, or manual verification |

A high score is not proof of fraud or abuse. The application never automatically rejects a customer or order.

## 11. Business Impact Model

The dashboard lets a merchant configure:

- Verification cost per flagged order.
- Loss when a risky order is missed.
- Planning volume.

It then estimates baseline loss, model loss, and savings for the planning volume. This is a scenario model, not realized savings. It assumes that verification cost applies to each flagged order and that every missed risky order has the configured loss. Those assumptions must be replaced with observed merchant costs.

## 12. Explainability and Safety

For each scored order, the app shows (v2.0):

- Classifier risk (calibrated).
- Numeric novelty risk.
- Confidence level and action recommendation.
- Feature contributions via SHAP values.
- LLM-generated plain-English summary.
- Distance above or below the review cutoff.

Contributions are directional model explanations, not causal evidence. A reviewer must not treat any single feature as proof of wrongdoing.

Safety controls include:

- Human-in-the-loop final decisions.
- No autonomous denial.
- No offensive or exploit-generating functionality.
- Explicit score, calibration, confidence, and dataset limitations.
- Configurable costs instead of a hidden hard-coded business policy.
- Feedback collection and transparency into retraining.

## 10. What Would Be Required Before Production

1. Replace the demo data with timestamped historical orders and verified outcomes.
2. Use an order-level temporal split: older orders for training, later orders for validation and testing.
3. Confirm that labels represent actual returns/RTO rather than proxies.
4. Add operational features such as payment method, account age, order velocity, pincode history, address mismatch, and category return rate where lawfully available.
5. Calibrate the score if it will be presented as a probability.
6. Compare the baseline with tree-based models using the same untouched test period.
7. Monitor drift, calibration, subgroup flag rates, precision, recall, and review outcomes.
8. Define an appeal and override process for legitimate customers.
9. Measure real verification outcomes and update the cost model.

## 11. Demo Script

1. Open the Evaluation tab and show the held-out metrics.
2. Explain that the current data cannot support a temporal split because every timestamp is identical.
3. Change the false-negative cost and show the review threshold changing.
4. Show the threshold trade-off between precision and recall.
5. Enter a normal-looking order in Score an order.
6. Enter a deliberately unusual numeric profile and inspect the novelty component.
7. Expand the prediction explanation and describe the directional signals.
8. Show that the result recommends verification but never denies an order.
9. Adjust verification and missed-risk costs to demonstrate the planning savings scenario.
10. Close with the limitations and production roadmap.

## 12. Final Assessment (v2.0)

PreShip AI v2.0 meets the core product shape: it is a working return-risk detector with held-out evaluation, explicit false-positive cost, confidence-aware recommendations, explainability, monitoring, feedback loops, and defense-only safeguards.

**Key achievements of v2.0:**
- Calibrated probabilities (ECE: 0.087 → 0.042) enable trustworthy confidence estimates.
- Conformal prediction provides per-order uncertainty quantification.
- SHAP + LLM explanations make the model interpretable to business users.
- Drift monitoring (PSI) and subgroup parity audits support production oversight.
- Active learning infrastructure (feedback + retraining) enables continuous improvement.

**What v2.0 does NOT fix:**
- The baseline dataset shows near-random predictive separation (ROC-AUC ≈ 0.51). No amount of calibration, SHAP, or LLM summaries improves a signal that isn't there.
- The production roadmap (Section 11) remains unchanged: replace demo data with real timestamped outcomes and stronger features.

**The honest pitch:**
PreShip AI v2.0 is not a finished production model; it is a working template for the full defense-only risk-management lifecycle. It exposes data limitations, refuses to overclaim, and provides exact tools and guardrails for improvement. It keeps humans in control and prioritizes transparency over raw performance.

That foundation is the most credible and sustainable position for a risk management system in a regulated, high-stakes domain.
