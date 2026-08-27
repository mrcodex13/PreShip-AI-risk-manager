# PreShip AI Risk Manager
## Final Project Report

**Track:** AI Risk Manager  
**Product:** Defense-only return/RTO risk scorer and verification assistant  
**Date:** 28 August 2026

## 1. Executive Summary

PreShip AI is a working Streamlit application that helps an e-commerce merchant prioritize orders for return/RTO verification before dispatch. It combines a supervised classifier with a numeric novelty detector, tunes the review cutoff against explicit business costs, and explains the signals behind each prediction.

The system is intentionally defense-only. It does not collect offensive intelligence, automate abuse, or autonomously deny customers. Its output is a risk signal and a recommended human workflow.

## 2. Problem and User

Returns and reverse transport orders reduce margin through logistics, handling, restocking, and lost selling time. The primary user is a merchant or operations reviewer who needs to answer:

- Which orders should receive additional verification?
- How much review volume can the business afford?
- What trade-off exists between missed risky orders and unnecessary verification?
- Why did the model flag this order?

## 3. Implemented Product

The dashboard provides:

- Single-order risk scoring.
- Low, Watch, and High risk bands based on the live cost-tuned threshold.
- Known-pattern classifier risk and numeric novelty risk breakdown.
- Directional feature contributions for individual predictions.
- Cost-sensitive threshold selection.
- Precision, recall, AUC-PR, ROC-AUC, false-positive count, and false-negative count.
- Threshold trade-off chart for precision and recall.
- Planning-level loss and savings simulation.
- Dataset size, label rate, missing-value, and limitation summaries.
- Human verification guidance and defense-only notices.

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

### Preprocessing

- Numeric features: `Age`, `Quantity`, `Price`, `Discount`, and `Product Rating`.
- Categorical features: `Gender`, `State`, `Category`, and `Brand`.
- Numeric missing values are median-imputed and standardized.
- Categorical missing values are filled with the most frequent value and one-hot encoded.
- Unknown categories at scoring time are ignored safely by the encoder.

### Supervised Model

The known-pattern model is class-weighted Logistic Regression. Class weighting is used because missing a risky order and incorrectly reviewing a safe order have different business consequences.

### Novelty Signal

An Isolation Forest is fitted to numeric training fields. Its decision-function rank is converted into a 0 to 1 novelty signal. This catches unusual numeric profiles, but it does not identify intent and does not prove fraud.

### Hybrid Score

```text
hybrid score = 0.8 × classifier risk + 0.2 × novelty risk
```

This is a prioritization score. It should not be described as a calibrated individual-order return probability.

### Cutoff Selection

The review cutoff is selected on the validation set by minimizing:

```text
false-positive cost × false positives
+ false-negative cost × false negatives
```

The final test set is used for reporting and is not used to choose the cutoff.

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

## 8. Business Impact Model

The dashboard lets a merchant configure:

- Verification cost per flagged order.
- Loss when a risky order is missed.
- Planning volume.

It then estimates baseline loss, model loss, and savings for the planning volume. This is a scenario model, not realized savings. It assumes that verification cost applies to each flagged order and that every missed risky order has the configured loss. Those assumptions must be replaced with observed merchant costs.

## 9. Explainability and Safety

For each scored order, the app shows:

- Classifier risk.
- Numeric novelty risk.
- Contribution signals that moved the classifier prediction higher or lower.
- Distance above or below the review cutoff.
- A recommended verification action.

Contributions are directional model explanations, not causal evidence. A reviewer must not treat any single feature as proof of wrongdoing.

Safety controls include:

- Human-in-the-loop final decisions.
- No autonomous denial.
- No offensive or exploit-generating functionality.
- Explicit score and dataset limitations.
- Configurable costs instead of a hidden hard-coded business policy.

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

## 12. Final Assessment

PreShip AI meets the core product shape of the AI Risk Manager track: it is a working return-risk detector with held-out evaluation, explicit false-positive cost, a usable merchant workflow, explainability, and defense-only safeguards.

Its current measured predictive quality is a baseline rather than a finished production model. The strongest and most credible pitch is therefore not that the model is already highly accurate. The strongest pitch is that the project exposes the full risk-management loop, makes its trade-offs visible, refuses to overclaim from weak data, and identifies the exact evidence required for a production-grade next iteration.
