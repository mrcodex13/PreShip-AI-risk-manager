# PreShip AI Risk Manager

A defense-only return and RTO risk-scoring dashboard for Indian e-commerce. PreShip AI helps merchants decide which orders deserve verification before dispatch, while keeping the final decision with a human reviewer.

## Problem

Returns, RTO logistics, and chargebacks quietly reduce e-commerce margin. The product challenge is to build a working detector or verifier with measured performance on held-out data, explicit false-positive cost, and no offense-capable functionality.

## What This Project Does

- Scores an order using customer, product, price, discount, quantity, rating, and state signals.
- Combines a supervised return-risk classifier with an Isolation Forest novelty signal.
- Tunes the review cutoff using configurable false-positive and false-negative costs.
- Explains which known-pattern signals pushed a prediction up or down.
- Recommends a workflow: proceed, monitor, OTP/address/payment verification, or manual review.
- Shows precision, recall, AUC-PR, ROC-AUC, confusion costs, threshold trade-offs, and planning-level savings.
- Keeps the system defense-only: it scores and recommends verification; it never automatically rejects a customer or order.

## Dashboard

The Streamlit dashboard has four working areas:

- **Evaluation:** held-out test metrics, business impact assumptions, cutoff trade-offs, limitations, and model inputs.
- **Score an order:** enter an order and inspect its risk band, score components, action, and directional feature contributions.
- **Dataset:** inspect the source sample, label balance, data quality, and download a sample CSV.
- **Sidebar controls:** change false-positive cost, missed-risk loss, verification cost, and planning volume to see how the review policy changes.

## Current Evaluation

The supplied dataset contains 5,000 rows and a 50.92% positive high-risk label rate. With the default assumptions of ₹150 false-positive cost and ₹350 false-negative cost:

| Metric | Held-out result |
|---|---:|
| Train / validation / test split | 60% / 20% / 20% |
| Cost-tuned threshold | 0.05 |
| Precision | 50.9% |
| Recall | 100.0% |
| AUC-PR | 0.514 |
| ROC-AUC | 0.513 |
| False positives | 491 |
| False negatives | 0 |

These are benchmark results, not a production performance guarantee. The `0.05` cutoff means the configured costs strongly favor catching every labeled risky order, resulting in a very high review volume. Merchants should set costs from measured operational data.

## Important Data Limitation

The CSV includes a timestamp column, but all rows have the same timestamp. A genuine future-based temporal split cannot therefore be verified. The app discloses this and uses a stratified 60/20/20 train, validation, and test split. Before production use, replace the demo data with timestamped historical orders and evaluate on a strictly later period.

The supplied file also appears to be a benchmark/demo dataset. Confirm and document how `high_return_risk` was labeled before using this system for real customer decisions.

## Model Design

1. Numeric fields are median-imputed and standardized.
2. Categorical fields are most-frequent-imputed and one-hot encoded.
3. A class-weighted Logistic Regression model estimates known-pattern risk.
4. Isolation Forest identifies unusual numeric profiles relative to the training set.
5. The displayed hybrid prioritization score is:

   `0.8 × classifier risk + 0.2 × numeric novelty risk`

The hybrid score is a ranking signal, not a calibrated probability that an individual order will be returned.

## Run Locally

For the local demo, this workspace includes a Windows Python environment in `PreShipAIpython`. That environment is excluded from GitHub by `.gitignore`; use `requirements.txt` to recreate it elsewhere.

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
risk_manager_app.py                  Streamlit dashboard and model pipeline
data/Export_Product_Return_Data.csv  Input dataset
models/                               Saved model bundle and metadata
GOALS.md                              Product goals and judging notes
FINAL_REPORT.md                       Detailed methodology and final assessment
```

## Responsible Use

This is a decision-support tool for loss prevention. A high score is not proof of fraud, abuse, or customer intent. Do not use the score as the sole basis for denial. Use proportionate verification, retain an appeal path, monitor subgroup performance, and record human review outcomes.

## Recommended Next Steps

- Collect real timestamped order and outcome data.
- Define and audit the return/RTO label.
- Add payment method, account age, address mismatch, order velocity, and pincode-level historical signals where lawfully available.
- Calibrate probabilities and compare against a stronger model using the same untouched time-based test period.
- Add drift, subgroup fairness, and post-deployment precision/recall monitoring.
- Replace planning assumptions with measured merchant costs.

## License

The local bundled Python environment is excluded from the repository. If it is distributed separately, see `PreShipAIpython/LICENSE_PYTHON.txt` for its Python distribution license information.
