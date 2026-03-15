# AI Credit Assessor

An AI-powered credit risk assessment tool that combines a trained XGBoost model with SHAP explainability and GPT-4o-mini to produce plain-English credit memos — the kind of tool a bank's credit team would actually use.

**[Live Demo →](https://8mwpw2yoicmfimgsqla2ku.streamlit.app/)**

---

## What It Does

1. **Loan officer enters applicant details** — checking account status, loan amount, employment duration, savings, age, etc.
2. **XGBoost model predicts default probability** — trained on the German Credit dataset (1,000 applicants, 20 features)
3. **SHAP breakdown shows which factors drove the score** — red bars increase risk, blue bars decrease it
4. **GPT-4o-mini writes a credit assessment memo** — plain English, 3-4 sentences, suitable for a junior loan officer
5. **What-If Analysis** — adjust the top 3 risk drivers and instantly see how the probability changes
6. **Approval Path (counterfactual)** — model finds the minimum changes needed to move a High/Medium applicant toward Low risk
7. **Batch Assessment tab** — upload a CSV of applicants, score all at once, view risk distribution, filter by risk band, download results
8. **Model Overview tab** — global feature importance (XGBoost gain), model metrics, probability calibration plot, decision threshold optimisation with cost curve, and a local vs global explainability comparison

---

## Model Performance

Trained with 5-fold cross-validation and class-imbalance correction (`scale_pos_weight`):

| Metric | Value |
|---|---|
| ROC-AUC | **0.806** |
| Gini coefficient | **0.612** |
| KS statistic | **0.488** |
| 5-fold CV AUC | **0.792 ± 0.022** |

The Gini coefficient (= 2 × AUC − 1) and KS statistic are the standard metrics used by banks to evaluate credit scorecards under Basel III.

---

## Why These Metrics Matter (for Interviews)

- **ROC-AUC**: Probability that the model ranks a defaulter higher than a non-defaulter. 0.806 means it correctly ranks 80.6% of pairs.
- **Gini**: Industry-standard credit scorecard metric. > 0.60 is considered strong for a retail credit model.
- **KS statistic**: Maximum separation between the cumulative distribution of good and bad borrowers. > 0.40 is considered good.
- **Class imbalance**: The dataset is 70% good / 30% bad. Without correction, the model would be biased toward predicting "good". `scale_pos_weight` reweights the loss function to compensate.

---

## Tech Stack

| Layer | Tool |
|---|---|
| ML model | XGBoost (gradient boosted trees) |
| Explainability | SHAP TreeExplainer |
| LLM memo | GPT-4o-mini |
| UI | Streamlit |
| Dataset | German Credit (UCI / OpenML via scikit-learn) |

---

## Features

- **XGBoost with class-imbalance handling** — `scale_pos_weight` corrects for the 70/30 good/bad split without oversampling
- **SHAP feature attribution** — TreeExplainer computes exact Shapley values for tree models (no approximation), showing each feature's contribution to the individual prediction
- **LLM credit memo** — SHAP values are passed to GPT-4o-mini with a credit officer system prompt; the model translates statistical output into a readable business memo
- **Finance-standard evaluation** — reports ROC-AUC, Gini coefficient, and KS statistic (not just accuracy)
- **What-If sensitivity analysis** — top 3 SHAP-identified risk drivers exposed as interactive controls; re-runs the model live to show probability delta
- **Approval Path (counterfactual)** — greedy search over risk-increasing features; finds the minimum set of changes to reduce the risk label, with immutable features (age, personal status) excluded
- **Probability calibration** — reliability diagram showing predicted vs actual default rates; highlights whether the model over/underestimates risk
- **Threshold optimisation** — interactive slider with live precision, recall, F1, and relative cost (5×FN + FP); cost curve shows the optimal threshold vs current selection
- **Global vs local explainability** — Model Overview tab contrasts XGBoost gain (global) with per-applicant SHAP (local), with a comparison table
- **Batch assessment** — vectorised scoring of CSV uploads; summary metrics, risk distribution chart, filter by risk band, full CSV export
- **Bring your own API key** — paste key in sidebar; falls back to environment variable

---

## Project Structure

```
credit-ai-assessor/
├── app.py                   # Streamlit UI (assessment form, tabs, What-If, PDF download)
├── train.py                 # One-time model training script
├── model/
│   ├── predict.py           # Load model, encode input, compute SHAP values
│   ├── explain.py           # GPT-4o-mini memo generation (streaming)
│   ├── counterfactual.py    # Greedy approval path search (minimum changes to reduce risk)
│   ├── report.py            # PDF report generation (fpdf2)
│   └── batch.py             # Vectorised batch scoring for CSV uploads
├── artifacts/
│   └── model.pkl            # Trained model + SHAP explainer + encoders
├── requirements.txt
└── .env.example
```

---

## Run Locally

```bash
git clone https://github.com/Xiiiaowen/credit-ai-assessor
cd credit-ai-assessor

pip install -r requirements.txt

cp .env.example .env
# Add your OPENAI_API_KEY to .env

streamlit run app.py
```

The trained model is committed — no need to re-run `train.py`. If you want to retrain:

```bash
python train.py
```

---

## What Could Be Improved in Production

**Threshold calibration** — the 50% decision threshold is arbitrary. A bank would optimise it based on the cost ratio of a false negative (missed default) vs false positive (rejected good applicant), typically 5:1 or higher.

**Feature engineering** — debt-to-income ratio, payment-to-income ratio, and credit utilisation rate are derived features that typically improve model performance significantly.

**Model monitoring** — in production, you track data drift (input distribution shifts) and concept drift (the relationship between features and default changes over time). Tools: Evidently, WhyLogs.

**Regulatory explainability** — under GDPR Article 22 and the EU AI Act, automated credit decisions require human-interpretable explanations. SHAP provides local per-applicant attribution; the Approval Path feature provides counterfactual explanations ("what would need to change for this applicant to be approved?"), both of which are required in production credit systems.

---

## Disclaimer

For learning and portfolio demonstration only. Not intended for actual credit decisions.
