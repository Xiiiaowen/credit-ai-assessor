"""
Vectorised batch scoring: encode an entire DataFrame and score all rows at once.
Returns default probability and risk label for each applicant.
"""

import numpy as np
import pandas as pd
from model.predict import _load

REQUIRED_COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker",
]

_TEMPLATE_ROWS = [
    {
        "checking_status": ">=200", "duration": 24, "credit_history": "existing paid",
        "purpose": "new car", "credit_amount": 4000, "savings_status": "500<=X<1000",
        "employment": "4<=X<7", "installment_commitment": 2, "personal_status": "male single",
        "other_parties": "none", "residence_since": 3, "property_magnitude": "real estate",
        "age": 35, "other_payment_plans": "none", "housing": "own",
        "existing_credits": 1, "job": "skilled", "num_dependents": 1,
        "own_telephone": "yes", "foreign_worker": "no",
    },
    {
        "checking_status": "<0", "duration": 48, "credit_history": "delayed previously",
        "purpose": "furniture/equipment", "credit_amount": 8500, "savings_status": "<100",
        "employment": "<1", "installment_commitment": 4, "personal_status": "female div/dep/mar",
        "other_parties": "none", "residence_since": 1, "property_magnitude": "no known property",
        "age": 25, "other_payment_plans": "stores", "housing": "rent",
        "existing_credits": 2, "job": "unskilled resident", "num_dependents": 2,
        "own_telephone": "none", "foreign_worker": "yes",
    },
    {
        "checking_status": "no checking", "duration": 12, "credit_history": "all paid",
        "purpose": "radio/tv", "credit_amount": 1500, "savings_status": ">=1000",
        "employment": ">=7", "installment_commitment": 1, "personal_status": "male mar/wid",
        "other_parties": "guarantor", "residence_since": 4, "property_magnitude": "life insurance",
        "age": 52, "other_payment_plans": "none", "housing": "own",
        "existing_credits": 1, "job": "high qualif/self emp/mgmt", "num_dependents": 1,
        "own_telephone": "yes", "foreign_worker": "no",
    },
    {
        "checking_status": "0<=X<200", "duration": 36, "credit_history": "no credits/all paid",
        "purpose": "education", "credit_amount": 6000, "savings_status": "100<=X<500",
        "employment": "1<=X<4", "installment_commitment": 3, "personal_status": "male div/sep",
        "other_parties": "co applicant", "residence_since": 2, "property_magnitude": "car",
        "age": 29, "other_payment_plans": "bank", "housing": "for free",
        "existing_credits": 1, "job": "skilled", "num_dependents": 1,
        "own_telephone": "none", "foreign_worker": "no",
    },
    {
        "checking_status": "<0", "duration": 60, "credit_history": "critical/other existing credit",
        "purpose": "business", "credit_amount": 15000, "savings_status": "no known savings",
        "employment": "unemployed", "installment_commitment": 4, "personal_status": "female div/dep/mar",
        "other_parties": "none", "residence_since": 1, "property_magnitude": "no known property",
        "age": 45, "other_payment_plans": "none", "housing": "rent",
        "existing_credits": 3, "job": "unskilled resident", "num_dependents": 2,
        "own_telephone": "none", "foreign_worker": "yes",
    },
]


def make_template_csv() -> str:
    """Return CSV string with 5 example rows covering a range of risk profiles."""
    return pd.DataFrame(_TEMPLATE_ROWS)[REQUIRED_COLUMNS].to_csv(index=False)


def score_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score all rows in df. Returns df with two new leading columns:
      - default_probability (float, 0–1)
      - risk_label (str: Low / Medium / High)
    Raises ValueError if required columns are missing or values cannot be encoded.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    art = _load()
    model    = art["model"]
    encoders = art["encoders"]
    feature_names = art["feature_names"]

    X = {}
    for feat in feature_names:
        col = df[feat]
        if feat in encoders:
            X[feat] = encoders[feat].transform(col.astype(str).values)
        else:
            X[feat] = col.astype(float).values

    X_df  = pd.DataFrame(X, columns=feature_names)
    probs = model.predict_proba(X_df)[:, 1]
    labels = ["Low" if p < 0.35 else ("Medium" if p < 0.60 else "High") for p in probs]

    result = df.copy().reset_index(drop=True)
    result.insert(0, "risk_label", labels)
    result.insert(0, "default_probability", probs)
    return result
