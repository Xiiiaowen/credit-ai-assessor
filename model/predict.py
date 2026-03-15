"""
Load trained model, encode applicant input, return prediction + SHAP values.
"""

import joblib
import numpy as np
import pandas as pd
import shap

_artifact = None


def _load():
    global _artifact
    if _artifact is None:
        _artifact = joblib.load("artifacts/model.pkl")
    return _artifact


def predict(applicant: dict) -> dict:
    """
    Given a dict of raw feature values (as shown in the UI), return:
      {
        "probability": float,          # P(default)
        "risk_label": str,             # Low / Medium / High
        "shap_values": np.ndarray,     # per-feature SHAP values
        "shap_base": float,            # expected model output
        "feature_names": list[str],
        "encoded_input": pd.DataFrame, # single-row encoded DataFrame
      }
    """
    art = _load()
    model = art["model"]
    explainer = art["explainer"]
    encoders = art["encoders"]
    feature_names = art["feature_names"]

    # Encode categoricals
    row = {}
    for feat in feature_names:
        val = applicant[feat]
        if feat in encoders:
            row[feat] = encoders[feat].transform([str(val)])[0]
        else:
            row[feat] = float(val)

    X = pd.DataFrame([row], columns=feature_names)

    prob = float(model.predict_proba(X)[0, 1])

    if prob < 0.35:
        label = "Low"
    elif prob < 0.60:
        label = "Medium"
    else:
        label = "High"

    sv = explainer(X)
    shap_vals = sv.values[0]       # shape: (n_features,)
    shap_base = float(sv.base_values[0])

    return {
        "probability": prob,
        "risk_label": label,
        "shap_values": shap_vals,
        "shap_base": shap_base,
        "feature_names": feature_names,
        "encoded_input": X,
    }


def predict_prob(applicant: dict) -> float:
    """Fast probability-only prediction (no SHAP). Used for counterfactual search."""
    art = _load()
    model = art["model"]
    encoders = art["encoders"]
    feature_names = art["feature_names"]

    row = {}
    for feat in feature_names:
        val = applicant[feat]
        if feat in encoders:
            row[feat] = encoders[feat].transform([str(val)])[0]
        else:
            row[feat] = float(val)

    X = pd.DataFrame([row], columns=feature_names)
    return float(model.predict_proba(X)[0, 1])


def get_global_importance() -> dict:
    """
    Return global feature importance (XGBoost gain) and saved model metrics.
    """
    art = _load()
    model = art["model"]
    feature_names = art["feature_names"]
    importances = model.feature_importances_   # gain-based, sums to 1
    order = np.argsort(importances)[::-1]
    return {
        "feature_names": [feature_names[i] for i in order],
        "importances": [float(importances[i]) for i in order],
        "metrics": art["metrics"],
    }
