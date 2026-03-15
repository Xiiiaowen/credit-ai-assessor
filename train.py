"""
Train XGBoost credit risk model on the German Credit dataset.
Run once locally, then commit artifacts/model.pkl to git.

    pip install -r requirements.txt
    python train.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap


# ── Load data ─────────────────────────────────────────────────────────────────
print("Fetching German Credit dataset from OpenML…")
data = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
X_raw, y_raw = data.data, data.target

# Target: 1 = bad credit (default risk), 0 = good
y = (y_raw == "bad").astype(int)

# ── Encode categoricals ───────────────────────────────────────────────────────
encoders: dict[str, LabelEncoder] = {}
X = X_raw.copy()
for col in X.select_dtypes(include=["category", "object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

feature_names = list(X.columns)

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Train XGBoost ─────────────────────────────────────────────────────────────
# scale_pos_weight handles class imbalance (700 good : 300 bad)
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=neg / pos,
    eval_metric="auc",
    random_state=42,
    verbosity=0,
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
gini = 2 * auc - 1
fpr, tpr, _ = roc_curve(y_test, y_proba)
ks = float(np.max(tpr - fpr))

print(f"\n── Test set metrics ──────────────")
print(f"ROC-AUC : {auc:.4f}")
print(f"Gini    : {gini:.4f}")
print(f"KS stat : {ks:.4f}")
print(f"\n{classification_report(y_test, (y_proba >= 0.5).astype(int), target_names=['Good','Bad'])}")

# 5-fold CV AUC
cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring="roc_auc")
print(f"5-fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── SHAP explainer ────────────────────────────────────────────────────────────
print("\nBuilding SHAP TreeExplainer…")
explainer = shap.TreeExplainer(model)

# ── Save artifact ─────────────────────────────────────────────────────────────
os.makedirs("artifacts", exist_ok=True)
joblib.dump(
    {
        "model": model,
        "explainer": explainer,
        "encoders": encoders,
        "feature_names": feature_names,
        "metrics": {"auc": auc, "gini": gini, "ks": ks},
    },
    "artifacts/model.pkl",
)
print("Saved → artifacts/model.pkl")
