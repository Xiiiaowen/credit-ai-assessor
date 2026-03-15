"""
Greedy counterfactual search: find the minimum feature changes that reduce
an applicant's credit risk label (e.g. High → Low).
"""

import numpy as np
from model.predict import predict_prob

# Features the applicant cannot realistically change
_IMMUTABLE = {"age", "personal_status", "foreign_worker", "num_dependents"}


def find_counterfactual(
    applicant: dict,
    shap_vals,
    feature_names: list[str],
    options_map: dict,
    target_prob: float = 0.35,
    max_changes: int = 5,
) -> tuple[list[dict], float]:
    """
    Greedily change the highest-risk features one at a time, picking the
    option that most reduces default probability.

    Returns:
        changes   — list of {"feature", "old", "new"} dicts (in order applied)
        final_prob — predicted probability after all changes
    """
    # Risk-increasing features sorted by SHAP contribution (highest first)
    pos_idx = [
        (i, float(shap_vals[i]))
        for i in range(len(feature_names))
        if shap_vals[i] > 0 and feature_names[i] not in _IMMUTABLE
    ]
    pos_idx.sort(key=lambda x: -x[1])

    current = dict(applicant)
    changes = []

    for feat_idx, _ in pos_idx:
        feat = feature_names[feat_idx]
        opts = options_map.get(feat)
        if opts is None:
            continue

        old_val = current[feat]

        # Build candidate list
        if isinstance(opts, list):
            candidates = [v for v in opts if v != old_val]
        else:
            mn, mx, step = opts
            # Sample ~10 evenly-spaced values across the range
            n = min(10, int((mx - mn) / step) + 1)
            candidates = [
                int(mn + (mx - mn) * t / (n - 1)) for t in range(n)
            ]
            candidates = [v for v in candidates if v != old_val]

        # Find the candidate that minimises probability
        base_prob = predict_prob(current)
        best_val = None
        best_prob = base_prob

        for cand in candidates:
            trial = dict(current)
            trial[feat] = cand
            p = predict_prob(trial)
            if p < best_prob:
                best_prob = p
                best_val = cand

        if best_val is not None:
            changes.append({
                "feature": feat,
                "old": old_val,
                "new": best_val,
                "prob_before": base_prob,
                "prob_after": best_prob,
            })
            current[feat] = best_val

            if best_prob <= target_prob:
                break

        if len(changes) >= max_changes:
            break

    return changes, predict_prob(current)
