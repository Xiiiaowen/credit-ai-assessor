"""
AI Credit Assessor — Streamlit app.
XGBoost prediction + SHAP explainability + LLM plain-English memo + What-If Analysis.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from model.predict import predict, get_global_importance
from model.explain import explain_stream

st.set_page_config(page_title="AI Credit Assessor", layout="wide")

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem !important; }
  .element-container:has(svg) { margin-bottom: 0 !important; }
</style>
""", unsafe_allow_html=True)

_svg = open("logo.svg", encoding="utf-8").read()
_svg_sidebar = _svg.replace('width="200"', 'width="110"').replace('height="120"', 'height="70"')
_svg_header  = _svg.replace('width="200"', 'width="72"').replace('height="120"', 'height="56"')

# ── What-If options map ────────────────────────────────────────────────────────
_WHATIF_OPTIONS = {
    "checking_status":       ["<0", "0<=X<200", ">=200", "no checking"],
    "duration":              (6, 72, 1),
    "credit_history":        ["no credits/all paid", "all paid", "existing paid",
                              "delayed previously", "critical/other existing credit"],
    "credit_amount":         (250, 20000, 100),
    "savings_status":        ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"],
    "employment":            ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
    "installment_commitment":(1, 4, 1),
    "age":                   (19, 75, 1),
    "residence_since":       (1, 4, 1),
    "existing_credits":      [1, 2, 3, 4],
    "num_dependents":        [1, 2],
    "purpose":               ["new car", "used car", "furniture/equipment", "radio/tv",
                              "domestic appliance", "repairs", "education", "business",
                              "retraining", "other"],
    "property_magnitude":    ["real estate", "life insurance", "car", "no known property"],
    "other_parties":         ["none", "co applicant", "guarantor"],
    "other_payment_plans":   ["none", "bank", "stores"],
    "housing":               ["own", "for free", "rent"],
    "job":                   ["unskilled resident", "unemp/unskilled non res",
                              "skilled", "high qualif/self emp/mgmt"],
    "personal_status":       ["male single", "female div/dep/mar", "male div/sep", "male mar/wid"],
    "own_telephone":         ["yes", "none"],
    "foreign_worker":        ["yes", "no"],
}

# ── API key ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(_svg_sidebar, unsafe_allow_html=True)
    st.title("⚙️ Settings")
    api_key_input = st.text_input("OpenAI API Key", type="password",
                                  placeholder="sk-… (leave blank to use env key)")
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input

    st.divider()
    st.caption("**Model:** XGBoost trained on German Credit dataset (UCI / OpenML)")
    st.caption("**Explainability:** SHAP TreeExplainer")
    st.caption("**Memo:** GPT-4o-mini")

# ── Page header ───────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown(_svg_header, unsafe_allow_html=True)
with col_title:
    st.title("AI Credit Assessor")
    st.caption("Enter applicant details → get default probability, SHAP breakdown, and an AI-generated assessment memo.")

st.divider()

tab_assess, tab_model = st.tabs(["Applicant Assessment", "Model Overview"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Applicant Assessment
# ════════════════════════════════════════════════════════════════════════════
with tab_assess:

    with st.form("applicant_form"):
        st.subheader("Applicant Profile")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Financial**")
            checking_status = st.selectbox(
                "Checking account",
                ["<0", "0<=X<200", ">=200", "no checking"],
                help="Current balance in checking account (DM)"
            )
            savings_status = st.selectbox(
                "Savings account",
                ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"],
            )
            credit_amount = st.number_input("Loan amount (DM)", min_value=250, max_value=20000,
                                            value=3000, step=100)
            duration = st.slider("Loan duration (months)", 6, 72, 24)
            installment_commitment = st.slider("Installment rate (% of income)", 1, 4, 3)
            credit_history = st.selectbox(
                "Credit history",
                ["no credits/all paid", "all paid", "existing paid",
                 "delayed previously", "critical/other existing credit"],
            )

        with col2:
            st.markdown("**Personal**")
            age = st.number_input("Age", min_value=19, max_value=75, value=35)
            personal_status = st.selectbox(
                "Personal status",
                ["male single", "female div/dep/mar", "male div/sep", "male mar/wid"],
            )
            num_dependents = st.selectbox("Dependents", [1, 2])
            housing = st.selectbox("Housing", ["own", "for free", "rent"])
            own_telephone = st.selectbox("Telephone", ["yes", "none"])
            foreign_worker = st.selectbox("Foreign worker", ["yes", "no"])

        with col3:
            st.markdown("**Employment & Assets**")
            employment = st.selectbox(
                "Employment duration",
                ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
            )
            job = st.selectbox(
                "Job type",
                ["unskilled resident", "unemp/unskilled non res", "skilled",
                 "high qualif/self emp/mgmt"],
            )
            purpose = st.selectbox(
                "Loan purpose",
                ["new car", "used car", "furniture/equipment", "radio/tv",
                 "domestic appliance", "repairs", "education", "business",
                 "retraining", "other"],
            )
            property_magnitude = st.selectbox(
                "Collateral / property",
                ["real estate", "life insurance", "car", "no known property"],
            )
            other_parties = st.selectbox("Other debtors", ["none", "co applicant", "guarantor"])
            other_payment_plans = st.selectbox("Other payment plans", ["none", "bank", "stores"])
            residence_since = st.slider("Years at current address", 1, 4, 2)
            existing_credits = st.selectbox("Existing credits at this bank", [1, 2, 3, 4])

        submitted = st.form_submit_button("Assess Credit Risk", type="primary", use_container_width=True)

    # ── On form submit: run model and store in session_state ──────────────────
    if submitted:
        applicant = {
            "checking_status": checking_status,
            "duration": duration,
            "credit_history": credit_history,
            "purpose": purpose,
            "credit_amount": credit_amount,
            "savings_status": savings_status,
            "employment": employment,
            "installment_commitment": installment_commitment,
            "personal_status": personal_status,
            "other_parties": other_parties,
            "residence_since": residence_since,
            "property_magnitude": property_magnitude,
            "age": age,
            "other_payment_plans": other_payment_plans,
            "housing": housing,
            "existing_credits": existing_credits,
            "job": job,
            "num_dependents": num_dependents,
            "own_telephone": own_telephone,
            "foreign_worker": foreign_worker,
        }
        with st.spinner("Running model…"):
            result = predict(applicant)
        st.session_state["result"] = result
        st.session_state["applicant"] = applicant
        for key in list(st.session_state.keys()):
            if key.startswith("wi_"):
                del st.session_state[key]

    # ── Results ───────────────────────────────────────────────────────────────
    if "result" in st.session_state:
        result    = st.session_state["result"]
        applicant = st.session_state["applicant"]

        prob          = result["probability"]
        label         = result["risk_label"]
        shap_vals     = result["shap_values"]
        feature_names = result["feature_names"]

        st.divider()

        col_score, col_meter = st.columns([1, 2])
        with col_score:
            colour = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[label]
            st.metric("Default Probability", f"{prob:.1%}")
            st.markdown(f"### {colour} {label} Risk")
            st.caption("Low < 35% · Medium 35–60% · High > 60%")

        with col_meter:
            fig_bar, ax = plt.subplots(figsize=(5, 0.6))
            bar_color = "#2ecc71" if label == "Low" else ("#f39c12" if label == "Medium" else "#e74c3c")
            ax.barh([0], [prob], color=bar_color, height=0.5)
            ax.barh([0], [1 - prob], left=[prob], color="#ecf0f1", height=0.5)
            ax.set_xlim(0, 1)
            ax.axis("off")
            ax.set_facecolor("none")
            fig_bar.patch.set_alpha(0)
            st.pyplot(fig_bar, use_container_width=True)
            plt.close(fig_bar)

        st.divider()

        col_shap, col_memo = st.columns(2)

        with col_shap:
            st.subheader("Factor Breakdown (SHAP)")
            st.caption("Bars show how each factor pushed the risk score up (red) or down (blue) from the baseline.")

            indices   = np.argsort(np.abs(shap_vals))[::-1][:10]
            top_names = [feature_names[i] for i in indices]
            top_vals  = [shap_vals[i] for i in indices]

            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#e74c3c" if v > 0 else "#3498db" for v in top_vals]
            y_pos = range(len(top_names))
            ax.barh(list(y_pos), top_vals, color=colors)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(top_names, fontsize=9)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP value (impact on default probability)")
            ax.set_facecolor("#0e1117")
            fig.patch.set_facecolor("#0e1117")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.spines["bottom"].set_color("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_memo:
            st.subheader("Credit Assessment Memo")
            if not os.environ.get("OPENAI_API_KEY"):
                st.warning("Add your OpenAI API key in the sidebar to generate the memo.")
            else:
                if submitted:
                    memo_placeholder = st.empty()
                    memo_text = ""
                    for token in explain_stream(applicant, prob, label, shap_vals, feature_names):
                        memo_text += token
                        memo_placeholder.markdown(memo_text)
                    st.session_state["memo"] = memo_text
                elif "memo" in st.session_state:
                    st.markdown(st.session_state["memo"])

        # ── What-If Analysis ──────────────────────────────────────────────────
        st.divider()
        st.subheader("What-If Analysis")
        st.caption(
            "Adjust the top 3 risk drivers to see how the default probability changes. "
            "All other factors stay fixed at the original values."
        )

        top3_idx      = np.argsort(np.abs(shap_vals))[::-1][:3]
        top3_features = [feature_names[i] for i in top3_idx]

        whatif_applicant = dict(applicant)

        wi_cols = st.columns(3)
        for i, feat in enumerate(top3_features):
            with wi_cols[i]:
                opts        = _WHATIF_OPTIONS[feat]
                label_text  = feat.replace("_", " ").title()
                current_val = applicant[feat]

                if isinstance(opts, list):
                    idx = opts.index(current_val) if current_val in opts else 0
                    whatif_applicant[feat] = st.selectbox(
                        label_text, opts, index=idx, key=f"wi_{feat}"
                    )
                else:
                    mn, mx, step = opts
                    whatif_applicant[feat] = st.slider(
                        label_text, mn, mx, int(current_val), step, key=f"wi_{feat}"
                    )

        wi_result = predict(whatif_applicant)
        wi_prob   = wi_result["probability"]
        wi_label  = wi_result["risk_label"]
        delta     = wi_prob - prob

        wi_colour = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[wi_label]

        res_cols = st.columns(3)
        with res_cols[0]:
            orig_colour = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[label]
            st.metric("Original Probability", f"{prob:.1%}")
            st.markdown(f"**{orig_colour} {label} Risk**")
        with res_cols[1]:
            st.metric("New Probability", f"{wi_prob:.1%}", delta=f"{delta:+.1%}", delta_color="inverse")
            st.markdown(f"**{wi_colour} {wi_label} Risk**")
        with res_cols[2]:
            if delta < -0.05:
                st.success("Risk reduced — these changes improve the applicant's profile.")
            elif delta > 0.05:
                st.error("Risk increased — these changes worsen the applicant's profile.")
            else:
                st.info("Minimal change — the model is not very sensitive to these adjustments.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Overview
# ════════════════════════════════════════════════════════════════════════════
with tab_model:

    gi = get_global_importance()
    metrics = gi["metrics"]

    # ── Model metrics ─────────────────────────────────────────────────────────
    st.subheader("Model Performance")
    st.caption("Evaluated on a held-out 20% test set. Trained on German Credit dataset (1,000 applicants, 20 features).")

    m1, m2, m3 = st.columns(3)
    m1.metric("ROC-AUC", f"{metrics['auc']:.3f}", help="Probability the model ranks a defaulter above a non-defaulter. Random = 0.5.")
    m2.metric("Gini Coefficient", f"{metrics['gini']:.3f}", help="= 2×AUC − 1. Industry standard for credit scorecards. >0.60 is considered strong.")
    m3.metric("KS Statistic", f"{metrics['ks']:.3f}", help="Maximum separation between good and bad borrower distributions. >0.40 is considered good.")

    st.divider()

    # ── Global feature importance ─────────────────────────────────────────────
    st.subheader("Global Feature Importance")
    st.caption(
        "Shows which features the model relies on most across **all** applicants (XGBoost gain). "
        "Compare with the SHAP chart in Tab 1, which shows impact for a **single** applicant — "
        "the two can differ because a globally important feature may not matter for every individual case."
    )

    names = gi["feature_names"][:15]   # top 15
    imps  = gi["importances"][:15]

    fig, ax = plt.subplots(figsize=(7, 5))
    y_pos = range(len(names))
    ax.barh(list(y_pos), imps, color="#3498db")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Importance (gain)")
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.spines["bottom"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()

    # ── Local vs Global explainer ─────────────────────────────────────────────
    st.subheader("Local vs Global Explainability")
    st.markdown("""
| | **Global (this tab)** | **Local — SHAP (Tab 1)** |
|---|---|---|
| **Question** | Which features drive the model overall? | Why did *this* applicant get this score? |
| **Method** | XGBoost gain across all training splits | Shapley values — exact contribution per feature per prediction |
| **Use case** | Model validation, regulatory review | Individual loan officer decision support |
| **Limitation** | Doesn't explain individual decisions | Can differ from global ranking |
    """)
