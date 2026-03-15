"""
AI Credit Assessor — Streamlit app.
XGBoost prediction + SHAP explainability + LLM plain-English memo.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from model.predict import predict
from model.explain import explain_stream

st.set_page_config(page_title="AI Credit Assessor", layout="wide")

st.markdown("""
<style>
  /* Remove top padding from main content area */
  .block-container { padding-top: 1.5rem !important; }
  /* Remove margin below SVG in sidebar and header */
  .element-container:has(svg) { margin-bottom: 0 !important; }
</style>
""", unsafe_allow_html=True)

_svg = open("logo.svg", encoding="utf-8").read()
_svg_sidebar = _svg.replace('width="200"', 'width="110"').replace('height="120"', 'height="70"')
_svg_header  = _svg.replace('width="200"', 'width="72"').replace('height="120"', 'height="56"')

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

# ── Applicant input form ──────────────────────────────────────────────────────
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

# ── Prediction & output ───────────────────────────────────────────────────────
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

    prob = result["probability"]
    label = result["risk_label"]
    shap_vals = result["shap_values"]
    feature_names = result["feature_names"]

    st.divider()

    # ── Risk score display ────────────────────────────────────────────────────
    col_score, col_meter = st.columns([1, 2])

    with col_score:
        colour = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[label]
        st.metric("Default Probability", f"{prob:.1%}")
        st.markdown(f"### {colour} {label} Risk")
        st.caption(f"Low < 35% · Medium 35-60% · High > 60%")

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

    # ── SHAP waterfall chart ──────────────────────────────────────────────────
    col_shap, col_memo = st.columns(2)

    with col_shap:
        st.subheader("Factor Breakdown (SHAP)")
        st.caption("Bars show how each factor pushed the risk score up (red) or down (blue) from the baseline.")

        # Top 10 features by absolute SHAP value
        indices = np.argsort(np.abs(shap_vals))[::-1][:10]
        top_names = [feature_names[i] for i in indices]
        top_vals = [shap_vals[i] for i in indices]

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

    # ── LLM memo ─────────────────────────────────────────────────────────────
    with col_memo:
        st.subheader("Credit Assessment Memo")
        if not os.environ.get("OPENAI_API_KEY"):
            st.warning("Add your OpenAI API key in the sidebar to generate the memo.")
        else:
            memo_placeholder = st.empty()
            memo_text = ""
            for token in explain_stream(applicant, prob, label, shap_vals, feature_names):
                memo_text += token
                memo_placeholder.markdown(memo_text)
