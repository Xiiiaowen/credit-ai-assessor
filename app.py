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

import pandas as pd
from model.predict import predict, get_global_importance
from model.explain import explain_stream, FEATURE_LABELS
from model.report import generate_pdf
from model.counterfactual import find_counterfactual
from model.batch import score_batch, make_template_csv, REQUIRED_COLUMNS

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

tab_assess, tab_batch, tab_model = st.tabs(["Applicant Assessment", "Batch Assessment", "Model Overview"])

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
            if key.startswith("wi_") or key in ("cf_changes", "cf_prob"):
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

        # ── Approval Path (Counterfactual) ────────────────────────────────────
        if label != "Low":
            st.divider()
            st.subheader("Approval Path")
            st.markdown(
                "The model works backwards to find the **smallest set of changes** that would "
                "bring this applicant's default probability below the Low Risk threshold (35%). "
                "Each row is one change, applied in sequence — the *Probability After* column "
                "shows the updated risk at that step."
            )
            if "cf_changes" not in st.session_state:
                with st.spinner("Searching for approval path…"):
                    cf_changes, cf_prob = find_counterfactual(
                        applicant, shap_vals, feature_names, _WHATIF_OPTIONS
                    )
                    st.session_state["cf_changes"] = cf_changes
                    st.session_state["cf_prob"] = cf_prob
            else:
                cf_changes = st.session_state["cf_changes"]
                cf_prob    = st.session_state["cf_prob"]

            if cf_changes:
                cf_label  = "Low" if cf_prob < 0.35 else ("Medium" if cf_prob < 0.60 else "High")
                cf_colour = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[cf_label]
                orig_colour = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[label]
                st.markdown(
                    f"**Result:** {len(cf_changes)} change(s) move the applicant from "
                    f"{orig_colour} **{label} Risk ({prob:.1%})** → "
                    f"{cf_colour} **{cf_label} Risk ({cf_prob:.1%})**"
                )
                rows = [
                    {
                        "Step": i + 1,
                        "Feature": FEATURE_LABELS.get(ch["feature"], ch["feature"].replace("_", " ").title()),
                        "Current Value": str(ch["old"]),
                        "Suggested Value": str(ch["new"]),
                        "Probability After": f"{ch['prob_after']:.1%}  ({ch['prob_after'] - ch['prob_before']:+.1%})",
                    }
                    for i, ch in enumerate(cf_changes)
                ]
                st.table(pd.DataFrame(rows).set_index("Step"))
                st.caption(
                    "Changes are chosen greedily: at each step the model picks the single value "
                    "that reduces default probability the most. Immutable features (age, personal "
                    "status, nationality) are excluded."
                )
            else:
                st.info("No combination of changes found that meaningfully reduces risk for this applicant.")

        # ── PDF download ──────────────────────────────────────────────────────
        st.divider()
        memo_text = st.session_state.get("memo", "")
        pdf_bytes = generate_pdf(
            applicant, prob, label, shap_vals, feature_names, memo_text,
            cf_changes=st.session_state.get("cf_changes"),
            cf_prob=st.session_state.get("cf_prob"),
        )
        st.download_button(
            label="Download Assessment Report (PDF)",
            data=pdf_bytes,
            file_name=f"credit_assessment_{label.lower()}_risk.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Assessment
# ════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("Batch Assessment")
    st.markdown(
        "Score multiple applicants at once. Upload a CSV with one applicant per row — "
        "the model returns a default probability and risk label for each. "
        "Download the template below to see the expected column names and valid values."
    )

    # ── Template download ──────────────────────────────────────────────────
    st.download_button(
        "Download CSV Template (5 example rows)",
        data=make_template_csv(),
        file_name="applicant_template.csv",
        mime="text/csv",
    )

    st.divider()

    # ── File upload ────────────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload applicant CSV", type="csv", label_visibility="collapsed")

    if uploaded:
        try:
            raw_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        st.caption(f"{len(raw_df)} row(s) loaded.")

        with st.spinner(f"Scoring {len(raw_df)} applicants…"):
            try:
                result_df = score_batch(raw_df)
            except ValueError as e:
                st.error(str(e))
                st.markdown("**Expected columns:**")
                st.code(", ".join(REQUIRED_COLUMNS))
                st.stop()

        # ── Summary metrics ────────────────────────────────────────────────
        counts = result_df["risk_label"].value_counts()
        avg_prob = result_df["default_probability"].mean()

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Applicants", len(result_df))
        m2.metric("🟢 Low Risk",    counts.get("Low",    0))
        m3.metric("🟡 Medium Risk", counts.get("Medium", 0))
        m4.metric("🔴 High Risk",   counts.get("High",   0))
        m5.metric("Avg Default Prob", f"{avg_prob:.1%}")

        # ── Risk distribution chart ────────────────────────────────────────
        st.divider()
        risk_order  = ["Low", "Medium", "High"]
        bar_colors  = ["#2ecc71", "#f39c12", "#e74c3c"]
        bar_vals    = [counts.get(r, 0) for r in risk_order]

        fig_b, ax_b = plt.subplots(figsize=(4, 2.2))
        ax_b.bar(risk_order, bar_vals, color=bar_colors, width=0.5)
        for i, v in enumerate(bar_vals):
            if v:
                ax_b.text(i, v + 0.3, str(v), ha="center", va="bottom", color="white", fontsize=10)
        ax_b.set_ylabel("Count", color="white")
        ax_b.set_facecolor("#0e1117")
        fig_b.patch.set_facecolor("#0e1117")
        ax_b.tick_params(colors="white")
        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)
        ax_b.spines["left"].set_color("white")
        ax_b.spines["bottom"].set_color("white")
        plt.tight_layout()
        st.pyplot(fig_b, use_container_width=False)
        plt.close(fig_b)

        # ── Results table ──────────────────────────────────────────────────
        st.divider()
        filter_risk = st.radio(
            "Filter results", ["All", "High", "Medium", "Low"],
            horizontal=True, label_visibility="collapsed"
        )
        show_df = result_df if filter_risk == "All" else result_df[result_df["risk_label"] == filter_risk]

        # Display subset of columns for readability; full data in download
        display_cols = ["risk_label", "default_probability",
                        "checking_status", "credit_amount", "duration",
                        "credit_history", "savings_status", "employment", "age"]
        disp = show_df[display_cols].copy()
        disp["default_probability"] = disp["default_probability"].map(lambda x: f"{x:.1%}")
        disp = disp.rename(columns={
            "risk_label": "Risk",
            "default_probability": "Default Prob",
            "checking_status": "Checking",
            "credit_amount": "Loan (DM)",
            "duration": "Duration (mo)",
            "credit_history": "Credit History",
            "savings_status": "Savings",
            "employment": "Employment",
            "age": "Age",
        })
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.caption("Table shows key columns. Download below for the full dataset.")

        # ── Download results ───────────────────────────────────────────────
        st.divider()
        out_df = result_df.copy()
        out_df["default_probability"] = out_df["default_probability"].map(lambda x: f"{x:.1%}")
        st.download_button(
            "Download Full Results (CSV)",
            data=out_df.to_csv(index=False),
            file_name="batch_results.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Overview
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

    # ── Calibration plot ──────────────────────────────────────────────────────
    st.subheader("Probability Calibration")
    st.caption(
        "A well-calibrated model's predicted probabilities match actual default rates. "
        "Points above the diagonal mean the model underestimates risk; "
        "points below mean it overestimates. "
        "Most credit models are intentionally conservative (above the line)."
    )

    if gi.get("y_test") is not None:
        from sklearn.calibration import calibration_curve as _cal_curve
        y_test_arr   = gi["y_test"]
        y_proba_arr  = gi["y_proba_test"]
        frac_pos, mean_pred = _cal_curve(y_test_arr, y_proba_arr, n_bins=8, strategy="quantile")

        fig_cal, ax_cal = plt.subplots(figsize=(5, 4))
        ax_cal.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        ax_cal.plot(mean_pred, frac_pos, "o-", color="#3498db", linewidth=2, markersize=7, label="XGBoost")
        ax_cal.set_xlabel("Mean predicted probability")
        ax_cal.set_ylabel("Fraction of actual defaults")
        ax_cal.set_facecolor("#0e1117")
        fig_cal.patch.set_facecolor("#0e1117")
        ax_cal.tick_params(colors="white")
        ax_cal.xaxis.label.set_color("white")
        ax_cal.yaxis.label.set_color("white")
        ax_cal.spines["bottom"].set_color("white")
        ax_cal.spines["left"].set_color("white")
        ax_cal.spines["top"].set_visible(False)
        ax_cal.spines["right"].set_visible(False)
        leg = ax_cal.legend(fontsize=9)
        leg.get_frame().set_facecolor("#1e2530")
        for t in leg.get_texts():
            t.set_color("white")
        plt.tight_layout()
        st.pyplot(fig_cal, use_container_width=False)
        plt.close(fig_cal)
    else:
        st.info("Re-run `python train.py` to enable the calibration plot.")

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

    # ── Threshold optimisation ────────────────────────────────────────────────
    st.subheader("Decision Threshold Optimisation")
    st.caption(
        "The default 50% threshold is arbitrary. Banks tune this based on the cost ratio "
        "of a false negative (missed default) vs a false positive (rejected good applicant). "
        "A typical ratio is **5:1** — missing a default costs 5× more than a wrongful rejection."
    )

    if gi.get("y_test") is not None:
        from sklearn.metrics import precision_score, recall_score, f1_score
        y_test_arr  = gi["y_test"]
        y_proba_arr = gi["y_proba_test"]

        threshold = st.slider(
            "Decision threshold", 0.10, 0.90, 0.50, 0.05,
            help="Applicants above this probability are flagged as likely defaulters."
        )

        y_pred = (y_proba_arr >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_test_arr == 1)).sum())
        fp = int(((y_pred == 1) & (y_test_arr == 0)).sum())
        fn = int(((y_pred == 0) & (y_test_arr == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        cost      = 5 * fn + fp

        tc1, tc2, tc3, tc4, tc5 = st.columns(5)
        tc1.metric("Threshold", f"{threshold:.0%}")
        tc2.metric("Precision", f"{precision:.3f}", help="Of flagged defaulters, what fraction actually defaulted?")
        tc3.metric("Recall",    f"{recall:.3f}",    help="Of actual defaulters, what fraction did the model catch?")
        tc4.metric("F1 Score",  f"{f1:.3f}")
        tc5.metric("Cost (5×FN+FP)", cost,          help="Lower is better. FN (missed default) costs 5× more than FP.")

        # Cost curve across all thresholds
        ts = np.linspace(0.05, 0.95, 60)
        costs = []
        for t in ts:
            yp = (y_proba_arr >= t).astype(int)
            fn_t = int(((yp == 0) & (y_test_arr == 1)).sum())
            fp_t = int(((yp == 1) & (y_test_arr == 0)).sum())
            costs.append(5 * fn_t + fp_t)
        best_t = ts[int(np.argmin(costs))]

        fig_cost, ax_cost = plt.subplots(figsize=(7, 3))
        ax_cost.plot(ts, costs, color="#3498db", linewidth=2)
        ax_cost.axvline(threshold, color="#e74c3c", linewidth=1.5, linestyle="--", label=f"Current: {threshold:.0%}")
        ax_cost.axvline(best_t,   color="#2ecc71", linewidth=1.5, linestyle="--", label=f"Optimal: {best_t:.0%}")
        ax_cost.set_xlabel("Threshold")
        ax_cost.set_ylabel("Relative cost (5×FN + FP)")
        ax_cost.set_facecolor("#0e1117")
        fig_cost.patch.set_facecolor("#0e1117")
        ax_cost.tick_params(colors="white")
        ax_cost.xaxis.label.set_color("white")
        ax_cost.yaxis.label.set_color("white")
        ax_cost.spines["bottom"].set_color("white")
        ax_cost.spines["left"].set_color("white")
        ax_cost.spines["top"].set_visible(False)
        ax_cost.spines["right"].set_visible(False)
        leg2 = ax_cost.legend(fontsize=9)
        leg2.get_frame().set_facecolor("#1e2530")
        for t2 in leg2.get_texts():
            t2.set_color("white")
        plt.tight_layout()
        st.pyplot(fig_cost, use_container_width=True)
        plt.close(fig_cost)
        st.caption(f"Optimal threshold at 5:1 cost ratio: **{best_t:.0%}** — move the slider to explore trade-offs.")
    else:
        st.info("Re-run `python train.py` to enable threshold optimisation.")

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
