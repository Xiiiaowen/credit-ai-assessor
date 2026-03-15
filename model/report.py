"""
Generate a PDF credit assessment report using fpdf2.
Returns bytes that can be passed to st.download_button().
"""

import io
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF

from model.explain import FEATURE_LABELS

# Characters the LLM commonly outputs that Latin-1 (Helvetica) can't encode
_UNICODE_MAP = str.maketrans({
    "\u2014": "--",   # em dash
    "\u2013": "-",    # en dash
    "\u2018": "'",    # left single quote
    "\u2019": "'",    # right single quote
    "\u201c": '"',    # left double quote
    "\u201d": '"',    # right double quote
    "\u2026": "...",  # ellipsis
    "\u00b7": "-",    # middle dot
    "\u2022": "-",    # bullet
})

def _safe(text: str) -> str:
    """Replace common Unicode chars with ASCII equivalents, drop anything else outside Latin-1."""
    text = text.translate(_UNICODE_MAP)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _shap_chart_bytes(shap_values, feature_names) -> bytes:
    """Render SHAP bar chart and return PNG bytes."""
    indices   = np.argsort(np.abs(shap_values))[::-1][:10]
    top_names = [feature_names[i] for i in indices]
    top_vals  = [shap_values[i] for i in indices]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in top_vals]
    y_pos = range(len(top_names))
    ax.barh(list(y_pos), top_vals, color=colors)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on default probability)")
    ax.set_title("Factor Breakdown", fontsize=11)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_pdf(
    applicant: dict,
    probability: float,
    risk_label: str,
    shap_values,
    feature_names: list[str],
    memo: str,
) -> bytes:
    """Return PDF bytes for the full credit assessment report."""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Header ────────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "AI Credit Assessor", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Credit Assessment Report  ·  {datetime.date.today().strftime('%d %B %Y')}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # ── Risk verdict ──────────────────────────────────────────────────────────
    colour_map = {"Low": (46, 204, 113), "Medium": (243, 156, 18), "High": (231, 76, 60)}
    r, g, b = colour_map[risk_label]

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Risk Assessment", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(60, 9, f"  {risk_label} Risk  |  {probability:.1%} default probability", fill=True, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # ── Applicant profile ─────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Applicant Profile", ln=True)
    pdf.set_font("Helvetica", "", 10)

    display_fields = [
        ("checking_status", "Checking account"),
        ("savings_status",  "Savings account"),
        ("credit_amount",   "Loan amount (DM)"),
        ("duration",        "Loan duration (months)"),
        ("credit_history",  "Credit history"),
        ("employment",      "Employment duration"),
        ("age",             "Age"),
        ("housing",         "Housing"),
        ("job",             "Job type"),
        ("purpose",         "Loan purpose"),
    ]

    fill = False
    for key, label in display_fields:
        val = applicant.get(key, "-")
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(80, 7, _safe(f"  {label}"), border=0, fill=fill)
        pdf.cell(0,  7, _safe(f"  {val}"), border=0, fill=fill, ln=True)
        fill = not fill
    pdf.ln(4)

    # ── SHAP chart ────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Factor Breakdown (SHAP)", ln=True)

    chart_bytes = _shap_chart_bytes(shap_values, feature_names)
    chart_buf   = io.BytesIO(chart_bytes)
    pdf.image(chart_buf, x=10, w=130)
    pdf.ln(2)

    # ── Top factors text ──────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Key Risk Drivers", ln=True)
    pdf.set_font("Helvetica", "", 10)

    top5 = np.argsort(np.abs(shap_values))[::-1][:5]
    for i in top5:
        name      = feature_names[i]
        label_str = FEATURE_LABELS.get(name, name)
        val       = applicant.get(name, "")
        direction = "increases" if shap_values[i] > 0 else "decreases"
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(
            pdf.epw, 6,
            _safe(f"  - {label_str} ('{val}'): {direction} default risk  [SHAP: {shap_values[i]:+.3f}]")
        )
    pdf.ln(3)

    # ── LLM memo ─────────────────────────────────────────────────────────────
    if memo:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Credit Assessment Memo", ln=True)
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_fill_color(248, 248, 248)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(pdf.epw, 6, _safe(memo), fill=True)
        pdf.ln(3)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        pdf.epw, 5,
        "For learning and portfolio demonstration only. "
        "Not intended for actual credit decisions. "
        "Model trained on the German Credit dataset (UCI / OpenML)."
    )

    return bytes(pdf.output())
