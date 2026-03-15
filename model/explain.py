"""
Generate a plain-English credit decision explanation using GPT-4o-mini.
Input: SHAP values + applicant data. Output: streamed explanation text.
"""

import os
import numpy as np
from openai import OpenAI

# Human-readable labels for German Credit dataset features
FEATURE_LABELS = {
    "checking_status": "Checking account status",
    "duration": "Loan duration (months)",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "credit_amount": "Loan amount (DM)",
    "savings_status": "Savings account balance",
    "employment": "Employment duration",
    "installment_commitment": "Installment rate (% of income)",
    "personal_status": "Personal status",
    "other_parties": "Other debtors / guarantors",
    "residence_since": "Years at current residence",
    "property_magnitude": "Property / collateral",
    "age": "Age",
    "other_payment_plans": "Other payment plans",
    "housing": "Housing situation",
    "existing_credits": "Number of existing credits",
    "job": "Employment type",
    "num_dependents": "Number of dependents",
    "own_telephone": "Has telephone",
    "foreign_worker": "Foreign worker",
}


def _get_openai() -> OpenAI:
    # Always read the current key so sidebar updates take effect immediately
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def explain_stream(
    applicant: dict,
    probability: float,
    risk_label: str,
    shap_values: "np.ndarray",
    feature_names: list[str],
):
    """
    Stream a plain-English explanation of the credit decision.
    Yields text tokens as they arrive from the LLM.
    """
    # Top 5 factors by absolute SHAP value
    indices = np.argsort(np.abs(shap_values))[::-1][:5]
    factors = []
    for i in indices:
        name = feature_names[i]
        label = FEATURE_LABELS.get(name, name)
        val = applicant.get(name, "")
        shap_v = shap_values[i]
        direction = "increases" if shap_v > 0 else "decreases"
        factors.append(f"- {label}: '{val}' → {direction} default risk (SHAP: {shap_v:+.3f})")

    factors_text = "\n".join(factors)

    system_prompt = (
        "You are a senior credit risk officer writing an internal credit assessment memo. "
        "Your audience is a junior loan officer who needs to understand the decision. "
        "Be professional, precise, and concise. Use plain English — avoid jargon. "
        "Do not use bullet points. Write 3-4 sentences maximum."
    )

    user_prompt = (
        f"An applicant has been assessed with a {risk_label} default risk "
        f"(probability of default: {probability:.1%}).\n\n"
        f"The top factors driving this assessment are:\n{factors_text}\n\n"
        f"Write a brief credit assessment memo explaining this outcome to the loan officer."
    )

    stream = _get_openai().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=200,
        stream=True,
    )

    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            yield delta
