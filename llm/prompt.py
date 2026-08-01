"""
LLM explanation orchestrator — builds the prompt and calls Groq.

To add a fallback provider later (e.g. if you hit Groq's rate limit):
1. Create llm/<provider>_provider.py with a call_<provider>(prompt) function
2. Import it here and wrap the call_groq(...) call in try/except

Usage (from api.py):
    from llm.explain_llm import get_llm_explanation
    explanation = get_llm_explanation(pred, prob, top_factors)
"""

from llm.groq import call_groq
import json


def build_prompt(pred: int, prob: float, top_factors: dict) -> str:
    return f"""A customer was predicted to churn: {'Yes' if pred == 1 else 'No'} \
(churn probability: {prob:.2%}).
Top contributing factors (SHAP values, feature: impact): {top_factors}
 
Respond with ONLY a JSON object (no markdown, no preamble) with exactly these two keys:
- "explanation": 2-3 plain-English sentences explaining why this customer is likely/unlikely
  to churn, based on the factors above. Frame it for a customer-retention team, not a data scientist.
- "solution": 1-2 concrete, actionable retention steps the team could take for this specific
  customer, grounded in the contributing factors (e.g. if contract type is a driver, suggest
  a contract upgrade incentive; if payment method is a driver, suggest a payment method switch offer).
  If the customer is not likely to churn, say there's no action needed.
 
Example format:
{{"explanation": "...", "solution": "..."}}"""

def get_llm_explanation(pred: int, prob: float, top_factors: dict) -> tuple[str, str]:
    """Returns (explanation, solution)."""
    prompt = build_prompt(pred, prob, top_factors)
 
    try:
        raw = call_groq(prompt)
        parsed = json.loads(raw)
        return parsed["explanation"], parsed["solution"]
    except Exception as e:
        # Fail gracefully — don't crash /predict if Groq is down/rate-limited or returns bad JSON
        fallback = f"(LLM unavailable: {e}). Top factors: {top_factors}"
        return fallback, "N/A"