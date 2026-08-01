"""
FastAPI backend for Customer Churn Prediction
- Serves predictions from a saved sklearn Pipeline (preprocessing + LogisticRegression)
- Computes SHAP values (LinearExplainer) for explainability
- Uses an LLM to translate SHAP output into plain-English explanations

NOTE: model.pkl must be the FULL clf_pipeline object (joblib.dump(clf_pipeline, ...)),
not just the bare classifier — get_top_factors() relies on model.named_steps to pull
out the preprocessor and classifier separately.

Schemas live in schemas.py, LLM logic in llm.py, SHAP logic in explain.py —
keep this file focused on request/response orchestration only.
"""

from dotenv import load_dotenv
load_dotenv()
import os


import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schema import CustomerInput, ChurnPrediction
from app.shap import get_top_factors
from llm.prompt import get_llm_explanation


DATA_PATH = os.path.join("data", "Telco_Customer_Churn.csv")
background_df = pd.read_csv(DATA_PATH).sample(100, random_state=42)
# drop target column + customerID if present, keep only the feature columns
background_df = background_df.drop(columns=["Churn", "customerID"], errors="ignore")

app = FastAPI(title="Churn Prediction API")

# ---- Load model once at startup ----
# Update this path/filename to whichever pipeline .pkl you're actually serving
MODEL_PATH = os.path.join("models", "logistic_model.pkl")
model = joblib.load(MODEL_PATH)


def to_dataframe(data: CustomerInput) -> pd.DataFrame:
    """
    Pipeline handles imputation/scaling/one-hot encoding internally —
    just pass raw request values through as-is.
    """
    payload = data.model_dump(mode="json") if hasattr(data, "model_dump") else data.dict()
    return pd.DataFrame([payload])


@app.post("/predict", response_model=ChurnPrediction)
def predict(data: CustomerInput):
    try:
        input_df = to_dataframe(data)
        pred = int(model.predict(input_df)[0])
        prob = float(model.predict_proba(input_df)[0][1])
        top_factors = get_top_factors(model, input_df,background_df)
        explanation,solution = get_llm_explanation(pred, prob, top_factors)
        

        return ChurnPrediction(
            prediction=pred,
            churn_probability=prob,
            top_factors=top_factors,
            explanation=explanation,
            solution=solution
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


# Run with: uvicorn api:app --reload --port 8000