"""
SHAP explainability layer — extracts top contributing features for a
prediction. Kept separate so the SHAP/explainer logic can be swapped
(e.g. LinearExplainer -> TreeExplainer if you switch to the RF/XGB model)
without touching api.py.
"""

import pandas as pd
import shap


def get_top_factors(model,background_df, input_df, n: int = 3) -> dict:
    """
    SHAP needs the *transformed* features (post preprocessor), not raw input,
    since that's what the classifier actually sees.

    `model` must be the full sklearn Pipeline (with 'preprocessor' and
    'classifier' named steps) — not the bare classifier.
    """
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed = preprocessor.transform(input_df)
    feature_names = preprocessor.get_feature_names_out()

    # LogisticRegression -> LinearExplainer (NOT TreeExplainer, that's for RF/XGB only)
    background = preprocessor.transform(background_df)

    explainer = shap.LinearExplainer(classifier, background)
    shap_values = explainer.shap_values(transformed)
    values = shap_values[0] if len(shap_values.shape) > 1 else shap_values

    pairs = sorted(zip(feature_names, values), key=lambda x: abs(x[1]), reverse=True)[:n]
    return {f: round(float(v), 3) for f, v in pairs}