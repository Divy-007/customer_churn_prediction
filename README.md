# Customer Churn Prediction App

## 📌 Project Overview

Customer churn is a critical problem for subscription-based businesses, as retaining existing customers is significantly more cost-effective than acquiring new ones.

This project builds an **end-to-end Machine Learning system** to predict customer churn probability using customer demographics, service usage, and billing data.

The solution goes beyond just modeling and focuses on the **complete ML lifecycle**, including:

- Exploratory Data Analysis (EDA)
- Feature engineering & preprocessing
- Model training and comparison
- Class imbalance handling
- Threshold optimization
- Model explainability using SHAP
- **LLM-powered plain-English explanations and retention recommendations**
- Deployment via FastAPI (Streamlit frontend planned)

Users can input customer information and receive a **real-time churn probability prediction, a SHAP-grounded explanation of why, and a concrete retention recommendation** — all in one API call.

---

## 📊 Dataset

This project uses the **Telco Customer Churn Dataset** available on Kaggle.

Dataset Link:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

- **7043 customer records**
- **21 features**

### Feature Categories:

- Customer demographics (gender, partner, dependents, senior citizen)
- Account information (tenure, contract type, billing method)
- Services subscribed (internet, streaming, security, tech support)
- Billing information (monthly charges, total charges)
- Target variable: **Churn**

The goal is to build models that **predict customer churn probability** and enable data-driven retention strategies.

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Joblib
- **FastAPI** — model serving API
- **Pydantic** — request/response validation
- **Groq (Llama 3.1)** — LLM-based explanation & retention recommendation layer
- Streamlit — planned for frontend UI (not yet implemented)

---

## ⚙️ Machine Learning Pipeline

### 🔹 Numerical Features

- Tenure
- MonthlyCharges
- TotalCharges

Processing:
- Missing value imputation
- Standard scaling

---

### 🔹 Categorical Features

- Gender
- Contract type
- Internet service
- Payment method
- Service-related features

Processing:
- Missing value imputation
- One-hot encoding

---

All preprocessing is handled using a **Scikit-learn ColumnTransformer**, ensuring a clean and reproducible pipeline.

---

## 🤖 Models Used

### 🔹 Logistic Regression
- Baseline interpretable model
- High recall → good at identifying churn customers
- **Serves as the production model behind the API**

---

### 🔹 Random Forest
- Ensemble model
- Captures non-linear relationships
- Balanced performance across metrics

---

### 🔹 XGBoost
- Gradient boosting model
- Handles class imbalance using `scale_pos_weight`
- Strong performance with optimized learning and regularization

---

## 📈 Model Performance

| Model | ROC-AUC | F1 Score | Precision | Recall |
|------|--------|--------|--------|--------|
| Logistic Regression | 0.86 | 0.64 | 0.52 | 0.84 |
| Random Forest | 0.85 | 0.65 | 0.56 | 0.78 |
| XGBoost | 0.85 | 0.63 | 0.55 | 0.75 |

---

## ⚖️ Threshold Optimization

Instead of using a default threshold (0.5), different thresholds were evaluated to balance:

- Precision (avoiding false positives)
- Recall (capturing churn customers)

This allows businesses to **customize risk tolerance based on strategy**.

---

## 🔍 Model Explainability (SHAP + LLM)

To make the model interpretable, **SHAP (SHapley Additive Explanations)** was used:

- Explains **individual predictions**
- Identifies **global feature importance**
- Helps understand **why a customer is likely to churn**

### Key Insights from SHAP:

- 📉 Low tenure → higher churn risk
- 📉 Month-to-month contracts → strong churn driver
- 📈 Higher monthly charges → increased churn probability
- ❌ Lack of services (security, tech support) → higher churn

### 🔹 From SHAP numbers to plain-English insight

Raw SHAP values are useful for a data scientist, but not for a retention team. To close that gap, the top SHAP-attributed features for each prediction are passed to an LLM (**Groq / Llama 3.1**), which returns:

1. **A plain-English explanation** — why this specific customer is likely (or unlikely) to churn, grounded strictly in the SHAP output (not free-form LLM guessing)
2. **A retention recommendation** — a concrete, actionable next step tailored to that customer's specific churn drivers (e.g. a contract-upgrade incentive if `Contract` is the top driver)

Both are generated in a single structured LLM call and returned alongside the prediction — the model explains *what* will happen, SHAP explains *why*, and the LLM translates that into something a retention team can act on immediately.

---

## 📊 Key Insights

### 📌 Contract Type
Customers on **month-to-month contracts** show the highest churn probability.

### 📌 Customer Tenure
Customers with shorter tenure are significantly more likely to churn.

### 📌 Monthly Charges
Higher monthly charges correlate with increased churn.

### 📌 Internet Service Type
Customers with **fiber optic service** have higher churn rates.

### 📌 Value-added Services
Customers without:
- Online Security
- Tech Support
- Device Protection

are more likely to churn.

---

## 💡 Business Recommendations

Based on the model insights:

- Encourage **long-term contracts** via incentives
- Offer **bundled services** to improve retention
- Provide **targeted offers to high-paying customers**
- Focus retention strategies on **new customers with low tenure**

These general insights are now also generated **per-customer, in real time**, via the LLM-powered `solution` field in the API response — see below.

---

## 🌐 Architecture & Deployment

The project currently exposes a **FastAPI backend**. A Streamlit frontend is planned but not yet built — for now, the API is testable directly via `/docs` (FastAPI's auto-generated interactive docs) or tools like Postman/curl.

```
app/
├── api.py              # FastAPI orchestration — /predict, /health
├── schema.py            # Pydantic request/response validation
├── explain.py            # SHAP explainability logic
└── llm/
    ├── groq_provider.py    # Groq API call
    └── explain_llm.py       # Prompt building + explanation/solution parsing
```

**Why split this way:** each layer has one job — the model predicts, SHAP explains the prediction, the LLM translates that into language and action, and the API just orchestrates the three. This keeps prompt tuning, explainability logic, and model serving independently testable and swappable (e.g. adding a fallback LLM provider is a one-file change).

### `/predict` endpoint

**Request:** customer profile (validated via Pydantic enums — invalid categorical values are rejected before they ever reach the model)

**Response:**
```json
{
  "prediction": 1,
  "churn_probability": 0.62,
  "top_factors": {
    "num__tenure": 1.14,
    "cat__Contract_Month-to-month": 0.67,
    "num__TotalCharges": -0.52
  },
  "explanation": "This customer is likely to churn primarily due to their short tenure and month-to-month contract...",
  "solution": "Offer a discounted incentive to upgrade to a one-year contract, and highlight the value of tenure-based loyalty perks."
}
```

Run the API:
```bash
uvicorn app.api:app --reload --port 8000
```

Interactive docs (test the API without a frontend): `http://localhost:8000/docs`

---

## 📌 Conclusion

This project demonstrates how to build a **production-ready ML solution** that combines:

- Predictive modeling
- Business understanding
- Explainability (SHAP)
- LLM-powered, per-customer actionable recommendations
- Deployment via FastAPI (Streamlit frontend to follow)

It highlights the importance of going beyond modeling to deliver **actionable business insights** — not just a probability score, but a grounded explanation and a concrete next step.