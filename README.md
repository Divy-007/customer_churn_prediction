# Customer Churn Prediction App

🔗 **Live Demo:** https://customer-churn-prediction-streamlit.onrender.com
*(Free-tier hosting — may take ~30-60s to wake up if idle)*

## 📌 Project Overview

Customer churn is a critical problem for subscription-based businesses, as retaining existing customers is significantly more cost-effective than acquiring new ones.

This project builds an **end-to-end Machine Learning system** to predict customer churn probability using customer demographics, service usage, and billing data — and goes a step further by explaining *why* a customer is at risk and *what to do about it*, using SHAP and an LLM.

The solution covers the **complete ML lifecycle**, including:

- Exploratory Data Analysis (EDA)
- Feature engineering & preprocessing
- Model training and comparison
- Class imbalance handling
- Threshold optimization
- Model explainability using SHAP
- **LLM-powered plain-English explanations and retention recommendations**
- Deployment via FastAPI (backend) + Streamlit (frontend)
- **Containerized with Docker Compose, deployed on AWS EC2**

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

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- SHAP
- FastAPI
- Pydantic
- Streamlit
- Groq (Llama 3.1) — LLM explanation & retention recommendation layer
- Docker & Docker Compose
  

---

## ⚙️ Machine Learning Pipeline

### 🔹 Numerical Features
- Tenure, MonthlyCharges, TotalCharges

Processing: missing value imputation, standard scaling

### 🔹 Categorical Features
- Gender, Contract type, Internet service, Payment method, service-related features

Processing: missing value imputation, one-hot encoding

All preprocessing is handled using a **Scikit-learn ColumnTransformer**, saved as part of the full model pipeline — so the API doesn't need to re-implement any encoding logic manually.

---

## 🤖 Models Used

| Model | ROC-AUC | F1 Score | Precision | Recall |
|------|--------|--------|--------|--------|
| **Logistic Regression** (production model) | 0.86 | 0.64 | 0.52 | 0.84 |
| Random Forest | 0.85 | 0.65 | 0.56 | 0.78 |
| XGBoost | 0.85 | 0.63 | 0.55 | 0.75 |

Logistic Regression was chosen for the served API — highest recall (best at catching actual churners) and, being a linear model, pairs cleanly with `shap.LinearExplainer` for fast, exact explainability.

---

## 🔍 Explainability: SHAP → LLM

Raw model output (a probability) doesn't tell a retention team *what to do*. This project closes that gap in two steps:

**1. SHAP (SHapley Additive exPlanations)**
For each prediction, SHAP values are computed against the trained classifier — using the model's own `ColumnTransformer` to get post-preprocessing feature names, and a **background sample from real training data** (not the input row itself) as the comparison baseline. This surfaces the top 3 features actually driving that specific customer's prediction.

**2. LLM explanation + recommendation (Groq / Llama 3.1)**
The SHAP output is passed — as numbers, not raw model internals — to an LLM, which returns two things in a single structured JSON response:
- `explanation`: a plain-English reason for the prediction, grounded strictly in the SHAP values
- `solution`: a concrete, actionable retention step tailored to that customer's specific churn drivers

This keeps the LLM from guessing — it reasons only over the SHAP numbers it's given, not the model's internals.

### Key Insights from SHAP
- 📉 Low tenure → higher churn risk
- 📉 Month-to-month contracts → strongest churn driver
- 📈 Higher monthly charges → increased churn probability
- ❌ Lack of add-on services (security, tech support) → higher churn

---

## 🌐 Project Structure

```
customer-churn-prediction/
├── app/
│   ├── app.py            # FastAPI backend — /predict, /health
│   ├── schema.py           # Pydantic request/response validation
│   ├── shap.py               # SHAP explainability logic
│   └── streamlit.py           # Streamlit frontend (thin client, calls the API)
├── llm/
│   ├── groq.py             # Groq API call
│   └── prompt.py             # Prompt building + JSON parsing (explanation + solution)
├── models/                # trained model pipelines (.pkl)
├── data/
│   └── Telco_Customer_Churn.csv
├── notebooks/
│   └── Customer_Churn_Prediction.ipynb
├── dockerfile.api           # backend image
├── dockerfile.streamlit      # frontend image
├── docker-compose.yml
├── requirements.txt
└── .env                    # GROQ_API_KEY — gitignored
```

---

## 📡 API

**`POST /predict`**

Request: full customer profile (validated via Pydantic enums — invalid categorical values are rejected before reaching the model).

Response:
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
  "solution": "Offer a discounted incentive to upgrade to a one-year contract, and highlight tenure-based loyalty perks."
}
```

Interactive docs (Swagger UI): `/docs`

---

## 🐳 Running Locally with Docker

```bash
docker compose up --build
```

- API → `http://localhost:8000/docs`
- Streamlit → `http://localhost:8501`

Requires a `.env` file at the repo root with:
```
GROQ_API_KEY=your_key_here
```

The project runs as two independent containers (`api`, `streamlit`) orchestrated via `docker-compose.yml` — each with its own Dockerfile, wired together on Docker's internal network so Streamlit reaches the API by service name rather than `localhost`.

---

## 🐳 Docker Hub

Pre-built images are published and don't require cloning the repo to run:

```bash
docker pull divy049/customer-churn-prediction-api:latest
docker pull divy049/customer-churn-prediction-streamlit:latest
```

---

## ☁️ Deployment

**Live demo:** https://customer-churn-prediction-streamlit.onrender.com

Deployed on **Render**, pulling directly from the pre-built Docker Hub images (no build step on the server):

- **API** service — image `divy049/customer-churn-prediction-api:latest`, port `8000`
- **Streamlit** service — image `divy049/customer-churn-prediction-streamlit:latest`, port `8501`, with `API_URL` set to the deployed API's public URL

The two services are deployed independently (matching the local Docker Compose setup) — Streamlit calls the API over its public Render URL rather than an internal Docker network, since Render services don't share a network by default.

> Note: running on Render's free tier, so the app may take ~30-60s to wake up after a period of inactivity.

---

## 💡 Business Recommendations

- Encourage long-term contracts via incentives
- Offer bundled services to improve retention
- Provide targeted offers to high-paying customers
- Focus retention strategies on new customers with low tenure

These are now also generated **per-customer, in real time**, via the `solution` field in the API response.

---

## 📌 Conclusion

This project demonstrates a **production-style ML system**: predictive modeling, SHAP-based explainability, LLM-powered actionable recommendations, and containerized deployment — going beyond a notebook model to something a retention team could actually use.
