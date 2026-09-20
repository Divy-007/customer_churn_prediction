# Customer Churn Prediction App

🔗 **Live Demo:**
https://customer-churn-prediction-streamlit.onrender.com

> Free-tier hosting --- the application may take some time to wake up
> after inactivity.

## 📌 Project Overview

Customer churn is a critical problem for subscription-based businesses.
Retaining existing customers can be more cost-effective than acquiring
new ones.

This project builds an **end-to-end Machine Learning system** that
predicts customer churn probability using customer demographics, service
usage, contract information, and billing data.

It goes beyond a basic prediction model by combining:

-   Machine Learning for churn prediction
-   SHAP for customer-level explainability
-   An LLM for plain-English explanations
-   LLM-generated retention recommendations
-   FastAPI for the backend API
-   Streamlit for the user interface
-   Docker for containerization
-   GitHub Actions for automated testing and Docker image publishing
-   Docker Hub as the container registry
-   Render for cloud deployment

Users can enter a customer profile and receive a **real-time churn
probability, the top factors influencing the prediction, a
human-readable explanation, and a recommended retention action**.

------------------------------------------------------------------------

## ✨ Key Features

-   Real-time customer churn prediction
-   Churn probability score
-   Pydantic-based request validation
-   SHAP-based local explainability
-   Top contributing features for every prediction
-   LLM-powered explanation and retention recommendation
-   FastAPI REST API with Swagger documentation
-   Streamlit frontend
-   Separate Docker images for API and frontend
-   Docker Compose support for local development
-   Automated Pytest test suite
-   GitHub Actions CI pipeline
-   Docker Hub image publishing
-   Render deployment from pre-built Docker images

------------------------------------------------------------------------

## 📊 Dataset

This project uses the **Telco Customer Churn Dataset** available on
Kaggle.

**Dataset:**
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

-   **7,043 customer records**
-   **21 features**

### Feature Categories

-   Customer demographics --- gender, partner, dependents, senior
    citizen
-   Account information --- tenure, contract type, billing method
-   Services --- internet, streaming, security, backup, tech support
-   Billing --- monthly charges and total charges
-   Target variable --- `Churn`

------------------------------------------------------------------------

## 🛠️ Tech Stack

  Category               Technology
  ---------------------- ------------------------
  Programming Language   Python
  Data Processing        Pandas, NumPy
  Machine Learning       Scikit-learn
  Explainability         SHAP
  Backend                FastAPI
  Validation             Pydantic
  LLM                    Groq API
  Frontend               Streamlit
  Testing                Pytest
  Containerization       Docker, Docker Compose
  CI                     GitHub Actions
  Container Registry     Docker Hub
  Deployment             Render
  Version Control        Git, GitHub

------------------------------------------------------------------------

## ⚙️ Machine Learning Pipeline

### Numerical Features

-   `tenure`
-   `MonthlyCharges`
-   `TotalCharges`

Processing includes:

-   Missing-value handling
-   Standard scaling

### Categorical Features

Examples include:

-   Gender
-   Contract
-   Internet service
-   Payment method
-   Online security
-   Online backup
-   Device protection
-   Tech support
-   Streaming services

Processing includes:

-   Missing-value handling
-   One-hot encoding

All preprocessing is handled through a **Scikit-learn preprocessing
pipeline**, allowing the same transformations used during training to be
applied during inference.

The trained pipeline is saved and loaded by the FastAPI backend.

------------------------------------------------------------------------

## 🤖 Models Used

The project evaluates multiple classification models before selecting
the production model.

  Model                       ROC-AUC   F1 Score   Precision   Recall
  ------------------------- --------- ---------- ----------- --------
  **Logistic Regression**        0.86       0.64        0.52     0.84
  Random Forest                  0.85       0.65        0.56     0.78
  XGBoost                        0.85       0.63        0.55     0.75

**Logistic Regression** is used by the served API because it provides
strong recall and works well with `shap.LinearExplainer` for efficient
local explanations.

> Metrics above are based on the project's evaluation setup and should
> be interpreted in that context.

------------------------------------------------------------------------

## 🔍 Explainability: SHAP → LLM

A raw probability does not explain *why* a customer is considered at
risk.

This project uses a two-stage explanation pipeline.

### 1. SHAP

SHAP values are calculated for the individual customer prediction.

The system:

-   Uses the trained model pipeline
-   Obtains post-preprocessing feature names
-   Uses a background sample from customer data
-   Calculates feature contributions
-   Selects the most influential factors

Example:

``` text
Top factors:
- num__tenure: 0.545
- num__MonthlyCharges: -0.350
- cat__OnlineSecurity_Yes: 0.262
```

### 2. LLM Explanation

The important SHAP factors are passed to the Groq LLM.

The LLM produces:

-   `explanation` --- a plain-English interpretation of the prediction
-   `solution` --- an actionable retention recommendation

The LLM is used as an interpretation layer rather than as the prediction
model itself.

``` text
Customer Data
     │
     ▼
ML Model
     │
     ├──────────────► Churn Probability
     │
     ▼
    SHAP
     │
     ▼
Top Contributing Factors
     │
     ▼
   Groq LLM
     │
     ├──────────────► Explanation
     │
     └──────────────► Retention Recommendation
```

### Important Note

The LLM provider/model configuration can change over time. The
configured Groq model should therefore be kept configurable rather than
treated as a permanent dependency.

------------------------------------------------------------------------

## 📡 API

### Base URL

``` text
https://customer-churn-prediction-api-sx7w.onrender.com
```

### Health Check

``` http
GET /health
```

Response:

``` json
{
  "status": "ok"
}
```

### Root Endpoint

``` http
GET /
```

Response:

``` json
{
  "message": "Customer Churn Prediction API is running",
  "status": "healthy"
}
```

### Prediction Endpoint

``` http
POST /predict
```

Example request:

``` json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.5,
  "TotalCharges": 846.0
}
```

Example response structure:

``` json
{
  "prediction": 1,
  "churn_probability": 0.62,
  "top_factors": {
    "num__tenure": 1.14,
    "cat__Contract_Month-to-month": 0.67,
    "num__TotalCharges": -0.52
  },
  "explanation": "This customer is likely to churn primarily due to their customer profile and the factors identified by SHAP.",
  "solution": "Consider a targeted retention offer based on the customer's main churn drivers."
}
```

### Swagger UI

Interactive API documentation:

``` text
https://customer-churn-prediction-api-sx7w.onrender.com/docs
```

------------------------------------------------------------------------

## 🌐 Project Structure

``` text
customer-churn-prediction/
│
├── app/
│   ├── app.py                 # FastAPI backend
│   ├── schema.py              # Pydantic request/response schemas
│   └── shap.py                # SHAP explainability logic
│
├── llm/
│   ├── groq.py                # Groq API integration
│   └── prompt.py              # Prompt construction and response parsing
│
├── models/
│   └── logistic_model.pkl     # Trained ML pipeline
│
├── data/
│   └── Telco_Customer_Churn.csv
│
├── notebooks/
│   └── Customer_Churn_Prediction.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_model.py
│   └── test_schema.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── dockerfile.api
├── dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

------------------------------------------------------------------------

## 🧪 Testing

The project includes automated tests using **Pytest**.

Current test coverage includes:

### API Tests

-   Root endpoint
-   Invalid prediction request
-   Valid prediction request
-   Prediction response structure

### Schema Tests

-   Valid customer data
-   Invalid categorical values
-   Invalid numeric ranges
-   Missing required fields

### Model Tests

-   Model loading
-   Prediction generation
-   Probability output validation

Run the complete test suite:

``` bash
pytest -v
```

Current test suite:

``` text
12 passed
```

The LLM call is mocked in the API test so that tests do not depend on
the external Groq API.

------------------------------------------------------------------------

## 🔄 CI/CD Pipeline

GitHub Actions is used to automate testing and Docker image publishing.

### Pipeline

``` text
                 Git Push / Pull Request
                          │
                          ▼
                  GitHub Actions
                          │
                          ▼
                  Install Dependencies
                          │
                          ▼
                     Run Pytest
                          │
                    ┌─────┴─────┐
                    │           │
                  FAIL         PASS
                    │           │
                    ▼           ▼
                  Stop      Build Docker
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
              API Image              Streamlit Image
                    │                       │
                    └───────────┬───────────┘
                                ▼
                           Docker Hub
                                │
                                ▼
                              Render
                                │
                                ▼
                       Production Services
```

### GitHub Actions

The workflow:

1.  Checks out the repository
2.  Sets up Python
3.  Installs dependencies
4.  Runs the automated test suite
5.  Builds the API Docker image
6.  Builds the Streamlit Docker image
7.  Pushes both images to Docker Hub

Docker publishing runs after the test job succeeds and is restricted to
pushes on `main`.

### Docker Hub Images

**API**

``` text
divy049/customer-churn-prediction-api:latest
```

**Streamlit**

``` text
divy049/customer-churn-prediction-streamlit:latest
```

### CI/CD Improvement

The current pipeline publishes the images successfully. A further
automation step can trigger a Render deployment after the Docker Hub
push so that a new image is deployed immediately after every successful
production build.

------------------------------------------------------------------------

## 🐳 Running Locally with Docker Compose

Create a `.env` file in the repository root:

``` env
GROQ_API_KEY=your_key_here
```

Then run:

``` bash
docker compose up --build
```

Services:

``` text
API       → http://localhost:8000
Swagger   → http://localhost:8000/docs
Streamlit → http://localhost:8501
```

The local setup uses two independent containers:

``` text
┌─────────────────────────┐
│       Streamlit         │
│        :8501            │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│        FastAPI          │
│         :8000           │
└─────────────────────────┘
```

Docker Compose provides the local service-to-service networking.

------------------------------------------------------------------------

## 🐳 Docker Images

Pre-built images can be pulled from Docker Hub.

### API

``` bash
docker pull divy049/customer-churn-prediction-api:latest
```

### Streamlit

``` bash
docker pull divy049/customer-churn-prediction-streamlit:latest
```

Build locally:

``` bash
docker build -f dockerfile.api -t customer-churn-prediction-api .
docker build -f dockerfile.streamlit -t customer-churn-prediction-streamlit .
```

------------------------------------------------------------------------

## ☁️ Deployment

The application is deployed on **Render** using pre-built Docker images
from Docker Hub.

### API Service

``` text
Image:
divy049/customer-churn-prediction-api:latest

Port:
8000
```

### Streamlit Service

``` text
Image:
divy049/customer-churn-prediction-streamlit:latest

Port:
8501
```

The Streamlit service uses the deployed API's public URL through its
`API_URL` configuration.

### Deployment Architecture

``` text
                 GitHub
                    │
                    ▼
             GitHub Actions
                    │
             ┌──────┴──────┐
             ▼             ▼
          API Image   Streamlit Image
             │             │
             └──────┬──────┘
                    ▼
                Docker Hub
                    │
              ┌─────┴─────┐
              ▼           ▼
         Render API   Render UI
              │           │
              └─────┬─────┘
                    ▼
              Live Application
```

Render services are configured as **Image** services, so Render consumes
the Docker images instead of building the application directly from the
GitHub repository.

> **Free-tier note:** Render free instances can spin down after
> inactivity, which can make the first request slower.

------------------------------------------------------------------------

## 🔐 Environment Variables & Secrets

### Local

Use:

``` env
GROQ_API_KEY=your_key_here
```

Keep `.env` in `.gitignore`.

### GitHub Actions

Docker publishing uses repository secrets:

``` text
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
```

Secrets must never be hardcoded in workflow files.

### Render

Configure required application environment variables through the Render
service environment settings.

------------------------------------------------------------------------

## 💡 Business Recommendations

Common churn-reduction strategies identified by the model/explanation
layer include:

-   Encourage long-term contracts through targeted incentives
-   Offer relevant service bundles
-   Focus retention efforts on customers with short tenure
-   Identify customers with high monthly charges
-   Use service-related churn drivers to personalize retention offers

The application also generates a **customer-specific recommendation**
through the `solution` field.

------------------------------------------------------------------------

## 🛡️ Error Handling

The API validates incoming customer data using Pydantic before sending
it to the model.

Invalid requests are rejected with validation errors.

The core ML prediction and SHAP explanation are separate from the LLM
layer. If the LLM provider is unavailable or its configured model is
unavailable, the application can report the LLM failure while preserving
the underlying prediction information according to the API's
error-handling path.

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Automatically trigger Render deployment after successful Docker Hub
    pushes
-   Add structured application logging
-   Add production API rate limiting
-   Add authentication
-   Add model monitoring
-   Add model drift detection
-   Add automated model retraining
-   Version Docker images using Git commit SHA
-   Add end-to-end integration tests
-   Add application monitoring and alerting
-   Add CI checks for linting and formatting
-   Add security scanning for Docker images and dependencies

------------------------------------------------------------------------

## 👨‍💻 Author

**Divy Kushwaha**

B.Tech Computer Science & Engineering

GitHub: https://github.com/Divy-007

------------------------------------------------------------------------

## ⭐ Project Goal

This project demonstrates a production-oriented Machine Learning
workflow:

**Data → ML → Explainability → LLM → API → Frontend → Docker → CI/CD →
Cloud Deployment**
