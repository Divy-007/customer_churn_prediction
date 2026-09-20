from fastapi.testclient import TestClient
from app.app import app

client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_invalid_prediction_request():
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_valid_prediction(monkeypatch):
    # Mock the LLM response so the test doesn't call Groq
    def mock_llm_explanation(pred, prob, top_factors):
        return (
            "Customer is likely to churn because of the identified factors.",
            "Consider improving customer retention and support."
        )

    monkeypatch.setattr(
        "app.app.get_llm_explanation",
        mock_llm_explanation
    )

    customer_data = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.85,
        "TotalCharges": 958.20
    }

    response = client.post("/predict", json=customer_data)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "churn_probability" in result
    assert "top_factors" in result
    assert "explanation" in result
    assert "solution" in result

    assert result["prediction"] in [0, 1]
    assert 0 <= result["churn_probability"] <= 1
    assert isinstance(result["top_factors"], dict)
    assert isinstance(result["explanation"], str)
    assert isinstance(result["solution"], str)