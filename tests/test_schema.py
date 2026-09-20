import pytest
from pydantic import ValidationError

from app.schema import CustomerInput


VALID_CUSTOMER = {
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
    "TotalCharges": 958.20,
}


def test_valid_customer():
    customer = CustomerInput(**VALID_CUSTOMER)

    assert customer.gender.value == "Male"
    assert customer.tenure == 12
    assert customer.MonthlyCharges == 79.85


def test_invalid_senior_citizen():
    data = VALID_CUSTOMER.copy()
    data["SeniorCitizen"] = 2

    with pytest.raises(ValidationError):
        CustomerInput(**data)


def test_negative_tenure():
    data = VALID_CUSTOMER.copy()
    data["tenure"] = -1

    with pytest.raises(ValidationError):
        CustomerInput(**data)


def test_invalid_monthly_charges():
    data = VALID_CUSTOMER.copy()
    data["MonthlyCharges"] = 600

    with pytest.raises(ValidationError):
        CustomerInput(**data)


def test_invalid_gender():
    data = VALID_CUSTOMER.copy()
    data["gender"] = "Unknown"

    with pytest.raises(ValidationError):
        CustomerInput(**data)


def test_missing_required_field():
    data = VALID_CUSTOMER.copy()
    del data["tenure"]

    with pytest.raises(ValidationError):
        CustomerInput(**data)