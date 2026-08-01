"""
Pydantic schemas for the Churn Prediction API.
Keep this file in sync with whatever preprocessing/encoding your model
was trained on — if the CSV's category values change, update the enums here.
"""

from enum import Enum
from pydantic import BaseModel, Field


# ---- Enums: constrain categorical fields to exactly what the model was trained on ----
class YesNo(str, Enum):
    yes = "Yes"
    no = "No"


class InternetDependentYesNo(str, Enum):
    """For fields where 'No internet service' is also a valid category."""
    yes = "Yes"
    no = "No"
    no_internet = "No internet service"


class Gender(str, Enum):
    male = "Male"
    female = "Female"


class MultipleLines(str, Enum):
    yes = "Yes"
    no = "No"
    no_phone_service = "No phone service"


class InternetService(str, Enum):
    dsl = "DSL"
    fiber = "Fiber optic"
    no = "No"


class Contract(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class PaymentMethod(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer (automatic)"
    credit_card = "Credit card (automatic)"


# ---- Request schema (standard Telco Customer Churn columns) ----
class CustomerInput(BaseModel):
    gender: Gender
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(..., ge=0, le=100, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: MultipleLines
    InternetService: InternetService
    OnlineSecurity: InternetDependentYesNo
    OnlineBackup: InternetDependentYesNo
    DeviceProtection: InternetDependentYesNo
    TechSupport: InternetDependentYesNo
    StreamingTV: InternetDependentYesNo
    StreamingMovies: InternetDependentYesNo
    Contract: Contract
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethod
    MonthlyCharges: float = Field(..., ge=0, le=500)
    TotalCharges: float = Field(..., ge=0, le=50000)


# ---- Response schema (nice to have — documents your API output too) ----
class ChurnPrediction(BaseModel):
    prediction: int
    churn_probability: float
    top_factors: dict
    explanation: str
    solution:str