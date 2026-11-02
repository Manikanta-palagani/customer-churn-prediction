from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Customer Churn Prediction API")

model = joblib.load("best_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
label_encoder = joblib.load("label_encoder.pkl")


class CustomerData(BaseModel):
    Gender: str
    Senior_Citizen: int
    Tenure_Months: float
    Contract_Type: str
    Internet_Service: str
    Payment_Method: str
    Monthly_Charges_INR: float
    Total_Charges_INR: float
    Tech_Support: str
    Online_Security: str
    Support_Tickets_6M: float
    Avg_Data_Usage_GB: float


@app.get("/health")
def health():
    return {"status": "API is running"}


@app.post("/predict")
def predict(data: CustomerData):

    customer = pd.DataFrame([data.dict()])

    customer["Tenure_Category"] = pd.cut(
        customer["Tenure_Months"],
        bins=[-1, 12, 24, 48, 72],
        labels=["New", "Medium", "Long", "Loyal"]
    )

    customer["Average_Monthly_Spending"] = (
        customer["Total_Charges_INR"] /
        customer["Tenure_Months"].replace(0, float("nan"))
    ).fillna(0)

    customer["Internet_Usage"] = (
        customer["Internet_Service"] != "No"
    ).astype(int)

    customer["Support_Usage"] = (
        customer["Tech_Support"] == "Yes"
    ).astype(int)

    customer["Security_Usage"] = (
        customer["Online_Security"] == "Yes"
    ).astype(int)

    customer["Total_Services_Used"] = (
        customer["Internet_Usage"] +
        customer["Support_Usage"] +
        customer["Security_Usage"]
    )

    processed_data = preprocessor.transform(customer)
    prediction = model.predict(processed_data)
    result = label_encoder.inverse_transform(prediction)

    return {"Prediction": result[0]}
