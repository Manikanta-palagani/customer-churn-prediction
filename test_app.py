from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


# Test 1: Health endpoint
def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "API is running"


# Test 2: Prediction endpoint
def test_prediction():
    customer_data = {
        "Gender": "Male",
        "Senior_Citizen": 0,
        "Tenure_Months": 12,
        "Contract_Type": "Month-to-Month",
        "Internet_Service": "Fiber Optic",
        "Payment_Method": "Electronic Check",
        "Monthly_Charges_INR": 1500,
        "Total_Charges_INR": 18000,
        "Tech_Support": "No",
        "Online_Security": "No",
        "Support_Tickets_6M": 5,
        "Avg_Data_Usage_GB": 25
    }

    response = client.post("/predict", json=customer_data)

    assert response.status_code == 200
    assert response.json()["Prediction"] in ["Churn", "Not Churn"]
