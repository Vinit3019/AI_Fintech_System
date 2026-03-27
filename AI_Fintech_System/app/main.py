from fastapi import FastAPI
import joblib
import os

app = FastAPI(title="AI Fintech Risk Engine API")

# Load models safely
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

credit_model = load_model("models/credit_model.pkl")
fraud_model = load_model("models/fraud_model.pkl")
kyc_model = load_model("models/kyc_model.pkl")


@app.get("/")
def home():
    return {"message": "AI Fintech Backend Running Successfully"}


from fastapi import Form

@app.post("/predict-credit-risk")
def predict_credit(data: str = Form(...)):
    if credit_model is None:
        return {"error": "Credit model not available"}

    try:
        values = list(map(float, data.split(",")))
        prediction = credit_model.predict([values])
        return {"credit_risk_prediction": int(prediction[0])}

    except Exception as e:
        return {"error": str(e)}
    if credit_model is None:
        return {"error": "Credit model not available"}

    prediction = credit_model.predict([data])
    return {"credit_risk_prediction": int(prediction[0])}


@app.post("/predict-fraud")
def predict_fraud(data: str = Form(...)):
    values = list(map(float, data.split(",")))
    prediction = fraud_model.predict([values])
    return {"fraud_prediction": int(prediction[0])}


@app.post("/predict-kyc")
def predict_kyc(data: str = Form(...)):
    values = list(map(float, data.split(",")))
    prediction = kyc_model.predict([values])
    return {"kyc_prediction": int(prediction[0])}