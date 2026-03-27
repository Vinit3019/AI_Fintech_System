from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

os.makedirs("models", exist_ok=True)

X, y = make_classification(n_samples=1000, n_features=4)

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "models/kyc_model.pkl")

print("kyc_model.pkl created successfully!")