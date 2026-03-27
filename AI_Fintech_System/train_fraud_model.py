from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# ensure models folder exists
os.makedirs("models", exist_ok=True)

# create dummy dataset
X, y = make_classification(n_samples=1000, n_features=6)

# train model
model = RandomForestClassifier()
model.fit(X, y)

# save model
joblib.dump(model, "models/fraud_model.pkl")

print("fraud_model.pkl created successfully!")