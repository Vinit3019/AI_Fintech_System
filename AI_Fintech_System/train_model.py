from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# create models folder if missing
os.makedirs("models", exist_ok=True)

# generate sample dataset
X, y = make_classification(n_samples=1000, n_features=5)

# train model
model = RandomForestClassifier()
model.fit(X, y)

# save model
joblib.dump(model, "models/credit_model.pkl")

print("credit_model.pkl created successfully!")