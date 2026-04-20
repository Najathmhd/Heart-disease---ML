import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Loading dataset...")
df = pd.read_csv("data/Heart_disease_cleveland_new.csv")

# Ensure no missing values by filling or dropping (just in case)
df.dropna(inplace=True)

X = df.drop("target", axis=1)
y = df["target"]

print("Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

print(f"Model accuracy on training data: {rf_model.score(X, y) * 100:.2f}%")

model_filename = 'heart_disease_model.pkl'
joblib.dump(rf_model, model_filename)
print(f"Model successfully saved to {model_filename}")
