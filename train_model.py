import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load Dataset
df = pd.read_csv("credit_card.csv")

# Features
X = df.drop("Approved", axis=1)

# Target
y = df["Approved"]

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save Model
pickle.dump(model, open("model.pkl", "wb"))

print("Model Trained Successfully!")