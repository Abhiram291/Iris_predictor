# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle


df = pd.read_csv("/Users/abhiramayla/Downloads/Iris.csv")
df.columns = df.columns.str.strip()


df = df.drop('Id', axis=1)

# ✅ Use correct feature columns
feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X = df[feature_cols]
y = df["Species"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = RandomForestClassifier()
model.fit(X, y_encoded)

# Save model and encoder
with open("model.pkl", "wb") as f:
    pickle.dump((model, le), f)

print("✅ Model and label encoder saved.")
