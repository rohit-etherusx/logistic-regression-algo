import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
print("📂 Loading dataset...")
data = pd.read_csv("data/default of credit card clients.csv")  # Fixed file path

# Remove irrelevant columns
data = data.iloc[:, 1:]  # Remove ID column

# Define features and target
X = data.iloc[:, :-1]  # All columns except the last (features)
y = data.iloc[:, -1]   # Last column (target)

# Handle categorical features
categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_encoded = encoder.fit_transform(X[categorical_features])

# Replace categorical columns with encoded values
X = X.drop(columns=categorical_features)
X = np.hstack((X.values, X_encoded))

# Apply feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression(class_weight="balanced", max_iter=5000, solver="saga", verbose=1, random_state=42)

# Training Model
print("🚀 Training Model...")
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print("\n✅ Model Training Complete!")
print(f"🎯 Final Test Accuracy: {test_acc:.2%}")
print("\n📜 Classification Report:")
print(classification_report(y_test, y_pred))

# Save Model, Encoder & Scaler
joblib.dump(model, "model/logistic_regression_model.pkl")
joblib.dump(encoder, "model/encoder.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("💾 Model, Encoder, and Scaler saved successfully!")
