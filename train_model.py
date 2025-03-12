import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load dataset; note that the CSV file is "default of credit card clients.csv" (without underscore)
df = pd.read_csv("data/default of credit card clients.csv", skiprows=1)

# Optionally, rename columns if necessary (here we assume the CSV's columns match expected order)
# For example, if the first column is an ID, drop it:
df = df.iloc[:, 1:]

# Assume that the last column is the target variable (0 = No Default, 1 = Default)
X = df.iloc[:, :-1]  # All feature columns
y = df.iloc[:, -1]   # Target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Apply feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
# Here, we increase max_iter to ensure convergence and set C to 10 for less regularization.
model = LogisticRegression(max_iter=2000, solver='saga', class_weight='balanced', C=10, random_state=42)
model.fit(X_train_scaled, y_train_res)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Training Complete! Accuracy: {acc * 100:.2f}%")
print("📜 Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model, scaler, and feature names
joblib.dump(model, "model/logistic_regression_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(X.columns.tolist(), "model/feature_names.pkl")
print("💾 Model, scaler, and feature names saved successfully!")
