# PROJECT 1: AI-Powered Fraud Detection System (Improved Version)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# Load Dataset
# -------------------------------
print("Loading dataset...")
data = pd.read_csv("creditcard.csv")

print("Dataset shape:", data.shape)
print(data.head())

# -------------------------------
# Data Understanding
# -------------------------------
print("\nClass distribution:")
print(data['Class'].value_counts())

# -------------------------------
# Feature & Target Separation
# -------------------------------
X = data.drop('Class', axis=1)
y = data['Class']

# -------------------------------
# Feature Scaling (VERY IMPORTANT)
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

# -------------------------------
# Train Fraud Detection Model
# -------------------------------
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

print("\nFraud Detection Model Trained Successfully")

# -------------------------------
# Model Evaluation
# -------------------------------
pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n")
print(classification_report(y_test, pred))

# -------------------------------
# Confusion Matrix Visualization
# -------------------------------
cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

# -------------------------------
# Test a Sample Transaction
# -------------------------------
sample = X_test[0].reshape(1, -1)
result = model.predict(sample)

print("\nSample Transaction Result:")
if result[0] == 1:
    print("Fraudulent Transaction Detected")
else:
    print("Normal Transaction")
