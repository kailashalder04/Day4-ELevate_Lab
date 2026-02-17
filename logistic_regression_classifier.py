
# TASK 4: LOGISTIC REGRESSION - BINARY CLASSIFICATION


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)
from sklearn.impute import SimpleImputer

# ----------------------------------------------------
# Create Output Folder
# ----------------------------------------------------
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------
df = pd.read_csv(r"E:\New folder (3)\OneDrive\Desktop\Elevate Labs\Day4\data.csv")


print("First 5 rows:")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

# Drop ID column if present
if "id" in df.columns:
    df = df.drop("id", axis=1)

# Convert diagnosis to binary
# M = 1 (Malignant), B = 0 (Benign)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# ----------------------------------------------------
# Feature & Target Split
# ----------------------------------------------------
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# ----------------------------------------------------
# Handle Missing Values (Fix for NaN Error)
# ----------------------------------------------------
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# ----------------------------------------------------
# Train-Test Split
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ----------------------------------------------------
# Feature Scaling
# ----------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------------------
# Model Training
# ----------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ----------------------------------------------------
# Predictions
# ----------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ----------------------------------------------------
# Evaluation Metrics
# ----------------------------------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc_auc)

# ----------------------------------------------------
# Confusion Matrix Plot
# ----------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

# ----------------------------------------------------
# ROC Curve Plot
# ----------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
plt.close()

# ----------------------------------------------------
# Sigmoid Function Plot (Concept Visualization)
# ----------------------------------------------------
z = np.linspace(-10, 10, 100)
sigmoid = 1 / (1 + np.exp(-z))

plt.figure(figsize=(6, 5))
plt.plot(z, sigmoid)
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("Probability")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sigmoid_curve.png"))
plt.close()

print("\nAll graphs saved inside 'output' folder.")
print("Task 4 completed successfully.")
