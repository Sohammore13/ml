# 7.py: Logistic Regression for Binary Classification

# 1. Import libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, RocCurveDisplay

# 2. Data Collection (Standard Binary Classification Dataset)
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# 3. Data Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Prediction
y_pred = model.predict(X_test)

# 7. Model Evaluation & Visualization
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualizations: Confusion Matrix & ROC Curve
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=axes[0], cmap='Blues')
axes[0].set_title("Confusion Matrix")

RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[1])
axes[1].set_title("ROC Curve")

plt.show()
