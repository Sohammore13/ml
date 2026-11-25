# 3.py: K-Fold Cross Validation, Learning & Validation Curves

# 1. Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve

# 2. Data Collection
iris = load_iris()
X = iris.data
y = iris.target

# 3. Model Definition & K-Fold Cross Validation
model = DecisionTreeClassifier(random_state=42)
cv_scores = cross_val_score(model, X, y, cv=5)
print("5-Fold Cross Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# 4. Learning Curve (Bias vs Variance)
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)

# 5. Validation Curve (Varying max_depth)
param_range = [1, 2, 3, 4, 5]
v_train, v_test = validation_curve(
    model, X, y, param_name="max_depth", param_range=param_range, cv=5
)

# 6. Visualization
plt.figure(figsize=(12, 4))

# Plot 1: Learning Curve
plt.subplot(1, 2, 1)
plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Training Score", color="blue")
plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Validation Score", color="green")
plt.title("Learning Curve (Bias vs Variance)")
plt.xlabel("Training Examples")
plt.ylabel("Score")
plt.legend()

# Plot 2: Validation Curve
plt.subplot(1, 2, 2)
plt.plot(param_range, np.mean(v_train, axis=1), label="Training Score", color="blue")
plt.plot(param_range, np.mean(v_test, axis=1), label="Validation Score", color="green")
plt.title("Validation Curve (max_depth)")
plt.xlabel("Max Depth")
plt.ylabel("Score")
plt.legend()

plt.show()
