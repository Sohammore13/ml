"""
MACHINE LEARNING LAB EXPERIMENTS (ALL IN ONE)
Ready for Google Colab / Exam Practice
Pattern in each code:
  1. Import libraries
  2. Data Collection (Library Dataset)
  3. Data Preprocessing
  4. Split the dataset
  5. Model Training
  6. Prediction
  7. Model Evaluation & Visualization
"""

# ==============================================================================
# EXPERIMENT 1: Complete ML Pipeline & Data Preprocessing
# ==============================================================================
print("\n" + "="*50)
print("EXPERIMENT 1: ML Pipeline & Preprocessing")
print("="*50)

# 1. Import libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 2. Data Collection
iris = load_iris()
X = iris.data
y = iris.target

# 3. Data Preprocessing (Scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 5. Model Evaluation & Visualization
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.boxplot(X)
plt.title("Before Scaling (Raw Iris)")

plt.subplot(1, 2, 2)
plt.boxplot(X_scaled)
plt.title("After Scaling (StandardScaler)")
plt.show()

print("Preprocessing complete!")
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)


# ==============================================================================
# EXPERIMENT 2: Model Evaluation (Classification Task)
# ==============================================================================
print("\n" + "="*50)
print("EXPERIMENT 2: Model Evaluation")
print("="*50)

# 1. Import libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# 2. Data Collection
iris = load_iris()
X = iris.data
y = iris.target

# 3. Data Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Model Training
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Prediction
y_pred = model.predict(X_test)

# 7. Model Evaluation & Visualization
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualizing Confusion Matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix - Decision Tree")
plt.show()


# ==============================================================================
# EXPERIMENT 3: K-Fold Cross Validation, Learning & Validation Curves
# ==============================================================================
print("\n" + "="*50)
print("EXPERIMENT 3: Cross Validation & Bias-Variance Curves")
print("="*50)

# 1. Import libraries
import numpy as np
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


# ==============================================================================
# EXPERIMENT 4: Gradient Descent Algorithm for Cost Function Optimization
# ==============================================================================
print("\n" + "="*50)
print("EXPERIMENT 4: Gradient Descent Algorithm")
print("="*50)

# 1. Import libraries
from sklearn.datasets import make_regression

# 2. Data Collection (Using Scikit-Learn's dataset)
X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)
X = X.flatten()

# 3. Initialize Parameters
m = 0.0
c = 0.0
learning_rate = 0.01
epochs = 100
n = len(X)
cost_history = []

# 4. Gradient Descent Optimization Loop
for epoch in range(epochs):
    y_pred = m * X + c
    cost = (1/n) * sum((y - y_pred) ** 2)
    cost_history.append(cost)
    
    # Compute partial derivatives
    dm = (-2/n) * sum(X * (y - y_pred))
    dc = (-2/n) * sum(y - y_pred)
    
    # Update parameters
    m = m - learning_rate * dm
    c = c - learning_rate * dc

print(f"Final Model: y = {m:.2f} * x + {c:.2f}")

# 5. Visualization
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(cost_history, color='red')
plt.title("Cost Function vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Cost (MSE)")

plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', label='Data Points', alpha=0.6)
plt.plot(X, m * X + c, color='red', label='Fitted Line')
plt.title("Gradient Descent Fit")
plt.legend()
plt.show()


# ==============================================================================
# EXPERIMENT 5: Linear Regression with L1 (Lasso) and L2 (Ridge) Regularization
# ==============================================================================
print("\n" + "="*50)
print("EXPERIMENT 5: Linear Regression with L1 & L2")
print("="*50)

# 1. Import libraries
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error

# 2. Data Collection
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# 3. Data Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Model Training (Linear, Lasso / L1, Ridge / L2)
lr = LinearRegression().fit(X_train, y_train)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)
ridge = Ridge(alpha=0.1).fit(X_train, y_train)

# 6. Prediction and Evaluation
mse_lr = mean_squared_error(y_test, lr.predict(X_test))
mse_lasso = mean_squared_error(y_test, lasso.predict(X_test))
mse_ridge = mean_squared_error(y_test, ridge.predict(X_test))

print("Linear Regression MSE :", mse_lr)
print("Lasso (L1) MSE        :", mse_lasso)
print("Ridge (L2) MSE        :", mse_ridge)

# 7. Visualization (Comparing MSE of models)
models = ['Linear', 'Lasso (L1)', 'Ridge (L2)']
errors = [mse_lr, mse_lasso, mse_ridge]

plt.bar(models, errors, color=['skyblue', 'salmon', 'lightgreen'])
plt.title("Model Comparison - Mean Squared Error")
plt.ylabel("MSE")
plt.show()


# ==============================================================================
# EXPERIMENT 6: Support Vector Machines (SVM) for Classification
# ==============================================================================
print("\n" + "="*50)
print("EXPERIMENT 6: Support Vector Machines (SVM)")
print("="*50)

# 1. Import libraries
from sklearn.svm import SVC

# 2. Data Collection
iris = load_iris()
X = iris.data
y = iris.target

# 3. Data Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Model Training (SVM with RBF Kernel)
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)

# 6. Prediction
y_pred = model.predict(X_test)

# 7. Model Evaluation & Visualization
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualizing Confusion Matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix - SVM (RBF)")
plt.show()


# ==============================================================================
# EXPERIMENT 7: Logistic Regression for Binary Classification
# ==============================================================================
print("\n" + "="*50)
print("EXPERIMENT 7: Logistic Regression for Binary Classification")
print("="*50)

# 1. Import libraries
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay

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


# ==============================================================================
# EXPERIMENT 8: Logic Gates using MP Neurons
# ==============================================================================
print("\n" + "="*50)
print("EXPERIMENT 8: Logic Gates using MP Neurons")
print("="*50)

# 1. Import libraries (already imported numpy and matplotlib)

# 2. Data Collection (Logic Gate Truth Tables)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
y_or  = np.array([0, 1, 1, 1])

# 3. MP Neuron Function: sum(x * w) >= threshold
def mp_neuron(inputs, weights, threshold):
    total = np.dot(inputs, weights)
    return np.where(total >= threshold, 1, 0)

# 4. Prediction
pred_and = mp_neuron(X, weights=[1, 1], threshold=2)
pred_or  = mp_neuron(X, weights=[1, 1], threshold=1)

print("AND Gate Predictions:", pred_and)
print("OR Gate Predictions :", pred_or)

# 5. Visualization (Decision Boundaries)
plt.figure(figsize=(10, 4))

# Subplot 1: AND Gate
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_and, cmap='bwr', s=100, edgecolors='k')
x_line = np.linspace(-0.5, 1.5, 10)
plt.plot(x_line, 2 - x_line, 'g--', label="Boundary (x1+x2=2)")
plt.xlim(-0.5, 1.5); plt.ylim(-0.5, 1.5)
plt.title("AND Gate (Threshold = 2)")
plt.legend()

# Subplot 2: OR Gate
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_or, cmap='bwr', s=100, edgecolors='k')
plt.plot(x_line, 1 - x_line, 'g--', label="Boundary (x1+x2=1)")
plt.xlim(-0.5, 1.5); plt.ylim(-0.5, 1.5)
plt.title("OR Gate (Threshold = 1)")
plt.legend()
plt.show()


# ==============================================================================
# EXPERIMENT 9: Multi-Layer Perceptron (MLP) with Backpropagation Weight Changes
# ==============================================================================
print("\n" + "="*50)
print("EXPERIMENT 9: Multi-Layer Perceptron (MLP) with Backprop")
print("="*50)

# 1. Import libraries
from sklearn.datasets import make_classification

# 2. Data Collection (Library Dataset)
X, y = make_classification(n_samples=50, n_features=2, n_informative=2, n_redundant=0, random_state=42)
y = y.reshape(-1, 1)

# 3. Activation Function & Derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 4. Initialize Network Weights (2 -> 2 -> 1)
np.random.seed(42)
W1 = np.random.uniform(size=(2, 2))  # Input to Hidden
W2 = np.random.uniform(size=(2, 1))  # Hidden to Output
lr = 0.1
losses = []

print("Initial W1:\n", W1)
print("Initial W2:\n", W2)

# 5. Training with Backpropagation (Record Weight Changes)
for epoch in range(1, 501):
    # Forward Pass
    hidden_output = sigmoid(np.dot(X, W1))
    output = sigmoid(np.dot(hidden_output, W2))
    
    # Loss Calculation
    error = y - output
    losses.append(np.mean(error ** 2))
    
    # Backward Pass (Gradients)
    d_output = error * sigmoid_derivative(output)
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_output)
    
    # Weight Changes (Delta W)
    delta_W2 = hidden_output.T.dot(d_output) * lr
    delta_W1 = X.T.dot(d_hidden) * lr
    
    # Update Weights
    W2 += delta_W2
    W1 += delta_W1
    
    if epoch in [1, 250, 500]:
        print(f"\n--- Epoch {epoch} ---")
        print("Delta W2 (Weight Change):\n", delta_W2)
        print("Updated W2:\n", W2)

# 6. Visualization (Loss Reduction Curve)
plt.plot(losses, color='purple')
plt.title("MLP Backpropagation - Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.show()


# ==============================================================================
# EXPERIMENT 10: Supervised Hebbian Learning Rule
# ==============================================================================
print("\n" + "="*50)
print("EXPERIMENT 10: Supervised Hebbian Learning Rule")
print("="*50)

# 1. Import libraries (already imported)

# 2. Data Collection (Bipolar Dataset: AND Gate)
X = np.array([
    [-1, -1],
    [-1,  1],
    [ 1, -1],
    [ 1,  1]
])
y = np.array([-1, -1, -1, 1])

# 3. Initialize Weights and Bias
w = np.zeros(2)
b = 0.0

# 4. Supervised Hebbian Learning (Record Weight Changes)
print(f"{'Step':<5} | {'Input x':<10} | {'Target y':<8} | {'Delta w':<10} | {'New w':<10} | {'New b':<5}")
print("-" * 60)

for i in range(len(X)):
    delta_w = X[i] * y[i]
    delta_b = y[i]
    
    # Update
    w += delta_w
    b += delta_b
    
    print(f"{i+1:<5} | {str(X[i]):<10} | {y[i]:<8} | {str(delta_w):<10} | {str(w):<10} | {b:<5}")

# 5. Prediction & Classification Behaviour
print("\n--- Classification Results ---")
for i in range(len(X)):
    net = np.dot(X[i], w) + b
    y_pred = 1 if net >= 0 else -1
    print(f"Pattern {X[i]} -> Predicted: {y_pred:2d}, Actual: {y[i]:2d}")

# 6. Visualization (Decision Boundary)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=120, edgecolors='k')
# Decision boundary line: w1*x1 + w2*x2 + b = 0 => x2 = (-w1*x1 - b) / w2
x_vals = np.linspace(-2, 2, 10)
y_vals = (-w[0] * x_vals - b) / w[1]
plt.plot(x_vals, y_vals, 'k--', label="Decision Boundary")

plt.xlim(-2, 2); plt.ylim(-2, 2)
plt.title("Hebbian Learning - Decision Boundary")
plt.xlabel("X1"); plt.ylabel("X2")
plt.legend()
plt.show()

print("\nAll 10 experiments executed successfully!")


# ==============================================================================
# EXPERIMENT 11: Dimensionality Reduction using PCA and t-SNE
# ==============================================================================
print("\n" + "="*50)
print("EXPERIMENT 11: Dimensionality Reduction (PCA & t-SNE)")
print("="*50)

# 1. Import libraries
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 2. Data Collection (64-dimensional Digits dataset)
digits = load_digits()
X = digits.data
y = digits.target
print(f"Original Dataset Shape: {X.shape}")

# 3. Data Preprocessing (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Dimensionality Reduction using PCA (Linear)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA Variance Explained: {sum(pca.explained_variance_ratio_):.2f}")

# 5. Dimensionality Reduction using t-SNE (Non-Linear)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)
print("t-SNE embedding completed!")

# 6. Visualization & Comparison
plt.figure(figsize=(12, 5))

# Subplot 1: PCA Projection
plt.subplot(1, 2, 1)
scatter1 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=15, alpha=0.7)
plt.title(f"PCA (Linear Reduction, Var: {sum(pca.explained_variance_ratio_):.2f})")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter1)

# Subplot 2: t-SNE Projection
plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=15, alpha=0.7)
plt.title("t-SNE (Non-Linear Manifold)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.colorbar(scatter2)

plt.tight_layout()
plt.show()

print("Experiment 11 complete!")


# ==============================================================================
# EXPERIMENT 12: Automated End-to-End MLOps Pipeline
# ==============================================================================
print("\n" + "="*50)
print("EXPERIMENT 12: Automated End-to-End MLOps Pipeline")
print("="*50)

# 1. Import libraries
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Setup MLOps Logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logging.info("Step 1: Initializing MLOps Automated Pipeline...")

# 2. Data Collection
iris = load_iris()
X = iris.data
y = iris.target
logging.info(f"Step 2: Loaded raw dataset with shape {X.shape}")

# 3. Data Splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Build Automated End-to-End Pipeline (Preprocessing + Model)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Automated Pipeline Training
logging.info("Step 3: Training the end-to-end pipeline...")
pipeline.fit(X_train, y_train)

# 6. Prediction & Evaluation
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Step 4: Evaluation complete. Test Accuracy = {accuracy:.4f}")

print("\n--- Pipeline Evaluation Report ---")
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Model Versioning & Artifact Persistence
model_artifact = "model_pipeline_v1.joblib"
joblib.dump(pipeline, model_artifact)
logging.info(f"Step 5: Saved versioned model artifact to '{model_artifact}'")

# Visualization: Confusion Matrix
ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, cmap='Blues')
plt.title("MLOps Automated Pipeline - Confusion Matrix")
plt.show()

# Verification: Simulating inference with loaded model
loaded_pipeline = joblib.load(model_artifact)
sample_pred = loaded_pipeline.predict(X_test[:2])
print(f"\nReloaded Model Inference Check on 2 test samples: {sample_pred}")
print("Experiment 12 complete!")
