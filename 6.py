# 6.py: Support Vector Machine (SVM) Classification

# 1. Import libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

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
