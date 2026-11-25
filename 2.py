# 2.py: Model Evaluation (Classification Task)

# 1. Import libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
