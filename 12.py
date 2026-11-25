# 12.py: Automated End-to-End MLOps Pipeline

# 1. Import libraries
import logging
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

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
