# Machine Learning Lab Experiments & Exam Preparation Manual

Comprehensive implementation of Machine Learning lab experiments following a universal, exam-friendly **7-step pattern** with built-in library datasets and visualizations.

---

## 📌 Syllabus & Course Outcome (CO) Mapping

| Exp # | Experiment Title | Course Outcome | Built-in Dataset | Visualization |
| :---: | :--- | :---: | :--- | :--- |
| **1** | Complete ML Pipeline & Data Preprocessing | **CO1** | `load_iris()` | Feature boxplots (Before vs After Scaling) |
| **2** | Model Evaluation using Classification Task | **CO1** | `load_iris()` | `ConfusionMatrixDisplay` heatmap |
| **3** | Cross-Validation, Bias-Variance Curves | **CO1** | `load_iris()` | Learning Curve & Validation Curve |
| **4** | Gradient Descent Cost Function Optimization | **CO3** | `make_regression()` | Cost vs Epochs curve & regression line |
| **5** | Linear Regression with L1 (Lasso) & L2 (Ridge) | **CO3** | `load_diabetes()` | Mean Squared Error (MSE) comparison bar chart |
| **6** | Support Vector Machines (SVM) Classification | **CO3** | `load_iris()` | SVM confusion matrix display |
| **7** | Logistic Regression for Binary Classification | **CO3** | `load_breast_cancer()` | Side-by-side Confusion Matrix & ROC Curve |
| **8** | Logic Gates using MP Neurons | **CO4** | Truth Tables | 2D Decision boundaries for AND & OR gates |
| **9** | Multi-Layer Perceptron (MLP) Backprop | **CO4** | `make_classification()` | Loss curve + printed weight changes ($\Delta W$) |
| **10** | Supervised Hebbian Learning Rule | **CO4** | Bipolar Patterns | Step-by-step weight updates table + Decision line |
| **11** | Dimensionality Reduction (PCA & t-SNE) | **CO5** | `load_digits()` | Side-by-side 2D PCA vs t-SNE cluster plots |
| **12** | MLOps Automated End-to-End Pipeline | **CO6** | `load_iris()` | Structured logging, pipeline fit, model artifact |

---

## 🎯 The Universal 7-Step Pattern

Every experiment is structured using the exact same flow:
1. **Import libraries** (`sklearn`, `numpy`, `matplotlib.pyplot`)
2. **Data Collection** (Standard library datasets)
3. **Data Preprocessing** (`StandardScaler().fit_transform(X)`)
4. **Split Dataset** (`train_test_split(X, y, test_size=0.2, random_state=42)`)
5. **Model Training** (`model.fit(X_train, y_train)`)
6. **Prediction** (`y_pred = model.predict(X_test)`)
7. **Model Evaluation & Visualization** (`metrics` & `plt.show()`)

---

## 📚 Study Resources Included

* **`Machine_Learning_Lab_Manual_With_Theory.pdf`**: Complete printable manual with **theoretical background, mathematical formulations, and code** for all 12 experiments.
* **`all_codes.ipynb`**: Ready-to-use Google Colab notebook with 12 distinct runnable cells.
* **`all_codes.py`**: All 12 experiments in a single standalone Python script.
* **`1.py` to `12.py`**: Modular individual scripts for each experiment.

---

## 🚀 How to Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Click on **Upload** and select `all_codes.ipynb`.
3. Run any cell using `Shift + Enter`.
