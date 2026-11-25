# 5.py: Linear Regression with L1 (Lasso) and L2 (Ridge) Regularization

# 1. Import libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
