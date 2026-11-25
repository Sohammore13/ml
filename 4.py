# 4.py: Gradient Descent Algorithm for Cost Function Optimization

# 1. Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# 2. Data Collection (Using Scikit-Learn's dataset)
X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)
X = X.flatten()  # 1D array for simple linear regression

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

# Plot 1: Cost Reduction Curve
plt.subplot(1, 2, 1)
plt.plot(cost_history, color='red')
plt.title("Cost Function vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Cost (MSE)")

# Plot 2: Data Points and Fitted Line
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', label='Data Points', alpha=0.6)
plt.plot(X, m * X + c, color='red', label='Fitted Line')
plt.title("Gradient Descent Fit")
plt.legend()

plt.show()
