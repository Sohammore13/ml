# 10.py: Supervised Hebbian Learning Rule

# 1. Import libraries
import numpy as np
import matplotlib.pyplot as plt

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
