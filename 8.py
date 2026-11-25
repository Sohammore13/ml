# 8.py: Logic Gates using MP Neurons

# 1. Import libraries
import numpy as np
import matplotlib.pyplot as plt

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
