# 9.py: Multi-Layer Perceptron (MLP) with Backpropagation Weight Changes

# 1. Import libraries
import numpy as np
import matplotlib.pyplot as plt
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
