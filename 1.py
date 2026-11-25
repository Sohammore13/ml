# 1.py: Complete ML Pipeline & Data Preprocessing

# 1. Import libraries
import pandas as pd
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

# 5. Visualization (Feature distribution before vs after scaling)
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
