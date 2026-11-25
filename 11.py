# 11.py: Dimensionality Reduction using PCA and t-SNE

# 1. Import libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 2. Data Collection (64-dimensional Digits dataset)
digits = load_digits()
X = digits.data
y = digits.target
print(f"Original Dataset Shape: {X.shape}")

# 3. Data Preprocessing (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Dimensionality Reduction using PCA (Linear)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA Variance Explained: {sum(pca.explained_variance_ratio_):.2f}")

# 5. Dimensionality Reduction using t-SNE (Non-Linear)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)
print("t-SNE embedding completed!")

# 6. Visualization & Comparison
plt.figure(figsize=(12, 5))

# Subplot 1: PCA Projection
plt.subplot(1, 2, 1)
scatter1 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=15, alpha=0.7)
plt.title(f"PCA (Linear Reduction, Var: {sum(pca.explained_variance_ratio_):.2f})")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter1)

# Subplot 2: t-SNE Projection
plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=15, alpha=0.7)
plt.title("t-SNE (Non-Linear Manifold)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.colorbar(scatter2)

plt.tight_layout()
plt.show()

print("Experiment 11 complete!")
