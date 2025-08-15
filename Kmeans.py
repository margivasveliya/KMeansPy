
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

print("Loading dataset...")
try:
    data = pd.read_csv("Mall_Customers.csv")
except FileNotFoundError:
    raise FileNotFoundError("Mall_Customers.csv not found. Place it in the same folder as this script.")

print("\nFirst few rows:")
print(data.head())

#
if "CustomerID" in data.columns:
    data.drop("CustomerID", axis=1, inplace=True)


if "Gender" in data.columns:
    le = LabelEncoder()
    data["Gender"] = le.fit_transform(data["Gender"])  


X = data.copy()

inertia_scores = []
k_values = range(2, 11)

print("\nFinding optimal K using Elbow Method...")
for k in k_values:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    inertia_scores.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(k_values, inertia_scores, 'o-', color='blue', markersize=6)
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


optimal_k = 5 
print(f"\nTraining KMeans with K={optimal_k}...")
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
labels = kmeans.fit_predict(X)


data["Cluster"] = labels

print("\nCluster counts:")
print(data["Cluster"].value_counts())

sil_score = silhouette_score(X, labels)
print(f"\nSilhouette Score: {sil_score:.3f}")

print("\nReducing dimensions for visualization...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(7, 5))
for c in range(optimal_k):
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')


centroids_2d = pca.transform(kmeans.cluster_centers_)
plt.scatter(
    centroids_2d[:, 0],
    centroids_2d[:, 1],
    s=200,
    c='red',
    marker='X',
    label='Centroids'
)

plt.title("Mall Customers Clusters (PCA Projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

