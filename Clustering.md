# Scikit-Learn Clustering Complete Guide

## Quick Reference Table

| Algorithm            | Best For                            | Time Complexity | Key Parameters                       | Pros                                  | Cons                                                      |
| -------------------- | ----------------------------------- | --------------- | ------------------------------------ | ------------------------------------- | --------------------------------------------------------- |
| **K-Means**          | Spherical clusters, known k         | O(n×k×i×d)      | `n_clusters`, `init`, `random_state` | Fast, simple, scalable                | Assumes spherical clusters, needs k                       |
| **DBSCAN**           | Arbitrary shapes, noise detection   | O(n log n)      | `eps`, `min_samples`                 | Finds arbitrary shapes, handles noise | Sensitive to parameters, struggles with varying densities |
| **Hierarchical**     | Hierarchy needed, small datasets    | O(n³)           | `n_clusters`, `linkage`              | No k needed, creates hierarchy        | Slow on large data, sensitive to noise                    |
| **Gaussian Mixture** | Probabilistic, overlapping clusters | O(n×k×i×d)      | `n_components`, `covariance_type`    | Soft clustering, flexible shapes      | Needs k, can overfit                                      |
| **Mean Shift**       | Unknown k, non-parametric           | O(n²)           | `bandwidth`, `cluster_all`           | Finds k automatically                 | Very slow, bandwidth selection critical                   |
| **Spectral**         | Non-convex shapes, graph data       | O(n³)           | `n_clusters`, `affinity`, `gamma`    | Handles complex shapes                | Expensive, needs k                                        |

## 1. K-Means Clustering

### Basic Usage

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, n_features=2,
                       random_state=42, cluster_std=0.60)

# Basic K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X)

# Get cluster centers
centers = kmeans.cluster_centers_
```

### Key Parameters

- **`n_clusters`**: Number of clusters (must specify)
- **`init`**: Initialization method
  - `'k-means++'` (default): Smart initialization
  - `'random'`: Random initialization
  - Array: Custom initial centers
- **`n_init`**: Number of random initializations (default: 10)
- **`max_iter`**: Maximum iterations (default: 300)
- **`tol`**: Convergence tolerance (default: 1e-4)
- **`algorithm`**: `'lloyd'`, `'elkan'`, `'auto'`, `'full'`

### Finding Optimal K

```python
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# Elbow Method
def elbow_method(X, max_k=10):
    inertias = []
    K_range = range(1, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()

# Silhouette Analysis
def silhouette_analysis(X, max_k=10):
    silhouette_scores = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)

    plt.plot(K_range, silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.show()
```

### Advanced K-Means

```python
# Mini-batch K-Means for large datasets
from sklearn.cluster import MiniBatchKMeans

mini_kmeans = MiniBatchKMeans(n_clusters=4, batch_size=100, random_state=42)
y_pred = mini_kmeans.fit_predict(X)
```

## 2. DBSCAN (Density-Based Spatial Clustering)

### Basic Usage

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Standardize features (important for DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_pred = dbscan.fit_predict(X_scaled)

# Number of clusters (excluding noise)
n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
n_noise = list(y_pred).count(-1)

print(f'Clusters: {n_clusters}, Noise points: {n_noise}')
```

### Key Parameters

- **`eps`**: Maximum distance between samples in same neighborhood
- **`min_samples`**: Minimum samples in neighborhood to form core point
- **`metric`**: Distance metric (`'euclidean'`, `'manhattan'`, `'cosine'`, etc.)
- **`algorithm`**: `'auto'`, `'ball_tree'`, `'kd_tree'`, `'brute'`

### Parameter Tuning

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Find optimal eps using k-distance graph
def find_optimal_eps(X, min_samples=5):
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)

    # Sort distances to min_samples-th nearest neighbor
    distances = np.sort(distances[:, min_samples-1], axis=0)

    plt.plot(distances)
    plt.ylabel(f'{min_samples}-NN Distance')
    plt.xlabel('Data Points sorted by distance')
    plt.title('K-distance Graph')
    plt.show()

    return distances

# Grid search for parameters
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score

def dbscan_grid_search(X, param_grid):
    best_score = -1
    best_params = None

    for params in ParameterGrid(param_grid):
        dbscan = DBSCAN(**params)
        labels = dbscan.fit_predict(X)

        # Skip if all points are noise or all in one cluster
        if len(set(labels)) <= 1 or len(set(labels)) == len(labels):
            continue

        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score

param_grid = {
    'eps': [0.1, 0.3, 0.5, 0.7, 1.0],
    'min_samples': [3, 5, 7, 10]
}
```

## 3. Hierarchical Clustering

### Basic Usage

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')
y_pred = agg_clustering.fit_predict(X)

# Create dendrogram
def plot_dendrogram(X, method='ward'):
    linkage_matrix = linkage(X, method=method)

    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    plt.title(f'Hierarchical Clustering Dendrogram ({method})')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()

    return linkage_matrix
```

### Key Parameters

- **`n_clusters`**: Number of clusters to find
- **`linkage`**: Linkage criterion
  - `'ward'`: Minimizes variance (only with Euclidean)
  - `'complete'`: Maximum distances
  - `'average'`: Average distances
  - `'single'`: Minimum distances
- **`affinity`**: Distance metric
- **`distance_threshold`**: Distance threshold (alternative to n_clusters)

### Advanced Hierarchical Clustering

```python
# Using distance threshold instead of n_clusters
agg_clustering = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=1.5,
    linkage='ward'
)
y_pred = agg_clustering.fit_predict(X)

# Connectivity constraints (for image segmentation, etc.)
from sklearn.feature_extraction import image
from sklearn.datasets import load_sample_image

# Example with connectivity
connectivity = image.grid_to_graph(*X.shape[:2]) if X.ndim > 1 else None
agg_clustering = AgglomerativeClustering(
    n_clusters=4,
    connectivity=connectivity,
    linkage='ward'
)
```

## 4. Gaussian Mixture Models

### Basic Usage

```python
from sklearn.mixture import GaussianMixture

# Gaussian Mixture Model
gmm = GaussianMixture(n_components=4, random_state=42)
y_pred = gmm.fit_predict(X)

# Get probabilities
probabilities = gmm.predict_proba(X)

# Get parameters
means = gmm.means_
covariances = gmm.covariances_
weights = gmm.weights_
```

### Key Parameters

- **`n_components`**: Number of mixture components
- **`covariance_type`**: Covariance matrix type
  - `'full'`: Each component has its own general covariance matrix
  - `'tied'`: All components share the same general covariance matrix
  - `'diag'`: Each component has its own diagonal covariance matrix
  - `'spherical'`: Each component has its own single variance
- **`init_params`**: Method to initialize parameters (`'kmeans'`, `'random'`)
- **`max_iter`**: Maximum EM iterations
- **`tol`**: Convergence threshold

### Model Selection

```python
from sklearn.model_selection import cross_val_score

# AIC/BIC for model selection
def select_gmm_components(X, max_components=10):
    n_components_range = range(1, max_components + 1)
    aic_scores = []
    bic_scores = []

    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X)
        aic_scores.append(gmm.aic(X))
        bic_scores.append(gmm.bic(X))

    plt.figure(figsize=(10, 6))
    plt.plot(n_components_range, aic_scores, 'bo-', label='AIC')
    plt.plot(n_components_range, bic_scores, 'ro-', label='BIC')
    plt.xlabel('Number of Components')
    plt.ylabel('Information Criterion')
    plt.legend()
    plt.title('Model Selection for GMM')
    plt.show()

    return n_components_range[np.argmin(bic_scores)]
```

## 5. Mean Shift

### Basic Usage

```python
from sklearn.cluster import MeanShift, estimate_bandwidth

# Estimate bandwidth
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

# Mean Shift
mean_shift = MeanShift(bandwidth=bandwidth)
y_pred = mean_shift.fit_predict(X)

# Cluster centers
cluster_centers = mean_shift.cluster_centers_
n_clusters = len(cluster_centers)
```

### Key Parameters

- **`bandwidth`**: Kernel bandwidth
- **`seeds`**: Seeds for optimization
- **`bin_seeding`**: Use binning for seeding
- **`min_bin_freq`**: Minimum frequency for bin seeding
- **`cluster_all`**: Whether to cluster all points

### Bandwidth Selection

```python
from sklearn.model_selection import ParameterGrid

def select_bandwidth(X, quantile_range=np.arange(0.1, 0.9, 0.1)):
    bandwidths = []
    n_clusters_list = []

    for quantile in quantile_range:
        bandwidth = estimate_bandwidth(X, quantile=quantile)
        if bandwidth > 0:  # Ensure bandwidth is positive
            ms = MeanShift(bandwidth=bandwidth)
            ms.fit(X)
            bandwidths.append(bandwidth)
            n_clusters_list.append(len(ms.cluster_centers_))

    plt.figure(figsize=(10, 6))
    plt.plot(bandwidths, n_clusters_list, 'bo-')
    plt.xlabel('Bandwidth')
    plt.ylabel('Number of Clusters')
    plt.title('Bandwidth vs Number of Clusters')
    plt.show()
```

## 6. Spectral Clustering

### Basic Usage

```python
from sklearn.cluster import SpectralClustering

# Spectral Clustering
spectral = SpectralClustering(n_clusters=4, affinity='rbf', random_state=42)
y_pred = spectral.fit_predict(X)
```

### Key Parameters

- **`n_clusters`**: Number of clusters
- **`affinity`**: Affinity matrix construction
  - `'rbf'`: RBF kernel
  - `'nearest_neighbors'`: k-nearest neighbors
  - `'precomputed'`: User-provided matrix
- **`gamma`**: Kernel coefficient for RBF
- **`n_neighbors`**: Number of neighbors for kNN graph
- **`eigen_solver`**: Eigenvalue decomposition strategy

### Advanced Spectral Clustering

```python
# Custom affinity matrix
from sklearn.metrics.pairwise import rbf_kernel

def custom_spectral_clustering(X, n_clusters=4, gamma=1.0):
    # Create custom affinity matrix
    affinity_matrix = rbf_kernel(X, gamma=gamma)

    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42
    )
    y_pred = spectral.fit_predict(affinity_matrix)
    return y_pred
```

## Evaluation Metrics

### Internal Metrics (No ground truth needed)

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate_clustering_internal(X, labels):
    # Silhouette Score (higher is better, range: -1 to 1)
    silhouette = silhouette_score(X, labels)

    # Calinski-Harabasz Score (higher is better)
    calinski_harabasz = calinski_harabasz_score(X, labels)

    # Davies-Bouldin Score (lower is better)
    davies_bouldin = davies_bouldin_score(X, labels)

    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz:.3f}")
    print(f"Davies-Bouldin Score: {davies_bouldin:.3f}")

    return silhouette, calinski_harabasz, davies_bouldin
```

### External Metrics (Ground truth available)

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score

def evaluate_clustering_external(y_true, y_pred):
    # Adjusted Rand Index (higher is better, range: -1 to 1)
    ari = adjusted_rand_score(y_true, y_pred)

    # Normalized Mutual Information (higher is better, range: 0 to 1)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    # Fowlkes-Mallows Score (higher is better, range: 0 to 1)
    fmi = fowlkes_mallows_score(y_true, y_pred)

    print(f"Adjusted Rand Index: {ari:.3f}")
    print(f"Normalized Mutual Information: {nmi:.3f}")
    print(f"Fowlkes-Mallows Index: {fmi:.3f}")

    return ari, nmi, fmi
```

## Preprocessing and Data Preparation

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standard Scaling (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-Max Scaling (range 0-1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Robust Scaling (median-based)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

### Dimensionality Reduction

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PCA before clustering
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
```

## Complete Workflow Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def complete_clustering_workflow(X, algorithms=None):
    if algorithms is None:
        algorithms = {
            'K-Means': KMeans(n_clusters=4, random_state=42),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
            'Hierarchical': AgglomerativeClustering(n_clusters=4)
        }

    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dimensionality reduction for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Apply clustering algorithms
    results = {}
    fig, axes = plt.subplots(1, len(algorithms), figsize=(15, 5))

    for i, (name, algorithm) in enumerate(algorithms.items()):
        # Fit and predict
        labels = algorithm.fit_predict(X_scaled)

        # Calculate silhouette score
        if len(set(labels)) > 1:
            sil_score = silhouette_score(X_scaled, labels)
        else:
            sil_score = -1

        results[name] = {
            'labels': labels,
            'silhouette_score': sil_score,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0)
        }

        # Plot results
        scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
        axes[i].set_title(f'{name}\nClusters: {results[name]["n_clusters"]}\nSilhouette: {sil_score:.3f}')
        axes[i].set_xlabel('First Principal Component')
        axes[i].set_ylabel('Second Principal Component')

    plt.tight_layout()
    plt.show()

    return results

# Example usage
X, _ = make_blobs(n_samples=300, centers=4, n_features=8, random_state=42)
results = complete_clustering_workflow(X)
```

## Tips and Best Practices

### 1. Algorithm Selection Guide

- **Spherical, similar-sized clusters**: K-Means
- **Arbitrary shapes, noise present**: DBSCAN
- **Need hierarchy or dendogram**: Hierarchical
- **Probabilistic assignments**: Gaussian Mixture
- **Unknown number of clusters**: Mean Shift, DBSCAN
- **Complex, non-convex shapes**: Spectral Clustering

### 2. Common Pitfalls

- **Not scaling data**: Critical for distance-based algorithms
- **Wrong distance metric**: Choose appropriate metric for your data
- **Ignoring curse of dimensionality**: Consider PCA for high-dimensional data
- **Not validating results**: Always evaluate with multiple metrics
- **Assuming clusters exist**: Not all data has meaningful clusters

### 3. Performance Optimization

- Use **MiniBatchKMeans** for large datasets
- Consider **approximate algorithms** for speed
- **Parallelize** when possible (`n_jobs=-1`)
- **Sample data** for parameter tuning on large datasets

### 4. Validation Strategies

- **Multiple runs**: Use different random seeds
- **Cross-validation**: For parameter selection
- **Stability analysis**: Consistent results across runs
- **Domain knowledge**: Results should make sense

This guide provides a comprehensive foundation for clustering with scikit-learn. Remember to always validate your results and choose algorithms based on your specific data characteristics and requirements.
