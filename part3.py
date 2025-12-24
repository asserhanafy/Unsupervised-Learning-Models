import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import comb
from collections import Counter
import time

# ============================================================================
# INTERNAL VALIDATION METRICS
# ============================================================================

def compute_euclidean_distance(X, centers):
    """
    Compute Euclidean distance between points and centers.
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
    centers : array, shape (n_centers, n_features)
    
    Returns:
    --------
    distances : array, shape (n_samples, n_centers)
    """
    # Efficient vectorized computation: ||x - c||^2 = ||x||^2 + ||c||^2 - 2<x,c>
    X_sq = np.sum(X**2, axis=1, keepdims=True)  # (n, 1)
    C_sq = np.sum(centers**2, axis=1, keepdims=True).T  # (1, k)
    XC = np.dot(X, centers.T)  # (n, k)
    
    distances = np.sqrt(np.maximum(X_sq + C_sq - 2*XC, 0))
    return distances


def silhouette_score(X, labels):
    """
    Compute Silhouette Score for clustering.
    
    Score ranges from -1 to 1:
    - 1: Perfect clustering
    - 0: Overlapping clusters
    - -1: Wrong clustering
    
    Formula for sample i:
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    
    where:
    - a(i) = average distance to points in same cluster
    - b(i) = average distance to points in nearest cluster
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
    labels : array, shape (n_samples,)
        Cluster labels
    
    Returns:
    --------
    score : float
        Average silhouette score
    sample_scores : array, shape (n_samples,)
        Silhouette score for each sample
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters == 1 or n_clusters == n_samples:
        return 0.0, np.zeros(n_samples)
    
    # Compute pairwise distances
    distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        distances[i] = np.sqrt(np.sum((X - X[i])**2, axis=1))
    
    sample_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Find points in same cluster
        same_cluster_mask = labels == labels[i]
        same_cluster_points = np.sum(same_cluster_mask) - 1  # Exclude point itself
        
        if same_cluster_points == 0:
            # Singleton cluster
            sample_scores[i] = 0
            continue
        
        # a(i): mean distance to points in same cluster
        a_i = np.sum(distances[i, same_cluster_mask]) / same_cluster_points
        
        # b(i): mean distance to points in nearest other cluster
        b_i = np.inf
        for cluster_label in unique_labels:
            if cluster_label == labels[i]:
                continue
            
            other_cluster_mask = labels == cluster_label
            mean_dist = np.mean(distances[i, other_cluster_mask])
            b_i = min(b_i, mean_dist)
        
        # Silhouette coefficient
        sample_scores[i] = (b_i - a_i) / max(a_i, b_i)
    
    return np.mean(sample_scores), sample_scores


def davies_bouldin_index(X, labels):
    """
    Compute Davies-Bouldin Index (lower is better).
    
    Measures average similarity between each cluster and its most similar cluster.
    
    Formula:
    DB = (1/k) * Σ max_{j≠i} [(σᵢ + σⱼ) / d(cᵢ, cⱼ)]
    
    where:
    - σᵢ = average distance of points in cluster i to centroid
    - d(cᵢ, cⱼ) = distance between centroids
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
    labels : array, shape (n_samples,)
    
    Returns:
    --------
    db_index : float
        Davies-Bouldin index (lower is better)
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters == 1:
        return 0.0
    
    # Compute centroids
    centroids = np.array([X[labels == k].mean(axis=0) for k in unique_labels])
    
    # Compute cluster dispersions (σᵢ)
    dispersions = np.zeros(n_clusters)
    for i, k in enumerate(unique_labels):
        cluster_points = X[labels == k]
        dispersions[i] = np.mean(np.sqrt(np.sum((cluster_points - centroids[i])**2, axis=1)))
    
    # Compute pairwise centroid distances
    centroid_distances = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            dist = np.sqrt(np.sum((centroids[i] - centroids[j])**2))
            centroid_distances[i, j] = dist
            centroid_distances[j, i] = dist
    
    # Compute DB index
    db_values = np.zeros(n_clusters)
    for i in range(n_clusters):
        max_ratio = 0
        for j in range(n_clusters):
            if i != j:
                ratio = (dispersions[i] + dispersions[j]) / centroid_distances[i, j]
                max_ratio = max(max_ratio, ratio)
        db_values[i] = max_ratio
    
    return np.mean(db_values)


def calinski_harabasz_index(X, labels):
    """
    Compute Calinski-Harabasz Index (Variance Ratio Criterion).
    Higher values indicate better clustering.
    
    Formula:
    CH = [Tr(B_k) / (k-1)] / [Tr(W_k) / (n-k)]
    
    where:
    - B_k = between-cluster dispersion matrix
    - W_k = within-cluster dispersion matrix
    - Tr() = trace (sum of diagonal elements)
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
    labels : array, shape (n_samples,)
    
    Returns:
    --------
    ch_index : float
        Calinski-Harabasz index (higher is better)
    """
    n_samples, n_features = X.shape
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters == 1 or n_clusters == n_samples:
        return 0.0
    
    # Overall mean
    overall_mean = np.mean(X, axis=0)
    
    # Between-cluster sum of squares (SSB)
    ssb = 0
    for k in unique_labels:
        cluster_points = X[labels == k]
        n_k = len(cluster_points)
        cluster_mean = np.mean(cluster_points, axis=0)
        ssb += n_k * np.sum((cluster_mean - overall_mean)**2)
    
    # Within-cluster sum of squares (SSW)
    ssw = 0
    for k in unique_labels:
        cluster_points = X[labels == k]
        cluster_mean = np.mean(cluster_points, axis=0)
        ssw += np.sum((cluster_points - cluster_mean)**2)
    
    # Calinski-Harabasz index
    ch_index = (ssb / (n_clusters - 1)) / (ssw / (n_samples - n_clusters))
    
    return ch_index


def within_cluster_sum_of_squares(X, labels):
    """
    Compute Within-Cluster Sum of Squares (WCSS).
    Also known as inertia. Lower is better.
    
    Formula:
    WCSS = Σᵢ Σ_{x∈Cᵢ} ||x - μᵢ||²
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
    labels : array, shape (n_samples,)
    
    Returns:
    --------
    wcss : float
    """
    unique_labels = np.unique(labels)
    wcss = 0
    
    for k in unique_labels:
        cluster_points = X[labels == k]
        cluster_mean = np.mean(cluster_points, axis=0)
        wcss += np.sum((cluster_points - cluster_mean)**2)
    
    return wcss


def compute_bic_aic(X, n_components, log_likelihood, covariance_type):
    """
    Compute BIC and AIC for GMM.
    
    BIC = -2 * log(L) + k * log(n)
    AIC = -2 * log(L) + 2k
    
    where:
    - log(L) = log-likelihood
    - k = number of parameters
    - n = number of samples
    
    Lower BIC/AIC indicates better model.
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
    n_components : int
    log_likelihood : float
    covariance_type : str
    
    Returns:
    --------
    bic : float
    aic : float
    """
    n_samples, n_features = X.shape
    
    # Count parameters based on covariance type
    # Means: K * d
    # Weights: K - 1 (sum to 1)
    n_params = n_components * n_features + (n_components - 1)
    
    # Covariances:
    if covariance_type == 'full':
        # K * d * (d + 1) / 2
        n_params += n_components * n_features * (n_features + 1) // 2
    elif covariance_type == 'tied':
        # d * (d + 1) / 2
        n_params += n_features * (n_features + 1) // 2
    elif covariance_type == 'diagonal':
        # K * d
        n_params += n_components * n_features
    elif covariance_type == 'spherical':
        # K
        n_params += n_components
    
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    aic = -2 * log_likelihood + 2 * n_params
    
    return bic, aic


# ============================================================================
# EXTERNAL VALIDATION METRICS
# ============================================================================

def adjusted_rand_index(true_labels, pred_labels):
    """
    Compute Adjusted Rand Index.
    
    Measures similarity between two clusterings, adjusted for chance.
    Range: [-1, 1], where 1 is perfect agreement.
    
    Formula:
    ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
    
    Parameters:
    -----------
    true_labels : array, shape (n_samples,)
    pred_labels : array, shape (n_samples,)
    
    Returns:
    --------
    ari : float
    """
    n = len(true_labels)
    
    # Build contingency table
    true_unique = np.unique(true_labels)
    pred_unique = np.unique(pred_labels)
    
    contingency = np.zeros((len(true_unique), len(pred_unique)))
    
    for i, t in enumerate(true_unique):
        for j, p in enumerate(pred_unique):
            contingency[i, j] = np.sum((true_labels == t) & (pred_labels == p))
    
    # Sum over rows and columns
    sum_rows = np.sum(contingency, axis=1)
    sum_cols = np.sum(contingency, axis=0)
    
    # Compute combination sums
    sum_comb_c = np.sum([comb(n_ij, 2, exact=True) for n_ij in contingency.flatten() if n_ij >= 2])
    sum_comb_rows = np.sum([comb(n_i, 2, exact=True) for n_i in sum_rows if n_i >= 2])
    sum_comb_cols = np.sum([comb(n_j, 2, exact=True) for n_j in sum_cols if n_j >= 2])
    
    # Total combinations
    n_combinations = comb(n, 2, exact=True)
    
    # Expected index
    expected_index = (sum_comb_rows * sum_comb_cols) / n_combinations
    
    # Max index
    max_index = (sum_comb_rows + sum_comb_cols) / 2
    
    # Adjusted Rand Index
    if max_index - expected_index == 0:
        return 0.0
    
    ari = (sum_comb_c - expected_index) / (max_index - expected_index)
    
    return ari


def normalized_mutual_information(true_labels, pred_labels):
    """
    Compute Normalized Mutual Information.
    
    Measures mutual dependence between clusterings, normalized.
    Range: [0, 1], where 1 is perfect agreement.
    
    Formula:
    NMI = 2 * I(U;V) / [H(U) + H(V)]
    
    where:
    - I(U;V) = mutual information
    - H(U), H(V) = entropies
    
    Parameters:
    -----------
    true_labels : array, shape (n_samples,)
    pred_labels : array, shape (n_samples,)
    
    Returns:
    --------
    nmi : float
    """
    n = len(true_labels)
    
    # Build contingency table
    true_unique = np.unique(true_labels)
    pred_unique = np.unique(pred_labels)
    
    contingency = np.zeros((len(true_unique), len(pred_unique)))
    
    for i, t in enumerate(true_unique):
        for j, p in enumerate(pred_unique):
            contingency[i, j] = np.sum((true_labels == t) & (pred_labels == p))
    
    # Compute marginals
    sum_rows = np.sum(contingency, axis=1)
    sum_cols = np.sum(contingency, axis=0)
    
    # Compute mutual information
    mi = 0
    for i in range(len(true_unique)):
        for j in range(len(pred_unique)):
            if contingency[i, j] > 0:
                mi += contingency[i, j] * np.log(
                    (n * contingency[i, j]) / (sum_rows[i] * sum_cols[j])
                )
    mi /= n
    
    # Compute entropies
    entropy_true = -np.sum((sum_rows / n) * np.log(sum_rows / n + 1e-10))
    entropy_pred = -np.sum((sum_cols / n) * np.log(sum_cols / n + 1e-10))
    
    # Normalized mutual information
    if entropy_true + entropy_pred == 0:
        return 0.0
    
    nmi = 2 * mi / (entropy_true + entropy_pred)
    
    return nmi


def purity_score(true_labels, pred_labels):
    """
    Compute Purity score.
    
    Measures how "pure" each cluster is with respect to true labels.
    Range: [0, 1], where 1 is perfect purity.
    
    Formula:
    Purity = (1/n) * Σₖ max_j |Cₖ ∩ Tⱼ|
    
    Parameters:
    -----------
    true_labels : array, shape (n_samples,)
    pred_labels : array, shape (n_samples,)
    
    Returns:
    --------
    purity : float
    """
    n = len(true_labels)
    pred_unique = np.unique(pred_labels)
    
    total_correct = 0
    for cluster in pred_unique:
        # Get true labels for points in this cluster
        mask = pred_labels == cluster
        cluster_true_labels = true_labels[mask]
        
        # Find most common true label
        if len(cluster_true_labels) > 0:
            most_common_count = Counter(cluster_true_labels).most_common(1)[0][1]
            total_correct += most_common_count
    
    purity = total_correct / n
    return purity


def confusion_matrix(true_labels, pred_labels):
    """
    Compute confusion matrix for clustering.
    
    Parameters:
    -----------
    true_labels : array, shape (n_samples,)
    pred_labels : array, shape (n_samples,)
    
    Returns:
    --------
    cm : array, shape (n_true_classes, n_pred_clusters)
    true_classes : array
    pred_clusters : array
    """
    true_unique = np.unique(true_labels)
    pred_unique = np.unique(pred_labels)
    
    cm = np.zeros((len(true_unique), len(pred_unique)), dtype=int)
    
    for i, t in enumerate(true_unique):
        for j, p in enumerate(pred_unique):
            cm[i, j] = np.sum((true_labels == t) & (pred_labels == p))
    
    return cm, true_unique, pred_unique


# ============================================================================
# DIMENSIONALITY REDUCTION QUALITY METRICS
# ============================================================================

def reconstruction_error(X_original, X_reconstructed):
    """
    Compute Mean Squared Error between original and reconstructed data.
    
    MSE = (1/n) * Σᵢ ||xᵢ - x̂ᵢ||²
    
    Parameters:
    -----------
    X_original : array, shape (n_samples, n_features)
    X_reconstructed : array, shape (n_samples, n_features)
    
    Returns:
    --------
    mse : float
        Mean squared error
    rmse : float
        Root mean squared error
    """
    mse = np.mean((X_original - X_reconstructed)**2)
    rmse = np.sqrt(mse)
    return mse, rmse


def explained_variance_ratio(X, X_transformed, transform_matrix):
    """
    Compute explained variance ratio for PCA-like methods.
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
        Original centered data
    X_transformed : array, shape (n_samples, n_components)
        Transformed data
    transform_matrix : array, shape (n_features, n_components)
        Transformation matrix (e.g., principal components)
    
    Returns:
    --------
    explained_var_ratio : array, shape (n_components,)
    cumulative_var_ratio : array, shape (n_components,)
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Total variance
    total_var = np.sum(np.var(X_centered, axis=0))
    
    # Variance explained by each component
    n_components = X_transformed.shape[1]
    explained_var = np.zeros(n_components)
    
    for i in range(n_components):
        component = transform_matrix[:, i]
        projected = np.dot(X_centered, component)
        explained_var[i] = np.var(projected)
    
    explained_var_ratio = explained_var / total_var
    cumulative_var_ratio = np.cumsum(explained_var_ratio)
    
    return explained_var_ratio, cumulative_var_ratio


# ============================================================================
# ELBOW METHOD
# ============================================================================

def compute_elbow_curve(X, max_k=10, method='wcss'):
    """
    Compute elbow curve for determining optimal number of clusters.
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
    max_k : int
        Maximum number of clusters to test
    method : str
        'wcss' or 'silhouette'
    
    Returns:
    --------
    k_values : array
    scores : array
    optimal_k : int
    """
    from scipy.signal import argrelextrema
    
    k_values = np.arange(2, max_k + 1)
    scores = []
    
    print(f"Computing elbow curve for k = 2 to {max_k}...")
    
    for k in k_values:
        # Simple k-means for elbow method
        labels = simple_kmeans(X, k)
        
        if method == 'wcss':
            score = within_cluster_sum_of_squares(X, labels)
        elif method == 'silhouette':
            score, _ = silhouette_score(X, labels)
        
        scores.append(score)
        print(f"k={k}: {method}={score:.4f}")
    
    scores = np.array(scores)
    
    # Find optimal k using elbow method (maximum curvature)
    if method == 'wcss':
        # For WCSS, find point of maximum curvature (elbow)
        # Use second derivative
        if len(scores) >= 3:
            second_deriv = np.diff(scores, n=2)
            optimal_k = k_values[np.argmax(second_deriv) + 1]
        else:
            optimal_k = k_values[len(k_values) // 2]
    else:
        # For silhouette, just pick maximum
        optimal_k = k_values[np.argmax(scores)]
    
    return k_values, scores, optimal_k


def simple_kmeans(X, k, max_iter=100):
    """
    Simple k-means implementation for elbow method.
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
    k : int
        Number of clusters
    max_iter : int
    
    Returns:
    --------
    labels : array, shape (n_samples,)
    """
    n_samples = X.shape[0]
    
    # Initialize centers randomly
    indices = np.random.choice(n_samples, k, replace=False)
    centers = X[indices]
    
    for _ in range(max_iter):
        # Assign to nearest center
        distances = compute_euclidean_distance(X, centers)
        labels = np.argmin(distances, axis=1)
        
        # Update centers
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Check convergence
        if np.allclose(centers, new_centers):
            break
        
        centers = new_centers
    
    return labels


# ============================================================================
# STATISTICAL COMPARISON
# ============================================================================

def create_comparison_table(results_dict, metrics):
    """
    Create comparison table across all experiments.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with experiment names as keys and metric dictionaries as values
    metrics : list
        List of metric names to include
    
    Returns:
    --------
    table : array, shape (n_experiments, n_metrics)
    experiment_names : list
    """
    experiment_names = list(results_dict.keys())
    n_experiments = len(experiment_names)
    n_metrics = len(metrics)
    
    table = np.zeros((n_experiments, n_metrics))
    
    for i, exp_name in enumerate(experiment_names):
        for j, metric in enumerate(metrics):
            if metric in results_dict[exp_name]:
                table[i, j] = results_dict[exp_name][metric]
            else:
                table[i, j] = np.nan
    
    return table, experiment_names


def paired_ttest(scores1, scores2):
    """
    Perform paired t-test between two sets of scores.
    
    H0: mean(scores1) = mean(scores2)
    
    Parameters:
    -----------
    scores1 : array
    scores2 : array
    
    Returns:
    --------
    t_statistic : float
    p_value : float
    significant : bool (at α=0.05)
    """
    n = len(scores1)
    diff = scores1 - scores2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    
    # t-statistic
    t_stat = mean_diff / (std_diff / np.sqrt(n))
    
    # Two-tailed p-value (approximation using normal distribution for large n)
    from scipy.stats import t as t_dist
    p_value = 2 * (1 - t_dist.cdf(np.abs(t_stat), df=n-1))
    
    significant = p_value < 0.05
    
    return t_stat, p_value, significant


def compute_time_complexity(n_samples, n_features, n_clusters, algorithm='kmeans'):
    """
    Theoretical time complexity for different algorithms.
    
    Returns:
    --------
    complexity_str : str
        Big-O notation
    """
    complexities = {
        'kmeans': f'O({n_clusters} * {n_samples} * {n_features} * iterations)',
        'gmm': f'O({n_clusters} * {n_samples} * {n_features}² * iterations)',
        'pca': f'O(min({n_samples}², {n_features}³))',
        'autoencoder': f'O({n_samples} * {n_features} * hidden_dim * epochs)'
    }
    
    return complexities.get(algorithm, 'Unknown')


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_2d_projection(X_2d, labels, title='2D Projection', true_labels=None, 
                       centers=None, save_path=None):
    """
    Plot 2D projection of data with cluster assignments.
    
    Parameters:
    -----------
    X_2d : array, shape (n_samples, 2)
    labels : array, shape (n_samples,)
        Predicted cluster labels
    title : str
    true_labels : array or None
        True labels for comparison
    centers : array or None, shape (n_clusters, 2)
        Cluster centers
    save_path : str or None
    """
    fig, axes = plt.subplots(1, 2 if true_labels is not None else 1, 
                             figsize=(15 if true_labels is not None else 8, 6))
    
    if true_labels is None:
        axes = [axes]
    
    # Plot predicted labels
    scatter = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=labels, 
                             cmap='tab10', alpha=0.6, s=30)
    if centers is not None:
        axes[0].scatter(centers[:, 0], centers[:, 1], c='red', 
                       marker='X', s=200, edgecolors='black', linewidths=2,
                       label='Centers')
        axes[0].legend()
    
    axes[0].set_title(f'{title} - Predicted Clusters')
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')
    plt.colorbar(scatter, ax=axes[0])
    
    # Plot true labels if available
    if true_labels is not None:
        scatter = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=true_labels,
                                 cmap='tab10', alpha=0.6, s=30)
        axes[1].set_title(f'{title} - True Labels')
        axes[1].set_xlabel('Component 1')
        axes[1].set_ylabel('Component 2')
        plt.colorbar(scatter, ax=axes[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_elbow_curve(k_values, scores, optimal_k, metric_name='WCSS', save_path=None):
    """
    Plot elbow curve with marked optimal k.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, scores, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2,
                label=f'Optimal k = {optimal_k}')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'Elbow Method - {metric_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_bic_aic_curves(k_values, bic_scores, aic_scores, save_path=None):
    """
    Plot BIC and AIC curves for GMM.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, bic_scores, 'bo-', label='BIC', linewidth=2, markersize=8)
    plt.plot(k_values, aic_scores, 'rs-', label='AIC', linewidth=2, markersize=8)
    
    optimal_k_bic = k_values[np.argmin(bic_scores)]
    optimal_k_aic = k_values[np.argmin(aic_scores)]
    
    plt.axvline(x=optimal_k_bic, color='b', linestyle='--', alpha=0.5)
    plt.axvline(x=optimal_k_aic, color='r', linestyle='--', alpha=0.5)
    
    plt.xlabel('Number of Components (k)', fontsize=12)
    plt.ylabel('Information Criterion', fontsize=12)
    plt.title('GMM Model Selection - BIC & AIC', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)