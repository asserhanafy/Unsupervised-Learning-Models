import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from Kmeans import KMeans
from PCA import PCA
import part3


# ----------------------------
# Experiment runner
# ----------------------------
def run_pca_kmeans(
	X,
	y_true,
	n_components_list,
	k=2,
	init="kmeans++",
	max_iter=300,
	tol=1e-4,
):
	"""Run PCA -> KMeans for each n_components and collect metrics."""
	results = {
		"n_components": [],
		"reconstruction_error": [],
		"explained_variance": [],
		"silhouette": [],
		"davies_bouldin": [],
		"calinski_harabasz": [],
		"wcss": [],
		"ari": [],
		"nmi": [],
		"purity": [],
		"labels": [],
	}

	for n_comp in n_components_list:
		pca = PCA(n_components=n_comp)
		pca.fit(X)
		X_pca = pca.transform(X)

		rec_err = pca.reconstruction_error(X)
		explained = np.sum(pca.explained_variance_ratio_[:n_comp])

		km = KMeans(k=k, init=init, max_iter=max_iter, tol=tol)
		km.fit(X_pca)
		labels = km.labels_

		sil, _ = part3.silhouette_score(X_pca, labels) if k > 1 else (np.nan, None)
		dbi = part3.davies_bouldin_index(X_pca, labels) if k > 1 else np.nan
		chi = part3.calinski_harabasz_index(X_pca, labels) if k > 1 else np.nan
		wcss = part3.within_cluster_sum_of_squares(X_pca, labels)

		ari = part3.adjusted_rand_index(y_true, labels)
		nmi = part3.normalized_mutual_information(y_true, labels)
		purity = part3.purity_score(y_true, labels)

		results["n_components"].append(n_comp)
		results["reconstruction_error"].append(rec_err)
		results["explained_variance"].append(explained)
		results["silhouette"].append(sil)
		results["davies_bouldin"].append(dbi)
		results["calinski_harabasz"].append(chi)
		results["wcss"].append(wcss)
		results["ari"].append(ari)
		results["nmi"].append(nmi)
		results["purity"].append(purity)
		results["labels"].append(labels)

	return results


# ----------------------------
# Plot helpers
# ----------------------------
def plot_reconstruction_and_variance(n_components, rec_errors, explained):
	plt.figure(figsize=(12, 5))

	plt.subplot(1, 2, 1)
	plt.plot(n_components, rec_errors, marker="o", color="red")
	plt.xlabel("Number of PCA Components")
	plt.ylabel("Reconstruction Error (MSE)")
	plt.title("Reconstruction Error vs PCA Components")
	plt.grid(True, alpha=0.3)

	plt.subplot(1, 2, 2)
	plt.plot(n_components, explained, marker="o", color="blue")
	plt.xlabel("Number of PCA Components")
	plt.ylabel("Explained Variance (cumulative)")
	plt.title("Explained Variance vs PCA Components")
	plt.grid(True, alpha=0.3)

	plt.tight_layout()
	plt.show()


def plot_clustering_metrics(n_components, results):
	plt.figure(figsize=(14, 12))

	plt.subplot(3, 2, 1)
	plt.plot(n_components, results["silhouette"], marker="o")
	plt.xlabel("PCA Components")
	plt.ylabel("Silhouette")
	plt.title("Silhouette vs PCA components")
	plt.grid(True, alpha=0.3)

	plt.subplot(3, 2, 2)
	plt.plot(n_components, results["davies_bouldin"], marker="o", color="orange")
	plt.xlabel("PCA Components")
	plt.ylabel("Davies-Bouldin (lower better)")
	plt.title("Davies-Bouldin vs PCA components")
	plt.grid(True, alpha=0.3)

	plt.subplot(3, 2, 3)
	plt.plot(n_components, results["calinski_harabasz"], marker="o", color="green")
	plt.xlabel("PCA Components")
	plt.ylabel("Calinski-Harabasz (higher better)")
	plt.title("Calinski-Harabasz vs PCA components")
	plt.grid(True, alpha=0.3)

	plt.subplot(3, 2, 4)
	plt.plot(n_components, results["wcss"], marker="o", color="red")
	plt.xlabel("PCA Components")
	plt.ylabel("WCSS")
	plt.title("WCSS vs PCA components")
	plt.grid(True, alpha=0.3)

	plt.subplot(3, 2, 5)
	plt.plot(n_components, results["ari"], marker="o", color="blue")
	plt.xlabel("PCA Components")
	plt.ylabel("Adjusted Rand Index")
	plt.title("ARI vs PCA components")
	plt.grid(True, alpha=0.3)

	plt.subplot(3, 2, 6)
	plt.plot(n_components, results["purity"], marker="o", color="brown")
	plt.xlabel("PCA Components")
	plt.ylabel("Purity")
	plt.title("Purity vs PCA components")
	plt.grid(True, alpha=0.3)

	plt.tight_layout()
	plt.show()


def plot_confusion(cm, true_classes, pred_clusters, title):
	plt.figure(figsize=(6, 5))
	sns.heatmap(
		cm,
		annot=True,
		fmt="d",
		cmap="Blues",
		xticklabels=[f"Cluster {c}" for c in pred_clusters],
		yticklabels=[f"True {c}" for c in true_classes],
	)
	plt.title(title)
	plt.xlabel("Predicted Clusters")
	plt.ylabel("True Classes")
	plt.tight_layout()
	plt.show()


def plot_2d_projection(X_2d, labels, true_labels=None):
	plt.figure(figsize=(8, 6))
	sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=labels, palette="tab10", s=40, alpha=0.7)
	plt.title("K-Means on 2D PCA projection")
	plt.xlabel("PC1")
	plt.ylabel("PC2")
	if true_labels is not None:
		plt.legend(title="Cluster")
	plt.tight_layout()
	plt.show()


# ----------------------------
# Main experiment
# ----------------------------
def main():
	data = load_breast_cancer()
	X = data.data
	y_true = data.target

	X = StandardScaler().fit_transform(X)
	n_components_list = [2, 5, 10, 15, 20]

	results = run_pca_kmeans(X, y_true, n_components_list, k=2, init="kmeans++")

	# Identify best silhouette configuration for confusion matrix
	best_idx = int(np.nanargmax(results["silhouette"]))
	best_n = results["n_components"][best_idx]
	best_labels = results["labels"][best_idx]
	cm, true_classes, pred_clusters = part3.confusion_matrix(y_true, best_labels)

	print(f"Best silhouette at n_components={best_n}: score={results['silhouette'][best_idx]:.3f}")
	print(f"Reconstruction error={results['reconstruction_error'][best_idx]:.4f}, "
		  f"Explained variance={results['explained_variance'][best_idx]:.3f}")

	plot_reconstruction_and_variance(
		results["n_components"],
		results["reconstruction_error"],
		results["explained_variance"],
	)

	plot_clustering_metrics(results["n_components"], results)

	title = f"Confusion Matrix (best silhouette, n_components={best_n})"
	plot_confusion(cm, true_classes, pred_clusters, title)

	# 2D visualization for n_components = 2
	if 2 in n_components_list:
		idx2 = n_components_list.index(2)
		pca2 = PCA(n_components=2)
		pca2.fit(X)
		X2 = pca2.transform(X)
		km2 = KMeans(k=2, init="kmeans++", max_iter=300, tol=1e-4)
		km2.fit(X2)
		plot_2d_projection(X2, km2.labels_, y_true)


if __name__ == "__main__":
	main()
