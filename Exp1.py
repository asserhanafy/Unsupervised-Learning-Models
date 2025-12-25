import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from Kmeans import KMeans
import part3


# ----------------------------
# Data loading and prep
# ----------------------------
def load_dataset():
	"""Load the breast cancer dataset from sklearn and return (X, y)."""
	data = load_breast_cancer()
	return data.data, data.target


# ----------------------------
# Gap statistic
# ----------------------------
def _within_cluster_ss(X, labels, centroids):
	"""Sum of squared distances of points to their assigned centroids."""
	ss = 0.0
	for i in range(centroids.shape[0]):
		pts = X[labels == i]
		if pts.size == 0:
			continue
		diff = pts - centroids[i]
		ss += np.sum(diff * diff)
	return ss


def gap_statistic(X, k_max=10, B=10, init="kmeans++", max_iter=300, tol=1e-4, random_state=42):
	"""Compute gap statistic and its standard error for k = 1..k_max."""
	rng = np.random.default_rng(random_state)
	n_samples, n_features = X.shape
	mins = X.min(axis=0)
	maxs = X.max(axis=0)

	Wks = np.zeros(k_max)
	Wkbs = np.zeros((k_max, B))

	for k in range(1, k_max + 1):
		km = KMeans(k=k, init=init, max_iter=max_iter, tol=tol)
		rng_bit = rng.integers(0, 2**32 - 1)
		np.random.seed(int(rng_bit))
		km.fit(X)
		Wks[k - 1] = _within_cluster_ss(X, km.labels_, km.centroids_)

		for b in range(B):
			Xb = rng.uniform(mins, maxs, size=(n_samples, n_features))
			km_b = KMeans(k=k, init=init, max_iter=max_iter, tol=tol)
			rng_bit_b = rng.integers(0, 2**32 - 1)
			np.random.seed(int(rng_bit_b))
			km_b.fit(Xb)
			Wkbs[k - 1, b] = _within_cluster_ss(Xb, km_b.labels_, km_b.centroids_)

	logWks = np.log(Wks)
	logWkbs = np.log(Wkbs)
	gaps = np.mean(logWkbs, axis=1) - logWks
	sk = np.sqrt(1 + 1.0 / B) * np.std(logWkbs, axis=1)
	return gaps, sk


def select_gap_k(gaps, sk):
	"""Tibshirani rule: smallest k s.t. Gap(k) >= Gap(k+1) - s_{k+1}."""
	k_vals = np.arange(1, len(gaps) + 1)
	for i in range(len(gaps) - 1):
		if gaps[i] >= gaps[i + 1] - sk[i + 1]:
			return k_vals[i]
	return k_vals[-1]


# ----------------------------
# Experiment helpers
# ----------------------------
def run_kmeans_series(X, y_true, k_values, init_method, tol=1e-4, max_iter=300, random_state=42):
	"""Run KMeans for a list of k and collect metrics."""
	rng = np.random.default_rng(random_state)
	records = {
		"inertia": [],
		"silhouette": [],
		"davies_bouldin": [],
		"calinski_harabasz": [],
		"wcss": [],
		"ari": [],
		"nmi": [],
		"purity": [],
		"time": [],
		"iterations": [],
		"labels": []
	}

	for k in k_values:
		seed = int(rng.integers(0, 2**32 - 1))
		np.random.seed(seed)

		km = KMeans(k=k, init=init_method, max_iter=max_iter, tol=tol)
		start = time.time()
		km.fit(X)
		elapsed = time.time() - start
		labels = km.labels_

		records["inertia"].append(km.inertia_)
		records["time"].append(elapsed)
		records["iterations"].append(km.iterations_)

		if k > 1:
			sil, _ = part3.silhouette_score(X, labels)
			dbi = part3.davies_bouldin_index(X, labels)
			chi = part3.calinski_harabasz_index(X, labels)
		else:
			sil, dbi, chi = np.nan, np.nan, np.nan

		records["silhouette"].append(sil)
		records["davies_bouldin"].append(dbi)
		records["calinski_harabasz"].append(chi)
		records["wcss"].append(part3.within_cluster_sum_of_squares(X, labels))
		records["labels"].append(labels)

		# External metrics (requires ground truth)
		records["ari"].append(part3.adjusted_rand_index(y_true, labels))
		records["nmi"].append(part3.normalized_mutual_information(y_true, labels))
		records["purity"].append(part3.purity_score(y_true, labels))

	return records


def pick_elbow_k(k_values, inertia_values):
	"""Heuristic elbow: k with max second derivative of inertia curve."""
	inertia_arr = np.array(inertia_values)
	if len(inertia_arr) < 3:
		return k_values[len(k_values) // 2]
	second = np.diff(inertia_arr, n=2)
	idx = np.argmax(second) + 1
	return k_values[idx]


# ----------------------------
# Plotting
# ----------------------------
def plot_metric_curves(k_values, results_by_init):
	plt.figure(figsize=(14, 6))
	plt.subplot(1, 2, 1)
	for init_method, res in results_by_init.items():
		plt.plot(k_values, res["inertia"], marker="o", label=init_method)
	plt.xlabel("k")
	plt.ylabel("Inertia (WCSS)")
	plt.title("Elbow (Inertia)")
	plt.grid(True, alpha=0.3)
	plt.legend()

	plt.subplot(1, 2, 2)
	for init_method, res in results_by_init.items():
		plt.plot(k_values, res["silhouette"], marker="o", label=init_method)
	plt.xlabel("k")
	plt.ylabel("Silhouette")
	plt.title("Silhouette Analysis")
	plt.grid(True, alpha=0.3)
	plt.legend()

	plt.tight_layout()
	plt.show()


def plot_internal_metrics(k_values, results_by_init):
	plt.figure(figsize=(14, 6))
	plt.subplot(1, 2, 1)
	for init_method, res in results_by_init.items():
		plt.plot(k_values, res["davies_bouldin"], marker="o", label=init_method)
	plt.xlabel("k")
	plt.ylabel("Davies-Bouldin (lower better)")
	plt.title("Davies-Bouldin vs k")
	plt.grid(True, alpha=0.3)
	plt.legend()

	plt.subplot(1, 2, 2)
	for init_method, res in results_by_init.items():
		plt.plot(k_values, res["calinski_harabasz"], marker="o", label=init_method)
	plt.xlabel("k")
	plt.ylabel("Calinski-Harabasz (higher better)")
	plt.title("Calinski-Harabasz vs k")
	plt.grid(True, alpha=0.3)
	plt.legend()

	plt.tight_layout()
	plt.show()


def plot_convergence(k_values, results_by_init):
	plt.figure(figsize=(14, 6))
	plt.subplot(1, 2, 1)
	for init_method, res in results_by_init.items():
		plt.plot(k_values, res["iterations"], marker="o", label=init_method)
	plt.xlabel("k")
	plt.ylabel("Iterations")
	plt.title("Convergence speed (iterations)")
	plt.grid(True, alpha=0.3)
	plt.legend()

	plt.subplot(1, 2, 2)
	for init_method, res in results_by_init.items():
		plt.plot(k_values, res["time"], marker="o", label=init_method)
	plt.xlabel("k")
	plt.ylabel("Time (s)")
	plt.title("Convergence speed (time)")
	plt.grid(True, alpha=0.3)
	plt.legend()

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


def plot_gap(k_values, gaps, sk, optimal_k):
	plt.figure(figsize=(8, 5))
	plt.errorbar(k_values, gaps, yerr=sk, fmt="o-", capsize=5, label="Gap")
	plt.axvline(optimal_k, color="r", linestyle="--", label=f"Optimal k={optimal_k}")
	plt.xlabel("k")
	plt.ylabel("Gap statistic")
	plt.title("Gap Statistic")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.show()


# ----------------------------
# Main experiment
# ----------------------------
def main():
	X, y_true = load_dataset()
	X = StandardScaler().fit_transform(X)

	k_values = list(range(2, 11))
	results = {}

	for init_method in ["kmeans++", "random"]:
		res = run_kmeans_series(X, y_true, k_values, init_method)
		results[init_method] = res
		elbow_k = pick_elbow_k(k_values, res["inertia"])
		best_sil_k = k_values[int(np.nanargmax(res["silhouette"]))]
		print(f"Init={init_method}: elbow k={elbow_k}, silhouette-opt k={best_sil_k}")

		# Best-by-metric (from part3.py): max for all except WCSS and DBI (min)
		idx_ari_exp1 = int(np.nanargmax(res["ari"]))
		best_ari_k_exp1 = k_values[idx_ari_exp1]
		best_ari_value_exp1 = res["ari"][idx_ari_exp1]

		idx_nmi_exp1 = int(np.nanargmax(res["nmi"]))
		best_nmi_k_exp1 = k_values[idx_nmi_exp1]
		best_nmi_value_exp1 = res["nmi"][idx_nmi_exp1]

		idx_purity_exp1 = int(np.nanargmax(res["purity"]))
		best_purity_k_exp1 = k_values[idx_purity_exp1]
		best_purity_value_exp1 = res["purity"][idx_purity_exp1]

		idx_sil_exp1 = int(np.nanargmax(res["silhouette"]))
		best_sil_k_exp1 = k_values[idx_sil_exp1]
		best_sil_value_exp1 = res["silhouette"][idx_sil_exp1]

		idx_ch_exp1 = int(np.nanargmax(res["calinski_harabasz"]))
		best_ch_k_exp1 = k_values[idx_ch_exp1]
		best_ch_value_exp1 = res["calinski_harabasz"][idx_ch_exp1]

		idx_wcss_exp1 = int(np.nanargmin(res["wcss"]))
		best_wcss_k_exp1 = k_values[idx_wcss_exp1]
		best_wcss_value_exp1 = res["wcss"][idx_wcss_exp1]

		idx_dbi_exp1 = int(np.nanargmin(res["davies_bouldin"]))
		best_dbi_k_exp1 = k_values[idx_dbi_exp1]
		best_dbi_value_exp1 = res["davies_bouldin"][idx_dbi_exp1]

		print(
			f"Init={init_method} best-by-metric: "
			f"ARI k={best_ari_k_exp1} v={best_ari_value_exp1:.3f}; "
			f"NMI k={best_nmi_k_exp1} v={best_nmi_value_exp1:.3f}; "
			f"Purity k={best_purity_k_exp1} v={best_purity_value_exp1:.3f}; "
			f"Silhouette k={best_sil_k_exp1} v={best_sil_value_exp1:.3f}; "
			f"CH k={best_ch_k_exp1} v={best_ch_value_exp1:.3f}; "
			f"WCSS k={best_wcss_k_exp1} v={best_wcss_value_exp1:.1f}; "
			f"DBI k={best_dbi_k_exp1} v={best_dbi_value_exp1:.3f}"
		)

	plot_metric_curves(k_values, results)
	plot_internal_metrics(k_values, results)
	plot_convergence(k_values, results)

	print("Computing gap statistic (kmeans++ init)...")
	gaps, sk = gap_statistic(X, k_max=max(k_values), init="kmeans++", B=10, random_state=42)
	gap_k = select_gap_k(gaps, sk)
	plot_gap(np.arange(1, len(gaps) + 1), gaps, sk, gap_k)
	print(f"Gap statistic optimal k={gap_k}")

	# External metrics snapshot at k=2 for reporting
	k_report = 2
	for init_method, res in results.items():
		idx = k_values.index(k_report)
		print(
			f"Init={init_method}, k={k_report}: "
			f"ARI={res['ari'][idx]:.3f}, NMI={res['nmi'][idx]:.3f}, Purity={res['purity'][idx]:.3f}, "
			f"WCSS={res['wcss'][idx]:.1f}, Silhouette={res['silhouette'][idx]:.3f}"
		)

		cm, true_classes, pred_clusters = part3.confusion_matrix(y_true, res["labels"][idx])
		title = f"Confusion Matrix (init={init_method}, k={k_report})"
		plot_confusion(cm, true_classes, pred_clusters, title)


if __name__ == "__main__":
	main()
