import numpy as np

class KMeans:
    def __init__(self, k, max_iter=300, tol=1e-4, init="kmeans++"):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.iterations_ = 0

    def _init_random(self, X):
        idx = np.random.choice(len(X), self.k, replace=False)
        return X[idx]

    def _init_kmeanspp(self, X):
        centroids = [X[np.random.randint(len(X))]]
        for _ in range(1, self.k):
            dist = np.min(
                np.linalg.norm(X[:, None] - np.array(centroids), axis=2) ** 2,
                axis=1
            )
            total = np.sum(dist)
            if total == 0:
                # All points coincide with existing centroid(s); fall back to uniform choice.
                probs = np.full(len(X), 1 / len(X))
            else:
                probs = dist / total

            # If any numerical issues remain (NaN), revert to uniform.
            if not np.all(np.isfinite(probs)):
                probs = np.full(len(X), 1 / len(X))

            centroids.append(X[np.random.choice(len(X), p=probs)])
        return np.array(centroids)

    def fit(self, X):
        self.centroids_ = (
            self._init_kmeanspp(X) if self.init == "kmeans++"
            else self._init_random(X)
        )

        self.inertia_history_ = []

        for _ in range(self.max_iter):
            dist = np.linalg.norm(X[:, None] - self.centroids_, axis=2)
            labels = np.argmin(dist, axis=1)

            new_centroids = np.array([
                X[labels == i].mean(axis=0) for i in range(self.k)
            ])

            inertia = np.sum((X - new_centroids[labels]) ** 2)
            self.inertia_history_.append(inertia)

            if np.linalg.norm(new_centroids - self.centroids_) < self.tol:
                break

            self.centroids_ = new_centroids
            self.iterations_ += 1

        self.labels_ = labels
        self.inertia_ = self.inertia_history_[-1]
