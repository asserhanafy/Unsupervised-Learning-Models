import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        Xs = (X - self.mean_) / self.std_

        cov = np.cov(Xs, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)

        idx = np.argsort(eigvals)[::-1]
        self.eigvals_ = eigvals[idx]
        self.eigvecs_ = eigvecs[:, idx]

        self.components_ = self.eigvecs_[:, :self.n_components]
        self.explained_variance_ratio_ = (
            self.eigvals_ / np.sum(self.eigvals_)
        )

    def transform(self, X):
        Xs = (X - self.mean_) / self.std_
        return Xs @ self.components_

    def inverse_transform(self, Z):
        Xs_rec = Z @ self.components_.T
        return Xs_rec * self.std_ + self.mean_

    def reconstruction_error(self, X):
        Z = self.transform(X)
        X_rec = self.inverse_transform(Z)
        return np.mean((X - X_rec) ** 2)
