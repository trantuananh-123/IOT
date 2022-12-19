import numpy as np
from pandas import DataFrame, Series


class SVM:
    def __init__(self, C=10, lr=0.001, epochs=1000):
        self._support_vectors = None
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.beta = None
        self.b = None
        self.X = None
        self.y = None

        self.n = 0

        self.d = 0

    def _constrain(self, X, y):
        return y * X.dot(self.beta) + self.b

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n, self.d = X.shape
        self.beta = np.random.randn(self.d)
        self.b = 0

        if isinstance(X, DataFrame):
            X = X.values
        if isinstance(y, Series):
            y = y.values

        for _ in range(self.epochs):
            print(_)
            margin = self._constrain(X, y)

            misclassified_pts_idx = np.where(margin < 1)[0]
            d_beta = self.beta - self.C * \
                     y[misclassified_pts_idx].dot(X[misclassified_pts_idx])
            self.beta = self.beta - self.lr * d_beta

            d_b = - self.C * np.sum(y[misclassified_pts_idx])
            self.b = self.b - self.lr * d_b
        return self

    def predict(self, X):
        if isinstance(X, DataFrame):
            X = X.values
        return np.sign(X.dot(self.beta) + self.b)
