import numpy as np
import warnings
from pandas import DataFrame, Series
from RandomForest.Utils import _information_gain, _most_common_label

warnings.filterwarnings("ignore")


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=100, max_feature=15):
        self.max_depth = max_depth
        self.max_feature = max_feature
        self.root = None

    def _is_finished(self, depth):
        if depth >= self.max_depth or self.n_class_labels == 1:
            return True
        return False

    def _best_split(self, X, y, features):
        split = {'info_gain': - 1, 'feature': None, 'threshold': None}

        for feature in features:
            X_feature = X[:, feature]
            thresholds = np.unique(X_feature)
            for threshold in thresholds:
                info_gain = _information_gain(X_feature, y, threshold)

                if info_gain > split['info_gain']:
                    split['info_gain'] = info_gain
                    split['feature'] = feature
                    split['threshold'] = threshold

        return split['feature'], split['threshold']

    def _build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        if self._is_finished(depth):
            most_common_Label = _most_common_label(y)
            return Node(value=most_common_Label)

        random_features = np.random.choice(self.n_features, self.max_feature, replace=False)
        best_feature, best_threshold = self._best_split(X, y, random_features)

        left_idx, right_idx = np.argwhere(X[:, best_feature] <= best_threshold).flatten(), np.argwhere(
            X[:, best_feature] > best_threshold).flatten()
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feature, best_threshold, left_child, right_child)

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def fit(self, X, y):
        if isinstance(X, DataFrame):
            X = X.values
        if isinstance(y, Series):
            y = y.values
        self.root = self._build_tree(X, y)

    def predict(self, X):
        if isinstance(X, DataFrame):
            X = X.values
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)
