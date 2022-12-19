import numpy as np
import warnings
from RandomForest.DecisionTree import DecisionTree
from collections import Counter
from pandas import DataFrame, Series
from RandomForest.Utils import _random_samples, _most_common_label

warnings.filterwarnings("ignore")


class RandomForest:
    def __init__(self, n_trees=100, max_depth=150, max_feature=15):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_feature = max_feature
        self.trees = []
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.trees = []
        for _ in range(self.n_trees):
            print(_)
            tree = DecisionTree(max_depth=self.max_depth, max_feature=self.max_feature)
            if isinstance(X, DataFrame):
                X = X.values
            if isinstance(y, Series):
                y = y.values
            X_sample, y_sample = _random_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([_most_common_label(pred) for pred in tree_preds])
        return predictions
