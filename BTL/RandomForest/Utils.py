import numpy as np
import math


def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


def _entropy(y):
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy


def _information_gain(X, y, threshold):
    parent_loss = _entropy(y)
    left_idx, right_idx = np.argwhere(X <= threshold).flatten(), np.argwhere(X > threshold).flatten()
    n, n_left, n_right = len(y), len(left_idx), len(right_idx)
    if n_left == 0 or n_right == 0:
        return 0
    child_loss = (n_left / n) * _entropy(y[left_idx]) + (n_right / n) * _entropy(y[right_idx])

    return parent_loss - child_loss


def _most_common_label(y):
    most_common = None
    max_count = 0
    for label in np.unique(y):
        count = len(y[y == label])
        if count > max_count:
            most_common = label
            max_count = count
    return most_common


def _random_samples(X, y):
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, n_samples, replace=True)
    return X[indices], y[indices]
