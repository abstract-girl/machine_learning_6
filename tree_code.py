import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    sort_idx = np.argsort(feature_vector)
    feature_vector, target_vector = feature_vector[sort_idx], target_vector[sort_idx]

    unique_thresholds = np.unique(feature_vector[:-1] + np.diff(feature_vector) / 2)

    best_threshold = None
    best_gini = float('inf')
    thresholds = []
    ginis = []

    for threshold in unique_thresholds:
        left = target_vector[feature_vector < threshold]
        right = target_vector[feature_vector >= threshold]

        if len(left) == 0 or len(right) == 0:
            continue

        gini_left = 1 - sum((np.count_nonzero(left == k) / len(left))**2 for k in np.unique(left))
        gini_right = 1 - sum((np.count_nonzero(right == k) / len(right))**2 for k in np.unique(right))

        weighted_gini = (len(left) / len(target_vector)) * gini_left + (len(right) / len(target_vector)) * gini_right

        thresholds.append(threshold)
        ginis.append(weighted_gini)

        if weighted_gini < best_gini:
            best_gini = weighted_gini
            best_threshold = threshold

    return np.array(thresholds), np.array(ginis), best_threshold, best_gini


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if any(ft not in {"real", "categorical"} for ft in feature_types):
            raise ValueError("There is unknown feature type")
        
        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth):
        if np.all(sub_y == sub_y[0]) or len(sub_y) < self._min_samples_split or depth == self._max_depth:
            node['type'] = 'terminal'
            node['class'] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best = None, None, float('inf')
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            unique_values = np.unique(sub_X[:, feature])

            for value in unique_values:
                if feature_type == 'categorical':
                    split = sub_X[:, feature] == value
                else:
                    split = sub_X[:, feature] < value

                if np.sum(split) < self._min_samples_leaf or np.sum(~split) < self._min_samples_leaf:
                    continue

                gini = self._gini(sub_y[split]) * (np.sum(split) / len(sub_X)) + self._gini(sub_y[~split]) * (np.sum(~split) / len(sub_X))
                if gini < gini_best:
                    feature_best = feature
                    gini_best = gini
                    threshold_best = value

        if feature_best is not None:
            node['type'] = 'nonterminal'
            node['feature_split'] = feature_best
            node['threshold'] = threshold_best
            node['left_child'], node['right_child'] = {}, {}
            self._fit_node(sub_X[sub_X[:, feature_best] == threshold_best], sub_y[sub_X[:, feature_best] == threshold_best], node['left_child'], depth + 1)
            self._fit_node(sub_X[sub_X[:, feature_best] != threshold_best], sub_y[sub_X[:, feature_best] != threshold_best], node['right_child'], depth + 1)

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1 - np.sum(probs**2)

    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['class']

        feature_value = x[node['feature_split']]
        if self._feature_types[node['feature_split']] == 'categorical':
            if feature_value == node['threshold']:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])
        else:
            if feature_value < node['threshold']:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])

    def fit(self, X, y):
        self._tree = {}
        self._fit_node(np.array(X), np.array(y), self._tree, 0)
        return self

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in np.array(X)])

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def get_params(self, deep=True):
        return {
            'feature_types': self._feature_types,
            'max_depth': self._max_depth,
            'min_samples_split': self._min_samples_split,
            'min_samples_leaf': self._min_samples_leaf
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
