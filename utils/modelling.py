import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#Grid Search for Scratch Models
def grid_search_custom_model(model_class, param_grid, X_train, y_train, cv=5):
    best_params = None
    best_score = float('inf')  # Lower is better (e.g., MSE)
    all_results = []

    # Iterate over parameter combinations
    for params in ParameterGrid(param_grid):
        cv_scores = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            # Train the model
            model = model_class(**params)
            model.fit(X_train_fold, y_train_fold)

            # Predict and evaluate
            y_pred = model.predict(X_val_fold)
            cv_scores.append(mean_squared_error(y_val_fold, y_pred))

        avg_score = np.mean(cv_scores)
        all_results.append((params, avg_score))
        if avg_score < best_score:
            best_score = avg_score
            best_params = params

    return best_params, best_score, all_results




class RegressionTreeScratch:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _build_tree(self, X, y, depth=0):
        num_samples = len(y)
        if (self.max_depth is not None and depth >= self.max_depth) or num_samples < self.min_samples_split:
            leaf_value = np.mean(y)
            return leaf_value

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return np.mean(y)

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {'feature_index': best_feature, 'threshold': best_threshold,
                'left': left_subtree, 'right': right_subtree}

    def _best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_variance_reduction = -float('inf')
        n_samples, n_features = X.shape
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold
                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue
                variance_reduction = self._variance_reduction(y, y[left_indices], y[right_indices])
                if variance_reduction > best_variance_reduction:
                    best_variance_reduction = variance_reduction
                    best_feature = feature_index
                    best_threshold = threshold
        return best_feature, best_threshold

    def _variance_reduction(self, parent, left, right):
        weight_left = len(left) / len(parent)
        weight_right = len(right) / len(parent)
        reduction = np.var(parent) - (weight_left * np.var(left) + weight_right * np.var(right))
        return reduction

    def _predict(self, inputs):
        node = self.tree
        while isinstance(node, dict):
            if inputs[node['feature_index']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node


class RandomForestRegressorScratch:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.feature_indices = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.trees = []
        self.feature_indices = []
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            if self.max_features is None:
                max_features = n_features
            elif self.max_features == 'sqrt':
                max_features = int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                max_features = int(np.log2(n_features))
            else:
                max_features = self.max_features
            features = np.random.choice(n_features, max_features, replace=False)
            self.feature_indices.append(features)
            X_sample = X_sample[:, features]
            tree = RegressionTreeScratch(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = []
        for tree, features in zip(self.trees, self.feature_indices):
            X_subset = X[:, features]
            predictions.append(tree.predict(X_subset))
        predictions = np.mean(predictions, axis=0)
        return predictions


class RegressionTreeLLM:
    """
    A regression tree for predicting continuous values using variance reduction.
    """

    def __init__(self, max_depth=None, min_samples_split=2):
        """
        Initializes the regression tree with hyperparameters.

        Parameters:
        - max_depth (int): Maximum depth of the tree.
        - min_samples_split (int): Minimum number of samples required to split an internal node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        """
        Builds the tree by fitting the model to the data.

        Parameters:
        - X (array-like): Feature matrix.
        - y (array-like): Target vector.
        """
        self.root = self._build_tree(X, y)

    def predict(self, X):
        """
        Predicts target values for given input data.

        Parameters:
        - X (array-like): Feature matrix.

        Returns:
        - predictions (array-like): Predicted values.
        """
        return np.array([self._predict(inputs, self.root) for inputs in X])

    def _build_tree(self, X, y, depth=0):
        num_samples = len(y)

        # Stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or num_samples < self.min_samples_split:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        # Find the best split
        best_split = self._get_best_split(X, y)
        if not best_split or best_split['variance_reduction'] <= 0:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        # Recursive splitting
        left_subtree = self._build_tree(best_split['X_left'], best_split['y_left'], depth + 1)
        right_subtree = self._build_tree(best_split['X_right'], best_split['y_right'], depth + 1)

        return Node(
            feature_index=best_split['feature_index'],
            threshold=best_split['threshold'],
            left=left_subtree,
            right=right_subtree
        )

    def _get_best_split(self, X, y):
        """
        Finds the best split by maximizing variance reduction.

        Returns:
        - best_split (dict): Dictionary containing the best split parameters.
        """
        n_samples, n_features = X.shape
        best_split = {}
        max_variance_reduction = -float('inf')

        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                X_left, y_left, X_right, y_right = self._split(X, y, feature_index, threshold)
                if len(y_left) > 0 and len(y_right) > 0:
                    variance_reduction = self._variance_reduction(y, y_left, y_right)
                    if variance_reduction > max_variance_reduction:
                        max_variance_reduction = variance_reduction
                        best_split = {
                            'feature_index': feature_index,
                            'threshold': threshold,
                            'X_left': X_left,
                            'y_left': y_left,
                            'X_right': X_right,
                            'y_right': y_right,
                            'variance_reduction': variance_reduction
                        }
        return best_split if best_split else None

    def _split(self, X, y, feature_index, threshold):
        """
        Splits the dataset into left and right subsets.

        Parameters:
        - feature_index (int): Index of the feature to split on.
        - threshold (float): Threshold value to split at.

        Returns:
        - X_left, y_left, X_right, y_right: Split datasets.
        """
        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold
        return X[left_indices], y[left_indices], X[right_indices], y[right_indices]

    def _variance_reduction(self, y, y_left, y_right):
        """
        Calculates variance reduction from the split.

        Returns:
        - variance_reduction (float): Amount of variance reduced.
        """
        weight_left = len(y_left) / len(y)
        weight_right = len(y_right) / len(y)
        reduction = np.var(y) - (weight_left * np.var(y_left) + weight_right * np.var(y_right))
        return reduction

    def _calculate_leaf_value(self, y):
        """
        Calculates the value of a leaf node.

        Returns:
        - leaf_value (float): Mean of the target values.
        """
        return np.mean(y)

    def _predict(self, inputs, node):
        """
        Recursively traverses the tree to make a prediction.

        Parameters:
        - inputs (array-like): Single sample features.
        - node (Node): Current node in the tree.

        Returns:
        - prediction (float): Predicted value.
        """
        if node.is_leaf():
            return node.value
        if inputs[node.feature_index] <= node.threshold:
            return self._predict(inputs, node.left)
        else:
            return self._predict(inputs, node.right)
class Node:
    """
    Represents a node in the regression tree.
    """

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        """
        Initializes a node.

        Parameters:
        - feature_index (int): Index of the feature used for splitting.
        - threshold (float): Threshold value for splitting.
        - left (Node): Left child node.
        - right (Node): Right child node.
        - value (float): Value at the leaf node.
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        """
        Checks if the node is a leaf node.

        Returns:
        - is_leaf (bool): True if leaf node, else False.
        """
        return self.value is not None


class RandomForestRegressorLLM:
    """
    A random forest regressor using the LLM's Regression Tree implementation.
    """

    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, max_features=None):
        """
        Initializes the random forest regressor.

        Parameters:
        - n_estimators (int): Number of trees in the forest.
        - max_depth (int): Maximum depth of each tree.
        - min_samples_split (int): Minimum samples to split an internal node.
        - max_features (int, str): Number of features to consider when looking for the best split.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.feature_indices = []

    def fit(self, X, y):
        """
        Fits the random forest to the data.

        Parameters:
        - X (array-like): Feature matrix.
        - y (array-like): Target vector.
        """
        n_samples, n_features = X.shape
        self.trees = []
        self.feature_indices = []
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            # Feature selection
            if self.max_features == 'sqrt':
                max_features = int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                max_features = int(np.log2(n_features))
            elif isinstance(self.max_features, int):
                max_features = self.max_features
            else:
                max_features = n_features  # Use all features

            features = np.random.choice(n_features, max_features, replace=False)
            self.feature_indices.append(features)
            X_sample = X_sample[:, features]

            # Train a regression tree
            tree = RegressionTreeLLM(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """
        Predicts target values for given input data.

        Parameters:
        - X (array-like): Feature matrix.

        Returns:
        - predictions (array-like): Predicted values.
        """
        predictions = []
        for tree, features in zip(self.trees, self.feature_indices):
            X_subset = X[:, features]
            predictions.append(tree.predict(X_subset))
        predictions = np.mean(predictions, axis=0)
        return predictions


def grid_search_custom_model_parallel(model_class, param_grid, X_train, y_train, cv=5, n_jobs=-1):
    """
    Parallelized grid search for custom models.

    Parameters:
    - model_class: Model class to be instantiated with parameters.
    - param_grid: Grid of hyperparameters to search.
    - X_train, y_train: Training data.
    - cv: Number of cross-validation folds.
    - n_jobs: Number of parallel jobs (-1 uses all cores).

    Returns:
    - best_params: Best hyperparameters.
    - best_score: Best cross-validation score.
    - all_results: All parameter combinations and their scores.
    """
    def evaluate_params(params):
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        cv_scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            model = model_class(**params)
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            cv_scores.append(mean_squared_error(y_val_fold, y_pred))

        avg_score = np.mean(cv_scores)
        return params, avg_score

    # Parallel evaluation
    results = Parallel(n_jobs=n_jobs)(delayed(evaluate_params)(params) for params in ParameterGrid(param_grid))

    # Find best parameters
    best_params, best_score = min(results, key=lambda x: x[1])
    all_results = results

    return best_params, best_score, all_results