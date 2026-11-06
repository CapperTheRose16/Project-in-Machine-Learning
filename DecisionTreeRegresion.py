import numpy as np
class DecisionTreeRegresion:
    def __init__(self, max_depth=5, min_mse_reduction=0.05):
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative.")
        if min_mse_reduction < 0:
            raise ValueError("min_mse_reduction must be non-negative.")

        self.max_depth = max_depth
        self.min_mse_reduction = min_mse_reduction
        self.tree_ = None
        self.feature_names_in_ = None
        self.n_features_in_ = None

    @staticmethod
    def _mse(values):
        return np.mean((values - np.mean(values)) ** 2)

    def _find_best_split(self, X, y):
        parent_mse = self._mse(y)
        best_split = None

        for axis in range(X.shape[1]):
            values = np.sort(np.unique(X[:, axis]))
            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:
                left_mask = X[:, axis] < threshold
                left_y = y[left_mask]
                right_y = y[~left_mask]

                weighted_mse = (
                    len(left_y) * self._mse(left_y)
                    + len(right_y) * self._mse(right_y)
                ) / len(y)

                if best_split is None or weighted_mse < best_split["weighted_mse"]:
                    best_split = {
                        "axis": axis,
                        "threshold": threshold,
                        "weighted_mse": weighted_mse,
                        "mse_reduction": parent_mse - weighted_mse,
                    }

        return best_split

    @staticmethod
    def _make_leaf(y):
        return {
            "leaf": True,
            "prediction": float(np.mean(y)),
            "n_samples": len(y),
            "mse": float(DecisionTreeRegresion._mse(y)),
        }

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < 2:
            return self._make_leaf(y)

        split = self._find_best_split(X, y)
        if split is None or split["mse_reduction"] <= self.min_mse_reduction:
            return self._make_leaf(y)

        left_mask = X[:, split["axis"]] < split["threshold"]
        return {
            "leaf": False,
            "axis": split["axis"],
            "threshold": split["threshold"],
            "n_samples": len(y),
            "mse_reduction": split["mse_reduction"],
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[~left_mask], y[~left_mask], depth + 1),
        }

    def fit(self, X, y):
        columns = getattr(X, "columns", None)
        if columns is not None:
            self.feature_names_in_ = list(columns)
            X = X.to_numpy(dtype=float)
        else:
            X = np.asarray(X, dtype=float)
            if X.ndim != 2:
                raise ValueError("X must be a 2D array or DataFrame.")
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]

        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array or DataFrame.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows.")
        if len(y) == 0:
            raise ValueError("Cannot fit the tree with an empty dataset.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain only finite numeric values.")

        self.n_features_in_ = X.shape[1]
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def _predict_one(self, node, row):
        while not node["leaf"]:
            if row[node["axis"]] < node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["prediction"]

    def predict(self, X):
        if self.tree_ is None:
            raise RuntimeError("Call fit before predict.")

        if self.feature_names_in_ and hasattr(X, "columns"):
            X = X[self.feature_names_in_].to_numpy(dtype=float)
        else:
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(1, -1)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must have {self.n_features_in_} features in the training order."
            )
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite numeric values.")

        return np.array([self._predict_one(self.tree_, row) for row in X])

    def print_tree(self, node=None, spacing=""):
        if self.tree_ is None:
            raise RuntimeError("Call fit before print_tree.")
        if node is None:
            node = self.tree_

        if node["leaf"]:
            print(
                f"{spacing}Predict {node['prediction']:.2f} "
                f"(samples: {node['n_samples']}; MSE: {node['mse']:.3f})"
            )
            return

        feature = self.feature_names_in_[node["axis"]]
        print(
            f"{spacing}{feature} < {node['threshold']:.2f}? "
            f"(samples: {node['n_samples']})"
        )
        print(f"{spacing}--> True:")
        self.print_tree(node["left"], spacing + "  ")
        print(f"{spacing}--> False:")
        self.print_tree(node["right"], spacing + "  ")
