import numpy
import json

import AILibs

class LargeScaleRandomBoostingForest:
    """
    Large Scale Random Boosting Forest (Gradient Boosted Trees) for regression.

    Trees are built sequentially. Each subsequent tree is trained on a fresh 
    data batch to predict the residual error (y - prediction) left unmodeled 
    by all previously built trees in the forest.
    """
    def __init__(self, batch_size=10000, num_trees=128, learning_rate=0.1, max_depth=6, min_leaf_size=1, feature_subsample_ratio=1.0):
        """
        Args:
            batch_size: Number of samples drawn per tree fitting iteration.
            num_trees: Total number of sequential boosted trees.
            learning_rate: Shrinkage factor (eta) applied to each tree's prediction.
            max_depth: Maximum tree depth.
            min_leaf_size: Minimum samples in a leaf.
            feature_subsample_ratio: Ratio of features randomly selected per tree.
        """
        self.batch_size                 = batch_size
        self.num_trees                  = num_trees
        self.learning_rate              = learning_rate
        self.max_depth                  = max_depth
        self.min_leaf_size              = min_leaf_size
        self.feature_subsample_ratio    = feature_subsample_ratio


    def fit(self, x_sampler):
        """
        Build an ensemble of sequentially boosted trees from training data sampler.

        Args:
            x_sampler: Sampler yielding (x_batch, y_batch) pairs.
        Returns:
            List of tree root nodes (dicts).
        """
        self.forest = []

        # Obtain initial batch to compute input statistics & initial target baseline
        x_batch, y_batch = self._sample_random_batch(x_sampler, self.batch_size)

        self.x_mean = x_batch.mean(axis=0)
        self.x_std  = x_batch.std(axis=0) + 1e-12

        # Base prediction (mean of initial target batch)
        self.base_pred = float(numpy.mean(y_batch))

        # Sequentially build boosted trees
        for n in range(self.num_trees):
            # sample fresh random batch for current boosting step
            x_batch, y_batch = self._sample_random_batch(x_sampler, self.batch_size)

            # compute current predictions using all previous trees
            current_preds = self._predict_batch(x_batch)

            # residual error becomes the target for the new tree
            residuals = y_batch - current_preds

            # standardise inputs
            x_norm = (x_batch - numpy.expand_dims(self.x_mean, 0)) / numpy.expand_dims(self.x_std, 0)

            # select random feature subset ratio
            mask = numpy.random.rand(x_norm.shape[-1]) <= self.feature_subsample_ratio
            while numpy.sum(mask) == 0:
                mask = numpy.random.rand(x_norm.shape[-1]) <= self.feature_subsample_ratio

            features_indices = numpy.where(mask)[0]
            x_selected = x_norm[:, features_indices]

            # fit tree to residuals
            result_tree = {}
            result_tree["features_indices"] = features_indices
            result_tree["tree"]             = self._tree_recursion(x_selected, residuals, 0, 1e-6)

            self.forest.append(result_tree)

        return self.forest


    def predict(self, x):
        """
        Args:
            x: Single sample vector of shape (num_features, ) or batch (num_samples, num_features)

        Returns:
            Predicted continuous value (scalar or array).
        """
        if x.ndim == 1:
            return float(self._predict_batch(numpy.expand_dims(x, 0))[0])
        else:
            return self._predict_batch(x)


    def _predict_batch(self, x_batch):
        """
        Computes cumulative predictions across base model and all trained trees.
        """
        preds = numpy.full(x_batch.shape[0], self.base_pred, dtype=numpy.float64)

        if not self.forest:
            return preds

        # Standardise batch
        x_norm = (x_batch - numpy.expand_dims(self.x_mean, 0)) / numpy.expand_dims(self.x_std, 0)

        # Accumulate predictions from each tree scaled by learning rate
        for item in self.forest:
            x_selected = x_norm[:, item["features_indices"]]
            tree = item["tree"]

            tree_preds = numpy.array([self._eval_tree(sample, tree) for sample in x_selected])
            preds += self.learning_rate * tree_preds

        return preds


    def save(self, path):
        data = {}

        data["version"]                 = 1
        data["batch_size"]              = self.batch_size
        data["num_trees"]               = self.num_trees
        data["learning_rate"]           = self.learning_rate
        data["max_depth"]               = self.max_depth
        data["min_leaf_size"]           = self.min_leaf_size
        data["feature_subsample_ratio"] = self.feature_subsample_ratio
        data["base_pred"]               = self.base_pred

        data["x_mean"] = self.x_mean.tolist()
        data["x_std"]  = self.x_std.tolist()

        forest_json = []

        for item in self.forest:
            tree_item = {}
            tree_item["features_indices"] = item["features_indices"].tolist()
            tree_item["tree"]             = item["tree"]
            forest_json.append(tree_item)

        data["forest"] = forest_json

        with open(path, "w") as f:
            json.dump(data, f)


    @classmethod
    def load(cls, path):

        with open(path, "r") as f:
            data = json.load(f)

        model = cls(
            batch_size=data["batch_size"],
            num_trees=data["num_trees"],
            learning_rate=data["learning_rate"],
            max_depth=data["max_depth"],
            min_leaf_size=data["min_leaf_size"],
            feature_subsample_ratio=data["feature_subsample_ratio"],
        )

        model.base_pred = data["base_pred"]

        model.x_mean = numpy.array(
            data["x_mean"],
            dtype=numpy.float32
        )

        model.x_std = numpy.array(
            data["x_std"],
            dtype=numpy.float32
        )

        model.forest = []

        for item in data["forest"]:
            tree_item = {}
            tree_item["features_indices"] = numpy.array(
                item["features_indices"],
                dtype=numpy.int64
            )
            tree_item["tree"] = item["tree"]
            model.forest.append(tree_item)

        return model


    def _sample_random_batch(self, x_sampler, batch_size):

        if isinstance(x_sampler, AILibs.BatchSampler):
            x_batch, y_batch = x_sampler.sample(batch_size)
        elif isinstance(x_sampler, tuple) or isinstance(x_sampler, list):
            x_data, y_data = x_sampler[0], x_sampler[1]
            indices = numpy.random.randint(0, len(x_data), (batch_size, ))
            x_batch = x_data[indices]
            y_batch = y_data[indices]
        else:
            raise Exception("Unsupported input data type")

        return numpy.array(x_batch, dtype=numpy.float64), numpy.array(y_batch, dtype=numpy.float64)


    def _eval_tree(self, x_sample, tree):
        """
        Recursively traverse tree until reaching leaf prediction.
        """
        if "value" in tree:
            return tree["value"]

        feature_idx = tree["feature_idx"]
        threshold   = tree["threshold"]

        if x_sample[feature_idx] < threshold:
            return self._eval_tree(x_sample, tree["left"])
        else:
            return self._eval_tree(x_sample, tree["right"])


    def _tree_recursion(self, x, y, current_depth, eps=0.001):
        """
        Recursively build a decision tree node predicting residual target y.
        """
        n_samples, n_features = x.shape

        # Stopping criteria
        if current_depth >= self.max_depth or n_samples <= self.min_leaf_size or numpy.all(y == y[0]):
            return self._create_leaf(y)

        # Select a random feature dimension
        feature_idx = numpy.random.randint(0, n_features)
        col = x[:, feature_idx]

        min_v = numpy.min(col)
        max_v = numpy.max(col)

        if abs(min_v - max_v) <= eps:
            return self._create_leaf(y)

        # Choose a random split threshold uniformly between min and max
        threshold = numpy.random.uniform(min_v, max_v)

        mask      = col < threshold
        left_idx  = numpy.where(mask)[0]
        right_idx = numpy.where(~mask)[0]

        left_x,  right_x  = x[left_idx],  x[right_idx]
        left_y,  right_y  = y[left_idx],  y[right_idx]

        if left_idx.size > 0:
            left_child = self._tree_recursion(left_x, left_y, current_depth + 1, eps)
        else:
            left_child = self._create_leaf(y)

        if right_idx.size > 0:
            right_child = self._tree_recursion(right_x, right_y, current_depth + 1, eps)
        else:
            right_child = self._create_leaf(y)

        result = {}
        result["feature_idx"] = int(feature_idx)
        result["threshold"]   = float(threshold)
        result["left"]        = left_child
        result["right"]       = right_child
        result["size"]        = int(n_samples)

        return result


    def _create_leaf(self, y):
        # Leaf predicts the mean residual error of samples in the node
        leaf = {
            "size": int(len(y)),
            "value": float(numpy.mean(y)) if len(y) > 0 else 0.0
        }
        return leaf