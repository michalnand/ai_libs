import numpy
import json

import AILibs

class LargeScaleRandomForest:
    """
    Simple Random Forest for large-scale data fitting in batches.

    Uses random feature sub-selection and random threshold splits 
    similar to Isolation Forest, aggregating target predictions across trees.
    """
    def __init__(self, batch_size=10000, num_trees=128, max_depth=15, min_leaf_size=1, feature_subsample_ratio=1.0):
        
        self.batch_size                 = batch_size
        self.num_trees                  = num_trees
        self.max_depth                  = max_depth
        self.min_leaf_size              = min_leaf_size
        self.feature_subsample_ratio    = feature_subsample_ratio


    def fit(self, x_sampler):
        """
        Build an ensemble of decision trees from training data sampler.

        Args:
            x_sampler: Sampler yielding (x_batch, y_batch) pairs.
        Returns:
            List of tree root nodes (dicts).
        """
        self.forest = []

        x_batch, y_batch = self._sample_random_batch(x_sampler, self.batch_size)

        self.x_mean = x_batch.mean(axis=0)
        self.x_std  = x_batch.std(axis=0) + 1e-12

        # Build each decision tree independently
        for n in range(self.num_trees):
            # subsample batch
            x_batch, y_batch = self._sample_random_batch(x_sampler, self.batch_size)

            # standardise
            x_norm = (x_batch - numpy.expand_dims(self.x_mean, 0)) / numpy.expand_dims(self.x_std, 0)

            # select only random features subset
            mask = numpy.random.rand(x_norm.shape[-1]) <= self.feature_subsample_ratio
            while numpy.sum(mask) == 0:
                mask = numpy.random.rand(x_norm.shape[-1]) <= self.feature_subsample_ratio

            features_indices = numpy.where(mask)[0]
            x_selected = x_norm[:, features_indices]

            # tree saving
            result_tree = {}

            result_tree["features_indices"] = features_indices
            result_tree["tree"]             = self._tree_recursion(x_selected, y_batch, 0, 1e-6)

            self.forest.append(result_tree)

        return self.forest


    def predict(self, x):
        """
        Args:
            x: Single sample vector of shape (num_features, )

        Returns:
            Aggregated prediction across the forest.
        """
        # standardise
        x_norm = (x - self.x_mean) / self.x_std

        predictions = []
        for item in self.forest:
            x_selected = x_norm[item["features_indices"]]
            tree = item["tree"]
            pred = self._eval_tree(x_selected, tree)
            predictions.append(pred)

        # Average prediction across trees (or majority vote if targets are integer labels)
        if isinstance(predictions[0], (int, numpy.integer)):
            values, counts = numpy.unique(predictions, return_counts=True)
            return values[numpy.argmax(counts)]
        else:
            return float(numpy.mean(predictions))



    def save(self, path):
        data = {}

        data["version"]                 = 1
        data["batch_size"]              = self.batch_size
        data["num_trees"]               = self.num_trees
        data["max_depth"]               = self.max_depth
        data["min_leaf_size"]           = self.min_leaf_size
        data["feature_subsample_ratio"] = self.feature_subsample_ratio

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
            max_depth=data["max_depth"],
            min_leaf_size=data["min_leaf_size"],
            feature_subsample_ratio=data["feature_subsample_ratio"],
        )

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

        return numpy.array(x_batch), numpy.array(y_batch)


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
        Recursively build a decision tree node using random feature and random split value.
        """
        n_samples, n_features = x.shape

        # Stopping criteria
        if current_depth >= self.max_depth or n_samples <= self.min_leaf_size or len(numpy.unique(y)) <= 1:
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
        leaf = {"size": int(len(y))}

        if isinstance(y[0], (int, numpy.integer)):
            values, counts = numpy.unique(y, return_counts=True)
            leaf["value"] = values[numpy.argmax(counts)].item() if hasattr(values[numpy.argmax(counts)], 'item') else values[numpy.argmax(counts)]
        else:
            leaf["value"] = float(numpy.mean(y))

        return leaf
