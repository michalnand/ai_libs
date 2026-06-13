import numpy


class LargeScaleIsolationForest:
    """
    Isolation Forest for anomaly detection.

    Anomalies are isolated quickly because they are few and different,
    resulting in shorter average path lengths in randomly constructed
    binary trees. The anomaly score is derived from the expected path
    length: shorter paths → higher scores → more anomalous.

    Reference:
        Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou.
        "Isolation forest." ICDM, 2008.
    """

    def __init__(self, batch_size = 10000, num_trees = 128, min_leaf_size = 1, feature_subsample_ratio = 1.0, projection_dim = -1):
        
        self.batch_size                 = batch_size
        self.num_trees                  = num_trees
        self.min_leaf_size              = min_leaf_size
        self.feature_subsample_ratio    = feature_subsample_ratio
        self.projection_dim             = projection_dim
        self.max_depth                  = max(int(numpy.log2(batch_size)), 1)



    def fit(self, x_sampler):
        """
        Build an ensemble of isolation trees from training data.

        Args:
            x:              Training data of shape (n_samples, n_features).
            max_depth:      Maximum depth for each isolation tree.
            num_trees:      Number of isolation trees in the forest.
            num_subsamples: If > 0, each tree is built on a random subsample
                            of this size (recommended for large datasets).
            eps:            Minimum range threshold for a feature; if the
                            feature's range is <= eps the node becomes a leaf.

        Returns:
            List of isolation tree root nodes (dicts).
        """
        self.forest = []

        # Build each isolation tree independently
        for n in range(self.num_trees):
            # subsample batch
            x_batch = self._sample_random_batch(x_sampler, self.batch_size)

            # select only random features subset
            mask = numpy.random.rand(x_batch.shape[-1]) <= self.feature_subsample_ratio
            while numpy.sum(mask) == 0:
                mask = numpy.random.rand(x_batch.shape[-1]) <= self.feature_subsample_ratio

            features_indices = numpy.where(mask)[0] 
            
            x_selected = x_batch[:, features_indices]

            # standardise
            x_mean = x_selected.mean(axis=0)
            x_std  = x_selected.std(axis=0) + 1e-12

            x_norm = (x_selected - numpy.expand_dims(x_mean, 0))/numpy.expand_dims(x_std, 0)

            # random projection matrix
            if self.projection_dim > 0:
                projection_matrix = numpy.random.randn(x_norm.shape[1], self.projection_dim)
                x_proj = x_norm@projection_matrix
            else:
                projection_matrix = None
                x_proj = x_norm

            # tree saving
            result_tree = {}

            result_tree["features_indices"]     = features_indices
            result_tree["mean"]                 = x_mean
            result_tree["x_std"]                = x_std
            result_tree["projection_matrix"]    = projection_matrix
            result_tree["tree"]                 = self._tree_recursion(x_proj, 0, 1e-6)  

            self.forest.append(result_tree)

        return self.forest


    def score(self, x):
        """
        Args:
            x: data numpy array of shape (num_features, )

        Returns:
            Anomaly score scalar, range [0, 1]
        """        
        path_length = 0

        
        for item in self.forest:
            # only selected features
            x_selected = x[item["features_indices"]]
            
            # normalise
            x_norm = (x_selected - item["mean"])/item["x_std"]

            # random projection, if any
            if item["projection_matrix"] is not None:
                x_proj = numpy.expand_dims(x_norm, 0)@item["projection_matrix"]
                x_proj = x_proj[0]
            else:
                x_proj = x_norm

            tree = item["tree"]
            path_length+= self._eval_path_length(x_proj, tree, 0)  
        
        # Average path length across all trees for each sample
        avg_path_lengths = path_length/len(self.forest)

        
        # Normalise by c(n) and convert to anomaly score: s = 2^(-E[h(x)] / c(n))
        c_n = self._compute_c(self.batch_size)
        score_result = numpy.power(2, -avg_path_lengths / c_n)
        
        return score_result



    def _sample_random_batch(self, x_sampler, batch_size):

        indices = numpy.random.randint(0, len(x_sampler), (batch_size, ))

        if isinstance(x_sampler, numpy.ndarray):
            result = x_sampler[indices]
        elif isinstance(x_sampler, list):
            result = x_sampler[indices]
        #elif isinstance(x_sampler, BatchSampler):
        #    result = x_sampler.sample(batch_size)
        else:
            raise Exception("Unsupported input data type")
        
        return numpy.array(result)

    def _eval_path_length(self, x_sample, tree, current_depth):
        """
        Recursively traverse the tree to find the path length for a single sample.

        At each internal node the sample is routed left or right based on the
        stored split (feature index + threshold). The recursion stops when a
        leaf (empty dict) is reached, and the accumulated depth is returned.

        Args:
            x_sample:      Single sample, shape (n_features,).
            tree:          Current node (dict with keys feature_idx, threshold,
                           left, right) or empty dict for a leaf.
            current_depth: Depth accumulated so far.

        Returns:
            Path length (int) from root to the terminating leaf.
        """
        # Leaf node — return the depth reached
        if not tree:
            return current_depth
        
        # External node (leaf with size info) — no children to recurse into
        if "feature_idx" not in tree:
            return current_depth

        
        feature_idx = tree["feature_idx"]
        threshold   = tree["threshold"]   

        # Route sample to left or right child based on the split
        if x_sample[feature_idx] < threshold:
            return self._eval_path_length(x_sample, tree["left"], current_depth + 1)
        else:
            return self._eval_path_length(x_sample, tree["right"], current_depth + 1)
        

    def _tree_recursion(self, x, current_depth, eps=0.001):
        """
        Recursively build a single isolation tree.

        At each node a random feature and a random split value (uniform
        between the feature's min and max) are chosen. Data is partitioned
        into left (< threshold) and right (>= threshold) subsets. Recursion
        stops when:
          - max_depth is reached,
          - the node contains <= 1 sample, or
          - the selected feature has near-zero range (< eps).

        Args:
            x:             Data subset for this node, shape (n_samples, n_features).
            current_depth: Current depth in the tree.
            max_depth:     Maximum allowed depth.
            eps:           Minimum feature range to allow a split.

        Returns:
            A dict representing the node with keys:
                feature_idx, threshold, left, right
            or an empty dict {} for a leaf node.
        """
        n_samples, n_features = x.shape

        # Stopping criteria: max depth reached or node is pure (single sample)
        if current_depth >= self.max_depth or n_samples <= self.min_leaf_size:
            return {}
        
        # Randomly select a feature dimension for splitting
        feature_idx = numpy.random.randint(0, n_features)
        col = x[:, feature_idx] 

        min_v = numpy.min(col)
        max_v = numpy.max(col)

        # If the feature values are nearly constant, no useful split exists
        if abs(min_v - max_v) <= eps:
            return {}
        
        # Choose a random split threshold uniformly between min and max
        threshold = numpy.random.uniform(min_v, max_v)
        
        # Partition data into left (< threshold) and right (>= threshold)
        mask      = col < threshold
        left_idx  = numpy.where(mask)[0]
        right_idx = numpy.where(~mask)[0]

        left_x  = x[left_idx]
        right_x = x[right_idx]  

        # Recurse into non-empty children
        if left_idx.size > 0:
            left_child = self._tree_recursion(left_x, current_depth + 1, eps)
        else:
            left_child = {"size": 0}

        if right_idx.size > 0:
            right_child = self._tree_recursion(right_x, current_depth + 1, eps) 
        else:
            right_child = {"size": 0}
        
        # Build and return the internal node
        result = {} 

        result["feature_idx"]   = feature_idx
        result["threshold"]     = threshold
        result["left"]          = left_child
        result["right"]         = right_child
        result["size"]          = n_samples

        return result   


    def _compute_c(self, n):
        """
        Compute c(n), the average path length of unsuccessful searches in a
        Binary Search Tree, used to normalise the anomaly score.

        Formula: c(n) = 2·H(n-1) - 2·(n-1)/n
        where H(i) ≈ ln(i) + γ  (γ = 0.5772… is the Euler–Mascheroni constant).

        Args:   
            n: Number of samples.

        Returns:
            Average path length c(n). Returns 0.0 when n <= 1.
        """
        if n <= 1:
            return 0.0
        return 2.0 * (numpy.log(n - 1.0) + 0.5772156649) - (2.0 * (n - 1.0) / n)


    def _anomaly_scores(self, path_lengths_all):
        """
        Convert raw path lengths into anomaly scores using the formula:
            s(x, n) = 2^(-E[h(x)] / c(n))

        Args:
            path_lengths_all: Array of average path lengths, shape (n_samples,).

        Returns:
            Anomaly scores array, values in (0, 1].
        """
        n_samples = path_lengths_all.shape[0]
        c_n = self._compute_c(n_samples)
        scores = numpy.power(2, -path_lengths_all / c_n)
        return scores

