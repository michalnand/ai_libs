import numpy


class ExtendedIsolationForest:
    """
    Extended Isolation Forest for anomaly detection.

    Anomalies are isolated quickly because they are few and different,
    resulting in shorter average path lengths in randomly constructed
    binary trees. The anomaly score is derived from the expected path
    length: shorter paths → higher scores → more anomalous.

        
    """

    def fit(self, x, max_depth, num_trees = 32, num_subsamples = -1, eps = 0.001):
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

        # Store training set size for score normalisation in predict()
        self.n_train_samples = x.shape[0]

        # Standardise features so no single feature dominates the random projections
        #self._mean = numpy.mean(x, axis=0)
        #self._std  = numpy.std(x, axis=0)
        #self._std[self._std < 1e-10] = 1.0          # avoid division by zero for constant features
        #x = (x - self._mean) / self._std

        # Build each isolation tree independently
        for n in range(num_trees):
            # Optionally subsample the data for diversity and efficiency
            if num_subsamples > 0:
                idx = numpy.random.choice(x.shape[0], num_subsamples, replace=False)
                x_sampled = x[idx]
            else:
                x_sampled = x          

            tree = self._tree_recursion(x_sampled, 0, max_depth, eps)
            self.forest.append(tree)

        return self.forest


    def predict(self, x):
        """
        Compute anomaly scores for each sample in x.

        Each sample is passed through every tree to obtain its path length
        (number of edges from root to the terminating node). The path
        lengths are averaged across all trees and normalised using c(n),
        the average path length of unsuccessful searches in a BST, to
        produce a score in (0, 1]. Scores close to 1 indicate anomalies;
        scores close to 0.5 indicate normal points.

        Args:
            x: Data of shape (n_samples, n_features).

        Returns:
            Anomaly scores array of shape (n_samples,), values in (0, 1].
        """
        n_samples = x.shape[0]
        num_trees = len(self.forest)

        # Apply the same standardisation used during fit
        #x = (x - self._mean) / self._std    
        
        # Collect path lengths: one row per tree, one column per sample
        path_lengths = numpy.zeros((num_trees, n_samples))
        
        for t, tree in enumerate(self.forest):
            for n in range(n_samples):
                path_lengths[t, n] = self._eval_path_length(x[n], tree, 0)
        
        # Average path length across all trees for each sample
        avg_path_lengths = path_lengths.mean(axis=0)
        
        # Normalise by c(n) and convert to anomaly score: s = 2^(-E[h(x)] / c(n))
        c_n = self._compute_c(self.n_train_samples)
        scores = numpy.power(2, -avg_path_lengths / c_n)
        
        return scores


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
        if "v_proj" not in tree:
            return current_depth #+ self._compute_c(tree.get("size", 1))

        
        v_proj      = tree["v_proj"]
        threshold   = tree["threshold"]   

        x_proj = x_sample @ v_proj

        # Route sample to left or right child based on the split
        if x_proj < threshold:
            return self._eval_path_length(x_sample, tree["left"], current_depth + 1)
        else:
            return self._eval_path_length(x_sample, tree["right"], current_depth + 1)
        
    def _tree_recursion(self, x, current_depth, max_depth, eps=0.001):
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
        if current_depth >= max_depth or n_samples <= 1:
            return {}
        
      

       
        # Random projection vector
        v_proj = numpy.random.randn(n_features)         

        # Random sparse projection: zero out a random subset of coordinates so each split focuses on a few features
        p_active = numpy.sqrt(n_features) / n_features 
        inactive = numpy.random.rand(n_features) > p_active
        v_proj[inactive] = 0.0

        # Project data onto random vector
        x_proj = x @ v_proj                            
        
        min_v = numpy.min(x_proj)
        max_v = numpy.max(x_proj)   


        # If the feature values are nearly constant, no useful split exists
        if abs(min_v - max_v) <= eps:
            return {}   

        # Choose a random split threshold uniformly between min and max
        threshold = numpy.random.uniform(min_v, max_v)
        
        # Partition data into left (< threshold) and right (>= threshold)
        mask      = x_proj < threshold
        left_idx  = numpy.where(mask)[0]
        right_idx = numpy.where(~mask)[0]

        left_x  = x[left_idx]
        right_x = x[right_idx]  

        # Recurse into non-empty children
        if left_idx.size > 0:
            left_child = self._tree_recursion(left_x, current_depth + 1, max_depth, eps)
        else:
            left_child = {"size": 0}

        if right_idx.size > 0:
            right_child = self._tree_recursion(right_x, current_depth + 1, max_depth, eps) 
        else:
            right_child = {"size": 0}
        
        # Build and return the internal node
        result = {} 

        result["v_proj"]        = v_proj
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

