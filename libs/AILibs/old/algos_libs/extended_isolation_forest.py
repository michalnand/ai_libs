import numpy
import json

class ExtendedIsolationForest:

    def fit(self, x, max_depth, num_trees = 32, eps = 0.001):

        self.proj   = []
        self.forest = []

        self.n_train_samples = x.shape[0]
        self.n_dims = x.shape[1]
        for n in range(num_trees):
            # random orthogonal projection
            p = self._random_orthogonal_matrix(self.n_dims)

            # project data
            xp   = x @ p    

            # fit tree
            tree = self._tree_recursion(xp, 0, max_depth, eps)

            self.proj.append(p)
            self.forest.append(tree)

          

        return self.forest, self.proj


    def predict(self, x):
        n_samples = x.shape[0]
        num_trees = len(self.forest)
        
        path_lengths = numpy.zeros((num_trees, n_samples))
        
        for t, tree in enumerate(self.forest):
            x_proj = x @ self.proj[t]  # project input data using the same projection as the tree
            for n in range(n_samples):  
                path_lengths[t, n] = self._eval_path_length(x_proj[n], tree, 0)
        
        # Average path length across all trees
        avg_path_lengths = path_lengths.mean(axis=0)    
        
        # Use training set size for normalization
        c_n = self._compute_c(self.n_train_samples)
        scores = numpy.power(2, -avg_path_lengths / c_n)
        
        return scores


    def _eval_path_length(self, x_sample, tree, current_depth):
        if not tree:
            return current_depth
        
        feature_idx = tree["feature_idx"]
        v_split     = tree["v_split"]   

        if x_sample[feature_idx] < v_split:
            return self._eval_path_length(x_sample, tree["left_child"], current_depth + 1)
        else:
            return self._eval_path_length(x_sample, tree["right_child"], current_depth + 1)
        
    def _tree_recursion(self, x, current_depth, max_depth, eps=0.001):

        n_samples, n_features = x.shape
        if current_depth >= max_depth or n_samples <= 1:
            return {}
        
        # pick random feature for splitting
        feature_idx = numpy.random.randint(0, n_features)
        col = x[:, feature_idx] 

        min_v = numpy.min(col)
        max_v = numpy.max(col)

        # features too close
        if abs(min_v - max_v) <= eps:
            return {}
        
        # pick random value for splitting   
        v_split = numpy.random.uniform(min_v, max_v)

        # split indices into left/right using local col
        mask      = col < v_split
        left_idx  = numpy.where(mask)[0]
        right_idx = numpy.where(~mask)[0]

        left_x  = x[left_idx]
        right_x = x[right_idx]  

        if left_idx.size > 0:
            left_child = self._tree_recursion(left_x, current_depth + 1, max_depth, eps)
        else:
            left_child = {}

        if right_idx.size > 0:
            right_child = self._tree_recursion(right_x, current_depth + 1, max_depth, eps) 
        else:
            right_child = {}
        
        result = {} 

        result["feature_idx"] = feature_idx
        result["v_split"]     = v_split
        result["left_child"]  = left_child
        result["right_child"] = right_child

        return result


    def _compute_c(self, n):
        if n <= 1:
            return 0.0
        return 2.0 * (numpy.log(n - 1.0) + 0.5772156649) - (2.0 * (n - 1.0) / n)


    def _anomaly_scores(self, path_lengths_all):
        n_samples = path_lengths_all.shape[0]
        c_n = self._compute_c(n_samples)
        scores = numpy.power(2, -path_lengths_all / c_n)
        return scores
    
    def _random_orthogonal_matrix(self, d):
        # Step 1: Gaussian random matrix
        A = numpy.random.randn(d, d)

        # Step 2: QR decomposition
        Q, R = numpy.linalg.qr(A)

        # Step 3: Make distribution uniform (fix sign ambiguity)
        D = numpy.diag(numpy.sign(numpy.diag(R)))
        Q = Q @ D

        return Q



if __name__ == "__main__":

    x = numpy.array([
        [1.0, 2.0],
        [1.5, 1.8],
        [3.0, 3.5],
        [3.2, 3.0],
        [5.0, 100.0],
        [5.0, 8.0],
        [8.0, 8.0]
    ])

    i_forest = IsolationForest()
    i_forest.fit(x, 8, num_trees=128)

    scores = i_forest.predict(x)
    print(scores)