import numpy


class IsolationForest:

    def fit(self, x, max_depth, num_trees = 8, eps = 0.001):

        n_samples, n_features = x.shape
        path_lengths_all = []

        for n in range(num_trees):
            path_lengths    = numpy.zeros((n_samples, ), dtype=int)
            all_indices     = numpy.arange(n_samples)

            self._tree_recursion(x, path_lengths, all_indices, 0, max_depth, eps)

            path_lengths_all.append(path_lengths)

        path_lengths_all = numpy.array(path_lengths_all)
        path_lengths_all = path_lengths_all.mean(axis=0)

      
        scores = self._anomaly_scores(path_lengths_all)

        return path_lengths_all, scores
        

    def _tree_recursion(self, x, path_lengths, indices, current_depth, max_depth, eps=0.001):
        # select subset
        x_sub = x[indices]
        n_samples, n_features = x_sub.shape

        if current_depth >= max_depth or n_samples <= 1:
            path_lengths[indices] = int(current_depth)
            return

        # pick random feature for splitting
        feature_id = numpy.random.randint(0, n_features)
        col = x_sub[:, feature_id] 

        min_v = numpy.min(col)
        max_v = numpy.max(col)

        # features too close — treat as leaf
        if abs(min_v - max_v) <= eps:
            path_lengths[indices] = int(current_depth)
            return

        # pick value for splitting 
        v_split = numpy.random.uniform(min_v, max_v)

        # split indices into left/right using local col
        mask = col < v_split
        left_idx  = indices[mask]
        right_idx = indices[~mask]

        if left_idx.size > 0:
            self._tree_recursion(x, path_lengths, left_idx, current_depth + 1, max_depth, eps)

        if right_idx.size > 0:
            self._tree_recursion(x, path_lengths, right_idx, current_depth + 1, max_depth, eps)


    def _compute_c(self, n):
        if n <= 1:
            return 0.0
        return 2.0 * (numpy.log(n - 1.0) + 0.5772156649) - (2.0 * (n - 1.0) / n)


    def _anomaly_scores(self, path_lengths_all):
        n_samples = path_lengths_all.shape[0]
        c_n = self._compute_c(n_samples)
        scores = numpy.power(2, -path_lengths_all / c_n)
        return scores

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
    path_lengths, scores = i_forest.fit(x, 8)


    print(scores)