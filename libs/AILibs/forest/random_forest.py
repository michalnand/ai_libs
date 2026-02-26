import numpy

class RandomDecisionTree:
    def fit(self, x, y, max_depth, num_candidates = 1):
        self.max_depth = max_depth
        self.num_candidates = num_candidates
        self.tree = self._build_tree(x, y, depth=0)

    def _build_tree(self, x, y, depth):
        if depth >= self.max_depth or len(x) <= 1:
            return {"value": y.mean(axis=0)}  # leaf node

        feature_idx, feature_val, threshold = self._select_split(x, y, self.num_candidates)

        # Avoid trivial split       
        left_mask  = feature_val <= threshold
        right_mask = feature_val > threshold

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return {"value": y.mean(axis=0)}

        # Recursively build tree
        return {
            "feature_idx"   : feature_idx,
            "threshold"     : threshold,
            "left"          : self._build_tree(x[left_mask], y[left_mask], depth + 1),
            "right"         : self._build_tree(x[right_mask], y[right_mask], depth + 1)
        }   

    def predict(self, x, node = None):
        if node is None:
            node = self.tree

        if "value" in node: 
            return node["value"]
        if x[node["feature_idx"]] <= node["threshold"]:
            return self.predict(x, node["left"])
        else:
            return self.predict(x, node["right"])

    
    def _select_split_random(self, x):
        # random feature index
        feature_idx = numpy.random.randint(x.shape[1])
        col         = x[:, feature_idx] 

        min_v = numpy.min(col)
        max_v = numpy.max(col)

        # Choose a random split threshold uniformly between min and max
        threshold = numpy.random.uniform(min_v, max_v)
        
        return feature_idx, col, threshold
    
    def _select_split(self, x, y, num_candidates=16):
        best_error     = numpy.inf
        best_split     = None

        for _ in range(num_candidates):
            feature_idx, col, threshold = self._select_split_random(x)

            left_mask  = col <= threshold
            right_mask = col > threshold

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue    

            # compute MSE for this split
            left_error  = numpy.sum((y[left_mask]  - y[left_mask].mean(axis=0))  ** 2)
            right_error = numpy.sum((y[right_mask] - y[right_mask].mean(axis=0)) ** 2)
            error = left_error + right_error

            if error < best_error:
                best_error = error
                best_split = (feature_idx, col, threshold)

        # fallback: use a pure random split if no valid candidate was found
        if best_split is None:
            return self._select_split_random(x)

        return best_split
    


class RandomForest:

    def fit(self, x, y, max_depth, num_trees = 8, num_subsamples = -1, num_random_candidates=16):
        self.trees = []

        self.num_trees = num_trees
        for n in range(self.num_trees):

            if num_subsamples > 0:
                idx = numpy.random.choice(x.shape[0], num_subsamples, replace=False)
                x_sampled = x[idx]
                y_sampled = y[idx]
            else:
                x_sampled = x 
                y_sampled = y      

            tree = RandomDecisionTree()    
            tree.fit(x_sampled, y_sampled, max_depth, num_candidates=num_random_candidates)
            self.trees.append(tree)

    def predict(self, x):
        result = 0.0

        for n in range(self.num_trees):
            result+= self.trees[n].predict(x)

        result = result/self.num_trees
        return result
    

    def predict_batch(self, x):
        result = []

        for n in range(x.shape[0]):
            y_hat = self.predict(x[n])
            result.append(y_hat)
        
        result = numpy.array(result)
        return result
        
    
    

class RandomBoostingForest:

    def fit(self, x, y, max_depth, num_trees = 8, num_subsamples = -1, learning_rate = 0.1):
        self.trees = []

        self.num_trees      = num_trees
        self.learning_rate  = learning_rate
        self.initial_prediction = y.mean(axis=0)

        residual = y - self.initial_prediction
        for n in range(self.num_trees):            
            # Subsample for training, but keep full residuals
            if num_subsamples > 0 and num_subsamples < x.shape[0]:
                idx = numpy.random.choice(x.shape[0], num_subsamples, replace=False)
                x_sampled = x[idx]
                r_sampled = residual[idx]
            else:
                x_sampled = x
                r_sampled = residual

            tree = RandomDecisionTree(num)
            tree.fit(x_sampled, r_sampled, max_depth)
            self.trees.append(tree) 

            # Update residuals on the FULL dataset
            residual = residual - learning_rate * self._predict_tree_batch(tree, x)

            

    def predict(self, x):
        result = numpy.copy(self.initial_prediction)

        for n in range(self.num_trees):
            result+= self.learning_rate*self.trees[n].predict(x)

        return result
    
    def predict_batch(self, x):
        result = []

        for n in range(x.shape[0]):
            y_hat = self.predict(x[n])
            result.append(y_hat)
        
        result = numpy.array(result)
        return result

    def _predict_tree_batch(self, tree, x):
        result = numpy.array([tree.predict(x[n]) for n in range(x.shape[0])])
        return result