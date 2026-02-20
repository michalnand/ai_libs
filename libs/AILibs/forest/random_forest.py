import numpy

class RandomDecissionTree:
    def fit(self, x, y, max_depth):
        self.max_depth = max_depth
        self.tree = self._build_tree(x, y, depth=0)

    def _build_tree(self, x, y, depth):
        if depth >= self.max_depth or len(x) <= 1:
            return {"value": y.mean(axis=0)}  # leaf node

        feature_idx, feature_val, threshold = self._select_split(x, y)

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


    def _select_split(self, x, y):
        # random feature index
        feature_idx = numpy.random.randint(x.shape[1])
        col         = x[:, feature_idx] 

        min_v = numpy.min(col)
        max_v = numpy.max(col)

     
        # Choose a random split threshold uniformly between min and max
        threshold = numpy.random.uniform(min_v, max_v)
        

        return feature_idx, col, threshold
    


class RandomForest:

    def fit(self, x, y, max_depth, num_trees = 8, num_subsamples = -1):
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

            tree = RandomDecissionTree()    
            tree.fit(x_sampled, y_sampled, max_depth)
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

    def fit(self, x, y, max_depth, num_trees = 8, num_subsamples = -1, learning_rate = 0.25):
        self.trees = []

        self.num_trees      = num_trees
        self.learning_rate  = learning_rate

        residual = y.copy()
        for n in range(self.num_trees):
            # Subsample for training, but keep full residuals
            if num_subsamples > 0 and num_subsamples < x.shape[0]:
                idx = numpy.random.choice(x.shape[0], num_subsamples, replace=False)
                x_sampled = x[idx]
                r_sampled = residual[idx]
            else:
                x_sampled = x
                r_sampled = residual

            tree = RandomDecissionTree()
            tree.fit(x_sampled, r_sampled, max_depth)
            self.trees.append(tree)

            # Update residuals on the FULL dataset
            residual = residual - learning_rate * self._prediction(tree, x)

    def predict(self, x):
        result = 0.0

        for n in range(self.num_trees):
            tmp = self.trees[n].predict(x)

            if n == 0:
                result+= tmp
            else:
                result+= self.learning_rate*tmp

        return result
    
    def predict_batch(self, x):
        result = []

        for n in range(x.shape[0]):
            y_hat = self.predict(x[n])
            result.append(y_hat)
        
        result = numpy.array(result)
        return result

    def _prediction(self, tree, x):
        result = []

        for n in range(x.shape[0]):
            y_hat = tree.predict(x[n])
            result.append(y_hat)

        result = numpy.array(result)
        return result