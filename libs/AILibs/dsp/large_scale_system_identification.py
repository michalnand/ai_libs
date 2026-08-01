import numpy

class LargeScaleSystemIdentification:


    def __init__(self, solver_instance):
        self.solver_instance = solver_instance


    def fit(self, x, u=None, lags=0):
            """
            Fits the sparse model to predict x(n+1).
    
            Parameters:
            - x: np.ndarray, shape (num_samples, num_states)
            - u: np.ndarray, shape (num_samples, num_inputs), optional
            - dictionary: callable, maps X_aug -> Z = dictionary(X_aug), optional
            - lags: int, number of past delays to include (default: 0)
            """
        
            # Step 1: Perform time-delay augmentation & alignment (optional)
            X_aug, Y = self._build_lagged_features(x, u, lags=lags)
    
            # Step 2: Fit sparse regression model
            return self.solver_instance.fit(X_aug, Y)


    def _build_lagged_features(self, x, u=None, lags=0):
        """
        Aligns and stacks state x and input u across time delays.
        
        Returns:
            X_aug: Shape (N - lags - 1, feature_dim)
            Y:     Target x(n+1), Shape (N - lags - 1, num_states)
        """
        num_samples, num_states = x.shape
        if lags < 0:
            raise ValueError("Lags must be a non-negative integer.")

        # Target Y is x(n+1), starting from index (lags + 1) up to num_samples - 1
        Y = x[lags + 1:]    

        # Collect slices for x(n), x(n-1), ..., x(n-lags)
        lagged_blocks = []
        for delay in range(lags + 1):
            start = lags - delay
            end = num_samples - 1 - delay
            lagged_blocks.append(x[start:end])

            # Optionally include lagged control inputs u(n), u(n-1), ...
            if u is not None:
                lagged_blocks.append(u[start:end])

        # Stack features horizontally along columns
        X_aug = numpy.hstack(lagged_blocks)
        return X_aug, Y
