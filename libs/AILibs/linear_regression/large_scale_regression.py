import numpy as np
import json
import AILibs

class LargeScaleRegression:
    """
        Large scale, batch based sparse regression using SR3
    """
    def __init__(self, 
                 batch_size=4096, 
                 num_batches=256, 
                 num_steps=20, 
                 lambda_max=1.0, 
                 lambda_min=1e-4, 
                 rho=1.0, 
                 n_iter=100, 
                 rel_tol=1e-6, 
                 dictionary=None):  
        
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_steps = num_steps
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.rho = rho
        self.n_iter = n_iter
        self.rel_tol = rel_tol
        self.dictionary = dictionary
        
        self.models = []

    def fit(self, dataset_sampler):
        """
        Args:
            dataset_sampler: Sampler yielding (x_batch, y_batch) pairs.
        Returns:
            None
        """
        A = None
        b = None
        total_samples = 0

        # 1. Aggregate sufficient statistics across multiple batches
        for n in range(self.num_batches):
            x_batch, y_batch = self._sample_random_batch(dataset_sampler, self.batch_size)

            # Optional augmentation
            if self.dictionary is not None:
                z_aug = self.dictionary(x_batch) 
            else:
                z_aug = x_batch

            if A is None:
                A = np.zeros((z_aug.shape[1], z_aug.shape[1]))
                b = np.zeros((z_aug.shape[1], y_batch.shape[1]))

            A += z_aug.T @ z_aug      # Accumulate (num_features, num_features)
            b += z_aug.T @ y_batch    # Accumulate (num_features, num_outputs)
            total_samples += z_aug.shape[0]

        # Normalize by total samples so lambda is scale-invariant
        A /= total_samples
        b /= total_samples

        self.models = []
        
        # 2. Continuation path: progressively denser models
        # Logarithmic sweep from high sparsity (lambda_max) to low sparsity (lambda_min)
        lambdas = np.logspace(np.log10(self.lambda_max), np.log10(self.lambda_min), self.num_steps)
        
        # Track Z and U for warm-starting the ADMM loop
        Z_prev = None
        U_prev = None

        for lam in lambdas:
            Z, U = self._sr3_fit(A, b, lambda_=lam, rho=self.rho, 
                                 n_iter=self.n_iter, rel_tol=self.rel_tol, 
                                 Z_init=Z_prev, U_init=U_prev)
            
            # Update warm starts
            Z_prev, U_prev = Z, U

            # Evaluate (using a fresh validation batch)
            loss, r2 = self._eval(Z, dataset_sampler)    

            # Count non-zeros to observe sparsity density
            sparsity = np.mean(Z != 0.0)

            model = {
                "lambda": float(lam),
                "params": Z,
                "loss": float(loss),
                "r2_score": float(r2),
                "density": float(sparsity)
            }
            self.models.append(model)

        return None

    def predict(self, x, model_idx=None):
        """
        Args:
            x: input vector (num_inputs,) or batch (num_samples, num_inputs)
            model_idx: index of model to predict. If None, the one with lowest loss is chosen.
        Returns:
            y_pred: prediction
        """ 
        # Optional augmentation
        if self.dictionary is not None:
            z_aug = self.dictionary(x) 
        else:
            z_aug = x
        

        if model_idx is None:
            # Find the index of the model with the lowest validation loss
            model_idx = np.argmin([m["loss"] for m in self.models])

        w = self.models[model_idx]["params"]
        y_pred = z_aug @ w
        
        return y_pred

    def save(self, path):
        # We must convert numpy arrays to lists for JSON serialization
        save_data = {
            "hyperparams": {
                "batch_size": self.batch_size,
                "num_batches": self.num_batches,
                "lambda_max": self.lambda_max,
                "lambda_min": self.lambda_min
            },
            "models": []
        }
        
        for m in self.models:
            m_copy = m.copy()
            m_copy["params"] = m["params"].tolist()
            save_data["models"].append(m_copy)

        with open(path, "w") as f:
            json.dump(save_data, f)
            
    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
            
        # Reconstruct object
        hp = data["hyperparams"]
        instance = cls(batch_size=hp["batch_size"], num_batches=hp["num_batches"], 
                       lambda_max=hp["lambda_max"], lambda_min=hp["lambda_min"])
        
        # Convert params back to numpy arrays
        instance.models = data["models"]
        for m in instance.models:
            m["params"] = np.array(m["params"])
            
        return instance

    def _sample_random_batch(self, dataset_sampler, batch_size):   
        if isinstance(dataset_sampler, AILibs.BatchSampler):
            x_batch, y_batch = dataset_sampler.sample(batch_size)
        elif isinstance(dataset_sampler, (tuple, list)):
            x_data, y_data = dataset_sampler[0], dataset_sampler[1]
            indices = np.random.randint(0, len(x_data), size=(batch_size,))
            x_batch = x_data[indices]
            y_batch = y_data[indices]
        else:
            raise Exception("Unsupported input data type")

        return np.array(x_batch), np.array(y_batch)
        
    def _sr3_fit(self, XtX, XtY, lambda_=0.001, rho=1.0, n_iter=100, rel_tol=1e-6, Z_init=None, U_init=None):
        """
        Sparse linear regression using SR3 algorithm based on precomputed covariance matrices.
        
        :param XtX: numpy array, shape (n_features, n_features) equivalent to X.T @ X
        :param XtY: numpy array, shape (n_features, n_outputs) equivalent to X.T @ Y
        :param Z_init: numpy array, warm start for Z
        :param U_init: numpy array, warm start for U
        """
        n_features = XtX.shape[0]
        n_outputs = XtY.shape[1]

        # Initialize with warm starts if provided
        A = np.zeros((n_features, n_outputs))
        Z = np.zeros((n_features, n_outputs)) if Z_init is None else Z_init.copy()
        U = np.zeros((n_features, n_outputs)) if U_init is None else U_init.copy()

        # The left-hand side is constant across ADMM iterations
        # XtX + rho * I is positive definite, making np.linalg.solve safe and fast
        LHS = XtX + rho * np.eye(n_features)

        for n in range(n_iter):
            # 1. Update A: solve exact system
            RHS = XtY + rho * (Z - U)
            A = np.linalg.solve(LHS, RHS)

            # 2. Update Z: soft thresholding
            Z_prev = Z.copy()
            Z = np.sign(A + U) * np.maximum(np.abs(A + U) - lambda_ / rho, 0)

            # 3. Update dual variable U
            U += (A - Z)

            # Check ADMM convergence
            primal_res = np.linalg.norm(A - Z)
            dual_res   = rho * np.linalg.norm(Z - Z_prev)

            if primal_res < rel_tol and dual_res < rel_tol:
                break

        # Hard-threshold to remove numerical dust
        threshold = 0.5 * lambda_ / rho
        Z[np.abs(Z) < threshold] = 0.0

        return Z, U   

    def _eval(self, w, dataset_sampler):
        x_batch, y_batch = self._sample_random_batch(dataset_sampler, self.batch_size)
        z_aug = self.dictionary(x_batch) if self.dictionary is not None else x_batch

        y_pred = z_aug @ w

        # Mean Squared Error
        loss = np.mean((y_batch - y_pred)**2)
        
        # R^2 Score
        ss_res = np.sum((y_batch - y_pred)**2)
        ss_tot = np.sum((y_batch - np.mean(y_batch, axis=0))**2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))

        return loss, r2