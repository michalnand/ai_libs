"""
    Tests for AILibs.linear_regression (LargeScaleRegression)
"""
import pytest
import numpy
import os

import AILibs

@pytest.mark.regression
class TestLargeScaleRegression:

    def test_lsr_basic_aggregation_and_sparsity(self, rng):
        """Test if the batch aggregation correctly finds the global sparse solution."""
        x = rng.standard_normal((5000, 20))
        
        # Create true sparse weights (20 features, 3 outputs, only 5 non-zeros per output)
        a = numpy.zeros((20, 3))
        for col in range(3):
            idxs = rng.choice(20, size=5, replace=False)
            a[idxs, col] = rng.standard_normal(size=5) * 2.0

        y = x @ a

        # Initialize LargeScaleRegression
        # 10 batches of 500 = 5000 samples (full dataset)
        lsr = AILibs.LargeScaleRegression(
            batch_size=500, 
            num_batches=10, 
            num_steps=15,          # 15 steps in the continuation path
            lambda_max=1.0,
            lambda_min=1e-4,
            rho=1.0
        )

        # Fit using the tuple sampler format
        lsr.fit((x, y))

        assert len(lsr.models) == 15, "Continuation path should generate exactly num_steps models."

        # The model with the lowest loss should closely approximate the true sparse weights
        best_idx = numpy.argmin([m["loss"] for m in lsr.models])
        a_est = lsr.models[best_idx]["params"]

        assert a_est.shape == a.shape

        # Verify predictions
        y_pred = lsr.predict(x, model_idx=best_idx)
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)
        print("Best Model Metrics:\n", AILibs.metrics.format_metrics(metrics))

        # Check sparsity pattern and value recovery
        true_zeros = (a == 0.0)
        assert numpy.allclose(a_est[true_zeros], 0.0, atol=1e-3), "Failed to zero out true zeros."
        assert numpy.allclose(a_est, a, atol=1e-2), "Failed to recover true coefficients."


    def test_lsr_noisy_data(self, rng):
        """Test if the continuation path handles noise and stops at the right phase transition."""
        x = rng.standard_normal((4000, 30))
        a = numpy.zeros((30, 2))
        
        # 4 non-zeros per output
        for col in range(2):
            idxs = rng.choice(30, size=4, replace=False)
            a[idxs, col] = rng.standard_normal(size=4) * 1.5

        y = x @ a
        y_noisy = y + rng.standard_normal(y.shape) * 0.2  # Add significant noise

        lsr = AILibs.LargeScaleRegression(
            batch_size=1000,
            num_batches=4,
            num_steps=10,
            lambda_max=2.0,
            lambda_min=1e-2
        )
        lsr.fit((x, y_noisy))

        best_idx = numpy.argmin([m["loss"] for m in lsr.models])
        a_est = lsr.models[best_idx]["params"]

        # Ensure the selected model remains sparse despite noise
        n_nonzero_true = numpy.count_nonzero(a)
        n_nonzero_est = numpy.count_nonzero(a_est)
        
        # It shouldn't overfit the noise by adding dozens of fake features
        assert n_nonzero_est <= 2 * n_nonzero_true, \
            f"Overfitted noise: {n_nonzero_est} non-zeros found vs {n_nonzero_true} true."

        # The estimated non-zeros should be relatively close to true values
        assert numpy.allclose(a_est, a, atol=3e-1)


    def test_lsr_dictionary_expansion(self, rng):
        """Test the optional non-linear dictionary mapping during batch streaming."""
        x = rng.standard_normal((2000, 5))
        
        # Custom dictionary: concatenate original features with squared features -> 10 dims
        def poly_dict(inp):
            return numpy.hstack([inp, inp**2])

        # True weights depend on the expanded 10-dimensional space
        a = numpy.zeros((10, 1))
        a[0, 0] = 1.5   # x_0
        a[7, 0] = -2.0  # x_2 squared

        # Generate y using the expanded space
        z = poly_dict(x)
        y = z @ a

        lsr = AILibs.LargeScaleRegression(
            batch_size=400,
            num_batches=5,
            num_steps=5,
            dictionary=poly_dict
        )
        lsr.fit((x, y))

        best_idx = numpy.argmin([m["loss"] for m in lsr.models])
        a_est = lsr.models[best_idx]["params"]

        # The parameters should map to the 10-dimensional expanded space
        assert a_est.shape == (10, 1)
        assert numpy.allclose(a_est, a, atol=1e-2)

        # Predict should automatically apply the dictionary to raw `x`
        y_pred = lsr.predict(x)
        assert y_pred.shape == y.shape
        assert numpy.allclose(y_pred, y, atol=1e-2)


    def test_lsr_save_and_load(self, rng, tmp_path):
        """Test JSON serialization of hyperparameters and numpy model parameters."""
        x = rng.standard_normal((1000, 8))
        a = rng.standard_normal((8, 2))
        y = x @ a

        # Train a model
        lsr_orig = AILibs.LargeScaleRegression(
            batch_size=250, 
            num_batches=4, 
            num_steps=3
        )
        lsr_orig.fit((x, y))
        
        # Save to temporary path provided by pytest
        file_path = os.path.join(tmp_path, "test_model.json")
        lsr_orig.save(file_path)

        # Load into new instance
        lsr_loaded = AILibs.LargeScaleRegression.load(file_path)

        # Verify hyperparameters are restored
        assert lsr_loaded.batch_size == lsr_orig.batch_size
        assert lsr_loaded.num_batches == lsr_orig.num_batches

        # Verify parameters are numpy arrays and match exactly
        w_orig = lsr_orig.models[-1]["params"]
        w_loaded = lsr_loaded.models[-1]["params"]
        
        assert isinstance(w_loaded, numpy.ndarray)
        assert numpy.allclose(w_orig, w_loaded, atol=1e-8)

        # Verify predictions match exactly
        y_pred_orig = lsr_orig.predict(x, model_idx=-1)
        y_pred_loaded = lsr_loaded.predict(x, model_idx=-1)
        
        assert numpy.allclose(y_pred_orig, y_pred_loaded, atol=1e-8)