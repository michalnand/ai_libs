"""
    Tests for AILibs.linear_regression  (lr_fit)
"""
import pytest
import numpy

import AILibs




@pytest.mark.regression
class TestLrFit:

    def test_single_output(self, rng):
        x = rng.standard_normal((1000, 5))
        a = rng.standard_normal((5, 1))
        y = x @ a
        a_est = AILibs.linear_regression.lr_fit(x, y)

        assert a_est.shape == a.shape

        y_pred = x @ a_est
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)

        print(AILibs.metrics.format_metrics(metrics))

        assert numpy.allclose(a_est, a, atol=1e-6)


    def test_multiple_outputs(self, rng):
        x = rng.standard_normal((1000, 5))
        a = rng.standard_normal((5, 7))

        y = x @ a
        a_est = AILibs.linear_regression.lr_fit(x, y)

        assert a_est.shape == a.shape

        y_pred = x @ a_est
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)

        print(AILibs.metrics.format_metrics(metrics))

        assert numpy.allclose(a_est, a, atol=1e-6)


    def test_noisy_data(self, rng):
        x = rng.standard_normal((1000, 11))
        a = rng.standard_normal((11, 14))

        y = x @ a
        y_noisy = y + rng.standard_normal(y.shape) * 0.1

        a_est = AILibs.linear_regression.lr_fit(x, y_noisy)

        assert a_est.shape == a.shape

        y_pred = x @ a_est
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)

        print(AILibs.metrics.format_metrics(metrics))

        assert numpy.allclose(a_est, a, atol=1e-1)



    def test_sparse_fit(self, rng):
        sparsity = 0.8
        x = rng.standard_normal((1000, 11))
        a = rng.standard_normal((11, 13))

        mask = rng.random(a.shape) < sparsity
        a[mask] = 0.0   

        y = x @ a
        y_noisy = y + rng.standard_normal(y.shape) * 0.1

        a_est = AILibs.linear_regression.lr_sparse_fit(x, y_noisy)

        assert a_est.shape == a.shape

        y_pred = x @ a_est
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)

        print(AILibs.metrics.format_metrics(metrics))

        assert numpy.allclose(a_est, a, atol=1e-2)


    def test_sr3_fit(self, rng):
        sparsity = 0.8
        x = rng.standard_normal((1000, 11))
        a = rng.standard_normal((11, 13))   

        mask = rng.random(a.shape) < sparsity
        a[mask] = 0.0   

        y = x @ a

        # lambda_/rho = 0.02, small enough to keep coefficients of any magnitude
        a_est = AILibs.linear_regression.sr3_fit(x, y, lambda_=0.1, rho=5.0)

        assert a_est.shape == a.shape

        y_pred = x @ a_est
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)

        print(AILibs.metrics.format_metrics(metrics))

        assert numpy.allclose(a_est, a, atol=1e-3)


    def test_sr3_fit_noisy(self, rng):
        """SR3 on noisy data — should still recover a reasonable sparse model."""
        sparsity = 0.8
        x = rng.standard_normal((1000, 11))
        a = rng.standard_normal((11, 13))

        mask = rng.random(a.shape) < sparsity
        a[mask] = 0.0

        y = x @ a
        y_noisy = y + rng.standard_normal(y.shape) * 0.1

        a_est = AILibs.linear_regression.sr3_fit(x, y_noisy, lambda_=0.5, rho=5.0)

        assert a_est.shape == a.shape

        y_pred = x @ a_est
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)
        print(AILibs.metrics.format_metrics(metrics))

        assert numpy.allclose(a_est, a, atol=2e-1)


    def test_sr3_fit_sparsity_structure(self, rng):
        """SR3 should recover the correct zero/non-zero pattern."""
        sparsity = 0.9
        x = rng.standard_normal((2000, 8))
        a = rng.standard_normal((8, 5))

        mask = rng.random(a.shape) < sparsity
        a[mask] = 0.0

        y = x @ a

        a_est = AILibs.linear_regression.sr3_fit(x, y, lambda_=0.5, rho=5.0, n_iter=500)

        assert a_est.shape == a.shape

        # true zero positions should remain zero (or near-zero) in the estimate
        true_zeros = (a == 0.0)
        assert numpy.allclose(a_est[true_zeros], 0.0, atol=1e-3), \
            f"SR3 failed to zero out true-zero positions, max residual = {numpy.max(numpy.abs(a_est[true_zeros]))}"

        # true non-zero positions should be recovered
        true_nonzeros = ~true_zeros
        if numpy.any(true_nonzeros):
            assert numpy.allclose(a_est[true_nonzeros], a[true_nonzeros], atol=5e-2), \
                f"SR3 failed to recover non-zero coefficients"

        y_pred = x @ a_est
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)
        print(AILibs.metrics.format_metrics(metrics))


    def test_sr3_fit_single_output(self, rng):
        """SR3 with a single output column."""
        sparsity = 0.7
        x = rng.standard_normal((500, 20))
        a = rng.standard_normal((20, 1))

        mask = rng.random(a.shape) < sparsity
        a[mask] = 0.0

        y = x @ a

        a_est = AILibs.linear_regression.sr3_fit(x, y, lambda_=0.3, rho=5.0)

        assert a_est.shape == a.shape

        y_pred = x @ a_est
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)
        print(AILibs.metrics.format_metrics(metrics))

        assert numpy.allclose(a_est, a, atol=5e-2)


    def test_sr3_fit_large_system(self, rng):
        """SR3 on a wider system (many features, few non-zeros)."""
        x = rng.standard_normal((4000, 256))
        a = numpy.zeros((256, 16))    

        # only 20 non-zero coefficients per output
        for col in range(3):
            idxs = rng.choice(256, size=20, replace=False)
            a[idxs, col] = rng.standard_normal() * 2.0

        y = x @ a
        y_noisy = y + rng.standard_normal(y.shape) * 0.05

        # lambda_/rho = 0.2, safely above noise floor, below true coeff magnitude ~2.0
        a_est = AILibs.linear_regression.sr3_fit(x, y_noisy, lambda_=1.0, rho=5.0, n_iter=500)

        assert a_est.shape == a.shape

        y_pred = x @ a_est
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)
        print(AILibs.metrics.format_metrics(metrics))

        # check sparsity: the estimate should have few non-zero entries
        n_nonzero_true = numpy.count_nonzero(a)
        n_nonzero_est  = numpy.count_nonzero(a_est)
        print(f"true non-zeros: {n_nonzero_true}, estimated non-zeros: {n_nonzero_est}")

        assert n_nonzero_est < 2 * n_nonzero_true, \
            f"SR3 result not sparse enough: {n_nonzero_est} vs {n_nonzero_true} true non-zeros"

        assert numpy.allclose(a_est, a, atol=1e-2)


