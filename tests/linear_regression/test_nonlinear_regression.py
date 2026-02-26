"""
    Tests for AILibs.linear_regression  (lr_fit)
"""
import pytest
import numpy

import AILibs



def _make_sparse_system(rng, x_aug, n_outputs, sparsity):
    """Helper: create sparse coefficients and noiseless targets."""
    a = rng.standard_normal((x_aug.shape[1], n_outputs))
    mask = rng.random(a.shape) < sparsity
    a[mask] = 0.0
    y = x_aug @ a
    return a, y




@pytest.mark.regression
class TestNLrFit:

    def test_polynomial_fit(self, rng):
        sparsity = 0.8
        x = rng.standard_normal((1000, 11))
        
        x_tmp = AILibs.common.dictionary.dictionary_polynomial(x, order=3)

        x_const = AILibs.common.dictionary.dictionary_constant(x)

        x_aug = numpy.concatenate([x, x_const, x_tmp], axis=1)

        a = rng.standard_normal((x_aug.shape[1], 7))


        mask = rng.random(a.shape) < sparsity
        a[mask] = 0.0  

         
        y = x_aug @ a

        a_est = AILibs.linear_regression.lr_sparse_fit(x_aug, y)

        assert a_est.shape == a.shape

        y_pred = x_aug @ a_est
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)

        print(AILibs.metrics.format_metrics(metrics))

        assert numpy.allclose(a_est, a, atol=1e-6)


    def test_dictionary_cross_fit(self, rng):
        sparsity = 0.8
        x = rng.standard_normal((1000, 11))
        
        x_tmp = AILibs.common.dictionary.dictionary_cross_products(x)
        x_const = AILibs.common.dictionary.dictionary_constant(x)
        x_aug = numpy.concatenate([x, x_const, x_tmp], axis=1)

        a = rng.standard_normal((x_aug.shape[1], 7))



        mask = rng.random(a.shape) < sparsity
        a[mask] = 0.0  

         
        y = x_aug @ a

        a_est = AILibs.linear_regression.lr_sparse_fit(x_aug, y)

        assert a_est.shape == a.shape

        y_pred = x_aug @ a_est
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)

        print(AILibs.metrics.format_metrics(metrics))

        assert numpy.allclose(a_est, a, atol=1e-6)


    def test_dictionary_sin_cos(self, rng):
        sparsity = 0.8
        x = rng.standard_normal((1000, 11))
        
        x_tmp   = AILibs.common.dictionary.dictionary_sin_cos(x, n_harmonics=5)
        x_const = AILibs.common.dictionary.dictionary_constant(x)   
        x_aug   = numpy.concatenate([x, x_const, x_tmp], axis=1)

        a = rng.standard_normal((x_aug.shape[1], 7))



        mask = rng.random(a.shape) < sparsity
        a[mask] = 0.0  

         
        y = x_aug @ a

        a_est = AILibs.linear_regression.lr_sparse_fit(x_aug, y)

        assert a_est.shape == a.shape

        y_pred = x_aug @ a_est
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)

        print(AILibs.metrics.format_metrics(metrics))

        assert numpy.allclose(a_est, a, atol=1e-6)


    def test_dictionary_sin_cos_cross(self, rng):
        sparsity = 0.8
        x = rng.standard_normal((1000, 11))
        
        x_tmp   = AILibs.common.dictionary.dictionary_sin_cos_cross(x)
        x_const = AILibs.common.dictionary.dictionary_constant(x)
        x_aug   = numpy.concatenate([x, x_const, x_tmp], axis=1)

        a = rng.standard_normal((x_aug.shape[1], 7))

        mask = rng.random(a.shape) < sparsity
        a[mask] = 0.0  


        y = x_aug @ a

        a_est = AILibs.linear_regression.lr_sparse_fit(x_aug, y)

        assert a_est.shape == a.shape

        y_pred = x_aug @ a_est
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)

        print(AILibs.metrics.format_metrics(metrics))

        assert numpy.allclose(a_est, a, atol=1e-6)


@pytest.mark.regression
class TestNLrSR3Fit:
    """SR3 sparse regression with nonlinear dictionary features."""

    def test_sr3_polynomial_fit(self, rng):
        """SR3 on polynomial dictionary features."""
        sparsity = 0.9
        x = rng.standard_normal((1000, 8))

        x_poly  = AILibs.common.dictionary.dictionary_polynomial(x, order=3)
        x_const = AILibs.common.dictionary.dictionary_constant(x)
        x_aug   = numpy.concatenate([x, x_const, x_poly], axis=1)

        a, y = _make_sparse_system(rng, x_aug, n_outputs=5, sparsity=sparsity)

        a_est = AILibs.linear_regression.sr3_fit(x_aug, y, lambda_=0.5, rho=5.0, n_iter=500)

        assert a_est.shape == a.shape

        y_pred = x_aug @ a_est
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)
        print(AILibs.metrics.format_metrics(metrics))

        assert numpy.allclose(a_est, a, atol=5e-2)


    def test_sr3_cross_products_fit(self, rng):
        """SR3 on cross-product dictionary features."""
        sparsity = 0.9
        x = rng.standard_normal((1000, 8))

        x_cross = AILibs.common.dictionary.dictionary_cross_products(x)
        x_const = AILibs.common.dictionary.dictionary_constant(x)
        x_aug   = numpy.concatenate([x, x_const, x_cross], axis=1)

        a, y = _make_sparse_system(rng, x_aug, n_outputs=5, sparsity=sparsity)

        a_est = AILibs.linear_regression.sr3_fit(x_aug, y, lambda_=0.5, rho=5.0, n_iter=500)

        assert a_est.shape == a.shape

        y_pred = x_aug @ a_est
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)
        print(AILibs.metrics.format_metrics(metrics))

        assert numpy.allclose(a_est, a, atol=5e-2)


    def test_sr3_sin_cos_fit(self, rng):
        """SR3 on sin/cos harmonic dictionary features."""
        sparsity = 0.9
        x = rng.standard_normal((1000, 8))

        x_sc    = AILibs.common.dictionary.dictionary_sin_cos(x, n_harmonics=3)
        x_const = AILibs.common.dictionary.dictionary_constant(x)
        x_aug   = numpy.concatenate([x, x_const, x_sc], axis=1)

        a, y = _make_sparse_system(rng, x_aug, n_outputs=5, sparsity=sparsity)

        a_est = AILibs.linear_regression.sr3_fit(x_aug, y, lambda_=0.5, rho=5.0, n_iter=500)

        assert a_est.shape == a.shape

        y_pred = x_aug @ a_est
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)
        print(AILibs.metrics.format_metrics(metrics))

        assert numpy.allclose(a_est, a, atol=5e-2)


    def test_sr3_sin_cos_cross_fit(self, rng):
        """SR3 on sin/cos cross-product dictionary features."""
        sparsity = 0.95
        x = rng.standard_normal((2000, 5))

        x_scc   = AILibs.common.dictionary.dictionary_sin_cos_cross(x)
        x_const = AILibs.common.dictionary.dictionary_constant(x)
        x_aug   = numpy.concatenate([x, x_const, x_scc], axis=1)

        a, y = _make_sparse_system(rng, x_aug, n_outputs=3, sparsity=sparsity)

        a_est = AILibs.linear_regression.sr3_fit(x_aug, y, lambda_=0.5, rho=5.0, n_iter=500)

        assert a_est.shape == a.shape

        y_pred = x_aug @ a_est
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)
        print(AILibs.metrics.format_metrics(metrics))

        assert numpy.allclose(a_est, a, atol=5e-2)


    def test_sr3_polynomial_noisy(self, rng):
        """SR3 on polynomial dictionary with noisy observations."""
        sparsity = 0.9
        x = rng.standard_normal((1500, 6))

        x_poly  = AILibs.common.dictionary.dictionary_polynomial(x, order=2)
        x_const = AILibs.common.dictionary.dictionary_constant(x)
        x_aug   = numpy.concatenate([x, x_const, x_poly], axis=1)

        a, y = _make_sparse_system(rng, x_aug, n_outputs=4, sparsity=sparsity)
        y_noisy = y + rng.standard_normal(y.shape) * 0.1

        a_est = AILibs.linear_regression.sr3_fit(x_aug, y_noisy, lambda_=0.3, rho=5.0, n_iter=500)

        assert a_est.shape == a.shape

        y_pred = x_aug @ a_est
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)
        print(AILibs.metrics.format_metrics(metrics))

        assert numpy.allclose(a_est, a, atol=3e-1)


    def test_sr3_sparsity_recovery(self, rng):
        """SR3 should recover the correct zero/non-zero pattern on dictionary features."""
        sparsity = 0.95
        x = rng.standard_normal((2000, 6))

        x_poly  = AILibs.common.dictionary.dictionary_polynomial(x, order=2)
        x_const = AILibs.common.dictionary.dictionary_constant(x)
        x_aug   = numpy.concatenate([x, x_const, x_poly], axis=1)

        a, y = _make_sparse_system(rng, x_aug, n_outputs=3, sparsity=sparsity)

        a_est = AILibs.linear_regression.sr3_fit(x_aug, y, lambda_=0.5, rho=5.0, n_iter=500)

        # true zero positions should remain zero
        true_zeros = (a == 0.0)
        assert numpy.allclose(a_est[true_zeros], 0.0, atol=1e-3), \
            f"SR3 failed to zero out true-zero positions, max = {numpy.max(numpy.abs(a_est[true_zeros]))}"

        # true non-zero positions should be recovered
        true_nonzeros = ~true_zeros
        if numpy.any(true_nonzeros):
            assert numpy.allclose(a_est[true_nonzeros], a[true_nonzeros], atol=5e-2)

        n_nonzero_true = numpy.count_nonzero(a)
        n_nonzero_est  = numpy.count_nonzero(a_est)
        print(f"true non-zeros: {n_nonzero_true}, estimated non-zeros: {n_nonzero_est}")

        y_pred = x_aug @ a_est
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)
        print(AILibs.metrics.format_metrics(metrics))