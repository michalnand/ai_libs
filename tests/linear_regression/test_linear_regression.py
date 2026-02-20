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


