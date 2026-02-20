"""
    Tests for AILibs.linear_regression  (lr_fit)
"""
import pytest
import numpy

import AILibs




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


