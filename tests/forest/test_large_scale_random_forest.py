"""
    Tests for AILibs.forest

    LargeScaleRandomForest — regression tests with synthetic nonlinear targets.
"""
import pytest
import numpy

import AILibs


# ------------------------------------------------------------------
# Helper: batch prediction for LargeScaleRandomForest (predicts one sample at a time)
# ------------------------------------------------------------------
def _predict_batch(model, x):
    """Run model.predict(x[n]) for every row and stack the results."""
    preds = [model.predict(x[n]) for n in range(x.shape[0])]
    return numpy.array(preds)


@pytest.mark.forest
class TestLargeScaleRandomForest:

    # ------------------------------------------------------------------
    # 1. Linear target — y = X @ a  (sanity check)
    #    A deep-enough random forest should approximate a linear map
    #    reasonably well, especially with enough trees.
    # ------------------------------------------------------------------
    def test_linear_target(self, rng):
        n_samples  = 800
        n_features = 5

        x = rng.standard_normal((n_samples, n_features))
        a = rng.standard_normal((n_features, 1))
        y = (x @ a).ravel()

        forest = AILibs.LargeScaleRandomForest(
            batch_size=512, 
            num_trees=64, 
            max_depth=12
        )
        forest.fit((x, y))

        y_pred  = _predict_batch(forest, x)
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)
        print(AILibs.metrics.format_metrics(metrics))

        assert metrics["r2"] > 0.80

    # ------------------------------------------------------------------
    # 2. Polynomial target — y = poly(X)
    #    Random forests handle nonlinear relationships via piece-wise
    #    constant approximation; a polynomial should be well captured.
    # ------------------------------------------------------------------
    def test_polynomial_target(self, rng):
        n_samples  = 1000
        n_features = 4

        x = rng.standard_normal((n_samples, n_features))

        x_poly = AILibs.common.dictionary.dictionary_polynomial(x, order=3)
        x_aug  = numpy.concatenate([x, x_poly], axis=1)

        a = rng.standard_normal((x_aug.shape[1], 1))
        y = (x_aug @ a).ravel()

        forest = AILibs.LargeScaleRandomForest(
            batch_size=512, 
            num_trees=64, 
            max_depth=14
        )
        forest.fit((x, y))

        y_pred  = _predict_batch(forest, x)
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)
        print(AILibs.metrics.format_metrics(metrics))

        assert metrics["r2"] > 0.70

    # ------------------------------------------------------------------
    # 3. Sinusoidal target — y = sin(x1) + cos(x2)
    #    Tests the forest's ability to capture periodic structure from
    #    raw features (no dictionary augmentation at train time).
    # ------------------------------------------------------------------
    def test_sinusoidal_target(self, rng):
        n_samples  = 1000
        n_features = 4

        x = rng.uniform(-numpy.pi, numpy.pi, (n_samples, n_features))
        y = numpy.sin(x[:, 0]) + numpy.cos(x[:, 1]) + 0.5 * numpy.sin(2 * x[:, 2])

        forest = AILibs.LargeScaleRandomForest(
            batch_size=512, 
            num_trees=64, 
            max_depth=14
        )
        forest.fit((x, y))

        y_pred  = _predict_batch(forest, x)
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)
        print(AILibs.metrics.format_metrics(metrics))

        assert metrics["r2"] > 0.75

    # ------------------------------------------------------------------
    # 4. Noisy linear target — y = X @ a + noise
    #    The forest should still recover most of the signal even when
    #    the target is corrupted by moderate Gaussian noise.
    # ------------------------------------------------------------------
    def test_noisy_linear_target(self, rng):
        n_samples  = 1000
        n_features = 6

        x = rng.standard_normal((n_samples, n_features))
        a = rng.standard_normal((n_features, 1))
        y_clean = (x @ a).ravel()
        noise   = rng.standard_normal(n_samples) * 0.3 * numpy.std(y_clean)
        y = y_clean + noise

        forest = AILibs.LargeScaleRandomForest(
            batch_size=512, 
            num_trees=64, 
            max_depth=12
        )
        forest.fit((x, y))

        y_pred  = _predict_batch(forest, x)
        metrics = AILibs.metrics.regression_evaluation(y_clean, y_pred)
        print(AILibs.metrics.format_metrics(metrics))

        assert metrics["r2"] > 0.65

    # ------------------------------------------------------------------
    # 5. Cross-product interaction — y = Σ x_i * x_j
    #    Verifies the forest picks up feature interactions that a single
    #    decision stump per node cannot represent directly.
    # ------------------------------------------------------------------
    def test_cross_product_target(self, rng):
        n_samples  = 1000
        n_features = 5

        x = rng.standard_normal((n_samples, n_features))

        x_cross = AILibs.common.dictionary.dictionary_cross_products(x)
        a = rng.standard_normal((x_cross.shape[1], 1))
        y = (x_cross @ a).ravel()

        forest = AILibs.LargeScaleRandomForest(
            batch_size=512, 
            num_trees=64, 
            max_depth=14
        )
        forest.fit((x, y))

        y_pred  = _predict_batch(forest, x)
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)
        print(AILibs.metrics.format_metrics(metrics))

        assert metrics["r2"] > 0.70

    # ------------------------------------------------------------------
    # 6. Subsampling / Large-scale batching
    #    Ensures the batching strategy produces competitive results on
    #    a larger dataset with random sampling.
    # ------------------------------------------------------------------
    def test_subsampling(self, rng):
        n_samples  = 2000
        n_features = 5

        x = rng.standard_normal((n_samples, n_features))

        x_poly = AILibs.common.dictionary.dictionary_polynomial(x, order=2)
        x_aug  = numpy.concatenate([x, x_poly], axis=1)

        a = rng.standard_normal((x_aug.shape[1], 1))
        y = (x_aug @ a).ravel()

        forest = AILibs.LargeScaleRandomForest(
            batch_size=512, 
            num_trees=64, 
            max_depth=14
        )
        forest.fit((x, y))

        y_pred  = _predict_batch(forest, x)
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)
        print(AILibs.metrics.format_metrics(metrics))

        assert metrics["r2"] > 0.60 