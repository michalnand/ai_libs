"""
    Tests for AILibs.forest

    IsolationForest — anomaly detection tests with synthetic outlier datasets.
    RandomForest    — regression tests with nonlinear synthetic targets.
"""
import pytest
import numpy

import AILibs


# ------------------------------------------------------------------
# Helper: batch prediction for RandomForest (predicts one sample at a time)
# ------------------------------------------------------------------
def _predict_batch(model, x):
    """Run model.predict(x[n]) for every row and stack the results."""
    preds = [model.predict(x[n]) for n in range(x.shape[0])]
    return numpy.array(preds)


@pytest.mark.forest
class TestIsolationForest:

    # ------------------------------------------------------------------
    # 1. Gaussian cluster with distant outliers
    #    Normal data lives in a tight 5-D Gaussian blob; anomalies are
    #    placed far from the cluster centre.
    # ------------------------------------------------------------------
    def test_gaussian_cluster_with_outliers(self, rng):
        n_normal   = 500
        n_anomaly  = 20
        n_features = 5

        # tight normal cluster around the origin
        x_normal  = rng.standard_normal((n_normal, n_features)) * 0.5

        # anomalies scattered far from the origin (shift + large variance)
        x_anomaly = rng.standard_normal((n_anomaly, n_features)) * 3.0 + 8.0

        x    = numpy.concatenate([x_normal, x_anomaly], axis=0)
        y_gt = numpy.concatenate([numpy.zeros(n_normal), numpy.ones(n_anomaly)])

        forest = AILibs.forest.IsolationForest()
        forest.fit(x, max_depth=10, num_trees=128)
        scores = forest.predict(x)

        assert scores.shape == (x.shape[0],)

        # anomalies should score clearly higher than normal points
        mean_normal  = scores[:n_normal].mean()
        mean_anomaly = scores[n_normal:].mean()

        print(f"mean normal score:  {mean_normal:.4f}")
        print(f"mean anomaly score: {mean_anomaly:.4f}")

        assert mean_anomaly > mean_normal, (
            f"anomaly mean ({mean_anomaly:.4f}) should exceed "
            f"normal mean ({mean_normal:.4f})"
        )

        # full evaluation via anomaly metrics
        th = AILibs.metrics.tune_threshold(y_gt, scores, metric="f1")
        metrics = AILibs.metrics.anomaly_evaluation(y_gt, scores, th=th)
        print(AILibs.metrics.format_metrics(metrics))

        assert metrics["auc_roc"] > 0.90
        assert metrics["f1_score"] > 0.70


    # ------------------------------------------------------------------
    # 2. Multi-modal normal distribution with sparse anomalies
    #    Normal data comes from three well-separated Gaussian clusters;
    #    anomalies are placed in the gaps between clusters.
    # ------------------------------------------------------------------
    def test_multimodal_clusters(self, rng):
        n_per_cluster = 200
        n_anomaly     = 15
        n_features    = 4

        centres = numpy.array([
            [ 5.0,  5.0,  0.0,  0.0],
            [-5.0, -5.0,  0.0,  0.0],
            [ 0.0,  0.0,  5.0, -5.0],
        ])

        clusters = []
        for c in centres:
            cluster = rng.standard_normal((n_per_cluster, n_features)) * 0.6 + c
            clusters.append(cluster)

        x_normal = numpy.concatenate(clusters, axis=0)
        n_normal = x_normal.shape[0]

        # anomalies in the sparse region between clusters
        x_anomaly = rng.uniform(-3, 3, size=(n_anomaly, n_features))

        x    = numpy.concatenate([x_normal, x_anomaly], axis=0)
        y_gt = numpy.concatenate([numpy.zeros(n_normal), numpy.ones(n_anomaly)])

        forest = AILibs.forest.IsolationForest()
        forest.fit(x, max_depth=12, num_trees=128, num_subsamples=256)
        scores = forest.predict(x)

        th = AILibs.metrics.tune_threshold(y_gt, scores, metric="f1")
        metrics = AILibs.metrics.anomaly_evaluation(y_gt, scores, th=th)
        print(AILibs.metrics.format_metrics(metrics))

        assert metrics["auc_roc"] > 0.85


    # ------------------------------------------------------------------
    # 3. Sinusoidal pattern with off-manifold anomalies
    #    Normal points lie on a noisy sinusoidal curve in 2-D; anomalies
    #    are injected away from the curve.
    # ------------------------------------------------------------------
    def test_sinusoidal_manifold(self, rng):
        n_normal  = 600
        n_anomaly = 25

        t = rng.uniform(0, 2 * numpy.pi, n_normal)
        x_normal = numpy.column_stack([
            t,
            numpy.sin(t) + rng.standard_normal(n_normal) * 0.1,
        ])

        # anomalies: random points far from the sinusoidal curve
        x_anomaly = numpy.column_stack([
            rng.uniform(0, 2 * numpy.pi, n_anomaly),
            rng.uniform(3, 5, n_anomaly) * rng.choice([-1, 1], n_anomaly),
        ])

        x    = numpy.concatenate([x_normal, x_anomaly], axis=0)
        y_gt = numpy.concatenate([numpy.zeros(n_normal), numpy.ones(n_anomaly)])

        forest = AILibs.forest.IsolationForest()
        forest.fit(x, max_depth=12, num_trees=128)
        scores = forest.predict(x)

        th = AILibs.metrics.tune_threshold(y_gt, scores, metric="f1")
        metrics = AILibs.metrics.anomaly_evaluation(y_gt, scores, th=th)
        print(AILibs.metrics.format_metrics(metrics))

        assert metrics["auc_roc"] > 0.85
        assert metrics["recall"]  > 0.70


    # ------------------------------------------------------------------
    # 4. Correlated features with axis-aligned outliers
    #    Normal data has strongly correlated features; anomalies break
    #    the correlation (e.g. high x1 with low x2).
    # ------------------------------------------------------------------
    def test_correlated_features(self, rng):
        n_normal  = 500
        n_anomaly = 20
        n_features = 6

        # generate correlated normal data via a low-rank factor model
        latent  = rng.standard_normal((n_normal, 2))
        weights = rng.standard_normal((2, n_features))
        x_normal = latent @ weights + rng.standard_normal((n_normal, n_features)) * 0.2

        # anomalies: break correlations by shuffling each feature independently
        x_anomaly = rng.standard_normal((n_anomaly, n_features)) * 5.0

        x    = numpy.concatenate([x_normal, x_anomaly], axis=0)
        y_gt = numpy.concatenate([numpy.zeros(n_normal), numpy.ones(n_anomaly)])

        forest = AILibs.forest.IsolationForest()
        forest.fit(x, max_depth=10, num_trees=128)
        scores = forest.predict(x)

        th = AILibs.metrics.tune_threshold(y_gt, scores, metric="f1")
        metrics = AILibs.metrics.anomaly_evaluation(y_gt, scores, th=th)
        print(AILibs.metrics.format_metrics(metrics))

        assert metrics["auc_roc"]  > 0.90
        assert metrics["f1_score"] > 0.70


    # ------------------------------------------------------------------
    # 5. Subsampling consistency
    #    Verify that using subsampling still produces good detection on
    #    a larger dataset while keeping training efficient.
    # ------------------------------------------------------------------
    def test_subsampling(self, rng):
        n_normal  = 2000
        n_anomaly = 40
        n_features = 8

        x_normal  = rng.standard_normal((n_normal, n_features))
        x_anomaly = rng.standard_normal((n_anomaly, n_features)) * 2.0 + 6.0

        x    = numpy.concatenate([x_normal, x_anomaly], axis=0)
        y_gt = numpy.concatenate([numpy.zeros(n_normal), numpy.ones(n_anomaly)])

        forest = AILibs.forest.IsolationForest()
        forest.fit(x, max_depth=10, num_trees=128, num_subsamples=256)
        scores = forest.predict(x)

        th = AILibs.metrics.tune_threshold(y_gt, scores, metric="f1")
        metrics = AILibs.metrics.anomaly_evaluation(y_gt, scores, th=th)
        print(AILibs.metrics.format_metrics(metrics))

        assert metrics["auc_roc"] > 0.90


@pytest.mark.forest
class TestRandomForest:

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

        forest = AILibs.forest.RandomForest()
        forest.fit(x, y, max_depth=12, num_trees=64)

        y_pred  = _predict_batch(forest, x)
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)
        print(AILibs.metrics.format_metrics(metrics))

        assert metrics["r2"] > 0.85

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

        forest = AILibs.forest.RandomForest()
        forest.fit(x, y, max_depth=14, num_trees=64)

        y_pred  = _predict_batch(forest, x)
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)
        print(AILibs.metrics.format_metrics(metrics))

        assert metrics["r2"] > 0.75

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

        forest = AILibs.forest.RandomForest()
        forest.fit(x, y, max_depth=14, num_trees=64)

        y_pred  = _predict_batch(forest, x)
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)
        print(AILibs.metrics.format_metrics(metrics))

        assert metrics["r2"] > 0.80

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

        forest = AILibs.forest.RandomForest()
        forest.fit(x, y, max_depth=12, num_trees=64)

        y_pred  = _predict_batch(forest, x)
        metrics = AILibs.metrics.regression_evaluation(y_clean, y_pred)
        print(AILibs.metrics.format_metrics(metrics))

        assert metrics["r2"] > 0.70

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

        forest = AILibs.forest.RandomForest()
        forest.fit(x, y, max_depth=14, num_trees=64)

        y_pred  = _predict_batch(forest, x)
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)
        print(AILibs.metrics.format_metrics(metrics))

        assert metrics["r2"] > 0.75

    # ------------------------------------------------------------------
    # 6. Subsampling — same polynomial target but with num_subsamples
    #    Ensures the subsampling path produces competitive results on
    #    a larger dataset.
    # ------------------------------------------------------------------
    def test_subsampling(self, rng):
        n_samples  = 2000
        n_features = 5

        x = rng.standard_normal((n_samples, n_features))

        x_poly = AILibs.common.dictionary.dictionary_polynomial(x, order=2)
        x_aug  = numpy.concatenate([x, x_poly], axis=1)

        a = rng.standard_normal((x_aug.shape[1], 1))
        y = (x_aug @ a).ravel()

        forest = AILibs.forest.RandomForest()
        forest.fit(x, y, max_depth=14, num_trees=64, num_subsamples=512)

        y_pred  = _predict_batch(forest, x)
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)
        print(AILibs.metrics.format_metrics(metrics))

        assert metrics["r2"] > 0.65

    # ------------------------------------------------------------------
    # 7. Multi-output regression — y has several columns
    #    The RandomDecissionTree stores y.mean(axis=0) at leaves, so
    #    multi-output should work out of the box.
    # ------------------------------------------------------------------
    def test_multi_output(self, rng):
        n_samples   = 800
        n_features  = 5
        n_outputs   = 3

        x = rng.standard_normal((n_samples, n_features))
        a = rng.standard_normal((n_features, n_outputs))
        y = x @ a

        forest = AILibs.forest.RandomForest()
        forest.fit(x, y, max_depth=12, num_trees=64)

        y_pred  = _predict_batch(forest, x)
        metrics = AILibs.metrics.regression_evaluation(y, y_pred)
        print(AILibs.metrics.format_metrics(metrics))

        assert metrics["r2"] > 0.80


