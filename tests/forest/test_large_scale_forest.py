"""
    Tests for AILibs.forest
    
    LargeScaleIsolationForest — anomaly detection tests with synthetic outlier datasets.
"""
import pytest
import numpy

import AILibs


# ------------------------------------------------------------------
# Helper: batch prediction for LargeScaleIsolationForest
# ------------------------------------------------------------------
def _predict_batch(model, x):
    """Run model.score(x[n]) for every row and stack the results."""
    preds = [model.score(x[n]) for n in range(x.shape[0])]
    return numpy.array(preds)


@pytest.mark.forest
class TestLargeScaleIsolationForest:

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

        # Initialize with batch_size (equivalent to previous num_subsamples)
        forest = AILibs.LargeScaleIsolationForest(batch_size=256, num_trees=128)
        forest.fit(x)
        
        scores = _predict_batch(forest, x)

        assert scores.shape == (x.shape[0],)
        assert scores.shape == y_gt.shape   

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

        forest = AILibs.LargeScaleIsolationForest(batch_size=256, num_trees=128)
        forest.fit(x)
        
        scores = _predict_batch(forest, x)

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

        forest = AILibs.LargeScaleIsolationForest(batch_size=256, num_trees=128)
        forest.fit(x)
        
        scores = _predict_batch(forest, x)

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

        forest = AILibs.LargeScaleIsolationForest(batch_size=256, num_trees=128)
        forest.fit(x)
        
        scores = _predict_batch(forest, x)

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

        # Testing with a larger batch size explicitly
        forest = AILibs.LargeScaleIsolationForest(batch_size=256, num_trees=128)
        forest.fit(x)
        
        scores = _predict_batch(forest, x)

        th = AILibs.metrics.tune_threshold(y_gt, scores, metric="f1")
        metrics = AILibs.metrics.anomaly_evaluation(y_gt, scores, th=th)
        print(AILibs.metrics.format_metrics(metrics))

        assert metrics["auc_roc"] > 0.90