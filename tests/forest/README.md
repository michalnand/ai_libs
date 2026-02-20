# Forest Tests

Tests for `AILibs.forest` — **IsolationForest** (anomaly detection) and **RandomForest** (regression).

All datasets are synthetic with a seeded RNG (`seed=42`) for full reproducibility.

```bash
# run all forest tests
pytest tests/forest/ -v -s

# by marker
pytest -m forest

# single class
pytest tests/forest/test_forest.py::TestIsolationForest
pytest tests/forest/test_forest.py::TestRandomForest
```

---

## IsolationForest — Anomaly Detection

Each test constructs a dataset with a known normal region and injected anomalies,
then verifies that the isolation forest assigns higher anomaly scores to the outliers.
Metrics are computed via `anomaly_evaluation` (AUC-ROC, AUC-PR, F1, etc.) after
optimal threshold selection with `tune_threshold`.

| # | Test | Dataset / Target | What it checks |
|---|------|-----------------|----------------|
| 1 | `test_gaussian_cluster_with_outliers` | Tight 5-D Gaussian blob (σ = 0.5) + distant outliers (shift = 8, σ = 3) | Basic detection: anomaly scores > normal scores, AUC-ROC > 0.90, F1 > 0.70 |
| 2 | `test_multimodal_clusters` | 3 well-separated 4-D Gaussian clusters + anomalies in the gaps between them | Multi-modal density; anomalies live in sparse inter-cluster regions. AUC-ROC > 0.85 |
| 3 | `test_sinusoidal_manifold` | Points on a noisy sin(t) curve in 2-D + off-manifold outliers (y ∈ [3, 5]) | Non-linear manifold structure — the forest must capture shape, not just range. AUC-ROC > 0.85, Recall > 0.70 |
| 4 | `test_correlated_features` | Low-rank correlated 6-D data (2 latent factors) + correlation-breaking outliers (σ = 5) | Feature-correlation anomalies — outliers are not just magnitude-extreme but structurally different. AUC-ROC > 0.90, F1 > 0.70 |
| 5 | `test_subsampling` | Large 8-D Gaussian (2000 normal) + shifted outliers (40), trained with `num_subsamples=256` | Verifies the subsampling path still achieves AUC-ROC > 0.90 on a larger dataset |

---

## RandomForest — Regression

Each test generates a synthetic target from known features, fits a `RandomForest`,
and evaluates predictions with `regression_evaluation` (R², RMSE, MAE, etc.).
Since `RandomForest.predict` operates on single samples, a `_predict_batch` helper
runs batch inference.

| # | Test | Dataset / Target | What it checks |
|---|------|-----------------|----------------|
| 1 | `test_linear_target` | y = X a — pure linear map (5 features) | Sanity check: forest should approximate a linear function. R² > 0.85 |
| 2 | `test_polynomial_target` | y = [X, X², X³] · a — polynomial up to order 3 (4 features) | Piece-wise constant approximation of nonlinear polynomials. R² > 0.75 |
| 3 | `test_sinusoidal_target` | y = sin(x₁) + cos(x₂) + 0.5·sin(2x₃) — periodic (4 features) | Periodic structure from raw features, no dictionary augmentation at train time. R² > 0.80 |
| 4 | `test_noisy_linear_target` | y = X a + ε — linear with 30% noise (6 features) | Robustness to noise; predictions are evaluated against the *clean* target. R² > 0.70 |
| 5 | `test_cross_product_target` | y = Σ xᵢ·xⱼ — pairwise feature interactions (5 features) | Captures feature interactions that a single split cannot represent directly. R² > 0.75 |
| 6 | `test_subsampling` | Quadratic target (order 2), 2000 samples, `num_subsamples=512` | Subsampling path still produces competitive results on a larger dataset. R² > 0.65 |
| 7 | `test_multi_output` | y = X A, A ∈ ℝ⁵ˣ³ — 3 output columns | Multi-output regression works via `y.mean(axis=0)` at tree leaves. R² > 0.80 |
