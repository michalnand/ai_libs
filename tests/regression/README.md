# Linear Regression Tests

Tests for `AILibs.linear_regression` — **lr_fit** (ordinary least-squares) and **lr_sparse_fit** (sparse regression), combined with `AILibs.common.dictionary` augmentations for nonlinear regression.

All datasets are synthetic with a seeded RNG (`seed=42`) for full reproducibility.

```bash
# run all linear-regression tests
pytest tests/linear_regression/ -v -s

# by marker
pytest -m regression

# single class
pytest tests/linear_regression/test_linear_regression.py::TestLrFit
pytest tests/linear_regression/test_nonlinear_regression.py::TestNLrFit
```

---

## TestLrFit — Linear Regression

Each test constructs a random linear system y = X a (+ optional noise), fits
coefficients with `lr_fit` or `lr_sparse_fit`, and verifies that the estimated
parameters match the ground truth. Metrics are computed via `regression_evaluation`.

| # | Test | Dataset / Target | What it checks |
|---|------|-----------------|----------------|
| 1 | `test_single_output` | y = X a, 1000 × 5 features, single output | Basic OLS: recovered coefficients match ground truth (atol = 1e-6) |
| 2 | `test_multiple_outputs` | y = X A, 1000 × 5 features, 7 outputs | Multi-output OLS: shape and coefficient accuracy (atol = 1e-6) |
| 3 | `test_noisy_data` | y = X a + ε (σ = 0.1), 1000 × 11 features, 14 outputs | Robustness to noise: coefficients close to ground truth (atol = 1e-1) |
| 4 | `test_sparse_fit` | y = X a + ε, 80% of coefficients zeroed, 1000 × 11 features, 13 outputs | Sparse regression (`lr_sparse_fit`) recovers sparse structure (atol = 1e-2) |

---

## TestNLrFit — Nonlinear Regression (Dictionary Methods)

Each test augments the raw features with a dictionary expansion (polynomial,
cross-products, sin/cos harmonics, or combined), zeroes 80% of the coefficients
to create a sparse target, and fits with `lr_sparse_fit`. All tests verify
exact coefficient recovery (atol = 1e-6).

| # | Test | Dictionary / Augmentation | What it checks |
|---|------|--------------------------|----------------|
| 1 | `test_polynomial_fit` | `dictionary_polynomial(x, order=3)` + `dictionary_constant` | Sparse polynomial regression up to order 3 over 11 features, 7 outputs |
| 2 | `test_dictionary_cross_fit` | `dictionary_cross_products(x)` + `dictionary_constant` | Pairwise feature interactions with sparse coefficients, 7 outputs |
| 3 | `test_dictionary_sin_cos` | `dictionary_sin_cos(x, n_harmonics=5)` + `dictionary_constant` | Harmonic (sin/cos) basis expansion with sparse coefficients, 7 outputs |
| 4 | `test_dictionary_sin_cos_cross` | `dictionary_sin_cos_cross(x)` + `dictionary_constant` | Combined sin/cos cross-product dictionary with sparse coefficients, 7 outputs |
