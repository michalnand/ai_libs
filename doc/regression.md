<p align="center">
  <h1 align="center">📈 AILibs.linear_regression & AILibs.common</h1>
  <p align="center">
    <strong>Linear, sparse, and regularised regression with nonlinear dictionary augmentation</strong>
  </p>
  <p align="center">
    <em>From ordinary least squares to ADMM-based sparse recovery — clean, transparent, pure NumPy.</em>
  </p>
  <p align="center">
    <a href="#-overview">Overview</a> •
    <a href="#-regression-methods">Regression Methods</a> •
    <a href="#-dictionary-augmentation">Dictionary Augmentation</a> •
    <a href="#-api-reference">API Reference</a> •
    <a href="#-applications">Applications</a>
  </p>
</p>

---

## ✨ Overview

The `AILibs.linear_regression` module provides a progression of regression solvers — from the classical pseudoinverse to modern sparse recovery algorithms. Combined with the **dictionary augmentation** functions in `AILibs.common`, these linear methods can capture highly nonlinear relationships.

### Regression Methods — `AILibs.linear_regression`

| | Method | Type | Key Idea |
|---|---|---|---|
| 📐 | **`lr_fit`** | Closed-form | Ordinary Least Squares via pseudoinverse |
| 🔬 | **`lr_sparse_fit`** | Iterative greedy | Top-$k$ thresholding on residual regression |
| 💎 | **`lr_lasso_fit`** | Coordinate descent | L1-regularised regression (Lasso) with soft-thresholding |
| ⚗️ | **`sr3_fit`** | ADMM | Sparse Relaxed Regularised Regression (SR3) |

### Dictionary Functions — `AILibs.common`

| | Function | Output Columns |
|---|---|---|
| 1️⃣ | **`dictionary_constant`** | $1$ (bias / intercept) |
| 📊 | **`dictionary_polynomial`** | $D \cdot (\text{order} - 1)$ |
| ✖️ | **`dictionary_cross_products`** | $\binom{D}{2}$ |
| 🌊 | **`dictionary_sin_cos`** | $2 \cdot D \cdot n$ |
| 🔀 | **`dictionary_sin_cos_cross`** | $2 \cdot D \cdot (D{-}1)$ |

---

## 📐 Regression Methods

### 📐 Ordinary Least Squares — `lr_fit`

The foundational linear regression. Solves $\mathbf{Y} = \mathbf{X}\mathbf{A}$ for the coefficient matrix $\mathbf{A}$ in closed form via the pseudoinverse:

$$\mathbf{A} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{Y} = \mathbf{X}^+ \mathbf{Y}$$

Internally uses `numpy.linalg.lstsq`, which computes the minimum-norm solution via SVD — numerically stable even for rank-deficient or ill-conditioned $\mathbf{X}$.

| Parameter | Description |
|---|---|
| `X` | Feature matrix, shape `(n_samples, n_inputs)` |
| `Y` | Target matrix, shape `(n_samples, n_outputs)` |

| Returns | Description |
|---|---|
| `A` | Coefficient matrix, shape `(n_inputs, n_outputs)` |

```python
from AILibs.linear_regression import lr_fit

A = lr_fit(X, Y)
Y_pred = X @ A
```

**Key features:**
- ⚡ Single-pass closed-form solution — no iterations, no hyperparameters
- 📊 Multi-output — fits all output columns simultaneously
- 🔢 SVD-based — handles rank-deficient and overdetermined systems gracefully

---

### 🔬 Sparse Iterative Thresholding — `lr_sparse_fit`

Recovers a **sparse** coefficient matrix by iteratively fitting regression on the current residual and adding only the top-$k$ largest-magnitude new coefficients per iteration.

**Algorithm:**

1. Initialise $\mathbf{A} = \mathbf{0}$, residual $\mathbf{r} = \mathbf{Y}$
2. Fit OLS on the residual: $\tilde{\mathbf{A}} = \text{lr\_fit}(\mathbf{X}, \mathbf{r})$
3. Zero out already-selected positions in $\tilde{\mathbf{A}}$
4. Select the top-$k$ largest $|\tilde{A}_{ij}|$ entries and add them to $\mathbf{A}$
5. Update residual: $\mathbf{r} = \mathbf{Y} - \mathbf{X}\mathbf{A}$
6. Repeat until convergence or `n_iter` reached

where $k = \max(1,\; \lfloor \text{density} \cdot n_{\text{inputs}} \cdot n_{\text{outputs}} \rceil)$.

| Parameter | Description | Default |
|---|---|---|
| `X` | Feature matrix `(n_samples, n_inputs)` | — |
| `Y` | Target matrix `(n_samples, n_outputs)` | — |
| `density` | Fraction of non-zero coefficients added per iteration | `0.01` |
| `n_iter` | Maximum iterations | `100` |
| `rel_tol` | Relative MSE improvement threshold for early stopping | `1e-6` |

```python
from AILibs.linear_regression import lr_sparse_fit

A_sparse = lr_sparse_fit(X, Y, density=0.01, n_iter=100)
Y_pred = X @ A_sparse
```

**Key features:**
- 🎯 Greedy forward selection — builds sparsity pattern incrementally
- 📉 Early stopping when relative MSE improvement drops below `rel_tol`
- 🔧 `density` controls the speed/sparsity trade-off — lower = sparser but slower
- 📊 Multi-output — sparse pattern is shared across all outputs

---

### 💎 Lasso Regression — `lr_lasso_fit`

**L1-regularised linear regression** solved via coordinate descent with soft-thresholding. Minimises:

$$\min_{\mathbf{A}} \;\frac{1}{2n}\|\mathbf{Y} - \mathbf{X}\mathbf{A}\|_F^2 \;+\; \lambda \|\mathbf{A}\|_1$$

Each coordinate update applies the **soft-thresholding operator**:

$$w_j \leftarrow \frac{S_{\lambda n}(\mathbf{x}_j^\top \mathbf{r}_j)}{\|\mathbf{x}_j\|_2^2}, \quad S_\tau(z) = \text{sign}(z)\max(|z| - \tau,\; 0)$$

where $\mathbf{r}_j$ is the partial residual with the $j$-th feature contribution added back.

| Parameter | Description | Default |
|---|---|---|
| `X` | Feature matrix `(n_samples, n_inputs)` | — |
| `Y` | Target matrix `(n_samples, n_outputs)` | — |
| `lambda_` | Regularisation strength (higher = sparser) | `1.0` |
| `n_iter` | Maximum coordinate descent iterations | `10000` |
| `rel_tol` | Convergence threshold on weight change | `1e-6` |

```python
from AILibs.linear_regression import lr_lasso_fit

A_lasso = lr_lasso_fit(X, Y, lambda_=0.5, n_iter=5000)
Y_pred = X @ A_lasso
```

**Key features:**
- 💎 True L1 penalty — drives coefficients exactly to zero
- 🔄 Coordinate descent — updates one weight at a time, scales to high dimensions
- 📉 Convergence check on $\|\Delta \mathbf{w}\|$ for reliable early stopping
- 🎛️ Single hyperparameter $\lambda$ controls the sparsity level

---

### ⚗️ SR3 — Sparse Relaxed Regularised Regression — `sr3_fit`

An **ADMM-based** sparse regression algorithm that introduces a relaxation variable $\mathbf{Z}$ to decouple the least-squares fit from the sparsity constraint:

$$\min_{\mathbf{A}, \mathbf{Z}} \;\frac{1}{2}\|\mathbf{Y} - \mathbf{X}\mathbf{A}\|_F^2 \;+\; \lambda\|\mathbf{Z}\|_1 \;+\; \frac{\rho}{2}\|\mathbf{A} - \mathbf{Z}\|_F^2$$

The ADMM iterations alternate between three updates:

$$\mathbf{A} \leftarrow (\mathbf{X}^\top\mathbf{X} + \rho\,\mathbf{I})^{-1}(\mathbf{X}^\top\mathbf{Y} + \rho(\mathbf{Z} - \mathbf{U}))$$

$$\mathbf{Z} \leftarrow S_{\lambda/\rho}(\mathbf{A} + \mathbf{U})$$

$$\mathbf{U} \leftarrow \mathbf{U} + \mathbf{A} - \mathbf{Z}$$

Convergence is checked via primal and dual residual norms. A final hard-threshold pass removes numerical dust below $0.1\,\lambda/\rho$.

| Parameter | Description | Default |
|---|---|---|
| `X` | Feature matrix `(n_samples, n_inputs)` | — |
| `Y` | Target matrix `(n_samples, n_outputs)` | — |
| `lambda_` | L1 regularisation strength | `1.0` |
| `rho` | ADMM penalty parameter | `1.0` |
| `n_iter` | Maximum ADMM iterations | `100` |
| `rel_tol` | Primal/dual residual convergence threshold | `1e-6` |

```python
from AILibs.linear_regression import sr3_fit

A_sr3 = sr3_fit(X, Y, lambda_=0.5, rho=1.0, n_iter=200)
Y_pred = X @ A_sr3
```

**Key features:**
- ⚗️ ADMM splitting — decouples regression from sparsity for faster convergence
- 🔧 Two-knob tuning: $\lambda$ (sparsity level) and $\rho$ (ADMM convergence speed)
- 🧹 Post-hoc hard thresholding removes residual ADMM noise
- 📊 Multi-output — all output columns solved simultaneously
- 📐 Primal + dual residual monitoring for principled convergence

---

## 📚 Dictionary Augmentation

The dictionary functions in `AILibs.common` transform a nonlinear problem into a linear one by **lifting** the input features into a higher-dimensional space of nonlinear basis terms:

$$\mathbf{x}_{\text{aug}} = \begin{bmatrix} \mathbf{x} & 1 & x_i^2 & x_i^3 & \cdots & x_i x_j & \cdots & \sin(k x_i) & \cos(k x_i) & \cdots \end{bmatrix}$$

After augmentation, any linear solver can capture nonlinear relationships by fitting $\mathbf{Y} = \mathbf{X}_{\text{aug}} \mathbf{A}$.

---

### 1️⃣ Constant — `dictionary_constant`

Appends a column of ones (bias / intercept term).

$$\phi(x) = 1$$

| Input Shape | Output Shape |
|---|---|
| `(n, d)` | `(n, 1)` |

```python
from AILibs.common import dictionary_constant

bias = dictionary_constant(x)   # shape (n_samples, 1), all ones
```

---

### 📊 Polynomial — `dictionary_polynomial`

Generates element-wise powers $x^2, x^3, \ldots, x^{\text{order}}$ for each feature independently:

$$\phi(x_i) = \{x_i^2, x_i^3, \ldots, x_i^p\}$$

| Parameter | Description |
|---|---|
| `x` | Input features `(n, d)` |
| `order` | Maximum polynomial degree $p$ (terms start at $x^2$) |

| Input Shape | Output Shape |
|---|---|
| `(n, d)` | `(n, d × (order − 1))` |

```python
from AILibs.common import dictionary_polynomial

poly = dictionary_polynomial(x, order=4)   # x², x³, x⁴ for each feature
```

> 💡 Powers start at 2 because $x^1$ is already present in the raw features.

---

### ✖️ Cross Products — `dictionary_cross_products`

Generates all unique pairwise products $x_i \cdot x_j$ for $i < j$:

$$\phi(x_i, x_j) = x_i \cdot x_j \quad \forall\; i < j$$

| Input Shape | Output Shape |
|---|---|
| `(n, d)` | `(n, \binom{d}{2})` |

```python
from AILibs.common import dictionary_cross_products

cross = dictionary_cross_products(x)   # x₀x₁, x₀x₂, x₁x₂, ...
```

---

### 🌊 Sin/Cos Harmonics — `dictionary_sin_cos`

Generates sine and cosine harmonics up to the $n$-th harmonic for each feature:

$$\phi_k(x_i) = \{\sin(k\,x_i),\; \cos(k\,x_i)\} \quad k = 1, \ldots, n$$

| Parameter | Description |
|---|---|
| `x` | Input features `(n, d)` |
| `n_harmonics` | Number of harmonic frequencies |

| Input Shape | Output Shape |
|---|---|
| `(n, d)` | `(n, 2 × d × n_harmonics)` |

```python
from AILibs.common import dictionary_sin_cos

harmonics = dictionary_sin_cos(x, n_harmonics=3)   # sin(x), cos(x), sin(2x), cos(2x), sin(3x), cos(3x)
```

> 🌊 This is a truncated Fourier basis — excellent for periodic or oscillatory signals.

---

### 🔀 Sin/Cos Cross Terms — `dictionary_sin_cos_cross`

Generates mixed trigonometric interaction terms $x_a \sin(x_b)$ and $x_a \cos(x_b)$ for all pairs $a \neq b$:

$$\phi(x_a, x_b) = \{x_a \sin(x_b),\; x_a \cos(x_b)\} \quad \forall\; a \neq b$$

These terms naturally arise in the equations of motion for mechanical and dynamical systems (e.g., pendulums, robot arms, orbital mechanics).

| Input Shape | Output Shape |
|---|---|
| `(n, d)` | `(n, 2 × d × (d − 1))` |

```python
from AILibs.common import dictionary_sin_cos_cross

cross_trig = dictionary_sin_cos_cross(x)   # x₀sin(x₁), x₀cos(x₁), x₁sin(x₀), ...
```

---

### 🧱 Building an Augmented Feature Matrix

Dictionary functions are designed to be **concatenated** freely. The typical workflow:

```python
import numpy as np
from AILibs.common import (
    dictionary_constant,
    dictionary_polynomial,
    dictionary_cross_products,
    dictionary_sin_cos,
)
from AILibs.linear_regression import lr_sparse_fit

# Raw features: (200, 3)
x = np.random.randn(200, 3)

# Build augmented feature matrix
x_aug = np.concatenate([
    x,                                    # 3 cols  — raw features
    dictionary_constant(x),               # 1 col   — bias
    dictionary_polynomial(x, order=3),    # 6 cols  — x², x³
    dictionary_cross_products(x),         # 3 cols  — x₀x₁, x₀x₂, x₁x₂
    dictionary_sin_cos(x, n_harmonics=2), # 12 cols — sin/cos harmonics
], axis=1)
# Total: 25 augmented features

# Sparse regression finds the few terms that matter
A = lr_sparse_fit(x_aug, y, density=0.05)
```

---

## 📋 API Reference

### Regression Functions

| Function | Inputs | Returns | Type |
|---|---|---|---|
| `lr_fit(X, Y)` | Features, targets | `A` | Closed-form OLS |
| `lr_sparse_fit(X, Y, density, n_iter, rel_tol)` | Features, targets, sparsity params | `A` (sparse) | Iterative greedy |
| `lr_lasso_fit(X, Y, lambda_, n_iter, rel_tol)` | Features, targets, L1 penalty | `A` (sparse) | Coordinate descent |
| `sr3_fit(X, Y, lambda_, rho, n_iter, rel_tol)` | Features, targets, ADMM params | `A` (sparse) | ADMM |

### Dictionary Functions

| Function | Inputs | Output Shape |
|---|---|---|
| `dictionary_constant(x)` | `(n, d)` | `(n, 1)` |
| `dictionary_polynomial(x, order)` | `(n, d)`, int | `(n, d × (order−1))` |
| `dictionary_cross_products(x)` | `(n, d)` | `(n, d(d−1)/2)` |
| `dictionary_sin_cos(x, n_harmonics)` | `(n, d)`, int | `(n, 2dn)` |
| `dictionary_sin_cos_cross(x)` | `(n, d)` | `(n, 2d(d−1))` |

### Import

```python
from AILibs.linear_regression import (
    lr_fit,
    lr_sparse_fit,
    lr_lasso_fit,
    sr3_fit,
)

from AILibs.common import (
    dictionary_constant,
    dictionary_polynomial,
    dictionary_cross_products,
    dictionary_sin_cos,
    dictionary_sin_cos_cross,
)
```

---

## 🗺️ Applications

### 📐 Regression Solvers

| Application Domain | Recommended Method | Why |
|---|---|---|
| 🔧 **System Identification** | `lr_fit` | Recover $\dot{x} = Ax + Bu$ matrices from state/input trajectories — fast, exact for linear systems |
| 🧬 **Genomics & Biomarker Discovery** | `lr_lasso_fit` | Thousands of gene features, few relevant — L1 penalty selects the active genes |
| 🔭 **Equation Discovery (SINDy)** | `lr_sparse_fit` | Identify governing equations from data — sparse $A$ reveals which dictionary terms appear in the dynamics |
| 📡 **Compressed Sensing** | `sr3_fit` | Recover sparse signals from underdetermined measurements — ADMM handles large, ill-conditioned problems |
| 🏗️ **Structural Load Modelling** | `lr_fit` + `dictionary_polynomial` | Augment sensor data with polynomial terms, then fit — captures nonlinear stress-strain relationships |
| 🔋 **Battery Degradation** | `lr_sparse_fit` + `dictionary_sin_cos` | Identify dominant degradation modes from cycling data — sparse fit isolates the few relevant harmonics |
| 📈 **Financial Factor Models** | `lr_lasso_fit` | Select the handful of economic factors that explain asset returns from a large candidate set |
| 🤖 **Robot Dynamics** | `sr3_fit` + `dictionary_sin_cos_cross` | Learn Lagrangian dynamics with $q\sin(q)$ and $q\cos(q)$ coupling terms |

### 📚 Dictionary Augmentation

| Dictionary | Natural Fit |
|---|---|
| `dictionary_polynomial` | Polynomial relationships, Taylor-series approximations, material stress-strain curves |
| `dictionary_cross_products` | Feature interactions, multi-variable coupling effects, ANOVA-style modelling |
| `dictionary_sin_cos` | Periodic phenomena — vibrations, seasonal patterns, oscillatory dynamics |
| `dictionary_sin_cos_cross` | Mechanical systems with rotational coupling — pendulums, robotic joints, satellites |
| `dictionary_constant` | Always include — captures offsets and steady-state biases |

---

## ⚖️ Method Comparison

| | `lr_fit` | `lr_sparse_fit` | `lr_lasso_fit` | `sr3_fit` |
|---|---|---|---|---|
| **Solution type** | Dense | Sparse (greedy) | Sparse (L1) | Sparse (ADMM) |
| **Solver** | SVD pseudoinverse | Iterative OLS + thresholding | Coordinate descent | ADMM splitting |
| **Sparsity control** | — | `density` | `lambda_` | `lambda_`, `rho` |
| **Iterations** | 0 (closed-form) | `n_iter` (default 100) | `n_iter` (default 10000) | `n_iter` (default 100) |
| **Convergence** | Exact | Early stop on MSE | Early stop on $\|\Delta w\|$ | Primal + dual residuals |
| **Multi-output** | ✅ simultaneous | ✅ simultaneous | ✅ column-by-column | ✅ simultaneous |
| **Best for** | Full-rank, well-conditioned | Equation discovery, SINDy | High-dimensional feature selection | Large sparse recovery, compressed sensing |

### When to Use What

```
            ┌──────────────────────────┐
            │  Is the true model sparse │
            │  (few active terms)?      │
            └────────┬─────────────────┘
                 No  │  Yes
                 │   │
          ┌──────▼┐  └──────────────────────────┐
          │lr_fit │  ┌──────────────────────────┐│
          │(OLS)  │  │ Do you know the sparsity ││
          └───────┘  │ pattern approximately?   ││
                     └────────┬────────────┬────┘│
                          Yes │            │ No   │
                              │            │      │
                    ┌─────────▼──┐   ┌─────▼─────┐
                    │lr_sparse_fit│   │ How many  │
                    │(top-k greedy)│  │ features? │
                    └─────────────┘  └──┬─────┬──┘
                                  Few   │     │ Many
                                        │     │
                                 ┌──────▼┐  ┌─▼────────┐
                                 │sr3_fit│  │lr_lasso_fit│
                                 │(ADMM) │  │(coord desc)│
                                 └───────┘  └───────────┘
```

---

<p align="center">
  <sub>Sparse, dense, polynomial, trigonometric — if it's linear in the parameters, we solve it. 📈</sub>
</p>