# Dictionary – Nonlinear Feature Augmentation

The `dictionary` module provides **feature lifting** (augmentation) functions that map an input matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$ into a higher-dimensional space by appending nonlinear terms. This transforms a nonlinear regression problem into a **linear** one, solvable with ordinary or sparse least-squares.

The idea comes from the **dictionary of functions** approach: given raw features $\mathbf{x}$, construct an augmented feature vector

$$\mathbf{x}_{\text{aug}} = \begin{bmatrix} \mathbf{x} & 1 & \phi_1(\mathbf{x}) & \phi_2(\mathbf{x}) & \cdots \end{bmatrix}$$

and then solve the linear system $\mathbf{Y} = \mathbf{X}_{\text{aug}} \mathbf{A}$.

---

## Available Functions

| Function | Description | Output columns |
|---|---|---|
| `dictionary_constant(x)` | Appends a bias / intercept column of ones | 1 |
| `dictionary_polynomial(x, order)` | Powers $x^2, x^3, \dots, x^{\text{order}}$ for every feature | $D \cdot (\text{order} - 1)$ |
| `dictionary_cross_products(x)` | Unique pairwise products $x_i \cdot x_j$ for $i < j$ | $\binom{D}{2}$ |
| `dictionary_sin_cos(x, n_harmonics)` | $\sin(kx),\;\cos(kx)$ for $k = 1 \dots n$ for every feature | $2 \cdot D \cdot n$ |
| `dictionary_sin_cos_cross(x)` | Cross terms $x_a \sin(x_b),\; x_a \cos(x_b)$ for all $a \neq b$ | $2 \cdot D \cdot (D-1)$ |

All functions accept a 2-D NumPy array of shape `(batch_size, num_features)` and return a 2-D array that can be concatenated with the original features.

---

## Usage

### Basic – polynomial regression

```python
import numpy
import AILibs

x = numpy.random.randn(1000, 5)          # raw features

# lift into polynomial space
x_poly  = AILibs.common.dictionary.dictionary_polynomial(x, order=3)
x_const = AILibs.common.dictionary.dictionary_constant(x)

x_aug = numpy.concatenate([x, x_const, x_poly], axis=1)

# fit linear model on the augmented features
a = AILibs.linear_regression.lr_fit(x_aug, y)
```

### Stacking multiple dictionaries

Dictionaries can be **stacked** (concatenated) to build richer feature spaces. This is the key design principle — each function produces an independent block of features, and you compose the augmentation you need:

```python
x = numpy.random.randn(1000, 11)

x_poly   = AILibs.common.dictionary.dictionary_polynomial(x, order=3)
x_cross  = AILibs.common.dictionary.dictionary_cross_products(x)
x_sincos = AILibs.common.dictionary.dictionary_sin_cos(x, n_harmonics=5)
x_const  = AILibs.common.dictionary.dictionary_constant(x)

x_aug = numpy.concatenate([x, x_const, x_poly, x_cross, x_sincos], axis=1)
```

Because stacking many terms creates a wide (often over-determined) matrix, it pairs naturally with **sparse regression** (`lr_sparse_fit`), which recovers only the truly active coefficients:

```python
a_est = AILibs.linear_regression.lr_sparse_fit(x_aug, y)
```

### Evaluating the fit

```python
y_pred  = x_aug @ a_est
metrics = AILibs.metrics.regression_evaluation(y, y_pred)
print(AILibs.metrics.format_metrics(metrics))
```

---

## Design Principles

1. **Composable** – each function returns one block of features; combine them freely with `numpy.concatenate`.
2. **Stateless** – pure functions, no fitting or state to manage.
3. **Sparse-friendly** – the augmented matrix is intentionally high-dimensional; use `lr_sparse_fit` to automatically select only the relevant terms.
4. **Interpretable** – because the dictionary terms are known analytic functions, the resulting sparse coefficient matrix $\mathbf{A}$ directly tells you which nonlinear relationships are present in the data.

---

## Real-World Systems & Which Dictionaries to Use

The dictionary approach is especially powerful for **system identification** — learning governing equations from data. Below are classic systems and the dictionary terms that naturally appear in their dynamics.

### 1. Lorenz Attractor

The Lorenz system is one of the most studied chaotic dynamical systems:

$$\dot{x} = \sigma(y - x), \quad \dot{y} = x(\rho - z) - y, \quad \dot{z} = xy - \beta z$$

The right-hand side contains **linear terms** ($x$, $y$, $z$) and **cross products** ($xz$, $xy$). A suitable dictionary:

```python
x_cross = AILibs.common.dictionary.dictionary_cross_products(x)
x_const = AILibs.common.dictionary.dictionary_constant(x)
x_aug   = numpy.concatenate([x, x_const, x_cross], axis=1)
```

Sparse regression on $\dot{\mathbf{x}} = \mathbf{X}_{\text{aug}} \mathbf{A}$ will recover exactly the six non-zero Lorenz coefficients — this is the core idea behind **SINDy** (Sparse Identification of Nonlinear Dynamics).

### 2. DC Motor (Electromechanical System)

A DC motor with state $[\omega, i]^T$ (angular velocity, current) obeys:

$$\dot{\omega} = -\frac{b}{J}\omega + \frac{K_t}{J}i, \quad \dot{i} = -\frac{K_e}{L}\omega - \frac{R}{L}i + \frac{1}{L}u$$

This is a **linear system** — only a constant term and the raw features are needed:

```python
x_const = AILibs.common.dictionary.dictionary_constant(x)
x_aug   = numpy.concatenate([x, x_const], axis=1)
```

### 3. Pendulum / Nonlinear Oscillators

A simple pendulum: $\ddot{\theta} = -\frac{g}{l}\sin\theta$, or with damping and drive:

$$\dot{\theta} = \omega, \quad \dot{\omega} = -\frac{g}{l}\sin\theta - c\,\omega$$

The sinusoidal nonlinearity calls for the **sin/cos dictionary**:

```python
x_sincos = AILibs.common.dictionary.dictionary_sin_cos(x, n_harmonics=3)
x_const  = AILibs.common.dictionary.dictionary_constant(x)
x_aug    = numpy.concatenate([x, x_const, x_sincos], axis=1)
```

This also works for **Van der Pol oscillators**, **Duffing oscillators**, and any system with trigonometric terms.

### 4. Coupled Oscillators / Attitude Dynamics

Rigid-body rotation (e.g. satellite attitude, gyroscopes) produces equations like:

$$\dot{\omega}_1 = \frac{(I_2 - I_3)}{I_1}\,\omega_2\,\omega_3$$

where angular velocities multiply each other **and** interact with trigonometric terms of Euler angles. The **sin_cos_cross** dictionary captures exactly this:

```python
x_sc_cross = AILibs.common.dictionary.dictionary_sin_cos_cross(x)
x_cross    = AILibs.common.dictionary.dictionary_cross_products(x)
x_const    = AILibs.common.dictionary.dictionary_constant(x)
x_aug      = numpy.concatenate([x, x_const, x_cross, x_sc_cross], axis=1)
```

### 5. Fluid Dynamics / Navier–Stokes (Reduced-Order)

POD/Galerkin projections of the Navier–Stokes equations yield systems with **quadratic interactions** between modal coefficients:

$$\dot{a}_i = \sum_j L_{ij}\,a_j + \sum_{j,k} Q_{ijk}\,a_j\,a_k$$

A polynomial + cross-product dictionary is the natural fit:

```python
x_poly  = AILibs.common.dictionary.dictionary_polynomial(x, order=2)
x_cross = AILibs.common.dictionary.dictionary_cross_products(x)
x_const = AILibs.common.dictionary.dictionary_constant(x)
x_aug   = numpy.concatenate([x, x_const, x_poly, x_cross], axis=1)
```

### 6. Chemical Reaction Networks

Mass-action kinetics produce polynomial rate laws, e.g. $\dot{c}_A = -k\,c_A\,c_B$. Higher-order reactions may need cubic terms:

```python
x_poly  = AILibs.common.dictionary.dictionary_polynomial(x, order=3)
x_cross = AILibs.common.dictionary.dictionary_cross_products(x)
x_const = AILibs.common.dictionary.dictionary_constant(x)
x_aug   = numpy.concatenate([x, x_const, x_poly, x_cross], axis=1)
```

### Quick Reference

| System | Key nonlinearities | Recommended dictionaries |
|---|---|---|
| Lorenz / Rössler / Chen | $xy$, $xz$ | `cross_products` |
| DC motor, mass-spring | linear | `constant` only |
| Pendulum, oscillators | $\sin\theta$, $\cos\theta$ | `sin_cos` |
| Rigid body, gyroscope | $\omega_i \omega_j$, $\omega\sin\theta$ | `cross_products` + `sin_cos_cross` |
| Navier–Stokes (ROM) | $a_i a_j$ | `polynomial(2)` + `cross_products` |
| Chemical kinetics | $c_A^2 c_B$, $c_A c_B$ | `polynomial(3)` + `cross_products` |
| Unknown / exploratory | all of the above | stack everything, let sparse regression select |

> **Tip:** When in doubt, stack all dictionaries and rely on `lr_sparse_fit` to zero out irrelevant terms. The sparse solver acts as automatic model selection.

---

## Worked Example (from tests)

The test `test_nonlinear_regression.py` demonstrates the full pipeline:

```python
# 1. Generate raw features
x = rng.standard_normal((1000, 11))

# 2. Build augmented feature matrix by stacking dictionaries
x_tmp   = AILibs.common.dictionary.dictionary_sin_cos_cross(x)
x_const = AILibs.common.dictionary.dictionary_constant(x)
x_aug   = numpy.concatenate([x, x_const, x_tmp], axis=1)

# 3. Create a sparse ground-truth coefficient matrix
a = rng.standard_normal((x_aug.shape[1], 7))
mask = rng.random(a.shape) < 0.8        # 80 % sparsity
a[mask] = 0.0

# 4. Generate targets
y = x_aug @ a

# 5. Recover coefficients with sparse linear regression
a_est = AILibs.linear_regression.lr_sparse_fit(x_aug, y)

# 6. Verify
assert numpy.allclose(a_est, a, atol=1e-6)
```

The same pattern is repeated for every dictionary type (`polynomial`, `cross_products`, `sin_cos`, `sin_cos_cross`), confirming that the lifting + sparse solve pipeline recovers the true model.
