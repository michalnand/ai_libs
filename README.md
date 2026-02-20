<p align="center">
  <h1 align="center">🧠 AILibs</h1>
  <p align="center">
    <strong>A from-scratch machine learning & signal processing toolkit in pure NumPy</strong>
  </p>
  <p align="center">
    <em>No black boxes — every algorithm implemented transparently for learning, research, and rapid prototyping.</em>
  </p>
  <p align="center">
    <a href="#-modules">Modules</a> •
    <a href="#-installation">Installation</a> •
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-examples">Examples</a> •
    <a href="#-testing">Testing</a> •
    <a href="#-project-structure">Project Structure</a>
  </p>
</p>

---

## ✨ Highlights

| | Feature | Description |
|---|---|---|
| 🌲 | **Isolation Forest** | Anomaly detection via random path-length isolation |
| 🌳 | **Random Forest & Boosting** | Ensemble decision-tree regression and gradient-boosted forests |
| 📈 | **Linear & Sparse Regression** | OLS, iterative sparse thresholding, and RANSAC-based robust estimation |
| 📐 | **Dictionary Augmentation** | Polynomial, cross-product, and sin/cos feature lifting for nonlinear regression |
| 📡 | **Kalman Filter** | Steady-state discrete Kalman filter for linear dynamical systems |
| 📊 | **Comprehensive Metrics** | Regression (R², RMSE, MAE), detection (F1, MCC, IoU), and anomaly (AUC-ROC, AUC-PR) evaluation |
| 📁 | **Dataset Utilities** | CSV loader and multi-dataset collator |
| 🧪 | **Fully Tested** | Reproducible synthetic test suites with seeded RNG (`seed=42`) |

---

## 📦 Modules

### 🌲 Forest — `AILibs.forest`

Tree-based ensemble methods built from scratch.

#### `IsolationForest`

Unsupervised anomaly detection based on the principle that anomalies are *few and different*, leading to shorter isolation path lengths in random binary trees.

$$s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}$$

where $E[h(x)]$ is the average path length and $c(n)$ is the average path length of unsuccessful BST searches, used for normalization.

```python
from AILibs.forest import IsolationForest

forest = IsolationForest()
forest.fit(x_train, max_depth=12, num_trees=128, num_subsamples=4096)
scores = forest.predict(x_test)  # scores ∈ (0, 1] — closer to 1 = anomaly
```

#### `RandomForest` & `RandomBoostingForest`

Ensemble regression via bagging (`RandomForest`) and gradient boosting (`RandomBoostingForest`) of random decision trees.

```python
from AILibs.forest import RandomForest, RandomBoostingForest

# Bagging
rf = RandomForest()
rf.fit(x_train, y_train, max_depth=12, num_trees=128)
y_pred = rf.predict_batch(x_test)

# Gradient Boosting
rbf = RandomBoostingForest()
rbf.fit(x_train, y_train, max_depth=12, num_trees=128, learning_rate=0.25)
y_pred = rbf.predict_batch(x_test)
```

---

### 📈 Linear Regression — `AILibs.linear_regression`

Closed-form and iterative methods for linear and sparse regression.

#### `lr_fit` — Ordinary Least Squares

Solves $\mathbf{Y} = \mathbf{X}\mathbf{A}$ via the pseudoinverse:

```python
from AILibs.linear_regression import lr_fit

A = lr_fit(X, Y)  # shape: (n_features, n_outputs)
```

#### `lr_sparse_fit` — Sparse Regression (Iterative Thresholding)

Recovers a sparse coefficient matrix by iteratively adding the top-$k$ largest-magnitude coefficients:

```python
from AILibs.linear_regression import lr_sparse_fit

A_sparse = lr_sparse_fit(X, Y, density=0.01, n_iter=100)
```

#### `ransac_estimate_A_B` — RANSAC System Identification

Robust estimation of state-space matrices $\mathbf{A}, \mathbf{B}$ in $x_{n+1} = Ax_n + Bu_n$ using RANSAC:

```python
from AILibs.linear_regression.ransac import ransac_estimate_A_B

A_hat, B_hat, inlier_mask = ransac_estimate_A_B(x, u, n_iter=1000, tol=1e-3)
```

---

### 📐 Dictionary — `AILibs.common`

**Feature lifting functions** that transform a nonlinear problem into a linear one by augmenting the input with nonlinear basis terms:

$$\mathbf{x}_{\text{aug}} = \begin{bmatrix} \mathbf{x} & 1 & \phi_1(\mathbf{x}) & \phi_2(\mathbf{x}) & \cdots \end{bmatrix}$$

| Function | Description | Output Columns |
|---|---|---|
| `dictionary_constant(x)` | Bias / intercept column of ones | $1$ |
| `dictionary_polynomial(x, order)` | Powers $x^2, x^3, \dots, x^{\text{order}}$ | $D \cdot (\text{order} - 1)$ |
| `dictionary_cross_products(x)` | Unique pairwise products $x_i \cdot x_j$ for $i < j$ | $\binom{D}{2}$ |
| `dictionary_sin_cos(x, n)` | $\sin(kx), \cos(kx)$ for $k = 1 \dots n$ | $2 \cdot D \cdot n$ |
| `dictionary_sin_cos_cross(x)` | Cross terms $x_a \sin(x_b), x_a \cos(x_b)$ for $a \neq b$ | $2 \cdot D \cdot (D-1)$ |

```python
import numpy as np
from AILibs.common import dictionary_polynomial, dictionary_constant

x_aug = np.concatenate([x, dictionary_constant(x), dictionary_polynomial(x, order=3)], axis=1)
A = lr_sparse_fit(x_aug, y, density=0.05)
```

---

### 📡 DSP — `AILibs.dsp`

#### `KalmanFilter`

Steady-state discrete Kalman filter for linear dynamical systems:

$$x(n+1) = A\,x(n) + B\,u(n)$$

$$y(n) = H\,x(n) + \text{noise}$$

The Kalman gain $K$ is pre-computed by solving the **Discrete Algebraic Riccati Equation (DARE)**, making each `step()` call $O(n^2)$:

```python
from AILibs.dsp import KalmanFilter

kf = KalmanFilter(A, B, H, q_noise, r_noise)
kf.reset()

for y_obs, u in zip(observations, controls):
    x_est = kf.step(y_obs, u)  # filtered state estimate
```

---

### 📊 Metrics — `AILibs.metrics`

A comprehensive evaluation toolkit returning JSON-serializable dictionaries.

#### Regression Evaluation

| Metric | Key |
|---|---|
| Mean Squared Error | `mse` |
| Root MSE | `rmse` |
| Mean Absolute Error | `mae` |
| Median Absolute Error | `medae` |
| Max Absolute Error | `max_ae` |
| Mean Absolute % Error | `mape` |
| R² (coefficient of determination) | `r2` |
| Adjusted R² | `adjusted_r2` |
| σ-interval MSE (1σ, 2σ, 3σ) | `mse_1sigma`, `mse_2sigma`, `mse_3sigma` |

```python
from AILibs.metrics import regression_evaluation

metrics = regression_evaluation(y_gt, y_pred, n_features=5)
```

#### Detection Evaluation

Binary classification metrics including accuracy, precision, recall, F1, MCC, specificity, balanced accuracy, IoU, and Dice coefficient.

```python
from AILibs.metrics import detection_evaluation

metrics = detection_evaluation(y_gt, y_pred, th=0.5)
```

#### Anomaly Evaluation

Combines threshold-independent metrics (AUC-ROC, AUC-PR, score distribution analysis) with threshold-dependent binary metrics:

```python
from AILibs.metrics import anomaly_evaluation, tune_threshold

th = tune_threshold(y_gt, scores, metric="f1", steps=100)
metrics = anomaly_evaluation(y_gt, scores, th=th)
```

> **`tune_threshold`** sweeps thresholds from 0→1 and returns the optimal value maximizing `f1`, `mcc`, or `youden`.

---

### 📁 Datasets — `AILibs.datasets`

#### `CSVDataset`

Loads a CSV file, auto-detects numeric columns, and exposes the data as a NumPy array:

```python
from AILibs.datasets import CSVDataset

dataset = CSVDataset("data.csv")
print(dataset.x.shape)  # (n_samples, n_numeric_features)
sample = dataset[42]
```

#### `DatasetCollator`

Combines multiple datasets into a single unified dataset with seamless indexing:

```python
from AILibs.datasets import DatasetCollator

collator = DatasetCollator([dataset_a, dataset_b, dataset_c])
print(len(collator))  # total samples across all datasets
sample = collator[100]
```

---

### 🧪 Test Data — `AILibs.test_data`

Synthetic data generators for testing and benchmarking:

| Generator | Description |
|---|---|
| `test_data_linear_dynamics(...)` | Linear dynamical system $\dot{x} = Ax$ with configurable states |
| `test_data_lorenz_attractor(...)` | Classic Lorenz attractor (chaotic 3D system, $\sigma$=10, $\rho$=28, $\beta$=8/3) |

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/michalnand/ai_libs.git
cd ai_libs

# Install in development mode
pip install -e libs/

# Install test dependencies
pip install -r tests/requirements.txt
```

### Requirements

- **Python** ≥ 3.8
- **NumPy** (core dependency)
- **SciPy** (for Kalman filter DARE solver)
- **Matplotlib** (for examples / notebooks)

---

## ⚡ Quick Start

### Anomaly Detection with Isolation Forest

```python
import numpy as np
from AILibs.forest import IsolationForest
from AILibs.metrics import anomaly_evaluation, tune_threshold

# Generate synthetic data
rng = np.random.default_rng(42)
x_normal  = rng.standard_normal((500, 5))
x_anomaly = rng.standard_normal((25, 5)) * 3 + 8

x_test = np.vstack([x_normal, x_anomaly])
y_gt   = np.array([0]*500 + [1]*25)

# Fit and predict
forest = IsolationForest()
forest.fit(x_normal, max_depth=10, num_trees=64)
scores = forest.predict(x_test)

# Evaluate
th = tune_threshold(y_gt, scores, metric="f1")
metrics = anomaly_evaluation(y_gt, scores, th=th)
print(f"AUC-ROC: {metrics['auc_roc']:.3f} | F1: {metrics['f1_score']:.3f}")
```

### Nonlinear Regression with Dictionary Augmentation

```python
import numpy as np
from AILibs.linear_regression import lr_sparse_fit
from AILibs.common import dictionary_polynomial, dictionary_constant
from AILibs.metrics import regression_evaluation

# Create polynomial data
rng = np.random.default_rng(42)
x = rng.uniform(-2, 2, (200, 3))
y = (x[:, 0:1]**2 + 0.5*x[:, 1:2]**3 - x[:, 2:3])

# Augment features with polynomial dictionary
x_aug = np.concatenate([x, dictionary_constant(x), dictionary_polynomial(x, order=3)], axis=1)

# Sparse fit
A = lr_sparse_fit(x_aug, y, density=0.05)
y_pred = x_aug @ A

metrics = regression_evaluation(y, y_pred)
print(f"R²: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.4f}")
```

### Kalman Filtering

```python
import numpy as np
from AILibs.dsp import KalmanFilter

# Simple 2-state system
dt = 0.01
A = np.array([[1, dt], [0, 1]])           # position + velocity
B = np.array([[0.5*dt**2], [dt]])          # acceleration input
H = np.array([[1, 0]])                     # observe position only
Q = np.eye(2) * 0.001                      # process noise
R = np.array([[0.1]])                      # measurement noise

kf = KalmanFilter(A, B, H, Q, R)

for t in range(100):
    u = np.array([[1.0]])                  # constant acceleration
    y_obs = np.array([[true_pos + noise]]) # noisy measurement
    x_est = kf.step(y_obs, u)
```

---

## 📓 Examples

Interactive Jupyter notebooks and Python scripts demonstrating real-world usage:

| Example | Description |
|---|---|
| [`examples/regression/01_linear_regression.ipynb`](examples/regression/01_linear_regression.ipynb) | Linear regression with OLS and evaluation |
| [`examples/regression/02_nonlinear_regression.ipynb`](examples/regression/02_nonlinear_regression.ipynb) | Nonlinear regression using dictionary augmentation |
| [`examples/regression/03_sparse_regression.ipynb`](examples/regression/03_sparse_regression.ipynb) | Sparse coefficient recovery with iterative thresholding |
| [`examples/forest/01_isolation_forest_anomaly_detection.ipynb`](examples/forest/01_isolation_forest_anomaly_detection.ipynb) | Isolation Forest anomaly detection walkthrough |
| [`examples/forest/02_random_boosting_forest_classification.ipynb`](examples/forest/02_random_boosting_forest_classification.ipynb) | Random Boosting Forest for classification tasks |
| [`examples/forest/anomaly_detection.py`](examples/forest/anomaly_detection.py) | Credit card fraud detection with Isolation Forest |
| [`examples/forest/random_forest.py`](examples/forest/random_forest.py) | Random Forest classification on tabular data |
| [`examples/forest/random_boosting_forest.py`](examples/forest/random_boosting_forest.py) | Gradient-boosted forest classification |

---

## 🧪 Testing

The project uses **pytest** with reproducible synthetic datasets (`seed=42`) and HTML report generation.

```bash
# Run all tests
pytest tests/ -v -s

# Run by category
pytest -m regression      # linear & nonlinear regression
pytest -m forest          # isolation forest & random forest
pytest -m dsp             # Kalman filter

# Generate HTML report
pytest tests/ --html=tests/reports/report.html --self-contained-html
```

### Test Coverage Overview

| Module | Tests | What's Verified |
|---|---|---|
| **Linear Regression** | 4 tests | Single/multi-output OLS, noisy data robustness, sparse coefficient recovery |
| **Nonlinear Regression** | 4 tests | Polynomial, cross-product, sin/cos, and sin/cos-cross dictionary fits |
| **Isolation Forest** | 5 tests | Gaussian blobs, multi-modal clusters, sinusoidal manifolds, correlated features, subsampling |
| **Random Forest** | 4 tests | Linear, polynomial, sinusoidal, and noisy targets |
| **Kalman Filter** | 8 tests | Noiseless/noisy estimation, partial observability, reset, shapes, convergence, gain limits |

---

## 🗂 Project Structure

```
ai_libs/
├── libs/
│   ├── pyproject.toml                  # Package configuration
│   └── AILibs/
│       ├── common/                     # Dictionary feature augmentation
│       │   └── dictionary.py
│       ├── datasets/                   # CSV loader & dataset collator
│       │   ├── csv_dataset.py
│       │   └── dataset_collator.py
│       ├── dsp/                        # Signal processing (Kalman filter)
│       │   └── kalman_filter.py
│       ├── forest/                     # Tree-based ensembles
│       │   ├── isolation_forest.py
│       │   └── random_forest.py
│       ├── linear_regression/          # OLS, sparse, RANSAC
│       │   ├── linear_regression.py
│       │   └── ransac.py
│       ├── metrics/                    # Evaluation metrics
│       │   ├── anomaly_evaluation.py
│       │   ├── detection_evaluation.py
│       │   ├── regression_evaluation.py
│       │   └── tune_threshold.py
│       └── test_data/                  # Synthetic data generators
│           └── dynamical_system.py
├── examples/
│   ├── forest/                         # Forest algorithm demos
│   └── regression/                     # Regression notebooks
└── tests/
    ├── pytest.ini                      # Test configuration
    ├── conftest.py                     # Shared fixtures (seeded RNG)
    ├── dsp/                            # Kalman filter tests
    ├── forest/                         # Forest tests
    └── linear_regression/              # Regression tests
```

---

## 📄 License

This project is provided as-is for educational and research purposes.

---

<p align="center">
  <sub>Built with ❤️ and pure NumPy — no sklearn, no PyTorch, just math.</sub>
</p>
