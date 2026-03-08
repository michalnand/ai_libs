<p align="center">
  <h1 align="center">🌲 AILibs.forest</h1>
  <p align="center">
    <strong>Tree-based ensemble methods built from scratch in pure NumPy</strong>
  </p>
  <p align="center">
    <em>Isolation forests for anomaly detection, random forests for regression, gradient boosting for precision — all transparent.</em>
  </p>
  <p align="center">
    <a href="#-overview">Overview</a> •
    <a href="#-algorithms">Algorithms</a> •
    <a href="#-api-reference">API Reference</a> •
    <a href="#-applications">Applications</a>
  </p>
</p>

---

## ✨ Overview

The `AILibs.forest` module provides tree-based ensemble algorithms implemented entirely from scratch. Every split, every tree, every score — built transparently with only NumPy.

| | Algorithm | Task | Key Idea |
|---|---|---|---|
| 🌲 | **IsolationForest** | Anomaly Detection | Anomalies isolate faster → shorter paths |
| 🌳 | **ExtendedIsolationForest** | Anomaly Detection | Random hyperplane splits for better high-dimensional isolation |
| 🌿 | **RandomDecisionTree** | Regression | Single tree with MSE-based candidate split selection |
| 🏔️ | **RandomForest** | Regression | Bagged ensemble of random decision trees |
| 🚀 | **RandomBoostingForest** | Regression | Gradient-boosted sequential ensemble of trees |

---

## 🔬 Algorithms

### 🌲 Isolation Forest — `IsolationForest`

Unsupervised anomaly detection based on the principle that anomalies are *few and different*, leading to shorter isolation path lengths in random binary trees.

At each internal node a **single feature** is chosen at random and a threshold is drawn uniformly between the feature's min and max. Points that are outliers land in small partitions quickly, producing short root-to-leaf paths.

The anomaly score normalises the average path length $E[h(x)]$ by $c(n)$, the expected path length of unsuccessful searches in a Binary Search Tree:

$$s(x, n) = 2^{\;-\,E[h(x)]\,/\,c(n)}$$

$$c(n) = 2\,H(n{-}1) - \frac{2(n{-}1)}{n}, \quad H(i) \approx \ln(i) + \gamma$$

where $\gamma = 0.5772\ldots$ is the Euler–Mascheroni constant.

| Score | Interpretation |
|---|---|
| $s \to 1$ | **Anomaly** — isolated very quickly |
| $s \approx 0.5$ | **Normal** — average path length |
| $s \to 0$ | **Dense region** — hard to isolate |

| Parameter | Description | Default |
|---|---|---|
| `max_depth` | Maximum tree depth (controls resolution) | — |
| `num_trees` | Number of isolation trees in the ensemble | `32` |
| `num_subsamples` | Subsample size per tree (`-1` = use all data) | `-1` |
| `eps` | Minimum feature range to allow a split | `0.001` |

```python
from AILibs.forest import IsolationForest

iforest = IsolationForest()
iforest.fit(x_train, max_depth=12, num_trees=128, num_subsamples=4096)
scores = iforest.predict(x_test)   # shape (n_samples,), values in (0, 1]
```

**Key features:**
- 🎲 Axis-aligned random splits — one feature per node
- ⚡ Linear-time tree construction, sublinear scoring
- 📏 Score normalisation via BST path-length theory
- 🔀 Subsampling for scalability on large datasets

---

### 🌳 Extended Isolation Forest — `ExtendedIsolationForest`

An improved variant that replaces axis-aligned splits with **random hyperplane splits**. Instead of choosing a single feature, each node projects the data onto a random direction vector $\mathbf{v}$ and splits along the projected value:

$$p = \mathbf{x} \cdot \mathbf{v}, \quad \text{split if } p < \theta$$

This eliminates the bias towards axis-aligned artifacts that the standard Isolation Forest can exhibit, producing more faithful anomaly contours in high-dimensional or correlated feature spaces.

**Sparse random projections:** To keep each split interpretable and efficient, the projection vector $\mathbf{v}$ is randomly sparsified — each coordinate is zeroed out with probability $1 - \sqrt{D}/D$, where $D$ is the number of features. This means each split focuses on a small random subset of dimensions.

| Parameter | Description | Default |
|---|---|---|
| `max_depth` | Maximum tree depth | — |
| `num_trees` | Number of isolation trees | `32` |
| `num_subsamples` | Subsample size per tree (`-1` = use all) | `-1` |
| `eps` | Minimum projected range to allow a split | `0.001` |

```python
from AILibs.forest import ExtendedIsolationForest

eiforest = ExtendedIsolationForest()
eiforest.fit(x_train, max_depth=12, num_trees=128, num_subsamples=4096)
scores = eiforest.predict(x_test)   # shape (n_samples,), values in (0, 1]
```

**Key features:**
- 📐 Random hyperplane splits — not constrained to axis alignment
- 🎯 Better isolation contours for correlated and high-dimensional data
- 🔀 Sparse projections — $O(\sqrt{D})$ active coordinates per split
- 📏 Same $c(n)$ normalisation and scoring as standard Isolation Forest

---

### 🌿 Random Decision Tree — `RandomDecisionTree`

The base building block for both `RandomForest` and `RandomBoostingForest`. A single regression tree that selects splits by evaluating `num_candidates` random (feature, threshold) pairs and keeping the one with the lowest total MSE:

$$\text{MSE}_{\text{split}} = \sum_{i \in L}(y_i - \bar{y}_L)^2 + \sum_{i \in R}(y_i - \bar{y}_R)^2$$

When `num_candidates=1`, the tree behaves as a purely random tree (Extra-Tree style). Increasing `num_candidates` adds greedy search quality at the cost of more computation.

| Parameter | Description | Default |
|---|---|---|
| `max_depth` | Maximum tree depth | — |
| `num_candidates` | Number of random splits evaluated per node | `1` |

```python
from AILibs.forest import RandomDecisionTree

tree = RandomDecisionTree()
tree.fit(x_train, y_train, max_depth=10, num_candidates=16)
y_pred = tree.predict(x_test[0])   # predict a single sample
```

**Key features:**
- 🎲 Random feature + threshold selection with optional MSE refinement
- 📊 Leaf nodes store the mean target value
- 🔄 Supports multi-output regression (target can be a vector)

---

### 🏔️ Random Forest — `RandomForest`

Ensemble regression via **bagging** (Bootstrap AGGregating). Each tree is trained independently on a (optionally subsampled) copy of the data, and predictions are averaged across all trees:

$$\hat{y}(x) = \frac{1}{T}\sum_{t=1}^{T} \text{tree}_t(x)$$

Averaging reduces variance while each individual tree provides low bias, yielding robust predictions even on noisy data.

| Parameter | Description | Default |
|---|---|---|
| `max_depth` | Maximum depth per tree | — |
| `num_trees` | Number of trees in the ensemble | `8` |
| `num_subsamples` | Subsample size per tree (`-1` = use all) | `-1` |
| `num_random_candidates` | Split candidates per node (passed to `RandomDecisionTree`) | `16` |

```python
from AILibs.forest import RandomForest

rf = RandomForest()
rf.fit(x_train, y_train, max_depth=12, num_trees=128, num_random_candidates=16)

y_single = rf.predict(x_test[0])        # single sample
y_batch  = rf.predict_batch(x_test)     # full batch, shape (n_samples, n_outputs)
```

**Key features:**
- 🔀 Bagging with optional subsampling for diversity
- 📉 Variance reduction through averaging — robust to noise
- 🎲 Each tree uses randomised split selection (configurable greediness)
- 📊 Supports multi-output regression

---

### 🚀 Random Boosting Forest — `RandomBoostingForest`

Ensemble regression via **gradient boosting**. Trees are trained *sequentially*, where each new tree fits the **residual error** of the current ensemble. A learning rate $\eta$ controls the contribution of each tree:

$$\hat{y}^{(0)} = \bar{y}$$

$$\hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta \cdot \text{tree}_t(x)$$

where $\text{tree}_t$ is fitted on the residual $r^{(t)} = y - \hat{y}^{(t-1)}$.

| Parameter | Description | Default |
|---|---|---|
| `max_depth` | Maximum depth per tree | — |
| `num_trees` | Number of boosting stages | `8` |
| `num_subsamples` | Subsample size per stage (`-1` = use all) | `-1` |
| `learning_rate` | Shrinkage factor $\eta$ (step size) | `0.1` |

```python
from AILibs.forest import RandomBoostingForest

rbf = RandomBoostingForest()
rbf.fit(x_train, y_train, max_depth=8, num_trees=200, learning_rate=0.1)

y_single = rbf.predict(x_test[0])
y_batch  = rbf.predict_batch(x_test)
```

**Key features:**
- 📈 Sequential residual fitting — each tree corrects previous errors
- 🎛️ Learning rate for regularisation — smaller $\eta$ + more trees = better generalisation
- 🏗️ Initial prediction set to the training mean $\bar{y}$
- 🔀 Optional subsampling per stage (stochastic gradient boosting)

---

## 📋 API Reference

### Anomaly Detection Interface

| Class | Method | Inputs | Returns |
|---|---|---|---|
| `IsolationForest` | `fit(x, max_depth, ...)` | Training data `(n, d)` | List of tree dicts |
| `IsolationForest` | `predict(x)` | Test data `(n, d)` | Scores `(n,)` in $(0, 1]$ |
| `ExtendedIsolationForest` | `fit(x, max_depth, ...)` | Training data `(n, d)` | List of tree dicts |
| `ExtendedIsolationForest` | `predict(x)` | Test data `(n, d)` | Scores `(n,)` in $(0, 1]$ |

### Regression Interface

| Class | Method | Inputs | Returns |
|---|---|---|---|
| `RandomDecisionTree` | `fit(x, y, max_depth, ...)` | Features `(n, d)`, targets `(n, k)` | — |
| `RandomDecisionTree` | `predict(x)` | Single sample `(d,)` | Prediction `(k,)` |
| `RandomForest` | `fit(x, y, max_depth, ...)` | Features `(n, d)`, targets `(n, k)` | — |
| `RandomForest` | `predict(x)` | Single sample `(d,)` | Prediction `(k,)` |
| `RandomForest` | `predict_batch(x)` | Batch `(n, d)` | Predictions `(n, k)` |
| `RandomBoostingForest` | `fit(x, y, max_depth, ...)` | Features `(n, d)`, targets `(n, k)` | — |
| `RandomBoostingForest` | `predict(x)` | Single sample `(d,)` | Prediction `(k,)` |
| `RandomBoostingForest` | `predict_batch(x)` | Batch `(n, d)` | Predictions `(n, k)` |

### Import

```python
from AILibs.forest import (
    IsolationForest,
    ExtendedIsolationForest,
    RandomDecisionTree,
    RandomForest,
    RandomBoostingForest,
)
```

---

## 🗺️ Applications

Tree-based ensembles are versatile workhorses across many domains. Below is a guide to where each algorithm in `AILibs.forest` fits naturally.

### 🌲 Isolation Forest & Extended Isolation Forest

| Application Domain | Description |
|---|---|
| 💳 **Fraud Detection** | Flag rare, unusual transactions in financial data by scoring each transaction's isolation path length |
| 🏭 **Industrial Fault Detection** | Detect sensor anomalies in manufacturing lines — abnormal readings isolate quickly |
| 🌐 **Network Intrusion Detection** | Identify malicious traffic patterns that deviate from normal network behaviour |
| 🏥 **Medical Outlier Screening** | Flag unusual patient records or lab results for manual review |
| 📊 **Data Quality & Cleaning** | Automatically identify corrupted, mislabelled, or erroneous entries in large datasets |
| 🛰️ **Satellite & Remote Sensing** | Detect anomalous spectral signatures or unusual land-cover changes |

> **When to choose Extended over Standard:**
> Use `ExtendedIsolationForest` when features are **correlated** or the anomaly boundary is **not axis-aligned** — the random hyperplane splits capture diagonal and curved isolation contours that axis-aligned splits miss.

### 🏔️ Random Forest

| Application Domain | Description |
|---|---|
| 🏠 **Property Valuation** | Predict real estate prices from heterogeneous tabular features (area, location, age, …) |
| 🌡️ **Sensor Fusion** | Combine multiple noisy sensor readings into a robust state estimate |
| ⚙️ **System Identification** | Learn input–output mappings of physical systems for simulation or model-based control |
| 📈 **Financial Forecasting** | Model nonlinear relationships in economic indicators for time-series prediction |
| 🧬 **Bioinformatics** | Gene expression regression — predict phenotype from high-dimensional genomic features |
| 🤖 **Robotics Calibration** | Map raw sensor values to calibrated physical quantities via nonlinear regression |

> **When to choose Random Forest:**
> Best when you want a **robust, low-variance** predictor and are willing to trade some bias. Works well out of the box with minimal tuning. Handles noisy targets gracefully due to averaging.

### 🚀 Random Boosting Forest

| Application Domain | Description |
|---|---|
| 🎯 **High-Accuracy Regression** | When every decimal of RMSE matters — boosting squeezes out residual error iteratively |
| 📉 **Residual Modelling** | Model the error of a physics-based simulator to create a hybrid physics + ML predictor |
| 🔋 **Battery State Estimation** | Predict state-of-charge or remaining useful life from voltage/current/temperature curves |
| 🏗️ **Structural Health Monitoring** | Detect degradation trends by modelling the expected vs. observed response of structures |
| 📡 **Signal Reconstruction** | Reconstruct clean signals from noisy or incomplete observations via sequential refinement |
| 🧪 **Experimental Design** | Build fast surrogate models of expensive simulations for optimisation and sensitivity analysis |

> **When to choose Boosting over Bagging:**
> Use `RandomBoostingForest` when the data is relatively clean and you want **maximum predictive accuracy**. Boosting corrects systematic errors that bagging cannot fix, but is more sensitive to noisy targets and requires tuning `learning_rate` × `num_trees`.

---

## ⚖️ Algorithm Comparison

| | IsolationForest | ExtendedIsolationForest | RandomForest | RandomBoostingForest |
|---|---|---|---|---|
| **Task** | Anomaly detection | Anomaly detection | Regression | Regression |
| **Training** | Unsupervised | Unsupervised | Supervised | Supervised |
| **Ensemble** | Independent trees | Independent trees | Bagged (parallel) | Boosted (sequential) |
| **Split type** | Axis-aligned random | Random hyperplane | MSE-optimised random | MSE-optimised random |
| **Bias** | — | — | Moderate | Low |
| **Variance** | — | — | Low | Moderate |
| **Noise tolerance** | High | High | High | Moderate |
| **Key hyperparameter** | `num_subsamples` | `num_subsamples` | `num_random_candidates` | `learning_rate` |

---

<p align="center">
  <sub>Every tree grown by hand — no sklearn, no XGBoost, just NumPy. 🌲</sub>
</p>