<p align="center">
  <h1 align="center">🎮 AILibs.control</h1>
  <p align="center">
    <strong>Classical & modern control algorithms in pure NumPy</strong>
  </p>
  <p align="center">
    <em>From PID to Model Predictive Control — transparent implementations for learning, simulation, and deployment.</em>
  </p>
  <p align="center">
    <a href="#-overview">Overview</a> •
    <a href="#-controllers">Controllers</a> •
    <a href="#-api-reference">API Reference</a>
  </p>
</p>

---

## ✨ Overview

The `AILibs.control` module provides a collection of control algorithms implemented from scratch using only NumPy (and SciPy for Riccati solvers). Every controller is designed to be **readable**, **testable**, and **hackable**.

| | Controller | Type | Use Case |
|---|---|---|---|
| 🎚️ | **PIDTextbook** | Classical | Educational continuous-form PID |
| 🎛️ | **PID** | Classical | Discrete incremental PID with anti-windup & rate limiting |
| 🔄 | **LQRDiscrete** | Optimal | Full-state feedback minimizing a quadratic cost |
| 🔁 | **LQRIDiscrete** | Optimal | LQR with integral action for zero steady-state error |
| 📡 | **LQGDiscrete** | Optimal | LQR + Kalman observer for noisy partial observations |
| 🔮 | **MPCAnalytical** | Predictive | Closed-form receding-horizon MPC via pre-computed gain |
| ⚡ | **MPCFGM** | Predictive | MPC solved online via Fast Gradient Method (Nesterov) with box constraints |

---

## 🎛️ Controllers

### 🎚️ PID Textbook — `PIDTextbook`

A **textbook-style PID controller** in continuous form. Directly applies proportional, integral, and derivative terms without discretization safeguards. Primarily for **educational and illustrative purposes**.

$$u(t) = K_p \, e(t) \;+\; K_i \sum e(t) \;+\; K_d \, \Delta e(t)$$

| Parameter | Description |
|---|---|
| `kp` | Proportional gain |
| `ki` | Integral gain |
| `kd` | Derivative gain |

```python
from AILibs.control import PIDTextbook

pid = PIDTextbook(kp=2.0, ki=0.5, kd=0.1)

for t in range(1000):
    u = pid.forward(xr, x)    # xr = reference, x = measurement
```

> ⚠️ **Note:** This version lacks anti-windup, derivative filtering, and saturation limits. Use `PID` for practical applications.

---

### 🎛️ PID Discrete — `PID`

**Discrete-time PID controller** using the velocity (incremental) form — a difference equation that avoids derivative noise amplification and supports both **anti-windup** and **output rate limiting**.

The update law computes a control *increment* $\Delta u$ from three error samples:

$$u(n) = u(n{-}1) + k_0\,e(n) + k_1\,e(n{-}1) + k_2\,e(n{-}2)$$

where:

$$k_0 = K_p + K_i + K_d, \quad k_1 = -K_p - 2K_d, \quad k_2 = K_d$$

| Parameter | Description | Default |
|---|---|---|
| `kp` | Proportional gain | — |
| `ki` | Integral gain | — |
| `kd` | Derivative gain | — |
| `antiwindup` | Output saturation clamp | `10¹⁰` |
| `du_max` | Maximum control signal change per step (rate limit) | `10¹⁰` |

```python
from AILibs.control import PID

pid = PID(kp=2.0, ki=0.5, kd=0.1, antiwindup=10.0, du_max=1.0)

u = 0.0
for t in range(1000):
    u = pid.forward(xr, x, u)   # xr = reference, x = measurement, u = previous control
```

**Key features:**
- 🔒 Anti-windup via output saturation clamp
- 📉 Rate limiting on $\Delta u$ for actuator safety
- 🔄 Stateful — call `reset()` to clear error history
- ⚡ Incremental form avoids integral wind-up by construction

---

### 🔄 LQR Discrete — `LQRDiscrete`

**Linear Quadratic Regulator** for discrete-time systems. Computes the optimal full-state feedback gain $K$ that minimizes the infinite-horizon cost:

$$J = \sum_{n=0}^{\infty} \left[ \mathbf{x}^\top Q \, \mathbf{x} \;+\; \mathbf{u}^\top R \, \mathbf{u} \right]$$

The gain is obtained by solving the **Discrete Algebraic Riccati Equation (DARE)**:

$$K = (R + B^\top P B)^{-1} B^\top P A$$

The control law tracks a reference state:

$$u = K \, (x_r - x)$$

| Parameter | Description | Default |
|---|---|---|
| `a` | State transition matrix $(n \times n)$ | — |
| `b` | Control input matrix $(n \times m)$ | — |
| `q` | State cost matrix $(n \times n)$, positive semi-definite | — |
| `r` | Control cost matrix $(m \times m)$, positive definite | — |
| `antiwindup` | Output saturation clamp | `10¹⁰` |

```python
from AILibs.control import LQRDiscrete

lqr = LQRDiscrete(A, B, Q, R, antiwindup=5.0)

for t in range(1000):
    u = lqr.forward(xr, x)       # xr = desired state (n_states, 1), x = current state
    x = A @ x + B @ u
```

**Key features:**
- ⚡ Gain $K$ pre-computed at init — each `forward()` is $O(nm)$
- 🔒 Output saturation via `antiwindup` clamp
- 📐 Tuning via $Q/R$ ratio: large $Q$ → aggressive tracking, large $R$ → gentle control

---

### 🔁 LQRI Discrete — `LQRIDiscrete`

**LQR with Integral action** — augments the state with integral error accumulators to achieve **zero steady-state tracking error**, even under constant disturbances or model mismatch.

The augmented system stacks the original state with its integral:

$$\begin{bmatrix} x(n+1) \\ z(n+1) \end{bmatrix} = \begin{bmatrix} A & 0 \\ I & I \end{bmatrix} \begin{bmatrix} x(n) \\ z(n) \end{bmatrix} + \begin{bmatrix} B \\ 0 \end{bmatrix} u(n)$$

The resulting control law splits into two terms:

$$u(n) = -K\,x(n) + K_i \cdot \text{integral\_action}(n)$$

| Parameter | Description | Default |
|---|---|---|
| `a` | State transition matrix $(n \times n)$ | — |
| `b` | Control input matrix $(n \times m)$ | — |
| `q` | State cost matrix $(n \times n)$ | — |
| `r` | Control cost matrix $(m \times m)$ | — |
| `antiwindup` | Output saturation clamp | `10¹⁰` |
| `aw_enabled` | Enable conditional anti-windup | `True` |

```python
from AILibs.control import LQRIDiscrete

lqri = LQRIDiscrete(A, B, Q, R, antiwindup=5.0)

integral_action = numpy.zeros((n_inputs, 1))

for t in range(1000):
    u, integral_action = lqri.forward(xr, x, integral_action)
    x = A @ x + B @ u
```

**Key features:**
- 🎯 Zero steady-state error via integral action
- 🔒 Conditional anti-windup — back-calculates the integral when output saturates
- 📐 Augmented DARE solution yields both $K$ and $K_i$ simultaneously

---

### 📡 LQG Discrete — `LQGDiscrete`

**Linear Quadratic Gaussian** — the separation principle in action. Combines an **LQR gain with integral action** for optimal control with a **Kalman observer** for optimal state estimation from noisy, partial observations:

$$\hat{x}(n{+}1) = A\,\hat{x}(n) + B\,u(n) + F\bigl(y(n) - C\,\hat{x}(n)\bigr)$$

$$u(n) = -K\,\hat{x}(n) + K_i \cdot \text{integral\_action}(n)$$

The Kalman gain $F$ is computed by solving DARE on the dual system $(A^\top, C^\top)$.

| Parameter | Description | Default |
|---|---|---|
| `a` | State transition matrix $(n \times n)$ | — |
| `b` | Control input matrix $(n \times m)$ | — |
| `c` | Observation matrix $(p \times n)$ | — |
| `q` | LQR state cost $(n \times n)$ | — |
| `r` | LQR control cost $(m \times m)$ | — |
| `noise_q` | Process noise covariance $(n \times n)$ | — |
| `noise_r` | Measurement noise covariance $(p \times p)$ | — |
| `antiwindup` | Output saturation clamp | `10¹⁰` |
| `di_max` | Maximum integral action change per step | `10¹⁰` |

```python
from AILibs.control import LQGDiscrete

lqg = LQGDiscrete(A, B, C, Q, R, noise_Q, noise_R, antiwindup=5.0)

integral_action = numpy.zeros((n_inputs, 1))
x_hat = numpy.zeros((n_states, 1))

for t in range(1000):
    u, integral_action, x_hat = lqg.forward(yr, y, integral_action, x_hat)
    x = A @ x + B @ u
    y = C @ x + noise
```

**Key features:**
- 📡 Handles noisy partial observations — no full-state access needed
- 🎯 Integral action for zero steady-state output error
- 🔒 Conditional anti-windup with integral back-calculation
- 📉 Integral rate limiting via `di_max`

---

### 🔮 MPC Analytical — `MPCAnalytical`

**Closed-form Model Predictive Control** — builds the prediction matrices and pre-computes the optimal gain offline. At each step, only a matrix-vector multiply is needed (no online optimization).

The controller minimizes:

$$\min_{\mathbf{U}} \;\sum_{k=1}^{H_p} \|\mathbf{x}_k - \mathbf{x}_k^r\|_Q^2 \;+\; \sum_{k=0}^{H_c-1} \|\mathbf{u}_k\|_R^2$$

The prediction model uses stacked matrices:

$$\mathbf{X} = \Phi\,x(n) + \Theta\,\mathbf{U}$$

where $\Phi$ stacks powers of $A$ and $\Theta$ is the lower-triangular convolution of $A^i B$. The unconstrained optimum is:

$$\mathbf{U}^* = \underbrace{({\Theta^\top \tilde{Q}\,\Theta + \tilde{R}})^{-1}\Theta^\top \tilde{Q}}_{\Sigma}\;(\mathbf{X}_r - \Phi\,x)$$

Only the first $n_u$ rows of $\Sigma$ are retained ($\Sigma_0$), giving a single pre-computed gain.

| Parameter | Description | Default |
|---|---|---|
| `A` | State transition matrix $(n_x \times n_x)$ | — |
| `B` | Control input matrix $(n_x \times n_u)$ | — |
| `Q` | State cost matrix $(n_x \times n_x)$ | — |
| `R` | Control cost matrix $(n_u \times n_u)$ | — |
| `prediction_horizon` | Number of predicted future states $H_p$ | `16` |
| `control_horizon` | Number of optimized future inputs $H_c$ ($\leq H_p$) | `4` |
| `u_max` | Box constraint on control output | `10¹⁰` |

```python
from AILibs.control import MPCAnalytical

mpc = MPCAnalytical(A, B, Q, R, prediction_horizon=20, control_horizon=5, u_max=1.0)

for t in range(1000):
    u = mpc.forward_traj(Xr, x)   # Xr = stacked reference trajectory (nx*Hp, 1)
    x = A @ x + B @ u
```

**Key features:**
- ⚡ Gain pre-computed at init — each `forward_traj()` is a single matrix-vector multiply
- 🔮 Prediction & control horizon decoupling ($H_c \leq H_p$) for reduced computation
- 📎 Box constraint on output via `numpy.clip`

---

### ⚡ MPC Fast Gradient Method — `MPCFGM`

**Online MPC via Nesterov's Fast Gradient Method** — solves the constrained QP at each timestep using an accelerated first-order method. Ideal when box constraints on $u$ are active and a closed-form solution is insufficient.

The same QP cost as `MPCAnalytical`:

$$\min_{\mathbf{U}} \;\mathbf{U}^\top H\,\mathbf{U} - 2\,h^\top\mathbf{U} \quad \text{s.t.} \;\; |u_i| \leq u_{\max}$$

where $H = \Theta^\top \tilde{Q}\,\Theta + \tilde{R}$ and $h = \Theta^\top \tilde{Q}\,(X_r - \Phi\,x)$.

The solver uses **Nesterov momentum** with step size $\alpha = 1/L$, where $L = 2\,\lambda_{\max}(H)$ is estimated via power iteration.

| Parameter | Description | Default |
|---|---|---|
| `A` | State transition matrix $(n_x \times n_x)$ | — |
| `B` | Control input matrix $(n_x \times n_u)$ | — |
| `Q` | State cost matrix $(n_x \times n_x)$ | — |
| `R` | Control cost matrix $(n_u \times n_u)$ | — |
| `prediction_horizon` | $H_p$ | `16` |
| `control_horizon` | $H_c$ | `4` |
| `u_max` | Box constraint on each control input | `10¹⁰` |

```python
from AILibs.control import MPCFGM

mpc = MPCFGM(A, B, Q, R, prediction_horizon=20, control_horizon=5, u_max=1.0)

for t in range(1000):
    u = mpc.forward_traj(Xr, x, iters=16)   # iters = FGM iterations per step
    x = A @ x + B @ u
```

**Key features:**
- 🚀 Nesterov-accelerated convergence — $O(1/k^2)$ rate
- 🔒 Hard box constraints via projected gradient (clip after each iteration)
- 🔧 Tunable iteration count — trade compute budget for solution quality
- 📐 Lipschitz constant $L$ estimated via power iteration (32 iterations)

---

## 📋 API Reference

### Interface Summary

| Controller | Method | Inputs | Returns |
|---|---|---|---|
| `PIDTextbook` | `forward(xr, x)` | reference, measurement | `u` |
| `PID` | `forward(xr, x, u_prev)` | reference, measurement, previous control | `u` |
| `LQRDiscrete` | `forward(xr, x)` | reference state, current state | `u` |
| `LQRIDiscrete` | `forward(xr, x, ia)` | reference, state, integral action | `u, ia_new` |
| `LQGDiscrete` | `forward(yr, y, ia, x_hat)` | reference output, measurement, integral action, state estimate | `u, ia_new, x_hat_new` |
| `MPCAnalytical` | `forward_traj(Xr, x)` | stacked reference trajectory, current state | `u` |
| `MPCFGM` | `forward_traj(Xr, x, iters)` | stacked reference trajectory, current state, solver iterations | `u` |

### Shapes Convention

All controllers use **column vectors** with shape `(n, 1)`:

| Symbol | Shape | Description |
|---|---|---|
| `x` | `(n_states, 1)` | System state |
| `u` | `(n_inputs, 1)` | Control signal |
| `y` | `(n_outputs, 1)` | System output (measurement) |
| `Xr` | `(n_states × Hp, 1)` | Stacked reference trajectory (MPC) |

### Import

```python
from AILibs.control import (
    PIDTextbook,
    PID,
    LQRDiscrete,
    LQRIDiscrete,
    LQGDiscrete,
    MPCAnalytical,
    MPCFGM,
)
```

---

## 🗺️ Controller Selection Guide

```
                    ┌──────────────────────┐
                    │  Do you have a model? │
                    └────────┬─────────────┘
                        No   │   Yes
                  ┌──────────┴──────────┐
                  │                     │
           ┌──────▼──────┐    ┌─────────▼─────────┐
           │  PID /      │    │ Full state         │
           │  PIDTextbook│    │ available?          │
           └─────────────┘    └───┬─────────┬──────┘
                               Yes│         │No
                                  │         │
                          ┌───────▼──┐  ┌───▼──────────┐
                          │ Need     │  │  LQGDiscrete  │
                          │ constraints? │ (observer +  │
                          └──┬────┬──┘  │  integral)   │
                          Yes│    │No   └──────────────┘
                             │    │
                      ┌──────▼┐  ┌▼───────────────┐
                      │  MPC  │  │ Steady-state    │
                      │       │  │ error matters?  │
                      └──┬──┬─┘  └──┬──────────┬──┘
                         │  │    Yes│          │No
                  ┌──────▼┐ │      │    ┌─────▼──────┐
                  │MPCFGM │ │  ┌───▼────┐│LQRDiscrete │
                  │(online)│ │  │LQRIDisc││            │
                  └────────┘ │  │(+ integ)│└────────────┘
                      ┌──────▼──────┐ └────────┘
                      │MPCAnalytical│
                      │(precomputed)│
                      └─────────────┘
```

---

<p align="center">
  <sub>Pure NumPy control — no MATLAB, no CasADi, just math. 🎮</sub>
</p>