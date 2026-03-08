## PID Controller Implementation

### **1. Overview**

This module implements two forms of **Proportional–Integral–Derivative (PID)** controllers for control system applications, following classical control theory conventions.

### **2. Classes**

#### `PIDTextbook`

* **Type:** Continuous-form, simple PID.
* **Equation:**
  [
  u(t) = K_p e(t) + K_i \int e(t) dt + K_d \frac{de(t)}{dt}
  ]
* **Use case:** Educational or conceptual understanding.
* **Limitations:**

  * Not discretized.
  * No saturation or anti-windup.
  * May produce unstable results in real digital systems.

#### `PID`

* **Type:** Discrete-time, incremental PID controller.
* **Difference equation:**
  [
  u(n+1) = u(n) + k_0 e(n) + k_1 e(n-1) + k_2 e(n-2)
  ]
* **Key Features:**

  * **Antiwindup:** Limits the accumulated control action (`antiwindup` parameter).
  * **Rate limiting:** Restricts maximum change per step (`du_max` parameter).
  * **Incremental form:** Reduces derivative noise and numerical instability.

### **3. Parameters**

| Parameter        | Description                                               |
| ---------------- | --------------------------------------------------------- |
| `kp`, `ki`, `kd` | Proportional, integral, and derivative gains              |
| `antiwindup`     | Maximum allowed control output (prevents integral windup) |
| `du_max`         | Maximum rate of control signal change per time step       |
| `xr`, `x`        | Reference and current system outputs                      |
| `u_prev`         | Previous control command (for incremental update)         |

### **4. Practical Usage Notes**

* For real-time control (e.g., embedded systems or simulators), always prefer the **discrete PID** implementation.
* Use `PID.reset()` when restarting control loops to avoid residual errors.
* Tune gains carefully—poor tuning may cause oscillations or instability.
* Combine with plant models (`n_states`, `n_inputs`) when embedding in larger control frameworks such as **LQR**, **MPC**, or **Kalman filter-based controllers**.
