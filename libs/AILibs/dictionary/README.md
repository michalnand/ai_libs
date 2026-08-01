# AILibs Dictionary Engine: Architecture & Reference

The `AILibs` dictionary framework treats feature engineering as a **compositional computational graph**. Instead of writing monolithic data transformation functions, features are built modularly using graph nodes. Every dictionary node inherits from a shared base class (`BaseDictionary`) and can be stacked **serially** (pipeline) or **parallelly** (branching and concatenation).

---

## Main Idea of the Structured Output

The core philosophy of this architecture is **declarative feature graph composition** driven by recursion:

1. **Unified Base Class (`BaseDictionary`)**: All nodes share the same interface (`__call__`). When you call the terminal node of your graph, it automatically triggers a bottom-up recursive evaluation of all its upstream parents.
2. **Flexible Upstream Routing**:
* `upstream=None`: Treats raw input $x$ as its source.
* `upstream=SingleNode`: Creates a **serial pipeline** (the output of the upstream node feeds directly into the current node).
* `upstream=[Node1, Node2, ...]` or `Concatenate([Node1, Node2])`: Creates a **parallel branch** (evaluates all nodes independently on the same input and fuses their outputs horizontally via `numpy.hstack`).


3. **Inspectable & Extensible**: Because every node maintains references to its parents, the entire architecture is transparent and easy to serialize, debug, or visualize.

---

## Summary Table of Dictionaries

| Dictionary Name | Class Name | Exact Mathematical Equation / Operation | Description |
| --- | --- | --- | --- |
| **Identity** | `Identity` | $f(x) = x$ | Passes input features through unchanged. |
| **Constant** | `Constant` | $f(x) = \mathbf{1} \in \mathbb{R}^{N \times 1}$ | Generates a column of ones for intercept/bias terms. |
| **Polynomial** | `Polynomial` | $f(x) = \left[x^2, x^3, \dots, x^d\right]$ | Raises features to powers up to the specified `degree` ($d \ge 2$). |
| **Rational** | `Rational` | $f(x) = \text{sign}(x) \cdot \frac{1}{\vert{}x\vert{} + \epsilon}$ | Sign-preserving inverse mapping for decay/saturation laws. |
| **Rational Quadratic** | `RationalQuadratic` | $f(x) = \frac{1}{x^2 + \epsilon}$ | Inverse-square mapping for physical distance/gravity laws. |
| **Cross Terms** | `CrossTerms` | $f(x_i, x_j) = x_i x_j \quad (\forall i \le j)$ | Computes all pairwise feature interactions. |
| **Wave** | `Wave` | $f(x) = \bigcup_{k=1}^{n} \Big\{\sin(kx), \cos(kx)\Big\}$ | Generates trigonometric Fourier harmonics. |
| **Non-Linear** | `NonLinear` | $f(x) = \left[\text{sgn}(x), \vert{}x\vert{}, \text{ReLU}(x), \text{neg}(x), \tanh(x), x^2, x\vert{}x\vert{}\right]$ | Comprehensive bucket of activations and aerodynamic drag terms. |
| **Concatenate** | `Concatenate` | $f(x) = \big[ d_1(x) \parallel d_2(x) \parallel \dots \big]$ | Pure parallel stacking node that fuses multiple branches horizontally. |

---

## Example Usage: Building a Feature Graph

The code below demonstrates how to compose serial and parallel nodes using your exact syntax, culminating in a single callable graph terminal.

```python
import numpy as np
from AILibs.dictionaries import Identity, Constant, Polynomial, Wave, Concatenate, NonLinear

# 1. Define base root transformations
A = Identity()          # Raw features x
B = Constant()          # Bias column [1]

# 2. Serial Pipeline: C wraps A (first evaluates A, then raises to degree 3)
C = Polynomial(upstream=A, degree=3)

# 3. Parallel & Fused Graph: 
# Concatenate evaluates B, C, and Wave features in parallel on raw input, 
# then hstacks them together into a unified wide feature matrix.
GraphTerminal = Concatenate([
    B, 
    C, 
    Wave(upstream=A, n_harmonics=2),
    NonLinear(upstream=A)
])

# 4. Execution: Calling only the terminal node recursively runs the entire graph!
x_sample = np.random.standard_normal((100, 2))  # 100 samples, 2 features
z_features = GraphTerminal(x_sample)

print("Graph Architecture Structure:")
print(GraphTerminal)
print(f"\nInput shape: {x_sample.shape}")
print(f"Augmented Feature shape: {z_features.shape}")

```

---

## How to Create Your Own Custom Dictionary

To add a new custom mathematical transformation (for example, a **Gaussian / RBF kernel feature mapper** or a custom physical constraint), you only need to inherit from `BaseDictionary` and implement the `_transform(self, x)` method.

### Custom Dictionary Example: Absolute Logarithm / Log-Sum-Exp feature

```python
import numpy as np
from AILibs.dictionaries import BaseDictionary

class LogarithmicFeature(BaseDictionary):
    """
    Custom Dictionary: Computes natural log of absolute values: f(x) = ln(|x| + 1)
    Useful for heavy-tailed financial or physical scaling data.
    """
    def __init__(self, upstream=None, eps=1e-6):
        super().__init__(upstream)
        self.eps = eps

    def _transform(self, x):
        return np.log(np.abs(x) + self.eps)

    def __repr__(self):
        parent_str = f"upstream={self.upstream}" if self.upstream else ""
        return f"LogarithmicFeature({parent_str})"

# Usage of the custom dictionary:
custom_node = LogarithmicFeature(upstream=Identity())
x_test = np.array([[-10.0, 0.0], [5.0, -2.0]])
print(custom_node(x_test))

```