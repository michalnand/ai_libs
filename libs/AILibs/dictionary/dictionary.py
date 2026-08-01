import numpy
from .base_dictionary import BaseDictionary


class Identity(BaseDictionary):
    """1:1 mapping, returns the input features as-is."""
    def _transform(self, x):
        return x


class Concatenate(BaseDictionary):
    """
    Pure parallel stacking node. 
    Takes a list of dictionaries, evaluates them in parallel, and passes 
    the concatenated results downstream without any extra transformation.
    """
    def __init__(self, dictionaries):
        if not isinstance(dictionaries, (list, tuple)):
            raise TypeError("Concatenate expects a list or tuple of dictionaries.")
        super().__init__(upstream=dictionaries)

    def _transform(self, x):
        return x  # Pass-through after horizontal stacking

class Constant(BaseDictionary):
    """Returns a single column of ones (bias term)."""
    def _transform(self, x):
        return numpy.ones((x.shape[0], 1))


class Polynomial(BaseDictionary):
    """Raises features to powers from 2 up to the specified degree."""
    def __init__(self, upstream=None, degree=3):
        super().__init__(upstream)
        self.degree = degree
        if degree < 2:
            raise ValueError("Polynomial degree should be >= 2 (use Identity for degree 1).")

    def _transform(self, x):
        features = [x**p for p in range(2, self.degree + 1)]
        return numpy.hstack(features) if features else numpy.empty((x.shape[0], 0))

    def __repr__(self):
        parent_str = f"upstream={self.upstream}" if self.upstream else ""
        return f"Polynomial(degree={self.degree}, {parent_str})"



class Rational(BaseDictionary):
    """
        Computes inverse features 1 / (x + eps) to capture inverse laws 
        or saturation kinetics.
    """
    def _transform(self, x):
        eps = 10e-8
        return numpy.sgn(x)*1.0/(numpy.abs(x) + eps)
 
class RationalQuadratic(BaseDictionary):
    """
        Computes inverse quadratc features 1 / (x**2 + eps) to capture inverse-square laws 
        or saturation kinetics.
    """
    def _transform(self, x):
        eps = 10e-8
        return 1.0/((x**2) + eps)
     

class CrossTerms(BaseDictionary):
    """Adds pairwise interaction terms (Xi * Xj) for all i <= j."""
    def _transform(self, x):
        n_samples, n_features = x.shape
        cross = []
        for i in range(n_features):
            for j in range(i, n_features):
                cross.append((x[:, i] * x[:, j])[:, numpy.newaxis])
        return numpy.hstack(cross) if cross else numpy.empty((n_samples, 0))


class Wave(BaseDictionary):
    """Adds sine and cosine harmonic terms up to specified N harmonics."""
    def __init__(self, upstream=None, n_harmonics=1):
        super().__init__(upstream)
        self.n_harmonics = n_harmonics

    def _transform(self, x):
        features = []
        for k in range(1, self.n_harmonics + 1):
            features.append(numpy.sin(k * x))
            features.append(numpy.cos(k * x))
        return numpy.hstack(features)


class NonLinear(BaseDictionary):
    """Bucket of useful non-linear terms: sgn, abs, ReLU, negative part, tanh."""
    def _transform(self, x):
        features = [
            numpy.sign(x),
            numpy.abs(x),
            numpy.maximum(x, 0),        # ReLU
            numpy.minimum(x, 0),        # Negative part
            numpy.tanh(x),
            x**2,                       # quadratic term
            numpy.abs(x)*x              # quadratic term with sign, velocity drag
        ]
        return numpy.hstack(features)

