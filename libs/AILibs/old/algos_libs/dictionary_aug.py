import numpy

class DictionaryAug:

    def __init__(self):
        pass


    def apply(self, x, poly_order = 0, n_harmonics = 0):
        x_const     = self._augment_constant(x)
        x_poly      = self._augment_polynomial(x, poly_order)
        x_sincos    = self._augment_sin_cos(x, n_harmonics)


    def _augment_constant(self, x):
        return numpy.ones((x.shape[0], 1), dtype=numpy.float32)
    
 

    def _augment_polynomial(self, x, order):
        """
        Generate polynomial powers (excluding original features).

        Parameters:
        - x: np.ndarray of shape (batch_size, num_features)
        - order: int, highest polynomial order (>= 2)

        Returns:
        - np.ndarray of shape (batch_size, num_features * (order - 1))
        """
        poly_terms = [x ** p for p in range(2, order + 1)]
        return numpy.concatenate(poly_terms, axis=1)


    def _augment_sin_cos(self, x, n_harmonics = 1):
        """
        Generate sine and cosine harmonics of input features.

        Parameters:
        - x: np.ndarray of shape (batch_size, num_features)
        - harmonics: int, number of harmonics

        Returns:
        - np.ndarray of shape (batch_size, num_features * 2 * harmonics)
        """
        result = numpy.zeros(x.shape[0], x.shape[1], n_harmonics*2)
        for k in range(n_harmonics):
            sine   = numpy.sin(k*x)
            cosine = numpy.cos(k*x) 

        return numpy.concatenate(sin_cos, axis=1)



    def _augment_pairwise_products(self, x):
        """
        Generate unique pairwise feature products (i < j).

        Parameters:
        - x: np.ndarray of shape (batch_size, num_features)

        Returns:
        - np.ndarray of shape (batch_size, num_unique_pairs)
        """
        batch_size, num_features = x.shape
        indices = [(i, j) for i in range(num_features) for j in range(i + 1, num_features)]
        products = [x[:, i] * x[:, j] for i, j in indices]
        return numpy.stack(products, axis=1)


    def _augment_nonlinear_cross(self, x):
        """
        Generate nonlinear cross terms : x_a * sin(x_b) and x_a * cos(x_b)
        for all unique pairs (a != b), common in nonlinear dynamical systems.

        Parameters:
        - x: np.ndarray of shape (batch_size, num_features)

        Returns:
        - np.ndarray of shape (batch_size, num_features * (num_features - 1) * 2)
          columns ordered as:
          x0*sin(x1), x0*cos(x1), x0*sin(x2), x0*cos(x2), ..., x1*sin(x0), x1*cos(x0), ...
        """
        num_features = x.shape[1]
        terms = []
        for a in range(num_features):
            for b in range(num_features):
                if a != b:
                    terms.append(x[:, a:a+1] * numpy.sin(x[:, b:b+1]))
                    terms.append(x[:, a:a+1] * numpy.cos(x[:, b:b+1]))

        return numpy.concatenate(terms, axis=1)

