"""
    Tests for AILibs Dictionary and their integration with LargeScaleRegression
"""
import pytest
import numpy
import AILibs

@pytest.mark.regression
class TestLargeScaleNonLinearRegression:

    def test_polynomial_dictionary(self, rng):
        """Test polynomial dictionary expansion and shape."""
        x = rng.standard_normal((1000, 3))
        poly = AILibs.dictionary.Polynomial(degree=3)
        
        z = poly(x)
        # Degree 2 and 3 powers for 3 features: x^2 (3 cols), x^3 (3 cols) -> total 6 cols
        assert z.shape == (1000, 6)
        
        # Verify mathematical correctness
        assert numpy.allclose(z[:, :3], x**2)
        assert numpy.allclose(z[:, 3:], x**3)


    def test_cross_terms_dictionary(self, rng):
        """Test cross-terms interaction dictionary (Xi * Xj for i <= j)."""
        x = rng.standard_normal((1000, 3))
        cross = AILibs.dictionary.CrossTerms()
        
        z = cross(x)
        # For 3 features, combinations with replacement: i <= j -> 6 terms
        expected_cols = 3 * (3 + 1) // 2
        assert z.shape == (1000, expected_cols)
        
        # Check specific cross term x_0 * x_1 (second column after (0,0))
        expected_term = (x[:, 0] * x[:, 1])[:, numpy.newaxis]
        assert numpy.allclose(z[:, 1], expected_term[:, 0])


    def test_complex_dictionary_tree_graph(self, rng):
        """Test complex compositional dictionary tree graph and its integration with LargeScaleRegression."""
        x = rng.standard_normal((2000, 3))

        # 1. Build computational graph tree using upstream references
        d_identity = AILibs.dictionary.Identity()
        d_constant = AILibs.dictionary.Constant()
        d_cross    = AILibs.dictionary.CrossTerms()
        d_wave     = AILibs.dictionary.Wave(upstream=d_cross, n_harmonics=5)
        
        # 2. Parallel stack via Concatenate 
        d_all = AILibs.dictionary.Concatenate([d_identity, d_constant, d_wave])

        # 3. Evaluate graph directly
        x_aug = numpy.clip(d_all(x), -10.0, 10.0)
        
        # Dimension verification:
        # - Identity: 3 cols
        # - Constant: 1 col
        # - CrossTerms on 3 features: 6 cols
        # - Wave on CrossTerms output (6 features, 5 harmonic -> sin & cos): 60 cols
        # Total expected columns = 3 + 1 + 6*5*2
        expected_features = 3 + 1 + (6 * 5 * 2)
        assert x_aug.shape == (2000, expected_features)

        # 4. Create synthetic target y using the augmented feature graph
        a = numpy.zeros((expected_features, 1))
        a[0, 0] = 1.5   # Identity feature 0
        a[3, 0] = 0.8   # Constant feature
        a[4, 0] = -2.0  # First cross term element
        
        y = x_aug @ a

        # 5. Fit LargeScaleRegression using the complex dictionary graph
        lsr = AILibs.LargeScaleRegression(
            batch_size=200,
            num_batches=5,
            num_steps=10,
            dictionary=d_all
        )
        lsr.fit((x, y))

        best_idx = numpy.argmin([m["loss"] for m in lsr.models])
        a_est = lsr.models[best_idx]["params"]

        assert a_est.shape == a.shape
        assert numpy.allclose(a_est, a, atol=1e-2)

        # 6. Verify end-to-end predictions
        y_pred = lsr.predict(x)
        assert y_pred.shape == y.shape
        assert numpy.allclose(y_pred, y, atol=1e-2)