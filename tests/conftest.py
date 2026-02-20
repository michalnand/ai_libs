"""
Root conftest.py — shared fixtures available to ALL test categories.
Add only truly global fixtures here.
"""
import pytest
import numpy


@pytest.fixture
def rng():
    """Seeded random generator for reproducible tests."""
    return numpy.random.default_rng(seed=42)
