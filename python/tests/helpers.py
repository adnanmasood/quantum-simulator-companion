"""Test utilities; these are not part of the simulator API."""
import numpy as np


def assert_same_pure_state(actual, expected, atol=1e-12):
    """Assert normalized complex vectors agree up to one global phase."""
    actual, expected = np.asarray(actual, dtype=complex), np.asarray(expected, dtype=complex)
    assert actual.shape == expected.shape and actual.ndim == 1
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(np.vdot(actual, actual), 1, atol=atol, rtol=0)
    np.testing.assert_allclose(np.vdot(expected, expected), 1, atol=atol, rtol=0)
    overlap = np.vdot(expected, actual)
    assert abs(overlap) > 0, "orthogonal states cannot be related by global phase"
    phase = overlap / abs(overlap)
    np.testing.assert_allclose(actual, phase*expected, atol=atol, rtol=0)
