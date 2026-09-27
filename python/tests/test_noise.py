import numpy as np
import pytest
from qsim_tutorial import DensityMatrixSimulator, StatevectorSimulator, amplitude_damping, bit_flip


@pytest.mark.parametrize("p", [0, 0.23, 1])
def test_analytic_channels(p):
    rho = DensityMatrixSimulator(1)
    rho.apply_channel(bit_flip(p), 0)
    np.testing.assert_allclose(rho.state, np.diag([1-p, p]), atol=1e-12, rtol=0)
    rho = DensityMatrixSimulator(1)
    rho.x(0)
    rho.apply_channel(amplitude_damping(p), 0)
    np.testing.assert_allclose(rho.state, np.diag([p, 1-p]), atol=1e-12, rtol=0)
    rho = DensityMatrixSimulator(1)
    rho.h(0)
    rho.apply_channel(amplitude_damping(p), 0)
    expected = np.array([[(1+p)/2, np.sqrt(1-p)/2], [np.sqrt(1-p)/2, (1-p)/2]])
    np.testing.assert_allclose(rho.state, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("make_channel,prepare,pauli", [(bit_flip, False, "Z"), (amplitude_damping, True, "Z")])
def test_trajectory_ensemble_matches_density_with_sampling_bound(make_channel, prepare, pauli):
    n_samples = 6000
    ops = make_channel(0.23)
    rho = DensityMatrixSimulator(1)
    if prepare: rho.x(0)
    rho.apply_channel(ops, 0)
    rng = np.random.default_rng(1729)
    values = []
    ensemble = np.zeros((2, 2), complex)
    for _ in range(n_samples):
        sim = StatevectorSimulator(1, seed=rng)
        if prepare: sim.x(0)
        sim.apply_channel(ops, 0)
        state = sim.state
        ensemble += np.outer(state, state.conj())
        values.append(sim.expectation(pauli))
    # Exact variance of the +/-1 trajectory observable for this experiment.
    standard_error = np.sqrt((1 - rho.expectation(pauli)**2) / n_samples)
    assert abs(np.mean(values) - rho.expectation(pauli)) <= 6 * standard_error
    np.testing.assert_allclose(ensemble/n_samples, rho.state, atol=3*standard_error, rtol=0)


def test_kraus_sampling_uses_raw_weight_for_normalization():
    # Tiny allowed completeness drift distinguishes raw weights from normalized p.
    ops = [np.sqrt(1 + 5e-11) * np.eye(2)]
    sim = StatevectorSimulator(1)
    sim.apply_channel(ops, 0)
    np.testing.assert_allclose(sim.state, [1, 0], atol=1e-15, rtol=0)


def test_state_dependent_damping_preserves_complex_coherences_in_ensemble():
    samples = 6000
    original = np.array([np.sqrt(.3), np.exp(.713j)*np.sqrt(.7)])
    ops = amplitude_damping(.41)
    rho = DensityMatrixSimulator(1)
    rho.set_state(np.outer(original, original.conj()))
    rho.apply_channel(ops, 0)
    rng = np.random.default_rng(818)
    observations = []
    for _ in range(samples):
        sim = StatevectorSimulator(1, seed=rng)
        sim.set_state(original)
        sim.apply_channel(ops, 0)
        assert np.vdot(sim.state, sim.state).real == pytest.approx(1, abs=1e-12)
        observations.append([sim.expectation(p) for p in ("X", "Y", "Z")])
    observations = np.array(observations)
    estimates = observations.mean(axis=0)
    standard_errors = observations.std(axis=0, ddof=1)/np.sqrt(samples)
    exact = np.array([rho.expectation(p) for p in ("X", "Y", "Z")])
    assert np.all(abs(estimates-exact) <= 6*standard_errors)


def test_density_two_qubit_reference_and_entangled_noise():
    rho, sim = DensityMatrixSimulator(3), StatevectorSimulator(3)
    for engine in (rho, sim):
        engine.ry(0.37, 0)
        engine.rx(-0.7, 2)
        engine.h(1)
        engine.cx(1, 2)
        engine.cz(0, 2)
        engine.swap(0, 1)
    np.testing.assert_allclose(rho.state, np.outer(sim.state, sim.state.conj()), atol=1e-12, rtol=0)
    assert rho.expectation("XYZ") == pytest.approx(sim.expectation("XYZ"), abs=1e-12)
    rho = DensityMatrixSimulator(2)
    rho.h(0)
    rho.cx(0, 1)
    rho.apply_channel(amplitude_damping(0.4), 0)
    expected = np.array([[.5, 0, 0, np.sqrt(.6)/2], [0, 0, 0, 0],
                         [0, 0, .2, 0], [np.sqrt(.6)/2, 0, 0, .3]])
    np.testing.assert_allclose(rho.state, expected, atol=1e-12, rtol=0)
    assert np.trace(rho.state) == pytest.approx(1)
    assert np.linalg.eigvalsh(rho.state).min() >= -1e-12


def test_invalid_channel_and_density_inputs():
    for constructor in (bit_flip, amplitude_damping):
        for p in (-.1, 1.1, np.nan, np.inf):
            with pytest.raises(ValueError): constructor(p)
    for engine in (StatevectorSimulator(1), DensityMatrixSimulator(1)):
        for ops in ([], [np.eye(2)*2], [np.ones((3, 3))], [np.eye(2)*np.nan]):
            with pytest.raises(ValueError): engine.apply_channel(ops, 0)
    with pytest.raises(MemoryError): DensityMatrixSimulator(9)
    with pytest.raises(MemoryError): DensityMatrixSimulator(2, max_state_bytes=255)
    rho = DensityMatrixSimulator(1)
    for bad in (np.diag([1.1, -.1]), np.eye(2), [[1, 1], [0, 0]], np.ones((3, 3))):
        with pytest.raises(ValueError): rho.set_state(bad)
    rho.set_state(np.diag([0.3, 0.7]))
    np.testing.assert_allclose(rho.probabilities(), [0.3, 0.7], atol=1e-12, rtol=0)
