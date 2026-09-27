import numpy as np
import pytest
from qsim_tutorial import Circuit, StatevectorSimulator, amplitude_damping


def teleportation_circuit(theta, phi):
    return (Circuit(3, 2).add("ry", 0, theta=theta).add("rz", 0, theta=phi)
            .add("h", 1).add("cx", 1, 2).add("cx", 0, 1).add("h", 0)
            .add("measure", 0, cbit=0).add("measure", 1, cbit=1)
            .add("x", 2, condition=(1, 1)).add("z", 2, condition=(0, 1)))


@pytest.mark.parametrize("theta,phi", [(0, 0), (np.pi, 0), (np.pi/2, np.pi/2), (0.713, -1.129)])
def test_teleportation_all_branches_fidelity(theta, phi):
    expected = np.array([np.exp(-1j*phi/2)*np.cos(theta/2), np.exp(1j*phi/2)*np.sin(theta/2)])
    seen = set()
    for seed in range(64):
        result = teleportation_circuit(theta, phi).execute(seed=seed)
        seen.add(result.classical_bits)
        # Destination is q2; trace out measured q1,q0 using independent reshape.
        amplitudes = result.state.reshape(2, 4)
        reduced = amplitudes @ amplitudes.conj().T
        fidelity = np.vdot(expected, reduced @ expected).real
        assert fidelity == pytest.approx(1, abs=1e-12)
    assert seen == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_independent_shots_feed_forward_reset_and_noise():
    circuit = (Circuit(2).add("h", 0).add("measure", 0, cbit=0)
               .add("x", 1, condition=(0, 1)).add("measure", 1, cbit=1))
    counts = circuit.run(100, seed=102)
    assert counts == circuit.run(100, seed=102)
    assert set(counts) == {"00", "11"} and sum(counts.values()) == 100
    assert circuit.run(0) == {}
    assert Circuit(1).add("x", 0).add("reset", 0).add("measure", 0, cbit=0).run(4) == {"0": 4}
    noisy = Circuit(1).add("x", 0).add("channel", 0, kraus=amplitude_damping(1)).add("measure", 0, cbit=0)
    assert noisy.run(10) == {"0": 10}
    conditional = Circuit(1).add("measure", 0, cbit=0).add("x", 0, condition=(0, 0))
    np.testing.assert_array_equal(conditional.execute().state, [0, 1])


def test_condition_requires_executed_prior_measurement():
    with pytest.raises(ValueError, match="not yet measured"):
        Circuit(1).add("x", 0, condition=(0, 0)).execute()
    # A skipped conditional measurement does not initialize its output bit.
    c = (Circuit(2).add("measure", 0, cbit=0)
         .add("measure", 1, cbit=1, condition=(0, 1)).add("x", 0, condition=(1, 0)))
    with pytest.raises(ValueError, match="not yet measured"):
        c.execute()
    assert Circuit(2).execute().classical_bits == (0, 0)


def test_validation_before_execution():
    c = Circuit(2)
    invalid = [lambda: c.add("unknown", 0), lambda: c.add("x", 0, 1),
               lambda: c.add("cx", 0, 0), lambda: c.add("x", 2),
               lambda: c.add("rx", 0), lambda: c.add("h", 0, theta=1),
               lambda: c.add("measure", 0), lambda: c.add("measure", 0, cbit=2),
               lambda: c.add("x", 0, cbit=0), lambda: c.add("x", 0, condition=(0, 2)),
               lambda: c.add("x", 0, condition=(3, 1)), lambda: c.add("x", 0, condition=(0,)),
               lambda: c.add("channel", 0), lambda: c.add("h", 0, kraus=amplitude_damping(0.5))]
    for call in invalid:
        with pytest.raises(ValueError): call()
        assert c.instructions == ()
    with pytest.raises(ValueError): c.execute(simulator=StatevectorSimulator(1))
    with pytest.raises(ValueError): c.execute(simulator=StatevectorSimulator(2), seed=4)


def test_supplied_state_executes_without_reinitialization():
    sim = StatevectorSimulator(2)
    sim.x(1)
    result = Circuit(2).add("x", 0).execute(simulator=sim)
    np.testing.assert_array_equal(result.state, [0, 0, 0, 1])
