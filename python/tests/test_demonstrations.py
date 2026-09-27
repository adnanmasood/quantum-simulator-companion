from pathlib import Path
import runpy
import numpy as np
import pytest
from qsim_tutorial import StatevectorSimulator
from .helpers import assert_same_pure_state

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_grover_oracle_and_gate_diffusion_agree_with_analytical_reflection():
    definitions = runpy.run_path(str(EXAMPLES / "grover.py"))
    oracle = StatevectorSimulator(2)
    oracle.h(0)
    oracle.h(1)
    oracle.cz(0, 1)
    np.testing.assert_allclose(oracle.state, [.5, .5, .5, -.5], atol=1e-12, rtol=0)
    reflected = 2*np.mean(oracle.state) - oracle.state
    np.testing.assert_allclose(reflected, [0, 0, 0, 1], atol=1e-12, rtol=0)
    sim = StatevectorSimulator(2, seed=1729)
    definitions["grover_circuit"]().execute(simulator=sim)
    # This exact gate decomposition differs from the direct reflection by -1.
    np.testing.assert_allclose(sim.state, -reflected, atol=1e-12, rtol=0)
    assert_same_pure_state(sim.state, reflected)
    np.testing.assert_allclose(sim.probabilities(), [0, 0, 0, 1], atol=1e-12, rtol=0)
    assert sim.sample(128) == {"11": 128}


def test_phase_helper_accepts_rotation_pauli_decompositions():
    state = np.array([np.sqrt(.37), np.sqrt(.63)*np.exp(.81j)])
    rotation, pauli = StatevectorSimulator(1), StatevectorSimulator(1)
    rotation.set_state(state)
    pauli.set_state(state)
    rotation.rz(np.pi, 0)
    pauli.z(0)
    assert not np.allclose(rotation.state, pauli.state)
    assert_same_pure_state(rotation.state, pauli.state)
    # A relative phase error preserves Z probabilities but must still fail.
    wrong = pauli.state.copy()
    wrong[1] *= -1
    np.testing.assert_allclose(abs(wrong)**2, pauli.probabilities(), atol=1e-12, rtol=0)
    with pytest.raises(AssertionError): assert_same_pure_state(wrong, pauli.state)
    with pytest.raises(AssertionError): assert_same_pure_state([1, 0], [0, 1])
    with pytest.raises(AssertionError): assert_same_pure_state([2, 0], [1, 0])


@pytest.mark.parametrize("filename", ["grover.py", "mechanisms.py"])
def test_demonstration_entrypoints(filename, capsys):
    runpy.run_path(str(EXAMPLES / filename), run_name="__main__")
    assert capsys.readouterr().out
