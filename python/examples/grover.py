"""Two-qubit Grover search with an explicit phase oracle marking |11>."""
import numpy as np
from qsim_tutorial import Circuit, StatevectorSimulator


def grover_circuit():
    circuit = Circuit(2).add("h", 0).add("h", 1)
    circuit.add("cz", 0, 1)  # Oracle: negate precisely the amplitude for item 11.
    # Diffusion: -(2|s><s| - I), physically equivalent up to one global minus sign.
    for q in (0, 1): circuit.add("h", q)
    for q in (0, 1): circuit.add("x", q)
    circuit.add("cz", 0, 1)
    for q in (0, 1): circuit.add("x", q)
    for q in (0, 1): circuit.add("h", q)
    return circuit


def main():
    sim = StatevectorSimulator(2, seed=1729)
    grover_circuit().execute(simulator=sim)
    expected = np.array([0, 0, 0, 1], dtype=complex)
    fidelity = float(abs(np.vdot(expected, sim.state))**2)
    np.testing.assert_allclose(sim.probabilities(), [0, 0, 0, 1], atol=1e-12, rtol=0)
    assert np.isclose(fidelity, 1, atol=1e-12, rtol=0)
    print("Oracle after H on both: [0.5, 0.5, 0.5, -0.5]")
    print("Final amplitudes (global minus sign is immaterial):", sim.state)
    print("Probabilities:", sim.probabilities())
    print("Pure-state fidelity with |11>:", fidelity)
    print("Samples:", sim.sample(128))


if __name__ == "__main__":
    main()
