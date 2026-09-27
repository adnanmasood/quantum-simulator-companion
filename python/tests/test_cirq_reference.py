"""Optional independent library comparison with explicitly reversed qubit order."""
import json
from pathlib import Path
import numpy as np
import pytest
from qsim_tutorial import from_fixture, StatevectorSimulator

cirq = pytest.importorskip("cirq")
pytestmark = pytest.mark.reference
FIXTURES = json.loads((Path(__file__).resolve().parents[2] / "fixtures/core.json").read_text())


def cirq_operation(op, qubits):
    gate, targets = op["gate"], [qubits[t] for t in op["targets"]]
    gates = {"h": cirq.H, "x": cirq.X, "y": cirq.Y, "z": cirq.Z,
             "s": cirq.S, "t": cirq.T, "cx": cirq.CNOT, "cz": cirq.CZ, "swap": cirq.SWAP}
    if gate in {"rx", "ry", "rz"}:
        return getattr(cirq, gate)(op["theta"])(*targets)
    return gates[gate](*targets)


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["name"])
def test_shared_fixture_against_cirq(fixture):
    qubits = cirq.LineQubit.range(fixture["n_qubits"])
    circuit = cirq.Circuit(cirq_operation(op, qubits) for op in fixture["operations"])
    result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=list(reversed(qubits)))
    np.testing.assert_allclose(from_fixture(fixture).execute().state, result.final_state_vector, atol=1e-12, rtol=0)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_seeded_random_complex_circuits(n):
    rng = np.random.default_rng(218 + n)
    qubits = cirq.LineQubit.range(n)
    operations = []
    for _ in range(50):
        gate = str(rng.choice(["rx", "ry", "rz", "cx", "cz", "swap", "t", "h"]))
        targets = rng.choice(n, 2 if gate in {"cx", "cz", "swap"} else 1, replace=False).tolist()
        op = {"gate": gate, "targets": targets}
        if gate in {"rx", "ry", "rz"}: op["theta"] = float(rng.uniform(-np.pi, np.pi))
        operations.append(op)
    fixture = {"n_qubits": n, "operations": operations}
    result = cirq.Simulator(dtype=np.complex128).simulate(
        cirq.Circuit(cirq_operation(op, qubits) for op in operations), qubit_order=list(reversed(qubits)))
    np.testing.assert_allclose(from_fixture(fixture).execute().state, result.final_state_vector, atol=1e-12, rtol=0)
