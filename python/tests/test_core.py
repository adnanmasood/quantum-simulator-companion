import json
from pathlib import Path
import numpy as np
import pytest
from qsim_tutorial import StatevectorSimulator, H, X, Y, Z, from_fixture

FIXTURES = json.loads((Path(__file__).resolve().parents[2] / "fixtures/core.json").read_text())


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["name"])
def test_shared_fixture(fixture):
    result = from_fixture(fixture).execute().state
    expected = np.array([complex(*pair) for pair in fixture["expected_state"]])
    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_local_gates_against_independent_kronecker(n):
    rng = np.random.default_rng(927 + n)
    state = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    state /= np.linalg.norm(state)
    sim = StatevectorSimulator(n)
    for target in range(n):
        unitary, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2)))
        full = np.ones((1, 1), complex)
        for q in reversed(range(n)):
            full = np.kron(full, unitary if q == target else np.eye(2))
        sim.set_state(state)
        sim.apply_gate(unitary, target)
        np.testing.assert_allclose(sim.state, full @ state, atol=1e-12, rtol=0)
        sim.apply_gate(unitary.conj().T, target)
        np.testing.assert_allclose(sim.state, state, atol=1e-12, rtol=0)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_two_qubit_gates_all_basis_states(n):
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            for value in range(2**n):
                state = np.eye(2**n, dtype=complex)[value]
                for gate in ("cx", "cz", "swap"):
                    sim = StatevectorSimulator(n)
                    sim.set_state(state)
                    getattr(sim, gate)(a, b)
                    destination, phase = value, 1
                    if gate == "cx" and (value >> a) & 1:
                        destination ^= 1 << b
                    if gate == "cz" and (value >> a) & 1 and (value >> b) & 1:
                        phase = -1
                    if gate == "swap" and ((value >> a) ^ (value >> b)) & 1:
                        destination ^= (1 << a) | (1 << b)
                    expected = np.zeros_like(state)
                    expected[destination] = phase
                    np.testing.assert_array_equal(sim.state, expected)


@pytest.mark.parametrize("bit", [0, 1])
def test_collapse_regression_preserves_matching_basis_branch(bit):
    # Retaining the opposite measurement branch would erase |0>.
    sim = StatevectorSimulator(1, seed=1729)
    if bit:
        sim.x(0)
    for _ in range(4):
        assert sim.measure(0) == bit
        np.testing.assert_array_equal(sim.state, [1-bit, bit])
    sim.reset(0)
    np.testing.assert_array_equal(sim.state, [1, 0])


def test_bell_collapse_and_entangled_reset():
    seen = set()
    for seed in range(64):
        sim = StatevectorSimulator(2, seed=seed)
        sim.h(0)
        sim.cx(0, 1)
        outcome = sim.measure(0)
        seen.add(outcome)
        assert sim.measure(0) == sim.measure(1) == outcome
        expected = np.zeros(4)
        expected[3*outcome] = 1
        np.testing.assert_allclose(sim.state, expected, atol=1e-12, rtol=0)
        sim.reset(0)
        assert sim.measure(0) == 0
        assert sim.measure(1) == outcome
    assert seen == {0, 1}


def test_phase_is_not_probability_and_pauli_order():
    plus = StatevectorSimulator(1)
    minus = StatevectorSimulator(1)
    plus.h(0)
    minus.h(0)
    minus.z(0)
    np.testing.assert_allclose(plus.probabilities(), minus.probabilities(), atol=1e-12, rtol=0)
    plus.h(0)
    minus.h(0)
    np.testing.assert_allclose(plus.probabilities(), [1, 0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(minus.probabilities(), [0, 1], atol=1e-12, rtol=0)
    sim = StatevectorSimulator(3)
    sim.x(0)
    assert sim.expectation("IIZ") == -1
    assert sim.expectation("ZII") == 1
    sim.h(1)
    sim.s(1)
    assert sim.expectation("IYI") == pytest.approx(1)


def test_sampling_seed_state_copy_and_budget():
    a, b = StatevectorSimulator(2, seed=17), StatevectorSimulator(2, seed=17)
    for sim in (a, b):
        sim.h(0)
        sim.cx(0, 1)
    before = a.state
    counts = a.sample(10000)
    assert counts == b.sample(10000)
    assert set(counts) == {"00", "11"}
    assert abs(counts["00"] - 5000) < 6*np.sqrt(2500)
    np.testing.assert_array_equal(a.state, before)
    assert a.sample(0) == {}
    with pytest.raises(ValueError):
        a.state[0] = 7
    copy = a.state
    copy.setflags(write=True)
    copy[0] = 7
    np.testing.assert_array_equal(a.state, before)
    StatevectorSimulator(2, max_state_bytes=64)
    with pytest.raises(MemoryError):
        StatevectorSimulator(2, max_state_bytes=63)
    with pytest.raises(MemoryError):
        StatevectorSimulator(10000000)


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_invalid_registers(n):
    with pytest.raises(ValueError): StatevectorSimulator(n)


def test_invalid_inputs_leave_state_unchanged():
    sim = StatevectorSimulator(2)
    bad_calls = [lambda: sim.h(-1), lambda: sim.h(2), lambda: sim.h(True),
                 lambda: sim.rx(np.nan, 0), lambda: sim.ry(np.inf, 0),
                 lambda: sim.rz(1j, 0), lambda: sim.cx(0, 0), lambda: sim.cz(2, 0),
                 lambda: sim.swap(1, 1), lambda: sim.sample(-1), lambda: sim.sample(1.5),
                 lambda: sim.set_state([1, 0]), lambda: sim.set_state([2, 0, 0, 0]),
                 lambda: sim.set_state([np.nan, 0, 0, 0]),
                 lambda: sim.apply_gate([[1, 0], [0, 2]], 0),
                 lambda: sim.apply_gate(np.eye(4), 0),
                 lambda: sim.apply_gate([[np.inf, 0], [0, 1]], 0),
                 lambda: sim.expectation("X"), lambda: sim.expectation("iz")]
    for call in bad_calls:
        with pytest.raises(ValueError): call()
        np.testing.assert_array_equal(sim.state, [1, 0, 0, 0])


def test_roundoff_tolerance_is_explicit_and_small():
    sim = StatevectorSimulator(1)
    sim.set_state([np.sqrt(1 + 5e-11), 0])
    np.testing.assert_array_equal(sim.state, [1, 0])
    with pytest.raises(ValueError): sim.set_state([np.sqrt(1 + 1e-8), 0])


def test_expectation_rejects_corrupt_norm_and_nonreal_result(monkeypatch):
    sim = StatevectorSimulator(1)
    sim._state *= 2  # Deliberately inject internal arithmetic corruption.
    with pytest.raises(ArithmeticError, match="norm"):
        sim.expectation("Z")
    sim = StatevectorSimulator(1)
    monkeypatch.setattr("qsim_tutorial.core.local_vector", lambda state, matrix, target: 1j*state)
    with pytest.raises(ArithmeticError, match="nonreal"):
        sim.expectation("Z")
