"""Small, inspectable simulator. Qubit 0 is the least significant bit.

Local operators use O(2**n) storage and work, never a 2**n square gate.
Vectorized NumPy temporaries are intentional: the limit bounds the state itself,
not peak process memory. This reference favors clarity over minimum allocation.
"""
from __future__ import annotations

from collections import Counter
from numbers import Integral, Real
import numpy as np

TOL = 1e-10
I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.diag([1, -1]).astype(np.complex128)
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
S = np.diag([1, 1j]).astype(np.complex128)
T = np.diag([1, np.exp(1j * np.pi / 4)]).astype(np.complex128)
for _gate in (I, X, Y, Z, H, S, T):
    _gate.setflags(write=False)


def integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def angle(theta):
    if isinstance(theta, (bool, np.bool_)) or not isinstance(theta, Real) or not np.isfinite(theta):
        raise ValueError("angle must be a finite real number")
    return float(theta)


def operator(matrix, *, unitary=False):
    result = np.asarray(matrix, dtype=np.complex128)
    if result.shape != (2, 2) or not np.isfinite(result).all():
        raise ValueError("operator must be a finite 2 by 2 complex matrix")
    if unitary and not np.allclose(result.conj().T @ result, I, atol=TOL, rtol=0):
        raise ValueError("gate must be unitary")
    return result


def local_vector(state, matrix, target):
    """Apply a possibly nonunitary local matrix without building a global one."""
    blocks = state.reshape(-1, 2, 1 << target)
    out = np.empty_like(blocks)
    out[:, 0, :] = matrix[0, 0] * blocks[:, 0, :] + matrix[0, 1] * blocks[:, 1, :]
    out[:, 1, :] = matrix[1, 0] * blocks[:, 0, :] + matrix[1, 1] * blocks[:, 1, :]
    return out.reshape(-1)


def channel(operators):
    ops = tuple(operator(k) for k in operators)
    if not ops:
        raise ValueError("a channel requires at least one Kraus operator")
    completeness = sum((k.conj().T @ k for k in ops), start=np.zeros((2, 2), complex))
    if not np.allclose(completeness, I, atol=TOL, rtol=0):
        raise ValueError("Kraus operators must satisfy sum(K†K) = I")
    return ops


def pauli_matrices(pauli, n_qubits):
    if not isinstance(pauli, str) or len(pauli) != n_qubits or set(pauli) - set("IXYZ"):
        raise ValueError("Pauli string must contain one uppercase I/X/Y/Z per qubit, highest first")
    lookup = {"I": I, "X": X, "Y": Y, "Z": Z}
    return [lookup[p] for p in reversed(pauli)]


class GateMethods:
    def _target(self, target):
        target = integer(target, "target")
        if target >= self.n_qubits:
            raise ValueError("target is outside the register")
        return target

    def _pair(self, a, b):
        a, b = self._target(a), self._target(b)
        if a == b:
            raise ValueError("two-qubit gates require distinct targets")
        return a, b

    def h(self, target): self.apply_gate(H, target)
    def x(self, target): self.apply_gate(X, target)
    def y(self, target): self.apply_gate(Y, target)
    def z(self, target): self.apply_gate(Z, target)
    def s(self, target): self.apply_gate(S, target)
    def t(self, target): self.apply_gate(T, target)

    def rx(self, theta, target):
        theta = angle(theta)
        self.apply_gate(np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * X, target)

    def ry(self, theta, target):
        theta = angle(theta)
        self.apply_gate(np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * Y, target)

    def rz(self, theta, target):
        theta = angle(theta)
        self.apply_gate(np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * Z, target)


class StatevectorSimulator(GateMethods):
    def __init__(self, n_qubits, seed=None, max_state_bytes=268435456):
        self.n_qubits = integer(n_qubits, "n_qubits", 1)
        budget = integer(max_state_bytes, "max_state_bytes", 1)
        # Compare bit lengths before shifting; arbitrarily large inputs cannot allocate.
        if self.n_qubits > budget.bit_length() or 16 * (1 << self.n_qubits) > budget:
            raise MemoryError("statevector exceeds max_state_bytes (16 * 2**n bytes)")
        self._state = np.zeros(1 << self.n_qubits, dtype=np.complex128)
        self._state[0] = 1
        self.rng = np.random.default_rng(seed)

    @property
    def state(self):
        result = self._state.copy()
        result.setflags(write=False)
        return result

    def set_state(self, amplitudes):
        state = np.asarray(amplitudes, dtype=np.complex128)
        if state.shape != self._state.shape or not np.isfinite(state).all():
            raise ValueError("state must be a finite vector with 2**n amplitudes")
        norm = float(np.vdot(state, state).real)
        if not np.isclose(norm, 1, atol=TOL, rtol=0):
            raise ValueError("state must be normalized")
        self._state = state.copy() / np.sqrt(norm)

    def apply_gate(self, matrix, target):
        target = self._target(target)
        self._state = local_vector(self._state, operator(matrix, unitary=True), target)

    def cx(self, control, target):
        control, target = self._pair(control, target)
        indices = np.arange(self._state.size)
        left = indices[((indices >> control) & 1 == 1) & ((indices >> target) & 1 == 0)]
        right = left | (1 << target)
        self._state[left], self._state[right] = self._state[right], self._state[left]

    def cz(self, control, target):
        control, target = self._pair(control, target)
        indices = np.arange(self._state.size)
        self._state[((indices >> control) & 1 == 1) & ((indices >> target) & 1 == 1)] *= -1

    def swap(self, a, b):
        a, b = self._pair(a, b)
        indices = np.arange(self._state.size)
        left = indices[((indices >> a) & 1 == 0) & ((indices >> b) & 1 == 1)]
        right = left ^ (1 << a) ^ (1 << b)
        self._state[left], self._state[right] = self._state[right], self._state[left]

    def probabilities(self):
        """Return normalized probabilities, allowing only small arithmetic drift."""
        probabilities = np.abs(self._state) ** 2
        total = probabilities.sum()
        if not np.isfinite(total) or not np.isclose(total, 1, atol=TOL, rtol=0):
            raise ArithmeticError("state norm is invalid")
        return probabilities / total

    def measure(self, target):
        target = self._target(target)
        bits = (np.arange(self._state.size) >> target) & 1
        p_one = float(self.probabilities()[bits == 1].sum())
        outcome = int(self.rng.random() < p_one)
        keep = bits == outcome
        # Retain the branch that MATCHES the outcome, then normalize its raw weight.
        weight = float(np.vdot(self._state[keep], self._state[keep]).real)
        if weight <= 0:
            raise ArithmeticError("sampled a zero-probability branch")
        self._state[~keep] = 0
        self._state[keep] /= np.sqrt(weight)
        return outcome

    def reset(self, target):
        if self.measure(target) == 1:
            self.x(target)

    def sample(self, shots):
        shots = integer(shots, "shots")
        if not shots:
            return {}
        outcomes = self.rng.choice(self._state.size, size=shots, p=self.probabilities())
        return dict(sorted(Counter(format(int(i), f"0{self.n_qubits}b") for i in outcomes).items()))

    def expectation(self, pauli):
        self.probabilities()  # Reject material norm drift before evaluating an observable.
        transformed = self._state
        for target, matrix in enumerate(pauli_matrices(pauli, self.n_qubits)):
            transformed = local_vector(transformed, matrix, target)
        value = np.vdot(self._state, transformed)
        if not np.isfinite(value) or abs(value.imag) > TOL:
            raise ArithmeticError("Hermitian expectation has a nonreal or nonfinite result")
        return float(value.real)

    def apply_channel(self, kraus_ops, target):
        """Sample one local Kraus branch; return its index. This is one trajectory."""
        target = self._target(target)
        ops = channel(kraus_ops)
        branches = [local_vector(self._state, k, target) for k in ops]
        raw_weights = np.array([np.vdot(v, v).real for v in branches])
        total = raw_weights.sum()
        if not np.isfinite(total) or not np.isclose(total, 1, atol=TOL, rtol=0):
            raise ArithmeticError("channel branch weights do not sum to one")
        chosen = int(self.rng.choice(len(ops), p=raw_weights / total))
        # Raw branch weight normalizes amplitudes; normalized probabilities only sample.
        self._state = branches[chosen] / np.sqrt(raw_weights[chosen])
        return chosen


def bit_flip(p):
    p = angle(p)
    if not 0 <= p <= 1:
        raise ValueError("bit-flip probability must be in [0, 1]")
    return (np.sqrt(1 - p) * I, np.sqrt(p) * X)


def amplitude_damping(gamma):
    gamma = angle(gamma)
    if not 0 <= gamma <= 1:
        raise ValueError("damping probability must be in [0, 1]")
    return (np.diag([1, np.sqrt(1 - gamma)]).astype(complex),
            np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex))
