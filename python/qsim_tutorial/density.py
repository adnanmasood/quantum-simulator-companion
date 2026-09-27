"""Deliberately small dense reference, bounded to eight qubits.

Unlike the statevector engine, this oracle expands global matrices. It is for
checking noise on tiny systems, not for benchmarking or scaling up.
"""
import numpy as np
from .core import GateMethods, I, TOL, channel, integer, operator, pauli_matrices


class DensityMatrixSimulator(GateMethods):
    def __init__(self, n_qubits, max_state_bytes=16777216):
        self.n_qubits = integer(n_qubits, "n_qubits", 1)
        budget = integer(max_state_bytes, "max_state_bytes", 1)
        if self.n_qubits > 8 or 16 * (4 ** self.n_qubits) > budget:
            raise MemoryError("density reference is limited to 8 qubits and max_state_bytes")
        dim = 1 << self.n_qubits
        self._state = np.zeros((dim, dim), dtype=np.complex128)
        self._state[0, 0] = 1

    @property
    def state(self):
        result = self._state.copy()
        result.setflags(write=False)
        return result

    def set_state(self, density):
        rho = np.asarray(density, dtype=np.complex128)
        if rho.shape != self._state.shape or not np.isfinite(rho).all():
            raise ValueError("density matrix must have finite entries and shape (2**n, 2**n)")
        if not np.allclose(rho, rho.conj().T, atol=TOL, rtol=0):
            raise ValueError("density matrix must be Hermitian")
        if not np.isclose(np.trace(rho), 1, atol=TOL, rtol=0):
            raise ValueError("density matrix must have trace one")
        if np.linalg.eigvalsh(rho).min() < -TOL:
            raise ValueError("density matrix must be positive semidefinite")
        self._state = rho.copy()

    def _expand(self, matrix, target):
        result = np.ones((1, 1), dtype=complex)
        for q in reversed(range(self.n_qubits)):
            result = np.kron(result, matrix if q == target else I)
        return result

    def apply_gate(self, matrix, target):
        target = self._target(target)
        full = self._expand(operator(matrix, unitary=True), target)
        self._state = full @ self._state @ full.conj().T

    def cx(self, control, target):
        control, target = self._pair(control, target)
        indices = np.arange(self._state.shape[0])
        perm = indices ^ (((indices >> control) & 1) << target)
        self._state = self._state[np.ix_(perm, perm)]

    def cz(self, control, target):
        control, target = self._pair(control, target)
        indices = np.arange(self._state.shape[0])
        signs = 1 - 2 * (((indices >> control) & 1) * ((indices >> target) & 1))
        self._state = signs[:, None] * self._state * signs[None, :]

    def swap(self, a, b):
        a, b = self._pair(a, b)
        indices = np.arange(self._state.shape[0])
        different = ((indices >> a) ^ (indices >> b)) & 1
        perm = indices ^ (different << a) ^ (different << b)
        self._state = self._state[np.ix_(perm, perm)]

    def apply_channel(self, kraus_ops, target):
        target = self._target(target)
        expanded = [self._expand(k, target) for k in channel(kraus_ops)]
        self._state = sum((k @ self._state @ k.conj().T for k in expanded),
                          start=np.zeros_like(self._state))

    def probabilities(self):
        p = self._state.diagonal().real.copy()
        if p.min() < -TOL or not np.isclose(p.sum(), 1, atol=TOL, rtol=0):
            raise ArithmeticError("invalid density probabilities")
        p = np.maximum(p, 0)
        return p / p.sum()

    def expectation(self, pauli):
        self.probabilities()
        full = np.ones((1, 1), dtype=complex)
        for matrix in reversed(pauli_matrices(pauli, self.n_qubits)):
            full = np.kron(full, matrix)
        value = np.trace(self._state @ full)
        if not np.isfinite(value) or abs(value.imag) > TOL:
            raise ArithmeticError("Hermitian expectation has a nonreal or nonfinite result")
        return float(value.real)
