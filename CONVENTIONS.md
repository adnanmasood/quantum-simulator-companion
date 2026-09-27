# Numerical conventions and scope

- Qubit **0 is the least-significant index bit**. Array index 5 is displayed as `101`; the highest qubit is printed first.
- Gates act on column vectors. Python uses NumPy `complex128`; Rust uses `num_complex::Complex64`. Both contain two 64-bit floating-point components and occupy 16 bytes per amplitude.
- Initial registers are `|0…0⟩`. Zero-qubit registers are rejected. User states must have the right shape, finite entries, and unit norm within the documented tolerance.
- Rotations use radians and `Rₚ(θ) = exp(−iθP/2)`. Controlled-gate targets are ordered `(control, target)`.
- Public matrix gates are finite two-by-two unitaries. Local nonunitary Kraus application is a separate operation.
- Measurement retains the reported branch and renormalizes it. Reset measures and applies X when the result is 1. Terminal sampling preserves amplitudes while advancing the random generator.
- Displayed bitstrings and Pauli strings put the highest-numbered qubit first.
- Seeds reproduce a fixed implementation's random stream; different libraries need not produce the same shot sequence.
- Allocation limits cover the primary state array rather than all process memory or temporary arrays.

Python additionally provides classical feed-forward, local Kraus trajectories, and a small density reference. Rust implements the ideal core. Q# runs independently through Microsoft's QDK.

## Shared fixtures

`fixtures/core.json` is an array of deterministic cases. Each contains `name`, `n_qubits`, and `operations`; operations have `gate`, `targets`, and, for rotations, `theta`. `expected_state` stores `[real, imaginary]` pairs.

Supported fixture gates are `h`, `x`, `y`, `z`, `s`, `t`, `rx`, `ry`, `rz`, `cx`, `cz`, and `swap`. Controlled targets are `[control, target]`. There is no measurement in these deterministic fixtures.

The Rust CLI and `python/examples/fixtures.py` each consume **one object** from standard input, rather than the whole fixture array. Rust can additionally accept benchmark fields described in [its implementation notes](rust/IMPLEMENTATION.md).

## Validation

Tests cover asymmetric basis states, complex phase, inverse gates, nonadjacent controls, measurement branches, reset, sampling, input rejection, teleportation, and noise. Independent comparisons must account for qubit order and, where appropriate, global phase. Passing a finite test suite supplies evidence within that suite's scope; it is not a proof for arbitrary inputs.
