# Python implementation and verification

This package implements the Python companion engine. It implements a small **dense statevector simulator**, a deliberately bounded **density-matrix reference**, and a minimal instruction executor. NumPy is its only runtime dependency. It is educational numerical software, not an OpenQASM/QIR interpreter or production replacement for established simulators.

## Install and run

Commands below run from the `quantum-simulator-companion` repository directory. Python 3.12 or newer is required by the pinned NumPy version; verification used **CPython 3.13.7 on macOS arm64**.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -e ./python
.venv/bin/python python/examples/bell.py
.venv/bin/python python/examples/teleportation.py
.venv/bin/python python/examples/noise.py
.venv/bin/python python/examples/mechanisms.py
.venv/bin/python python/examples/grover.py
```

Optional verification groups keep large SDK dependencies out of the runtime:

```bash
.venv/bin/python -m pip install -e './python[test,reference,qsharp]'
.venv/bin/python -m pytest python/tests -q
.venv/bin/python qsharp/run_examples.py
```

The `test` extra installs pytest, `reference` installs Cirq, and `qsharp` installs QDK and PyQIR. Missing optional SDKs skip their corresponding tests. To reproduce the complete captured verification environment, install `python/requirements-lock.txt`, then install the local package with `--no-deps --no-build-isolation -e ./python`. The full lock is an observed environment with pinned transitive dependencies, not a cross-platform wheel guarantee. Separate `requirements-core.txt`, `requirements-test.txt`, `requirements-reference.txt`, and `requirements-qsharp.txt` support narrower installations.

Validated direct versions: NumPy **2.5.3**, pytest **9.1.1**, Cirq core **1.7.0**, QDK **1.32.3**, PyQIR **0.12.5**. `pip check` reports no broken requirements.

## Statevector API and conventions

```python
from qsim_tutorial import StatevectorSimulator

sim = StatevectorSimulator(3, seed=1729, max_state_bytes=268435456)
sim.h(0)
sim.cx(0, 2)
print(sim.state)
print(sim.sample(1000))
```

- Qubit **0 is the least-significant integer bit**. Index `5` means `|101⟩`; the displayed rightmost bit is Q0. Matrices act on column vectors. Array dtype is `numpy.complex128`, two 64-bit floating-point components.
- `state` returns a **read-only copy**, so callers cannot accidentally modify the engine through its result. `set_state(amplitudes)` copies a finite vector of length `2**n`; its squared norm must differ from 1 by at most `1e-10` with `rtol=0`. Only accepted roundoff is normalized.
- `apply_gate(matrix, target)` accepts a finite 2×2 matrix and checks `U.conj().T @ U` against identity with `atol=1e-10, rtol=0`.
- Named gates: `h/x/y/z/s/t(target)`, `rx/ry/rz(theta, target)`, `cx(control, target)`, `cz(control, target)`, `swap(a, b)`. Rotations use radians and `Rₚ(θ) = cos(θ/2)I − i sin(θ/2)P`. Gate methods return `None`.
- `probabilities()` returns a new normalized probability array after rejecting material norm drift. `measure(target)` returns 0 or 1, retains **matching** amplitudes, removes incompatible ones, and renormalizes by the retained raw branch weight. `reset(target)` measures then applies X if the outcome is 1.
- `sample(shots)` returns `{bitstring: count}` for nondestructive terminal full-register sampling. It advances the RNG, preserves the quantum state, and returns `{}` for zero shots. It is not a mid-circuit measurement operation.
- `expectation(pauli)` accepts a full uppercase Pauli string with the **highest qubit first**; `"IIZ"` applies Z to Q0 of a three-qubit register. It checks state normalization and rejects a nonfinite result or an imaginary residue larger than `1e-10`.
- `apply_channel(kraus_ops, target)` applies one local stochastic trajectory step and returns the selected Kraus-operator index. It checks `Σ K†K = I` using the same tolerance. Sampling uses normalized branch probabilities; amplitude normalization uses the **raw** selected weight.
- Invalid qubit indexes, repeated two-qubit targets, nonfinite matrices/angles, zero-qubit registers, invalid state norms, and negative/noninteger shots are rejected. Seeds initialize NumPy's generator; reproducibility applies to a fixed implementation/version, not identical random streams across Python, Rust, or QDK.

### Memory budget and implementation boundary

The default 256 MiB budget bounds the **primary state array**, `16 × 2**n` bytes. It does not bound total process memory. Local gates allocate an output state and multiplication temporaries; controlled gates allocate index arrays; `state` copies the state; Kraus trajectories retain one candidate state per operator; terminal sampling stores sampled integers. Leave substantial headroom. The local-gate kernel reshapes the vector into `(-1, 2, 2**target)` blocks and combines paired slices. It never builds a full global gate matrix. This clear NumPy implementation is not allocation-free and should not be presented as equivalent to the Rust in-place kernel's allocation behavior.

## Circuits, classical bits, and noise

```python
from qsim_tutorial import Circuit, amplitude_damping

circuit = (
    Circuit(2, n_clbits=2)
    .add("h", 0)
    .add("measure", 0, cbit=0)
    .add("x", 1, condition=(0, 1))
    .add("channel", 1, kraus=amplitude_damping(0.1))
    .add("measure", 1, cbit=1)
)
print(circuit.run(1000, seed=1729))
```

`Circuit(n_qubits, n_clbits=None)` defaults the classical-register width to the qubit count. `.add(gate, *targets, theta=None, cbit=None, condition=None, kraus=None)` validates and appends a frozen `Instruction`, and returns the circuit for chaining. Supported names are the ideal named gates plus `measure`, `reset`, and `channel`. Rotation angles use `theta=`; measurements require `cbit=`; channel steps require `kraus=`. Conditions are `(classical_bit, expected_0_or_1)`.

`execute(seed=None, simulator=None)` returns a `CircuitResult` with a read-only state copy and `classical_bits` in `(c0, c1, ...)` order. Its `bitstring` property reverses that order for display. Without an engine, execution creates a fresh all-zero register. With a matching supplied engine, execution mutates its current state; put the seed on that engine. These two seed modes cannot be combined.

Classical output bits start at zero, but **a condition on a bit that has not been written by an executed measurement raises `ValueError`**. A skipped conditional measurement does not initialize its output. An error encountered during execution does not roll back earlier operations. `run(shots, seed=None)` reinitializes quantum/classical state for every shot and returns classical-register counts. Classical output bits never measured remain zero; there is no implicit final measurement.

`bit_flip(p)` and `amplitude_damping(gamma)` return local Kraus tuples for probabilities in `[0,1]`. Circuit noise is sampled independently each execution. A trajectory remains pure; the average of outer products estimates a mixed state. The code separates trajectory uncertainty from optional terminal measurement sampling.

## Small density reference

`DensityMatrixSimulator(n_qubits, max_state_bytes=16777216)` rejects more than **eight qubits**. It offers `state`, `set_state(density)`, all ideal named gates, `apply_gate`, `apply_channel`, `probabilities`, and `expectation`. `set_state` checks shape, finite entries, Hermiticity, unit trace, and nonnegative eigenvalues within `1e-10`. Density noise computes the complete channel sum rather than sampling one Kraus branch.

This reference **does** expand full operators with Kronecker products. Its strict small-system bound keeps that intentional independent implementation useful as a noise oracle. It has no measurement/reset/shot executor API; use the statevector trajectory path for dynamic circuits.

## Verification results

The complete suite passed **80 tests**, including **21 independent Cirq comparisons** and one optional QDK companion check. Coverage includes:

- Seventeen shared Python/Rust fixtures with independently expanded-matrix expected vectors: basis-order asymmetry, Bell in both control directions, GHZ, interference, complex phase, all rotation families, CZ, nonadjacent SWAP, and a mixed circuit.
- Independent Kronecker-product checks for arbitrary complex one-qubit unitaries on every target in one- through five-qubit registers, plus inverse recovery.
- Exhaustive basis-state CX/CZ/SWAP checks for all ordered distinct targets in two-, three-, and four-qubit registers.
- Deterministic |0⟩/|1⟩ collapse regression for a reversed branch condition, repeated measurement, Bell correlations, conditional state, reset, state-copy isolation, and input/memory validation.
- Teleportation of four input states, including complex phase, over all four classical outcomes; destination reduced-state fidelity equals 1 within `1e-12`.
- A runnable interference/GHZ example checks equal initial probabilities, phase-sensitive H outputs, GHZ XXX coherence, and partial-measurement correlations. A runnable two-qubit Grover search exposes the CZ oracle and H/X/CZ diffusion. Its gate result differs from the analytical reflection by one global minus sign and returns item `11` with unit probability.
- A test-only phase-alignment helper accepts globally different decompositions (Rz(π) versus Z, and Grover's diffusion sign) while rejecting a relative-phase error with identical probabilities, orthogonal states, and invalid normalization. Cirq's named-gate comparisons additionally check exact matching conventions.
- Exact bit-flip/amplitude-damping density predictions, complex coherence, local noise on an entangled state, and density positivity/trace checks.
- **18,000 trajectories** across three noise tests, compared with exact density observables using six-standard-error acceptance bounds. These are finite statistical estimates, not exact ensemble claims.
- Cirq comparison with explicitly reversed `qubit_order`; four additional seeded random 50-operation complex circuits. Matching conventions allow direct vector comparison with `atol=1e-12, rtol=0`.
- Rejection of conditions on unwritten classical bits, including skipped measurements; rejection of material norm drift and nonreal Hermitian expectation results.

Observed example output, seed 1729: Bell terminal counts over 4096 samples are `{"00": 2109, "11": 1987}`; teleportation fidelity is `1.0`. The noise example prepares |+⟩ and applies damping `gamma=0.23`: 6000 trajectories estimate ⟨X⟩ as **0.8756786495**, versus density reference **0.8774964387**, with estimated standard error **0.0041121351**. These outputs document a run, not a performance claim.

The shared benchmark harness is documented in [benchmarks/README.md](../benchmarks/README.md). Run it locally to generate timing records for your environment.
