# Rust ideal statevector simulator

This is the complete ideal-state companion to the NumPy implementation. It is a teaching implementation, with checked inputs and an in-place gate kernel; it is not an optimized production simulator. The library uses safe Rust and `num_complex::Complex64`, which stores **two `f64` components**, or 16 bytes per amplitude. The name `Complex64` here does not mean NumPy's 8-byte `complex64`.

## Conventions and API

Qubit 0 is the least-significant integer bit. Array element `i` holds the amplitude of the binary representation of `i`; displayed bitstrings and Pauli strings put the highest-numbered qubit first. Gates act on column vectors. Controlled targets are `(control, target)`.

`StatevectorSimulator::new(n_qubits, seed, max_state_bytes)` returns `Result<StatevectorSimulator, SimError>`. The seed is `Option<u64>`. Use `DEFAULT_MAX_STATE_BYTES` for the default 268,435,456-byte limit. This is a limit on **the persistent amplitude vector**, not a promise about total process memory. Zero-qubit registers are rejected. Dimension shifts, allocation-size multiplication, byte limits, and the initial state-vector allocation are checked before use.

The public API is:

- `n_qubits()` and `state()`. The latter is an immutable slice; call `.to_vec()` when an owned snapshot is needed.
- `set_state(&[Complex64])`, requiring exactly `2^n` finite amplitudes and squared norm within `1e-10` of one. Accepted inputs are normalized to remove this small initial drift; rejected inputs leave the current state unchanged.
- `apply_gate([[Complex64; 2]; 2], target)`, validating finite entries and `U†U = I` within `1e-10` before mutation.
- `h/x/y/z/s/t(target)` and `rx/ry/rz(theta, target)`. Angles are in radians and must be finite; rotations are `exp(-i θ P / 2)`.
- `cx(control, target)`, `cz(control, target)`, and `swap(first, second)`. Targets must be valid and distinct.
- `probabilities() -> Result<Vec<f64>, SimError>`, returning squared magnitudes normalized for permitted arithmetic drift without changing the state. A squared-norm error larger than `1e-10`, or a nonfinite norm, is rejected.
- `measure(target) -> Result<u8, SimError>`, sampling the Born rule, zeroing the unselected branch, and normalizing the selected branch.
- `reset(target) -> Result<(), SimError>`, measuring and applying X when the outcome is one. On an entangled input this also conditions the rest of the register on the sampled measurement branch.
- `sample(shots) -> Result<BTreeMap<String, usize>, SimError>`, independently sampling the full register without collapsing or modifying its amplitudes. This advances the RNG. Zero shots returns an empty map.
- `expectation(pauli) -> Result<f64, SimError>`, for an `n`-character uppercase `I/X/Y/Z` string. For example, `"IX"` applies X to qubit 0 of a two-qubit register.

Single-qubit evolution updates amplitude pairs with stride `1 << target`. Both old amplitudes are copied before either output is written. No full-register gate matrix is materialized. CX and SWAP perform each required exchange once, while CZ negates only amplitudes with both selected bits set.

`sample` builds one cumulative probability array, then uses binary search per draw. Its cost is O(2^n + shots × log(2^n)), plus bitstring formatting and ordered-map updates. The temporary cumulative array contains one `f64` per amplitude. Expectation values use direct basis-index transformations and cost O(n × 2^n) without building a dense observable.

## Build and run

From this directory, on a machine with a working Rust installation:

```sh
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo build --release --locked
```

`Cargo.lock` records the exact resolved dependencies, including `num-complex 0.4.6`, `rand 0.9.5`, `serde 1.0.229`, and `serde_json 1.0.151`. An initial build needs access to crates.io; subsequent builds can add `--offline` once dependencies are cached. The tested compiler is Rust 1.97.1; a lower minimum supported version has not been established.

If a macOS build needs the standalone Command Line Tools instead of the active Xcode installation, select them for that command only:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools \
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
cargo test --locked
```

This does not change global Xcode selection. Use your configured linker and Rust toolchain on other platforms.

The built executable is `target/release/qsim-tutorial`. It reads one JSON object from standard input. Example:

```sh
target/release/qsim-tutorial <<'JSON'
{
  "name": "bell",
  "n_qubits": 2,
  "operations": [
    {"gate": "h", "targets": [0]},
    {"gate": "cx", "targets": [0, 1]}
  ]
}
JSON
```

The result contains `name`, `n_qubits`, `state` as `[real, imaginary]` pairs, `probabilities`, and `counts`. Without a `shots` field, ordinary fixture execution performs no sampling and returns empty counts. Gate names are lowercase; rotations require `theta`. A nonrotation with `theta` is rejected. Invalid JSON or invalid input produces a JSON object with an `error` string on stderr and exits with status 2; successful stdout contains only the JSON result. Fields such as a fixture's `expected_state` are allowed and ignored by the executor.

The shared `../fixtures/core.json` file is an array. Feed each object separately to the CLI, rather than feeding the complete array.

## Benchmark contract

Add `"benchmark": true`, `"shots": 4096`, `"seed": 1729`, and `"repeats": 5` to an ordinary request. The CLI performs one complete untimed warmup, followed by the requested positive number of timed repeats. Defaults are 4,096 shots, seed 1,729, and five repeats. Every repeat starts from a fresh zero state and the same seed.

Additional output fields are `warmup_runs`, `repeats`, `shots`, `seed`, and `timings`. Each element of `timings` contains:

```json
{
  "setup_seconds": 0.0,
  "evolution_seconds": 0.0,
  "sampling_seconds": 0.0
}
```

Those zeros illustrate the schema; they are not measurements. Actual values come from `std::time::Instant`. Setup includes state allocation/initialization and RNG construction. Evolution executes the already-parsed gate operations, including the public API's validation. Terminal sampling includes probability accumulation, RNG draws, bitstring formatting, and count aggregation. Parsing, request validation, final amplitude/probability conversion, and JSON serialization are outside these three timers. State, probabilities, and counts in the output are from the final repeat. The CLI reports every repeat and makes no speed claims; the root benchmark runner performs equivalent workload comparisons.

## Verification and limitations

Local debug and release test suites both passed on arm64 macOS with Rust/Cargo 1.97.1 and Command Line Tools clang 16.0.0. The test suite contains **21 integration tests**: 18 analytical/API tests and three CLI tests. One CLI test checks all **17 shared fixtures** against independently constructed expected vectors. Tests cover qubit-order asymmetry, both control directions, complex phases, rotations and inverses, interference, Bell/GHZ states, unitary validation, nonfinite inputs, allocation limits, accumulated norm-drift rejection, random-complex inverse identities, Pauli expectations, destructive measurement, correlated collapse, reset, nonmutating sampling, seeded reproducibility, and CLI error/benchmark behavior. Clippy with warnings denied passed.

The implementation is CPU-only, single-threaded, dense, and ideal. It has no noise channels, density matrices, arbitrary multi-qubit matrix API, compiler, GPU path, distributed execution, or circuit optimization. The Python companion supplies noise examples and additional circuit functionality. The CLI handles deterministic gate fixtures and terminal sampling; measurement/reset are available through the Rust library but are not fixture gate names.

Floating-point evolution is approximate. Validating a norm or unitary within tolerance does not make it mathematically exact, and the code does not silently renormalize after every gate. Probabilities, nonempty sampling, measurement, and expectations reject material accumulated norm drift. Measurement normalizes with the raw retained branch weight; sampling normalization does not modify amplitudes. Full-state output, JSON buffers, probability vectors, cumulative sampling arrays, and count maps add memory beyond the guarded amplitude vector. The initial vector allocation uses `try_reserve_exact`; later standard-library allocations can still fail if process memory is exhausted. Seeded runs are reproducible within this locked Rust environment, but do not promise the same random draws as NumPy, Cirq, QDK, or another RNG implementation.
