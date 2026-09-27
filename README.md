# Quantum Simulator Companion

An educational quantum simulator implemented from scratch in **Python/NumPy**, with an equivalent **Rust ideal-state core** and separate **Q# examples**. Everything runs locally on a CPU.

## What is included

- **Python:** dense statevectors; H/X/Y/Z/S/T and rotations; CX/CZ/SWAP; measurement, reset, sampling, Pauli expectations; a small circuit executor with classical conditions; local Kraus trajectories; and a small density-matrix reference.
- **Rust:** the ideal statevector core, measurement, reset, sampling, expectations, and a JSON CLI.
- **Q#:** Bell, feed-forward, and teleportation examples running through Microsoft's QDK, plus optional QIR export and LLVM verification.
- **Verification:** analytical tests, independent Cirq comparisons, shared Python/Rust fixtures, and bounded CPU benchmark scripts.

The Q# examples run through QDK. The custom Python and Rust engines do not execute Q# or QIR. This is a learning implementation with explicit limits; it does not provide GPU/distributed execution or hardware-calibrated noise.

## Python quick start

Python **3.12 or newer** is required. The recorded verification environment used Python 3.13.7. Run these commands from the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e './python[test]'
python python/examples/bell.py
python python/examples/grover.py
python python/examples/teleportation.py
python python/examples/noise.py
python -m pytest python/tests -q
```

On Windows, create the environment with `py -3.13 -m venv .venv`, then activate it with `.venv\Scripts\Activate.ps1`. Use your installed Python command if its name differs.

For the complete optional verification suite, including Cirq and QDK:

```bash
python -m pip install -e './python[test,reference,qsharp]'
python -m pytest python/tests -q
```

Installing only `-e ./python` gives the engine and its NumPy dependency. Optional Cirq/QDK tests skip when those dependencies are absent. The complete suite contains 80 tests. Exact direct versions are pinned in `python/pyproject.toml`; `python/requirements-lock.txt` records the full tested macOS environment, whose wheel availability can differ on other platforms.

To recreate that recorded dependency set:

```bash
python -m pip install -r python/requirements-lock.txt
python -m pip install --no-deps --no-build-isolation -e ./python
```

### Minimal example

```python
from qsim_tutorial import StatevectorSimulator

sim = StatevectorSimulator(2, seed=1729)
sim.h(0)
sim.cx(0, 1)
print(sim.sample(1024))  # Only 00 and 11 are possible.
```

[Python APIs, numerical checks, and limits](python/IMPLEMENTATION.md)

## Rust

Install a Rust toolchain supporting edition 2024; the recorded compiler is **Rust 1.97.1**. From the repository root:

```bash
cargo test --manifest-path rust/Cargo.toml --locked
cargo test --release --manifest-path rust/Cargo.toml --locked
cargo clippy --manifest-path rust/Cargo.toml --all-targets --locked -- -D warnings
cargo build --release --manifest-path rust/Cargo.toml --locked
```

`Cargo.lock` pins the resolved dependencies. The CLI at `rust/target/release/qsim-tutorial` reads one JSON circuit object from standard input; Windows builds use the `.exe` suffix. The 21 integration tests include the shared fixtures in `fixtures/core.json`.

[Rust API and CLI request format](rust/IMPLEMENTATION.md)

## Q# companion

After installing the Python `qsharp` extra:

```bash
python qsharp/run_examples.py
python qsharp/run_examples.py --emit-qir qsharp/generated/bell.ll
```

The runner checks Bell correlations, classical feed-forward, and teleportation, then compiles and verifies Bell QIR. No Azure account or quantum hardware is required. Generated QIR is ignored by Git.

[Q# workflow and examples](qsharp/README.md)

## Benchmarks

Install the `reference` extra and build Rust in release mode first:

```bash
python benchmarks/run_benchmarks.py --quick
python benchmarks/branching.py
```

Omit `--quick` to run the nine ideal workloads. Timings and generated workloads are saved under `benchmarks/` and ignored by Git. Use `--rust-bin` if your executable is elsewhere. See [methodology and timing boundaries](benchmarks/README.md) before interpreting the results.

## Repository layout

```text
python/       NumPy engine, examples, tests, dependency files
rust/         Ideal statevector library and JSON CLI
qsharp/       Q# project and local Python host
fixtures/     Deterministic cases shared by Python and Rust
benchmarks/   Ideal and branching/noise benchmark scripts
```

Keep these directories together: the tests resolve shared fixtures and Q# examples relative to the checkout. [CONVENTIONS.md](CONVENTIONS.md) defines bit order, precision, state updates, and fixture format.

## Check in to GitHub

Copy this directory's contents into your repository root. It contains source, tests, locks, and implementation documentation. Generated outputs, local environments, build targets, caches, and credentials are covered by `.gitignore`.
