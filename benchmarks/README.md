# CPU benchmarks

These scripts compare the tutorial implementations on identical workloads and check their results before retaining timings. They generate fresh local outputs; prior benchmark results and machine metadata are not included in this source package.

## Setup and execution

From the repository root, after activating the Python environment:

```bash
python -m pip install -e './python[reference]'
cargo build --release --manifest-path rust/Cargo.toml --locked
python benchmarks/run_benchmarks.py --quick
python benchmarks/run_benchmarks.py
python benchmarks/branching.py
```

The full ideal run uses 8, 12, and 16 qubits across Clifford, product-rotation, and entangling-rotation families. Each case has 20 layers, seed 1729, one warmup, five measured repeats, and 4,096 terminal samples. Quick mode runs the three 8-qubit cases with two measured repeats.

Use `--rust-bin PATH` for an executable outside the default Cargo target directory. The default includes `.exe` on Windows. `--cpu-model` and `--memory-bytes` accept independently verified hardware metadata. The script reads the compiler version from `rustc` on your command path.

For a requested single numerical-library thread on macOS/Linux, prefix the ideal command with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python benchmarks/run_benchmarks.py
```

This records the requested environment, not a guarantee of scheduler isolation. On Windows, set the corresponding environment variables in your shell.

## Outputs

- `results.json`: ideal cases, raw repeats, summaries, state comparisons, software versions, and available environment metadata.
- `workload-*.json`: exact ideal instruction streams for the run.
- `branching-results.json`: noisy and measurement-conditioned Python/Cirq comparisons.

These files are ignored by Git. Repeated runs overwrite matching output files, so copy a result elsewhere before starting another run if you need to retain it.

## Timing boundaries

- **Setup:** engine construction and allocation where performed by that implementation. Cirq allocates the state inside simulation, so consider setup plus evolution together when allocation matters.
- **Evolution:** application of the prepared ideal circuit.
- **Sampling:** probability construction and terminal sampling, including bitstring/count aggregation as implemented.
- Python/Cirq state extraction is reported separately. Rust's internal timers exclude JSON parsing and serialization. Each implementation has its own dispatch, validation, and allocation behavior.

All ideal states use double-complex values. Cirq runs with `split_untangled_states=False` for this dense-state comparison. Final states are compared up to global phase; measurement streams need not match across generators. This compares implementations and workloads, not programming languages in isolation.

The branching script runs four-qubit amplitude-damping and feed-forward cases with 1,024 independent shots, one warmup, and five repeats. Its timer includes full shot execution and result conversion, excluding circuit construction. Noise is checked against an analytical probability with a statistical tolerance; feed-forward must preserve the expected correlations. Rust is excluded because its delivered interface has no noisy circuit executor.

Do not compare branching-shot timings directly with ideal-evolution timings: they measure different work. These bounded CPU runs do not measure GPU, distributed, large-memory, or quantum-hardware performance.
