# Q# companion: programs, simulation, and verified QIR

The Q# companion runs on Microsoft's local QDK simulator. It demonstrates the same Bell correlations, measurement-conditioned control, and teleportation as the Python implementation. It also compiles a Bell entry point to QIR and runs LLVM verification through PyQIR. **The Python and Rust educational simulators do not ingest or execute this QIR.**

From the `quantum-simulator-companion` repository directory:

```bash
.venv/bin/python -m pip install -r python/requirements-qsharp.txt
.venv/bin/python qsharp/run_examples.py
```

If creating a new environment, run `python3.13 -m venv .venv` first. Verified with Python **3.13.7**, **qdk 1.32.3**, and **pyqir 0.12.5** on macOS arm64. This requires no Azure account, QPU, cloud submission, or .NET project.

The current import is:

```python
from pathlib import Path
from qdk import qsharp

qsharp.init(project_root=str(Path("qsharp").resolve()))
results = qsharp.run("Tutorial.Bell()", shots=128, seed=1729)
```

`qsharp.json` describes the project; `src/Main.qs` contains three operations:

- `Tutorial.Bell()` prepares a Bell pair and returns two measurement results, resetting both qubits before release.
- `Tutorial.FeedForward()` measures one superposition, conditionally applies X to the other qubit, and checks correlated classical results.
- `Tutorial.TeleportAndCheck(theta, phi)` prepares `Rz(phi) Ry(theta)|0⟩`, teleports it using two measured bits and X/Z corrections, undoes preparation on the destination, and returns its measurement. Correct transfer produces `Zero`.

The runner checks **128 Bell shots**, **128 feed-forward shots**, and **512 teleportation checks** over four input states, including an asymmetric complex state. On the verified installation, Bell and feed-forward each returned `{"00": 69, "11": 59}` at seed 1729; all teleportation checks returned `Zero`. Python and QDK use different random streams, so identical seeds need not produce matching counts.

For QIR output:

```bash
.venv/bin/python qsharp/run_examples.py --emit-qir qsharp/generated/bell.ll
```

The runner initializes the compilation context with `TargetProfile.Adaptive_RI`, compiles `Tutorial.Bell()`, confirms H/CX calls, and verifies the LLVM module. The entire project includes measurement-dependent branches, so loading it under the `Base` profile is not supported. In this QDK release the controlled-X symbol is `__quantum__qis__cx__body`; the output also contains measurement/reset and result-recording calls. QIR is an interface contract with a target profile and runtime requirements, not a guarantee that any simulator can execute every QIR program.

Generated QIR is excluded from this source package. The command above creates `generated/bell.ll` locally; its text may change with future SDK versions.

Sources for the current API and migration: [Microsoft's qdk.qsharp reference](https://learn.microsoft.com/en-us/python/qdk/qdk.qsharp?view=qsharp-py), [official migration from qsharp to qdk](https://github.com/microsoft/qdk/wiki/Migrating-from-qsharp-to-qdk), and [QIR Alliance PyQIR](https://github.com/qir-alliance/pyqir).
