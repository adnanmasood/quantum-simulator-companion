"""Run local QDK examples, check feed-forward, and optionally export Bell QIR.

This uses Microsoft's simulator; it does not execute QIR on our Python/Rust core.
"""
import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
from qdk import qsharp
import pyqir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-qir", type=Path, help="write verified Bell QIR to this local path")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    qsharp.init(project_root=str(root))
    shots, seed = 128, 1729
    summary = {"qdk_version": version("qdk"), "shots_per_case": shots, "seed": seed}
    for name in ("Bell", "FeedForward"):
        results = qsharp.run(f"Tutorial.{name}()", shots=shots, seed=seed)
        assert all(isinstance(r, tuple) and len(r) == 2 and r[0] == r[1] for r in results), results
        counts = Counter("".join("1" if bit == qsharp.Result.One else "0" for bit in reversed(r)) for r in results)
        assert set(counts) == {"00", "11"}
        summary[name] = dict(sorted(counts.items()))
    for theta, phi in ((0.0, 0.0), (3.141592653589793, 0.0), (1.5707963267948966, 1.5707963267948966), (0.713, -1.129)):
        results = qsharp.run(f"Tutorial.TeleportAndCheck({theta}, {phi})", shots=shots, seed=seed)
        assert all(result == qsharp.Result.Zero for result in results), results
    summary["teleportation"] = {"input_states": 4, "successful_checks": 4*shots}
    # The project includes measurement-conditioned operations, requiring adaptation.
    qsharp.init(project_root=str(root), target_profile=qsharp.TargetProfile.Adaptive_RI)
    qir = str(qsharp.compile("Tutorial.Bell()"))
    assert "__quantum__qis__h__body" in qir and "__quantum__qis__cx__body" in qir
    module = pyqir.Module.from_ir(pyqir.Context(), qir)
    assert module.verify() is None
    summary["bell_qir_compile"] = "passed (Adaptive_RI target profile)"
    if args.emit_qir:
        args.emit_qir.parent.mkdir(parents=True, exist_ok=True)
        args.emit_qir.write_text(qir)
        summary["qir_file"] = str(args.emit_qir.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
