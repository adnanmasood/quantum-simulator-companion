"""Exercise current QDK imports and the actual companion project, if installed."""
import json
from pathlib import Path
import subprocess
import sys
import pytest

pytest.importorskip("qdk")
pytestmark = pytest.mark.qsharp


def test_qsharp_simulation_and_qir_compilation(tmp_path):
    script = Path(__file__).resolve().parents[2] / "qsharp/run_examples.py"
    output = tmp_path / "bell.ll"
    result = subprocess.run([sys.executable, str(script), "--emit-qir", str(output)],
                            text=True, capture_output=True, check=True)
    summary = json.loads(result.stdout)
    assert summary["teleportation"]["successful_checks"] == 512
    assert sum(summary["Bell"].values()) == 128
    assert sum(summary["FeedForward"].values()) == 128
    assert "__quantum__qis__cx__body" in output.read_text()
