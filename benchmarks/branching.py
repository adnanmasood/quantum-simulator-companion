"""Small Python/Cirq measurements of noisy shots and classical feed-forward.

These are per-shot workflows, separately reported from the ideal terminal-sample
benchmark. Rust has no channel executor and is intentionally not included.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
import numpy as np
import cirq
from qsim_tutorial import Circuit, amplitude_damping

def main():
    shots,repeats,seed=1024,5,1729
    qs=cirq.LineQubit.range(4)
    noisy=Circuit(4).add("x",0)
    for _ in range(6):noisy.add("channel",0,kraus=amplitude_damping(.05))
    for q in range(4):noisy.add("measure",q,cbit=q)
    nc=cirq.Circuit(cirq.X(qs[0]),*[cirq.amplitude_damp(.05)(qs[0]) for _ in range(6)],cirq.measure(*reversed(qs),key="out"))
    feed=Circuit(4).add("h",0).add("measure",0,cbit=0).add("x",3,condition=(0,1))
    for q in range(4):feed.add("measure",q,cbit=q)
    fc=cirq.Circuit(cirq.H(qs[0]),cirq.measure(qs[0],key="m"),cirq.X(qs[3]).with_classical_controls("m"),cirq.measure(*reversed(qs),key="out"))
    result={"date_utc":datetime.now(timezone.utc).isoformat(),"shots":shots,"repeats":repeats,"warmups":1,"seed":seed,"timing_boundary":"Fresh simulator/circuit execution and result conversion for 1024 independent shots; circuit construction excluded. Counts are statistically checked; different RNGs need not agree shot-for-shot.","cases":[]}
    for name,ours,theirs in [("amplitude-damping-6",noisy,nc),("mid-circuit-feed-forward",feed,fc)]:
        row={"name":name,"n_qubits":4}
        for backend in ["python","cirq"]:
            timings=[]
            for rep in range(repeats+1):
                start=time.perf_counter()
                if backend=="python":counts=ours.run(shots,seed=seed)
                else:
                    raw=cirq.Simulator(dtype=np.complex128,seed=seed,split_untangled_states=False).run(theirs,repetitions=shots)
                    counts=dict(Counter("".join(map(str,r)) for r in raw.measurements["out"]))
                elapsed=time.perf_counter()-start
                if rep:timings.append(elapsed)
                if sum(counts.values())!=shots:raise AssertionError("shot total")
                if name.startswith("amplitude"):
                    expected=.95**6
                    observed=counts.get("0001",0)/shots
                    if set(counts)-{"0000","0001"} or abs(observed-expected)>6*np.sqrt(expected*(1-expected)/shots):raise AssertionError("noise ensemble mismatch")
                elif set(counts)-{"0000","1001"}:raise AssertionError("feed-forward correlation mismatch")
            row[backend]={"seconds":timings,"median_seconds":statistics.median(timings),"counts":counts}
        result["cases"].append(row)
        print(name,"passed",flush=True)
    (ROOT/"benchmarks/branching-results.json").write_text(json.dumps(result,indent=2))

if __name__=="__main__":main()
