#!/usr/bin/env python3
"""Bounded CPU measurements. Run from the repository root after installation.

Identical ideal instruction streams; setup/evolution/sampling timed separately.
This is an educational implementation comparison, not a vendor ranking.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import numpy as np
from qsim_tutorial import StatevectorSimulator

def apply(sim, ops):
    for op in ops:
        args = ([op["theta"]] if "theta" in op else []) + op["targets"]
        getattr(sim, op["gate"])(*args)

def workload(n, family, depth=20):
    rng = np.random.default_rng(1729 + n)
    ops=[]
    for layer in range(depth):
        for q in range(n):
            if family=="clifford":
                ops.append({"gate":"h" if rng.random()<.5 else "s", "targets":[q]})
            else:
                ops.append({"gate":"ry", "targets":[q], "theta":float(rng.uniform(-np.pi,np.pi))})
                ops.append({"gate":"rz", "targets":[q], "theta":float(rng.uniform(-np.pi,np.pi))})
        if family!="product":
            for q in range(layer%2,n-1,2):
                ops.append({"gate":"cx", "targets":[q,q+1]})
    return {"name":f"{family}-{n}","n_qubits":n,"operations":ops,"family":family,"depth":depth,
            "benchmark":True,"shots":4096,"seed":1729,"repeats":5}

def python_run(w):
    rows=[]
    for trial in range(w["repeats"]+1):
        start=time.perf_counter();sim=StatevectorSimulator(w["n_qubits"],seed=w["seed"])
        created=time.perf_counter();apply(sim,w["operations"])
        evolved=time.perf_counter();counts=sim.sample(w["shots"])
        sampled=time.perf_counter()
        if trial:rows.append({"setup_seconds":created-start,"evolution_seconds":evolved-created,"sampling_seconds":sampled-evolved})
        if trial<w["repeats"]:del sim
    t=time.perf_counter();state=sim.state;extraction=time.perf_counter()-t
    return {"timings":rows,"extraction_seconds":extraction,"counts":counts},state

def cirq_run(w):
    import cirq
    start=time.perf_counter()
    qs=cirq.LineQubit.range(w["n_qubits"])
    ops=[]
    fixed={"h":cirq.H,"x":cirq.X,"y":cirq.Y,"z":cirq.Z,"s":cirq.S,"t":cirq.T,"cx":cirq.CNOT,"cz":cirq.CZ,"swap":cirq.SWAP}
    for op in w["operations"]:
        gate=op["gate"]
        g={"rx":cirq.rx,"ry":cirq.ry,"rz":cirq.rz}[gate](op["theta"]) if gate in {"rx","ry","rz"} else fixed[gate]
        ops.append(g(*(qs[t] for t in op["targets"])))
    circuit=cirq.Circuit(ops)
    construction=time.perf_counter()-start
    rows=[]
    for trial in range(w["repeats"]+1):
        start=time.perf_counter();sim=cirq.Simulator(dtype=np.complex128,seed=w["seed"],split_untangled_states=False)
        created=time.perf_counter();result=sim.simulate(circuit,qubit_order=list(reversed(qs)))
        evolved=time.perf_counter()
        samples=cirq.sample_state_vector(result.final_state_vector,indices=list(range(w["n_qubits"])),repetitions=w["shots"],seed=w["seed"])
        counts=dict(Counter("".join(map(str,row)) for row in samples))
        sampled=time.perf_counter()
        if trial:rows.append({"setup_seconds":created-start,"evolution_seconds":evolved-created,"sampling_seconds":sampled-evolved})
        if trial<w["repeats"]:del sim,result,samples
    t=time.perf_counter();state=result.final_state_vector.copy();extraction=time.perf_counter()-t
    return {"construction_seconds":construction,"timings":rows,"extraction_seconds":extraction,"counts":counts},state

def compare(a,b):
    overlap=np.vdot(a,b)
    aligned=b*np.exp(-1j*np.angle(overlap))
    error=float(np.max(np.abs(a-aligned)))
    if error>1e-9:raise AssertionError(f"benchmark state mismatch: {error}")
    return error

def summaries(rows):
    return {key:{"median":statistics.median([r[key] for r in rows]),"min":min(r[key] for r in rows),"max":max(r[key] for r in rows)} for key in ["setup_seconds","evolution_seconds","sampling_seconds"]}

def main():
    p=argparse.ArgumentParser();p.add_argument("--quick",action="store_true");p.add_argument("--rust-bin",type=Path,default=ROOT/"rust/target/release"/("qsim-tutorial.exe" if os.name == "nt" else "qsim-tutorial"))
    p.add_argument("--cpu-model");p.add_argument("--memory-bytes",type=int)
    args=p.parse_args()
    data={"date_utc":datetime.now(timezone.utc).isoformat(),"platform":platform.platform(),"machine":platform.machine(),"python":sys.version,"processor":platform.processor(),"thread_environment":{k:os.environ.get(k) for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]},"versions":{p:importlib.metadata.version(p) for p in ["numpy","cirq-core"]},"cases":[],"policy":"One warmup, five measured repeats; no GPU; all states complex128; 4096 terminal samples. Python validates each named gate; Rust/Cirq have their own dispatch/validation. Cirq state allocation is inside evolution, so compare setup+evolution where allocation matters."}
    try:data["cpu"]=args.cpu_model or subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.SubprocessError):data["cpu"]=platform.processor()
    data["memory_bytes"]=args.memory_bytes
    try:data["rustc"]=subprocess.check_output([shutil.which("rustc") or "rustc","--version"],text=True).strip()
    except (OSError,subprocess.SubprocessError):data["rustc"]="unavailable"
    for n in ([8] if args.quick else [8,12,16]):
        for family in ["clifford","product","entangling"]:
            w=workload(n,family)
            if args.quick:w["repeats"]=2
            py,a=python_run(w);cr,b=cirq_run(w)
            raw=subprocess.run([str(args.rust_bin)],input=json.dumps(w),text=True,capture_output=True,check=True)
            rs=json.loads(raw.stdout)
            c=np.array([complex(*x) for x in rs["state"]])
            for r in [py,cr,rs]:
                if sum(r["counts"].values())!=w["shots"]:raise AssertionError("wrong shot count")
                r["summary"]=summaries(r["timings"])
            rs.pop("state",None);rs.pop("probabilities",None)
            record={"name":w["name"],"n_qubits":n,"depth":w["depth"],"gates":len(w["operations"]),"shots":w["shots"],"seed":w["seed"],"python":py,"cirq":cr,"rust":rs,"cirq_max_amplitude_error":compare(a,b),"rust_max_amplitude_error":compare(a,c)}
            # Keep count cardinality/checksum, not pages of histograms in the benchmark record.
            for engine in ["python","cirq","rust"]:
                counts=record[engine].pop("counts")
                record[engine]["observed_distinct_bitstrings"]=len(counts)
                record[engine]["sample_total"]=sum(counts.values())
            data["cases"].append(record)
            (ROOT/"benchmarks/results.json").write_text(json.dumps(data,indent=2))
            (ROOT/"benchmarks"/f"workload-{w['name']}.json").write_text(json.dumps(w))
            print(w["name"],"verified and measured",flush=True)
    print("Saved benchmarks/results.json",flush=True)

if __name__=="__main__":main()
