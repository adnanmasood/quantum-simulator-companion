use qsim_tutorial::{DEFAULT_MAX_STATE_BYTES, StatevectorSimulator};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    io::{self, Read},
    process,
    time::Instant,
};

#[derive(Clone, Deserialize)]
struct Operation {
    gate: String,
    targets: Vec<usize>,
    theta: Option<f64>,
}

#[derive(Deserialize)]
struct Request {
    name: Option<String>,
    n_qubits: usize,
    operations: Vec<Operation>,
    #[serde(default)]
    benchmark: bool,
    shots: Option<usize>,
    seed: Option<u64>,
    repeats: Option<usize>,
    max_state_bytes: Option<usize>,
}

fn validate(operations: &[Operation], n: usize) -> Result<(), String> {
    for (index, op) in operations.iter().enumerate() {
        let arity = match op.gate.as_str() {
            "h" | "x" | "y" | "z" | "s" | "t" | "rx" | "ry" | "rz" => 1,
            "cx" | "cz" | "swap" => 2,
            _ => return Err(format!("operation {index}: unknown gate {:?}", op.gate)),
        };
        if op.targets.len() != arity || op.targets.iter().any(|&q| q >= n) {
            return Err(format!("operation {index}: invalid targets"));
        }
        if arity == 2 && op.targets[0] == op.targets[1] {
            return Err(format!("operation {index}: targets must be distinct"));
        }
        let rotation = matches!(op.gate.as_str(), "rx" | "ry" | "rz");
        if rotation && !op.theta.is_some_and(f64::is_finite) {
            return Err(format!(
                "operation {index}: rotation requires a finite theta"
            ));
        }
        if !rotation && op.theta.is_some() {
            return Err(format!(
                "operation {index}: theta applies only to rotations"
            ));
        }
    }
    Ok(())
}

fn evolve(sim: &mut StatevectorSimulator, operations: &[Operation]) -> Result<(), String> {
    for op in operations {
        let q = op.targets[0];
        match op.gate.as_str() {
            "h" => sim.h(q),
            "x" => sim.x(q),
            "y" => sim.y(q),
            "z" => sim.z(q),
            "s" => sim.s(q),
            "t" => sim.t(q),
            "rx" => sim.rx(op.theta.unwrap(), q),
            "ry" => sim.ry(op.theta.unwrap(), q),
            "rz" => sim.rz(op.theta.unwrap(), q),
            "cx" => sim.cx(q, op.targets[1]),
            "cz" => sim.cz(q, op.targets[1]),
            "swap" => sim.swap(q, op.targets[1]),
            _ => unreachable!("validated before evolution"),
        }
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn execute(request: Request) -> Result<Value, String> {
    validate(&request.operations, request.n_qubits)?;
    let limit = request.max_state_bytes.unwrap_or(DEFAULT_MAX_STATE_BYTES);
    let seed = request.seed.unwrap_or(1729);
    let shots = request
        .shots
        .unwrap_or(if request.benchmark { 4096 } else { 0 });
    let repeats = if request.benchmark {
        request.repeats.unwrap_or(5)
    } else {
        1
    };
    if repeats == 0 {
        return Err("repeats must be positive".into());
    }
    let mut timings = Vec::new();
    let mut final_output = json!({});
    let runs = repeats
        .checked_add(usize::from(request.benchmark))
        .ok_or("repeats is too large")?;
    for run in 0..runs {
        let setup_start = Instant::now();
        let mut sim = StatevectorSimulator::new(request.n_qubits, Some(seed), limit)
            .map_err(|e| e.to_string())?;
        let setup_seconds = setup_start.elapsed().as_secs_f64();
        let evolution_start = Instant::now();
        evolve(&mut sim, &request.operations)?;
        let evolution_seconds = evolution_start.elapsed().as_secs_f64();
        let sampling_start = Instant::now();
        let counts = sim.sample(shots).map_err(|e| e.to_string())?;
        let sampling_seconds = sampling_start.elapsed().as_secs_f64();
        if request.benchmark && run > 0 {
            timings.push(json!({"setup_seconds": setup_seconds,
                "evolution_seconds": evolution_seconds, "sampling_seconds": sampling_seconds}));
        }
        if run == runs - 1 {
            let state: Vec<[f64; 2]> = sim.state().iter().map(|z| [z.re, z.im]).collect();
            let probabilities = sim.probabilities().map_err(|e| e.to_string())?;
            final_output = json!({"name": request.name, "n_qubits": request.n_qubits,
                "state": state, "probabilities": probabilities, "counts": counts});
        }
    }
    if request.benchmark {
        final_output["timings"] = json!(timings);
        final_output["warmup_runs"] = json!(1);
        final_output["repeats"] = json!(repeats);
        final_output["shots"] = json!(shots);
        final_output["seed"] = json!(seed);
    }
    Ok(final_output)
}

fn main() {
    let result = (|| -> Result<Value, String> {
        let mut input = String::new();
        io::stdin()
            .read_to_string(&mut input)
            .map_err(|e| e.to_string())?;
        let request: Request =
            serde_json::from_str(&input).map_err(|e| format!("invalid JSON request: {e}"))?;
        execute(request)
    })();
    match result {
        Ok(output) => println!("{output}"),
        Err(message) => {
            eprintln!("{}", json!({"error": message}));
            process::exit(2);
        }
    }
}
