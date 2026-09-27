use serde_json::{Value, json};
use std::{
    io::Write,
    process::{Command, Output, Stdio},
};

fn call(input: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_qsim-tutorial"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn bell_cli_and_benchmark_contract() {
    let request = json!({"name":"bell-smoke","n_qubits":2,
        "operations":[{"gate":"h","targets":[0]},{"gate":"cx","targets":[0,1]}],
        "benchmark":true,"shots":32,"seed":1729,"repeats":2});
    let output = call(&request.to_string());
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["name"], "bell-smoke");
    assert_eq!(value["warmup_runs"], 1);
    assert_eq!(value["repeats"], 2);
    assert_eq!(value["timings"].as_array().unwrap().len(), 2);
    assert!(
        (value["state"][0][0].as_f64().unwrap() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12
    );
    assert_eq!(value["probabilities"][1], 0.0);
    assert_eq!(
        value["counts"]
            .as_object()
            .unwrap()
            .values()
            .map(|v| v.as_u64().unwrap())
            .sum::<u64>(),
        32
    );
    for row in value["timings"].as_array().unwrap() {
        for key in ["setup_seconds", "evolution_seconds", "sampling_seconds"] {
            assert!(row[key].as_f64().unwrap() >= 0.0);
        }
    }
}

#[test]
fn cli_rejects_malformed_or_invalid_inputs() {
    let invalid = vec![
        "not json".to_string(),
        json!({"n_qubits":0,"operations":[]}).to_string(),
        json!({"n_qubits":2,"operations":[{"gate":"h","targets":[-1]}]}).to_string(),
        json!({"n_qubits":2,"operations":[{"gate":"rx","targets":[0]}]}).to_string(),
        json!({"n_qubits":2,"operations":[{"gate":"cx","targets":[1,1]}]}).to_string(),
        json!({"n_qubits":2,"operations":[{"gate":"h","targets":[0,1]}]}).to_string(),
        json!({"n_qubits":2,"operations":[{"gate":"unknown","targets":[0]}]}).to_string(),
        json!({"n_qubits":2,"operations":[{"gate":"h","targets":[0],"theta":0.1}]}).to_string(),
        json!({"n_qubits":2,"operations":[],"benchmark":true,"repeats":0}).to_string(),
        json!({"n_qubits":2,"operations":[],"max_state_bytes":1}).to_string(),
    ];
    for input in invalid {
        let output = call(&input);
        assert_eq!(output.status.code(), Some(2), "{input}");
        assert!(output.stdout.is_empty());
        let error: Value = serde_json::from_slice(&output.stderr).unwrap();
        assert!(error["error"].is_string());
    }
}

#[test]
fn shared_fixtures_match_expected_states() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/core.json");
    let input = std::fs::read_to_string(path).expect("shared core fixtures must be present");
    let fixtures: Vec<Value> = serde_json::from_str(&input).unwrap();
    assert!(!fixtures.is_empty());
    for fixture in fixtures {
        let output = call(&fixture.to_string());
        assert!(
            output.status.success(),
            "{}: {}",
            fixture["name"],
            String::from_utf8_lossy(&output.stderr)
        );
        let actual: Value = serde_json::from_slice(&output.stdout).unwrap();
        let expected = fixture["expected_state"]
            .as_array()
            .expect("fixture must have expected_state");
        let state = actual["state"].as_array().unwrap();
        assert_eq!(state.len(), expected.len());
        for (a, e) in state.iter().zip(expected) {
            for component in 0..2 {
                assert!(
                    (a[component].as_f64().unwrap() - e[component].as_f64().unwrap()).abs() < 1e-12,
                    "fixture {}: {a} != {e}",
                    fixture["name"]
                );
            }
        }
    }
}
