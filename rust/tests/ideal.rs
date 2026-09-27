use num_complex::Complex64 as C;
use qsim_tutorial::{DEFAULT_MAX_STATE_BYTES, StatevectorSimulator};

fn sim(n: usize) -> StatevectorSimulator {
    StatevectorSimulator::new(n, Some(1729), DEFAULT_MAX_STATE_BYTES).unwrap()
}
fn c(re: f64, im: f64) -> C {
    C::new(re, im)
}
fn assert_state(sim: &StatevectorSimulator, expected: &[C]) {
    assert_eq!(sim.state().len(), expected.len());
    for (actual, expected) in sim.state().iter().zip(expected) {
        assert!(
            (*actual - *expected).norm() < 1e-12,
            "{actual:?} != {expected:?}"
        );
    }
}

#[test]
fn zero_state_and_little_endian_indexing() {
    let mut s = sim(3);
    assert_eq!(s.n_qubits(), 3);
    s.x(0).unwrap();
    assert_eq!(s.probabilities().unwrap()[1], 1.0);
    s.x(2).unwrap();
    assert_eq!(s.probabilities().unwrap()[5], 1.0);
    assert_eq!(s.sample(10).unwrap().get("101"), Some(&10));
}

#[test]
fn bell_and_ghz_have_correct_amplitudes() {
    let mut s = sim(3);
    s.h(0).unwrap();
    s.cx(0, 1).unwrap();
    s.cx(1, 2).unwrap();
    let a = std::f64::consts::FRAC_1_SQRT_2;
    let mut expected = vec![c(0.0, 0.0); 8];
    expected[0] = c(a, 0.0);
    expected[7] = c(a, 0.0);
    assert_state(&s, &expected);
    assert!((s.expectation("XXX").unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn phase_gates_and_y_signs() {
    let mut s = sim(1);
    s.y(0).unwrap();
    assert_state(&s, &[c(0.0, 0.0), c(0.0, 1.0)]);
    s.y(0).unwrap();
    assert_state(&s, &[c(1.0, 0.0), c(0.0, 0.0)]);
    s.h(0).unwrap();
    s.s(0).unwrap();
    let a = std::f64::consts::FRAC_1_SQRT_2;
    assert_state(&s, &[c(a, 0.0), c(0.0, a)]);
    assert!((s.expectation("Y").unwrap() - 1.0).abs() < 1e-12);
    s.t(0).unwrap();
    assert_state(&s, &[c(a, 0.0), c(-0.5, 0.5)]);
}

#[test]
fn interference_is_not_classical_coin_flipping() {
    let mut s = sim(1);
    s.h(0).unwrap();
    s.z(0).unwrap();
    s.h(0).unwrap();
    assert_state(&s, &[c(0.0, 0.0), c(1.0, 0.0)]);
}

#[test]
fn rotations_match_analytic_values_and_inverse() {
    let angle = 0.713;
    let (sine, cosine) = (angle / 2.0_f64).sin_cos();
    let mut x = sim(1);
    x.rx(angle, 0).unwrap();
    assert_state(&x, &[c(cosine, 0.0), c(0.0, -sine)]);
    x.rx(-angle, 0).unwrap();
    assert_state(&x, &[c(1.0, 0.0), c(0.0, 0.0)]);
    let mut y = sim(1);
    y.ry(angle, 0).unwrap();
    assert_state(&y, &[c(cosine, 0.0), c(sine, 0.0)]);
    y.ry(-angle, 0).unwrap();
    assert_state(&y, &[c(1.0, 0.0), c(0.0, 0.0)]);
    let mut z = sim(1);
    z.h(0).unwrap();
    z.rz(angle, 0).unwrap();
    let a = std::f64::consts::FRAC_1_SQRT_2;
    assert_state(&z, &[c(a * cosine, -a * sine), c(a * cosine, a * sine)]);
    z.rz(-angle, 0).unwrap();
    z.h(0).unwrap();
    assert_state(&z, &[c(1.0, 0.0), c(0.0, 0.0)]);
}

#[test]
fn controlled_gates_and_swap_are_asymmetric_where_expected() {
    let mut s = sim(3);
    s.x(0).unwrap();
    s.cx(2, 0).unwrap(); // inactive control
    assert_eq!(s.probabilities().unwrap()[1], 1.0);
    s.cx(0, 2).unwrap();
    assert_eq!(s.probabilities().unwrap()[5], 1.0);
    s.cz(2, 0).unwrap();
    assert_eq!(s.state()[5], c(-1.0, 0.0));
    s.swap(0, 1).unwrap();
    assert_eq!(s.state()[6], c(-1.0, 0.0));
    s.cx(2, 1).unwrap();
    assert_eq!(s.state()[4], c(-1.0, 0.0));
}

#[test]
fn generic_unitary_uses_columns_and_does_not_overwrite_input_early() {
    let mut s = sim(2);
    let a = std::f64::consts::FRAC_1_SQRT_2;
    s.apply_gate([[c(a, 0.0), c(0.0, a)], [c(0.0, a), c(a, 0.0)]], 1)
        .unwrap();
    assert_state(&s, &[c(a, 0.0), c(0.0, 0.0), c(0.0, a), c(0.0, 0.0)]);
}

#[test]
fn set_state_validates_and_preserves_state_after_rejection() {
    let mut s = sim(1);
    s.set_state(&[c(0.0, 0.0), c(0.0, 1.0)]).unwrap();
    let original = s.state().to_vec();
    for bad in [
        vec![c(1.0, 0.0)],
        vec![c(1.0, 0.0), c(1.0, 0.0)],
        vec![c(f64::NAN, 0.0), c(0.0, 0.0)],
        vec![c(f64::MAX, 0.0), c(0.0, 0.0)],
    ] {
        assert!(s.set_state(&bad).is_err());
        assert_eq!(s.state(), &original);
    }
}

#[test]
fn allocation_and_register_size_are_checked_before_allocation() {
    assert!(StatevectorSimulator::new(0, Some(0), 1024).is_err());
    assert!(StatevectorSimulator::new(usize::BITS as usize, Some(0), usize::MAX).is_err());
    assert!(StatevectorSimulator::new((usize::BITS - 1) as usize, Some(0), usize::MAX).is_err());
    assert!(StatevectorSimulator::new(3, Some(0), 127).is_err());
    assert!(StatevectorSimulator::new(3, Some(0), 128).is_ok());
}

#[test]
fn invalid_targets_and_nonfinite_rotations_are_rejected_atomically() {
    let mut s = sim(2);
    let original = s.state().to_vec();
    assert!(s.h(2).is_err());
    assert!(s.x(usize::MAX).is_err());
    assert!(s.cx(0, 0).is_err());
    assert!(s.cx(2, 1).is_err());
    assert!(s.cz(1, 1).is_err());
    assert!(s.swap(0, 0).is_err());
    assert!(s.swap(1, 2).is_err());
    assert!(s.measure(2).is_err());
    assert!(s.reset(2).is_err());
    assert!(s.rx(f64::NAN, 0).is_err());
    assert!(s.ry(f64::INFINITY, 0).is_err());
    assert!(s.rz(f64::NEG_INFINITY, 0).is_err());
    assert_eq!(s.state(), &original);
}

#[test]
fn invalid_unitaries_are_rejected_atomically() {
    let mut s = sim(1);
    let original = s.state().to_vec();
    for matrix in [
        [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(0.0, 0.0)]],
        [[c(1.0, 0.0), c(1.0, 0.0)], [c(0.0, 0.0), c(0.0, 0.0)]],
        [[c(f64::NAN, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]],
        [[c(f64::MAX, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]],
    ] {
        assert!(s.apply_gate(matrix, 0).is_err());
        assert_eq!(s.state(), &original);
    }
}

#[test]
fn pauli_order_and_imaginary_phase_are_checked() {
    let mut s = sim(2);
    s.x(0).unwrap();
    assert_eq!(s.expectation("IZ").unwrap(), -1.0);
    assert_eq!(s.expectation("ZI").unwrap(), 1.0);
    assert_eq!(s.expectation("II").unwrap(), 1.0);
    assert!(s.expectation("ix").is_err());
    assert!(s.expectation("X").is_err());
    assert!(s.expectation("A!").is_err());
    assert!(s.expectation("é").is_err());
    let mut bell = sim(2);
    bell.h(0).unwrap();
    bell.cx(0, 1).unwrap();
    for (p, e) in [("XX", 1.0), ("YY", -1.0), ("ZZ", 1.0), ("XY", 0.0)] {
        assert!((bell.expectation(p).unwrap() - e).abs() < 1e-12);
    }
}

#[test]
fn measurement_is_repeatable_and_collapses_entangled_partner() {
    for seed in 0..64 {
        let mut s = StatevectorSimulator::new(2, Some(seed), 1024).unwrap();
        s.h(0).unwrap();
        s.cx(0, 1).unwrap();
        let result = s.measure(0).unwrap();
        assert_eq!(s.measure(0).unwrap(), result);
        assert_eq!(s.measure(1).unwrap(), result);
        assert!((s.probabilities().unwrap().iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(
            s.probabilities().unwrap()[if result == 0 { 0 } else { 3 }],
            1.0
        );
    }
}

#[test]
fn deterministic_measurements_and_reset_do_not_divide_by_zero() {
    let mut s = sim(1);
    assert_eq!(s.measure(0).unwrap(), 0);
    s.x(0).unwrap();
    assert_eq!(s.measure(0).unwrap(), 1);
    s.reset(0).unwrap();
    assert_eq!(s.measure(0).unwrap(), 0);
    let mut bell = sim(2);
    bell.h(0).unwrap();
    bell.cx(0, 1).unwrap();
    bell.reset(0).unwrap();
    assert_eq!(bell.measure(0).unwrap(), 0);
    assert!((bell.probabilities().unwrap().iter().sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn terminal_sampling_is_seeded_nonmutating_and_statistically_plausible() {
    let mut a = sim(2);
    let mut b = sim(2);
    for s in [&mut a, &mut b] {
        s.h(0).unwrap();
        s.cx(0, 1).unwrap();
    }
    let original = a.state().to_vec();
    assert!(a.sample(0).unwrap().is_empty());
    let counts = a.sample(10_000).unwrap();
    assert_eq!(counts, b.sample(10_000).unwrap());
    assert_eq!(a.state(), &original);
    assert_eq!(counts.values().sum::<usize>(), 10_000);
    assert_eq!(counts.len(), 2);
    assert!(counts["00"] > 4_700 && counts["00"] < 5_300);
    assert!(counts.contains_key("11"));
}

#[test]
fn material_norm_drift_is_rejected_at_observation_boundaries() {
    let mut s = sim(1);
    // Individually accepted as numerically unitary; repetition creates material drift.
    let scale = (1.0 + 5e-11_f64).sqrt();
    let near_unitary = [[c(scale, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(scale, 0.0)]];
    for _ in 0..16 {
        s.apply_gate(near_unitary, 0).unwrap();
    }
    let before = s.state().to_vec();
    assert!(s.probabilities().is_err());
    assert!(s.sample(8).is_err());
    assert!(s.measure(0).is_err());
    assert!(s.reset(0).is_err());
    assert!(s.expectation("Z").is_err());
    assert_eq!(s.state(), &before);
}

#[test]
fn tiny_norm_drift_is_corrected_only_at_defined_boundaries() {
    let mut s = sim(1);
    let scale = (1.0 + 5e-11_f64).sqrt();
    s.set_state(&[c(scale, 0.0), c(0.0, 0.0)]).unwrap();
    assert_state(&s, &[c(1.0, 0.0), c(0.0, 0.0)]);
    s.apply_gate(
        [[c(scale, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(scale, 0.0)]],
        0,
    )
    .unwrap();
    let before = s.state().to_vec();
    assert_eq!(s.probabilities().unwrap(), vec![1.0, 0.0]);
    assert_eq!(s.sample(8).unwrap().get("0"), Some(&8));
    assert_eq!(s.state(), &before);
    assert_eq!(s.measure(0).unwrap(), 0);
    assert_state(&s, &[c(1.0, 0.0), c(0.0, 0.0)]);
}

#[test]
fn inverse_gates_restore_random_complex_states_across_targets() {
    use rand::{Rng, SeedableRng, rngs::StdRng};
    let mut rng = StdRng::seed_from_u64(8128);
    for n in 1..=5 {
        let mut state: Vec<C> = (0..1usize << n)
            .map(|_| c(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
            .collect();
        let norm = state.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        for amplitude in &mut state {
            *amplitude /= norm;
        }
        let mut s = sim(n);
        s.set_state(&state).unwrap();
        for target in 0..n {
            s.h(target).unwrap();
            s.h(target).unwrap();
            s.x(target).unwrap();
            s.x(target).unwrap();
            s.y(target).unwrap();
            s.y(target).unwrap();
            s.z(target).unwrap();
            s.z(target).unwrap();
            let theta = rng.random_range(-3.0..3.0);
            s.rx(theta, target).unwrap();
            s.rx(-theta, target).unwrap();
            s.ry(theta, target).unwrap();
            s.ry(-theta, target).unwrap();
            s.rz(theta, target).unwrap();
            s.rz(-theta, target).unwrap();
            assert_state(&s, &state);
            for control in 0..n {
                if control == target {
                    continue;
                }
                s.cx(control, target).unwrap();
                s.cx(control, target).unwrap();
                s.cz(control, target).unwrap();
                s.cz(control, target).unwrap();
                s.swap(control, target).unwrap();
                s.swap(control, target).unwrap();
                assert_state(&s, &state);
            }
        }
    }
}
