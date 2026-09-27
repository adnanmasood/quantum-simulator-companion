//! Checked dense statevectors. Qubit zero is the least-significant index bit.
//! `Complex64` means two f64 components (128 bits per amplitude).

use num_complex::Complex64;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::{collections::BTreeMap, error::Error, fmt};

pub const DEFAULT_MAX_STATE_BYTES: usize = 268_435_456;
pub const VALIDATION_TOLERANCE: f64 = 1e-10;
pub type Matrix2 = [[Complex64; 2]; 2];
pub type Counts = BTreeMap<String, usize>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimError(pub String);

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for SimError {}
type SimResult<T> = Result<T, SimError>;

fn error(message: impl Into<String>) -> SimError {
    SimError(message.into())
}
fn finite(z: Complex64) -> bool {
    z.re.is_finite() && z.im.is_finite()
}
fn real(value: f64) -> Complex64 {
    Complex64::new(value, 0.0)
}

fn validate_norm(norm: f64) -> SimResult<()> {
    if !norm.is_finite() || (norm - 1.0).abs() > VALIDATION_TOLERANCE {
        return Err(error("state squared norm must equal one within 1e-10"));
    }
    Ok(())
}

/// State and RNG are owned. Sampling changes only the RNG; measurement changes both.
pub struct StatevectorSimulator {
    n_qubits: usize,
    state: Vec<Complex64>,
    rng: StdRng,
}

impl StatevectorSimulator {
    pub fn new(n_qubits: usize, seed: Option<u64>, max_state_bytes: usize) -> SimResult<Self> {
        if n_qubits == 0 || n_qubits >= usize::BITS as usize {
            return Err(error(
                "n_qubits must be positive and smaller than usize::BITS",
            ));
        }
        let dimension = 1usize << n_qubits;
        let bytes = dimension
            .checked_mul(std::mem::size_of::<Complex64>())
            .ok_or_else(|| error("state allocation size overflows usize"))?;
        if bytes > max_state_bytes || bytes > isize::MAX as usize {
            return Err(error(format!(
                "state requires {bytes} bytes; limit is {max_state_bytes}"
            )));
        }
        let mut state = Vec::new();
        state
            .try_reserve_exact(dimension)
            .map_err(|_| error(format!("could not allocate {bytes} bytes for state")))?;
        state.resize(dimension, real(0.0));
        state[0] = real(1.0);
        let rng = match seed {
            Some(value) => StdRng::seed_from_u64(value),
            None => StdRng::from_os_rng(),
        };
        Ok(Self {
            n_qubits,
            state,
            rng,
        })
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }
    /// Borrow immutable amplitudes; callers cannot invalidate the state.
    pub fn state(&self) -> &[Complex64] {
        &self.state
    }

    pub fn set_state(&mut self, amplitudes: &[Complex64]) -> SimResult<()> {
        if amplitudes.len() != self.state.len() || !amplitudes.iter().copied().all(finite) {
            return Err(error("state must contain 2^n finite complex amplitudes"));
        }
        let norm: f64 = amplitudes.iter().map(|z| z.norm_sqr()).sum();
        validate_norm(norm)?;
        let scale = norm.sqrt();
        for (output, input) in self.state.iter_mut().zip(amplitudes) {
            *output = *input / scale;
        }
        Ok(())
    }

    fn target(&self, target: usize) -> SimResult<()> {
        if target >= self.n_qubits {
            return Err(error(format!(
                "qubit {target} is outside a {}-qubit register",
                self.n_qubits
            )));
        }
        Ok(())
    }
    fn pair(&self, first: usize, second: usize) -> SimResult<()> {
        self.target(first)?;
        self.target(second)?;
        if first == second {
            return Err(error("two-qubit targets must be distinct"));
        }
        Ok(())
    }
    fn angle(theta: f64) -> SimResult<()> {
        if !theta.is_finite() {
            return Err(error("rotation angle must be finite"));
        }
        Ok(())
    }

    /// Apply a 2×2 unitary, using column-vector convention.
    pub fn apply_gate(&mut self, matrix: Matrix2, target: usize) -> SimResult<()> {
        self.target(target)?;
        if !matrix.iter().flatten().copied().all(finite) {
            return Err(error("gate entries must be finite"));
        }
        for a in 0..2 {
            for b in 0..2 {
                let inner = matrix[0][a].conj() * matrix[0][b] + matrix[1][a].conj() * matrix[1][b];
                let expected = real(if a == b { 1.0 } else { 0.0 });
                if !finite(inner) || (inner - expected).norm() > VALIDATION_TOLERANCE {
                    return Err(error("gate must be unitary within 1e-10"));
                }
            }
        }
        self.apply_local_unchecked(matrix, target);
        Ok(())
    }

    // No full-register matrix is formed. Copy both old amplitudes before writing.
    fn apply_local_unchecked(&mut self, u: Matrix2, target: usize) {
        let stride = 1usize << target;
        for block in (0..self.state.len()).step_by(2 * stride) {
            for offset in 0..stride {
                let i = block + offset;
                let j = i + stride;
                let a = self.state[i];
                let b = self.state[j];
                self.state[i] = u[0][0] * a + u[0][1] * b;
                self.state[j] = u[1][0] * a + u[1][1] * b;
            }
        }
    }

    pub fn h(&mut self, target: usize) -> SimResult<()> {
        let s = real(std::f64::consts::FRAC_1_SQRT_2);
        self.apply_gate([[s, s], [s, -s]], target)
    }
    pub fn x(&mut self, target: usize) -> SimResult<()> {
        self.apply_gate([[real(0.0), real(1.0)], [real(1.0), real(0.0)]], target)
    }
    pub fn y(&mut self, target: usize) -> SimResult<()> {
        let i = Complex64::new(0.0, 1.0);
        self.apply_gate([[real(0.0), -i], [i, real(0.0)]], target)
    }
    pub fn z(&mut self, target: usize) -> SimResult<()> {
        self.apply_gate([[real(1.0), real(0.0)], [real(0.0), real(-1.0)]], target)
    }
    pub fn s(&mut self, target: usize) -> SimResult<()> {
        self.apply_gate(
            [
                [real(1.0), real(0.0)],
                [real(0.0), Complex64::new(0.0, 1.0)],
            ],
            target,
        )
    }
    pub fn t(&mut self, target: usize) -> SimResult<()> {
        self.apply_gate(
            [
                [real(1.0), real(0.0)],
                [
                    real(0.0),
                    Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4),
                ],
            ],
            target,
        )
    }
    pub fn rx(&mut self, theta: f64, target: usize) -> SimResult<()> {
        Self::angle(theta)?;
        let (s, c) = (theta / 2.0).sin_cos();
        let off = Complex64::new(0.0, -s);
        self.apply_gate([[real(c), off], [off, real(c)]], target)
    }
    pub fn ry(&mut self, theta: f64, target: usize) -> SimResult<()> {
        Self::angle(theta)?;
        let (s, c) = (theta / 2.0).sin_cos();
        self.apply_gate([[real(c), real(-s)], [real(s), real(c)]], target)
    }
    pub fn rz(&mut self, theta: f64, target: usize) -> SimResult<()> {
        Self::angle(theta)?;
        self.apply_gate(
            [
                [Complex64::from_polar(1.0, -theta / 2.0), real(0.0)],
                [real(0.0), Complex64::from_polar(1.0, theta / 2.0)],
            ],
            target,
        )
    }
    pub fn cx(&mut self, control: usize, target: usize) -> SimResult<()> {
        self.pair(control, target)?;
        let c = 1usize << control;
        let t = 1usize << target;
        for i in 0..self.state.len() {
            if i & c != 0 && i & t == 0 {
                self.state.swap(i, i | t);
            }
        }
        Ok(())
    }
    pub fn cz(&mut self, control: usize, target: usize) -> SimResult<()> {
        self.pair(control, target)?;
        let mask = (1usize << control) | (1usize << target);
        for i in 0..self.state.len() {
            if i & mask == mask {
                self.state[i] = -self.state[i];
            }
        }
        Ok(())
    }
    pub fn swap(&mut self, first: usize, second: usize) -> SimResult<()> {
        self.pair(first, second)?;
        let a = 1usize << first;
        let b = 1usize << second;
        for i in 0..self.state.len() {
            if i & a == 0 && i & b != 0 {
                self.state.swap(i, i ^ a ^ b);
            }
        }
        Ok(())
    }

    pub fn probabilities(&self) -> SimResult<Vec<f64>> {
        let mut probabilities: Vec<f64> = self.state.iter().map(|z| z.norm_sqr()).collect();
        let total = probabilities.iter().sum();
        validate_norm(total)?;
        for probability in &mut probabilities {
            *probability /= total;
        }
        Ok(probabilities)
    }

    pub fn measure(&mut self, target: usize) -> SimResult<u8> {
        self.target(target)?;
        let mask = 1usize << target;
        let mut weights = [0.0f64; 2];
        for (index, amplitude) in self.state.iter().enumerate() {
            weights[usize::from(index & mask != 0)] += amplitude.norm_sqr();
        }
        let total = weights[0] + weights[1];
        validate_norm(total)?;
        let p_one = weights[1] / total;
        let outcome = usize::from(self.rng.random::<f64>() < p_one);
        if weights[outcome] <= 0.0 {
            return Err(error("sampled a zero-probability branch"));
        }
        let denominator = weights[outcome].sqrt();
        for (index, amplitude) in self.state.iter_mut().enumerate() {
            if usize::from(index & mask != 0) == outcome {
                *amplitude /= denominator;
            } else {
                *amplitude = real(0.0);
            }
        }
        Ok(outcome as u8)
    }

    pub fn reset(&mut self, target: usize) -> SimResult<()> {
        if self.measure(target)? == 1 {
            self.x(target)?;
        }
        Ok(())
    }

    /// Independent full-register draws. The quantum state is unchanged.
    pub fn sample(&mut self, shots: usize) -> SimResult<Counts> {
        let mut counts = Counts::new();
        if shots == 0 {
            return Ok(counts);
        }
        let mut cumulative = Vec::with_capacity(self.state.len());
        let mut total = 0.0;
        for amplitude in &self.state {
            total += amplitude.norm_sqr();
            cumulative.push(total);
        }
        validate_norm(total)?;
        for weight in &mut cumulative {
            *weight /= total;
        }
        for _ in 0..shots {
            let draw = self.rng.random::<f64>();
            let index = cumulative
                .partition_point(|weight| *weight <= draw)
                .min(self.state.len() - 1);
            let bitstring = format!("{index:0width$b}", width = self.n_qubits);
            *counts.entry(bitstring).or_insert(0) += 1;
        }
        Ok(counts)
    }

    /// Pauli string uses highest qubit first, e.g. "IX" acts with X on qubit 0.
    pub fn expectation(&self, pauli: &str) -> SimResult<f64> {
        if pauli.len() != self.n_qubits
            || !pauli
                .bytes()
                .all(|p| matches!(p, b'I' | b'X' | b'Y' | b'Z'))
        {
            return Err(error(
                "Pauli string must have n uppercase I/X/Y/Z characters",
            ));
        }
        validate_norm(self.state.iter().map(|z| z.norm_sqr()).sum())?;
        let mut result = real(0.0);
        for (source, amplitude) in self.state.iter().enumerate() {
            let mut destination = source;
            let mut phase = real(1.0);
            for (qubit, p) in pauli.bytes().rev().enumerate() {
                let bit = 1usize << qubit;
                let one = source & bit != 0;
                match p {
                    b'X' => destination ^= bit,
                    b'Y' => {
                        destination ^= bit;
                        phase *= Complex64::new(0.0, if one { -1.0 } else { 1.0 });
                    }
                    b'Z' if one => phase = -phase,
                    _ => {}
                }
            }
            result += self.state[destination].conj() * phase * amplitude;
        }
        if !finite(result) || result.im.abs() > VALIDATION_TOLERANCE {
            return Err(error(
                "Hermitian expectation must be finite and real within 1e-10",
            ));
        }
        Ok(result.re)
    }
}
