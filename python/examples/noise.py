"""Compare trajectory observables with a tiny exact density-matrix reference."""
import json
import numpy as np
from qsim_tutorial import StatevectorSimulator, DensityMatrixSimulator, amplitude_damping


def main():
    gamma, trajectories = 0.23, 6000
    noise = amplitude_damping(gamma)
    reference = DensityMatrixSimulator(1)
    reference.h(0)
    reference.apply_channel(noise, 0)
    rng = np.random.default_rng(1729)
    values = []
    ensemble = np.zeros((2, 2), complex)
    for _ in range(trajectories):
        sim = StatevectorSimulator(1, seed=rng)
        sim.h(0)
        sim.apply_channel(noise, 0)
        values.append(sim.expectation("X"))
        psi = sim.state
        ensemble += np.outer(psi, psi.conj())
    mean, standard_error = float(np.mean(values)), float(np.std(values, ddof=1)/np.sqrt(trajectories))
    exact = reference.expectation("X")
    assert abs(mean-exact) <= 6*standard_error
    print(json.dumps({"gamma": gamma, "trajectories": trajectories, "observable": "X",
                      "trajectory_mean": mean, "density_reference": exact,
                      "estimated_standard_error": standard_error,
                      "absolute_error": abs(mean-exact)}, indent=2))
    print("Ensemble density:\n", ensemble / trajectories)
    print("Reference density:\n", reference.state)


if __name__ == "__main__":
    main()
