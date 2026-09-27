"""Verify a complex input state after measurement and classical feed-forward."""
import numpy as np
from qsim_tutorial import Circuit


def teleportation(theta=0.713, phi=-1.129):
    return (Circuit(3, 2).add("ry", 0, theta=theta).add("rz", 0, theta=phi)
            .add("h", 1).add("cx", 1, 2).add("cx", 0, 1).add("h", 0)
            .add("measure", 0, cbit=0).add("measure", 1, cbit=1)
            .add("x", 2, condition=(1, 1)).add("z", 2, condition=(0, 1)))


def main():
    theta, phi = 0.713, -1.129
    original = np.array([np.cos(theta/2)*np.exp(-1j*phi/2), np.sin(theta/2)*np.exp(1j*phi/2)])
    result = teleportation(theta, phi).execute(seed=1729)
    amplitudes = result.state.reshape(2, 4)
    destination = amplitudes @ amplitudes.conj().T
    fidelity = float(np.vdot(original, destination @ original).real)
    assert np.isclose(fidelity, 1, atol=1e-12, rtol=0)
    print("Measured (c0,c1):", result.classical_bits)
    print("Destination density matrix:\n", destination)
    print("Fidelity:", fidelity)


if __name__ == "__main__":
    main()
