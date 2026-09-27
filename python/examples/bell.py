"""Run with: .venv/bin/python python/examples/bell.py (from package root)."""
import json
from qsim_tutorial import StatevectorSimulator


def main():
    sim = StatevectorSimulator(2, seed=1729)
    sim.h(0)
    sim.cx(0, 1)
    print("State:", sim.state)
    print("Probabilities:", sim.probabilities())
    print("Expectation ZZ:", sim.expectation("ZZ"))
    print("4096 nondestructive terminal samples:", json.dumps(sim.sample(4096), sort_keys=True))
    a, b = sim.measure(0), sim.measure(1)
    assert a == b
    print("One destructive measurement:", (a, b), "conditional state:", sim.state)


if __name__ == "__main__":
    main()
