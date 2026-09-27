"""Run phase interference and GHZ correlations as separate, inspectable mechanisms."""
import numpy as np
from qsim_tutorial import StatevectorSimulator


def main():
    plus, minus = StatevectorSimulator(1), StatevectorSimulator(1)
    plus.h(0)
    minus.h(0)
    minus.z(0)
    np.testing.assert_allclose(plus.probabilities(), minus.probabilities(), atol=1e-12, rtol=0)
    print("Before H: |+> and |-> have probabilities", plus.probabilities())
    plus.h(0)
    minus.h(0)
    np.testing.assert_allclose(plus.probabilities(), [1, 0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(minus.probabilities(), [0, 1], atol=1e-12, rtol=0)
    print("After H:", plus.probabilities(), "and", minus.probabilities())

    ghz = StatevectorSimulator(3, seed=1729)
    ghz.h(0)
    ghz.cx(0, 1)
    ghz.cx(0, 2)
    expected = np.zeros(8)
    expected[0] = expected[7] = 0.5
    np.testing.assert_allclose(ghz.probabilities(), expected, atol=1e-12, rtol=0)
    assert np.isclose(ghz.expectation("XXX"), 1, atol=1e-12, rtol=0)
    print("GHZ probabilities:", ghz.probabilities())
    print("GHZ <XXX>:", ghz.expectation("XXX"))
    print("GHZ terminal counts:", ghz.sample(128))
    measurements = [ghz.measure(q) for q in (1, 0, 2)]
    assert len(set(measurements)) == 1
    print("GHZ sequential outcomes q1,q0,q2:", measurements)


if __name__ == "__main__":
    main()
