"""Read a deterministic shared fixture from stdin and return comparable JSON."""
import json
import sys
from qsim_tutorial import from_fixture


def main():
    fixture = json.load(sys.stdin)
    result = from_fixture(fixture).execute()
    state = result.state
    print(json.dumps({"state": [[float(z.real), float(z.imag)] for z in state],
                      "probabilities": (abs(state)**2).tolist()}))


if __name__ == "__main__":
    main()
