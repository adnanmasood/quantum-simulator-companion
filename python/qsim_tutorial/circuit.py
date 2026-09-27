"""A small instruction list with measurement, classical conditions, and noise.

Classical output bits start at zero. Conditions require a prior measurement. Every
run() shot starts in |0...0>; terminal sample() on a simulator is a separate API.
This is an educational format, not an OpenQASM or QIR implementation.
"""
from collections import Counter
from dataclasses import dataclass
import numpy as np
from .core import StatevectorSimulator, angle, channel, integer


@dataclass(frozen=True)
class Instruction:
    gate: str
    targets: tuple[int, ...]
    theta: float | None = None
    cbit: int | None = None
    condition: tuple[int, int] | None = None
    kraus: tuple | None = None


@dataclass(frozen=True)
class CircuitResult:
    state: np.ndarray
    classical_bits: tuple[int, ...]

    @property
    def bitstring(self):
        return "".join(map(str, reversed(self.classical_bits)))


class Circuit:
    def __init__(self, n_qubits, n_clbits=None):
        self.n_qubits = integer(n_qubits, "n_qubits", 1)
        self.n_clbits = integer(n_qubits if n_clbits is None else n_clbits, "n_clbits", 1)
        self._instructions = []

    @property
    def instructions(self):
        return tuple(self._instructions)

    def _cbit(self, cbit):
        cbit = integer(cbit, "classical bit")
        if cbit >= self.n_clbits:
            raise ValueError("classical bit is outside the classical register")
        return cbit

    def add(self, gate, *targets, theta=None, cbit=None, condition=None, kraus=None):
        one = {"h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "measure", "reset", "channel"}
        two = {"cx", "cz", "swap"}
        if not isinstance(gate, str) or gate not in one | two:
            raise ValueError("unsupported instruction")
        if len(targets) != (2 if gate in two else 1):
            raise ValueError("wrong number of targets")
        targets = tuple(integer(t, "target") for t in targets)
        if max(targets) >= self.n_qubits or len(set(targets)) != len(targets):
            raise ValueError("targets must be valid and distinct")
        if gate in {"rx", "ry", "rz"}:
            theta = angle(theta)
        elif theta is not None:
            raise ValueError("theta is only valid for rotations")
        if gate == "measure":
            cbit = self._cbit(cbit)
        elif cbit is not None:
            raise ValueError("cbit is only valid for measurement")
        if condition is not None:
            if not isinstance(condition, (tuple, list)) or len(condition) != 2:
                raise ValueError("condition must be (classical_bit, 0_or_1)")
            bit, value = self._cbit(condition[0]), integer(condition[1], "condition value")
            if value not in (0, 1):
                raise ValueError("condition value must be 0 or 1")
            condition = (bit, value)
        if gate == "channel":
            if kraus is None:
                raise ValueError("channel instruction requires Kraus operators")
            ops = []
            for k in channel(kraus):
                copy = k.copy()
                copy.setflags(write=False)
                ops.append(copy)
            kraus = tuple(ops)
        elif kraus is not None:
            raise ValueError("Kraus operators are only valid for channel instructions")
        self._instructions.append(Instruction(gate, targets, theta, cbit, condition, kraus))
        return self

    def execute(self, *, seed=None, simulator=None):
        sim = StatevectorSimulator(self.n_qubits, seed=seed) if simulator is None else simulator
        if not isinstance(sim, StatevectorSimulator) or sim.n_qubits != self.n_qubits:
            raise ValueError("simulator must be a matching StatevectorSimulator")
        if simulator is not None and seed is not None:
            raise ValueError("seed belongs on the supplied simulator, not execute()")
        classical = [0] * self.n_clbits
        initialized = set()
        for ins in self._instructions:
            if ins.condition is not None:
                if ins.condition[0] not in initialized:
                    raise ValueError("condition references a classical bit not yet measured")
                if classical[ins.condition[0]] != ins.condition[1]:
                    continue
            if ins.gate == "measure":
                classical[ins.cbit] = sim.measure(ins.targets[0])
                initialized.add(ins.cbit)
            elif ins.gate == "channel":
                sim.apply_channel(ins.kraus, ins.targets[0])
            elif ins.gate in {"rx", "ry", "rz"}:
                getattr(sim, ins.gate)(ins.theta, *ins.targets)
            else:
                getattr(sim, ins.gate)(*ins.targets)
        return CircuitResult(sim.state, tuple(classical))

    def run(self, shots, *, seed=None):
        shots = integer(shots, "shots")
        rng = np.random.default_rng(seed)
        return dict(sorted(Counter(self.execute(seed=rng).bitstring for _ in range(shots)).items()))


def from_fixture(fixture):
    circuit = Circuit(fixture["n_qubits"])
    for op in fixture["operations"]:
        circuit.add(op["gate"], *op["targets"], theta=op.get("theta"))
    return circuit
