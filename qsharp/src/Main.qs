namespace Tutorial {
    operation Bell() : (Result, Result) {
        use qs = Qubit[2];
        H(qs[0]);
        CNOT(qs[0], qs[1]);
        return (MResetZ(qs[0]), MResetZ(qs[1]));
    }

    operation FeedForward() : (Result, Result) {
        use qs = Qubit[2];
        H(qs[0]);
        let first = MResetZ(qs[0]);
        if first == One {
            X(qs[1]);
        }
        return (first, MResetZ(qs[1]));
    }

    operation TeleportAndCheck(theta : Double, phi : Double) : Result {
        use qs = Qubit[3];
        Ry(theta, qs[0]);
        Rz(phi, qs[0]);
        H(qs[1]);
        CNOT(qs[1], qs[2]);
        CNOT(qs[0], qs[1]);
        H(qs[0]);
        let m0 = MResetZ(qs[0]);
        let m1 = MResetZ(qs[1]);
        if m1 == One { X(qs[2]); }
        if m0 == One { Z(qs[2]); }
        // Invert the original preparation on the destination, then measure.
        Rz(-phi, qs[2]);
        Ry(-theta, qs[2]);
        return MResetZ(qs[2]);
    }
}
