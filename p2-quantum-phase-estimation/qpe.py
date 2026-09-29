"""P2: Quantum Phase Estimation

Student: Usama Ahmad  |  ID: VR513221
Quantum Computing, University of Verona, AA 2025-2026
Prof. Alessandra Di Pierro

Estimate the eigenphase of the controlled phase gate U = CR_phi on the
eigenvector |11>.

    CR_phi |11> = e^{i phi} |11> = e^{2 pi i theta} |11>
    so theta = phi / (2 pi), with 0 < theta < 1.

Controlled-U^{2^k} is CR_(phi * 2^k), because (e^{i phi})^{2^k} = e^{i phi 2^k}.
That gate is applied only when the phase qubit and both target qubits are |1|.

Demonstrated instances (exact with 3 phase qubits)
    phi = pi/2  -> theta = 1/4 = 0.010 in binary
    phi = pi/4  -> theta = 1/8 = 0.001 in binary
"""

import sys

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


def inverse_qft(qc, qubits):
    """Inverse QFT. qubits[0] is the least significant bit.

    Adjoint of the standard QFT that ends with bit-reversal: swap first,
    then Hadamard and negative controlled-phase gates from qubit 0 upward.
    """
    n = len(qubits)
    for j in range(n // 2):
        qc.swap(qubits[j], qubits[n - 1 - j])
    for j in range(n):
        for k in range(j):
            qc.cp(-np.pi / 2 ** (j - k), qubits[j], qubits[k])
        qc.h(qubits[j])


def build_qpe(phi, n_phase=3):
    """QPE for U = CR_phi, eigenvector |11>.

    phase[k] controls U^{2^k}. After the inverse QFT the measured integer
    m satisfies theta ≈ m / 2^{n_phase}.
    """
    phase = QuantumRegister(n_phase, "phase")
    target = QuantumRegister(2, "u")
    measured = ClassicalRegister(n_phase, "m")
    qc = QuantumCircuit(phase, target, measured, name="qpe")

    qc.x(target[0])
    qc.x(target[1])
    qc.h(phase)

    for k in range(n_phase):
        # controlled-CR_(phi * 2^k): phase when phase[k], target[0], target[1] are |1>
        qc.mcp(phi * (2**k), [phase[k], target[0]], target[1])

    inverse_qft(qc, list(phase))
    qc.measure(phase, measured)
    return qc


def _most_likely(counts):
    bitstring = max(counts, key=counts.get)
    return bitstring, int(bitstring, 2), counts[bitstring]


def main():
    n_phase = 3
    shots = 2048
    backend = AerSimulator()

    # phi chosen so theta = phi / (2 pi) is a multiple of 1/2^{n_phase}
    demos = [
        (np.pi / 2, "pi/2"),
        (np.pi / 4, "pi/4"),
    ]

    print("P2 Quantum Phase Estimation")
    print("U = CR_phi, eigenvector |11>, theta = phi / (2 pi)")
    print(f"Phase register: {n_phase} qubits. Simulator: Aer, {shots} shots.\n")

    for phi, label in demos:
        theta = phi / (2 * np.pi)
        expected = round(theta * (2**n_phase))
        qc = build_qpe(phi, n_phase)
        compiled = transpile(qc, backend)
        counts = backend.run(compiled, shots=shots).result().get_counts()
        bitstring, measured, hits = _most_likely(counts)
        estimate = measured / (2**n_phase)
        print(
            f"phi = {label}"
            f"    theta = {theta:.4f}"
            f"    measured {bitstring} = {measured}/{2 ** n_phase} = {estimate:.4f}"
            f"    ({hits}/{shots} shots, expected integer {expected})"
        )
        if measured != expected:
            raise SystemExit(f"QPE failed for phi = {label}")

    print("\nCircuit for phi = pi/2:")
    print(build_qpe(np.pi / 2, n_phase).draw(output="text"))
    print("Both instances recover theta = phi / (2 pi).")


if __name__ == "__main__":
    main()
