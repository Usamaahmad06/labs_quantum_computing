"""P1: Quantum Desk Calculator

Student: Usama Ahmad  |  ID: VR513221
Quantum Computing, University of Verona, AA 2025-2026
Prof. Alessandra Di Pierro

Quantum circuits that add and subtract the binary representations of two
positive integers. Each circuit is reversible: the inputs stay in their
registers and the result is written into a fresh sum register.

Demonstrated instances
    addition:       3 + 5 = 8,   6 + 2 = 8
    subtraction:    7 - 3 = 4,   5 - 1 = 4
"""

import sys

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


N_BITS = 3  # each input is a 3-bit positive integer (0..7)


def _full_adder(qc, a, b, cin, sum_bit, cout, scratch):
    """One reversible full-adder step. Inputs a, b, cin are left unchanged.

    sum_bit <- a XOR b XOR cin
    cout    <- majority(a, b, cin) = (a AND b) XOR (cin AND (a XOR b))
    """
    qc.cx(a, scratch)
    qc.cx(b, scratch)

    qc.cx(scratch, sum_bit)
    qc.cx(cin, sum_bit)

    qc.ccx(a, b, cout)
    qc.ccx(cin, scratch, cout)

    qc.cx(b, scratch)
    qc.cx(a, scratch)


def _ripple_add(qc, a, b, carry, sum_reg, scratch, b_negated=False):
    """Ripple-carry a + b into sum_reg (n + 1 bits). carry[0] is the initial carry."""
    n = len(a)
    if b_negated:
        for i in range(n):
            qc.x(b[i])

    for i in range(n):
        _full_adder(qc, a[i], b[i], carry[i], sum_reg[i], carry[i + 1], scratch)

    qc.cx(carry[n], sum_reg[n])

    if b_negated:
        for i in range(n):
            qc.x(b[i])


def build_adder(a_value, b_value, n_bits=N_BITS):
    """|a>|b>|0> -> |a>|b>|a + b>. Returns the circuit and register map."""
    a = QuantumRegister(n_bits, "a")
    b = QuantumRegister(n_bits, "b")
    carry = QuantumRegister(n_bits + 1, "c")
    total = QuantumRegister(n_bits + 1, "sum")
    scratch = QuantumRegister(1, "scratch")
    result = ClassicalRegister(n_bits + 1, "result")

    qc = QuantumCircuit(a, b, carry, total, scratch, result, name="add")
    _load(qc, a, a_value, n_bits)
    _load(qc, b, b_value, n_bits)
    _ripple_add(qc, a, b, carry, total, scratch[0])
    qc.measure(total, result)
    return qc


def build_subtractor(a_value, b_value, n_bits=N_BITS):
    """|a>|b>|0> -> |a>|b>|a - b| for a >= b.

    Subtraction is addition of the two's complement: a + (~b) + 1.
    The extra +1 is the initial carry. The lower n bits are the difference.
    """
    a = QuantumRegister(n_bits, "a")
    b = QuantumRegister(n_bits, "b")
    carry = QuantumRegister(n_bits + 1, "c")
    total = QuantumRegister(n_bits + 1, "diff")
    scratch = QuantumRegister(1, "scratch")
    result = ClassicalRegister(n_bits, "result")

    qc = QuantumCircuit(a, b, carry, total, scratch, result, name="sub")
    _load(qc, a, a_value, n_bits)
    _load(qc, b, b_value, n_bits)
    qc.x(carry[0])  # the +1 of two's complement
    _ripple_add(qc, a, b, carry, total, scratch[0], b_negated=True)
    qc.measure(total[:n_bits], result)
    return qc


def _load(qc, register, value, n_bits):
    for i in range(n_bits):
        if (value >> i) & 1:
            qc.x(register[i])


def _run(qc, shots=1024):
    backend = AerSimulator()
    compiled = transpile(qc, backend)
    job = backend.run(compiled, shots=shots)
    counts = job.result().get_counts()
    bitstring = max(counts, key=counts.get)
    return int(bitstring, 2), counts[bitstring], shots


def main():
    print("P1 Quantum Desk Calculator")
    print(f"Inputs are {N_BITS}-bit positive integers. Simulator: Aer, 1024 shots.\n")

    additions = [(3, 5), (6, 2)]
    print("Addition  |a>|b>|0> -> |a>|b>|a+b>")
    for a_value, b_value in additions:
        qc = build_adder(a_value, b_value)
        got, hits, shots = _run(qc)
        expected = a_value + b_value
        print(
            f"  {a_value} + {b_value} = {got}"
            f"   (expected {expected}, {hits}/{shots} shots)"
        )
        if got != expected:
            raise SystemExit(f"adder failed on {a_value}+{b_value}")

    print("\nCircuit for 3 + 5:")
    print(build_adder(3, 5).draw(output="text"))

    subtractions = [(7, 3), (5, 1)]
    print("Subtraction  |a>|b>|0> -> |a>|b>|a-b>   (a >= b)")
    for a_value, b_value in subtractions:
        qc = build_subtractor(a_value, b_value)
        got, hits, shots = _run(qc)
        expected = a_value - b_value
        print(
            f"  {a_value} - {b_value} = {got}"
            f"   (expected {expected}, {hits}/{shots} shots)"
        )
        if got != expected:
            raise SystemExit(f"subtractor failed on {a_value}-{b_value}")

    print("\nBoth instances of each operation match the classical result.")


if __name__ == "__main__":
    main()
