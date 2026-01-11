import numpy as np
import math
from fractions import Fraction
from collections import Counter
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFTGate
from qiskit_aer import AerSimulator


def num2bin(x: int, n: int) -> list[int]:
    """
    Convert integer x into its n-bit binary representation.

    Args:
        x (int): Integer to be converted.
        n (int): Number of bits in the binary representation.

    Returns:
        blist (list[int]): The binary representation as a list of integers.
    """
    bstr = bin(x)[2:].zfill(n)
    blist = [int(i) for i in bstr]
    return blist


def continued_fractions(phase: float, Q: int, max_den: int | None = None) -> set[int]:
    """
    Approximates a phase using the Continued Fractions algorithm to find potential periods.

    According to the theory of Diophantine approximations, if |phase - p/q| < 1/(2Q),
    then p/q is a convergent of the continued fraction expansion of the phase.
    These denominators 'q' are candidate periods for Shor's algorithm.

    Args:
        phase (float): The measured phase from the quantum circuit.
        Q (int): The total number of measurement outcomes (2^t).
        max_den (int | None): The maximum denominator to consider. Defaults to 2*Q if not provided.

    Returns:
        cand (set[int]): A set of candidate denominators (potential periods 'r').
    """
    cands = set()
    if max_den is None:
        max_den = 2 * Q
    # Initial approximation
    frac = Fraction(phase).limit_denominator(max_den)
    cands.add(frac.denominator)

    # Recursively find out the best denominators q
    # current fraction, rounded integers, previous numerator, previous denominator, current numerator, current denominator
    x, cf, h, k, h1, k1 = phase, [], 0, 1, 1, 0
    while len(cf) < 20 and x > 1e-12:
        a = int(x)
        cf.append(a)
        # Update current numerator and denominator
        h2, k2 = a * h1 + h, a * k1 + k
        p, q = h2, k2
        if 1 < q <= max_den and abs(phase - p / q) < 1 / (2 * Q):
            cands.add(q)
        h, k, h1, k1 = h1, k1, h2, k2
        x -= a
        if x > 0:
            x = 1 / x
    return cands


class Shor:
    def __init__(self, N: int, a: int, t: int | None = None):
        """
        Args:
            N (int): The number to be factorized.
            a (int): The base number used to find the period 'r' such that a^r mod N = 1.
            t (int, optional): The number of phase (control) qubits. Defaults to n if None.
        """

        if N <= 0 or a <= 0 or type(N) is not int or type(a) is not int:
            raise ValueError("'N' and 'a' must be positive integers.")

        if math.gcd(a, N) != 1:
            raise ValueError(f"Please choose an 'a' coprime to {N}.")

        self.N = N
        self.a = a

        n = math.floor(np.log2(N)) + 1  # number of bits in the binary expansion of N
        self.n = n

        if t:
            if t < 0:
                raise ValueError("'t' (number of phase qubits) must be positive.")
            if type(t) is not int:
                raise ValueError("'t' (number of phase qubits) must be an integer.")
            self.t = t
        else:
            self.t = n

    def __str__(self):
        return f"Factorizing {self.N} with the base number {self.a} and {self.t} phase qubits."

    def set_a(self, a_new: int):
        """
        Set a new value for a.

        Args:
            a_new (int): The new value for a.
        """
        if a_new <= 0 or type(a_new) is not int:
            raise ValueError("'a' must be a positive integer.")
        if math.gcd(a_new, self.N) != 1:
            raise ValueError(f"Please choose an 'a' coprime to {self.N}.")

        self.a = a_new

    def set_t(self, t_new: int):
        """
        Set a new value for t.

        Args:
            t_new (int): The new value for t.
        """
        if t_new < 0:
            raise ValueError("'t' (number of phase qubits) must be positive.")
        if type(t_new) is not int:
            raise ValueError("'t' (number of phase qubits) must be an integer.")

        self.t = t_new

    def phi_ADD(self, y: int):
        """
        Draper QFT adder:
        Input: |φ(b)> Output: |φ(y+b)>
        """
        n = self.n
        qc = QuantumCircuit(n + 1)

        # 1. Compute the phase angle of each qubit in the Draper adder
        bitstring = num2bin(y, n + 1)

        n_angles = []
        # Iterate over input qubits (from LSB)
        for i in range(n + 1):
            # accumulated phase angles
            angles = [
                2 * np.pi / 2 ** (k + 1) if bit == 1 else 0
                for k, bit in enumerate(bitstring[i:])
            ]
            n_angles.append(sum(angles))

        # Apply phase shift gates
        for i in range(n + 1):
            qc.p(n_angles[i], i)

        gate = qc.to_gate()
        gate.name = f"φADD({y})"

        return gate

    def modular_adder(self, y: int):
        """
        Modular adder gate:
        Input: |φ(b)> Output: |φ(y+b mod N)>
        """
        n = self.n
        N = self.N

        qubits = QuantumRegister(n + 1)
        ancilla = QuantumRegister(1)

        qc = QuantumCircuit(qubits, ancilla)

        add_y = self.phi_ADD(y)
        add_N = self.phi_ADD(N)
        QFT = QFTGate(n + 1)

        qc.append(add_y, qubits)
        qc.append(add_N.inverse(), qubits)
        qc.append(QFT.inverse(), qubits)
        # Check if MSB = 1 (meaning y+b < N)
        qc.cx(qubits[-1], ancilla[0])
        qc.append(QFT, qubits)
        qc.append(add_N.control(1), list(ancilla) + list(qubits))

        qc.append(add_y.inverse(), qubits)
        qc.append(QFT.inverse(), qubits)
        # Check if MSB =0 (meaning (y+b) (mod N) -y > 0)
        qc.x(qubits[-1])
        qc.cx(qubits[-1], ancilla[0])
        qc.x(qubits[-1])
        qc.append(QFT, qubits)
        qc.append(add_y, qubits)

        gate = qc.to_gate()
        gate.name = f"φADD({y}) mod N"

        return gate

    def CMULT(self, y: int):
        """
        Controlled multiplier gate:
        Input: |x>|b> Output: |x>|b+yx mod N)>
        """
        n = self.n
        N = self.N

        xreg = QuantumRegister(n)
        qubits = QuantumRegister(n + 1)
        ancilla = QuantumRegister(1)
        qc = QuantumCircuit(xreg, qubits, ancilla)

        QFT = QFTGate(n + 1)

        qc.append(QFT, qubits)

        for i in range(n):
            # Compute the constant in the modular adder (2^(i)y mod N)
            Y = 2 ** (i) * y % N
            mod_add_Y = self.modular_adder(Y)
            qc.append(mod_add_Y.control(1), [xreg[i]] + list(qubits) + list(ancilla))

        qc.append(QFT.inverse(), qubits)

        gate = qc.to_gate()
        gate.name = f"CMULT({y}) mod N"

        return gate

    def Uy(self, y: int):
        """
        Uy gate built from CMULT.
        Input: |x>|0> Output: |yx mod N>|0>
        """
        n = self.n
        N = self.N

        xreg = QuantumRegister(n)
        qubits = QuantumRegister(n + 1)
        ancilla = QuantumRegister(1)
        qc = QuantumCircuit(xreg, qubits, ancilla)

        cmult_y_N = self.CMULT(y)
        qc.append(cmult_y_N, list(xreg) + list(qubits) + list(ancilla))

        for i in range(n):
            qc.swap(xreg[i], qubits[i])

        # Modular multiplicative inverse
        y_inv = pow(y, -1, N)
        cmult_y_inv_N = self.CMULT(y_inv).inverse()
        qc.append(cmult_y_inv_N, list(xreg) + list(qubits) + list(ancilla))

        gate = qc.to_gate()
        gate.name = f"U_{y}"

        return gate

    def circuit(self) -> QuantumCircuit:
        """
        Shor's algorithm circuit.

        Returns:
            qc (QuantumCircuit): The Shor's algorithm circuit
        """
        N, a, n, t = self.N, self.a, self.n, self.t

        phase = QuantumRegister(t, "p")
        xreg = QuantumRegister(n, "x")
        qubits = QuantumRegister(n + 1, "q")
        ancilla = QuantumRegister(1, "a")
        cbits = ClassicalRegister(t)
        qc = QuantumCircuit(phase, xreg, qubits, ancilla, cbits)

        QFT = QFTGate(t)

        qc.h(phase)
        qc.x(xreg[0])  # initial |x> = |1>
        for i in range(t):
            A = a ** (2**i) % N
            u_A = self.Uy(A)
            qc.append(
                u_A.control(1), [phase[i]] + list(xreg) + list(qubits) + list(ancilla)
            )

        qc.append(QFT.inverse(), phase)
        qc.measure(phase, cbits)

        return qc

    def run_circuit(self, shots: int = 10000) -> dict[str, int]:
        """
        Run the Shor's algorithm circuit.

        Args:
            shots (int): Number of shots in the simulation.

        Returns:
            counts (dict[str,int]): The sampling results stored as a dictionary.
        """
        qc = self.circuit()
        sim = AerSimulator()
        compiled = transpile(qc, sim)

        job = sim.run(compiled, shots=shots)
        result = job.result()
        counts = result.get_counts()

        return counts

    def factorize(
        self, counts: dict[str, int] | None = None, shots: int = 10000
    ) -> tuple[int, int] | None:
        """
        Factorize the target number N.

        Args:
            counts (dict[str,int], optional): The sampling results from running the Shor's algorithm.
            shots (int): Number of shots in the simulation.

        """
        N, a, t = self.N, self.a, self.t
        if counts is None:
            counts = self.run_circuit(shots)

        # Parse counts: map bitstring to integer
        Q = 2**t
        x_counts = Counter()
        for bitstring, cnt in counts.items():
            x = int(bitstring, 2)
            x_counts[x] += cnt

        # Collect candidates, weighted by counts
        r_candidates = Counter()
        for x, total_cnt in x_counts.items():
            if total_cnt < int(shots * 0.03):
                continue  # Skip low-occurrence outcomes (< 3% of shots)
            phase = x / Q
            denoms = continued_fractions(phase, Q)
            for r in denoms:
                r_candidates[r] += total_cnt

        # Validate candidates: even r and a^r mod N = 1
        valid_r = []
        for r, weight in r_candidates.items():
            if r % 2 == 0 and pow(a, r, N) == 1:
                valid_r.append((r, weight))

        if not valid_r:
            print("No valid r. Try larger shots, more phase qubits, or new 'a'.")
        else:
            # Use smallest valid r
            p, _ = min(valid_r, key=lambda x: x[0])
            print(f"Selected period p = {p}")

            # Test all even divisors d|p for best factors
            factors = []
            divisors = [d for d in range(2, p + 1, 2) if p % d == 0]
            for d in sorted(divisors):
                exp = d // 2
                f1 = math.gcd(pow(a, exp, N) - 1, N)
                f2 = math.gcd(pow(a, exp, N) + 1, N)
                if 1 < f1 < N:
                    factors.append((f1, N // f1))
                if 1 < f2 < N:
                    factors.append((f2, N // f2))

            if factors:
                factor1, factor2 = min(set(factors), key=lambda x: x[0])
                print(f"{N} = {factor1} * {factor2}")
            else:
                print("Trivial factors only. Try new 'a'.")

            return factor1, factor2
