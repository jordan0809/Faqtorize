# FaQTorize

This repo documents the implementation of QFT-adder-based Shor's factoring algorithm ([Beauregard 2003](https://arxiv.org/pdf/quant-ph/0205095)) using CUDA-Q (C++) and Qiskit (Python).

## Usage

**C++ Implementation (CUDA-Q)**

The C++ version utilizes the [NVIDIA CUDA-Q](https://nvidia.github.io/cuda-quantum/latest/index.html) toolchain for high-performance quantum simulation.

1. **Compilation**: Navigate to the `cudaq` directory and use the provided `Makefile`:

```bash
cd cudaq
make
```
This will generate the `shor.exe` executable using the `nvq++` compiler.

2. **Execution**: Run the executable by providing the number to factor ($N$), the coprime base ($a$), and an optional number of phase qubits ($t$).
```bash
# Syntax: ./shor.exe N a [t]
# Example: Factor 15 with base 7
./shor.exe 15 7
```

- **N**: The integer you wish to factorize.
- **a**: A chosen integer coprime to $N$.
- **t** (optional): The number of qubits in the phase register. Defaults to $n$ (bit-length of $N$) if not specified.

3. **Example Output**: When running the C++ implementation, you will see the distribution of measurement outcomes followed by the period extraction and final factors:

```bash
$ ./shor.exe 63 5
Using default t = 6
Total number of qubits = 20

Measurement outcomes: 
{ 000000:2525 000001:2553 100010:1 100011:9 100100:96 100101:113 100110:62 100111:73 101000:162 101001:153 101010:586 101011:550 101100:273 101101:299 101110:23 101111:28 110000:20 110001:25 110010:299 110011:305 110100:585 110101:544 110110:175 110111:162 111000:86 111001:79 111010:107 111011:99 111100:3 111101:5 }
Simulation runtime: 975.786s

Selected period p = 6

=== Factorization Result ===
63 = 3 x 21
```

**Python Implementation (Qiskit)**

The Python implementation provides better visualizations of the quantum circuit and the measurement distributions. All algorithmic components (Draper adders, modular exponentiation) are encapsulated within the `Shor` class in `shor_qiskit.py`.
- Interactive Demo: See `qiskit_demo.ipynb` for example usage. 

To run the Python version, ensure you have the requirements installed:
```bash
pip install -r qiskit/requirements.txt
```

**Key Performance Note**

For the C++ implementation, if you are factoring larger numbers, it is highly recommended to use an NVIDIA GPU backend (if available) for the simulation by specifying `--target nvidia`:

```bash
nvq++ shor.cpp -o shor.exe --target nvidia
```