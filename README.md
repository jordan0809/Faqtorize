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