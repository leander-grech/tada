# Dependency setup

`requirements-windowed.txt` specifies the JAX training and evaluation dependencies. The bundled simulator wheel requires CPython 3.12 on Linux x86_64 with glibc 2.34 or newer.

Use the CPU installation commands in the [single-agent guide](../README.md#setup). For CUDA training, install the requirements directly with a compatible NVIDIA driver:

```sh
python -m pip install -r recipes/requirements-windowed.txt
python -m pip install simulator/wheels/flight_simulator-0.2.80-cp312-cp312-manylinux_2_34_x86_64.whl
```

Choose the visible devices before starting Python. `XLA_PYTHON_CLIENT_PREALLOCATE=false` allows JAX to allocate device memory as needed. Model evaluation uses PyTorch on CPU.
