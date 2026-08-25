# Independent cuStateVec inverse-walk backend

This directory is independent from `sad/`.  Circuit gate sequences are built
by `python/sad_cuquantum/runner.py` and passed to
`build/libcuquantum_sad.so`.  The native CUDA implementation applies RX, RY,
RZ, and RZZ with `custatevecApplyPauliRotation`, applies CNOT and derivative
matrices with `custatevecApplyMatrix`, and performs the same forward,
Hamiltonian, and inverse-walk recurrence as the correctness reference.

Build explicitly with:

```bash
make CUDA_ARCH=sm_89
```

The Python runner also builds the library on first use when it is missing.
`SAD_CUQUANTUM_NATIVE=0` disables the native path only for parity/debug tests.

Run the repository benchmark with:

```bash
/home/rzzhang/sad/.venv/bin/python \
  /home/rzzhang/sad/benchmark/benchmark_cuQuantum.py
```
