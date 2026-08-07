# VQA/QML adjoint-diff benchmarks

项目包含 PennyLane `lightning.gpu` 基线和独立的 SAD CUDA/C++ adjoint 实现，二者求解完全相同的三类 HEA：

- RA-HEA：`RY -> ring CNOT`
- SU2-HEA：`RY -> RZ -> ring CNOT`（主 benchmark）
- RZZ-HEA：`RX -> RZ -> even RZZ -> odd RZZ`

所有电路从全零态开始，参数按 `U(-pi, pi)` 独立采样，默认随机种子为 42。固定 Hamiltonian 为

```text
H = -sum_i Z_i Z_(i+1) - sum_i X_i
```

入口：

- PennyLane 基线与安装：[`pennylane-lightning/README.md`](pennylane-lightning/README.md)
- SAD CUDA/C++ 实现：[`sad/README.md`](sad/README.md)
- PennyLane 批量测试：[`benchmark/benchmark_pennylane_lightning.py`](benchmark/benchmark_pennylane_lightning.py)
- SAD 批量测试：[`benchmark/benchmark_sad.py`](benchmark/benchmark_sad.py)
- energy/完整 gradient CSV 对比：[`benchmark/compare_sad_pennylane.py`](benchmark/compare_sad_pennylane.py)
