# VQA/QML adjoint-diff benchmarks

项目包含 PennyLane `lightning.gpu` 基线和独立的 SAD CUDA/C++ adjoint 实现，二者求解完全相同的五类电路：

- RA-HEA：`RY -> ring CNOT`
- SU2-HEA：`RY -> RZ -> ring CNOT`（主 benchmark）
- RZZ-HEA：`RX -> RZ -> even RZZ -> odd RZZ`
- QAOA：`H -> repeated(even/odd shared RZZ(gamma) -> shared RX(beta))`
- XXZ-HVA：Néel 初态，`even/odd bonds` 上逐 bond 的 `RXX -> RYY -> RZZ`

参数按 `U(-pi, pi)` 独立采样，默认随机种子为 42。三类 HEA 使用固定 TFIM Hamiltonian

```text
H = -sum_i Z_i Z_(i+1) - sum_i X_i
```

QAOA 使用环形 MaxCut cost Hamiltonian，XXZ-HVA 使用 `sum(XX+YY+0.5ZZ)`；详细定义、正确性与性能见研究报告。

入口：

- PennyLane 基线与安装：[`pennylane-lightning/README.md`](pennylane-lightning/README.md)
- SAD CUDA/C++ 实现：[`sad/README.md`](sad/README.md)
- PennyLane 批量测试：[`benchmark/benchmark_pennylane_lightning.py`](benchmark/benchmark_pennylane_lightning.py)
- 低层 Lightning-GPU native binding 测试：[`benchmark/benchmark_lightning_native.py`](benchmark/benchmark_lightning_native.py)
- SAD 批量测试：[`benchmark/benchmark_sad.py`](benchmark/benchmark_sad.py)
- energy/完整 gradient CSV 对比：[`benchmark/compare_sad_pennylane.py`](benchmark/compare_sad_pennylane.py)
- 优化研究、消融与主实验：[`OPTIMIZATION_REPORT.md`](OPTIMIZATION_REPORT.md)
- RX/RY 深入研究（mailbox、warp 连续性、persistent、backward）：[`ROTATION_OPTIMIZATION_REPORT.md`](ROTATION_OPTIMIZATION_REPORT.md)
- native/QNode/SAD 三方计时表：[`NATIVE_BASELINE_COMPARISON.md`](NATIVE_BASELINE_COMPARISON.md)
- 对角门、reduction、跨 matching fusion 与 launch follow-up：[`DIAGONAL_FUSION_REDUCTION_REPORT.md`](DIAGONAL_FUSION_REDUCTION_REPORT.md)
