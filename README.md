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
- 文档索引：[`docs/README.md`](docs/README.md)
- 最新主实验：[`docs/experiments/主实验.md`](docs/experiments/主实验.md)
- 主实验 CSV（只保留相对 Lightning GPU native 的外部加速比）：[`benchmark/results/main_experiment.csv`](benchmark/results/main_experiment.csv)
- 参数选择 CSV（按场景 dispatch vs 统一 `F128r2/B128r2` 缺省组）：[`benchmark/results/parameter_search_experiment.csv`](benchmark/results/parameter_search_experiment.csv)

主实验复现入口：

```bash
.venv/bin/python benchmark/benchmark_sad.py
.venv/bin/python benchmark/benchmark_sad_fixed_parameters.py
.venv/bin/python benchmark/benchmark_parameter_policy.py
.venv/bin/python benchmark/benchmark_lightning_native.py
.venv/bin/python benchmark/generate_experiment_tables.py
```

执行策略搜索可复现入口：

```bash
python3 benchmark/search_execution_strategies.py --preset standard \
  --stage shape --stage calibration --qubits 20,24,26 \
  --repetitions 3 --iterations 6 \
  --output benchmark/results/execution_search_exhaustive.csv
python3 benchmark/search_execution_strategies.py --preset standard \
  --stage schedule --qubits 20,24,26 --repetitions 1 --iterations 6 \
  --output benchmark/results/execution_search_exhaustive.csv
python3 benchmark/search_execution_strategies.py --preset standard \
  --stage schedule --qubits 20,24,26 --repetitions 3 --iterations 6 \
  --refine-schedules --schedule-refine-top 5 \
  --schedule-refine-percent 5 \
  --output benchmark/results/execution_search_exhaustive.csv
python3 benchmark/search_adaptive_mailbox.py --iterations 12 --repetitions 9
python3 benchmark/analyze_adaptive_mailbox.py
python3 benchmark/search_xxz_strategies.py
python3 benchmark/search_fusion_strategies.py
python3 benchmark/generate_strategy_report.py
```

搜索器均采用轮换/打乱测量顺序并保存原始逐次样本；重复执行会按完整实验键断点续跑。RX/RY schedule 先对所有 DP frontier 候选做一次筛选，再只给每场景前 5 名及最佳值 5% 内候选补足三次测量；最终排名排除仅有一次的筛选样本。搜索报告同时给出硬件资源项、留出 qubit 验证和可移植的两阶段调优方法，模型只用于筛选候选，不替代目标机器上的实测确认。
