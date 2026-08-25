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
- 唯一参数选择报告：[`docs/experiments/参数选择.md`](docs/experiments/参数选择.md)
- 主实验 CSV（只保留相对 Lightning GPU native 的外部加速比）：[`benchmark/results/main_experiment.csv`](benchmark/results/main_experiment.csv)
- 参数选择三级 CSV（全固定 → 结构已选/执行固定 → 最佳）：[`benchmark/results/parameter_selection_stages.csv`](benchmark/results/parameter_selection_stages.csv)
- 参数搜索 CSV（保持融合/对角策略相同，比较按场景执行参数 dispatch 与统一保守 `F128r2/B128r2` + canonical phase）：[`benchmark/results/parameter_search_experiment.csv`](benchmark/results/parameter_search_experiment.csv)
- 结构与对角策略 CSV（同 shape/phase 的单变量相邻配对）：[`benchmark/results/structure_strategy_experiment.csv`](benchmark/results/structure_strategy_experiment.csv)

主实验复现入口：

```bash
.venv/bin/python benchmark/benchmark_sad.py
.venv/bin/python benchmark/benchmark_sad_fixed_parameters.py
.venv/bin/python benchmark/benchmark_parameter_policy.py
.venv/bin/python benchmark/benchmark_structure_policy.py --category fusion --resume
.venv/bin/python benchmark/benchmark_structure_policy.py --category diagonal --resume
.venv/bin/python benchmark/benchmark_structure_policy.py --category bond_schedule --resume
.venv/bin/python benchmark/benchmark_parameter_selection_stages.py
.venv/bin/python benchmark/benchmark_lightning_native.py
.venv/bin/python benchmark/generate_experiment_tables.py
.venv/bin/python benchmark/generate_parameter_selection_report.py
```

执行策略搜索可复现入口：

```bash
python3 benchmark/search_execution_strategies.py --preset production \
  --stage shape --qubits 4,6,8,10,12,14,16,18 \
  --output benchmark/results/execution_search_production.csv
python3 benchmark/search_small_qubit_shapes.py
python3 benchmark/search_phase_plans_end_to_end.py
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

现有搜索器采用轮换/打乱测量顺序并保存原始逐次样本；重复执行会按完整实验键断点续跑。`production` 预设先以 `m1` 稀疏筛选 L7--L11，schedule 再用 DP 遍历硬件等价类，只生成最少 phase 数及多一个 phase 的 frontier。历史宽 mailbox 网格仅作研究证据保留；L2 hit 与 mailbox/active-CTA 未进入生产 dispatch。后续按结构、diagonal、rotation phase 分层做增量搜索。具体候选裁剪规则、参数字符串语法及当前 A/B 的可归因边界见《参数选择》；模型只用于筛选候选，不替代目标机器上的实测确认。
