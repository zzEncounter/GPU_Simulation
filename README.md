# PennyLane Qubit Rotation Demo (Current Baseline)

这个仓库当前聚焦于两条可对比的实现路径：

- `PennyLane baseline`：基于 `lightning.gpu + adjoint`。
- `Standalone CUDA backend`：自定义 C++/CUDA 扩展，直接提供 `energy` 与 `energy_and_grad`。

当前目标是保持一个可维护、可扩展、便于性能实验的基线实现。

多人协作说明见 [COLLABORATION.md](./COLLABORATION.md)（当前统一远端：`git@github.com:zzEncounter/GPU_Simulation.git`）。

## 目录结构

- `ring_ising/workflows/`：两条主工作流实现（`pennylane.py` 与 `standalone.py`）。
- `ring_ising/training/`：共享训练循环与统一返回结果模型。
- `ring_ising/baseline/` 与 `standalone_backend/workflow.py`：兼容层（旧导入路径仍可用）。
- `ring_ising/`：其余前端 Python 包内容（CLI、模型构件、runtime 辅助层）。
- `run_pennylane_baseline.py`：PennyLane 基线路径入口兼容包装。
- `run_standalone_backend.py`：Standalone CUDA 路径入口兼容包装。
- `standalone_backend/`：Python 侧 runtime。
- `cpp/`：C++/CUDA 扩展实现。
- `benchmarks/compare_pennylane_saveall_checkpoint.py`：PennyLane 与 standalone 策略对比脚本。
- `benchmarks/compare_save_vs_dense_scan.py`：`save_param_states` 与 `dense_scan` 专项对比脚本（含时间拆解与 GPU 遥测）。
- `tests/test_backends_parity.py`：数值一致性测试。
- `old/`：历史归档（已从版本控制排除）。

## Standalone Backend 当前策略

梯度内存策略仅保留：

- `save_param_states`
- `checkpoint`
- `dense_scan`（实验模式，要求 `num_qubits <= 6`）
- `auto`（仅在上面两者间自动选择）

补充说明：

- `save_all` 路径已经从对外接口移除。
- 默认开启 gate fusion。
- 如需 A/B 对照，可用 `--disable-gate-fusion` 关闭。

## C++/CUDA 拆分状态

为提升可维护性，`cpp` 已拆分为：

- `ising_cuda_backend.cu`：调度、策略流程、对外接口。
- `ising_cuda_kernels.cu`：kernel、launch wrapper、inner product/fused grad。
- `ising_cuda_backend_internal.cuh`：内部共享类型与函数声明。
- `ising_cuda_bindings.cpp`：pybind11 绑定。

## 环境安装

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 构建扩展

```bash
.venv/bin/python setup.py build_ext --inplace
```

如果修改了 `cpp/*.cu` 或 `cpp/*.cpp`，需要重新执行上面命令。

## 运行

### PennyLane baseline

```bash
.venv/bin/python run_pennylane_baseline.py
```

### Standalone backend

```bash
.venv/bin/python run_standalone_backend.py --qubits 12 --layers 3
```

可选参数示例：

```bash
# 指定梯度策略
.venv/bin/python run_standalone_backend.py --gradient-strategy save_param_states
.venv/bin/python run_standalone_backend.py --gradient-strategy checkpoint
.venv/bin/python run_standalone_backend.py --gradient-strategy dense_scan

# checkpoint 粒度
.venv/bin/python run_standalone_backend.py --gradient-strategy checkpoint --checkpoint-interval 8

# 关闭 gate fusion 做对照
.venv/bin/python run_standalone_backend.py --disable-gate-fusion

# 关闭 standalone 后端内部细粒度计时（仅保留训练循环总时间）
.venv/bin/python run_standalone_backend.py --disable-backend-timings
```

## 函数调用接口

两条路径都支持直接函数调用，并返回统一风格的结构化结果（最终能量、训练循环时间、step 指标、可选 GPU 遥测，standalone 还可包含后端细粒度计时）：

```python
from ring_ising.baseline import BaselineConfig, run_baseline
from standalone_backend import StandaloneRunConfig, run_standalone

baseline_result = run_baseline(BaselineConfig(num_qubits=12, layers=3, steps=20))
standalone_result = run_standalone(
    StandaloneRunConfig(num_qubits=12, layers=3, steps=20, measure_backend_timings=True)
)
```

## 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Benchmark

```bash
.venv/bin/python benchmarks/compare_pennylane_saveall_checkpoint.py \
  --cases 20x4 \
  --standalone-modes save_param_states checkpoint auto dense_scan
```

`dense_scan` 仅适用于 `--cases` 中 `qubits <= 6` 的条目。

可选：

```bash
.venv/bin/python benchmarks/compare_pennylane_saveall_checkpoint.py \
  --cases 20x4 \
  --standalone-modes save_param_states checkpoint \
  --disable-gate-fusion
```

专项对比（仅 `save_param_states` vs `dense_scan`）：

```bash
.venv/bin/python benchmarks/compare_save_vs_dense_scan.py \
  --cases 5x8 6x8
```
