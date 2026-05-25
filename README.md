# PennyLane Qubit Rotation Demo (Current Baseline)

这个仓库当前聚焦于两条可对比的实现路径：

- `PennyLane baseline`：基于 `lightning.gpu + adjoint`。
- `Standalone CUDA backend`：自定义 C++/CUDA 扩展，直接提供 `energy` 与 `energy_and_grad`。

当前目标是保持一个可维护、可扩展、便于性能实验的基线实现。

多人 SSH 协作说明见 [COLLABORATION.md](./COLLABORATION.md)。

## 目录结构

- `run_pennylane_baseline.py`：PennyLane 基线路径入口。
- `run_standalone_backend.py`：Standalone CUDA 路径入口。
- `ising_model.py`：Ising Hamiltonian 和电路相关可复用构件。
- `runtime_utils.py`：计时、资源采样等运行辅助工具。
- `standalone_backend/`：Python 侧 runtime。
- `cpp/`：C++/CUDA 扩展实现。
- `benchmarks/compare_pennylane_saveall_checkpoint.py`：PennyLane 与 standalone 策略对比脚本。
- `tests/test_backends_parity.py`：数值一致性测试。
- `old/`：历史归档（已从版本控制排除）。

## Standalone Backend 当前策略

梯度内存策略仅保留：

- `save_param_states`
- `checkpoint`
- `bruteforce_parallel_q6`（实验模式，要求 `num_qubits <= 6`）
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
.venv/bin/python run_standalone_backend.py --gradient-strategy bruteforce_parallel_q6

# checkpoint 粒度
.venv/bin/python run_standalone_backend.py --gradient-strategy checkpoint --checkpoint-interval 8

# 关闭 gate fusion 做对照
.venv/bin/python run_standalone_backend.py --disable-gate-fusion
```

## 测试

```bash
.venv/bin/python -m unittest tests.test_backends_parity -v
```

## Benchmark

```bash
.venv/bin/python benchmarks/compare_pennylane_saveall_checkpoint.py \
  --cases 20x4 \
  --standalone-modes save_param_states checkpoint auto bruteforce_parallel_q6
```

`bruteforce_parallel_q6` 仅适用于 `--cases` 中 `qubits <= 6` 的条目。

可选：

```bash
.venv/bin/python benchmarks/compare_pennylane_saveall_checkpoint.py \
  --cases 20x4 \
  --standalone-modes save_param_states checkpoint \
  --disable-gate-fusion
```
