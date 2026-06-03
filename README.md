# PennyLane Qubit Rotation Demo

这个仓库当前聚焦于两条可对比的实现路径：

- `PennyLane backend`：基于 `lightning.gpu + adjoint`，主要作为参考实现与结果校验路径。
- `Standalone CUDA backend`：自定义 C++/CUDA 扩展，直接提供 `energy` 与 `energy_and_grad`。

当前目标是保持一个可维护、可扩展、便于性能实验的统一前端结构。

多人协作说明见 [COLLABORATION.md](./COLLABORATION.md)（当前统一远端：`git@github.com:zzEncounter/GPU_Simulation.git`）。

## 目录结构

- `ring_ising/config.py`：统一用户侧运行配置。
- `ring_ising/workflows/`：统一 workflow 入口，以及 `pennylane.py` / `standalone.py` 两条运行路径。
- `ring_ising/backends/standalone/`：Standalone CUDA 后端的 Python 侧封装。
- `ring_ising/training/`：共享训练循环与统一返回结果模型。
- `ring_ising/cli/run.py`：统一命令行入口。
- `ring_ising/`：其余前端 Python 包内容（CLI、模型构件、runtime 辅助层）。
- `run_workflow.py`：统一脚本入口。
- `standalone_backend/`：仅保留编译后的 CUDA 扩展模块挂载点。
- `cpp/`：C++/CUDA 扩展实现。
- `benchmarks/compare_gradient_strategy.py`：Standalone 梯度策略对比脚本。
- `tests/test_backends_parity.py`：数值一致性测试。
- `old/`：历史归档（已从版本控制排除）。

## 统一前端接口

现在推荐的主入口是：

- `ring_ising.workflows.RunConfig`
- `ring_ising.workflows.run`

其中：

- `backend="pennylane"` 选择 PennyLane 路径
- `backend="standalone"` 选择自定义 C++/CUDA 路径
- `gradient_strategy` 仅对 standalone 生效
- PennyLane 路径固定要求 `lightning.gpu`

## Standalone Backend 当前策略

梯度内存策略仅保留：

- `save_param_states`
- `checkpoint`
- `dense_scan`（实验模式，要求 `num_qubits <= 6`）

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

### PennyLane backend

```bash
.venv/bin/python run_workflow.py --backend pennylane
```

### Standalone backend

```bash
.venv/bin/python run_workflow.py --backend standalone --qubits 12 --layers 3
```

可选参数示例：

```bash
# 指定梯度策略
.venv/bin/python run_workflow.py --backend standalone --gradient-strategy save_param_states
.venv/bin/python run_workflow.py --backend standalone --gradient-strategy checkpoint
.venv/bin/python run_workflow.py --backend standalone --gradient-strategy dense_scan

# checkpoint 粒度
.venv/bin/python run_workflow.py --backend standalone --gradient-strategy checkpoint --checkpoint-interval 8

# 关闭 gate fusion 做对照
.venv/bin/python run_workflow.py --backend standalone --disable-gate-fusion

# 开启详细 step 报告；默认只显示进度条
.venv/bin/python run_workflow.py --backend standalone --report-steps

```

## 函数调用接口

推荐使用统一入口；两条路径都会返回同一风格的结构化结果（最终能量、训练循环时间、step 指标、可选 GPU 遥测）：

```python
from ring_ising.workflows import RunConfig, run

pennylane_result = run(
    RunConfig(
        backend="pennylane",
        num_qubits=12,
        layers=3,
        steps=20,
    )
)

standalone_result = run(
    RunConfig(
        backend="standalone",
        num_qubits=12,
        layers=3,
        steps=20,
        gradient_strategy="save_param_states",
    )
)
```

## 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Benchmark

```bash
.venv/bin/python benchmarks/compare_gradient_strategy.py \
  --cases 4x8 5x32 6x128 \
  --modes save_param_states checkpoint dense_scan \
  --reference-mode save_param_states
```

`dense_scan` 仅适用于 `qubits <= 6` 的条目。
