# PennyLane Lightning-GPU baseline

## 安装

建议使用独立虚拟环境。当前基线固定 PennyLane/Lightning-GPU 0.45.0，避免 benchmark 过程中依赖版本漂移。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e './pennylane-lightning[test]'
```

Lightning-GPU 需要 NVIDIA GPU、兼容驱动和 cuQuantum/CUDA 运行库；PyPI wheel 会安装其 CUDA 12 运行时依赖。

## API

```python
from pennylane_lightning_baseline import energy_and_grad

result = energy_and_grad(
    circuit="su2-hea",
    random_seed=42,
    scalability=(16, 8),
    batches=1,
    precision="float64",
    steps=5,
)

print(result.energy)
print(result.grad.shape)
print(result.step_times_s)
print(result.memory)
```

也支持解包为四项：

```python
energy, grad, step_times_s, memory = energy_and_grad(...)
```

约定：

- `steps` 是正式计时次数；默认先额外 warmup 一次，因此返回的计时长度恰好等于 `steps`。
- 每个 step 都在同一组固定参数上执行一次完整的 energy + adjoint gradient，不做优化器更新。
- `batches` 当前只允许为 1；其它值会显式报错。
- `precision` 接受 `float32/fp32/single/complex64` 或 `float64/fp64/double/complex128`。
- 默认设备固定为 `lightning.gpu`。测试时可显式传入 `device_name="lightning.qubit"`。
- 显存使用量是本进程在 device 创建前、warmup 后和各 step 后的边界采样；`gpu_peak_observed_mib` 是观测峰值，不保证捕获 step 内瞬态分配。计时区间不包含内存采样开销。

如需增加电路，用 `CircuitSpec` 和 `register_circuit` 注册，不需要修改 runner。

## 批量 benchmark

默认运行 3 类电路、4/6/.../28 qubits、8 layers，并随 qubit 数增加逐步减少重复次数：

```bash
python benchmark/benchmark_pennylane_lightning.py
```

所有配置都位于脚本开头的全局常量中。结果逐行 flush 到 `benchmark/results/pennylane_lightning_gpu.csv`，单个配置失败也会写入 `status/error` 后继续。

