# SAD custom CUDA adjoint baseline

这是不依赖 PennyLane 的 CUDA/C++ state-vector 实现。Python 只负责与基线一致的参数采样、加载共享库以及整理返回值；forward、Hamiltonian、generator-overlap backward 和分项计时均在 native CUDA/C++ 层执行。

## 构建与调用

```bash
make -C sad
python -m pip install -e './sad[test]'
```

如果共享库不存在，首次调用也会自动执行 `make -C sad`。`.env` 中暂时没有启用任何 override；可按需开启 device、CUDA arch、nvcc 或共享库路径设置。

```python
from sad_baseline import energy_and_grad

result = energy_and_grad(
    circuit="su2-hea",
    random_seed=42,
    scalability=(16, 8),
    batches=1,
    precision="float64",
    steps=5,
)

print(result.energy, result.grad)
print(result.forward_times_s)
print(result.hamiltonian_times_s)
print(result.backward_times_s)
print(result.step_times_s)
```

`step_times_s[i]` 是同一步三个分项的和。分项使用 native wall-clock，并在每个分项末尾做 CUDA event synchronization；因此既包含 kernel，也包含该分项的 launch 开销。额外 warmup 不进入返回时间；forward 包含全零态初始化，Hamiltonian 包含 energy 回传，backward 包含 gradient 回传。

## Benchmark 与数值对比

两个求解器都会把完整 gradient 以 JSON 数组写进各自 CSV。先生成结果，再按 circuit/qubits/layers/precision/seed/batches join：

```bash
python benchmark/benchmark_pennylane_lightning.py
python benchmark/benchmark_sad.py
python benchmark/compare_sad_pennylane.py
```

输出分别为：

- `benchmark/results/pennylane_lightning_gpu.csv`
- `benchmark/results/sad_gpu.csv`
- `benchmark/results/sad_vs_pennylane.csv`

对比文件包含 energy absolute error、gradient max/L2 error、allclose 结果、总时间 speedup，以及 SAD 的 forward/Hamiltonian/backward 分项均值。

## Kernel 结构

- RX/RY：固定 `r=3`、4 warps/CTA。每个 thread 保存 8 个 amplitude；5 个 tile bit 通过 lane shuffle，3 个通过 thread registers，2 个通过 warp/shared-memory mailbox。slot 使用编译期模板常量，振幅 index 在 load/store 时按标量重算，不保存动态索引数组。persistent cooperative grid 根据 CUDA occupancy 查询结果投放可同时驻留的全部 CTA，并在 phase 之间 `grid.sync()`。
- 所有参数的半角 sine/cosine 在资源创建时一次性预计算，timed kernel 不调用 device `sincos`。当前 float32/float64 的 RX/RY persistent kernel 均为 zero-stack。
- RZ/RZZ：资源创建时按每 8 个 generator 生成 256-entry phase lookup；kernel 每个 amplitude 只组合 lookup factor。backward 在相同 state pass 中累计各 generator overlap并反向演化 `phi/lambda`，partial 使用 gate-major shared memory，避免 thread-local `overlaps[]`。当前 diagonal forward/backward 也为 zero-stack。
- ring CNOT：将顺序 CNOT 层合成为一个可逆 basis permutation，coalesced 写入另一 state vector，随后交换 input/output；backward 同时轮换 `phi/lambda`。
- Hamiltonian：直接计算 `H|psi>`，其中 ZZ 部分为对角能量，X 部分读取对应 bit-flip amplitude。
- backward：从 `phi=|psi>`、`lambda=H|psi>` 出发逆序扫描，使用 `dE/dtheta = Im(<lambda|G|phi>)`，并同时反演两条 state vector。

## CUDA 源码布局

- `src/sad_cuda.cu`：公共 kernel、launch 封装、显存/计时资源、Hamiltonian 和顶层 circuit dispatch。
- `src/circuits/ra_hea.cuh`：RA-HEA 参数规则、逐层 forward/backward。
- `src/circuits/su2_hea.cuh`：SU2-HEA 参数规则、RZ lookup、逐层 forward/backward。
- `src/circuits/rzz_hea.cuh`：RZZ-HEA 偶数 qubit 约束、RZ/RZZ lookup、逐层 forward/backward。

三种电路通过 `CircuitExecutor<Circuit, T>` 特化接入公共执行器。运行时只在完整的 layer loop 外 dispatch 一次；层内没有 circuit 分支。新增电路时增加一个特化和顶层注册项即可，不需要修改公共 kernel。

## 当前验证

在 RTX 6000 Ada、float64、4/6/.../28 qubits × 8 layers 的默认 sweep 上，三类电路共 39 个配置全部通过逐元素对比。最大 energy absolute error 为约 `6.7e-15`，最大 gradient element absolute error 为约 `2.4e-14`。这些数据保存在 `benchmark/results/sad_vs_pennylane.csv`。
