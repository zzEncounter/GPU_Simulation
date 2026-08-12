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
print(result.kernel_variant)
```

`step_times_s[i]` 是同一步三个分项的和。分项使用 native wall-clock，并在每个分项末尾做 CUDA event synchronization；因此既包含 kernel，也包含该分项的 launch 开销。额外 warmup 不进入返回时间；forward 包含首层状态生成，Hamiltonian 包含 energy 回传，backward 包含 gradient 回传。

默认 `SAD_EXECUTION_MODE=optimized`：首层直接生成层末态；后续层把最后一个 RX/RY phase、对角门和 ring CNOT 合并。RA 使用纯实数 state/H/backward；SU2、RZZ backward 按规模选择 split、对角 single-pass 或 phase fusion。runner 还会根据 circuit/qubits 分别选择已测的 forward/backward tile 共享库，选择结果在 `kernel_variant` 中；`SAD_DISABLE_VARIANT_DISPATCH=1` 固定使用安全的 128×4 变体。设置 `legacy` 可切回完整逐算子实现，`initial-only`、`fused-forward`、`phased-forward`、`all-fused` 用于消融。

## Benchmark 与数值对比

两个求解器都会把完整 gradient 以 JSON 数组写进各自 CSV。先生成结果，再按 circuit/qubits/layers/precision/seed/batches join：

```bash
python benchmark/benchmark_pennylane_lightning.py
python benchmark/benchmark_sad.py
python benchmark/compare_sad_pennylane.py
```

输出分别为：

- `benchmark/results/pennylane_lightning_gpu.csv`
- `benchmark/results/sad_optimized_gpu.csv`（`sad_gpu.csv` 保留固定 baseline）
- `benchmark/results/sad_vs_pennylane.csv`

对比文件包含 energy absolute error、gradient max/L2 error、allclose 结果、总时间 speedup，以及 SAD 的 forward/Hamiltonian/backward 分项均值。

## Kernel 结构

- RX/RY：安全变体为 `r=2`、4 warps/CTA。每个 thread 保存 4 个 amplitude；5 个 tile bit 通过 lane shuffle，2 个通过 thread registers，2 个通过 warp/shared-memory mailbox。大规模会按 circuit/qubits 分别 dispatch 64/128-thread、8/16-amplitude 的 forward/backward 变体。RY 的符号用 sign-bit XOR 实现，无 warp 分支；backward 的 `phi/lambda` 顺序复用同一 mailbox。phase target 用 bit mask 表示，支持紧凑和 fixed-low-lane 布局。默认每个 phase 使用一个普通 kernel，由同一 stream 的 kernel 边界提供全局顺序；`SAD_ROTATION_PERSISTENT=1` 可重建旧 cooperative persistent 路径用于消融。
- 所有参数的半角 sine/cosine 在资源创建时一次性预计算，timed kernel 不调用 device `sincos`。当前 float32/float64 的 RX/RY ordinary/persistent kernel 均为 zero-stack。
- RZ/RZZ：资源创建时按每 8 个 generator 生成 256-entry phase lookup；kernel 每个 amplitude 只组合 lookup factor。backward 在相同 state pass 中累计各 generator overlap并反向演化 `phi/lambda`，partial 使用 gate-major shared memory，避免 thread-local `overlaps[]`。独立多梯度 diagonal 使用 64-thread CTA，共享单梯度 QAOA 使用 128-thread CTA。当前 diagonal forward/backward 也为 zero-stack。
- 首层：资源创建时生成 8-qubit chunk product lookup。RA/SU2 直接生成 `rotation(+RZ)+CNOT` 后的完整状态；RZZ 直接生成 `RX+RZ+RZZ` 后的完整状态，不再执行 memset、零态 kernel 和首层的多个 full-state pass。
- fused layer：最终 RX/RY phase 在写回前应用 RZ/RZZ lookup，并按需直接 scatter ring CNOT。反向 kernel 从 CNOT 后的索引 gather，同时完成 diagonal gradient/inverse 和 RX/RY gradient/inverse。
- ring CNOT：将顺序 CNOT 层合成为一个可逆 basis permutation，forward 使用 scatter，adjoint 使用 gather；写入另一 state vector 后交换 input/output，backward 同时轮换 `phi/lambda`。
- Hamiltonian：直接计算 `H|psi>`，其中 ZZ 部分为对角能量，X 部分读取对应 bit-flip amplitude。
- QAOA：每层只有共享的 `beta/gamma` 两个参数，cost-first；cost phase 使用按 domain-wall count 索引的 `q/2+1` 小表，共享 RZZ backward 一次规约整层 gamma gradient，并在 q>=24 融合进 mixer 的最终 adjoint phase。
- XXZ-HVA：bond-aware tile 将同一 bond 的 RXX/RYY/RZZ 合成一次 partner update；dependency-preserving schedule 还把 block 内 odd bond 接在相邻 even bond 后，boundary odd bond 最后处理。目标 Hamiltonian 为周期 `XX+YY+0.5ZZ`。
- backward：从 `phi=|psi>`、`lambda=H|psi>` 出发逆序扫描，使用 `dE/dtheta = Im(<lambda|G|phi>)`，并同时反演两条 state vector。RZ/ZZ 可在一个层端点上一次规约多个梯度。

## CUDA 源码布局

- `src/sad_cuda.cu`：C ABI、float/double 入口和异常边界。
- `src/core/`：复数类型、CUDA 错误检查、显存/锁页内存与计时资源。
- `src/kernels/`：rotation、diagonal、ring CNOT 和 Hamiltonian kernel；`fused/` 再按 initial、forward、backward 分开。
- `src/runtime/`：运行选项、唯一的 circuit dispatch、lookup、workspace 准备、单步执行与结果回传。
- `src/circuits/`：每种电路自己的参数 `LayerLayout`、约束以及逐层 forward/backward 编排。

五种电路通过 `CircuitExecutor<Circuit, T>` 特化接入公共执行器。运行时通过 `visit_circuit` 统一 dispatch，并且只发生在完整的 layer loop 外；层内没有 circuit 分支。新增电路时增加一个特化和 visitor 注册项即可，不需要修改公共 kernel。

## 当前验证

在 RTX 6000 Ada、float64、4/6/.../28 qubits × 8 layers 的 sweep 上，五类电路共 65 个配置全部通过逐元素对比。最大 energy absolute error 为 `3.91e-14`，最大 gradient element absolute error 为 `9.96e-13`；55 项回归测试通过。完整数据与消融见根目录 `OPTIMIZATION_REPORT.md` 和 `benchmark/results/research_main.csv`。
