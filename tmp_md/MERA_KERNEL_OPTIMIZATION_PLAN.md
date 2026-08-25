# MERA CUDA Kernel 详细优化方案

## 1. 目标与边界

本文档描述 MERA 第一版正确性实现之后的 CUDA 性能优化路径。优化必须保持
`MERA_IMPLEMENTATION_PLAN.md` 已固定的电路语义：

```text
Block(left,right):
    RY(left)
    RY(right)
    CX(control=left,target=right)

每个 stage:
    D matching
    U matching

layers = ceil(log2(qubits))
observable = Z(qubits - 1)
```

优化不修改 topology、参数顺序、Hamiltonian 或公共 API，并且必须与原有五种电路
已经采用的代码分层和扩展方式完全对齐。PennyLane 仍只有一个 `_mera()`，C++ 仍
只有一个 `MeraLayerLayout` 和一个 executor 特化；优化不能借机引入 MERA 独有的
公共框架、上下文字段或重复执行路径。

## 2. 当前第一版基线

### 2.1 当前 kernel

生产实现位于：

```text
sad/src/kernels/mera.cuh
sad/src/circuits/mera.cuh
sad/src/kernels/hamiltonian.cuh
```

MERA 使用：

```text
mera_matching_forward_kernel
mera_matching_backward_kernel
mera_hamiltonian_kernel
initialise_zero_state_kernel
```

同时复用公共 `Complex<T>`、`RotationCoefficients<T>`、`rotate_amplitude()`、
`generator_overlap()`、`block_atomic_sum()`、cooperative launch 和 occupancy 查询。

### 2.2 当前 forward 成本

一个 matching 只 launch 一次 kernel，但 kernel 内对每个 pair 顺序执行：

```text
for pair in matching:
    full-state RY(left)
    grid.sync()
    full-state RY(right)
    grid.sync()
    full-state CX(left,right)
    grid.sync()
```

若 matching 有 `p` 个 pair，就有约 `3p` 次全状态操作和 `3p` 次 grid barrier。
整个 MERA 的 block 数为：

```text
block_count = 2 * (qubits - 1) - popcount(qubits - 1)
```

因此第一版 forward 约有 `3 * block_count` 次 full-state pass。block 数线性增长，
每次 pass 的 state 大小指数增长，这是主要性能问题。

### 2.3 当前 backward 成本

每个 block 逆序执行：

```text
CX^-1(phi)
CX^-1(lambda)
grid.sync()
gradient(right) + RY^-1(right, phi/lambda)
grid.sync()
gradient(left) + RY^-1(left, phi/lambda)
grid.sync()
```

backward 同时访问 `phi`、`lambda` 和 gradient accumulator，通常比 forward 更重。
第一版正确且无全局 CX race，但没有利用同一 matching 内 pair 互不共享 wire 的性质。

### 2.4 当前可调参数

默认形状：

```text
forward  = 128 threads, register bits 2
backward = 128 threads, register bits 2
ordinary = 128 threads
variant  = f128r2_b128r2
```

当前 register bits 已控制每线程工作量，但 `kForwardTileBits/kTileBits` 还没有用于
MERA matching 的局部 tile 演化。

## 3. 必须保持的结构约束

优化后继续只保留：

```cpp
mera_matching_forward_kernel
mera_matching_backward_kernel
launch_mera_matching_forward
launch_mera_matching_backward
```

不新增平行的 `mera_optimized_*`、`mera_legacy_*` 或 `mera_block_*` 顶层 API。

同一个 matching 的 pair 互不共享 wire，可以分组；但 D 与 U 可能共享 wire，因此
必须保持：

```text
forward:  D 完整结束 -> global barrier/kernel boundary -> U
backward: U^-1 完整结束 -> D^-1
```

所有优化阶段都必须在 kernel 内按公式构造 tile map，不给 `context.cuh`、
`workspace.cuh`、`lookups.cuh` 或 `run_forward/run_backward` 增加 MERA 字段、
buffer、map builder 或参数。该约束与原实现计划的 production 符号审计一致，不是
只适用于第一轮的临时约束。

## 4. 核心优化：tile-aware matching

### 4.1 phase 划分

把 matching 的 pair 分成多个 phase。每个 phase 最多容纳：

```text
pairs_per_phase = floor(tile_bits / 2)
phase_count = ceil(pair_count / pairs_per_phase)
```

默认 backward：

```text
kTileBits = 5 lane bits + 2 register bits + 2 warp bits = 9
pairs_per_phase = 4
```

例如 7 个 pair 分为 `pair 0..3` 和 `pair 4..6` 两个 phase。

每个 phase：

```text
1. 选择本 phase 的 pair wires
2. 用其它 physical wires 填满 tile slots
3. 每个 CTA 加载一个 state tile
4. 在 registers/shared memory 内执行所有 local pairs
5. 一次写回 global state
6. phase 间 grid.sync()
```

### 4.2 kernel 内构造 selected map

第 `phase` 的第 `local_pair` 对应：

```text
pair = phase * pairs_per_phase + local_pair

D: first_active = 2 * pair + 1
U: first_active = 2 * pair
```

physical wire：

```cpp
left  = min(((first_active + 1) << layer) - 1, qubits - 1);
right = min(((first_active + 2) << layer) - 1, qubits - 1);
```

相邻放入 tile slots：

```text
selected[2 * local_pair]     = left
selected[2 * local_pair + 1] = right
```

剩余 slots 用未选择的 physical wires 从小到大填充。thread 0 写 shared selected map，
然后 `__syncthreads()`。这样不需要新增 device metadata。

### 4.3 tile 地址映射

复用：

```cpp
scatter_tile_assignment<TileBits>()
scatter_local_assignment<TileBits>()
```

```text
tile_bits  = min(qubits, configured_tile_bits)
tile_count = 2^(qubits - tile_bits)
```

CTA 使用 persistent-grid：

```cpp
for (tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
    load values[];
    apply all local pairs;
    store values[];
}
```

第一版是 `3 * pair_count` 个 full-state operation；tile 版变为
`ceil(pair_count / pairs_per_phase)` 个 tile load/store phase。

## 5. Forward tile kernel

### 5.1 tile load

每个 thread 保存：

```cpp
Complex<T> values[kForwardRegisterAmplitudes];
```

local index 延续公共布局：

```cpp
local = lane |
        (reg << kLaneBits) |
        (warp << (kLaneBits + kForwardRegisterBits));
```

读取一次 global state 后，本 phase 内不再访问 global state。

### 5.2 tile-local RY

根据 slot 所在位置使用：

```text
lane slot     -> warp shuffle
register slot -> register-pair update
warp slot     -> shared mailbox
```

复用 `apply_tile_gate_forward<T, RY, Slot>()` 的数学和数据交换逻辑。因为 `Slot`
是模板参数而 pair slot 是运行时值，使用固定范围 switch dispatch；tile bits 最大 12，
编译器可以内联每个 case。

### 5.3 tile-local CX

CX 是 basis permutation，control/target 位于相邻 slots。安全做法：

```text
values 写 mailbox
__syncthreads()
每个 local amplitude 从唯一 source_local gather
__syncthreads()
```

CX 自逆，forward/backward 可复用同一 permutation helper。禁止在 shared/global
地址上进行可能由两个 thread 同时写入的 naive swap。

### 5.4 phase 内完整顺序

```text
load tile once
for local_pair from left to right:
    RY(left slot)
    RY(right slot)
    CX(left slot, right slot)
store tile once
```

matching 内 pair 数学上互相对易，但仍保留从左到右顺序，确保参数 trace 与
PennyLane golden order 一致。

## 6. Backward tile kernel

### 6.1 tile state

每个 thread 保存：

```cpp
Complex<T> phi[kRegisterAmplitudes];
Complex<T> lambda[kRegisterAmplitudes];
```

每个 phase 对 `phi/lambda` 各读取一次，全部 local pair 完成后各写回一次。

### 6.2 block 逆序

```text
for local_pair in reverse order:
    CX^-1(phi/lambda)
    gradient(right)
    RY^-1(right, phi/lambda)
    gradient(left)
    RY^-1(left, phi/lambda)
```

梯度继续使用 `generator_overlap<T, RY>()`。parameter index 为：

```text
parameter_offset + 2 * pair
parameter_offset + 2 * pair + 1
```

第一轮继续逐 parameter 使用 `block_atomic_sum()`。只有 profiling 证明规约是瓶颈，
才实现 phase 内多参数 shared reduction；禁止用大型 thread-local overlap array
导致 register spill。

### 6.3 shared-memory 规划

需要容纳：

```text
phi/lambda amplitude mailbox（优先分时复用）
gradient reduction buffer
selected map
```

launcher 必须按新的 shared bytes 重新调用
`cudaOccupancyMaxActiveBlocksPerMultiprocessor()`，不能沿用第一版 occupancy 假设。

## 7. 公共 kernel 参数继续可调

优化后仍使用：

| 宏 | optimized kernel 中的作用 |
|---|---|
| `SAD_FORWARD_BLOCK_THREADS` | forward CTA threads、warp 数和 tile bits |
| `SAD_FORWARD_REGISTER_BITS` | forward amplitudes/thread 和 tile bits |
| `SAD_BLOCK_THREADS` | backward CTA threads、warp 数和 tile bits |
| `SAD_REGISTER_BITS` | backward phi/lambda amplitudes/thread 和 tile bits |
| `SAD_ORDINARY_BLOCK_THREADS` | MERA Hamiltonian threads |

禁止写死 128 threads、4 amplitudes/thread 或 9-bit tile。必须使用：

```cpp
kForwardBlockThreads
kForwardRegisterAmplitudes
kForwardTileBits
kForwardTileAmplitudes
kBlockThreads
kRegisterAmplitudes
kTileBits
kTileAmplitudes
```

增大 register bits 会扩大 tile、减少 phase，但也会提高 register pressure；增大
block threads 会扩大 tile，但也增加 shared memory 并可能降低 resident CTA 数。
必须比较完整 forward + Hamiltonian + backward，不能只比较单 kernel。

这些仍是编译期参数。运行时切换表示选择不同预编译 `.so`，不是在线修改 kernel：

```text
f128r2_b128r2
f64r4_b64r4
f64r3_b64r4
f128r3_b64r4
f64r4_b128r3
```

电路角度 `theta` 始终是运行时参数，不需要重新编译。

## 8. Launcher 与 executor

tile kernel 的 work unit 是 tile：

```text
tile_count = 2^(qubits - min(qubits, tile_bits))
resident_blocks = occupancy_per_sm * multiprocessors
grid_size = min(tile_count, resident_blocks)
```

pair_count 为 0 时不 launch。`CircuitExecutor` 边界保持：

```text
forward:  launch D matching, launch U matching
backward: launch U matching, launch D matching
```

不得改变 `MeraLayerLayout` 参数 offset，也不得把 D/U 合并成无 barrier 的集合。

## 9. 明确排除 host-precomputed MERA maps

本优化计划不允许为 MERA 增加 host-precomputed topology maps。原因不是这种方案
绝对无法优化，而是它会破坏已经确认的结构契约：

```text
不新增 build_mera_pair_maps()
不新增 MeraTopology/MeraTopologyView
不新增 forward/backward MERA selected-map buffers
不修改 ForwardCircuitContext/BackwardCircuitContext
不修改 PreparedWorkspace
不扩展 run_forward/run_backward 参数
```

XXZ 的 maps 是其既有 executor/kernel 结构的一部分，不能因为 XXZ 存在这些字段，
就把它们复制成一套 MERA 专属公共状态。MERA wire 有闭式公式，optimized kernel
必须由 `qubits/layer/matching/phase/pair` 直接构造 selected slots。

如果将来 profiling 明确证明公式构造成为瓶颈，需要先单独修改架构文档和 production
符号审计，并获得新的结构决策；它不属于本文档授权的优化范围。

因此以下宏在本轮及本文档覆盖的后续优化中都不启用：

```text
SAD_FORWARD_FIXED_LOW_LANES
SAD_FIXED_LOW_LANES
SAD_ALTERNATE_PHASES
SAD_DIAGONAL_LOOKUP_BITS
```

它们不得静默改变 MERA traversal 或电路语义。

## 10. Hamiltonian 与初始化

`mera_hamiltonian_kernel` 已是一次线性 state pass：

```text
lambda[index] = Z_target * phi[index]
energy += real(conj(phi[index]) * lambda[index])
```

第一轮保持 `SAD_ORDINARY_BLOCK_THREADS=128`。只有 Hamiltonian 占比显著时才测试
64/256/512，不为低占比 kernel 增加大量 variants。

初始化当前清零两个 phi buffer 并设置 `state[0]=1`。为保持与当前 executor 方法集合
和“一对 matching kernel”约束完全一致，本优化计划不增加 MERA 专用初始化 kernel，
也不增加第二条 stage-0 执行路径。小规模固定开销只通过现有 launcher 参数和 matching
实现优化。

## 11. Profiling 计划

### 11.1 配置

```text
GPU       = RTX 6000 Ada
precision = float64 主 sweep，float32 关键规模
seed      = 42
qubits    = 4,6,8,...,28
layers    = ceil(log2(qubits))
```

结果写入独立文件，不覆盖第一版：

```text
benchmark/results/mera_v1_sad_gpu.csv
benchmark/results/mera_tile_sad_gpu.csv
benchmark/results/mera_variant_<name>.csv
```

### 11.2 指标

记录：

```text
forward/Hamiltonian/backward mean 和 median
total mean 和 median
energy/full gradient
workspace bytes
kernel variant
```

Nsight 重点查看：

```text
kernel launch 数
cooperative barrier 数
DRAM bytes 和 throughput
achieved occupancy
registers/thread
shared memory/CTA
eligible warps/cycle
global load/store efficiency
atomic throughput
local-memory spill
```

按以下顺序 profile：

```text
A. 第一版 baseline
B. 只替换 forward tile matching
C. forward + backward tile matching
D. 多梯度规约（只有 C 显示规约瓶颈时）
E. variant sweep
```

每项单独保存结果，避免无法归因。

## 12. Variant sweep 与 dispatch

第一轮只测已有安全组合：

| Variant | Forward | Backward |
|---|---|---|
| `f128r2_b128r2` | `128 x r2` | `128 x r2` |
| `f64r4_b64r4` | `64 x r4` | `64 x r4` |
| `f64r3_b64r4` | `64 x r3` | `64 x r4` |
| `f128r3_b64r4` | `128 x r3` | `64 x r4` |
| `f64r4_b128r3` | `64 x r4` | `128 x r3` |

写入 `_select_library()` 前必须满足：

```text
1. 连续 qubit 区间稳定胜出
2. 至少两轮独立 sweep 可重复
3. float32/float64 正确
4. forward/backward 无严重性能反转
5. 无 spill、非法访问和 race
```

建议：

```text
改善 < 5%：不增加 dispatch
改善 >= 5% 且跨轮稳定：可考虑
改善 >= 10% 且连续区间稳定：优先写入
```

只在 `sad/python/sad_baseline/runner.py::_select_library()` 增加 circuit ID 5 分支；
阈值必须来自 CSV，不能复制其它电路的经验表。

## 13. 正确性与安全

### 13.1 topology

覆盖：

```text
3q  odd carry
4q  最小二幂
6q  非二幂 golden topology
8q  标准 golden topology
12q 跨 tile/phase
```

检查 pair wires、parameter offsets、phase coverage、无重复遗漏和 odd carry。

### 13.2 数值

每个关键规模检查：

```text
SAD vs PennyLane energy/full gradient
optimized vs legacy
all-fused vs optimized
float32/float64
```

容差：

```text
float32 absolute = 3e-5
float64 energy   = 1e-10
float64 gradient = 1e-9
```

独立中心有限差分覆盖第一个 D-left/D-right、第一个 U、最后一个 U，以及非 2 次幂
carry 附近参数。

### 13.3 CUDA sanitizer

运行：

```text
compute-sanitizer --tool memcheck
compute-sanitizer --tool racecheck
```

重点覆盖 tile-local CX、最后一个不完整 phase、qubits 小于 tile bits、tile_count=1、
最大地址计算和 float32/float64 shared-memory alignment。

若抽取公共 tile helper，必须运行原五种电路完整回归。

## 14. 实施阶段

### Phase A：冻结 baseline

1. 保存当前 MERA 第一版 4..28q CSV。
2. 记录 CUDA、driver、GPU、variant 和环境。
3. 验证重复 sweep 的波动。

### Phase B：forward tile matching

1. kernel 内构造 selected map。
2. tile load/store。
3. tile-local RY。
4. tile-local safe CX gather。
5. executor/launcher API 不变。

验收：forward 和完整 energy/gradient 对齐，forward 时间下降，racecheck 通过。

### Phase C：backward tile matching

1. 同时加载 phi/lambda。
2. local inverse CX。
3. local gradient + inverse RY。
4. 一次写回 phi/lambda。
5. 重新计算 occupancy。

验收：full gradient 和有限差分通过，backward 时间下降，无不可接受 spill。

### Phase D：微调

仅按 profiling 结果选择：

```text
多参数 shared reduction
mailbox 分时复用
dynamic shared sizing
slot dispatch 展开
selected-map 构造简化
```

### Phase E：variant sweep

编译五个 variants，运行完整 qubit sweep，重复验证 winner。

### Phase F：dispatch 和文档

写入 measured dispatch，更新 MERA benchmark、README 和优化报告。

## 15. 提交拆分

```text
1. benchmark: preserve MERA v1 baseline
2. kernel: tile-local MERA forward
3. kernel: tile-local MERA backward
4. tests: cross-tile and sanitizer coverage
5. perf: MERA variant sweep
6. runtime: measured MERA dispatch
7. docs: final architecture and results
```

不要把 topology、tile kernel 和 dispatch 放入一个提交。本计划禁止新增 MERA
workspace metadata。

## 16. 风险与回退

| 风险 | 表现 | 回退 |
|---|---|---|
| register pressure | occupancy 降低、spill | 降 register bits 或缩小 phase |
| shared memory 过大 | resident CTA 太少 | 分时 mailbox、减小 tile |
| CX race | 偶发错误、racecheck 报告 | mailbox gather，禁止并发 swap |
| kernel 内 phase map 错误 | 非 2 次幂错位 | 3/6/10q golden phase 测试 |
| gradient atomic 过重 | backward 收益低 | 受控的多参数规约 |
| 小规模变慢 | setup 大于收益 | qubit-dependent dispatch |
| variants 过多 | 构建维护成本增长 | 只保留稳定改善组合 |

第一版正确性 kernel 是优化 oracle。若不长期保留两套 production kernel，则依靠 git
baseline、PennyLane、有限差分和 execution-mode 对照提供回退能力。

## 17. 完成标准

```text
1. 电路语义、参数顺序和 Hamiltonian 不变
2. 仍只有一对 matching kernel/launcher
3. 公共 block/register 参数仍可调
4. 3/4/6/8/12q topology 和数值测试通过
5. float32/float64 full gradient 对齐
6. 有限差分通过
7. compute-sanitizer 无非法访问和 race
8. 原五种电路回归通过
9. 4..28q 有独立 CSV
10. dispatch 只包含数据证明的稳定 winner
11. README 记录参数区间、阈值和测试环境
12. workspace/context/lookups/run signatures 保持不变
13. 主文件结构审计逐项通过
```

性能提升以完整 qubit 区间的稳定 median total time 和正确性共同判断，不以单个小规模
点或单个 kernel 的最好结果判断。

## 18. 主文件结构审计

本节是实现和 code review 的强制检查表。没有列出的 production 文件不得因为 MERA
优化新增函数、类型、字段、buffer、分支或参数。允许“修改内部实现”不等于允许增加
第二套 API。

### 18.1 Python/PennyLane 主文件

| 文件 | 优化阶段允许变化 | 强制保持的结构 | 对齐对象 |
|---|---|---|---|
| `pennylane-lightning/src/pennylane_lightning_baseline/circuits.py` | 不修改 | 仍只有一个 `_mera()`、一个 `CircuitSpec` 注册和一个 Hamiltonian 分支 | 原五种电路的一类电路一个 builder |
| `pennylane-lightning/src/pennylane_lightning_baseline/native.py` | 不修改 production topology；只允许测试需要的非行为性修正 | 仍只有 `_build_operations()` 内一个 MERA 分支和 `_build_hamiltonian()` 内一个分支 | 原五种 native 分支结构 |
| `pennylane-lightning/src/pennylane_lightning_baseline/runner.py` | 不修改 | 公共 QNode runner 不增加 MERA 参数或专用执行路径 | 所有电路共享 runner |

特别禁止：

```text
第二个 def _mera_*
PennyLane topology/count helper
MERA 专用 QNode runner
修改 CircuitSpec 数据结构
```

### 18.2 SAD 注册与公共 runtime

| 文件 | 优化阶段允许变化 | 强制保持的结构 | 对齐对象 |
|---|---|---|---|
| `sad/include/sad_api.h` | 不修改 | 保持唯一 `SAD_CIRCUIT_MERA=5` | 现有 SadCircuit enum |
| `sad/src/runtime/circuit_dispatch.cuh` | 不修改 | 保持唯一 ID 5 case | 原五种 compile-time dispatch |
| `sad/src/runtime/circuit_execution.cuh` | 不修改 | 保持一个 include 和一个参数量特判；run signatures 不变 | QAOA 特判和公共 layer loop |
| `sad/src/circuits/context.cuh` | 不修改 | 不增加 MERA pointer/count/phase 字段 | 五种电路共享 context；MERA 不需要 metadata |
| `sad/src/runtime/lookups.cuh` | 不修改 | 不增加 `build_mera_*` 或 MERA map 数据结构 | MERA topology 由闭式公式计算 |
| `sad/src/runtime/workspace.cuh` | 不修改 | 不增加 MERA host vector、DeviceBuffer、phase count 或 accounting 项 | 公共 workspace 保持现状 |
| `sad/src/runtime/runner.cuh` | 不修改 matching 调用签名 | 保持一个 MERA Hamiltonian 分支；不新增 MERA forward/backward 参数 | QAOA/XXZ Hamiltonian 分派模式 |

这一组文件在优化 diff 中原则上应为零改动。若为了非 MERA 的公共 bug 修复必须修改，
应拆成独立提交，并运行原五种电路完整回归，不能夹在 MERA 优化提交中。

### 18.3 MERA circuit 与 kernel 主文件

| 文件 | 优化阶段允许变化 | 强制保持的结构 | 对齐对象 |
|---|---|---|---|
| `sad/src/circuits/mera.cuh` | 只允许修改现有方法体中的 launcher 调用细节 | 一个 `MeraLayerLayout`、一个 executor 特化、现有方法集合和签名不变 | `xxz_hva.cuh` 的 layout + executor |
| `sad/src/kernels/mera.cuh` | 修改现有 device helper、matching kernel 和 launcher 的内部算法；可增加仅供同文件复用的 `__device__` helper | 只能有一对 `__global__` matching kernel 和一对 host launcher；不得新增平行路径 | `kernels/xxz.cuh` 的一对 kernel + launcher |
| `sad/src/kernels/hamiltonian.cuh` | 原则上不修改；只有 profile 证明必要时调整现有 `mera_hamiltonian_kernel` 方法体 | 仍只有一个 MERA Hamiltonian kernel，不新增 launcher/variant kernel | QAOA/XXZ 各一个专用 Hamiltonian |
| `sad/src/core/cuda_common.cuh` | 不为 MERA 增加专用宏或类型 | 继续只提供公共 block/register/tile 常量 | 五种电路共享公共配置 |
| `sad/src/kernels/rotation_primitives.cuh` | 优先不修改；若抽取真正通用 helper，必须独立提交并证明其它电路不回归 | 不增加 MERA 命名符号 | 公共 rotation primitive |

`sad/src/kernels/mera.cuh` 最终允许的顶层 production 形状固定为：

```text
若干同文件 private __device__ helpers
mera_matching_forward_kernel
launch_mera_matching_forward
mera_matching_backward_kernel
launch_mera_matching_backward
```

不能同时保留第一版和 optimized 两套 `__global__` kernel。execution modes 继续通过同一
对 matching API，不能用 mode 分支选择重复实现。

### 18.4 Python SAD、benchmark 与测试

| 文件 | 优化阶段允许变化 | 强制保持的结构 | 对齐对象 |
|---|---|---|---|
| `sad/python/sad_baseline/runner.py` | 完整 sweep 后，只允许在现有 `_select_library()` 增加 circuit ID 5 的 measured 分支 | `_CIRCUITS`、校验、参数量和 canonical-name 结构不重构 | 现有 RA/SU2/RZZ variant dispatch |
| `benchmark/benchmark_sad.py` | MERA 独立输出、variant sweep 配置 | 继续调用公共 `energy_and_grad()`，不增加 MERA 执行器 | 现有 benchmark loop |
| `benchmark/benchmark_pennylane_lightning.py` | 仅 MERA 独立结果配置 | 继续调用公共 PennyLane runner | 现有 benchmark loop |
| `benchmark/benchmark_lightning_native.py` | 仅 MERA 独立结果配置 | 继续调用公共 native runner | 现有 benchmark loop |
| `sad/tests/test_sad_runner.py` | 增加 cross-tile、variant 和 execution-mode cases | 不建立第二套 production topology helper | 现有 SAD-vs-Lightning 测试 |
| `pennylane-lightning/tests/test_circuits.py` | 增加 golden topology/parameter cases | topology 只在测试期望值中表达 | 现有 tape 测试 |

### 18.5 明确禁止的结构变化

无论性能数据如何，本文档不授权以下变化：

```text
1. 在 circuits.py 增加第二个 MERA def
2. 新增 MeraTopology、MeraTopologyView 或 topology registry
3. 新增 build_mera_pair_maps/build_mera_phase_maps
4. 给 context/workspace 增加 MERA buffers 或 phase counts
5. 修改 run_forward/run_backward 公共签名
6. 同时保留 legacy 和 optimized 两套 MERA matching kernel API
7. 增加 MERA 专用公共编译宏 SAD_MERA_*
8. 修改其它五个 CircuitExecutor 的方法签名
9. 为 MERA 新建独立 runner 或绕过 visit_circuit
10. 为调优参数重构 CircuitSpec 或公共参数量接口
```

### 18.6 Code review 判定

优化提交合并前逐项执行：

```text
[ ] circuits.py 中 MERA builder 数量仍为 1
[ ] MeraLayerLayout 数量仍为 1
[ ] CircuitExecutor<SAD_CIRCUIT_MERA,T> 数量仍为 1
[ ] MERA matching __global__ kernel 数量仍为 2
[ ] MERA matching launcher 数量仍为 2
[ ] MERA Hamiltonian kernel 数量仍为 1
[ ] context/workspace/lookups 没有 MERA 字段或 builder
[ ] run_forward/run_backward 签名未变化
[ ] 公共宏仍为 SAD_FORWARD_BLOCK_THREADS/SAD_FORWARD_REGISTER_BITS/
    SAD_BLOCK_THREADS/SAD_REGISTER_BITS/SAD_ORDINARY_BLOCK_THREADS
[ ] 原五种电路测试通过
[ ] MERA topology、full gradient、有限差分和 sanitizer 通过
```

任何一项不满足，都应视为结构未对齐；不能仅凭性能提升接受。
