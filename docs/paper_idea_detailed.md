# 论文 Idea 详细技术方案
## Adaptive Kernel Generation for QML Simulation and Training on GPU

> 本文档整理自对 GPU_Simulation 项目代码的深度分析，结合老板思路，给出面向 PLDI/SC/PPoPP 的论文技术细节补充。

---

## 一、MemoryBond / ComputeBond 的精确定义与 Roofline 形式化

### 1.1 两类 Kernel 的精确 Arithmetic Intensity

**ComputeBond Kernel（`backward_rotation_chunk_w8`）：**

从代码 `inverse_walk_ryrz_rotation_chunk_kernel<W>` 来看，每个 thread 处理一个 amplitude pair：
- **读取**：`current[i0]`, `current[i1]`, `lambda[i0]`, `lambda[i1]` → 4 × 16 bytes = 64 bytes
- **写入**：`current[i0]`, `current[i1]`, `lambda[i0]`, `lambda[i1]` → 4 × 16 bytes = 64 bytes
- **计算**：每个 wire 做 RZ inverse（4 FMA）+ RY inverse（4 FMA）+ gradient（4 FMA）≈ 12 FMA per wire，W=8 时约 96 FMA per pair
- **Arithmetic Intensity** ≈ 96 × 2 FLOP / 128 bytes ≈ **1.5 FLOP/byte**

RTX 6000 Ada 的 DRAM bandwidth ≈ 960 GB/s，FP64 peak ≈ 1457 GFLOPS，roofline 交叉点 ≈ 1.52 FLOP/byte。**这个 kernel 恰好在 roofline 交叉点附近**，既不是纯 compute-bound 也不是纯 memory-bound，这解释了为什么 Nsight 显示 SM throughput 84% 但 DRAM throughput 只有 6-11%（L2 cache 命中率高）。

**MemoryBond Kernel（`ring_cnot_layer`）：**

每个 thread 做一次 gather（读 source index）+ 一次 scatter（写 output）：
- **读取**：1 × 16 bytes（source amplitude）
- **写入**：1 × 16 bytes（output amplitude）
- **计算**：`inverse_ring_cnot_source_index` 函数，约 10 个整数 XOR/shift 操作，≈ 10 integer ops
- **Arithmetic Intensity** ≈ 10 × 0.5 FLOP / 32 bytes ≈ **0.16 FLOP/byte**

远低于 roofline 交叉点，是纯 memory-bound。Nsight 数据验证：DRAM throughput 89.5%，SM throughput 0.8%。

### 1.2 Arithmetic Intensity 预测公式（论文贡献）

```
AI(chunk_start, W) = (12W × 2) / (4 × 16 × 2)   [ComputeBond]
AI(ring_cnot)      = 10 / 32                       [MemoryBond]
```

当 `chunk_start + W ≤ log2(THREADS)` 时，amplitude pair 在同一 warp 内，L2 hit rate 高，实际 effective bandwidth 远低于 DRAM bandwidth，kernel 表现为 compute-bound。当 `chunk_start > log2(THREADS)` 时，partner amplitude 跨 warp，L2 miss 率上升，kernel 向 memory-bound 漂移。**这就是为什么高位 chunk 需要 transposed mapping**。

---

## 二、Tile-local / Tile-nonlocal 的精确边界与 Kernel 选择规则

### 2.1 从代码推导精确边界

从 `build_structured_forward_rotation_chunks` 和 `structured_effective_backward_chunk_width` 可以提取出当前的启发式规则：

```
if num_qubits <= 15:
    forward: W=8 Cooperative
    backward: W=4 chunk
elif 16 <= num_qubits <= 19:
    forward: W=8 CooperativePair512 (wire 0-15) + W=4 Register (wire 16+)
    backward: W=1 (per-wire)  [注意：W>=8 且 num_qubits 16-19 时退化为 W=1]
elif num_qubits >= 20:
    forward: W=8 CooperativePair512 (wire 0-15) + W=4 Register (wire 16+)
    backward: W=8 cell2 (wire 0-7) + W=4 transposed cell2 (wire 8+)
```

### 2.2 Qubit Locality Score (QLS) 形式化（论文贡献）

```
QLS(wire, num_qubits) = log2(state_size) - wire = num_qubits - wire
```

- `QLS >= log2(THREADS) = 8`：amplitude pair 在同一 block 内，**Tile-local**，适合 Cooperative/cell2 kernel
- `QLS < 8`：amplitude pair 跨 block，**Tile-nonlocal**，需要 transposed mapping 或 register tile

**Tile-local 的精确定义**：wire `w` 是 tile-local 当且仅当 `w < log2(THREADS)`，即 `w < 8`（THREADS=256）。

**Tile-nonlocal 的精确定义**：wire `w` 是 tile-nonlocal 当且仅当 `w >= 8`。

这个定义直接对应代码中 `wire 0-7` 用 W8 cell2，`wire 8+` 用 transposed W4 cell2 的分层策略。

### 2.3 已知 Gap：num_qubits = 16-19 的 backward 退化

当前代码在 `num_qubits = 16-19` 时，backward 退化为 W=1（per-wire），而不是使用 W4 或 W8。这是一个已知的性能 gap，**可能可以用 transposed W4 cell2 填补**，预计有 2-3x 的提升空间。这是一个直接可以做的实验。

---

## 三、MemoryBond 的深度优化：Ring-CNOT Kernel 的精确分析

### 3.1 当前实现的 GF(2) 闭式公式

从代码 `inverse_ring_cnot_source_index` 可以看到：

```cpp
prefix ^= prefix << 1U;
prefix ^= prefix << 2U;
prefix ^= prefix << 4U;
prefix ^= prefix << 8U;
prefix ^= prefix << 16U;
prefix ^= prefix << 32U;
prefix &= mask;
auto transformed = prefix & ~std::size_t{1};
transformed |= __popcll(index >> 1U) & 1U;
```

这是一个 **GF(2) 线性变换的 prefix XOR**，把 ring-CNOT 的逐 bit 模拟换成了 O(log n) 的整数操作。

### 3.2 优化方向 A：Shared-Memory Block Transpose for Ring-CNOT

对于 `num_qubits <= 16`（state_size <= 65536），整个 statevector 可以放入 L2 cache（65536 × 16 bytes = 1 MB，RTX 6000 Ada L2 = 96 MB）。此时 ring-CNOT 的随机访存实际上是 L2 hit，DRAM throughput 会下降。

对于 `num_qubits >= 20`，state_size = 16 MB，L2 放不下，每次 gather 都是 DRAM miss。**优化方案**：把 statevector 分成 `2^(num_qubits - 16)` 个 64K-amplitude 的 block，每个 block 内的 ring-CNOT 置换可以用 shared memory 完成。

### 3.3 优化方向 B：Fused Ring-CNOT + Rotation（已在代码中实现）

代码中已有 `inverse_ring_cnot_then_w4_rotation_chunk_kernel`，把 inverse ring-CNOT 和 W4 rotation chunk 融合成一个 kernel：

```cpp
const auto out_index = base | (local_index << chunk_start);
return current_in[inverse_ring_cnot_source_index(out_index, num_qubits)];
```

即在 load 阶段直接从 ring-CNOT 的 source index 读取，省掉了一次 global memory round-trip。

**理论收益**：
- 不融合：ring-CNOT（1次 global read + 1次 global write）+ rotation（1次 global read + 1次 global write）= 4次 global memory 访问
- 融合后：1次 global read（从 source index）+ 1次 global write（rotation 结果）= 2次 global memory 访问
- 理论节省 50% global memory traffic

**反直觉结果**：代码注释说这个融合在 24x1 backward 反而从 22.3ms 退化到 24.7ms。**根本原因分析**：fusion 后每个 thread 需要计算 `inverse_ring_cnot_source_index`（约 10 个整数操作），这增加了 warp 内的 instruction divergence（不同 thread 的 source index 不同，导致 memory access pattern 更随机），抵消了减少 launch 的收益。这说明 **memory access pattern 的 regularity 比 launch count 更重要**，是一个值得在论文中深入分析的 insight。

---

## 四、ComputeBond 的深度优化：Cell2 Kernel 的精确分析

### 4.1 Cell2 的数学结构

从 `inverse_walk_ryrz_rotation_chunk_w4_cell2_kernel` 代码可以看到，W=4 的 cell2 kernel 把 4 个 wire 的 backward 分成两个阶段：

**阶段 1（高位 wire 2,3，在寄存器中）**：
```
处理 wire 3: (c0,c2), (c1,c3) 两个 pair
处理 wire 2: (c0,c1), (c2,c3) 两个 pair
```
这 4 个 amplitude 全在寄存器中，**零 shared memory 访问**。

**阶段 2（低位 wire 0,1，从 shared memory 读取）**：
```
处理 wire 1: (c0,c2), (c1,c3) 两个 pair
处理 wire 0: (c0,c1), (c2,c3) 两个 pair
```

**关键洞察**：原来的 W4 cooperative kernel 每个 wire 都需要 `__syncthreads()`（4次），cell2 把 wire 2,3 合并到寄存器，只需要 1次 `__syncthreads()`（在阶段 1 和阶段 2 之间）。

### 4.2 `__syncthreads()` 成本的精确量化（论文贡献）

- 在 RTX 6000 Ada 上，`__syncthreads()` 的 latency 约 20-30 cycles
- W4 cooperative kernel：4次 sync × 30 cycles = 120 cycles per tile
- W4 cell2 kernel：1次 sync × 30 cycles = 30 cycles per tile
- **节省 75% 的 sync overhead**

这解释了为什么 cell2 的 active warps 从 65.6% 降到 32.9%，但 SM throughput 仍维持 84%：**更少的 sync 意味着更少的 warp stall，每个 warp 的有效计算时间更长**。

### 4.3 Cell2 的 Shared Memory 使用分析

W4 cell2 kernel 的 shared memory 布局：
```cpp
constexpr int block_threads = 128;
constexpr int tile_dim = 16;       // 2^4
constexpr int tiles_per_block = 32;
constexpr int shared_values = 512;

Complex current_tile[512];  // 8 KB
Complex lambda_tile[512];   // 8 KB
double theta_partial[4×128]; // 4 KB
double phi_partial[4×128];   // 4 KB
// Total: 24 KB
```

ptxas 报告 24KB shared，与计算一致。RTX 6000 Ada 每个 SM 有 128KB shared memory，24KB 允许 5 个 block 同时驻留（occupancy ≈ 42%）。

**潜在优化**：把 `theta_partial` 和 `phi_partial` 从 shared memory 移到寄存器（每个 thread 只需要 4 个 double），可以把 shared memory 从 24KB 降到 16KB，允许 8 个 block 驻留（occupancy ≈ 67%）。

### 4.4 W8 Cell2 的 Pair512 路径

W8 cell2 使用 128-thread kernel，每个 block 处理 2 个 256-amplitude tile：
```
shared memory: 2 × 256 × 16 bytes × 2 (current + lambda) = 16 KB
+ theta/phi partial: 8 × 128 × 8 bytes × 2 = 16 KB
Total: 32 KB
```

ptxas 报告 32KB shared，78 registers，no spill。32KB shared 允许 4 个 block 驻留（occupancy ≈ 33%），与 Nsight 报告的 active warps 24.7% 一致。

**为什么 256-thread/4-tile 版本（64KB shared）不可行**：超过 48KB 的 shared memory 上限（默认 carveout），需要 `cudaFuncSetAttribute` 设置 `cudaFuncAttributeMaxDynamicSharedMemorySize`，但这会降低 L1 cache 大小，对 ring-CNOT 等 memory-bound kernel 有负面影响。

### 4.5 Occupancy 不是好指标（论文 Insight）

Cell2 kernel 的 active warps 从 65.6%（transposed W4）降到 32.9%（W8 cell2），但性能反而更好。这说明 **occupancy 不是 GPU 性能的好指标**，真正重要的是 **instruction throughput per warp**。论文可以用这个例子说明为什么传统的 occupancy-based 优化方法不适用于 QML 梯度计算。

---

## 五、Persistent CUDA Graph 的精确实现方案

### 5.1 当前 Launch Overhead 的量化

从 Nsight 数据（24x1 structured_w8）：
- `backward_w4_cell2`：24 launches
- `backward_w8_cell2`：6 launches
- `ring_cnot_layer`：20 launches
- `forward_w4_register`：42 launches
- **总计：92 次 kernel launch**

每次 `cudaLaunchKernel` 约 5-10 μs，92 次 × 7 μs ≈ **0.64 ms launch overhead**。对于 8x8（total 0.41 ms），这个 overhead 甚至超过了总时间，说明小规模下 launch overhead 是主要瓶颈。

**CUDA Graph 的收益**：把整个 forward + backward sweep 编译成一个 CUDA Graph，launch overhead 从 0.64 ms 降到约 0.01 ms（一次 `cudaGraphLaunch`）。

### 5.2 CUDA Graph 的实现挑战与解决方案

当前代码的 backward sweep 有动态分支：
```cpp
if (try_launch_fused_inverse_cnot_first_backward_chunk(...)) {
    // 融合路径
} else {
    // 分离路径
}
```

**解决方案**：在 `run_energy_and_grad_structured_adjoint` 的第一次调用时，根据 `num_qubits` 和 `num_layers` 静态确定所有 kernel 的 launch 序列，然后用 `cudaStreamBeginCapture` / `cudaStreamEndCapture` 录制成 CUDA Graph，后续调用直接 `cudaGraphLaunch`。

参数（`params`）每次调用都不同，但 CUDA Graph 支持通过 `cudaGraphExecKernelNodeSetParams` 更新参数，或者把参数放在 device memory 中（已经是这样做的），Graph 只需要更新 device memory 指针。

**论文贡献**：给出一个 **Circuit-Aware CUDA Graph Construction** 算法，输入电路描述，输出静态 kernel launch 序列，并证明其正确性（所有动态分支都可以在编译时确定）。

---

## 六、参数搜索的精确技术方案

### 6.1 搜索空间的精确定义

从代码中可以提取出完整的 kernel 配置参数空间：

```python
Config = {
    "structured_rotation_chunk_width": [1, 2, 4, 8],
    "forward_chunk_type": ["Cooperative", "CooperativePair512", "Register"],
    "backward_chunk_type": ["standard", "transposed", "cell2"],
    "cnot_fusion": ["fused", "separate"],
    "pair512_threshold": [16, 20, 24],   # num_qubits threshold
    "cell2_threshold": [20, 24],          # num_qubits threshold
    "high_wire_boundary": [8, 12, 16],    # boundary between low/high wire
}
```

当前代码的配置是手工调优的（通过 A/B 实验）。**论文贡献**：把这个搜索空间形式化，并用一个轻量 predictor 自动搜索。

### 6.2 Predictor 的精确设计

**输入特征**（从电路描述提取）：
```python
features = [
    num_qubits,                    # 电路规模
    num_layers,                    # 层数
    log2(state_size),              # = num_qubits
    num_qubits / 8,                # tile-local qubit 比例
    num_layers * num_qubits,       # 总参数数
    log2(THREADS),                 # = 8（GPU 固定）
    shared_memory_per_sm,          # = 128 KB（GPU 固定）
    l2_cache_size,                 # = 96 MB（GPU 固定）
]
```

**输出**：最优 kernel 配置（分类问题）或预测 backward time（回归问题）

**训练数据**：现有 benchmark 结果（8x8, 12x8, 16x4, 20x2, 24x1）× 所有配置组合 ≈ 5 × 4 × 3 × 2 × 3 × 3 × 3 = 1620 个数据点。

**推荐模型**：
1. **决策树**：可解释性强，适合 PLDI 论文（可以把决策树的分支条件直接写成 kernel 选择规则）
2. **MLP（2层，32个隐藏节点）**：足够拟合，推理开销 < 1 μs
3. **Lookup Table**：对 `(num_qubits, num_layers)` 的离散组合直接查表，最简单

### 6.3 Online Adaptive Selection 机制（论文 Novelty）

**核心思路**：不是 predictor 本身，而是**把 kernel 配置搜索和 QML 训练循环统一**：
1. 训练的前 K 步（K=5-10）用 profiling 模式收集 stage timing
2. 用 predictor 选择最优配置
3. 后续步骤使用固定配置（CUDA Graph）

这是一个 **online adaptive kernel selection** 机制，overhead 只在前 K 步，后续零额外成本。

---

## 七、电路分析与变换的精确技术方案

### 7.1 门顺序交换的形式化（Circuit Transformation Pass）

**关键性质**：同一 layer 内不同 wire 的 RY/RZ 门两两对易（作用在不同 tensor factor 上）。

**形式化变换规则**：

```
Rule 1 (Rotation Chunk Fusion):
  If gates G_i and G_j act on disjoint qubits and are in the same layer,
  they can be fused into a single chunk kernel.
  Cost reduction: n kernel launches → ceil(n/W) kernel launches.

Rule 2 (CNOT Layer Fusion):
  If all CNOTs in a layer form a permutation (ring topology),
  they can be fused into a single permutation kernel.
  Cost reduction: n kernel launches → 1 kernel launch.

Rule 3 (CNOT-Rotation Fusion):
  If the last CNOT layer and the first rotation chunk of the next backward step
  can be fused (as in inverse_ring_cnot_then_w4_rotation_chunk_kernel),
  save 1 global memory round-trip.
  Condition: chunk_start + W >= num_qubits - 4 (high-wire chunk).
  Warning: May degrade due to memory access irregularity (see Section 3.3).
```

**论文贡献**：给出一个 **Circuit Transformation Pass** 的形式化描述，类似编译器的 IR pass，输入是门级电路描述（`OpDesc` 序列），输出是优化后的 kernel launch 序列。

### 7.2 电路压缩：参数化 vs 固定子图

从代码可以看到两类 op：
```cpp
enum class OpKind : int { 
    RY, RZ,           // 参数化，每次调用参数不同
    CNOT,             // 固定，每次调用相同
    FusedRYRZ,        // 参数化，fused
    RotationLayer,    // 参数化，整层
    RingCNOTLayer     // 固定，整层
};
```

**论文贡献**：把电路分析成两类子图：
1. **参数化子图**（RotationLayer）：每次训练步参数不同，必须重新计算
2. **固定子图**（RingCNOTLayer）：每次训练步相同，可以预计算或缓存

对于固定子图，可以预计算其对应的 permutation table（ring-CNOT 的 `inverse_ring_cnot_source_index` 函数），存储为 `uint32_t` 数组（24 qubit 需要 16M × 4 bytes = 64 MB），用 texture memory 加速随机访问。

---

## 八、QML 味道的核心：Adjoint Diff 的 GPU 瓶颈分析

### 8.1 Stage Profile 分析

从 stage profile 数据（24x1 structured_w8，最终版本）：
```
forward:      6.4 ms  (19%)
backward:    22.4 ms  (66%)
hamiltonian:  4.0 ms  (12%)
dot:          0.7 ms   (2%)
```

**Backward 占 66% 的时间**，这是 QML 训练的主要瓶颈。

### 8.2 批量梯度计算（Batch Gradient）

当前代码每次计算一个参数集的梯度。QML 训练中常见的需求是**批量梯度**（多个参数集同时计算梯度，用于 mini-batch 训练或 natural gradient）。

**技术方案**：把 `current` 和 `lambda` 从单个 statevector 扩展为 `batch × state_size` 的矩阵，每个 kernel 处理 batch 个 statevector。

- 对于 ComputeBond kernel（rotation chunk）：batch 维度作为额外的 grid dimension，`gridDim.y = batch_size`
- 对于 MemoryBond kernel（ring-CNOT）：同样可以 batch，但 DRAM bandwidth 会成为更严重的瓶颈

**论文贡献**：给出 batch adjoint diff 的 GPU 实现，分析 batch size 对 ComputeBond/MemoryBond 比例的影响，以及最优 batch size 的选择规则。

### 8.3 性能数据汇总

| case | inverse_walk ms/step | structured_adjoint ms/step | vs inverse_walk | vs PennyLane |
|---|---:|---:|---:|---:|
| 8x8 | 1.549 | 0.737 | 2.10x | 21.74x |
| 12x8 | 3.011 | 0.617 | 4.88x | — |
| 16x4 | 2.208 | 0.951 | 2.32x | 18.31x |
| 20x2 | 12.828 | 3.511 | 3.65x | 3.33x |
| 24x1 | 152.759 | 36.226 | 4.22x | 5.82x |

---

## 九、论文框架精确技术贡献总结

```
Title: Adaptive Kernel Generation for QML Simulation and Training on GPU

Section 3: Circuit Analysis
  3.1 Qubit Locality Score (QLS) 定义
      QLS(wire) = num_qubits - wire
      Tile-local: wire < log2(THREADS) = 8
      Tile-nonlocal: wire >= 8
  3.2 Arithmetic Intensity 预测公式
      AI(ComputeBond) ≈ 1.5 FLOP/byte（roofline 交叉点附近）
      AI(MemoryBond)  ≈ 0.16 FLOP/byte（纯 memory-bound）
  3.3 Circuit Transformation Pass
      Rule 1: Rotation Chunk Fusion（对易性）
      Rule 2: CNOT Layer Fusion（置换结构）
      Rule 3: CNOT-Rotation Fusion（条件性，需验证 memory regularity）
  3.4 参数化 vs 固定子图分析

Section 4: Kernel Facilities
  4.1 ComputeBond Kernel 设计
      - W4/W8 Cooperative Chunk（shared memory tile）
      - Cell2 Fusion（寄存器内两门合并，消除 75% sync overhead）
      - Transposed Mapping（高位 chunk 的 coalescing 优化）
      - Pair512（低位大规模的 block-level 并行）
  4.2 MemoryBond Kernel 设计
      - GF(2) 闭式 ring-CNOT 索引（O(log n) 整数操作）
      - 双缓冲指针交换（消除 device-to-device copy）
      - CNOT-Rotation Fusion（条件性使用）
  4.3 Persistent CUDA Graph（消除 launch overhead，小规模收益显著）
  4.4 Async Pipeline（forward/backward overlap，三缓冲）

Section 5: Adaptive Parameter Search
  5.1 搜索空间定义（chunk_width, kernel_type, fusion_threshold）
  5.2 Roofline-based 特征提取（AI, QLS, state_size）
  5.3 Decision Tree Predictor（可解释，直接生成 kernel 选择规则）
  5.4 Online Adaptive Selection（训练前 K 步 profiling，后续 CUDA Graph）

Section 6: Evaluation
  6.1 vs PennyLane：最高 21.74x（8x8），最低 3.33x（20x2）
  6.2 vs inverse_walk：最高 4.88x（12x8），最低 2.10x（8x8）
  6.3 Ablation：每个优化的独立贡献
      - W1 vs W8：backward chunk fusion 的收益
      - cell2 vs transposed：两种高位优化的对比
      - CUDA Graph：launch overhead 的消除
  6.4 Roofline 分析：验证 ComputeBond/MemoryBond 分类的准确性
  6.5 Batch Gradient：batch size 对性能的影响
  6.6 Gap 填补：num_qubits=16-19 的 transposed W4 cell2 实验
```

---

## 十、组会重点讨论的 3 个技术问题

### 问题 1：CNOT-Rotation Fusion 的退化原因

`inverse_ring_cnot_then_w4_rotation_chunk_kernel` 在 24x1 backward 反而从 22.3ms 退化到 24.7ms，但理论上应该节省 50% global memory traffic。

**分析**：fusion 后每个 thread 需要计算 `inverse_ring_cnot_source_index`（约 10 个整数操作），这增加了 warp 内的 instruction divergence（不同 thread 的 source index 不同，导致 memory access pattern 更随机），抵消了减少 launch 的收益。

**结论**：**memory access pattern 的 regularity 比 launch count 更重要**。这是一个值得深入的 insight，可以作为论文的一个 negative result 来分析。

### 问题 2：Occupancy 与 SM Throughput 的解耦

Cell2 kernel 的 active warps 从 65.6%（transposed W4）降到 32.9%（W8 cell2），但性能反而更好（SM throughput 仍 84%）。

**结论**：传统的 occupancy-based 优化方法不适用于 QML 梯度计算。真正重要的是 **instruction throughput per warp**，而不是 warp 数量。论文可以用 roofline model 的扩展版本（考虑 sync overhead）来解释这个现象。

### 问题 3：num_qubits = 16-19 的 Gap

当前代码在 `num_qubits = 16-19` 时，backward 退化为 W=1（per-wire）。

**可行实验**：对 16-19 qubit 使用 transposed W4 cell2（与 20+ qubit 的高位策略相同），预计 backward 时间可以从 W=1 的水平降低 2-3x。这是一个直接可以实现和验证的优化，可以作为论文的一个新实验结果。

---

## 附录：关键代码位置索引

| 概念 | 代码位置 |
|---|---|
| MemoryBond kernel（ring-CNOT） | `cpp/ising_cuda_kernels.cu` → `ring_cnot_layer_kernel` |
| ComputeBond kernel（W8 chunk） | `cpp/ising_cuda_statevector_grad_kernels.cu` → `inverse_walk_ryrz_rotation_chunk_kernel<8>` |
| Transposed mapping | `cpp/ising_cuda_statevector_grad_kernels.cu` → `inverse_walk_ryrz_rotation_chunk_transposed_kernel` |
| Cell2 kernel（W4） | `cpp/ising_cuda_statevector_grad_kernels.cu` → `inverse_walk_ryrz_rotation_chunk_w4_cell2_kernel` |
| CNOT-Rotation Fusion | `cpp/ising_cuda_statevector_grad_kernels.cu` → `inverse_ring_cnot_then_w4_rotation_chunk_kernel` |
| Kernel 选择逻辑 | `cpp/ising_cuda_structured_adjoint_modes.inc` → `structured_effective_backward_chunk_width` |
| Forward chunk 选择 | `cpp/ising_cuda_structured_adjoint_modes.inc` → `build_structured_forward_rotation_chunks` |
| GF(2) 闭式公式 | `cpp/ising_cuda_statevector_grad_kernels.cu` → `inverse_ring_cnot_source_index` |
| 双缓冲指针交换 | `cpp/ising_cuda_structured_adjoint_modes.inc` → `run_structured_backward_sweep` |
| QLS 对应的 THREADS 常量 | `cpp/ising_cuda_backend_internal.cuh` → `constexpr int THREADS = 256` |
