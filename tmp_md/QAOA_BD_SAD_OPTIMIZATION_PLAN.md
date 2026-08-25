# QAOA-BD SAD 性能优化计划

## 1. 当前实现基线

当前 `qaoa-bd` 已经支持显式的：

```text
CNOT -> RZ -> CNOT
```

其中所有 qubit 的共享角 RX 由一次整层 rotation kernel 批处理，但 cost 层和
RX mixer 之间尚未 fusion。

当前主要文件：

```text
sad/src/circuits/qaoa_bd.cuh
sad/src/kernels/pair_cnot.cuh
sad/python/sad_baseline/runner.py
benchmark/results/sad_optimized_gpu_bd.csv
```

当前实现的特征：

- `forward_layer_optimized()` 直接调用普通 split forward；
- `backward_layer_optimized()` 直接调用普通 split backward；
- `backward_layer_fused()` 也没有 BD 专用融合；
- 每条 ring edge 分别执行 pair-CNOT、RZ、pair-CNOT；
- pair-CNOT 使用 correctness-first 的普通 permutation kernel；
- 单 wire RZ backward 使用普通 kernel 和 `atomicAdd`；
- RX 层不是逐个 RX 启动 kernel，而是一次整层 shared-parameter rotation kernel；
- cost 层和 RX 层之间没有 fusion；
- 使用的是通用 SAD block/register variant，不是 BD 专用调优 variant。

因此当前版本应被视为：

> 使用通用 optimized runtime 的 QAOA-BD 正确性基线，而不是完成专项性能优化的版本。

## 2. 总体优化目标

优化必须满足以下不变量：

1. 电路数学语义仍为显式 `CNOT-RZ-CNOT`；
2. 参数布局保持 `[beta_0, gamma_0, beta_1, gamma_1, ...]`；
3. 每层所有 RZ 仍共享同一个 gamma 参数；
4. RX 仍为每层一个共享 beta 参数；
5. energy、完整 gradient 和有限差分结果不变；
6. 原始 `qaoa` 的专用 RZZ 路径不受影响；
7. 严格门级模式和融合模式可以并存并独立 benchmark。

优化目标不是让 BD 必须达到原始 QAOA 的速度，而是在保持分解语义的前提下：

- 让 4–28 qubits 全部可运行；
- 减少 kernel launch 数量；
- 减少中间 state 的 global memory 往返；
- 降低共享 gamma 梯度的 atomic contention；
- 建立 BD-specific 的 block/register/tile 配置；
- 让大规模 qubit 的性能增长更稳定。

## 3. 优先级 P0：正确性和 28q 可运行性

这是最高优先级。当前 benchmark 中 4–26q 有成功结果，但 28q 记录为 error，
在原因未明确前不应继续做激进 fusion。

### 3.1 排查内容

使用 `compute-sanitizer` 和独立的小规模测试检查：

- pair-CNOT 的 control/target mask；
- `(n-1, 0)` wrap-around edge；
- phi/lambda ping-pong buffer；
- backward 的 inverse gate order；
- RZ backward 的 gradient offset；
- `atomicAdd` 的地址和并发写入；
- 28q state vector 分配和 workspace 分配；
- kernel grid size 和 `uint64_t` index 计算；
- forward/backward 的临时 buffer 是否发生越界。

### 3.2 对照测试

对 4q、6q、8q、26q、28q 分别记录：

```text
state vector bytes
workspace bytes
forward time
Hamiltonian time
backward time
kernel launch count
GPU peak memory
```

### 3.3 P0 验收标准

```text
4–28q 全部成功
energy error <= 1e-10
gradient error <= 1e-9
optimized 与 legacy 结果一致
compute-sanitizer 无非法访问和 race
```

## 4. 优先级 P1：减少 kernel launch 数量

当前每层有 `n` 条 cost edge，每条 edge 需要：

```text
CNOT -> RZ -> CNOT
```

因此每层大约需要：

```text
2n 个 pair-CNOT kernel
n 个 RZ kernel
1 个 RX kernel
```

8 layers 时 kernel launch 数量很高，launch overhead 和中间 state 写回是主要
瓶颈之一。

### 4.1 matching CNOT kernel

新增按 matching 批处理的接口：

```cpp
launch_pair_cnot_matching(
    StatePair<T>* phi,
    StatePair<T>* lambda,
    uint64_t state_size,
    int parity,
    int qubits,
    int grid_size);
```

matching 定义：

```text
parity = 0: (0,1), (2,3), ...
parity = 1: (1,2), (3,4), ..., (n-1,0)
```

一个 kernel 内根据 basis index 同时执行该 matching 中的所有 CNOT。

约束：

- 偶 matching 和奇 matching 的先后顺序必须保留；
- 不能直接复用旧 `launch_cnot()`，因为它的语义是整圈固定 permutation；
- pair matching kernel 必须支持 phi 和 lambda 同步 permutation；
- forward/backward 必须使用正确的 gather/scatter 方向。

### 4.2 matching RZ kernel

新增：

```cpp
launch_rz_matching_forward(...)
launch_rz_matching_backward(...)
```

同一 matching 中所有 RZ 作用在不同 target wire，但共享 gamma。forward 中每个
amplitude 线程一次性计算所有 target 的 Z eigenvalue，并乘上整个 matching 的
RZ phase。

backward 中：

```text
phi   <- inverse matching RZ
lambda <- inverse matching RZ
gradient[gamma] += all target contributions
```

这样每层的 RZ kernel 可以从 `n` 个减少为 2 个 matching kernel。

## 5. 优先级 P2：BD cost fusion

P1 完成后再实现 BD 专用 cost fusion。

### 5.1 CNOT-RZ-CNOT 局部 fusion

新增 BD 专用 kernel：

```text
qaoa_bd_cost_forward_kernel
qaoa_bd_cost_backward_kernel
```

建议保留两个模式：

```text
SAD_QAOA_BD_FUSION=0
```

严格门级模式：

```text
CNOT -> RZ -> CNOT
```

```text
SAD_QAOA_BD_FUSION=1
```

在一个 kernel 内按照同样的门序执行等价变换，减少中间 state 写回和 kernel
launch。融合 kernel 不得调用旧的：

```text
shared_ring_rzz_factor()
launch_shared_ring_rzz_forward()
launch_shared_ring_rzz_backward()
```

也不能接入旧 QAOA 的 compact lookup 或 cost-RX fused kernel。

### 5.2 fusion 验收

严格模式和融合模式必须满足：

```text
energy difference <= 1e-10
gradient max difference <= 1e-9
finite-difference gradient passes
```

性能比较至少拆分为：

```text
strict split BD
matching BD
fused BD
```

## 6. 优先级 P3：Cost + RX fusion

当前 RX 层已经是一次整层 rotation kernel，但 cost 与 RX 之间没有 fusion：

```text
BD cost layer -> RX layer kernel
```

后续可新增：

```text
qaoa_bd_cost_rx_forward_kernel
qaoa_bd_cost_rx_backward_kernel
```

建议顺序：

1. 先做 forward fusion；
2. backward 保持 split，先验证 forward 的收益；
3. 再评估 backward fusion 是否值得增加复杂度；
4. 对 4–20q 和 22–28q 分别 benchmark。

cost+RX fusion 不能改变 layer 内的逻辑顺序。RX 必须在完整 cost layer 后执行，
不能把 RX 插入某条 CNOT-RZ-CNOT 中间。

## 7. 优先级 P4：pair-CNOT kernel 优化

当前 [`pair_cnot.cuh`](../sad/src/kernels/pair_cnot.cuh) 是普通全状态 permutation，
每次执行都会读取并写回完整 state。

### 7.1 wire topology 特化

建议拆分为：

```text
pair_cnot_adjacent_forward_kernel
pair_cnot_wrap_forward_kernel
pair_cnot_general_forward_kernel
```

其中：

- adjacent：`(0,1)`、`(2,3)` 等相邻 bit；
- wrap：`(n-1,0)`；
- general：其他非相邻 wire pair。

相邻 wire 可以在 tile/register 内完成 permutation，减少 global memory 访问。wrap
edge 需要单独处理高位和最低位 mask，不能假设 wire 连续。

### 7.2 register/tile 调优

第一轮候选：

```text
f32r2_b32r2
f64r2_b64r2
f128r2_b128r2
f128r3_b128r3
f64r4_b64r4
```

必须分别测量：

```text
pair-CNOT forward
pair-CNOT backward
RZ forward
RZ backward
完整 BD layer
energy + gradient
```

不能直接复制旧 QAOA 的 winner，因为旧 QAOA 的主要瓶颈是 diagonal/fused kernel，
BD 的主要瓶颈是大量 permutation 和 kernel launch。

## 8. 优先级 P5：RZ gradient reduction 优化

当前单 wire RZ backward 使用：

```cpp
atomicAdd(gradient + gradient_offset, local);
```

同一层的所有 edge 共享 gamma，可能造成多个 block 对同一个梯度地址竞争。

建议分阶段实现：

1. 每个 block 写入一个 partial gamma gradient；
2. 使用 warp/block reduction；
3. 用第二个 reduction kernel 汇总；
4. matching RZ 中一次性完成所有 target 的 gamma contribution；
5. fused cost backward 中直接完成 gamma reduction。

不要为每个 RZ 创建新的参数槽，否则会破坏 `2 * layers` 的参数布局。

## 9. 优先级 P6：减少 buffer swap

当前每个 CNOT 和 RZ 都可能触发 phi buffer swap；大量 swap 会增加状态管理复杂度。

优化方向：

1. matching kernel 内完成多个 CNOT permutation；
2. 将一个 `CNOT-RZ-CNOT` 作为一个逻辑 operation；
3. 每个 matching 只进行一次 phi/lambda swap；
4. 在 backward 中保持严格逆序；
5. 用连续两个相同 CNOT 恢复原 state 的单元测试验证实现。

## 10. BD-specific variant dispatch

当前 BD 使用通用 library variant，不应直接套用旧 QAOA 的 dispatch 表。

建议在完整 profile 后新增独立标签，例如：

```text
qaoa-bd-f128r2_b128r2
qaoa-bd-f64r4_b64r4
qaoa-bd-f128r3_b128r3
```

第一阶段可以通过 `SAD_LIBRARY_PATH` 手动选择。只有在完整 qubit sweep 和正确性
测试都通过后，才将 BD 规则写入：

```text
sad/python/sad_baseline/runner.py::_select_library()
```

dispatch 至少应区分：

```text
qubits
forward/backward geometry
strict/fused BD mode
```

## 11. Benchmark 设计

每个优化阶段都应输出独立 CSV，避免覆盖原始结果：

```text
qaoa_bd_split.csv
qaoa_bd_matching.csv
qaoa_bd_fused.csv
qaoa_bd_cost_rx_fused.csv
```

固定配置：

```text
qubits    = 4,6,8,...,28
layers    = 8
precision = float64
random_seed = 42
```

每条结果至少保存：

```text
kernel launch count
forward time
Hamiltonian time
backward time
end-to-end time
state vector bytes
workspace bytes
GPU peak memory
energy error
gradient max error
kernel variant
```

比较对象：

```text
原始 qaoa
严格 split qaoa-bd
matching qaoa-bd
fused qaoa-bd
```

不能只比较端到端时间；BD 的性能瓶颈需要通过 forward/backward 分项时间确认。

## 12. 分阶段实施顺序

### Phase 1：稳定性

- 修复 28q error；
- 完成 sanitizer；
- 完成 full-gradient 和 finite-difference；
- 固化 strict split BD 作为 correctness oracle。

### Phase 2：matching kernel

- 实现偶 matching CNOT kernel；
- 实现奇 matching CNOT kernel；
- 实现 matching RZ forward/backward；
- 对比 launch 数量、内存流量和端到端时间。

### Phase 3：BD cost fusion

- 实现 fused `CNOT-RZ-CNOT` forward；
- 实现 fused backward；
- 保留 strict/fused 双路径；
- 完成跨 backend 数值等价测试。

### Phase 4：Cost + RX fusion

- 实现 cost+RX forward fusion；
- 评估 backward fusion；
- 只保留通过端到端 benchmark 的版本。

### Phase 5：参数调优和 dispatch

- 调整 block threads；
- 调整 register bits；
- 调整 tile size；
- 优化 gamma reduction；
- 建立 qubit-dependent BD variant dispatch。

## 13. 完成标准

- 4–28q 全部成功运行；
- strict/fused BD energy 和 full gradient 对齐；
- compute-sanitizer 无错误；
- kernel launch 数量相较当前 split 版本显著下降；
- gamma gradient reduction 不再由高竞争 atomic 成为主要瓶颈；
- 28q 不再出现 benchmark error；
- BD-specific variant 经过完整 qubit sweep 验证；
- 原始 QAOA、RZZ-HEA 和其他电路性能不回退；
- 所有优化结果写入独立 CSV，不覆盖已有 baseline。
