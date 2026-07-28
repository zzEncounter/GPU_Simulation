# structured_adjoint 未来优化路线图

> 基于当前已实现的双 Stream 并行 Ring-CNOT，系统梳理 `structured_adjoint` 策略中所有可能的进一步优化方向，包含算法原理、实现方案、预期收益与实现难度。

---

## 当前状态总结

| 优化 | 状态 | 收益 |
|------|------|------|
| 双 Buffer Ping-Pong | ✅ 已实现 | 正确性基础 |
| 双 Stream 并行 Ring-CNOT（Backward） | ✅ 已实现 | backward ~1.27x |
| Fused CNOT+Rotation（串行版，大规模） | ✅ 已实现 | 减少显存读写 |

**当前 Backward 每层时序**（双 Stream 版本）：
```
stream_c: |--Ring-CNOT(current)--|
stream_l: |--Ring-CNOT(lambda)--|   ← 并行
                                 ↓ event 同步
default:                         |--Rotation+Grad--|
←──────────── T_cnot + T_rot ────────────────────→
```

---

## 优化方向一：Backward 跨层 Overlap（最高优先级）

### 1.1 核心思路

**问题**：当前每层的 Rotation+Grad 是 compute-bound（SM 84%），Ring-CNOT 是 memory-bound（DRAM 89.7%）。两者使用**不同的硬件资源**，理论上可以 overlap。

**障碍**：Layer L-1 的 Ring-CNOT 需要 Layer L Rotation 修改后的 `current` 和 `lambda`，存在数据依赖，无法直接 overlap。

**解决方案**：把 Rotation+Grad 拆成两个独立的 kernel：

| Kernel | 操作 | 读写关系 |
|--------|------|---------|
| `kernel_gradient_only` | 只计算梯度，**不修改状态** | 只读 `current`、`lambda`，只写 `grad` |
| `kernel_state_update` | 对状态施加逆向旋转 | 读写 `current`、`lambda` |

**关键洞察**：
- `kernel_gradient_only` 完成后，`current` 和 `lambda` 还未被修改
- Layer L-1 的 Ring-CNOT 需要的是 `kernel_state_update` 之后的状态
- 但 `kernel_gradient_only` 完成后，就可以**提前发射** Layer L-1 的 Ring-CNOT（它读的是 `state_update` 之后的 buffer，而 `state_update` 会在 Ring-CNOT 之前完成）

### 1.2 优化后的时序

```
时间轴 →

Layer L：
  stream_c: |--Ring-CNOT(current_L)--|
  stream_l: |--Ring-CNOT(lambda_L)--|
                                     ↓ event 同步
  default:  |--gradient_only(L)--|
  stream_c:                      |--Ring-CNOT(current_{L-1})--|  ← 提前发射！
  stream_l:                      |--Ring-CNOT(lambda_{L-1})--|   ← 提前发射！
  default:  |--state_update(L)--|
                                 ↓ event 同步（等 Ring-CNOT L-1 完成）
  default:  |--gradient_only(L-1)--|
  stream_c:                         |--Ring-CNOT(current_{L-2})--|
  ...
```

**节省时间**：每层节省 `min(T_grad_only, T_cnot)` 的时间（两者 overlap 的部分）。

### 1.3 详细实现方案

#### 新增 Kernel 接口

```cpp
// 只计算梯度，不修改 current 和 lambda
void launch_structured_backward_gradient_only(
    const Complex *current,    // 只读
    const Complex *lambda,     // 只读
    std::size_t state_size,
    std::size_t layer,
    StructuredRotationChunk chunk,
    double *gradients,         // 只写
    cudaStream_t stream = 0);

// 只对状态施加逆向旋转，不计算梯度
void launch_structured_backward_state_update(
    Complex *current,          // 读写
    Complex *lambda,           // 读写
    std::size_t state_size,
    std::size_t layer,
    StructuredRotationChunk chunk,
    const double *params,
    cudaStream_t stream = 0);
```

#### 新增 Stream 和 Event 资源

```cpp
// 在 Impl 结构体中新增（在现有 pipeline_stream_* 基础上）：
cudaStream_t overlap_stream_current;   // Layer L-1 的 Ring-CNOT(current)
cudaStream_t overlap_stream_lambda;    // Layer L-1 的 Ring-CNOT(lambda)
cudaEvent_t  overlap_event_grad_done;  // gradient_only 完成事件
cudaEvent_t  overlap_event_cnot_done_c; // overlap Ring-CNOT(current) 完成
cudaEvent_t  overlap_event_cnot_done_l; // overlap Ring-CNOT(lambda) 完成
```

#### 完整伪代码

```cpp
void run_structured_backward_sweep_cross_layer_overlap(
    Impl &impl, const double *params) {

    // 初始化指针（4 个 buffer）
    auto *cur  = impl.current.get();
    auto *ncur = impl.scratch.get();
    auto *lam  = impl.lambda.get();
    auto *nlam = impl.cnot_scratch.get();

    // 额外需要两个 buffer 用于 overlap 层的 Ring-CNOT 输出
    // （可以复用 cur/ncur/lam/nlam，但需要仔细管理指针）

    for (int layer = impl.num_layers - 1; layer >= 0; layer--) {

        // ── 步骤 1：当前层的 Ring-CNOT（并行） ──────────────────────────
        launch_ring_cnot_inv(ncur, cur, stream_c);
        launch_ring_cnot_inv(nlam, lam, stream_l);
        cudaEventRecord(event_c, stream_c);
        cudaEventRecord(event_l, stream_l);
        cudaStreamWaitEvent(default_stream, event_c);
        cudaStreamWaitEvent(default_stream, event_l);
        swap(cur, ncur); swap(lam, nlam);

        // ── 步骤 2：gradient_only（只读，不修改状态） ────────────────────
        launch_gradient_only(cur, lam, layer, grad_buf, default_stream);
        cudaEventRecord(event_grad_done, default_stream);

        // ── 步骤 3：提前发射下一层的 Ring-CNOT ──────────────────────────
        if (layer > 0) {
            // 等 gradient_only 完成（此时 cur/lam 还未被 state_update 修改）
            cudaStreamWaitEvent(overlap_stream_c, event_grad_done);
            cudaStreamWaitEvent(overlap_stream_l, event_grad_done);
            launch_ring_cnot_inv(ncur, cur, overlap_stream_c);  // 下一层
            launch_ring_cnot_inv(nlam, lam, overlap_stream_l);
            cudaEventRecord(event_overlap_c, overlap_stream_c);
            cudaEventRecord(event_overlap_l, overlap_stream_l);
        }

        // ── 步骤 4：state_update（与下一层 Ring-CNOT 并行） ─────────────
        launch_state_update(cur, lam, layer, params, default_stream);

        // ── 步骤 5：等待下一层 Ring-CNOT 完成 ───────────────────────────
        if (layer > 0) {
            cudaStreamWaitEvent(default_stream, event_overlap_c);
            cudaStreamWaitEvent(default_stream, event_overlap_l);
            swap(cur, ncur); swap(lam, nlam);
            // 下一层循环直接跳过 Ring-CNOT（已经完成）
        }
    }
}
```

### 1.4 预期收益

设：
- `T_grad`：gradient_only 的时间（约为原 Rotation+Grad 的 60-70%）
- `T_update`：state_update 的时间（约为原 Rotation+Grad 的 30-40%）
- `T_cnot`：单次 Ring-CNOT 的时间

**overlap 后每层时间**：
```
T_overlap = T_cnot + T_grad + max(T_update, T_cnot - T_grad)
          ≈ T_cnot + T_grad  （当 T_cnot < T_grad 时，state_update 完全被 overlap）
```

| 规模 | 当前（双 stream） | 跨层 overlap 后 | 额外加速比 |
|------|----------------|----------------|-----------|
| 24x1 | T_cnot + T_rot | T_cnot + T_grad | ~1.3-1.5x |
| 24x4 | T_cnot + T_rot | T_cnot + T_grad | ~1.3-1.5x |

**实现难度**：高（需要新增 kernel、管理复杂的 stream 依赖、验证正确性）

---

## 优化方向二：CUDA Graph for Forward（最容易实现）

### 2.1 核心思路

**问题**：Forward 阶段每个 step 都需要 CPU 逐个 launch kernel（每层 Rotation + Ring-CNOT），有 CPU launch overhead（每次 launch ~1-5 us，多层累积可达数十 us）。

**解决方案**：CUDA Graph 可以把一系列 kernel launch 录制成一个"图"，后续 replay 时只需一次 `cudaGraphLaunch`，消除所有 CPU launch overhead。

**代码里已有 benchmark 接口**（说明这个方向已被考虑）：
```cpp
auto RingIsingCudaBackend::benchmark_structured_forward_graph(
    const double *params, std::size_t num_params, std::size_t repeats)
    -> CudaGraphBenchmarkResult;
```

### 2.2 实现方案

#### 录制阶段（第一次 Forward）

```cpp
cudaGraph_t forward_graph;
cudaGraphExec_t forward_graph_exec;

// 开始录制
cudaStreamBeginCapture(default_stream, cudaStreamCaptureModeGlobal);

// 执行一次完整的 Forward（所有 kernel launch 被录制）
apply_structured_forward_layers(impl, params, rotation_chunks);

// 结束录制，生成 graph
cudaStreamEndCapture(default_stream, &forward_graph);
cudaGraphInstantiate(&forward_graph_exec, forward_graph, nullptr, nullptr, 0);
```

#### Replay 阶段（后续每个 step）

```cpp
// 更新参数（通过 cudaGraphExecKernelNodeSetParams 或 pinned memory）
update_graph_params(forward_graph_exec, new_params);

// 一次 launch 替代所有 kernel launch
cudaGraphLaunch(forward_graph_exec, default_stream);
cudaStreamSynchronize(default_stream);
```

#### 参数更新策略

由于每个 step 的参数都不同，有两种方案：

**方案 A（推荐）**：使用 pinned memory 存储参数，kernel 通过指针读取
- 每次 step 只需更新 pinned memory 中的参数值
- Graph 中的 kernel 指针不变，自动读取新参数
- 无需重新录制 graph

**方案 B**：使用 `cudaGraphExecKernelNodeSetParams` 更新每个 kernel 节点的参数
- 需要遍历所有 kernel 节点，overhead 较大
- 不推荐

### 2.3 预期收益

| 规模 | Forward kernel 数量 | CPU launch overhead | 预期加速 |
|------|-------------------|-------------------|---------|
| 12x4 | ~12 kernels | ~60 us | ~1.05x |
| 20x8 | ~40 kernels | ~200 us | ~1.10x |
| 24x16 | ~80 kernels | ~400 us | ~1.15x |

**实现难度**：低（代码里已有 benchmark 接口，主要工作是在训练循环中启用）

---

## 优化方向三：多 Step CPU-GPU 异步流水线

### 3.1 核心思路

**当前执行模式**（CPU 和 GPU 交替等待）：
```
CPU: |--准备参数--|                    |--读梯度+更新参数--|                    |--...|
GPU:             |--forward+backward--|                   |--forward+backward--|
     ←── step t ──────────────────────→←── step t+1 ──────────────────────────→
```

**优化后**（CPU 和 GPU 流水线并行）：
```
GPU: |--step t forward+backward--|--step t+1 forward+backward--|--step t+2--|
CPU:                              |--更新参数(t)--|              |--更新(t+1)--|
     ←── step t ──────────────────→←── step t+1 ──────────────────────────────→
```

### 3.2 实现方案

#### 关键技术

1. **Pinned Memory（页锁定内存）**：
   - 使用 `cudaMallocHost` 分配参数和梯度的 CPU 内存
   - Pinned memory 支持 DMA 直接传输，比普通内存快 2-3x
   - 支持 `cudaMemcpyAsync`（异步传输，不阻塞 CPU）

2. **双缓冲参数**：
   - `params_buf[0]`：step t 的参数（GPU 正在使用）
   - `params_buf[1]`：step t+1 的参数（CPU 正在更新）

3. **CUDA Event 同步**：
   - `event_grad_ready`：GPU 完成梯度计算，CPU 可以读取
   - `event_params_ready`：CPU 完成参数更新，GPU 可以开始下一个 step

#### 伪代码

```cpp
// 初始化双缓冲
double *params_pinned[2], *grad_pinned[2];
cudaMallocHost(&params_pinned[0], param_size);
cudaMallocHost(&params_pinned[1], param_size);
cudaMallocHost(&grad_pinned[0], param_size);
cudaMallocHost(&grad_pinned[1], param_size);

cudaEvent_t event_grad_ready[2], event_params_ready[2];
// ... 初始化 events ...

// 第一个 step 正常执行（流水线预热）
memcpy(params_pinned[0], initial_params, param_size);
cudaMemcpyAsync(gpu_params, params_pinned[0], param_size, H2D, stream);
launch_forward_backward(gpu_params, grad_pinned[0], stream);
cudaEventRecord(event_grad_ready[0], stream);

// 流水线主循环
for (int step = 1; step < total_steps; step++) {
    int cur = step % 2;
    int prev = 1 - cur;

    // CPU：等待上一个 step 的梯度，更新参数
    cudaEventSynchronize(event_grad_ready[prev]);
    update_params(params_pinned[cur], params_pinned[prev], grad_pinned[prev], lr);
    cudaEventRecord(event_params_ready[cur], cpu_stream);

    // GPU：等待新参数就绪，开始下一个 step
    cudaStreamWaitEvent(gpu_stream, event_params_ready[cur]);
    cudaMemcpyAsync(gpu_params, params_pinned[cur], param_size, H2D, gpu_stream);
    launch_forward_backward(gpu_params, grad_pinned[cur], gpu_stream);
    cudaEventRecord(event_grad_ready[cur], gpu_stream);
}
```

### 3.3 预期收益

| 场景 | CPU 参数更新时间 | GPU step 时间 | 整体加速比 |
|------|---------------|-------------|-----------|
| 小规模（12x4） | ~0.5 ms | ~2 ms | ~1.20x |
| 中规模（20x8） | ~1 ms | ~10 ms | ~1.10x |
| 大规模（24x16） | ~2 ms | ~50 ms | ~1.04x |

> 注：GPU step 时间越长，CPU 参数更新占比越小，流水线收益越小。

**实现难度**：高（需要修改训练循环、管理双缓冲、处理边界条件）

---

## 优化方向四：Ring-CNOT Shared Memory 优化

### 4.1 核心思路

**当前 Ring-CNOT 的访问模式**：

Ring-CNOT 是一个置换操作，对于 wire `k`，它把态向量中所有满足 `bit_k = 1` 的分量与 `bit_{k+1} = 0` 的分量交换（简化描述）。

对于 `wire < log2(THREADS_PER_BLOCK) = 8` 的情况（tile-local ring-CNOT），所有需要交换的数据都在同一个 thread block 的 shared memory 范围内。

**优化**：用 shared memory 做 block-level 的 transpose，避免 global memory 的随机访问模式：

```
当前：thread 直接读写 global memory（随机访问，cache miss 多）
优化：先把数据加载到 shared memory，在 shared memory 中做置换，再写回 global memory
```

### 4.2 实现方案

```cuda
__global__ void ring_cnot_shared_mem_kernel(
    Complex *output, const Complex *input,
    int num_qubits, int wire, bool inverse) {

    extern __shared__ Complex smem[];  // 大小 = BLOCK_SIZE * sizeof(Complex)

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // 加载到 shared memory
    smem[tid] = input[gid];
    __syncthreads();

    // 在 shared memory 中做置换（无 global memory 随机访问）
    int partner = compute_ring_cnot_partner(tid, wire, inverse);
    Complex val = smem[partner];
    __syncthreads();

    // 写回 global memory（合并访问）
    output[gid] = val;
}
```

### 4.3 适用范围与预期收益

| wire 范围 | 访问模式 | 优化方案 | 预期收益 |
|----------|---------|---------|---------|
| wire < 8 | tile-local（同一 block 内） | shared memory transpose | ~1.2-1.5x |
| wire >= 8 | cross-tile（跨 block） | 无法用 shared memory | 无收益 |

**实现难度**：中（需要理解 Ring-CNOT 的置换模式，正确实现 shared memory 版本）

---

## 优化方向五：Rotation Chunk 内部 ILP 优化

### 5.1 核心思路

当前 Rotation chunk kernel（处理 W 个 qubit 的旋转）内部，W 个 qubit 的旋转操作之间**完全独立**。编译器可能已经自动做了指令级并行（ILP），但可以通过手动展开循环进一步提升。

### 5.2 实现方案

**当前（隐式循环）**：
```cuda
for (int w = 0; w < chunk_width; w++) {
    apply_ry_rz(state, wire + w, theta_ry[w], theta_rz[w]);
}
```

**优化（手动展开，chunk_width=4）**：
```cuda
// 同时处理 4 个 qubit，编译器可以更好地调度指令
Complex val0 = state[idx ^ (1 << wire)];
Complex val1 = state[idx ^ (1 << (wire+1))];
Complex val2 = state[idx ^ (1 << (wire+2))];
Complex val3 = state[idx ^ (1 << (wire+3))];

// 4 个旋转同时计算（利用 FMA 指令）
val0 = apply_ry_rz_fused(val0, theta_ry[0], theta_rz[0]);
val1 = apply_ry_rz_fused(val1, theta_ry[1], theta_rz[1]);
val2 = apply_ry_rz_fused(val2, theta_ry[2], theta_rz[2]);
val3 = apply_ry_rz_fused(val3, theta_ry[3], theta_rz[3]);
```

### 5.3 预期收益

| chunk_width | 当前 SM 利用率 | 优化后预期 | 加速比 |
|------------|-------------|----------|-------|
| 4 | ~84% | ~90% | ~1.07x |
| 8 | ~84% | ~88% | ~1.05x |

**实现难度**：低（主要是 kernel 内部的代码优化，不涉及架构变化）

---

## 优化方向六：流水线版本重新引入 Fused CNOT+Rotation

### 6.1 背景

当前串行版本（`run_structured_backward_sweep`）已经实现了 Fused CNOT+Rotation：把逆向 Ring-CNOT 和第一个 Rotation chunk 融合为一个 kernel，减少一次 global memory 读写。

但流水线版本（`run_structured_backward_sweep_pipelined`）**跳过了这个融合**，因为融合后 stream 依赖关系变复杂。

### 6.2 实现方案

在跨层 Overlap（方向一）实现后，可以重新考虑融合：

```
当前流水线版本：
  stream_c: |--Ring-CNOT(current)--|
  stream_l: |--Ring-CNOT(lambda)--|
                                   ↓ event 同步
  default:                         |--gradient_only--|  ← 第一个 chunk 单独计算

融合后：
  stream_c: |--Ring-CNOT(current) + gradient_chunk_0(current)--|  ← 融合
  stream_l: |--Ring-CNOT(lambda)  + gradient_chunk_0(lambda)--|   ← 融合
                                                                ↓ event 同步
  default:                                                      |--gradient_chunk_1..N--|
```

**关键挑战**：融合 kernel 需要同时读 `current` 和 `lambda`（来自不同 stream），需要在 stream_c 上等待 stream_l 的 Ring-CNOT 完成（或反之），增加了 stream 间依赖。

**实现难度**：高（需要在 stream 内部做跨 stream 同步，或使用 CUDA Graph 管理依赖）

---

## 优化方向七：Batch 梯度计算（多初始态并行）

### 7.1 核心思路

当前每次训练 step 只处理一个初始量子态（`|0⟩`）。在 QML（量子机器学习）场景中，可能需要对多个不同的初始态计算梯度（类似经典 ML 的 mini-batch）。

**优化**：把 `current` 和 `lambda` 扩展为 `batch × state_size` 的矩阵，让多个初始态并行计算梯度。

### 7.2 实现方案

```cpp
// 当前：state_size = 2^N
// 批量：batch_state_size = batch_size × 2^N

// 所有 kernel 增加 batch 维度
launch_apply_ring_cnot_layer_batched(
    next_current, current,
    state_size, num_qubits, batch_size, inverse, stream);

// 梯度累加（对 batch 维度求平均）
launch_accumulate_batch_gradients(
    gradients, batch_gradients, batch_size, num_params);
```

### 7.3 预期收益

| batch_size | 吞吐量提升 | 单样本延迟 | 显存需求 |
|-----------|----------|----------|---------|
| 2 | ~1.8x | 不变 | 2x |
| 4 | ~3.5x | 不变 | 4x |
| 8 | ~6x | 不变 | 8x |

> 注：吞吐量提升不是线性的，受 GPU 并行度和显存带宽限制。

**适用场景**：QML data-driven 训练，每个训练样本对应不同初始态。

**实现难度**：中（需要修改所有 kernel 增加 batch 维度，以及梯度累加逻辑）

---

## 优化方向八：统一 Non-Default Stream（基础工作）

### 8.1 核心思路

当前 Forward 和 Backward 都在 default stream 上执行。Default stream 有隐式全局同步语义（任何 non-default stream 的操作都会等待 default stream 完成），可能引入不必要的同步点。

**优化**：把整个 Forward + Hamiltonian + Backward 都放在同一个 non-default stream 上，避免 default stream 的隐式同步开销。

### 8.2 实现方案

```cpp
// 创建主计算 stream
cudaStream_t main_compute_stream;
cudaStreamCreateWithFlags(&main_compute_stream, cudaStreamNonBlocking);

// 所有操作都在 main_compute_stream 上执行
apply_structured_forward_layers(impl, params, chunks, main_compute_stream);
launch_apply_hamiltonian_energy_partials(..., main_compute_stream);
run_structured_backward_sweep_pipelined(impl, params, main_compute_stream);

// 最终同步
cudaStreamSynchronize(main_compute_stream);
```

**实现难度**：低（主要是把 `stream=0` 改为 `stream=main_compute_stream`，需要仔细检查所有 kernel launch）

---

## 优先级总结与实施路线图

### 优先级矩阵

| 方向 | 预期收益 | 实现难度 | 推荐优先级 | 依赖 |
|------|---------|---------|-----------|------|
| ② CUDA Graph for Forward | forward 1.1-1.2x | **低** | ⭐⭐⭐ 最优先 | 无 |
| ⑧ 统一 Non-Default Stream | 微小 | **低** | ⭐⭐⭐ 基础工作 | 无 |
| ⑤ Rotation Chunk ILP | rotation 1.05-1.1x | **低** | ⭐⭐⭐ 低风险 | 无 |
| ① Backward 跨层 Overlap | backward 1.3-1.5x | **高** | ⭐⭐⭐ 高收益 | 无 |
| ③ 多 Step CPU-GPU 异步 | 整体 1.1-1.2x | **高** | ⭐⭐ 中期 | ② |
| ④ Ring-CNOT Shared Memory | ring-CNOT 1.2-1.5x | **中** | ⭐⭐ 中期 | 无 |
| ⑥ 流水线版 Fused CNOT+Rotation | 减少显存读写 | **高** | ⭐ 长期 | ① |
| ⑦ Batch 梯度计算 | 线性扩展 | **中** | ⭐ 按需 | 无 |

### 建议实施顺序

```
阶段 1（短期，1-2 周）：
  ⑧ 统一 Non-Default Stream → 为后续优化打基础
  ⑤ Rotation Chunk ILP → 低风险，直接提升 kernel 效率
  ② CUDA Graph for Forward → 代码里已有 benchmark 接口，容易落地

阶段 2（中期，2-4 周）：
  ① Backward 跨层 Overlap → 最高收益，需要新增 kernel 和复杂 stream 管理
  ④ Ring-CNOT Shared Memory → 针对小 qubit 规模的专项优化

阶段 3（长期，按需）：
  ③ 多 Step CPU-GPU 异步 → 需要修改训练循环架构
  ⑥ 流水线版 Fused CNOT+Rotation → 依赖方向①完成后再考虑
  ⑦ Batch 梯度计算 → 仅在 QML data-driven 场景下有意义
```

### 预期累积收益

| 阶段 | 新增优化 | 累积 backward 加速比 | 累积整体加速比 |
|------|---------|-------------------|-------------|
| 当前 | 双 Stream Ring-CNOT | 1.27x | 1.20x |
| 阶段 1 | CUDA Graph + ILP | 1.27x | 1.30x |
| 阶段 2 | 跨层 Overlap | 1.65x | 1.50x |
| 阶段 3 | CPU-GPU 异步 | 1.65x | 1.80x |

---

## 附录：性能测量建议

在实施每个优化后，建议使用以下工具验证收益：

### 宏观性能（wall time）
```bash
python run_workflow.py --backend standalone --qubits 24 --layers 4 --steps 20 \
    --gradient-strategy structured_adjoint --report-steps
```

### 阶段级 profiling
```bash
python benchmarks/profile_baseline_stages.py \
    --cases 24x4 20x8 16x16 \
    --repeats 20
```

### Kernel 级 profiling（Nsight Compute）
```bash
python benchmarks/profile_baseline_kernels.py \
    --cases 24x4 \
    --profile-count 5 \
    --csv-out benchmarks/results/optimized_kernel_shares.csv
```

### 数值正确性验证
```bash
python -m unittest tests/test_backends_parity.py -v
```
