# GPU Simulation 并行优化设计文档

> 文档记录已实现的并行优化和未来可做的加速方向，针对 `structured_adjoint` 策略的 backward sweep。

---

## 背景：Backward Sweep 的时间构成

在 `structured_adjoint` 的 backward sweep 中，每一层的执行结构如下：

```
Layer L（串行，原始版本）：
  ① ring-CNOT(current → next_current)   ~1190 us  [memory-bound, DRAM 89.7%]
  ② ring-CNOT(lambda  → next_lambda)    ~1190 us  [memory-bound, DRAM 89.7%]
  ③ rotation + 梯度(current, lambda)     ~3229 us  [compute-bound, SM 84%]
```

对于 24 qubit × 1 layer（24x1）规模：
- Forward：~6.4 ms
- Backward：~22.4 ms
- 整个训练 step：~28.8 ms

---

## 已实现的优化

### 1. 双 Buffer（Ping-Pong Buffer）——正确性基础

**位置**：`cpp/ising_cuda_structured_adjoint_modes.inc`，`apply_structured_forward_layers` 和 `run_structured_backward_sweep`

**原理**：Ring-CNOT 是一个置换操作（permutation），不能原地修改——输出位置 i 的值来自输入位置 f(i)，如果原地修改会破坏还未读取的旧值。因此必须用双 Buffer：读旧 buffer，写新 buffer，然后交换指针。

**Buffer 分配**：

| Buffer 名称 | 大小（24 qubit） | 用途 |
|------------|----------------|------|
| `impl.current` | 256 MB | 前向状态（主 buffer） |
| `impl.scratch` | 256 MB | 前向状态（副 buffer，ring-CNOT 输出） |
| `impl.lambda` | 256 MB | 伴随状态（主 buffer） |
| `impl.cnot_scratch` | 256 MB | 伴随状态（副 buffer，ring-CNOT 输出） |

**前向传播中的双 Buffer**（`apply_structured_forward_layers`）：

```cpp
auto *state = impl.current.get();   // 读 buffer
auto *next  = impl.scratch.get();   // 写 buffer

for (layer = 0; layer < num_layers; layer++) {
    // Rotation：原地修改（单 qubit 操作，不影响其他位置）
    launch_structured_forward_rotation_chunk(state, ...);

    // Ring-CNOT：读旧 buffer，写新 buffer
    launch_apply_ring_cnot_layer(next, state, ...);
    std::swap(state, next);  // 指针交换，角色互换
}
```

**反向传播中的双 Buffer**（`run_structured_backward_sweep`）：

```cpp
auto *current      = impl.current.get();      // current 读 buffer
auto *next_current = impl.scratch.get();      // current 写 buffer
auto *lambda       = impl.lambda.get();       // lambda 读 buffer
auto *next_lambda  = impl.cnot_scratch.get(); // lambda 写 buffer

for (layer = num_layers-1; layer >= 0; layer--) {
    launch_apply_ring_cnot_layer(next_current, current, ..., true);  // 逆向
    launch_apply_ring_cnot_layer(next_lambda,  lambda,  ..., true);
    std::swap(current, next_current);
    std::swap(lambda,  next_lambda);
    run_structured_backward_rotation_layer(...);  // 梯度计算
}
```

---

### 2. 双 Stream 并行 Ring-CNOT——已实现的流水线优化

**位置**：`cpp/ising_cuda_backend.cu`（Impl 结构体），`cpp/ising_cuda_structured_adjoint_modes.inc`（`run_structured_backward_sweep_pipelined`）

**核心 insight**：同一层的两个 ring-CNOT（`current` 和 `lambda`）输入/输出完全独立，没有任何数据依赖，可以放到两个独立的 CUDA stream 上并行执行。

**执行时序对比**：

```
串行（原来）：
  |--ring-CNOT(current)--|--ring-CNOT(lambda)--|--rotation--|

并行（流水线）：
  stream_0: |--ring-CNOT(current)--|
  stream_1: |--ring-CNOT(lambda)--|   ← 与 stream_0 同时执行
                                   ↓ cudaEvent 同步
  default:                         |--rotation--|
```

**实现细节**：

在 `Impl` 结构体中新增：
```cpp
cudaStream_t pipeline_stream_current{nullptr};  // 处理 current 的 stream
cudaStream_t pipeline_stream_lambda{nullptr};   // 处理 lambda 的 stream
cudaEvent_t pipeline_event_current{nullptr};    // current stream 完成事件
cudaEvent_t pipeline_event_lambda{nullptr};     // lambda stream 完成事件
bool pipeline_streams_initialized{false};
```

初始化（构造函数中，`structured_adjoint` 策略时）：
```cpp
cudaStreamCreateWithFlags(&pipeline_stream_current, cudaStreamNonBlocking);
cudaStreamCreateWithFlags(&pipeline_stream_lambda,  cudaStreamNonBlocking);
cudaEventCreateWithFlags(&pipeline_event_current, cudaEventDisableTiming);
cudaEventCreateWithFlags(&pipeline_event_lambda,  cudaEventDisableTiming);
```

每一层的执行逻辑（`run_structured_backward_sweep_pipelined`）：
```cpp
// ① 两个 ring-CNOT 在不同 stream 上并行
launch_apply_ring_cnot_layer(next_current, current, ..., stream_c);
launch_apply_ring_cnot_layer(next_lambda,  lambda,  ..., stream_l);

// ② 记录完成事件
cudaEventRecord(event_c, stream_c);
cudaEventRecord(event_l, stream_l);

// ③ 默认 stream 等待两个 ring-CNOT 都完成（GPU 端同步，无 CPU 阻塞）
cudaStreamWaitEvent(0, event_c, 0);
cudaStreamWaitEvent(0, event_l, 0);

// ④ 交换指针
swap(current, next_current);
swap(lambda,  next_lambda);

// ⑤ rotation 在默认 stream 上执行
run_structured_backward_rotation_layer(...);
```

调用入口自动选择：
```cpp
if (impl.pipeline_streams_initialized) {
    run_structured_backward_sweep_pipelined(impl, params);
} else {
    run_structured_backward_sweep(impl, params);
}
```

**理论加速倍数**：

设 `T_cnot` 为单次 ring-CNOT 时间，`T_rot` 为 rotation 时间：

```
加速比 = (2 × T_cnot + T_rot) / (T_cnot + T_rot)
       = 1 + T_cnot / (T_cnot + T_rot)
```

| 规模 | ring-CNOT 占比（估算） | backward 理论加速比 | 整体 step 加速比 |
|------|----------------------|-------------------|----------------|
| 24x1 | ~50% | ~1.33x | ~1.2x |
| 20x2 | ~30% | ~1.23x | ~1.15x |
| 12x8 | ~15% | ~1.13x | ~1.08x |

**注意**：两个 ring-CNOT 并行时 DRAM 带宽需求约 860 GB/s，接近 RTX 6000 Ada 峰值 960 GB/s，实际收益需要实测验证。

---

### 3. Fused Ring-CNOT + Rotation Chunk（已有的融合优化）

**位置**：`cpp/ising_cuda_structured_adjoint_modes.inc`，`try_launch_fused_inverse_cnot_first_backward_chunk`

**原理**：对于大规模（num_qubits >= 20）且 chunk_width = 4 的情况，把 ring-CNOT 和第一个 rotation chunk 融合成一个 kernel，减少 global memory 的读写次数。

**注意**：流水线版本（`run_structured_backward_sweep_pipelined`）暂时跳过了这个融合路径，以保持 stream 依赖关系的简单性。

---

## 未来可做的加速方向

### 方向一：Rotation 和下一层 Ring-CNOT 的跨层 Overlap（收益大，难度高）

**原理**：Rotation 是 compute-bound（SM 84%），ring-CNOT 是 memory-bound（DRAM 89.7%），两者使用不同硬件资源，理论上可以 overlap。

**当前障碍**：Layer L-1 的 ring-CNOT 需要 Layer L rotation 的输出，存在数据依赖。

**解决方案**：把 rotation 拆成两个 kernel：
1. `kernel_gradient_only`：只读 current 和 lambda，计算并写梯度（不修改状态）
2. `kernel_state_update`：修改 current 和 lambda（逆向旋转）

梯度计算完成后，状态更新和下一层的 ring-CNOT 可以 overlap：

```
Layer L：
  stream_0: ring-CNOT(current_L)
  stream_1: ring-CNOT(lambda_L)
            ↓ event 同步
  default:  gradient_only(current_{L-1}, lambda_{L-1})  ← 只读，不修改
  stream_0: ring-CNOT(current_{L-1})  ← 可以提前开始！
  stream_1: ring-CNOT(lambda_{L-1})   ← 可以提前开始！
  default:  state_update(current_{L-1}, lambda_{L-1})
```

**预期收益**：backward 加速约 1.5-2x（在已有双 stream 基础上进一步提升）。

---

### 方向二：CUDA Graph for Forward（收益中，难度低）

**原理**：前向传播的 kernel 序列固定（每层：rotation → ring-CNOT），只有参数值变化。CUDA Graph 可以录制这个序列，后续 replay 时消除 CPU launch overhead。

**代码里已有 benchmark 接口**：
```cpp
auto RingIsingCudaBackend::benchmark_structured_forward_graph(
    const double *params, std::size_t num_params, std::size_t repeats)
    -> CudaGraphBenchmarkResult;
```

**待实现**：在实际训练中使用 CUDA Graph（第一次录制，后续 replay + 参数更新）。

**预期收益**：forward 加速约 1.1-1.2x（减少 CPU launch overhead）。

---

### 方向三：多训练 Step 之间的 CPU-GPU 异步流水线（收益大，难度高）

**原理**：当前每个 step 是 CPU 和 GPU 交替执行，两者互相等待。

```
当前：
  CPU: 准备参数 → GPU: forward+backward → CPU: 读梯度+更新参数 → GPU: ...

优化后：
  GPU: step t forward+backward
  CPU:                          读梯度+更新参数（step t）
  GPU:                          step t+1 forward+backward（用新参数）
```

**实现要点**：
- 使用 pinned memory（`cudaMallocHost`）存储参数和梯度，支持异步传输
- 用 `cudaMemcpyAsync` 异步传输参数
- 用 CUDA event 确保 step t 的梯度读取完成后才开始 step t+1

**预期收益**：整体训练速度提升约 1.2-1.5x。

---

### 方向四：Ring-CNOT 的 Shared Memory 优化（收益中，难度中）

**原理**：对于 tile-local 的 ring-CNOT（wire < log2(THREADS) = 8），所有需要的数据都在同一个 thread block 的 shared memory 范围内，可以用 shared memory 做 block-level 的 transpose，避免 global memory 的随机访问模式。

**预期收益**：tile-local ring-CNOT 加速约 1.2-1.5x（主要对小 qubit 规模有效）。

---

### 方向五：Batch 梯度计算（多初始态并行）（收益大，适用场景特定）

**原理**：把 `current` 和 `lambda` 扩展为 `batch × state_size` 的矩阵，让多个初始态（或多个训练样本）并行计算梯度。

**适用场景**：QML 中的 data-driven 训练，每个训练样本对应一个不同的初始量子态。

**预期收益**：线性扩展（batch=4 时吞吐量约 4x，但单样本延迟不变）。

---

## 优先级总结

| 方向 | 预期收益 | 实现难度 | 推荐优先级 |
|------|---------|---------|-----------|
| ✅ 双 stream 并行 ring-CNOT | backward 1.3x | 低 | **已完成** |
| ✅ 双 Buffer Ping-Pong | 正确性基础 | 低 | **已完成** |
| CUDA Graph for forward | forward 1.1-1.2x | 低 | ⭐⭐⭐ 优先 |
| Rotation/ring-CNOT 跨层 overlap | backward 1.5-2x | 高 | ⭐⭐⭐ 优先 |
| 多 step CPU-GPU 异步流水线 | 整体 1.2-1.5x | 高 | ⭐⭐ 中期 |
| Ring-CNOT shared memory 优化 | ring-CNOT 1.2-1.5x | 中 | ⭐⭐ 中期 |
| Batch 梯度计算 | 线性扩展 | 中 | ⭐ 按需 |
