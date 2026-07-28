# 双 Stream 并行 Ring-CNOT Backward 算法设计文档

> 针对 `structured_adjoint` 策略的 Backward Sweep 流水线优化设计，详细记录算法原理、数学正确性、实现细节与性能分析。

---

## 1. 背景与问题分析

### 1.1 Backward Sweep 的时间构成

在 `structured_adjoint` 的 Backward Sweep 中，每一层（layer）的执行结构如下：

```
Layer L（串行，原始版本）：
  ① ring-CNOT(current → next_current)   ~1190 us  [memory-bound, DRAM 89.7%]
  ② ring-CNOT(lambda  → next_lambda)    ~1190 us  [memory-bound, DRAM 89.7%]
  ③ rotation + 梯度(current, lambda)     ~3229 us  [compute-bound, SM 84%]

  总计：~5609 us / layer
```

对于 24 qubit × 1 layer（24x1）规模：
- Forward：~6.4 ms
- Backward：~22.4 ms（4 层 × ~5.6 ms）
- 整个训练 step：~28.8 ms

**瓶颈**：步骤 ① 和 ② 是两次完全独立的 Ring-CNOT 操作，串行执行浪费了约 1190 us / layer。

### 1.2 为什么 ① 和 ② 可以并行？

Ring-CNOT 操作的输入输出关系：
- 操作 ①：读 `current`，写 `next_current`
- 操作 ②：读 `lambda`，写 `next_lambda`

这两个操作的输入（`current` 和 `lambda`）是**完全独立的不同内存区域**，输出（`next_current` 和 `next_lambda`）也是**完全独立的不同内存区域**，不存在任何读写冲突（RAW / WAR / WAW hazard）。

因此，可以将它们放到两个独立的 CUDA Stream 上**同时执行**。

---

## 2. 核心概念解释

### 2.1 什么是 Ring-CNOT Layer？

Ring-CNOT Layer 是一种特殊的量子门排列：
- qubit 0 控制 qubit 1
- qubit 1 控制 qubit 2
- ...
- qubit N-2 控制 qubit N-1
- qubit N-1 控制 qubit 0（形成环）

在态向量（statevector）上，Ring-CNOT 是一个**置换操作（permutation）**：它把态向量的某些分量搬到其他位置。

**为什么不能原地修改？**

置换操作的输出位置 `i` 的值来自输入位置 `f(i)`。如果原地修改，比如先把位置 0 的值写到位置 1，再处理位置 1 时，原来位置 1 的值已经被覆盖，结果错误。

因此必须用**双 Buffer（Ping-Pong Buffer）**：读旧 buffer，写新 buffer，然后交换指针。

### 2.2 什么是 current 和 lambda？

在 Adjoint 微分方法中：
- **`current`（前向态）**：量子电路在某一层之后的量子态向量，大小为 `2^N` 个复数
- **`lambda`（伴随态）**：从哈密顿量出发，反向传播的辅助向量，大小同为 `2^N` 个复数

Backward Sweep 需要同时维护这两个向量，逐层"撤销"电路操作，并在每层计算参数梯度。

### 2.3 什么是 CUDA Stream？

CUDA Stream 是 GPU 上的一条**有序执行队列**。同一个 Stream 内的操作按顺序执行，不同 Stream 之间的操作可以**并行执行**（只要硬件资源允许）。

**CUDA Event** 是 Stream 之间的同步机制：
- `cudaEventRecord(event, stream)`：在 stream 中插入一个"完成标记"
- `cudaStreamWaitEvent(stream2, event)`：让 stream2 等待该 event 被触发后才继续执行

---

## 3. 算法设计

### 3.1 内存布局

| Buffer 名称 | 大小（24 qubit） | 用途 |
|------------|----------------|------|
| `impl.current` | 256 MB | 前向态主 Buffer（读） |
| `impl.scratch` | 256 MB | 前向态副 Buffer（Ring-CNOT 写入目标） |
| `impl.lambda` | 256 MB | 伴随态主 Buffer（读） |
| `impl.cnot_scratch` | 256 MB | 伴随态副 Buffer（Ring-CNOT 写入目标） |

**总显存占用**：4 × 256 MB = 1 GiB（24 qubit 规模）

### 3.2 Stream 和 Event 资源

```cpp
// 在 Impl 结构体中新增：
cudaStream_t pipeline_stream_current;  // 专门处理 current 的 Ring-CNOT
cudaStream_t pipeline_stream_lambda;   // 专门处理 lambda 的 Ring-CNOT
cudaEvent_t  pipeline_event_current;   // current Ring-CNOT 完成事件
cudaEvent_t  pipeline_event_lambda;    // lambda Ring-CNOT 完成事件
bool         pipeline_streams_initialized;
```

**初始化**（在构造函数中，仅 `structured_adjoint` 策略时）：

```cpp
cudaStreamCreateWithFlags(&pipeline_stream_current, cudaStreamNonBlocking);
cudaStreamCreateWithFlags(&pipeline_stream_lambda,  cudaStreamNonBlocking);
// cudaEventDisableTiming：禁用计时，减少 event 的 overhead
cudaEventCreateWithFlags(&pipeline_event_current, cudaEventDisableTiming);
cudaEventCreateWithFlags(&pipeline_event_lambda,  cudaEventDisableTiming);
pipeline_streams_initialized = true;
```

**销毁**（在析构函数中）：

```cpp
if (pipeline_streams_initialized) {
    cudaStreamDestroy(pipeline_stream_current);
    cudaStreamDestroy(pipeline_stream_lambda);
    cudaEventDestroy(pipeline_event_current);
    cudaEventDestroy(pipeline_event_lambda);
}
```

### 3.3 每一层的执行逻辑（核心算法）

```
输入：
  current      ← 当前层的前向态（读 buffer）
  next_current ← 前向态写 buffer
  lambda       ← 当前层的伴随态（读 buffer）
  next_lambda  ← 伴随态写 buffer
  stream_c     ← pipeline_stream_current
  stream_l     ← pipeline_stream_lambda
  event_c      ← pipeline_event_current
  event_l      ← pipeline_event_lambda

步骤：
  ① 在 stream_c 上发射：逆向 Ring-CNOT(current → next_current)
  ② 在 stream_l 上发射：逆向 Ring-CNOT(lambda  → next_lambda)
     （① 和 ② 在 GPU 上同时执行，无需等待）

  ③ 在 stream_c 上记录事件：cudaEventRecord(event_c, stream_c)
  ④ 在 stream_l 上记录事件：cudaEventRecord(event_l, stream_l)

  ⑤ 默认 stream 等待 event_c：cudaStreamWaitEvent(0, event_c, 0)
  ⑥ 默认 stream 等待 event_l：cudaStreamWaitEvent(0, event_l, 0)
     （⑤⑥ 是 GPU 端等待，不阻塞 CPU）

  ⑦ 交换指针：swap(current, next_current); swap(lambda, next_lambda)

  ⑧ 在默认 stream 上执行：逆向 Rotation + 梯度计算(current, lambda)
     （此时 current 和 lambda 已经是 Ring-CNOT 之后的状态）
```

### 3.4 完整 Backward Sweep 伪代码

```cpp
void run_structured_backward_sweep_pipelined(Impl &impl, const double *params) {
    auto *current      = impl.current.get();
    auto *next_current = impl.scratch.get();
    auto *lambda       = impl.lambda.get();
    auto *next_lambda  = impl.cnot_scratch.get();

    const auto stream_c = impl.pipeline_stream_current;
    const auto stream_l = impl.pipeline_stream_lambda;
    const auto event_c  = impl.pipeline_event_current;
    const auto event_l  = impl.pipeline_event_lambda;

    // 从最后一层逆序遍历到第 0 层
    for (int layer = impl.num_layers - 1; layer >= 0; layer--) {

        // ─── 步骤 1：两个 Ring-CNOT 在不同 stream 上并行发射 ───────────────
        launch_apply_ring_cnot_layer(
            next_current, current,
            impl.state_size, impl.num_qubits,
            /*inverse=*/true, stream_c);          // stream_c 上执行

        launch_apply_ring_cnot_layer(
            next_lambda, lambda,
            impl.state_size, impl.num_qubits,
            /*inverse=*/true, stream_l);          // stream_l 上执行

        // ─── 步骤 2：记录两个 stream 的完成事件 ──────────────────────────
        cudaEventRecord(event_c, stream_c);
        cudaEventRecord(event_l, stream_l);

        // ─── 步骤 3：默认 stream 等待两个 Ring-CNOT 都完成 ───────────────
        // 注意：这是 GPU 端等待，CPU 不阻塞，可以继续准备下一层的参数
        cudaStreamWaitEvent(0, event_c, 0);
        cudaStreamWaitEvent(0, event_l, 0);

        // ─── 步骤 4：交换指针（O(1) 操作，无数据拷贝）───────────────────
        std::swap(current, next_current);
        std::swap(lambda,  next_lambda);

        // ─── 步骤 5：在默认 stream 上执行 Rotation + 梯度计算 ────────────
        // 此时 current 和 lambda 已经是逆向 Ring-CNOT 之后的状态
        run_structured_backward_rotation_layer(
            impl, current, lambda, params, layer, impl.num_qubits);
    }

    // ─── 最终同步：确保所有 pipeline stream 工作完成 ──────────────────────
    cudaStreamSynchronize(stream_c);
    cudaStreamSynchronize(stream_l);
    cudaDeviceSynchronize();

    // ─── 如果指针发生了奇数次交换，需要拷贝回原始 buffer ─────────────────
    if (current != impl.current.get()) {
        copy_device_buffer(impl.current.get(), current, impl.state_size);
    }
    if (lambda != impl.lambda.get()) {
        copy_device_buffer(impl.lambda.get(), lambda, impl.state_size);
    }
}
```

### 3.5 调用入口（自动选择串行/流水线版本）

```cpp
void run_energy_and_grad_structured_adjoint(Impl &impl, ...) {
    // ... forward, hamiltonian ...

    if (impl.pipeline_streams_initialized) {
        run_structured_backward_sweep_pipelined(impl, params);  // 流水线版本
    } else {
        run_structured_backward_sweep(impl, params);            // 串行版本（fallback）
    }
}
```

---

## 4. 时序图

### 4.1 串行版本（原始）

```
时间轴 →

Layer L-1:
  default: |--Ring-CNOT(current)--|--Ring-CNOT(lambda)--|--Rotation+Grad--|
           ←    ~1190 us         →←    ~1190 us        →←    ~3229 us    →
           ←────────────────── ~5609 us ────────────────────────────────→

Layer L-2:
  default:                                                                 |--Ring-CNOT(current)--|...
```

### 4.2 流水线版本（双 Stream 并行）

```
时间轴 →

Layer L-1:
  stream_c: |--Ring-CNOT(current)--|
  stream_l: |--Ring-CNOT(lambda)--|    ← 与 stream_c 同时执行
            ←    ~1190 us         →
                                   ↓ event_c + event_l 同步（GPU 端）
  default:                         |--Rotation+Grad--|
                                   ←    ~3229 us    →
  ←──────────────── ~4419 us ──────────────────────→

Layer L-2:
  stream_c:                                          |--Ring-CNOT(current)--|
  stream_l:                                          |--Ring-CNOT(lambda)--|
  ...
```

**节省时间**：每层节省约 1190 us（一次 Ring-CNOT 的时间）

---

## 5. 数学正确性证明

### 5.1 数据依赖分析

设第 `L` 层的操作为：

```
Ring-CNOT_inv(current_L) → current_{L-1}
Ring-CNOT_inv(lambda_L)  → lambda_{L-1}
Rotation_inv(current_{L-1}, lambda_{L-1}) → grad_L, current_{L-2}', lambda_{L-2}'
```

**依赖关系**：
- `Ring-CNOT_inv(current_L)` 只读 `current_L`，只写 `next_current`
- `Ring-CNOT_inv(lambda_L)` 只读 `lambda_L`，只写 `next_lambda`
- 两者之间：**无任何共享内存访问** → 可以并行

**Rotation 的依赖**：
- 读 `current_{L-1}`（即 `next_current` 交换后的指针）
- 读 `lambda_{L-1}`（即 `next_lambda` 交换后的指针）
- 必须在两个 Ring-CNOT **都完成后**才能开始 → 通过 `cudaStreamWaitEvent` 保证

### 5.2 指针交换的正确性

```
初始状态：
  current      → Buffer A（存放 current_L）
  next_current → Buffer B（空）
  lambda       → Buffer C（存放 lambda_L）
  next_lambda  → Buffer D（空）

Ring-CNOT 执行后：
  Buffer B 存放 current_{L-1}
  Buffer D 存放 lambda_{L-1}

swap(current, next_current) 后：
  current      → Buffer B（存放 current_{L-1}）✓
  next_current → Buffer A（可用于下一层写入）

swap(lambda, next_lambda) 后：
  lambda       → Buffer D（存放 lambda_{L-1}）✓
  next_lambda  → Buffer C（可用于下一层写入）
```

每次 swap 只交换指针（8 字节），不移动数据，O(1) 操作。

### 5.3 奇偶层数的处理

经过 `num_layers` 次 swap 后：
- 若 `num_layers` 为偶数：`current` 指针回到 `impl.current`，无需拷贝
- 若 `num_layers` 为奇数：`current` 指针指向 `impl.scratch`，需要拷贝回 `impl.current`

代码末尾的检查处理了这种情况：

```cpp
if (current != impl.current.get()) {
    copy_device_buffer(impl.current.get(), current, impl.state_size);
}
```

---

## 6. 理论性能分析

### 6.1 加速比公式

设：
- `T_cnot`：单次 Ring-CNOT 的执行时间
- `T_rot`：单次 Rotation + 梯度计算的执行时间
- `T_sync`：event 同步的 overhead（通常 < 1 us，可忽略）

**串行版本**每层时间：
```
T_serial = 2 × T_cnot + T_rot
```

**流水线版本**每层时间（两个 Ring-CNOT 完全并行）：
```
T_pipeline = max(T_cnot, T_cnot) + T_rot = T_cnot + T_rot
```

**加速比**：
```
Speedup = T_serial / T_pipeline
        = (2 × T_cnot + T_rot) / (T_cnot + T_rot)
        = 1 + T_cnot / (T_cnot + T_rot)
```

### 6.2 不同规模的预期收益

| 规模 | T_cnot (us) | T_rot (us) | Ring-CNOT 占比 | Backward 加速比 | 整体 step 加速比 |
|------|------------|-----------|--------------|----------------|----------------|
| 12x2 | ~150 | ~800 | ~27% | ~1.16x | ~1.10x |
| 16x2 | ~350 | ~1500 | ~30% | ~1.19x | ~1.12x |
| 20x2 | ~700 | ~2500 | ~37% | ~1.22x | ~1.15x |
| 24x1 | ~1190 | ~3229 | ~50% | ~1.27x | ~1.20x |
| 24x2 | ~1190 | ~3229 | ~50% | ~1.27x | ~1.22x |

> 注：T_cnot 和 T_rot 随 qubit 数指数增长，但比值相对稳定。

### 6.3 带宽瓶颈分析

Ring-CNOT 是 memory-bound 操作（DRAM 利用率 89.7%）。两个 Ring-CNOT 并行时：

```
并行 DRAM 带宽需求 = 2 × (读 + 写) × state_size × sizeof(Complex)
                   = 2 × 2 × 2^24 × 16 bytes
                   ≈ 2 × 536 MB = 1072 MB
                   ≈ 860 GB/s（假设 ~1190 us 完成）
```

RTX 6000 Ada 峰值 DRAM 带宽：**960 GB/s**

**结论**：两个 Ring-CNOT 并行时，DRAM 带宽需求（~860 GB/s）接近峰值（960 GB/s），实际执行时间可能略长于单个 Ring-CNOT，但仍显著短于串行执行两次（~2380 us）。

**实测建议**：在 24 qubit 规模下，实际加速比可能在 1.15x-1.27x 之间，需要实测验证。

---

## 7. 与现有代码的对应关系

### 7.1 代码文件位置

| 组件 | 文件 | 函数/结构体 |
|------|------|-----------|
| Stream/Event 声明 | `cpp/ising_cuda_backend.cu` | `RingIsingCudaBackend::Impl` |
| Stream/Event 初始化 | `cpp/ising_cuda_backend.cu` | 构造函数 |
| 流水线 Backward | `cpp/ising_cuda_structured_adjoint_modes.inc` | `run_structured_backward_sweep_pipelined` |
| 串行 Backward（fallback） | `cpp/ising_cuda_structured_adjoint_modes.inc` | `run_structured_backward_sweep` |
| 调用入口 | `cpp/ising_cuda_structured_adjoint_modes.inc` | `run_energy_and_grad_structured_adjoint` |

### 7.2 与串行版本的差异对比

| 方面 | 串行版本 | 流水线版本 |
|------|---------|-----------|
| Ring-CNOT 执行方式 | 顺序执行，都在 default stream | 并行执行，分别在 stream_c 和 stream_l |
| 同步方式 | 隐式（同一 stream 顺序保证） | 显式（cudaEventRecord + cudaStreamWaitEvent） |
| Fused CNOT+Rotation | 支持（`try_launch_fused_inverse_cnot_first_backward_chunk`） | **暂时跳过**（保持 stream 依赖简单） |
| 最终同步 | 无需额外同步 | 需要 `cudaStreamSynchronize` × 2 + `cudaDeviceSynchronize` |

### 7.3 为什么流水线版本跳过 Fused CNOT+Rotation？

Fused CNOT+Rotation 把逆向 Ring-CNOT 和第一个 Rotation chunk 合并为一个 kernel，减少一次 global memory 读写。但在流水线版本中：

- Ring-CNOT 在 `stream_c` / `stream_l` 上执行
- Rotation 在 `default stream` 上执行
- 如果融合，需要把 Rotation 的一部分也放到 `stream_c` / `stream_l` 上，但 Rotation 需要同时读 `current` 和 `lambda`，而这两个 buffer 分别在不同 stream 上更新

这会导致 stream 依赖关系变得复杂，容易引入正确性问题。**当前选择保持简单，未来可以在验证正确性后重新引入融合优化。**

---

## 8. 已知限制与注意事项

### 8.1 带宽饱和风险

如第 6.3 节所述，24 qubit 规模下两个 Ring-CNOT 并行的 DRAM 带宽需求接近 GPU 峰值。如果实测发现两个 Ring-CNOT 的并行执行时间接近串行执行时间（即带宽完全饱和），则流水线优化的实际收益会低于理论值。

**建议**：在 benchmark 中同时测量 DRAM 利用率，如果并行时 DRAM 利用率 > 95%，说明带宽已饱和。

### 8.2 小规模（qubit <= 12）的收益有限

对于小规模问题，Ring-CNOT 时间很短（< 200 us），而 Rotation 时间也短，stream 创建和 event 同步的 overhead（~5-10 us）占比相对较大，实际收益可能不显著。

**建议**：对于 qubit <= 12 的情况，可以考虑回退到串行版本（通过 `pipeline_streams_initialized` 标志控制）。

### 8.3 多层（layers >= 4）的收益更显著

每层都能节省约 `T_cnot` 的时间，层数越多，总节省越大。对于 `layers >= 4` 的场景，流水线优化的整体收益更为显著。

### 8.4 与 CUDA Graph 的兼容性

当前流水线版本使用动态 stream 和 event，与 CUDA Graph 的录制机制存在兼容性问题（CUDA Graph 需要在录制时确定所有 stream 依赖）。如果未来引入 CUDA Graph for Backward，需要重新设计 stream 依赖图。

---

## 9. 未来改进方向

### 9.1 跨层 Overlap（更大收益）

在当前双 stream 基础上，进一步把 Rotation 拆成两个 kernel：
1. `kernel_gradient_only`：只读 `current` 和 `lambda`，计算梯度（不修改状态）
2. `kernel_state_update`：对 `current` 和 `lambda` 施加逆向旋转

这样，Layer L 的 `gradient_only` 完成后，Layer L-1 的 Ring-CNOT 就可以提前发射，实现跨层 overlap：

```
Layer L：
  stream_c: |--Ring-CNOT(current_L)--|
  stream_l: |--Ring-CNOT(lambda_L)--|
                                     ↓ event 同步
  default:  |--gradient_only(L)--|
  stream_c:                      |--Ring-CNOT(current_{L-1})--|  ← 提前开始！
  stream_l:                      |--Ring-CNOT(lambda_{L-1})--|   ← 提前开始！
  default:  |--state_update(L)--|
                                 ↓ event 同步
  default:  |--gradient_only(L-1)--|
  ...
```

**预期额外收益**：在已有双 stream 基础上，backward 再加速约 1.3-1.5x。

### 9.2 重新引入 Fused CNOT+Rotation

在跨层 overlap 实现后，可以重新考虑把 Ring-CNOT 和 `gradient_only` 的第一个 chunk 融合，进一步减少 global memory 读写。

---

## 10. 总结

| 优化层次 | 描述 | 状态 | 预期收益 |
|---------|------|------|---------|
| 双 Buffer Ping-Pong | 保证 Ring-CNOT 正确性 | ✅ 已实现 | 正确性基础 |
| 双 Stream 并行 Ring-CNOT | current 和 lambda 的 Ring-CNOT 同时执行 | ✅ 已实现 | backward ~1.27x |
| Fused CNOT+Rotation（串行版） | Ring-CNOT 和第一个 Rotation chunk 融合 | ✅ 已实现（仅串行版） | 减少显存读写 |
| 跨层 Overlap（Rotation 拆分） | gradient_only 和下一层 Ring-CNOT overlap | ⬜ 未实现 | backward ~1.5-2x |
| 流水线版 Fused CNOT+Rotation | 在流水线版本中重新引入融合 | ⬜ 未实现 | 减少显存读写 |
