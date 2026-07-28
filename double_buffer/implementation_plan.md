# Kernel 内部双 Buffer（Double Buffering）完整实现计划

## 1. 背景与目标

### 当前问题

在 `ising_cuda_rotation_kernels.cu` 和 `ising_cuda_statevector_grad_kernels.cu` 中，rotation chunk kernel 的每个 tile 处理流程是**串行**的：

```
tile_i:  [DRAM load] → [shared mem / register compute] → [DRAM store]
tile_i+1: [DRAM load] → [compute] → [DRAM store]
...
```

DRAM 访问（load/store）和 SM 计算单元是**交替工作**的，存在空闲等待。

### 目标

在 kernel 内部实现**双 buffer 流水线**，使 DRAM 带宽和 SM 计算单元**同时工作**，理论上可以隐藏 DRAM 延迟：

```
stage 0:  prefetch tile_0 → buf[0]
stage 1:  compute buf[0]  |  prefetch tile_1 → buf[1]
stage 2:  store tile_0    |  compute buf[1]  |  prefetch tile_2 → buf[0]
...
```

### 开关

通过 CLI 参数 `--double-buffer` 控制，默认关闭（`false`），方便对比测试。

---

## 2. Kernel 类型说明

### Register Kernel（`apply_ryrz_rotation_chunk_register_kernel<W>`）

- **并行粒度**：1 thread = 1 tile（`2^W` 个振幅）
- **数据存放**：寄存器数组 `values[tile_dim]`
- **线程同步**：不需要 `__syncthreads()`，tile 之间完全独立
- **支持 W**：W ≤ 4（W=5 时寄存器溢出）
- **调用时机**：`num_qubits ≥ 23`（前向），或 `num_qubits ≥ 16` 且 `structured_rotation_chunk_width=8` 的高位 qubit（反向）

### Cooperative Kernel（`apply_ryrz_rotation_chunk_cooperative_kernel<W>`）

- **并行粒度**：1 block = 1 tile（`2^W` 个振幅由 `tile_dim` 个 thread 协作处理）
- **数据存放**：Shared memory `tile[THREADS]`
- **线程同步**：需要 `__syncthreads()`（wire ≥ 5 时）；wire < 5 用 warp shuffle
- **支持 W**：W = 2~8
- **调用时机**：`num_qubits ≤ 15`（前向，W=8）

### CooperativePair512 Kernel（`apply_ryrz_rotation_chunk_cooperative_pair512_kernel`）

- **并行粒度**：1 block = 2 个 W=8 tile（512 个振幅）
- **数据存放**：Shared memory `tile[THREADS * 2]`
- **线程同步**：需要 `__syncthreads()`
- **支持 W**：固定 W=8
- **调用时机**：`num_qubits` 在 16~22 之间（前向，前 16 个 qubit）

### 反向 Kernel（`inverse_walk_ryrz_rotation_chunk_kernel<W>`）

- **并行粒度**：1 block = 1 tile（`2^W` 个振幅）
- **数据存放**：Shared memory `current_tile[THREADS]` + `lambda_tile[THREADS]`（两个状态向量）
- **线程同步**：每个 wire 处理后需要 `__syncthreads()`
- **支持 W**：W = 2~8
- **调用时机**：反向 sweep 中，`num_qubits ≤ 15` 或 `structured_rotation_chunk_width=8` 时

---

## 3. 三阶段实现方案

---

### 阶段一：前向 Register Kernel 双 Buffer（W=2/3/4）

**技术方案：寄存器级软件流水线（Register-level Software Pipelining）**

Register kernel 中每个 thread 独立处理一个 tile，tile 之间无任何共享。因此双 buffer 在**单个 thread 内部**实现，使用两组寄存器数组交替：

```
当前 tile 的数据在 values_cur[tile_dim] 中计算
下一个 tile 的数据预取到 values_next[tile_dim]
```

**伪代码**：

```cpp
template <int W>
__global__ void apply_ryrz_rotation_chunk_register_db_kernel(
    Complex *state, std::size_t size, std::size_t chunk_start,
    RotationChunkCoeffs coeffs) {

    constexpr auto tile_dim = std::size_t{1} << W;
    const auto tile_id_base = static_cast<std::size_t>(
        blockIdx.x * blockDim.x + threadIdx.x);
    const auto num_tiles = size / tile_dim;
    const auto stride = static_cast<std::size_t>(gridDim.x * blockDim.x);

    // 每个 thread 处理多个 tile（grid-stride loop）
    // 双 buffer：cur 和 nxt 交替
    Complex values_cur[tile_dim];
    Complex values_nxt[tile_dim];

    auto tile_id = tile_id_base;
    if (tile_id >= num_tiles) return;

    // Prologue：预取第一个 tile 到 values_cur
    const auto low_mask = (std::size_t{1} << chunk_start) - 1;
    auto load_tile = [&](std::size_t tid, Complex *buf) {
        const auto base = (tid & low_mask) | ((tid & ~low_mask) << W);
        #pragma unroll
        for (int i = 0; i < static_cast<int>(tile_dim); i++) {
            buf[i] = state[base | (static_cast<std::size_t>(i) << chunk_start)];
        }
    };
    auto store_tile = [&](std::size_t tid, const Complex *buf) {
        const auto base = (tid & low_mask) | ((tid & ~low_mask) << W);
        #pragma unroll
        for (int i = 0; i < static_cast<int>(tile_dim); i++) {
            state[base | (static_cast<std::size_t>(i) << chunk_start)] = buf[i];
        }
    };
    auto compute_tile = [&](Complex *buf) {
        #pragma unroll
        for (int local_wire = 0; local_wire < W; local_wire++) {
            // ... RyRz 变换 ...
        }
    };

    load_tile(tile_id, values_cur);  // 预取第一个 tile

    while (tile_id + stride < num_tiles) {
        // 预取下一个 tile 到 values_nxt（与计算当前 tile 重叠）
        load_tile(tile_id + stride, values_nxt);
        // 计算当前 tile
        compute_tile(values_cur);
        // 写回当前 tile
        store_tile(tile_id, values_cur);
        // 切换 buffer
        tile_id += stride;
        // swap cur/nxt（编译器会优化为寄存器重命名）
        #pragma unroll
        for (int i = 0; i < static_cast<int>(tile_dim); i++) {
            values_cur[i] = values_nxt[i];
        }
    }

    // Epilogue：处理最后一个 tile
    compute_tile(values_cur);
    store_tile(tile_id, values_cur);
}
```

**关键点**：
- 不需要 `cuda::pipeline`，依赖编译器/硬件的**指令级并行（ILP）**：load 指令发出后不阻塞，SM 可以在等待 DRAM 响应时执行计算指令
- 使用 **grid-stride loop**（而非原来的 1 thread = 1 tile），使每个 thread 处理多个 tile，从而有机会重叠 load 和 compute
- 也可以使用 `cuda::memcpy_async` + `cuda::pipeline<thread_scope_thread>` 显式异步预取（CUDA 11.1+）

**新增枚举值**：`RotationChunkKernelPreference::RegisterDoubleBuffer`

---

### 阶段二：前向 Cooperative Kernel 双 Buffer（W=5~8）

**技术方案：Shared Memory 双 Buffer + `cuda::pipeline<thread_scope_block>`**

Cooperative kernel 中一个 block 处理一个 tile，数据在 shared memory 中。双 buffer 需要在 shared memory 中分配两份 tile 空间，交替使用。

**核心挑战**：Cooperative kernel 在每个 wire 处理后需要 `__syncthreads()`，这意味着：
- 不能在计算 wire_k 的同时预取下一个 tile（因为所有 thread 都在参与 wire_k 的计算）
- 双 buffer 的收益窗口是：**当前 tile 的最后一个 wire 计算完成后，写回 DRAM 的同时，预取下一个 tile**

**伪代码**：

```cpp
template <int W>
__global__ void apply_ryrz_rotation_chunk_cooperative_db_kernel(
    Complex *state, std::size_t size, std::size_t chunk_start,
    RotationChunkCoeffs coeffs) {

    // 双 buffer：两份 shared memory tile
    __shared__ Complex tile[2][THREADS];
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pipe_state;

    constexpr auto tile_dim = std::size_t{1} << W;
    constexpr auto tiles_per_block = static_cast<std::size_t>(THREADS) / tile_dim;
    const auto local_thread = static_cast<std::size_t>(threadIdx.x);
    const auto local_tile = local_thread / tile_dim;
    const auto local_index = local_thread & (tile_dim - 1);

    auto pipeline = cuda::make_pipeline(threadIdx.x == 0, &pipe_state);

    // 对每个 block 负责的 tile 组：
    // 1. 用 cuda::memcpy_async 异步预取 tile[cur_buf] 的数据
    // 2. pipeline.consumer_wait() 等待数据就绪
    // 3. 执行 W 个 wire 的计算（含 __syncthreads()）
    // 4. 写回 DRAM（同时异步预取下一个 tile 到 tile[1-cur_buf]）

    // Prologue：预取第一批 tile
    int cur_buf = 0;
    // ... 异步 load tile[cur_buf] ...
    pipeline.producer_commit();

    for (每个 tile 批次) {
        // 预取下一批 tile 到 tile[1-cur_buf]
        pipeline.producer_acquire();
        // cuda::memcpy_async(tile[1-cur_buf] + ..., state + next_addr, ...)
        pipeline.producer_commit();

        // 等待当前批次数据就绪
        pipeline.consumer_wait();

        // 计算（含 __syncthreads()）
        for (int local_wire = 0; local_wire < W; local_wire++) {
            // warp shuffle（wire < 5）或 shared memory 交换（wire >= 5）
            __syncthreads();
        }

        // 写回 DRAM
        state[addr] = tile[cur_buf][local_thread];

        pipeline.consumer_release();
        cur_buf = 1 - cur_buf;
    }
}
```

**关键点**：
- 使用 `cuda::pipeline<cuda::thread_scope_block>`（CUDA 11.0+）
- `cuda::memcpy_async` 要求目标是 shared memory，源是 global memory，且 16 字节对齐（`Complex = thrust::complex<double>` = 16 字节，天然满足）
- Shared memory 用量翻倍：`2 * tile_dim * sizeof(Complex)` per block
- `__syncthreads()` 仍然需要，但现在它同时也起到了 pipeline 同步的作用

**新增枚举值**：`RotationChunkKernelPreference::CooperativeDoubleBuffer`

---

### 阶段三：反向 Kernel 双 Buffer（`inverse_walk_ryrz_rotation_chunk_kernel<W>`）

**技术方案：Shared Memory 双 Buffer，处理双状态向量（current + lambda）**

反向 kernel 比前向更复杂，因为它同时操作两个状态向量（`current` 和 `lambda`），且每个 wire 处理后需要 `__syncthreads()`。

**核心挑战**：
1. 需要为 `current_tile` 和 `lambda_tile` 各分配双 buffer，共 4 份 shared memory
2. 反向 kernel 还需要**写回梯度**（`theta_partial` 和 `phi_partial`），这部分不涉及 DRAM 读写，不需要双 buffer
3. 每个 wire 处理后的 `__syncthreads()` 限制了流水线深度

**Shared Memory 布局**：

```
current_tile[2][THREADS]  // 双 buffer for current
lambda_tile[2][THREADS]   // 双 buffer for lambda
theta_partial[W * THREADS]
phi_partial[W * THREADS]
```

**伪代码**：

```cpp
template <int W>
__global__ void inverse_walk_ryrz_rotation_chunk_db_kernel(
    Complex *current, Complex *lambda, std::size_t size,
    std::size_t chunk_start, RotationChunkCoeffs coeffs,
    double *out_gradients) {

    __shared__ Complex current_tile[2][THREADS];
    __shared__ Complex lambda_tile[2][THREADS];
    __shared__ double theta_partial[W * THREADS];
    __shared__ double phi_partial[W * THREADS];
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pipe_state;

    auto pipeline = cuda::make_pipeline(threadIdx.x == 0, &pipe_state);

    // Prologue：异步预取第一个 tile 的 current 和 lambda
    int cur_buf = 0;
    pipeline.producer_acquire();
    // cuda::memcpy_async(current_tile[cur_buf] + ..., current + addr, ...)
    // cuda::memcpy_async(lambda_tile[cur_buf] + ..., lambda + addr, ...)
    pipeline.producer_commit();

    for (每个 tile) {
        // 预取下一个 tile
        pipeline.producer_acquire();
        // cuda::memcpy_async(current_tile[1-cur_buf] + ..., ...)
        // cuda::memcpy_async(lambda_tile[1-cur_buf] + ..., ...)
        pipeline.producer_commit();

        // 等待当前 tile 数据就绪
        pipeline.consumer_wait();

        // 反向计算（从 wire W-1 到 wire 0）
        for (int local_wire = W - 1; local_wire >= 0; local_wire--) {
            // 计算梯度 + 更新 current_tile[cur_buf] 和 lambda_tile[cur_buf]
            __syncthreads();
        }

        // 写回 DRAM
        current[addr] = current_tile[cur_buf][local_thread];
        lambda[addr] = lambda_tile[cur_buf][local_thread];

        pipeline.consumer_release();
        cur_buf = 1 - cur_buf;
    }

    // 梯度 reduction（不变）
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) { ... }
    if (threadIdx.x == 0) { atomicAdd(...); }
}
```

**关键点**：
- Shared memory 用量：`4 * THREADS * sizeof(Complex) + 2 * W * THREADS * sizeof(double)`
  - W=4, THREADS=256：4×256×16 + 2×4×256×8 = 16384 + 16384 = 32 KB（在 48 KB 限制内）
  - W=8, THREADS=256：4×256×16 + 2×8×256×8 = 16384 + 32768 = 49 KB（**超出 48 KB！**）
  - 因此 W=8 的反向 kernel 双 buffer 需要减少 THREADS 或使用动态 shared memory
- 梯度 reduction 部分（`theta_partial`/`phi_partial`）不需要双 buffer，保持不变
- 每个 wire 的 `__syncthreads()` 同时充当 pipeline 的隐式同步点

**新增枚举值**：`RotationChunkKernelPreference::CooperativeBackwardDoubleBuffer`（反向专用）

---

## 4. 修改文件清单

### 4.1 CUDA 层（C++）

#### `cpp/ising_cuda_kernel_common.cuh`

新增枚举值：
```cpp
enum class RotationChunkKernelPreference : int {
    Cooperative,
    CooperativePair512,
    Register,
    RegisterDoubleBuffer,          // 阶段一：前向 Register kernel 双 buffer
    CooperativeDoubleBuffer,       // 阶段二：前向 Cooperative kernel 双 buffer
    CooperativeBackwardDoubleBuffer // 阶段三：反向 Cooperative kernel 双 buffer
};
```

#### `cpp/ising_cuda_rotation_kernels.cu`

- 新增头文件：`#include <cuda/pipeline>`
- **阶段一**：新增 `apply_ryrz_rotation_chunk_register_db_kernel<W>`（W=2,3,4）
- **阶段二**：新增 `apply_ryrz_rotation_chunk_cooperative_db_kernel<W>`（W=5,6,7,8）
- 在 `launch_apply_ryrz_rotation_chunk_specialized<W>` 中新增对应分支

#### `cpp/ising_cuda_statevector_grad_kernels.cu`

- **阶段三**：新增 `inverse_walk_ryrz_rotation_chunk_db_kernel<W>`（W=2,3,4）
- 在 `launch_inverse_walk_ryrz_rotation_chunk` 中新增对应分支

#### `cpp/ising_cuda_backend.hpp`

```cpp
RingIsingCudaBackend(std::size_t num_qubits, std::size_t num_layers,
                     double field, const std::string &gradient_strategy,
                     std::size_t structured_rotation_chunk_width = 8,
                     bool double_buffer = false);
```

#### `cpp/ising_cuda_backend.cu`

- `Impl` 新增字段：`bool double_buffer{false};`
- 构造函数新增参数 `bool double_buffer_`
- `build_structured_forward_rotation_chunks` 调用处：根据 `double_buffer` 替换 preference

#### `cpp/ising_cuda_bindings.cpp`

- `RingIsingCudaBackend` 的 `py::init` 新增 `double_buffer` 参数

### 4.2 Python 层

#### `ring_ising/config.py`

```python
double_buffer: bool = False
```

#### `ring_ising/backends/standalone/config.py`

```python
double_buffer: bool = False
```

#### `ring_ising/backends/standalone/runtime.py`

```python
self._cuda = self._backend.RingIsingCudaBackend(
    ...,
    bool(self.config.double_buffer),
)
```

#### `ring_ising/workflows/standalone.py`

```python
double_buffer=config.double_buffer,
```

#### `ring_ising/cli/run.py`

```python
parser.add_argument(
    "--double-buffer",
    dest="double_buffer",
    action="store_true",
    default=DEFAULT_RUN_CONFIG.double_buffer,
    help=(
        "Enable kernel-internal double buffering for rotation chunk kernels. "
        "Phase 1: forward Register kernel (W<=4). "
        "Phase 2: forward Cooperative kernel (W=5~8). "
        "Phase 3: backward rotation chunk kernel."
    ),
)
```

---

## 5. 各阶段 Shared Memory 用量分析

### 阶段一（Register kernel）

无 shared memory，纯寄存器操作。

### 阶段二（Cooperative kernel 双 buffer）

| W | tile_dim | 单 buffer | 双 buffer | 增量 |
|---|---|---|---|---|
| 5 | 32 | 32×16=512 B | 2×512=1 KB | +512 B |
| 6 | 64 | 64×16=1 KB | 2 KB | +1 KB |
| 7 | 128 | 128×16=2 KB | 4 KB | +2 KB |
| 8 | 256 | 256×16=4 KB | 8 KB | +4 KB |

W=8 时双 buffer 占用 8 KB，远低于 48 KB 限制，安全。

### 阶段三（反向 kernel 双 buffer）

| W | current+lambda 双 buffer | theta+phi partial | 总计 |
|---|---|---|---|
| 2 | 4×4×16=256 B | 2×2×256×8=8 KB | ~8.3 KB |
| 3 | 4×8×16=512 B | 2×3×256×8=12 KB | ~12.5 KB |
| 4 | 4×16×16=1 KB | 2×4×256×8=16 KB | ~17 KB |
| 8 | 4×256×16=16 KB | 2×8×256×8=32 KB | **48 KB（临界！）** |

W=8 反向 kernel 双 buffer 需要特殊处理：
- 方案 A：减少 `THREADS` 到 128（tile_dim=256 时需要 2 个 block 协作，复杂度高）
- 方案 B：使用动态 shared memory（`extern __shared__`），运行时分配
- 方案 C：W=8 反向 kernel 暂不实现双 buffer，仅实现 W=2/3/4

---

## 6. 实现步骤（有序）

### 阶段一（前向 Register kernel）

1. **[ ]** 修改 `ising_cuda_kernel_common.cuh`，新增 `RegisterDoubleBuffer` 枚举值
2. **[ ]** 在 `ising_cuda_rotation_kernels.cu` 中实现 `apply_ryrz_rotation_chunk_register_db_kernel<W>`（W=2,3,4）
3. **[ ]** 在 `launch_apply_ryrz_rotation_chunk_specialized` 中新增 `RegisterDoubleBuffer` 分支
4. **[ ]** 修改 `ising_cuda_backend.hpp/cu/bindings.cpp`，新增 `double_buffer` 参数
5. **[ ]** 修改 Python 层（config/runtime/workflow/cli）
6. **[ ]** 编译 + 正确性验证 + 性能测试

### 阶段二（前向 Cooperative kernel）

7. **[ ]** 新增 `CooperativeDoubleBuffer` 枚举值
8. **[ ]** 实现 `apply_ryrz_rotation_chunk_cooperative_db_kernel<W>`（W=5,6,7,8）
9. **[ ]** 在 `launch_apply_ryrz_rotation_chunk_specialized` 中新增分支
10. **[ ]** 更新 `build_structured_forward_rotation_chunks` 中的 preference 选择逻辑
11. **[ ]** 编译 + 正确性验证 + 性能测试

### 阶段三（反向 kernel）

12. **[ ]** 新增 `CooperativeBackwardDoubleBuffer` 枚举值
13. **[ ]** 实现 `inverse_walk_ryrz_rotation_chunk_db_kernel<W>`（W=2,3,4；W=8 视 shared memory 情况决定）
14. **[ ]** 在 `launch_inverse_walk_ryrz_rotation_chunk` 中新增分支
15. **[ ]** 更新反向 sweep 中的 preference 选择逻辑
16. **[ ]** 编译 + 正确性验证 + 性能测试

---

## 7. CUDA 版本要求

| 功能 | 最低版本 |
|---|---|
| `cuda::pipeline` | CUDA 11.0 |
| `cuda::memcpy_async`（shared memory 目标） | CUDA 11.1 |
| C++17（`if constexpr` 等） | 已满足（现有代码已使用） |

---

## 8. CLI 使用示例

```bash
# 不使用双 buffer（默认）
python -m ring_ising.cli.run \
    --qubits 24 --layers 2 \
    --gradient-strategy structured_adjoint \
    --structured-rotation-chunk-width 4 \
    --steps 5

# 使用双 buffer（阶段一：Register kernel W=4）
python -m ring_ising.cli.run \
    --qubits 24 --layers 2 \
    --gradient-strategy structured_adjoint \
    --structured-rotation-chunk-width 4 \
    --steps 5 \
    --double-buffer

# 使用双 buffer（阶段二：Cooperative kernel W=8，小 qubit 数）
python -m ring_ising.cli.run \
    --qubits 12 --layers 2 \
    --gradient-strategy structured_adjoint \
    --structured-rotation-chunk-width 8 \
    --steps 5 \
    --double-buffer
```

---

## 9. 预期效果与局限

### 预期效果

| 场景 | 适用阶段 | 预期收益 |
|---|---|---|
| `num_qubits ≥ 23`，W=4，Register kernel | 阶段一 | 中等（memory-bound 时有效） |
| `num_qubits ≤ 15`，W=8，Cooperative kernel | 阶段二 | 较小（tile 数量少，流水线深度有限） |
| 反向 sweep，W=2/3/4 | 阶段三 | 中等（两个状态向量的 DRAM 带宽翻倍） |

### 局限

- 阶段一的收益依赖 GPU 是否处于 memory-bound 状态
- 阶段二的 `cuda::pipeline` 引入额外的 shared memory 开销和同步开销，小 qubit 数时可能反而变慢
- 阶段三 W=8 反向 kernel 的 shared memory 用量达到 48 KB 临界，需要特殊处理

---

## 10. 文件结构

```
GPU_Simulation/
├── double_buffer/
│   └── implementation_plan.md          ← 本文件
├── cpp/
│   ├── ising_cuda_kernel_common.cuh    ← 新增 3 个枚举值
│   ├── ising_cuda_rotation_kernels.cu  ← 新增阶段一、二 kernel
│   ├── ising_cuda_statevector_grad_kernels.cu ← 新增阶段三 kernel
│   ├── ising_cuda_backend.hpp          ← 新增 double_buffer 参数
│   ├── ising_cuda_backend.cu           ← 传递 double_buffer
│   └── ising_cuda_bindings.cpp         ← 暴露 double_buffer 给 Python
└── ring_ising/
    ├── config.py                        ← 新增 double_buffer 字段
    ├── backends/standalone/
    │   ├── config.py                    ← 新增 double_buffer 字段
    │   └── runtime.py                   ← 传递 double_buffer
    ├── workflows/standalone.py          ← 传递 double_buffer
    └── cli/run.py                       ← 新增 --double-buffer 开关
```
