# 树归约（非交换独占扫描）分析

## 概述

GPU 仿真后端中的 `dense_scan` 梯度策略使用**树归约**（非交换独占扫描）来并行计算所有门矩阵的前缀/后缀乘积，实现 O(log N) 的并行深度，而非 O(N) 的顺序计算。本文档对 `cpp/ising_cuda_backend.cu` 和 `cpp/ising_cuda_kernels.cu` 中的实现进行详细分析。

---

## 1. 数学基础

### 问题描述

给定 N 个门矩阵的序列 `U₀, U₁, ..., U_{N-1}`，我们需要计算：

- **前缀乘积**：`P_k = U_{k-1} · U_{k-2} · ... · U₀`，对所有 k（从左到右的独占扫描）
- **后缀乘积**：`S_k = U†_{N-1} · U†_{N-2} · ... · U†_{k+1}`，对所有 k（从右到左的独占扫描）

由于矩阵乘法是**非交换的**（`A·B ≠ B·A`），我们不能使用为交换操作设计的标准并行前缀求和算法。相反，我们使用适配于非交换乘积的 **Blelloch 树扫描算法**。

### 为什么这对梯度计算很重要

对于参数化量子电路的梯度，通过参数移位法则：

```
∂E/∂θₖ = 2 · Re(⟨λ_k | dU_k/dθₖ | ψ_k⟩)
```

其中：
- `|ψ_k⟩ = U_{k-1} · ... · U₀ |0⟩` — 门 k 之前的前向状态
- `|λ_k⟩ = U†_{N-1} · ... · U†_{k+1} |H·ψ_final⟩` — 门 k 处的后向（lambda）状态

朴素地计算所有 `|ψ_k⟩` 和 `|λ_k⟩` 需要 O(N²) 次矩阵-向量乘法。树归约将其减少为 O(N log N) 次矩阵乘法，并行深度为 O(log N)。

---

## 2. 算法：Blelloch 非交换独占扫描

### 2.1 上扫（归约）阶段

从输入矩阵数组开始，上扫构建归约树：

```
第0层（输入）:     [U₀] [U₁] [U₂] [U₃] [U₄] [U₅] [U₆] [U₇]
                      \   /       \   /       \   /       \   /
第1层:              [U₁·U₀]     [U₃·U₂]     [U₅·U₄]     [U₇·U₆]
                        \     /                   \     /
第2层:              [U₃·U₂·U₁·U₀]           [U₇·U₆·U₅·U₄]
                          \              /
第3层:              [U₇·...·U₀]  （总乘积）
```

在每个层级 `d`，步长为 `2^d` 的矩阵对被相乘：右矩阵乘以左矩阵（保持非交换顺序）。

### 2.2 下扫（分配）阶段

上扫完成后，下扫分配部分乘积以计算独占扫描：

```
上扫后的起始状态：
  [U₀] [U₁·U₀] [U₂] [U₃·U₂] [U₄] [U₅·U₄] [U₆] [U₇·U₆·U₅·U₄·U₃·U₂·U₁·U₀]

步骤1 (span=4): 将左子替换为父节点，父节点替换为左·父
步骤2 (span=2): 继续分配
步骤3 (span=1): 最终分配

结果（独占扫描）：
  [I] [U₀] [U₁·U₀] [U₂·U₁·U₀] [U₃·U₂·U₁·U₀] [U₄·U₃·U₂·U₁·U₀] ...
```

关键不变量：下扫完成后，位置 k 包含其左侧所有矩阵的乘积：`P_k = U_{k-1} · U_{k-2} · ... · U₀`。

---

## 3. 实现细节

### 3.1 索引计算：`prepare_level_indices`

函数 `prepare_level_indices`（`ising_cuda_backend.cu` 第 490–540 行）计算树的每个层级中需要相乘的矩阵对：

**上扫**（span = 1, 2, 4, ..., < padded_size）：
- 对于步长为 `2*span` 的每个对组 `g`：
  - **右**索引为 `2*span*g + 2*span - 1`（对组的最右元素）
  - **左**索引为 `2*span*g + span - 1`（中间元素）
- 乘法：`result[right] = result[right] · result[left]`

**下扫**（span = padded_size/2, padded_size/4, ..., 1）：
- 索引结构相同，但操作不同：
  1. 将 `mats[left]` 保存到临时缓冲区（`left_tmp`）
  2. 将 `mats[right]` 复制到 `mats[left]`
  3. 乘法：`result[right] = tmp[left] · mats[left]`（即旧左 × 新左）

索引计算在主机端执行，然后将左/右索引数组上传到设备端供 CUDA 内核使用。

### 3.2 上扫实现

```cpp
for (std::size_t span = 1; span < padded; span *= 2) {
    const auto pairs = padded / (2 * span);
    prepare_level_indices(/* 上扫索引 */);
    upload_pointer_array(/* ptr_a = 右矩阵, ptr_b = 左矩阵, ptr_c = 输出 */);
    cublasZgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       dim, dim, dim, &alpha,
                       ptr_a, dim, ptr_b, dim, &beta, ptr_c, dim, pairs);
    launch_scatter_matrices(/* 将结果散射回树数组 */);
}
```

在每个层级，执行 `pairs = padded / (2*span)` 次独立的矩阵乘法。`cublasZgemmBatched` 调用在单个 GPU 内核启动中执行所有乘法。`scatter_matrices_kernel` 将连续的批处理输出缓冲区中的结果写回树数组中的正确位置。

### 3.3 下扫实现

```cpp
for (std::size_t span = padded / 2; span >= 1; span /= 2) {
    const auto pairs = padded / (2 * span);
    prepare_level_indices(/* 下扫索引 */);
    launch_prepare_downsweep_buffers(/* 保存左子，复制右→左 */);
    upload_pointer_array(/* ptr_a = left_tmp（保存的左）, ptr_b = mats（新左）, ptr_c = 输出 */);
    cublasZgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       dim, dim, dim, &alpha,
                       ptr_a, dim, ptr_b, dim, &beta, ptr_c, dim, pairs);
    launch_scatter_matrices(/* 将结果散射回树数组 */);
}
```

`prepare_downsweep_buffers_kernel` 至关重要：它将左子矩阵保存到临时缓冲区，并将右子矩阵复制到左子的位置。这实现了 Blelloch 算法的下扫交换操作。随后的 GEMM 计算 `old_left × new_left` 并将结果存储在右子的位置。

### 3.4 GPU 加速矩阵运算

树归约使用 **cuBLAS 批量矩阵乘法**（`cublasZgemmBatched`）来执行每个层级的 O(N/2) 次并行矩阵乘法：

```cpp
cublasZgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   dim, dim, dim,       // 2^n × 2^n 复数矩阵
                   &alpha,
                   ptr_a, dim,          // 指向右矩阵的指针数组
                   ptr_b, dim,          // 指向左矩阵的指针数组
                   &beta,
                   ptr_c, dim,          // 指向输出矩阵的指针数组
                   pairs);              // 并行乘法数量
```

对于 n ≤ 6 个量子比特，`dim = 2^n ≤ 64`，因此每个矩阵最大为 64×64 复数双精度（32 KB）。批量 GEMM 在每个树层级内高效地并行化独立的矩阵乘法。

### 3.5 辅助 CUDA 内核

多个自定义 CUDA 内核支持树归约：

| 内核 | 用途 | 并行方式 |
|------|------|----------|
| `fill_identity_matrices_kernel` | 将树叶初始化为单位矩阵（用于填充） | 每个矩阵元素一个线程 |
| `prepare_downsweep_buffers_kernel` | 保存左子，复制右→左（下扫阶段） | 每个矩阵元素一个线程 |
| `scatter_matrices_kernel` | 按索引将批量 GEMM 结果散射回树数组 | 每个矩阵元素一个线程 |
| `gather_vectors_kernel` | 按索引收集向量/矩阵（用于后缀扫描反转） | 每个向量元素一个线程 |
| `build_adjoint_batch_kernel` | 计算一批矩阵的共轭转置 | 每个矩阵元素一个线程 |
| `reduce_real_inner_products_kernel` | 使用共享内存归约计算批量 ⟨λ|dU|ψ⟩ 内积 | 每个参数一个块，256 个线程 |

### 3.6 指针数组管理

批量 cuBLAS 调用需要一个指向各个矩阵的设备指针数组。实现通过以下方式管理：

1. **主机端指针计算**：对于每对，计算左矩阵、右矩阵和输出矩阵的设备地址
2. **`upload_pointer_array`**：将主机指针数组复制到设备内存
3. **`cublasZgemmBatched`**：使用设备指针数组进行批量乘法

这是必要的，因为树归约操作的是矩阵数组的非连续子集（每个层级的特定对），因此简单的步长批量方法不够用。

---

## 4. 完整 GPU 流水线：`run_dense_scan_gpu_pipeline`

树归约嵌入在一个更大的 GPU 流水线中，该流水线通过 GPU 的一次遍历计算所有前向状态、后向状态和梯度：

### 阶段 1：前缀扫描（前向状态）

```
输入：门矩阵 U₀, U₁, ..., U_{N-1}
输出：前缀乘积 P_k = U_{k-1} · ... · U₀  （P₀ = I）

1. 将填充位置初始化为单位矩阵
2. 将门矩阵复制到扫描缓冲区
3. 运行非交换独占扫描（上扫 + 下扫）
4. 批量矩阵乘法：psi_before[k] = P_k · |0⟩   （对所有 k 同时执行）
5. 批量矩阵乘法：psi_after[k] = U_k · psi_before[k]
```

步骤 4 使用 `cublasZgemmStridedBatched` 在单次批量调用中将每个前缀乘积矩阵乘以初始状态 `|0⟩`。步骤 5 类似地将每个门应用于其对应的前门状态。

### 阶段 2：能量计算

```
1. psi_final = psi_after[N-1] = U_{N-1} · ... · U₀ · |0⟩
2. lambda_k = H · psi_final   （哈密顿量矩阵-向量乘积，通过 cublasZgemm）
3. energy = ⟨psi_final | lambda_k⟩   （通过 thrust::transform_reduce）
```

哈密顿量通过单次 `cublasZgemm` 调用作为密集矩阵-向量乘积应用。能量通过 Thrust 的并行归约计算为复内积。

### 阶段 3：后缀扫描（后向状态）

```
输入：门矩阵 U₀, U₁, ..., U_{N-1}
输出：后缀乘积 S_k = U†_{N-1} · ... · U†_{k+1}

1. 构建伴随批次：U†_k 对所有 k  （build_adjoint_batch_kernel）
2. 反转顺序：U†_{N-1}, U†_{N-2}, ..., U†₀  （使用反向索引的 gather_vectors_kernel）
3. 对反转的伴随矩阵运行非交换独占扫描
4. 将扫描结果反转回原始顺序  （使用反向索引的 gather_vectors_kernel）
5. 批量矩阵乘法：lambda_by_op[k] = S_k · lambda_k
```

后缀扫描巧妙地复用了前缀扫描：
1. 计算每个门矩阵的伴随（共轭转置）
2. 反转伴随矩阵的顺序
3. 运行相同的前缀扫描算法（计算从左到右的乘积）
4. 将结果反转回原始顺序

这之所以有效是因为：`S_k = U†_{N-1} · ... · U†_{k+1} = reverse_prefix_scan(reverse(U†₀, ..., U†_{N-1}))[N-1-k]`

### 阶段 4：梯度计算

```
对每个参数 θₖ：
1. 收集 lambda_for_params[k] = lambda_by_op[param_gate_index[k]]
2. 收集 psi_before_for_params[k] = psi_before[param_gate_index[k]]
3. 批量矩阵乘法：deriv_states[k] = dU_k/dθₖ · psi_before_for_params[k]
4. 批量内积：gradient[k] = 2 · Re(⟨lambda_for_params[k] | deriv_states[k]⟩)
```

步骤 1-2 使用 `gather_vectors_kernel` 仅收集对应于参数门的状态。步骤 3 使用 `cublasZgemmStridedBatched` 同时应用所有导数矩阵。步骤 4 使用 `reduce_real_inner_products_kernel` 通过共享内存归约并行计算所有梯度贡献。

---

## 5. 并行性分析

### 5.1 并行层次

| 层次 | 机制 | 粒度 |
|------|------|------|
| **操作间** | 所有 N 个门矩阵在批量操作中处理 | N 个操作并行 |
| **矩阵内** | cuBLAS 批量 GEMM 在每个 2^n × 2^n 矩阵内并行化 | 最多 64×64 = 4096 个元素 |
| **树层级** | 每个层级 O(N/2) 次独立矩阵乘法 | 每个树层级的对 |
| **内核级** | 自定义 CUDA 内核，每块 256 个线程 | 逐元素操作 |

### 5.2 计算复杂度

| 方法 | 前向状态 | 后向状态 | 总计 |
|------|----------|----------|------|
| **顺序** | O(N) 次矩阵-向量乘法 | O(N) 次矩阵-向量乘法 | O(N) |
| **树归约** | O(N log N) 次矩阵乘法 | O(N log N) 次矩阵乘法 | O(N log N) |
| **树归约（并行深度）** | O(log N) | O(log N) | O(log N) |

树归约做了更多的总工作量（O(N log N) vs O(N)），但实现了 O(log N) 的并行深度。对于 N = 2nq × nlayers 个操作且 nq ≤ 6，矩阵维度很小（≤ 64），因此批量矩阵乘法在 GPU 上效率很高。

### 5.3 内存使用

对于 n 个量子比特和 N 个操作：
- 门矩阵：`N × 2^n × 2^n` 个复数
- 前缀扫描缓冲区：`padded(N) × 2^n × 2^n` 个复数（填充到下一个 2 的幂）
- 后缀扫描缓冲区：`padded(N) × 2^n × 2^n` 个复数
- 临时缓冲区：`O(N/2 × 2^n × 2^n)` 个复数
- 前向/后向状态：各 `O(N × 2^n)` 个复数

对于 n=6，N=78（6 个量子比特，13 层）：每个矩阵为 64×64 = 4096 个复数 = 32 KB。填充到 128 后，扫描缓冲区为 128 × 32 KB = 4 MB。总 GPU 内存约为 20-30 MB，完全在现代 GPU 的能力范围内。

---

## 6. 关键加速技术

### 6.1 批量线性代数（cuBLAS）

最重要的加速来自使用 `cublasZgemmBatched` 和 `cublasZgemmStridedBatched` 在单次 GPU 内核启动中执行许多独立的矩阵乘法。这摊销了内核启动开销，并允许 GPU 在数千个线程间高效调度工作。

### 6.2 树归约实现并行前缀乘积

Blelloch 扫描算法将本质上顺序的前缀乘积计算转换为并行深度为 O(log N) 的并行算法。每个树层级执行 O(N/2) 次独立矩阵乘法，作为单次批量 GEMM 调用执行。

### 6.3 融合门操作

`FusedRYRZ` 门将 RY 和 RZ 旋转合并到单个内核中，减少了内存流量和内核启动开销。`apply_ring_cnot_layer_kernel` 将整层 CNOT 门融合到单个内核中，在寄存器中计算输出索引映射。

### 6.4 自定义 CUDA 内核处理数据移动

`prepare_downsweep_buffers_kernel` 在单个内核中执行下扫阶段所需的保存和交换操作，避免了多次设备到设备的复制。`scatter_matrices_kernel` 和 `gather_vectors_kernel` 高效处理树归约的不规则访问模式。

### 6.5 共享内存归约计算内积

`reduce_real_inner_products_kernel` 使用共享内存和步长减半归约模式高效计算批量内积。每个块使用 256 个线程计算一个参数的梯度贡献：

```cuda
__shared__ double partial[THREADS];  // 256 个元素
// 每个线程计算向量元素的部分和
for (std::size_t elem = threadIdx.x; elem < vector_size; elem += blockDim.x) {
    sum += conj(lhs[offset]) * rhs[offset]).real();
}
partial[threadIdx.x] = sum;
__syncthreads();
// 共享内存中的树归约
for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
}
// 线程 0 写入最终结果
if (threadIdx.x == 0) out[batch_index] = scale * partial[0];
```

### 6.6 异步内存操作

流水线使用 `cudaMemcpyAsync` 进行主机到设备和设备到设备的复制，在可能的情况下允许与计算重叠。

### 6.7 延迟同步

`maybe_synchronize_cuda` 函数（由 `STANDALONE_CUDA_EAGER_SYNC` 编译标志控制）允许实现最小化 `cudaDeviceSynchronize` 调用，依赖 CUDA 流顺序保证正确性，仅在主机需要结果时才同步。

---

## 7. 与顺序方法的比较

### 7.1 顺序状态向量（`SaveParamStates`）

`SaveParamStates` 策略顺序计算梯度：
1. 前向传播：逐个应用门，保存每个参数门之前的状态
2. 后向传播：逐个应用伴随门，在参数门处计算梯度

**代价**：O(N) 次状态向量操作，每次 O(2^n) 工作量
**内存**：O(P × 2^n)，其中 P 为参数数量

### 7.2 检查点策略（`Checkpoint`）

`Checkpoint` 策略以重新计算换取内存：
1. 前向传播：每 K 个操作保存一次检查点
2. 后向传播：从最近的检查点重新计算状态

**代价**：O(N + N·K/K) = O(N) 次状态向量操作（但有更多内存访问）
**内存**：O((N/K + K) × 2^n)

### 7.3 树归约（`BruteForceParallelQ6`）

**代价**：O(N log N) 次矩阵-矩阵乘法（每次 O(2^{2n})），但并行深度为 O(log N)
**内存**：O(N × 2^{2n}) 用于矩阵存储

对于小量子比特数（n ≤ 6），矩阵维度足够小，O(N log N) 次矩阵乘法可以在 GPU 上高效批量执行，O(log N) 的并行深度相比顺序 O(N) 方法提供了显著加速。

---

## 8. 局限性

1. **量子比特约束**：仅支持 n ≤ 6 个量子比特，因为矩阵维度 2^n 必须足够小以实现高效的批量 GEMM。在 n=6 时，矩阵为 64×64；在 n=7 时将为 128×128，显著增加内存和计算量。

2. **内存扩展**：总内存为 O(N × 2^{2n})，随状态维度二次增长。这限制了较大量子比特数的层数。

3. **工作量开销**：树归约执行 O(N log N) 次矩阵乘法，而顺序方法为 O(N)。仅当并行深度减少带来的收益超过额外工作量时才有益，这在 GPU 上执行小矩阵时成立。

4. **填充要求**：算法要求将 N 填充到下一个 2 的幂，在最坏情况下可能浪费高达 50% 的矩阵缓冲区空间。