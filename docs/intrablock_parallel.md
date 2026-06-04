# `intrablock_parallel` 模式说明

本文描述当前代码中 `intrablock_parallel` 梯度模式的工作原理、数据流以及复杂度分析。

当前实现对应文件主要是：

- `cpp/ising_cuda_intrablock_modes.inc`
- `cpp/ising_cuda_kernels.cu`
- `ring_ising/backends/standalone/config.py`

## 1. 设计目标

这个模式的目标是避免 `dense_scan` 那种显式 dense matrix 前缀扫描带来的高成本。

设：

- 总门数为 `k`
- qubit 数为 `q`
- statevector 维度为 `d = 2^q`
- block 数为 `M`
- 第 `j` 个 block 的长度为 `L_j`
- 若 block 大小统一为 `B`，则 `M = ceil(k / B)`，且 `L_j <= B`

我们希望：

- 不构造每个参数位置的 dense prefix matrix
- 不构造每个 block 的 full propagator `G_j`
- block 之间只传递 statevector 边界态
- block 内部再并行 replay，生成局部中间态并计算梯度

因此，当前实现已经是一条纯 statevector 路线。

## 2. 数学记号

设总电路门序列为：

```text
U_{k-1} ... U_1 U_0
```

按 block 切分后，第 `j` 个 block 内的门记为：

```text
U_{j,0}, U_{j,1}, ..., U_{j,L_j-1}
```

前向态定义为：

```text
psi_{j,0} = psi_start,j
psi_{j,r+1} = U_{j,r} psi_{j,r}
```

于是该 block 的结束边界态为：

```text
psi_end,j = psi_{j,L_j}
```

对能量

```text
E = <psi_final | H | psi_final>
```

定义反向边界态：

```text
lambda_{j,L_j} = lambda_end,j
lambda_{j,r} = U_{j,r}^\dagger lambda_{j,r+1}
```

则对 block 内第 `r` 个参数门的梯度：

```text
dE/dtheta = 2 Re <lambda_{j,r+1} | (dU_{j,r}/dtheta) | psi_{j,r}>
```

注意这里使用的是门作用之后的 adjoint 态 `lambda_{j,r+1}`，而不是 `lambda_{j,r}`。

## 3. 当前实现的数据流

当前实现可以分为四个阶段。

### 3.1 前向边界 sweep

实现位置：

- `propagate_intrablock_forward_boundaries(...)`
- `apply_ops_range_inplace(...)`

从零态 `|0...0>` 出发，按 block 顺序推进。

对于第 `j` 个 block：

1. 读取该 block 的起始边界态 `psi_start,j`
2. 顺序应用该 block 内的所有门
3. 得到结束边界态 `psi_end,j`
4. 将 `psi_end,j` 存入边界缓冲

这一步不会保存 block 内每一步的局部态，只保存 block 边界：

```text
psi_boundary[0], psi_boundary[1], ..., psi_boundary[M]
```

### 3.2 block 内前向 replay

实现位置：

- `launch_simulate_blocks_forward(...)`
- `simulate_blocks_forward_kernel(...)`

这一步以 block 为并行粒度。

每个 CUDA block 负责一个电路 block：

1. 从 `psi_boundary[j]` 读入起始边界态
2. 在 shared memory 中顺序应用该 block 内的门
3. 将每个局部位置的前向态写出到 `forward_states`

于是得到：

```text
psi_{j,0}, psi_{j,1}, ..., psi_{j,L_j}
```

这部分数据随后会被反向梯度 kernel 复用。

### 3.3 反向边界 sweep

实现位置：

- `propagate_intrablock_backward_boundaries(...)`

先构造：

```text
lambda_final = H psi_final
```

然后按 block 逆序推进：

1. 从 `lambda_end,j` 开始
2. 逆序应用 block 内各门的 `U^\dagger`
3. 得到 `lambda_start,j`

只保存 block 边界：

```text
lambda_boundary[0], lambda_boundary[1], ..., lambda_boundary[M]
```

### 3.4 block 内反向 replay + 梯度累积

实现位置：

- `launch_simulate_blocks_backward_and_gradient(...)`
- `simulate_blocks_backward_and_gradient_kernel(...)`

这一步同样以 block 为并行粒度。

每个 CUDA block：

1. 从 `lambda_boundary[j+1]` 取出该 block 的终点 adjoint 态
2. 从 `forward_states` 读取前向 replay 已保存的 `psi_{j,r}`
3. 按局部反向顺序：
   - 先用当前 `lambda_{j,r+1}` 与 `psi_{j,r}` 计算梯度
   - 再做一次 `U_{j,r}^\dagger`，更新到 `lambda_{j,r}`

因此，当前实现已经把“反向 block replay”和“梯度累积”融合在一个 kernel 里。

## 4. 与旧版 `G_j` 路线的区别

旧版思路是：

- 对每个 block 显式构造一个 dense matrix `G_j`
- 用 `G_j` 快速到达前向边界
- 用 `G_j^\dagger` 快速到达反向边界

它的问题是：

- 构造 `G_j` 本身就是对 `d x d` 结构做更新
- 即使不保存每个参数位置的 prefix matrix，单个 `G_j` 的构造仍是 `O(L_j d^2)`
- 在 `q` 增大时，`d^2` 项会迅速主导

当前版本完全去掉了 `G_j`：

- 边界传播回到 statevector-only 的 `O(L_j d)`
- block 内 replay 也是 statevector-only 的 `O(L_j d)`

这也是当前版本在 `9q / 10q` 上明显优于旧版的根本原因。

## 5. 复杂度分析

## 5.1 单个门作用到 statevector 的成本

当前电路中的门主要是：

- `FusedRYRZ`
- `RingCNOTLayer`

对长度为 `d` 的 statevector：

- 单 qubit 旋转类门：`O(d)`
- permutation / entangler 类门：`O(d)`

因此，任意一个 gate 的 statevector 作用成本都可以记为 `Theta(d)`。

## 5.2 前向边界 sweep

总共对全电路每个门作用一次：

```text
T_forward_boundary = Theta(k d)
```

## 5.3 block 内前向 replay

每个 block 再从边界重放一次局部前向，所有 block 合起来仍然覆盖全部 `k` 个门一次：

```text
T_forward_replay = Theta(k d)
```

## 5.4 反向边界 sweep

同理，对全电路每个门的伴随作用一次：

```text
T_backward_boundary = Theta(k d)
```

## 5.5 block 内反向 replay + 梯度

反向 replay 本身：

```text
Theta(k d)
```

参数梯度累积部分：

- 参数门数记为 `p`
- 每个参数门的梯度计算对 statevector 做一次 `O(d)` 级收缩

因此：

```text
T_grad = Theta(p d)
```

对于当前 ring ansatz，`p` 与 `k` 同阶，因此整体仍可写为：

```text
T_backward_replay_and_grad = Theta(k d)
```

## 5.6 总时间复杂度

把四个阶段加起来：

```text
T_total
= Theta(k d) + Theta(k d) + Theta(k d) + Theta(k d)
= Theta(k d)
```

更精确地说，它的常数项大约对应：

- 1 次前向边界 sweep
- 1 次局部前向 replay
- 1 次反向边界 sweep
- 1 次局部反向 replay
- 外加梯度收缩

所以虽然大 O 是 `Theta(k d)`，但常数因子不是 1，而是一个“多次 replay”的常数。

## 5.7 总空间复杂度

当前实现主要存储：

1. block 边界前向态

```text
(M + 1) * d
```

2. block 边界反向态

```text
(M + 1) * d
```

3. 所有 block 的局部前向态

```text
sum_j (L_j + 1) * d = (k + M) * d
```

4. 梯度与门描述符

```text
O(p) + O(k)
```

因此主导空间复杂度是：

```text
S_total = Theta((k + M) d) = Theta(k d)
```

这是当前版本最重要的性质之一：

- 没有 `Theta(M d^2)`
- 也没有 `Theta(k d^2)`

## 6. 为什么 `block_size` 的影响相对平缓

设统一 block 大小为 `B`，则：

```text
M = ceil(k / B)
```

时间主导项是 `Theta(k d)`，并不会因为 `B` 改变而改变阶数。

`block_size` 主要影响：

- block 数 `M`
- 边界态数量 `O(M d)`
- 单个 block replay kernel 的工作量和并行粒度

因此它更多影响常数项，而不是主导复杂度。

这也是 benchmark 中经常看到：

- `block_size` 在一个较大区间内性能差异不大的原因

## 7. 当前瓶颈

在去掉显式 `G_j` 后，profile 显示热点主要变成：

- `apply_ryrz_kernel`
- `apply_ry_kernel`
- `apply_rz_kernel`
- `simulate_blocks_backward_and_gradient_kernel`
- `simulate_blocks_forward_kernel`

这说明：

- `d^2` 级 dense block matrix 构造已经不是瓶颈
- 现在瓶颈更接近“多次 statevector replay 带来的总门作用次数”和 kernel launch 开销

因此下一步优化更适合朝这些方向走：

- 把边界 sweep 也 block 化，减少“一门一个 kernel”
- 进一步融合 forward replay / backward replay / gradient accumulation
- 减少局部前向态持久存储，转向更流式的重建策略

## 8. 与其他模式的对比

### `save_param_states`

- 时间复杂度：`Theta(k d)`
- 空间复杂度：约 `Theta(p d)`
- 优点：实现简单，适合较大 qubit
- 缺点：需要为参数门保存大量前向态

### `dense_scan`

- 当前实验路径中涉及 dense matrix product tree
- 主成本接近 `Theta(k d^2)` / `Theta(k d^3)` 的 dense linear algebra
- 小 `q` 时可能很快，但不适合更大 qubit

### 当前 `intrablock_parallel`

- 时间复杂度：`Theta(k d)`
- 空间复杂度：`Theta(k d)`
- 特点：用更多 replay 换掉 dense matrix
- 在 `9q / 10q` 上通常明显优于显式 dense block propagator 路线

## 9. 小结

当前版本的 `intrablock_parallel` 可以概括为：

- block 之间：只传 statevector 边界态
- block 之内：做局部前向/反向 replay
- 不显式构造 dense block propagator

它的主要优点是把算法重新拉回到 statevector 复杂度：

```text
Theta(k d)
```

而不是 dense matrix 的更高阶代价。

代价则是：

- 需要多次 replay
- 需要保存所有 block 内的局部前向态

所以当前实现更适合作为一个“去掉 dense matrix 之后的高 qubit 版本”基础形态，后续还能继续向更流式、更少 replay、更少 launch 的方向演化。
