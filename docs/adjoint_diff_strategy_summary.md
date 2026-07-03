# Adjoint-Diff CUDA 加速策略总结

本文面向后续读代码和复现实验的人，概述 standalone CUDA 后端中两个梯度计算策略：

- `structured_adjoint`: 大比特通用主方案。它保持 statevector 表示，利用 QML 电路的 layer 结构融合前向和反向计算，是当前项目的核心成果。
- `dense_scan`: 小比特辅助方案。它把每层压成 dense 矩阵节点，再用 dense product tree 并行扫描时间维度；只在 qubit 很少、layer 很多时有意义。

实验对象是 ring Ising QML 电路：

```text
for each layer:
    for each wire:
        RY(theta[layer, wire, 0])
        RZ(theta[layer, wire, 1])
    for each wire:
        CNOT(wire, (wire + 1) % num_qubits)
```

实验环境：

- GPU: NVIDIA RTX 6000 Ada Generation, 49140 MiB
- CUDA/Nsight Compute: `/usr/local/cuda-13.2`
- Python: project `.venv`
- 参数: `field=1.0`, `seed=7`, `init_scale=0.3`

## 主实验结果

### 大比特通用方案: `structured_adjoint`

下表来自 `benchmarks/results/codex_main_energy_grad_times.csv`。其中历史 `mode2` 即当前的 `structured_adjoint`。

| case | inverse_walk ms | ryrz_fused ms | structured_adjoint ms | PennyLane ms | vs inverse_walk | vs PennyLane |
|---|---:|---:|---:|---:|---:|---:|
| 8x8 | 1.549 | 1.010 | 0.737 | 16.030 | 2.10x | 21.74x |
| 12x8 | 2.595 | 2.020 | 0.940 | 22.073 | 2.76x | 23.48x |
| 16x4 | 2.208 | 1.423 | 0.951 | 17.424 | 2.32x | 18.31x |
| 20x2 | 13.823 | 8.277 | 6.532 | 21.737 | 2.12x | 3.33x |
| 24x1 | 147.009 | 104.376 | 65.453 | 381.034 | 2.25x | 5.82x |

后续继续优化 backward kernel 后，最终 stage profile 中 `structured_adjoint` 进一步降低到 24x1 约 33.7 ms 的阶段总和。主实验表仍保留最初对外比较用的一致 benchmark；后文会给出最终阶段数据。

### 小比特 dense 方案: `dense_scan`

下表来自 `benchmarks/results/codex_dense_scan_final_grid.csv` 与 q7/q8 边界探测。指标是训练脚本中的单步平均耗时，reference 为 `structured_adjoint`。

| case | structured_adjoint ms | dense_scan ms | dense_scan vs structured |
|---|---:|---:|---:|
| 4x32 | 0.467 | 0.181 | 2.58x |
| 4x512 | 7.146 | 0.590 | 12.10x |
| 5x32 | 0.623 | 0.236 | 2.64x |
| 5x512 | 9.710 | 0.977 | 9.93x |
| 6x32 | 0.818 | 0.377 | 2.17x |
| 6x512 | 12.840 | 2.998 | 4.28x |
| 7x128 | 4.369 | 4.213 | 1.04x |
| 7x512 | 17.627 | 16.474 | 1.07x |
| 7x2048 | 81.393 | 83.105 | 0.98x |
| 8x128 | 5.589 | 28.819 | 0.19x |
| 8x512 | 22.638 | 114.143 | 0.20x |

结论：

- `dense_scan` 在 4-6 qubit 且 layer 较多时很有效。
- 7 qubit 只有很窄的轻微收益区间。
- 8 qubit 虽然能正确运行，但 full dense matrix 的 `dim^3` 成本已经压倒收益，不应作为推荐路径。
- 因此，通用方案仍然是 `structured_adjoint`；若要把 dense 思路推广到更多 qubit，需要改成 hybrid/block dense scan，而不是继续扩大 full dense matrix。

## 数学原理

### Adjoint diff 基本公式

设电路由门或 fused block 组成：

```text
U = G_{m-1} ... G_1 G_0
psi_k = G_{k-1} ... G_0 |0>
E = <psi_m | H | psi_m>
```

反向伴随态定义为：

```text
lambda_m = H psi_m
lambda_k = G_k^dagger lambda_{k+1}
```

若参数 `p` 只出现在第 `k` 个算子 `G_k` 中，则：

```text
dE/dp = 2 Re <lambda_{k+1} | (dG_k/dp) | psi_k>
```

这说明 backward 不必保存所有参数平移结果，也不必对每个参数重新跑一遍电路；只需要在反向扫到含参算子时，同时知道进入该算子前的 state `psi_k` 和该算子后的 adjoint state `lambda_{k+1}`。

更重要的是，`G_k` 不一定必须是单个物理门。只要我们把一串连续门精确合成一个 block，公式仍然成立：

```text
B = G_{b-1} ... G_a
dE/dp = 2 Re <lambda_after_B | (dB/dp) | psi_before_B>
```

这就是 forward 和 backward 都能合并的数学基础。

### `structured_adjoint` 的完整流程

`structured_adjoint` 不改变 adjoint diff 的数学，只改变执行粒度。

Forward:

1. 初始化 `current = |0...0>`。
2. 对每一层：
   1. 把同一 wire 上连续的 `RY(theta)` 和 `RZ(phi)` 合成一个局部算子：

      ```text
      U_i(theta, phi) = RZ(phi) RY(theta)
      ```

   2. 将本层 rotation layer 分成若干连续 wire chunk 执行。低位 qubit 的相关振幅在 statevector 中相邻，适合一次 kernel 内用 register/shared memory 连续完成多个 rotation；高位 qubit stride 大，策略会自动避免过宽 chunk。
   3. 将整层 ring CNOT 作为一个 permutation kernel 执行，写入 scratch 并通过指针交换进入下一层。
3. 对最终 state 应用 Hamiltonian，得到 `lambda = H psi`。
4. 做 inner product 得到 energy。

Backward:

1. 从最后一层反向扫回第一层。
2. 对 `current` 和 `lambda` 先应用 inverse ring-CNOT permutation。CNOT 层无参数，反向只需把两个 statevector 同步按 `P^-1` 置换。
3. 反推本层 rotation layer。
   - `structured_rotation_chunk_width=1` 时，每个 kernel 处理一个 wire 上的 fused `RY+RZ`，同时计算两个参数梯度。
   - 默认宽度为 8。中大规模时低位保留 W8 chunk，高位根据 stride 拆成 W2/W4 chunk。
   - 当前最终实现对相邻两个 wire 使用 two-wire cell kernel：在一个 4-amplitude cell 内连续做两个 2x2 backward 更新，减少两门之间的 shared-memory round trip 和同步。
4. 对每个含参 rotation，根据公式计算：

   ```text
   grad_theta = 2 Re <lambda_after | d(RZ(phi)RY(theta))/dtheta | current_before>
   grad_phi   = 2 Re <lambda_after | d(RZ(phi)RY(theta))/dphi   | current_before>
   ```

同一 layer 中不同 wire 的 rotation 作用在不同 tensor factor 上，因此彼此对易。chunk kernel 中把多个 wire 连续执行，本质上仍是在计算同一个 tensor-product rotation layer。对于某个参数，`dB/dp` 只是把对应 wire 的局部 2x2 替换成导数矩阵，其余 wire 保持原矩阵；因此梯度公式保持不变。

### `dense_scan` 的数学原理

`dense_scan` 面向小 qubit，允许显式 dense matrix。当前版本每层只保留两个 dense leaf：

```text
R_layer = tensor_i [RZ(phi_i) RY(theta_i)]
C_layer = ring-CNOT permutation matrix
ops = [R_0, C_0, R_1, C_1, ...]
```

随后用 dense product tree 构造总矩阵，并用 downsweep 得到每个 leaf 的 `psi_before` 与 adjoint 侧状态。对于某个 rotation layer 参数 `p`：

```text
x = psi_before_R
y = R_layer^dagger lambda_after_R
lambda_after_R = R_layer y
dE/dp = 2 Re <lambda_after_R | (dR_layer/dp) x>
```

实现上不再物化 `dR_layer/dp` 的 dense 矩阵，而是在 CUDA kernel 中利用 tensor-product 结构直接计算。这样避免了 `num_params * dim^2` 的导数矩阵存储和上传。

dense_scan 的并行 span 更短，但总工作量随 `dim^3` 增长；这解释了为什么它在 4-6 qubit 很强，在 8 qubit 已经明显慢于 statevector 的 `structured_adjoint`。

## GPU 层面的加速原因

### Kernel launch 与全局内存流量

门级 baseline 每个 `RY`、`RZ`、`CNOT` 都启动 kernel。单独执行 `RY` 再执行 `RZ` 会对同一组振幅分别读写 global memory。`structured_adjoint` 把这些操作合在一个 kernel 或一个 chunk 内完成，减少：

- kernel launch 次数；
- global memory round trip；
- 中间 statevector 写回；
- backward 中 `current` 与 `lambda` 的重复调度开销。

### Statevector layout

statevector 的索引按二进制 basis 排列。低位 qubit 的 partner amplitude 相邻，适合 warp/cooperative tile；高位 qubit 的 partner amplitude stride 大，直接宽 chunk 会造成访存不连续和 shared-memory 同步成本上升。

当前策略因此分层处理：

- 低位保留较宽 W8 chunk，减少 launch 和 global read/write。
- 高位拆成 W2/W4 chunk，并使用 transposed/cell kernel 改善 global memory 访问形态。
- 对相邻两门使用 4-amplitude cell，在寄存器中连续做两次 2x2，避免 generic dense 4x4 带来的额外工作量。

### CNOT layer

整层 ring CNOT 是 basis-state permutation。逐门 CNOT 会多次读写 statevector；fused ring-CNOT 直接计算最终 index 映射。当前实现还用 GF(2) 线性变换的闭式索引公式替代逐 bit 模拟，减少每个线程内的整数分支。

CNOT permutation 是访存主导 kernel，SM arithmetic throughput 低并不表示实现差；关键指标是 DRAM throughput 和写入/读取模式。

## 调试数据与较优性证据

### 分阶段耗时

下表来自最终 `structured_adjoint` stage profile，单位 ms。`w1` 只启用 per-wire `RY+RZ` fusion 与 ring-CNOT fusion；`w8` 是完整默认策略。

| case | inverse total | structured_w1 total | structured_w8 total | w8 forward | w8 backward | Hamiltonian | dot |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8x8 | 1.362 | 0.546 | 0.413 | 0.063 | 0.310 | 0.006 | 0.019 |
| 12x8 | 2.463 | 1.125 | 0.600 | 0.097 | 0.457 | 0.008 | 0.023 |
| 16x4 | 2.079 | 0.817 | 0.598 | 0.098 | 0.444 | 0.016 | 0.024 |
| 20x2 | 15.442 | 6.402 | 3.699 | 0.658 | 2.687 | 0.227 | 0.039 |
| 24x1 | 141.242 | 60.816 | 33.711 | 6.428 | 22.366 | 3.959 | 0.651 |

观察：

- 24x1 中 forward 从门级约 49 ms 降到约 6.4 ms。
- `w1` 已经带来大幅收益，说明 per-wire `RY+RZ` fusion 和 CNOT layer fusion 是主要基础。
- `w8` 继续把 24x1 backward 从约 49.4-59.5 ms 量级降到约 22.4 ms，说明 backward chunk/cell 优化是必要的。
- Hamiltonian 和 dot 基本不变，说明加速来源确实是 circuit application 和 adjoint sweep。

### 只启用部分优化的 A/B 实验

Backward rotation 曾多轮调试。关键数据如下，单位 ms：

| strategy | 20x2 backward | 24x1 backward | 说明 |
|---|---:|---:|---|
| scalar + pair512 W8 | 3.788 | 33.560 | 手写 real/imag 后的 W8 基线 |
| high transposed W4 + low pair512 W8 | 2.823 | 29.848 | 高位拆成 W4，改善 stride 访存 |
| W4 cell2 + low pair512 W8 | 2.764 | 25.977 | 高位 two-wire cell |
| W4 cell2 + W8 cell2 | 2.687 | 22.366 | 当前最终策略 |

这个实验说明：

- 高位 chunk 不适合一味加宽；按 layout 拆分后更快。
- two-wire cell 的收益来自减少同步、shared-memory round trip、block reduction 与 atomicAdd，而不是把两个门硬做成 generic dense 4x4。
- 对 `current` 和 `lambda` 的 CNOT 尝试过双 statevector 合并 kernel，但 24x1 退化，说明少一次 launch 不一定抵得过更重的内存事务。

### Rotation chunk 分段策略消融

上一节说明了 backward kernel 本身的优化路径；这里单独看分段策略。最终方案不是简单地“chunk 越宽越好”，而是按 statevector layout 做 hybrid：低位保留 W8，20 qubit 以上的高位拆成 W2/W4，并在 backward 中使用 cell/transposed 变体。

同一批最终 stage profile 中，`w1` 表示只做 per-wire `RY+RZ` fusion 和 ring-CNOT fusion，不做 rotation layer chunk。`w8` 表示当前默认 hybrid 策略：

| case | w1 total | w8 total | w1 / w8 | w1 backward | w8 backward | backward ratio |
|---|---:|---:|---:|---:|---:|---:|
| 8x8 | 0.546 | 0.413 | 1.32x | 0.440 | 0.310 | 1.42x |
| 12x8 | 1.125 | 0.600 | 1.87x | 0.981 | 0.457 | 2.15x |
| 16x4 | 0.817 | 0.598 | 1.37x | 0.665 | 0.444 | 1.50x |
| 20x2 | 6.402 | 3.699 | 1.73x | 5.105 | 2.687 | 1.90x |
| 24x1 | 60.816 | 33.711 | 1.80x | 49.418 | 22.366 | 2.21x |

这说明完全退回 per-wire 虽然已经比门级 baseline 快很多，但 backward 会在所有规模上留下明显开销；rotation chunk 不是只对大规模有用。

历史 A/B 还验证了几个看上去合理但不够稳的替代方案：

| alternative | 20x2 total | 20x2 backward | 24x1 total | 24x1 backward | 问题 |
|---|---:|---:|---:|---:|---|
| all-W4 / 低位也拆窄 | 4.263 | 3.319 | 43.553 | 32.246 | 高位访存改善了，但低位连续 tile 的 W8 融合收益被放弃 |
| scalar + pair512 W8 | 4.762 | 3.816 | 44.871 | 33.560 | 对高位仍偏宽，stride 访存和同步成本重 |
| high transposed W4 + low pair512 W8 | 3.707 | 2.823 | 41.156 | 29.848 | 高位拆分有效，但低位 W8 backward 仍有 shared-memory round trip |
| W4 cell2 + low pair512 W8 | 3.706 | 2.764 | 37.277 | 25.977 | 高位 cell2 有效，但低位 pair512 仍未消掉两门之间的同步 |
| final W4 cell2 + W8 cell2 | 3.699 | 2.687 | 33.711 | 22.366 | 当前默认 |

这些消融支持当前分段策略：

- 低位不宜过度拆窄。低位 tile 在 global memory 中连续，W8 能把多次读写和 kernel launch 合并掉；all-W4 在大规模下比最终方案慢约 29% total / 44% backward。
- 高位不宜继续使用宽 W8。高位 chunk 的地址形态由 `base | (local_index << chunk_start)` 决定，`chunk_start` 大时 warp 内访问跨度变大；拆成 W2/W4 后，虽然 kernel 次数增加，但访存形态和 tile 内同步压力更合理。
- 只追求减少 launch 不可靠。双 statevector CNOT 合并、过宽 pair512 等尝试都显示，少一个 kernel launch 可能换来更重的内存事务、shared-memory 压力或同步成本。
- 最终方案在小规模没有明显退化，在 20x2/24x1 的 backward 热点上收益最大；这符合目标：作为通用 adjoint diff 路径，不追求单一设备上的极限特化，而是避免某一类规模明显掉队。

### Nsight Compute 指标

24x1 最终策略的 kernel group 汇总：

| kernel group | launches | mean time us | SM throughput | active warps | DRAM throughput | L2 throughput |
|---|---:|---:|---:|---:|---:|---:|
| backward_w4_cell2 | 24 | 3229.1 | 83.9% | 32.9% | 35.1% | 6.1% |
| backward_w8_cell2 | 6 | 6427.7 | 84.2% | 24.7% | 17.7% | 3.1% |
| forward_w4_register | 42 | 911.8 | 83.1% | 29.6% | 60.2% | 16.2% |
| ring_cnot_layer | 20 | 1190.6 | 1.7% | 77.7% | 89.7% | 23.9% |
| hamiltonian | 7 | 4246.2 | 84.0% | 98.2% | 34.3% | 19.3% |

解释：

- Rotation backward 的 active warps 并不高，但 SM throughput 在 84% 左右，说明它不是简单 occupancy-bound。cell2 后 active warps 甚至下降，但耗时显著下降，证明优化来自减少实际指令和同步工作。
- Ring-CNOT 的 SM throughput 很低、DRAM throughput 很高，符合 permutation/scatter kernel 的访存主导特征。
- Hamiltonian 的 active warps 和 SM throughput 都高，不是当前最优先瓶颈。
- Telemetry 中 24x1 最终 profile 的 avg SM util 约 86.4%，peak SM util 100%，peak process memory 约 1454 MiB，说明大规模下 GPU 被持续利用，显存也在合理范围。

## 当前结论

`structured_adjoint` 是当前应默认使用的通用方案。它保持 statevector 路径，适用于中大 qubit，并且在数学上严格等价于门级 adjoint diff。它的优势来自电路结构感知的融合，而不是近似。

`dense_scan` 是有价值的小 qubit 补充方案。它在 4-6 qubit、多层场景显著快于 structured_adjoint；但 full dense matrix 的 `dim^3` 成本使它无法自然扩展到 8 qubit 以上。后续若要延伸 dense 思路，应研究 hybrid/block dense_scan：只对局部 `k=6..7` qubit block 做 dense scan，其余维度继续走 statevector/block 调度。

## 复现实验命令

正确性：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

主实验：

```bash
.venv/bin/python benchmarks/compare_gradient_strategy.py \
  --cases 4x8 4x32 4x128 4x512 5x8 5x32 5x128 5x512 6x8 6x32 6x128 6x512 \
  --modes inverse_walk structured_adjoint dense_scan \
  --reference-mode structured_adjoint \
  --csv-out benchmarks/results/codex_dense_scan_final_grid.csv
```

q7/q8 dense 边界探测：

```bash
.venv/bin/python benchmarks/compare_gradient_strategy.py \
  --cases 7x8 7x32 7x128 7x512 8x8 8x32 8x128 8x512 \
  --modes structured_adjoint dense_scan \
  --reference-mode structured_adjoint \
  --csv-out benchmarks/results/codex_dense_scan_q7_q8_probe.csv
```

stage profile：

```bash
.venv/bin/python benchmarks/profile_baseline_stages.py \
  --cases 8x8 12x8 16x4 20x2 24x1 \
  --warmup 3 --repeats 20 \
  --structured-widths 1 8 \
  --csv-out benchmarks/results/codex_cell2_final_stages.csv
```
