# Structured Adjoint-Diff 性能报告

> 命名说明：本报告早期实验原名为 `mode2`。当前正式策略名为 `structured_adjoint`，代码仍接受 `mode2` 作为兼容 alias；部分原始 CSV 文件名保留历史标签。

本文总结当前 standalone CUDA 后端中 `structured_adjoint` 的实现、正确性依据与性能结果。实验对象是 ring Ising QML 电路：

```text
for each layer:
    for each wire:
        RY(theta[layer, wire, 0])
        RZ(theta[layer, wire, 1])
    for each wire:
        CNOT(wire, (wire + 1) % num_qubits)
```

实验机器：

- GPU: NVIDIA RTX 6000 Ada Generation, 49140 MiB
- CUDA/Nsight Compute: `/usr/local/cuda-13.2`
- Python: project `.venv`
- 参数: `field=1.0`, `seed=7`, `init_scale=0.3`
- 时间指标: 单次 `energy_and_grad` 调用的 median wall time，单位 ms

原始数据文件：

- 主实验: `benchmarks/results/codex_main_energy_grad_times.csv`
- stage profile: `benchmarks/results/codex_doc_stage_profile.csv`
- telemetry: `benchmarks/results/codex_doc_gpu_telemetry_*.csv`
- Nsight Compute: `benchmarks/results/codex_doc_ncu_24x1_kernel_profile.csv`

## 主实验

对比对象：

- `inverse_walk`: standalone 基线，严格按 `RY -> RZ -> CNOT` 门级结构执行。
- `ryrz_fused`: 新增对照，只融合每个 wire 上连续的 `RY+RZ`，CNOT 仍逐门执行。
- `structured_adjoint`: 当前优化模式，融合 rotation layer，融合 ring-CNOT layer，并在 backward 对 rotation 做部分 chunk fusion。
- `PennyLane`: `lightning.gpu` + `diff_method="adjoint"`。

| case | inverse_walk ms | ryrz_fused ms | structured_adjoint ms | PennyLane ms | structured_adjoint vs inverse | structured_adjoint vs PennyLane |
|---|---:|---:|---:|---:|---:|---:|
| 8x8 | 1.549 | 1.010 | 0.737 | 16.030 | 2.10x | 21.74x |
| 12x8 | 2.595 | 2.020 | 0.940 | 22.073 | 2.76x | 23.48x |
| 16x4 | 2.208 | 1.423 | 0.951 | 17.424 | 2.32x | 18.31x |
| 20x2 | 13.823 | 8.277 | 6.532 | 21.737 | 2.12x | 3.33x |
| 24x1 | 147.009 | 104.376 | 65.453 | 381.034 | 2.25x | 5.82x |

观察：

- `ryrz_fused` 已经带来明显收益，说明只减少 `RY/RZ` 的 kernel launch 和 statevector 读写就有价值。
- `structured_adjoint` 继续优于 `ryrz_fused`，收益来自整层 rotation chunk、ring-CNOT layer fusion、backward chunk fusion，以及 backward CNOT 的双缓冲调度。
- 大规模下 `PennyLane` 相对 standalone 的差距缩小，但 `structured_adjoint` 在 24x1 仍有 5.82x 优势。

所有主实验 case 中，standalone 与 PennyLane 的 energy 在显示精度内一致；`unittest tests.test_backends_parity` 也覆盖了 energy/gradient parity。

## Structured Adjoint 流程

`structured_adjoint` 使用与 inverse-walk adjoint 相同的整体算法，只改变局部线性算子的执行粒度。

Forward:

1. 初始化 `current = |0...0>`。
2. 对每个 layer：
   1. 将本层所有 `RY+RZ` 作为 structured rotation layer 执行。
   2. rotation layer 按连续 wire chunk 分段。低位 qubit 的相关振幅在 statevector 中相邻，适合 cooperative/shared-memory 或寄存器 tile；高位 qubit stride 大，策略会自动避免过重 chunk。
   3. 整层 ring CNOT 作为一个置换 kernel 执行，输出到 scratch，然后通过指针交换进入下一层。
3. 对最终 state 应用 Hamiltonian，得到 `lambda = H|psi>`。
4. inner product 得到 energy。

Backward:

1. 从最后一层倒序扫回第一层。
2. 先对 `current` 和 `lambda` 同时应用 inverse ring-CNOT layer。
   - 当前实现为两个输出 buffer，并通过指针交换避免每层额外 device-to-device copy。
3. 再反推本层 rotation。
   - `structured_width=1` 等价 per-wire fused backward: 一次 kernel 同时反推 `RY+RZ` 并计算两个参数梯度。
   - 默认 `structured_rotation_chunk_width=8`。12-15 qubit 自动降到有效宽度 4；16-19 qubit 只有请求 8 时启用 chunk；20 qubit 以上会把 8 号以上高位拆成 W2-W4 chunk，其中 W4 使用 two-wire cell kernel，W2/W3 使用 transposed kernel；低位 0-7 继续使用 W8 chunk，并在 24 qubit 以上使用 two-wire cell kernel。

## 正确性依据

### RY+RZ fusion

原始每个 wire 上的局部门为：

```text
U(theta, phi) = RZ(phi) RY(theta)
```

`ryrz_fused` 和 `structured_adjoint` 的 fused kernel 对每一对振幅 `(a0, a1)` 直接计算 `RZ(phi) RY(theta) [a0, a1]^T`，没有改变矩阵乘法顺序，只是把两个 kernel 合并成一个 kernel。

Backward 中使用 adjoint/inverse-walk：

```text
current_before = U^† current_after
lambda_before  = U^† lambda_after
grad_p = 2 Re <lambda_after | dU/dp | current_before>
```

对 `RY` 和 `RZ` 的两个参数分别计算 `dU/dtheta`、`dU/dphi`，因此数学上等价于门级 adjoint。

### Rotation chunk fusion

同一 layer 中不同 wire 的 single-qubit rotation 作用在不同 qubit 上：

```text
U_layer = Π_i U_i
```

这些 `U_i` 之间两两对易，因为它们作用在不同 tensor factor 上。chunk kernel 在一个 tile 内顺序应用多个 `U_i`，等价于原本逐 wire 应用。区别仅是执行粒度：把多次 global-memory load/store 和多次 launch 合成较少次数。

### Ring-CNOT layer fusion

一整层 CNOT 是确定性的 basis-state permutation。原始逐门 CNOT 做的是：

```text
index -> P(index)
```

fused ring-CNOT kernel直接计算同一个置换 `P`，并写到输出 buffer。因为它是 permutation，所以 unitary；反向传播中使用 inverse permutation `P^-1`，等价于逐个反向 CNOT。

当前 CNOT layer kernel 还把逐 bit 循环替换为 GF(2) 线性变换的闭式索引公式，减少每个线程的整数控制流。

## Stage Breakdown

下表来自 CUDA event stage profiling，单位 ms。

| case | mode | total | forward | backward | Hamiltonian | dot |
|---|---|---:|---:|---:|---:|---:|
| 8x8 | inverse_walk | 1.360 | 0.527 | 0.788 | 0.006 | 0.019 |
| 8x8 | structured_w1 | 0.627 | 0.063 | 0.520 | 0.006 | 0.020 |
| 8x8 | structured_w8 | 0.547 | 0.063 | 0.441 | 0.007 | 0.019 |
| 12x8 | inverse_walk | 2.463 | 0.950 | 1.461 | 0.008 | 0.024 |
| 12x8 | structured_w1 | 1.367 | 0.097 | 1.220 | 0.008 | 0.023 |
| 12x8 | structured_w8 | 0.800 | 0.097 | 0.656 | 0.008 | 0.023 |
| 16x4 | inverse_walk | 1.997 | 0.739 | 1.136 | 0.015 | 0.023 |
| 16x4 | structured_w1 | 0.903 | 0.091 | 0.759 | 0.014 | 0.023 |
| 16x4 | structured_w8 | 0.761 | 0.091 | 0.617 | 0.015 | 0.022 |
| 20x2 | inverse_walk | 13.511 | 4.976 | 8.228 | 0.211 | 0.039 |
| 20x2 | structured_w1 | 7.315 | 0.612 | 6.404 | 0.211 | 0.038 |
| 20x2 | structured_w8 | 6.543 | 0.612 | 5.635 | 0.212 | 0.038 |
| 24x1 | inverse_walk | 154.148 | 53.702 | 95.336 | 4.347 | 0.673 |
| 24x1 | structured_w1 | 76.497 | 6.834 | 64.390 | 3.945 | 0.657 |
| 24x1 | structured_w8 | 69.129 | 6.797 | 57.193 | 3.948 | 0.661 |

定量结论：

- Forward 是最直接的收益来源。24x1 中 forward 从 53.70 ms 降到 6.80 ms，约 7.9x。
- `structured_w1` 已经证明 per-wire `RY+RZ` fusion + ring-CNOT layer fusion 有明显收益。
- `structured_w8` 的额外收益主要来自 backward rotation chunk。24x1 backward 从 64.39 ms 进一步降到 57.19 ms。
- Hamiltonian 和 dot 在 structured_adjoint/inverse_walk 中基本相同，说明主要优化确实来自 circuit application 和 adjoint sweep，而不是 benchmark 偏差。

## GPU 利用率

Telemetry 结果：

| case | avg step ms | peak GPU util | avg SM util | peak SM util | peak process memory |
|---|---:|---:|---:|---:|---:|
| 20x2 structured_w8 | 6.815 | 100.0% | 14.0% | 14.0% | 494 MiB |
| 24x1 structured_w8 | 69.400 | 100.0% | 91.1% | 100.0% | 1454 MiB |

解释：

- 20x2 单步较短，1 秒采样粒度下 telemetry 容易低估 SM 平均利用率；peak GPU util 已达到 100%。
- 24x1 运行时间足够长，avg SM util 约 91%，peak SM util 100%，说明大规模下 GPU 已经被持续喂满。
- 24x1 workspace 约 1.0 GiB，实测 process memory peak 约 1454 MiB，显存占用合理。

## Nsight Compute Kernel 数据

24x1 `structured_w8` 的 Nsight Compute 汇总：

| kernel group | launches | median time (us) | SM throughput | active warps | DRAM throughput | L2 hit | grid x block |
|---|---:|---:|---:|---:|---:|---:|---|
| backward_rotation_chunk_w8 | 15 | 17679.4 | 84.4% | 33.1% | 6.7% | 74.4% | 65536 x 256 |
| forward_rotation_chunk_w4_register | 30 | 912.5 | 83.1% | 29.7% | 62.0% | 50.0% | 10923 x 96 |
| hamiltonian | 5 | 4254.3 | 84.0% | 98.2% | 35.1% | 79.0% | 65536 x 256 |
| ring_cnot_layer | 15 | 1532.9 | 0.8% | 76.0% | 89.5% | 40.9% | 65536 x 256 |
| inner_product_reduce | 10 | 308.4 | 12.0% | 56.8% | 49.3% | 33.5% | 2130 x 256 |
| zero_state | 5 | 273.0 | 2.2% | 84.7% | 96.4% | 100.0% | 65536 x 256 |

这组数据说明：

- `backward_rotation_chunk_w8` 是当前最大热点。它的 SM throughput 高，但 active warps 只有约 33%，符合 chunk kernel 使用较多 shared memory/同步和寄存器状态的特征。它不是完全 occupancy-bound，也不是单纯 DRAM-bound。
- `forward_rotation_chunk_w4_register` 的 SM throughput 约 83%，DRAM throughput 约 62%，说明 forward rotation 已经比较有效地把计算和读写合在一起。
- `ring_cnot_layer` active warps 高、DRAM throughput 高、SM throughput 低，符合 pure permutation/scatter 的访存主导特征。这里进一步优化应关注写入 coalescing、gather/scatter 方向、或分块置换，而不是增加算术强度。
- Hamiltonian kernel active warps 和 SM throughput 都较高，不是当前最优先瓶颈。

## 为什么 Structured Adjoint 能加速

定性上，structured_adjoint 减少了三个主要成本：

1. Kernel launch 数量
   门级实现每个 `RY`、`RZ`、`CNOT` 都启动 kernel。structured_adjoint 对 rotation layer 和 CNOT layer 做结构化合并，launch 数显著下降。

2. Global memory traffic
   单独执行 `RY` 再执行 `RZ` 会对同一组振幅读写两次。fused `RY+RZ` 和 rotation chunk 在寄存器/shared memory 中连续完成多步局部变换，减少 global read/write。

3. 更适合 statevector layout
   低位 qubit 的 partner amplitude 相邻，适合 warp shuffle/shared-memory tile。高位 qubit stride 大，structured_adjoint 避免对窄高位 chunk 过度融合，防止 cooperative chunk 的同步和访存代价超过收益。

此外，CNOT layer 作为 permutation 可以一次性计算目标 index。当前实现把 ring-CNOT 的逐 bit 模拟换成闭式 GF(2) 索引变换，减少每个线程内的分支和循环。

## 当前瓶颈与后续方向

1. Backward rotation chunk 是最大热点
   24x1 中 `inverse_walk_ryrz_rotation_chunk_kernel<8>` 单次约 17.7 ms。后续可以继续研究：
   - 更低 shared memory 占用的分块方式。
   - warp-level reduction 替代部分 shared-memory reduction。
   - 把梯度 reduction 分两阶段，减少 atomic 压力。

2. Ring-CNOT layer 是访存主导
   当前 CNOT permutation 的 DRAM throughput 高但 SM throughput 低。后续优化方向：
   - 尝试 gather 方向保证写入连续，比较 read scatter vs write scatter。
   - 对低位局部 permutation 使用 shared-memory block transpose。
   - 对 ring-CNOT 的线性变换预计算/分段以减少随机访存。

3. `ryrz_fused` 对照表明 per-wire fusion 仍值得保留
   它比 inverse_walk 快 1.3x-1.8x，但仍明显慢于 structured_adjoint。它适合作为 regression baseline，用于判断后续 structured_adjoint 改动是否真来自 layer/chunk 结构优化。

4. 小规模 telemetry 粒度不足
   20x2 的单步约 6.8 ms，用 1 秒粒度的 `nvidia-smi dmon` 只能看 peak，不能准确反映短 kernel 的 SM 时间分布。小规模应更多依赖 CUDA event stage profile 和 Nsight Compute。

## 后续 Backward Kernel 优化

2026-07-01 继续针对 backward rotation 做了两项实现级优化：

1. 将 fused backward `RY+RZ` 的 `thrust::complex` 临时对象展开为 real/imag 标量公式。
   - `phi` 梯度中 `RZ_forward * RZ_inverse = I`，因此可以直接由 `current_after` 和 `lambda_after` 计算，省去多次复数乘法。
   - 同样的标量公式应用到 per-wire fused backward kernel 和 rotation chunk backward kernel。
2. 为 24 qubit 及以上的 W8 backward chunk 增加 pair512 路径，每个 block 处理两个 256-amplitude tile，减少 block reduction 和 atomicAdd 数量；小中规模继续使用普通 W8/W4 路径，避免 shared footprint 变大导致退化。

更新后的 stage profile：

| case | mode | total | forward | backward | Hamiltonian | dot |
|---|---|---:|---:|---:|---:|---:|
| 8x8 | structured_w8 | 0.413 | 0.063 | 0.310 | 0.006 | 0.019 |
| 12x8 | structured_w8 | 0.602 | 0.097 | 0.457 | 0.008 | 0.023 |
| 16x4 | structured_w8 | 0.598 | 0.098 | 0.444 | 0.016 | 0.023 |
| 20x2 | structured_w8 | 4.732 | 0.658 | 3.788 | 0.226 | 0.039 |
| 24x1 | structured_w8 | 44.882 | 6.406 | 33.560 | 3.949 | 0.652 |

相对本报告前一版 stage profile：

| case | old structured_w8 backward | new structured_w8 backward | speedup |
|---|---:|---:|---:|
| 12x8 | 0.656 | 0.457 | 1.44x |
| 20x2 | 5.635 | 3.788 | 1.49x |
| 24x1 | 57.193 | 33.560 | 1.70x |

训练级比较：

| case | inverse_walk ms/step | ryrz_fused ms/step | structured_adjoint ms/step | structured_adjoint vs inverse |
|---|---:|---:|---:|---:|
| 12x8 | 3.073 | 1.777 | 0.666 | 4.62x |
| 20x2 | 14.229 | 6.105 | 4.411 | 3.23x |
| 24x1 | 151.022 | 95.359 | 47.585 | 3.17x |

更新后的 24x1 Nsight Compute 指标：

| kernel group | old time (us) | scalar time (us) | final time (us) | final SM throughput | final active warps | final DRAM throughput |
|---|---:|---:|---:|---:|---:|---:|
| backward_rotation_chunk_w8 | 17679.4 | 10692.8 | 10481.8 | 83.5% | 32.9% | 10.8% |

解释：

- active warp 没有明显上升，SM throughput 也基本保持在 83%-84%；性能提升来自每个 tile 的指令量下降，而不是 occupancy 提升。
- 这说明原 kernel 的主要问题不是 GPU 空转，而是复数表达式生成了多余的中间计算。手写 real/imag 后，同样的 SM throughput 执行了更少工作。
- pair512 对 24 qubit 有小幅收益，但对 12-20 qubit 会退化，因此只在 `state_size >= 2^24` 时启用。


## 再次优化：高位 Transposed Backward Chunk

2026-07-01 又继续针对 backward rotation chunk 的高位访存做了分段策略优化。

问题来源：W8 cooperative/pair512 chunk 对低位 qubit 很合适，但在高位 chunk 上，一个 block 内相邻线程会访问 `local_index << chunk_start` 形成的大 stride 地址，global load/store 不连续。Forward 侧已用 W4 register kernel 避开了类似问题；backward 由于同时需要 `current`、`lambda` 和梯度累加，完整 register tile 容易带来过高寄存器压力。

本轮保留 shared-memory tile，但新增 W2-W4 transposed cooperative kernel：

- 普通 mapping: 同一 tile 内 `local_index` 随 lane 连续变化，适合低位，遇到高位时 global 地址大 stride。
- Transposed mapping: 同一 warp 先跨多个 tile 固定 `local_index` 读写，使高位 chunk 的 global load/store 对相邻 tile 更连续；进入 shared memory 后仍按 tile 内局部索引执行同样的 backward 更新和梯度规约。
- 该轮策略: 对 `num_qubits >= 20` 且请求 W8 的 backward，8 号以上高位拆成 W2-W4 transposed chunk，低位 0-7 保留 W8 chunk。all-W4 也测试过，低位额外 kernel 开销超过访存收益。

A/B stage profile，单位 ms：

| strategy | 20x2 backward | 24x1 backward | 说明 |
|---|---:|---:|---|
| scalar + pair512 W8 | 3.788 | 33.560 | 上一轮最终实现 |
| high W4 above wire 16 | 3.313 | 31.774 | 只拆最高位段 |
| high W4 above wire 8 | 2.823 | 29.848 | 该轮最优策略 |
| all W4 | 3.319 | 32.246 | 低位拆太碎，launch/同步成本回升 |

该轮 stage profile：

| case | structured_w8 total | forward | backward | Hamiltonian | dot |
|---|---:|---:|---:|---:|---:|
| 8x8 | 0.414 | 0.063 | 0.310 | 0.006 | 0.019 |
| 12x8 | 0.557 | 0.090 | 0.423 | 0.007 | 0.022 |
| 16x4 | 0.554 | 0.091 | 0.411 | 0.014 | 0.023 |
| 20x2 | 3.707 | 0.612 | 2.823 | 0.211 | 0.038 |
| 24x1 | 41.156 | 6.402 | 29.848 | 3.944 | 0.651 |

端到端 strategy compare：

| case | inverse_walk ms/step | ryrz_fused ms/step | structured_adjoint ms/step | structured_adjoint vs inverse |
|---|---:|---:|---:|---:|
| 12x8 | 2.798 | 1.613 | 0.564 | 4.96x |
| 20x2 | 12.695 | 6.102 | 4.035 | 3.15x |
| 24x1 | 151.271 | 94.203 | 43.887 | 3.45x |

该轮 24x1 Nsight Compute 汇总：

| kernel group | launches | mean time (us) | SM throughput | active warps | DRAM throughput | L2 throughput |
|---|---:|---:|---:|---:|---:|---:|
| backward_transposed_w4 | 24 | 4269.5 | 83.4% | 65.6% | 26.7% | 4.7% |
| backward_pair512_w8 | 6 | 10420.9 | 84.0% | 32.9% | 11.0% | 1.9% |
| forward_w4_register | 42 | 914.5 | 83.1% | 29.6% | 60.0% | 16.2% |
| ring_cnot_layer | 20 | 1188.2 | 1.7% | 77.6% | 89.8% | 26.2% |
| hamiltonian | 7 | 4255.7 | 84.0% | 98.3% | 34.1% | 19.2% |

解释：

- 高位 transposed W4 的 active warps 约 65.6%，显著高于 W8 pair512 的 32.9%，且 SM throughput 仍维持在 83%-84%。这说明高位拆分不是单纯提高 occupancy，而是改善了高位 chunk 的访存与 tile 并行形态。
- 该轮低位仍保留 W8 pair512，因为低位相关振幅更接近，W8 减少 kernel launch 和 global round-trip 的收益更大。all-W4 的 A/B 结果验证了这一点。
- 之前尝试过 warp-level reduction，把 pair512 的 shared partial 从 49KB 降到 17KB，但 24x1 backward 反而从约 33.6ms 退到约 35.8ms；说明当前 pair512 更受每 tile 指令路径/同步结构影响，盲目提高 active warp 不一定收益。

新增原始数据文件：

- `benchmarks/results/codex_backward_transposed_final_stages.csv`
- `benchmarks/results/codex_backward_transposed_final_ncu_24x1_kernel_profile.csv`
- `benchmarks/results/codex_backward_transposed_final_ncu_24x1_runs.csv`

## 再进一步：Two-Wire Cell Backward Chunk

2026-07-02 继续验证“相邻两个 wire 合成一个 4-amplitude cell，在寄存器里连续完成两个 `RY+RZ` backward”的想法。实现上没有构造 dense 4x4 矩阵，而是把同一个 4-amplitude cell 内的两次 2x2 更新连续执行：

```text
wire hi: (00, 10), (01, 11)
wire lo: (00, 01), (10, 11)
```

这样数学顺序仍然等价于原来的逐 wire inverse walk，但省掉了两个 wire 之间的 shared-memory round trip 和 `__syncthreads()`。同时，一个 block 处理更多 cell 后，block-level reduction 和 atomicAdd 次数也下降。

实现与调参结果：

- W4 high-wire chunk: 使用 128-thread cell2 kernel，每个 block 处理 32 个 16-amplitude tile。ptxas: 62 registers, 24KB shared, no spill。
- W8 low-wire chunk: 24 qubit 以上使用 128-thread cell2 kernel，每个 block 处理 2 个 256-amplitude tile。ptxas: 78 registers, 32KB shared, no spill。
- W8 曾尝试 256-thread/4-tile 版本，但 static shared memory 需要 64KB，超过当前编译目标默认 48KB 上限，因此没有保留。
- W4 也测试过 256-thread/64-tile 版本；24q 和 20q 均略慢于 128-thread 版本，因此最终保留 128-thread。

A/B stage profile，单位 ms：

| strategy | 20x2 backward | 24x1 backward | 说明 |
|---|---:|---:|---|
| high transposed W4 + low pair512 W8 | 2.823 | 29.848 | 上一轮最优 |
| W4 cell2 + low pair512 W8 | 2.764 | 25.977 | 只替换高位 W4 |
| W4 cell2 + W8 cell2 | 2.687 | 22.366 | 当前最终策略 |

当前最终 stage profile：

| case | structured_w8 total | forward | backward | Hamiltonian | dot |
|---|---:|---:|---:|---:|---:|
| 8x8 | 0.413 | 0.063 | 0.310 | 0.006 | 0.019 |
| 12x8 | 0.600 | 0.097 | 0.457 | 0.008 | 0.023 |
| 16x4 | 0.598 | 0.098 | 0.444 | 0.016 | 0.024 |
| 20x2 | 3.699 | 0.658 | 2.687 | 0.227 | 0.039 |
| 24x1 | 33.711 | 6.428 | 22.366 | 3.959 | 0.651 |

端到端 strategy compare：

| case | inverse_walk ms/step | ryrz_fused ms/step | structured_adjoint ms/step | structured_adjoint vs inverse |
|---|---:|---:|---:|---:|
| 12x8 | 3.011 | 1.655 | 0.617 | 4.88x |
| 20x2 | 12.828 | 6.127 | 3.511 | 3.65x |
| 24x1 | 152.759 | 94.314 | 36.226 | 4.22x |

最终 24x1 Nsight Compute 汇总：

| kernel group | launches | mean time (us) | SM throughput | active warps | DRAM throughput | L2 throughput |
|---|---:|---:|---:|---:|---:|---:|
| backward_w4_cell2 | 24 | 3229.1 | 83.9% | 32.9% | 35.1% | 6.1% |
| backward_w8_cell2 | 6 | 6427.7 | 84.2% | 24.7% | 17.7% | 3.1% |
| forward_w4_register | 42 | 911.8 | 83.1% | 29.6% | 60.2% | 16.2% |
| ring_cnot_layer | 20 | 1190.6 | 1.7% | 77.7% | 89.7% | 23.9% |
| hamiltonian | 7 | 4246.2 | 84.0% | 98.2% | 34.3% | 19.3% |

解释：

- cell2 并没有提高 active warp；W8 cell2 active warps 甚至只有约 24.7%。但 SM throughput 仍在 84% 左右，说明性能来自更少的同步、shared round trip、block reduction 和 atomicAdd，而不是 occupancy 数字本身。
- W4 cell2 相比上一轮 transposed W4，单 kernel mean time 从约 4269.5 us 降到 3229.1 us。
- W8 cell2 相比上一轮 pair512，单 kernel mean time 从约 10420.9 us 降到 6427.7 us。
- 这验证了用户提出的“两门合并成 4-amplitude cell”方向：只要不用 generic dense 4x4，而是用结构化寄存器内两次 2x2，就能减少同步成本且不引入额外数学开销。
- 另一个未保留的试验是把 backward 的 `current` / `lambda` 两次 inverse ring-CNOT 合成一个双 statevector permutation kernel。它在 20x2 基本持平，但 24x1 backward 从约 22.3 ms 退化到约 24.7 ms；原因很可能是单线程内存事务加重后降低了 permutation kernel 的有效吞吐，少一次 launch 抵不过内存侧损失。

新增原始数据文件：

- `benchmarks/results/codex_cell2_final_stages.csv`
- `benchmarks/results/codex_cell2_final_ncu_24x1_kernel_profile.csv`
- `benchmarks/results/codex_cell2_final_ncu_24x1_runs.csv`
- `benchmarks/results/codex_pair_cnot_stages.csv`

## 复现实验命令

正确性：

```bash
.venv/bin/python -m unittest tests.test_backends_parity -v
```

主实验脚本是本次报告内联运行的 `energy_and_grad` timing，输出到：

```bash
benchmarks/results/codex_main_energy_grad_times.csv
```

Stage profile：

```bash
.venv/bin/python benchmarks/profile_baseline_stages.py \
  --cases 8x8 12x8 16x4 20x2 24x1 \
  --warmup 3 --repeats 20 \
  --structured-widths 1 8 \
  --csv-out benchmarks/results/codex_doc_stage_profile.csv
```

Telemetry：

```bash
.venv/bin/python benchmarks/run_standalone_gpu_telemetry_grid.py \
  --cases 20x2 24x1 \
  --modes structured_adjoint \
  --structured-widths 8 \
  --min-telemetry-samples 3 \
  --max-repeat-runs 2 \
  --telemetry-interval 1.0 \
  --allow-incomplete-telemetry \
  --out-dir benchmarks/results \
  --prefix codex_doc_gpu_telemetry
```

Nsight Compute：

```bash
.venv/bin/python benchmarks/run_standalone_gpu_telemetry_grid.py \
  --cases 24x1 \
  --modes structured_adjoint \
  --structured-widths 8 \
  --min-telemetry-samples 1 \
  --max-repeat-runs 1 \
  --telemetry-interval 1.0 \
  --allow-incomplete-telemetry \
  --kernel-profile-ncu \
  --ncu-launch-count 80 \
  --ncu-profile-count 1 \
  --allow-incomplete-kernel-profile \
  --out-dir benchmarks/results \
  --prefix codex_doc_ncu_24x1
```
