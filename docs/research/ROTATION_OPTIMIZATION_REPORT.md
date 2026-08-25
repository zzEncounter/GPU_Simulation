# RX/RY 深入优化研究：shared mailbox、phase、persistent 与 backward

## 1. 先给结论

本报告延续 `OPTIMIZATION_REPORT.md`，只研究 RX/RY，并以 SU2-HEA、RZZ-HEA 为端到端对象。结论不是“shared memory 一定慢”或“occupancy 越高越好”，而是下面五条。

1. **warp 间 mailbox 在当前 `complex<double>` 实现中是可见瓶颈，但不能靠缩短 tile 直接删除。** 24q、相同 9-bit tile 下，把 `128 threads × 4 amplitudes/thread` 改成单 warp 的 `32×16`，完全没有 amplitude mailbox，RX/RY forward 快 8.7%–9.3%，backward 快 12.6%–14.6%。Nsight Compute 同时看到 barrier stall 从 forward/backward 的 45.98%/38.45% 降到 0.20%/1.36%。但若仍让每 thread 只持有 4 个 amplitude，tile 从 9 bits 缩到 7 bits、phase 从 5 次增到 10 次，forward 慢 72%–73%，backward 慢 49%–55%。因此“phase 变小会慢”的猜测成立，但可以用 registers 补回 tile 容量。
2. **backward 已经顺序复用一份 mailbox；继续缩 mailbox 是一个可调参数，但只有资源边界被跨过时才明显。** `64×16` 的 mailbox 从 16 KiB 减到 8 KiB 后，forward active CTA/SM 从 5 增到 8，RY forward 快 4.7%；backward 仍由 registers 限制在 4 CTA/SM，反而慢约 0.3%。`64×32` 从 32 KiB 减到 8 KiB 时，backward 从 2 增到 4 CTA/SM，RX/RY 快 9.6%/10.8%。不过 `64×32` 的绝对时间仍输给 `64×16`，不能只看相对加速。
3. **相邻 warp 的连续性有价值，但优先级低于 phase 数。** 把物理 bit 5 固定到第一个 warp bit，使两个 warp 覆盖连续 64 amplitudes；只要这导致多一个 phase，20–26q 通常慢 6.9%–20.5%。唯一正例是 `64×16, 26q`：两种布局都是 5 phases，此时 RY forward 快 0.4%，backward 快 3.3%。推荐规则不是永久固定低 6 位，而是只在不会增加 phase 时启用。
4. **本实现没有从 persistent cooperative kernel 获益；逐 phase 普通 kernel 更好。** 24q isolated compact RY 的普通 kernel 比 persistent forward/backward 快 1.49×/1.24×。8-layer SU2-HEA 在 20/24/26q 的 total 分别快 1.27×/1.11×/1.34×，RZZ-HEA 分别快 1.03×/1.13×/1.17×。额外 launch 没有抵消移除 `grid.sync()`、resident-grid 限制和部分 register pressure 的收益。
5. **backward 复用了 forward 的逆旋转 primitive，但不是“差不多同价”。** 它读写两个 state vector，先算 generator overlap 并规约 gradient，再分别对 `phi/lambda` 施加逆旋转。24q DRAM traffic 恰为 forward 的约 2 倍（2.65 vs 5.36 GB），总时间为 2.10×–2.34×。RX/RY generator 与 slot 已经用模板和 `if constexpr` 编译；把 CTA tree reduction 换成 hierarchical reduction 或 per-warp atomic 没有稳定收益。

本轮已把 **普通逐 phase kernel** 设为 complex RX/RY 默认路径；`SAD_ROTATION_PERSISTENT=1` 保留旧 cooperative 路径用于复现。下一项最值得进入完整 dispatch sweep 的候选是 **单 warp `32×16`（零 mailbox）/`64×16`（一 warp bit）按方向和电路选择**。mailbox 分块应保留成编译期可调参数，不应统一设成一半。

## 2. 实验约定

硬件与旧报告相同：NVIDIA RTX 6000 Ada Generation 48 GiB，SM 8.9，driver 595.84，CUDA 13.2；日期为 2026-08-11。所有主计时使用 float64。

- isolated rotation：每个配置在独立进程内 warmup 5 次、计时 30 次 kernel sequence，独立运行 5 次，表中取 5 个均值的 median；`phi/lambda` 为零值，但执行路径、地址、同步和指令不依赖数值。
- HEA：随机种子 42、8 layers；20/24q 为 9 个 timed steps，26q 为 5 个 timed steps，表中为 step median。
- Nsight Compute：对 warmup 后的一次 24q full-fixed RY kernel 采样。profiler replay 会改变 wall time，所以只比较 counter，不把 profiler 下的时间混入主表。
- active CTA/SM 来自 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`；该 API 返回给定 kernel、block size、dynamic shared 大小时每个 SM 的最大 active blocks，[CUDA Runtime API 对此有明确说明](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html)。

原始数据：

- [`benchmark/results/rotation_deep_dive.csv`](../../benchmark/results/rotation_deep_dive.csv)：630 个 isolated runs；
- [`benchmark/results/rotation_full_circuit.csv`](../../benchmark/results/rotation_full_circuit.csv)：SU2/RZZ 端到端数据与完整 gradient；
- [`benchmark/results/rotation_ncu.csv`](../../benchmark/results/rotation_ncu.csv)：Nsight counters；
- [`benchmark/results/rotation_reduction.csv`](../../benchmark/results/rotation_reduction.csv)：三种 gradient reduction 的独立复测。

## 3. mailbox 当前到底是什么

对一个含 `2^t` 个 amplitudes 的 tile：

- lane bits 用 `shfl_xor`，不经过 shared；
- register bits 在一个 thread 的数组内交换；
- warp bits 必须跨 warp 读 partner，当前用 CTA shared mailbox。

forward 先把一份 tile 写入 mailbox，barrier 后读取 partner，再 barrier 防止下一次交换覆盖。backward 并没有同时保留 `phi_mailbox` 和 `lambda_mailbox`：它先用 mailbox 完成 `phi` 的 overlap/逆演化，gradient reduction 的同步点之后把同一地址覆写为 `lambda`，再完成 `lambda` 的逆演化。因此用户问题中“backward 两段串行，只需 forward 一样大小 mailbox”的优化**已经做了**。

以 `128×4, complex<double>` 为例，tile 为 512 amplitudes：

```text
forward mailbox               = 512 × 16 B = 8192 B
backward amplitude mailbox    = 512 × 16 B = 8192 B
backward kernel static shared = 9232 B
```

最后一项比 8192 B 多出的部分是 gradient reduction 和 `tile_base`，不是第二份 amplitude mailbox。

本轮加入两种继续减小的方法：

1. `C`-chunk complex mailbox：容量为 `tile_bytes/C`，每个 warp gate 分 `C` 段交换；barrier 数约从 `2` 增到 `2C`。
2. RY scalar mailbox：real/imag 复用同一地址，容量减半，两个分量串行。RY 的两个分量彼此独立，可以直接做；RX 的 `new.real` 依赖 `partner.imag`、`new.imag` 依赖 `partner.real`，若复用地址还要额外保存原值，所以本轮没有把它伪装成同成本方案。

## 4. 问题一：shared mailbox 是瓶颈吗

### 4.1 完全不用 mailbox

下表都是 24q full-fixed，一个 layer 的全部 24 个 RX/RY。`32×16` 和 `128×4` 都是 9-bit tile、phase 数相同；`32×4` 是 7-bit tile。

| gate | direction | `128×4` full mailbox | `32×4` no mailbox | phases | `32×16` no mailbox | phases |
|---|---|---:|---:|---:|---:|---:|
| RX | forward | 4.852 ms | 8.366 ms（1.72×慢） | 10 | 4.400 ms（快 9.3%） | 5 |
| RX | backward | 10.785 ms | 16.661 ms（1.55×慢） | 10 | 9.422 ms（快 12.6%） | 5 |
| RY | forward | 4.823 ms | 8.356 ms（1.73×慢） | 10 | 4.403 ms（快 8.7%） | 5 |
| RY | backward | 11.308 ms | 16.819 ms（1.49×慢） | 10 | 9.661 ms（快 14.6%） | 5 |

`32×16` 的 active warps 其实更少：例如 RY backward 是 `8 CTA × 1 warp = 8 warps/SM`，而 `128×4` 是 `6 CTA × 4 warps = 24 warps/SM`。它仍更快，说明收益不是简单来自更高 warp occupancy，而是少掉 shared exchange 与 CTA barrier 后，每 tile 的工作显著减少。

另一方面，`64×16` 是 10-bit tile，只用一个 warp bit。24q full-fixed RY forward/backward 为 3.992/8.827 ms，仍优于零-mailbox `32×16` 的 4.403/9.661 ms。也就是说最佳点不是“shared 数为零”，而是 **warp bits、register bits、phase 数和同步成本的联合最优**。

### 4.2 profiler 证据

NVIDIA 文档说明 Ada 这类设备的 shared memory 有 32 个 banks，连续 32-bit word 映射到连续 bank；同一 warp 的 bank conflict 会串行化请求并降低吞吐（[CUDA Programming Guide: Shared Memory Access Patterns](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory-access-patterns)）。`complex<double>` 为 16 bytes，mailbox 的宽访问本身就给 shared pipeline 很大压力。

| variant | direction | DRAM | shared load/store wavefronts | shared load/store bank conflicts | barrier stall |
|---|---|---:|---:|---:|---:|
| `128×4` mailbox | forward | 2.65 GB | 77.0M / 76.2M | 56.8M / 56.7M | 45.98% |
| `32×16` no mailbox | forward | 2.64 GB | 2.81M / 0.17M | 0.021M / 0.001M | 0.20% |
| `128×4` mailbox | backward | 5.36 GB | 170.4M / 167.1M | 113.2M / 113.2M | 38.45% |
| `32×16` no mailbox | backward | 5.36 GB | 11.5M / 5.77M | 0.103M / 0.037M | 1.36% |

零-mailbox kernel 仍有少量 shared counter，来自 `tile_base` 和 backward reduction，并不是 amplitude exchange。两边 DRAM bytes 几乎完全相同，而 barrier stall 大幅下降；Nsight 把该 stall 定义为等待同一 CTA sibling warps 到达 barrier（[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-stalls-per-warp-id)）。结合相同 tile 的时间对照，可以确认 mailbox/barrier 是实际瓶颈之一。

### 4.3 完整 HEA

24q、8 layers 的 exact-shape 结果如下。这里的 `128×4` 与 `32×16` 都是 persistent，只隔离 mailbox/shape；`64×16` 多一个 tile bit。

| circuit / variant | forward | backward | total |
|---|---:|---:|---:|
| SU2 `128×4` | 31.247 ms | 106.580 ms | 139.861 ms |
| SU2 `32×16`, no mailbox | 32.581 ms | 89.837 ms | **124.065 ms** |
| SU2 `64×16` | 33.505 ms | 89.706 ms | 125.072 ms |
| RZZ `128×4` | 35.840 ms | 116.827 ms | 154.978 ms |
| RZZ `32×16`, no mailbox | 33.877 ms | 99.486 ms | **135.918 ms** |
| RZZ `64×16` | 38.936 ms | 111.790 ms | 152.491 ms |

零-mailbox `32×16` 令 SU2/RZZ total 分别快 12.7%/12.3%。SU2 forward 略退化，但 backward 的收益占主导。这也证明 isolated 结果能落到 HEA，而不是只对空 state 的微基准成立。

## 5. 问题二：减小 mailbox 换 active CTA

### 5.1 isolated 数据

下表给出 24q full-fixed。括号是 active CTA/SM；scalar 只适用于 RY。

| shape / gate / direction | full | half | quarter | scalar real/imag reuse |
|---|---:|---:|---:|---:|
| `64×16` RX F | 3.936 (5) | 3.881 (8) | **3.858 (8)** | — |
| `64×16` RX B | **8.249 (4)** | 8.277 (4) | 8.275 (4) | — |
| `64×16` RY F | 3.992 (5) | **3.805 (8)** | 3.841 (8) | 3.808 (8) |
| `64×16` RY B | **8.827 (4)** | 8.856 (4) | 8.898 (4) | 9.100 (4) |
| `64×32` RX F | 4.058 (3) | 3.967 (4) | **3.900 (4)** | — |
| `64×32` RX B | 11.586 (2) | 10.810 (4) | **10.475 (4)** | — |
| `64×32` RY F | 3.977 (3) | 3.896 (4) | **3.854 (4)** | 3.861 (4) |
| `64×32` RY B | 11.653 (2) | 10.606 (4) | **10.390 (4)** | 10.707 (4) |

三种情形很清楚：

- **CTA 不变：** `64×16` backward 一直是 4 CTA/SM；增加分段只有同步成本，full 最好。
- **跨过 occupancy 台阶：** `64×16` forward 从 5 到 8 CTA/SM，half/quarter 小幅获益；`64×32` backward 从 2 到 4 CTA/SM，收益达到 9.6%–10.8%。
- **已经跨过台阶后继续减：** half 到 quarter 没再增加 CTA，但 `64×32` 仍略快，说明更小 working set/cache 行为也有贡献；收益已明显递减，不能外推为“越小越好”。

real/imag 地址复用对 RY forward 与 complex-half 几乎一样，但 RY backward 慢 2.7%（`64×16`），因为 `phi.real/imag`、`lambda.real/imag` 都要串行并增加 barrier。它没有优于更通用的 complex chunking。

### 5.2 端到端不只看 isolated occupancy

| 24q / 8 layers | SU2 total | RZZ total |
|---|---:|---:|
| `64×16` full | 124.526 ms | 152.927 ms |
| `64×16` half | 124.751 ms | **144.079 ms** |
| `64×32` full | 171.076 ms | 170.511 ms |
| `64×32` half | 150.850 ms | 170.093 ms |
| `64×32` quarter | 152.582 ms | 163.767 ms |

RZZ fused kernel 还有 diagonal gradient shared storage，half `64×16` 在完整路径中带来 5.8% total 收益；SU2 则基本持平。`64×32` 虽然相对自己的 full mailbox 明显改善，仍因 255 registers/thread 和 spill 风险输给 `64×16`。

因此建议把 `mailbox_chunks ∈ {1,2,4}` 作为 **shape × direction × circuit** 的编译期调参维度：

- 资源不跨 occupancy 台阶时优先 `1`；
- 跨台阶时测试 `2` 和 `4`；
- 选择必须看绝对时间，不能只看同 shape 的 speedup；
- real/imag 复用不需要进入默认候选集。

## 6. 问题三：不同 warp 之间的连续性

新布局 `full-pairs` 在后续高位 phase 中做如下映射：

```text
lane slots       <- physical bits 0..4
first warp slot  <- physical bit 5
register slots   <- high rotation targets
remaining warp slots, if any <- high rotation targets
```

这样 warp 0/1、warp 2/3 分别覆盖连续 64-amplitude 区间，但牺牲一个 high target slot。对照仍是只固定低 5 位的 `full-fixed`。

| shape | q | fixed-low-5 phases | pair-contiguous phases | RY forward ratio | RY backward ratio |
|---|---:|---:|---:|---:|---:|
| `128×4` | 20 | 4 | 5 | 1.111×慢 | 1.088×慢 |
| `128×4` | 24 | 5 | 6 | 1.159×慢 | 1.095×慢 |
| `128×4` | 26 | 6 | 7 | 1.131×慢 | 1.069×慢 |
| `64×16` | 20 | 3 | 4 | 1.205×慢 | 1.071×慢 |
| `64×16` | 24 | 4 | 5 | 1.177×慢 | 1.137×慢 |
| `64×16` | 26 | 5 | 5 | **快 0.4%** | **快 3.3%** |

24q RX 也重复同一趋势：多一个 phase 时，`128×4` forward/backward 慢 15.8%/11.8%，`64×16` 慢 20.3%/16.6%。

结论是 inter-warp 连续性确实有二级收益；`64×16, 26q` 在 phase 数不变时给出稳定正例。但一次额外 full-state pass 远比它贵。可实现的 dispatch 规则是：只有在固定 bit 5 后 `phase_count` 不变，或短尾 phase 能吸收这个固定 bit 时，才考虑 pair-contiguous；否则继续 fixed-low-5。

## 7. 问题四：为什么原来用 persistent，不用会怎样

### 7.1 原选择的逻辑

相邻 phase 会改变 physical-qubit 到 tile slot 的映射。phase `p+1` 不能在所有 CTA 完成 phase `p` 前开始，否则会读到新旧 state 混合值。原实现把 resident CTA 固定成 cooperative grid，在一个 kernel 内循环所有 tiles/phases，并在 phase 间 `grid.sync()`：

```text
one cooperative launch
  phase 0: resident CTAs grid-stride over all tiles
  grid.sync()
  phase 1: resident CTAs grid-stride over all tiles
  ...
```

这能省 kernel launches，并保持 layer fusion。CUDA 文档确认 grid-wide synchronization 需要 cooperative groups/launch，且 sync 前的访问对 sync 后组内线程可见（[Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)）。因此该设计在语义上合理。

替代实现是一 phase 一个普通 kernel。同一 stream 的 kernel 边界提供全局顺序：

```text
ordinary launch phase 0 over all tile CTAs
ordinary launch phase 1 over all tile CTAs
...
```

它保留 in-place tile 更新和最后 phase 的 diagonal/CNOT fusion，只增加 launch 次数。

### 7.2 isolated 结果

| q / layout / direction | persistent RY | ordinary per-phase RY | speedup |
|---|---:|---:|---:|
| 12 / compact / F | 0.0158 ms | 0.0147 ms | 1.08× |
| 12 / compact / B | 0.0328 ms | 0.0312 ms | 1.05× |
| 20 / compact / F | 0.214 ms | 0.154 ms | 1.39× |
| 20 / compact / B | 0.548 ms | 0.537 ms | 1.02× |
| 24 / compact / F | 4.591 ms | 3.089 ms | 1.49× |
| 24 / compact / B | 11.829 ms | 9.507 ms | 1.24× |
| 24 / fixed-low / F | 4.882 ms | 4.051 ms | 1.21× |
| 24 / fixed-low / B | 11.371 ms | 9.632 ms | 1.18× |

即使 12q 很短，普通 launch 也没有输。24q compact 的差距最大；fixed-low phase 更多且连续访问已经改善，普通 kernel 仍稳定获益。

### 7.3 完整 HEA

| circuit | q | persistent forward → ordinary | persistent backward → ordinary | total speedup |
|---|---:|---:|---:|---:|
| SU2 | 20 | 2.906 → 1.321 ms | 5.108 → 4.951 ms | 1.27× |
| SU2 | 24 | 31.200 → 21.875 ms | 105.534 → 101.339 ms | 1.11× |
| SU2 | 26 | 270.815 → 103.748 ms | 463.231 → 441.563 ms | 1.34× |
| RZZ | 20 | 1.748 → 1.609 ms | 6.086 → 5.983 ms | 1.03× |
| RZZ | 24 | 35.179 → 26.432 ms | 114.799 → 105.542 ms | 1.13× |
| RZZ | 26 | 200.898 → 125.201 ms | 506.849 → 480.419 ms | 1.17× |

可观察到三个来源：

- 去掉每 phase 的 `grid.sync()`；
- 普通 grid 可以投放全部 tile CTAs，由硬件持续调度，不要求整个 grid 同时 resident，tail/wave 调度更自由；
- 编译结果也变化：例如 24q RX forward/backward 从 55/74 registers 降到 46/68，active CTA 从 9/6 增到 10/7。RY 的资源数基本不变仍获益，说明 register 不是唯一原因。

后两项对总收益的精确占比没有单独拆出，因此它们是结合 source/attributes 的解释，不应写成独立测量结论。实测足以否定“persistent 必然因少 launch 更快”。本轮提交已把普通逐 phase kernel 设为 complex RX/RY 默认；后续仍应补齐 4–28q 的性能 sweep，并单独研究仍使用自身 kernel 的 RA real path 和 XXZ bond path。

## 8. 问题五：backward 与 forward 的关系

### 8.1 数学与执行顺序

设当前 gate 为

```text
U(theta) = exp(-i theta G / 2)
```

进入 backward gate 时，`phi` 和 `lambda` 都在该 gate 的输出端。当前实现的顺序是：

```text
partner_phi = exchange(phi, target)
local_overlap += generator_overlap(lambda, partner_phi)
reduce local_overlap -> gradient[theta]

phi    = U(theta)^dagger * phi
lambda = U(theta)^dagger * lambda
```

所以“先内积，再各自乘法/旋转”的理解是对的。细节上不是先构造完整 `G|phi>` state vector，而是每个 thread 用 partner amplitude 直接累计 `Im(<lambda|G|phi>)`；之后才对两条 state 分别施加同一个 inverse 2×2 rotation。

同轴、不同 qubit 的 generators 对易，因此一整个 phase 的 gradients 理论上可以在同一个端点求。但要真正获益，还需批量保存 partials，并把多个 warp-bit partner 访问和两条逆演化重排；这不是删除一行同步就能得到的优化。

### 8.2 为什么是约 2.1×–2.34×，而不是严格 2×

| shape / gate | forward | backward | B/F |
|---|---:|---:|---:|
| `128×4` RX | 4.852 ms | 10.785 ms | 2.22× |
| `128×4` RY | 4.823 ms | 11.308 ms | 2.34× |
| `64×16` RX | 3.936 ms | 8.249 ms | 2.10× |
| `64×16` RY | 3.992 ms | 8.827 ms | 2.21× |

基础的 2× 来自 `phi/lambda` 两条 global state 和两套 inverse rotation。超过 2× 的部分来自 generator partner、double gradient accumulation、block/atomic reduction，以及 backward 更高的 register pressure。Nsight 的 2.65/5.36 GB DRAM 数据与这一解释一致。

forward/backward 的参数本来就可以不同，当前也已经分离：`SAD_FORWARD_BLOCK_THREADS/REGISTER_BITS` 与 `SAD_BLOCK_THREADS/REGISTER_BITS` 独立编译，Python dispatch 的 variant 名也分别带 `f..._b...`。本轮结果进一步说明 direction-specific mailbox chunks 和单-warp选择是合理的。

### 8.3 generator 是否已经模板编译

已经是。关键层次均为编译期参数：

```cpp
template <typename T, NonDiagonalGate Gate, int Slot>
apply_tile_gate_backward(...)

if constexpr (Gate == NonDiagonalGate::RX) { ... }
else { ... }
```

- `Gate` 决定 RX/RY generator 和 rotation 公式，kernel 内没有 RX/RY runtime branch；
- `Slot` 决定 lane/register/warp exchange primitive；
- `T` 决定 float/double；
- angle 的 sine/cosine 是运行前预计算的数据，角度本身当然不能模板常量化。

### 8.4 两个“明显 reduction 优化”实测并不明显

本轮又比较：原 shared-memory CTA tree、warp-shuffle 后由一个 warp 汇总的 hierarchical reduction、每个 warp 直接 global atomic。下表是独立复测批次的 24q full-fixed backward；绝对值只在本表内比较。

| shape / gate | CTA tree | hierarchical | per-warp atomic |
|---|---:|---:|---:|
| `128×4` RX | **10.268 ms** | 10.347 ms | 10.477 ms |
| `128×4` RY | **10.778 ms** | 10.874 ms | 10.834 ms |
| `64×16` RX | **7.945 ms** | 8.039 ms | 7.952 ms |
| `64×16` RY | 8.501 ms | 8.473 ms | **8.414 ms** |

只有 `64×16` RY 的 per-warp atomic 快约 1.0%；同 shape RX 持平，`128×4` 则退化。更多 atomics、同步位置变化和 register lifetime 抵消了少几次 CTA reduction barrier，所以不作为默认优化。

仍值得后续研究、但不能在当前证据下宣称完成的方向有：

- ordinary per-phase kernel 下为每种 phase mask 生成更窄的专用 kernel，进一步去掉 runtime mask 与无关 slot code；
- 只在 phase 数不变时结合 `32×16` zero-mailbox 或 pair-contiguous 映射；
- 对 commuting generators 批量 overlap，再合并 inverse transform，评估是否能复用一次 mailbox snapshot；
- 针对 RZZ fused backward 单独扫描 mailbox chunks，因为它与 SU2 的最优点不同。

## 9. 正确性与建议的落地顺序

所有端到端 variant 与 exact-shape `128×4` 基准逐元素比较：最大 energy absolute difference 为 `2.67e-15`，最大 gradient element absolute difference 为 `1.40e-14`。普通 kernel 与零-mailbox custom library 跑 `sad/tests` 时，所有数值/路径测试通过；唯一 pytest failure 是测试硬编码要求 `kernel_variant == "f128r2_b128r2"`，custom library 正确返回了自己的文件名。

建议按以下顺序落地：

1. complex RX/RY 已默认使用 ordinary per-phase 路径；保留 persistent 编译开关作为回归对照。
2. 把 `32×16` zero-mailbox 加为候选，与现有 `64×16` 分 circuit/direction/q 选择；不要用 `32×4`。
3. 保留 `mailbox_chunks=1/2/4`，优先调 RZZ fused backward；只在跨过 occupancy 台阶或端到端有收益时选择分块。
4. pair-contiguous 只在 phase 数不增加时启用。
5. 不采用 scalar real/imag mailbox，也不统一替换 gradient reduction。

这几项的共同原则仍与旧报告一致：**先减少 full-state passes 和全局同步，再在不增加 phase 的前提下改善局部访问；occupancy 是约束和解释变量，不是单独的优化目标。**

## 10. 复现

```bash
# 全部 isolated ablations（约 630 rows）
.venv/bin/python benchmark/benchmark_rotation_deep_dive.py \
  --repetitions 5 --iterations 30

# 8-layer SU2/RZZ exact-shape 对照
.venv/bin/python benchmark/benchmark_rotation_full_circuit.py

# reduction 补充实验
.venv/bin/python benchmark/benchmark_rotation_deep_dive.py \
  --suite reduction-extended --repetitions 5 --iterations 30 \
  --output benchmark/results/rotation_reduction.csv

# 默认实现与回归测试
make -C sad
PYTHONPATH=sad/python:pennylane-lightning/src \
  .venv/bin/python -m pytest sad/tests pennylane-lightning/tests -q
```

Nsight 数据使用 `microbench_rotation.cu` 加 `-DSAD_ROTATION_PERSISTENT=1` 编译出的对应 shape binary，跳过 5 次 warmup 后采一个 kernel；核心 metrics 是：

```text
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{ld,st}.sum
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum
smsp__warp_issue_stalled_barrier_per_warp_active.pct
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
dram__bytes.sum
```
