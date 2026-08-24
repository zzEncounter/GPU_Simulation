# 执行策略搜索

由 `benchmark/generate_strategy_report.py` 生成。计时均取中位数；差距在 2% 以内的配置视为近似持平。

## 硬件与搜索覆盖范围

目标硬件：NVIDIA RTX 6000 Ada，计算能力 8.9，142 个 SM，每个 warp 32 个线程，每个 SM 有 65,536 个寄存器和 100 KiB 共享内存，L2 缓存为 96 MiB。

RX/RY 主搜索包含 12,406 行原始结果和 5,189 个硬件等价候选；各候选重复次数的分布为 `{1: 1570, 2: 21, 3: 3598}`。shape 阶段枚举所有能够编译且合法的 7--12 bit tile，以及 mailbox 从不切分到 64 份之间所有 2 的幂次切分；已知会超过 CUDA 资源限制的 launch 会被明确剔除。随后，针对每个场景留下的候选，在所有可达的 (lane, register, warp) 类别上用动态规划生成 phase 调度。每个生成的调度先测量一次；每个场景的前五名或距最优值 5% 以内的调度，再以打乱顺序补足三次测量。下文排名只采用拥有三次样本的调度。

融合搜索包含 1,218 个端到端样本；其中 1,218/1,218 通过了与旧实现的能量和梯度校验。

XXZ shape/分区搜索包含 4,425 个样本，每个候选均重复三次。canonical 与非均匀分区的直接对比检查通过了两种 parity：状态最大误差为 0，梯度最大误差为 1.137e-13。

自适应 mailbox 确认实验包含 39 个确实会改变执行路径的候选。每个候选按交替顺序测量九组相邻的 full/adaptive 配对；效果取配对比值的中位数，MAD 用于表示抖动。

## RX/RY phase 最优配置

| 门 | 方向 | q | variant/family | 各 phase 的 L/R/W | mailbox / CTA | 毫秒 |
|---|---:|---:|---|---|---|---:|
| RX | 反向 | 20 | `t32r2m1/compact` | `5,2,0/5,2,0/5,1,0` | 0 B / 24 | 0.425131 |
| RX | 反向 | 24 | `t64r4m16/compact` | `0,4,1/2,4,1/0,4,1/2,4,1` | 1024 B / 6 | 7.329301 |
| RX | 反向 | 26 | `t64r4m8/fixed` | `2,4,0/0,4,1/0,4,1/0,4,1/0,4,1` | 2048 B / 6 | 31.311361 |
| RX | 正向 | 20 | `t64r2m1/compact` | `3,2,1/4,2,1/4,2,1` | 4096 B / 19 | 0.145579 |
| RX | 正向 | 24 | `t256r2m2/compact` | `5,0,3/5,0,3/5,0,3` | 8192 B / 5 | 2.486613 |
| RX | 正向 | 26 | `t32r4m1/compact` | `5,4,0/5,4,0/4,4,0` | 0 B / 12 | 11.327472 |
| RY | 反向 | 20 | `t32r3m1/compact` | `3,3,0/1,3,0/3,3,0/1,3,0` | 0 B / 20 | 0.451584 |
| RY | 反向 | 24 | `t64r4m16/fixed` | `5,4,0/0,4,1/0,4,1/0,4,1` | 1024 B / 4 | 7.576235 |
| RY | 反向 | 26 | `t64r5m32/fixed` | `0,5,0/0,5,0/0,5,0/0,5,1/0,5,0` | 1024 B / 4 | 32.183979 |
| RY | 正向 | 20 | `t64r2m2/compact` | `5,2,1/2,2,1/4,2,1` | 2048 B / 16 | 0.147589 |
| RY | 正向 | 24 | `t128r5m8/compact` | `2,5,0/5,5,0/2,5,0` | 8192 B / 2 | 2.536453 |
| RY | 正向 | 26 | `t32r4m1/compact` | `5,4,0/5,4,0/4,4,0` | 0 B / 12 | 11.128150 |

低 target phase 的最优选择并不单调：正向路径通常偏好 phase 数最少的 compact 调度；反向路径有时则以较小的首尾 phase 或固定的低 lane 数量取胜。切分 mailbox 会增加 barrier，但能缩小共享内存和活跃变量范围，因此必须按门类型和方向分别评估。

## Mailbox 与 active CTA 的权衡

下表列出每个场景中提升最大的同 shape、同调度 chunked/full-mailbox 对比，从而把 mailbox 大小的影响与 phase 数量及 target 位置隔离开。

| 门 | 方向 | q | shape/family | 完整 mailbox | 分块 mailbox | CTA/SM | 耗时变化 |
|---|---:|---:|---|---:|---:|---|---:|
| RX | 反向 | 20 | `t64r5/pairs` | 32768 B | 16384 B (m2) | 2→4 | -18.35% |
| RX | 反向 | 24 | `t128r4/compact` | 32768 B | 4096 B (m8) | 2→2 | -9.93% |
| RX | 反向 | 26 | `t64r5/compact` | 32768 B | 1024 B (m32) | 2→4 | -17.75% |
| RX | 正向 | 20 | `t64r3/compact` | 8192 B | 2048 B (m4) | 10→16 | -7.34% |
| RX | 正向 | 24 | `t256r2/compact` | 16384 B | 8192 B (m2) | 5→5 | -11.82% |
| RX | 正向 | 26 | `t256r3/compact` | 32768 B | 4096 B (m8) | 3→3 | -11.68% |
| RY | 反向 | 20 | `t64r5/pairs` | 32768 B | 8192 B (m4) | 2→4 | -24.43% |
| RY | 反向 | 24 | `t64r5/fixed` | 32768 B | 8192 B (m4) | 2→4 | -16.56% |
| RY | 反向 | 26 | `t64r5/compact` | 32768 B | 2048 B (m16) | 2→4 | -21.37% |
| RY | 正向 | 20 | `t256r3/fixed` | 32768 B | 16384 B (m2) | 2→3 | -9.24% |
| RY | 正向 | 24 | `t128r4/compact` | 32768 B | 2048 B (m16) | 3→3 | -12.84% |
| RY | 正向 | 26 | `t64r5/compact` | 32768 B | 4096 B (m8) | 3→4 | -11.13% |

单 warp block 不需要 amplitude mailbox；这是一个真正的零 mailbox 方案，而不是 `m=infinity`。不过它的 tile 较小，可能增加遍历完整状态所需的 phase 数。反过来，切分多 warp tile 的 mailbox 能保持 phase map 不变，但每次 warp-target 交换大约会额外增加正向 `2m` 或反向 `4m` 个 CTA barrier。上表揭示了共享内存下降何时足以跨过 occupancy 台阶，从而补偿这些额外 barrier。

下面是逐 phase 自适应的配对结果（仅 W=0 的 phase 省略 mailbox）。单 warp/无 mailbox 以及所有 phase 均满足 W>0 的调度，已被识别为执行路径相同的对照组，因此不纳入最终配对实验。下表每行是在生成的 kernel 确有差异的候选中，自适应执行耗时最短者：

| 门 | 方向 | q | 候选 | 各 phase 的 W | CTA/SM 完整→省略 | 配对耗时变化 | 配对 MAD |
|---|---:|---:|---|---|---|---:|---:|
| RX | 反向 | 20 | `t64r4m1/pairs` | `1:0:0:0` | 5→6 | +0.11% | 7.99% |
| RX | 反向 | 26 | `t64r4m8/fixed` | `0:1:1:1:1` | 6→6 | -0.07% | 0.25% |
| RX | 正向 | 26 | `t128r5m8/compact` | `0:0:0` | 2→2 | -1.19% | 1.23% |
| RY | 反向 | 20 | `t64r5m16/fixed` | `1:1:0` | 4→4 | -5.54% | 4.98% |
| RY | 反向 | 24 | `t64r4m16/fixed` | `0:1:1:1` | 4→4 | +0.07% | 0.72% |
| RY | 反向 | 26 | `t64r5m32/fixed` | `0:0:0:1:0` | 4→4 | +0.21% | 0.20% |
| RY | 正向 | 24 | `t128r5m8/compact` | `0:0:0` | 2→2 | -1.15% | 1.46% |

## XX 与 YY 的差异及耦合边分区

| 方向 | q | 奇偶组 | 融合 XX+YY+ZZ 最优 shape | 耦合边分区 | 毫秒 |
|---|---:|---:|---|---|---:|
| 反向 | 20 | 0 | `t32r2` | `3:3:2:2` | 0.750598 |
| 反向 | 20 | 1 | `t32r2` | `3:3:2:2` | 0.749978 |
| 反向 | 24 | 0 | `t32r3` | `4:4:2:2` | 12.776448 |
| 反向 | 24 | 1 | `t32r3` | `4:4:4` | 12.848742 |
| 反向 | 26 | 0 | `t32r3` | `3:2:4:4` | 54.922035 |
| 反向 | 26 | 1 | `t32r3` | `4:4:2:3` | 54.715595 |
| 正向 | 20 | 0 | `t64r2` | `4:3:3` | 0.231782 |
| 正向 | 20 | 1 | `t64r2` | `4:2:4` | 0.231872 |
| 正向 | 24 | 0 | `t32r3` | `4:4:4` | 3.794739 |
| 正向 | 24 | 1 | `t32r3` | `4:4:4` | 3.781997 |
| 正向 | 26 | 0 | `t128r3` | `5:4:4` | 16.321945 |
| 正向 | 26 | 1 | `t128r3` | `5:3:5` | 16.335463 |

在各场景独立选择最优 shape 后，YY 比 XX 慢 1.8%--47.3%。YY 中依赖 `Z_i Z_j` 的 partner 符号会改变指令和寄存器压力；即使二者使用相同的 pair map，也不具备相同的资源特征。

RX/RY 调度的每个 slot 只计算一个 qubit target；XX/YY 则必须把每条互不相交 bond 的两个端点放入同一个 tile。因此，含 `b` 个 bit 的 tile 至多容纳 `floor(b/2)` 条 bond。XXZ 还需携带三个相互对易的系数，并在反向传播中为每条 bond 做三次梯度 reduction。因此，其 bond 组合搜索与 RX/RY 的 lane/register/warp 动态规划相互独立。

在 q=26 的 separate-matching 反向路径上，canonical 紧凑分区比非均匀分区慢 2.18%--2.94%。生产路径目前使用更快的 cross-matching kernel，所以这些 separate-kernel 分区结果用于说明分区成本，而不会直接改变生产 dispatch。

## 电路融合边界

下表的正向/反向选择采用因子化中位数：改变另一个方向的策略后将其影响取中位数消除，避免无关因素的计时噪声决定融合边界。`≈` 表示差距在 2% 以内。

| 电路 | q | 正向因子 | 反向因子 | 端到端最优方案 |
|---|---:|---|---|---|
| ra-hea | 20 | `split ≈` | `split` | `real-fused-both` |
| ra-hea | 22 | `split ≈` | `fused` | `real-fused-both` |
| ra-hea | 24 | `fused` | `fused` | `real-fused-both` |
| ra-hea | 26 | `fused` | `fused` | `real-fused-both` |
| ra-hea | 28 | `fused` | `fused` | `real-fused-both` |
| su2-hea | 20 | `lookup` | `split` | `auto` |
| su2-hea | 22 | `lookup` | `split` | `lookup-forward_split-backward` |
| su2-hea | 24 | `lookup` | `split` | `lookup-forward_split-backward` |
| su2-hea | 26 | `lookup` | `split` | `lookup-forward_split-backward` |
| su2-hea | 28 | `lookup` | `split` | `lookup-forward_split-backward` |
| rzz-hea | 20 | `fused` | `split` | `fused-forward_combined-backward` |
| rzz-hea | 22 | `fused` | `split ≈` | `fused-forward_split-backward` |
| rzz-hea | 24 | `fused` | `split` | `fused-forward_split-backward` |
| rzz-hea | 26 | `fused` | `split` | `fused-forward_split-backward` |
| rzz-hea | 28 | `fused` | `split` | `fused-forward_split-backward` |
| qaoa | 20 | `split` | `fused` | `fused-backward` |
| qaoa | 22 | `fused` | `fused` | `fused-both` |
| qaoa | 24 | `fused` | `fused` | `fused-both` |
| qaoa | 26 | `fused` | `fused` | `fused-both` |
| qaoa | 28 | `fused` | `fused` | `fused-both` |
| xxz-hva | 20 | `cross-matching` | `cross-matching` | `cross-matching` |
| xxz-hva | 22 | `cross-matching ≈` | `cross-matching` | `cross-matching` |
| xxz-hva | 24 | `cross-matching ≈` | `cross-matching` | `cross-matching` |
| xxz-hva | 26 | `cross-matching` | `cross-matching` | `cross-matching` |
| xxz-hva | 28 | `cross-matching` | `cross-matching` | `cross-matching` |

对实测融合边界的因子化解释如下：

- RA 保留实振幅快速路径。对于复振幅研究路径，正向 CNOT 融合在 22q 及以下近似持平，从 24q 起胜出；反向则从 22q 起胜出。
- SU2 正向使用 lookup 融合的 RY+RZ+CNOT；从 20q 起，反向拆分 CNOT/RZ/RY。分 phase 路径在较大 q 下明显落后。
- RZZ 正向保持将 RZ/RZZ 融入 RX，但从 20q 起将反向的四次 pass 全部拆开；全融合反向路径扩大的活跃变量范围，超过了减少状态 pass 所带来的收益。
- QAOA 从 22q 起在正向将 cost 融入 RX，并从 20q 起采用 compact 融合反向路径。XXZ 保留 cross-matching。

## 白盒模型与留出集验证

校准后的非负成本模型为：

`T = c_mem*state_pass_GiB + c_lane*lane_gate_G + c_reg*register_gate_G + c_warp*warp_gate_G + c_smem*mailbox_GiB + c_barrier*barrier_M + c_atomic*gradient_atomic_M + c_wave*CTA_waves + c_occ*occupancy_pressure_GiB + c_launch*phase_count`.

在 q=20/24/26 上训练，并在独立的 q=28 shape 测量中进行选择，结果如下：

| 门 | 方向 | 模型预测 | 实测最优 | APE 中位数 | 选择遗憾 |
|---|---:|---|---|---:|---:|
| RX | 正向 | `t128r3m2/compact` | `t64r4m1/compact` | 8.48% | 3.80% |
| RX | 反向 | `t64r4m2/compact` | `t64r4m2/compact` | 1.95% | 0.00% |
| RY | 正向 | `t128r3m2/compact` | `t64r4m2/compact` | 5.58% | 7.10% |
| RY | 反向 | `t64r4m1/compact` | `t64r4m2/fixed` | 2.17% | 8.86% |

因此，该模型适合作为搜索先验，而不是可移植的闭式最优选择公式：留出集上的选择遗憾接近 9%，按 family 外推 RY 时还可能更差。资源台阶、编译器寄存器分配以及 launch 合法性仍是不连续因素。

下表给出主实验数据上拟合的非负系数，特征顺序与上式一致。系数为零表示该相关项无法被独立辨识，并不表示其硬件成本真的为零。

| 门 | 方向 | state_pass_gib | lane_gates_g | register_gates_g | warp_gates_g | mailbox_gib | barrier_million | gradient_atomic_million | cta_waves | occupancy_pressure_gib | launches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RX | F | 0.3491 | 5.7116 | 7.1621 | 5.2003 | 0 | 0 | 0 | 0 | 0.42507 | 0 |
| RX | B | 0.27509 | 21.137 | 15.971 | 12.214 | 0 | 0 | 0 | 0 | 0.87418 | 0.0020204 |
| RY | F | 0.43841 | 5.609 | 6.3361 | 4.4028 | 0 | 0 | 0 | 0 | 0.50105 | 0 |
| RY | B | 0.22779 | 25.35 | 15.488 | 14.785 | 0 | 0.00044263 | 0 | 0 | 0.76173 | 0.0032445 |

## 可移植调优流程

1. 探测 SM 数量以及寄存器/共享内存上限；编译 7--12 bit tile 及所有 mailbox 切分因子，并剔除 launch 不合法的配置。
2. 对每个 `(gate,direction,q)`，以打乱顺序的重复轮次测量 compact/fixed/pair 完整 layer；保留五个 shape 候选，对其 DP 调度各筛选一次，再将前五名或 5% frontier 内的候选补足三次测量。
3. 校准 empty/lane/register/warp phase，拟合白盒模型，以动态规划遍历可达 phase 类别，再实测预测 frontier 以及多一个 phase 的方案。
4. 以旧实现的能量和梯度为正确性基准，重新进行端到端融合边界测量；绝不能只依据 ms/gate 选择融合方式。
5. 使用一个留出的 q 进行验证，并保留所有差距在 2% 以内的方案；当差距小于该阈值时，优先采用更简单的规则。

在这块 GPU 上，逐 phase 省略 mailbox 并不是普适规则：在各场景自适应执行最快且确实会改变执行路径的候选中，相邻配对耗时变化的中位数范围为 -5.54% 至 +0.21%；另有 5 个场景在其 2% frontier 上没有实际执行差异。因此，它应保留为需要实测后启用的可选开关；生产路径默认仍使用 full-mailbox。

任意 pure-rotation phase map 在语义上是正确的，但融合 RZZ 反向路径目前仍根据 canonical phase 公式推导 edge ownership。非 canonical 调度必须先加入显式 owner map，才能安全地成为通用生产 dispatch。
