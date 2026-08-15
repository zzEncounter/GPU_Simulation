# SAD CUDA 优化与调试研究报告

## 1. 研究范围与实验约定

本报告研究五类 8-layer、周期边界电路：

- RA-HEA：`RY -> ring CNOT`；
- SU2-HEA：`RY -> RZ -> ring CNOT`；
- RZZ-HEA：`RX -> RZ -> even RZZ -> odd RZZ`；
- 标准共享角 QAOA：`H^n -> repeated(even/odd RZZ(gamma_l) -> RX(beta_l))`；
- XXZ-HVA：Néel 初态，逐层 `even bonds -> odd bonds`，每条 bond 依次施加 `RXX(theta_x) RYY(theta_y) RZZ(theta_z)`。

前三类 HEA 沿用固定 TFIM 目标

```text
H = -sum_i Z_i Z_(i+1) - sum_i X_i
```

QAOA 使用环形 MaxCut cost Hamiltonian `H_C = 1/2 sum_i(ZZ-I)`（最小化其期望等价于最大化 cut）；XXZ-HVA 使用 `H_XXZ = sum_i(XX+YY+0.5 ZZ)`。这避免了“所有电路机械共用一个 Hamiltonian”造成的语义错配。

实验硬件是 NVIDIA RTX 6000 Ada 48 GiB（SM 8.9），CUDA 13.2；默认精度 float64，参数种子 42，参数从 `U(-pi, pi)` 采样。计时使用 CUDA event 同步后的 native wall-clock，预计算、显存分配和 warmup 不计入。小规模增加重复次数，大规模减少重复次数；主表报告 total median，同时列出 forward/H/backward mean。

`benchmark/results/sad_gpu.csv` 和 `pennylane_lightning_gpu.csv` 是 2026-08-07 已固定的原始 baseline，没有被覆盖。新结果分别写入：

- `benchmark/results/sad_optimized_gpu.csv`；
- `benchmark/results/qaoa_shared_sad_gpu.csv`；
- `benchmark/results/qaoa_xxz_pennylane_gpu.csv`；
- `benchmark/results/qaoa_xxz_sad_gpu.csv`（其中 XXZ-HVA 行）；
- `benchmark/results/research_main.csv`；
- `benchmark/results/research_ablation.csv`。

最终版本通过 55 个测试，覆盖五类电路、float32/float64、跨 phase 路径、legacy/optimized/all-fused/phased-forward 路径、规模 dispatch 以及 PennyLane 对照。65 个主实验配置的最大 energy absolute error 为 `3.91e-14`，最大 gradient element absolute error 为 `9.96e-13`。

## 2. 最终实现摘要

默认路径包含以下选择：

1. 首层从已知输入态直接生成层末状态；
2. RX/RY 保留 `128 threads × 4 amplitudes/thread` 安全变体，并按电路、qubit 数及 forward/backward 分别 dispatch 大 tile 变体；
3. RA/SU2 在 `q >= 18` 的 backward 固定 lane 对应物理低 5 位，其他路径使用紧凑布局；
4. RA 使用纯实数 state、Hamiltonian 和 backward，状态内存及流量减半；
5. RX/RY 最后一个 phase 与相邻 RZ/RZZ/CNOT 融合；
6. 对角相位使用 host 端 8-bit chunk lookup，timed kernel 内没有 `sincos`；
7. RZZ backward 按规模选择 phase 融合或独立的 RZ+全部 ZZ 单 pass；
8. SU2 在 28q 用逐门 phased `RY+RZ` 矩阵，并在最后 phase 直接 scatter CNOT；其他规模使用末 phase 的聚合 RZ lookup；
9. 不默认启用 phase 方向交替，也不把 RX 展成三层 `RZ-RY-RZ`。
10. QAOA 每层只保存一个共享 mixer gate angle 与一个共享 cost angle，backward 也只输出对应的两个梯度；XXZ-HVA 使用 bond-aware tile，一次合并同一 bond 的 RXX/RYY/RZZ。

## 3. 主实验：耗时与加速比

`vs fixed SAD` 使用两边的 total median。QAOA 与 XXZ-HVA 没有语义相同的旧 SAD 数据，故不报告该列。PennyLane 数据均为同电路、同参数、同 Hamiltonian、同精度。

### RA-HEA

| q | forward ms | H ms | backward ms | total ms | vs fixed SAD | vs PennyLane |
|--:|--:|--:|--:|--:|--:|--:|
| 4 | 0.046 | 0.019 | 0.079 | 0.144 | 9.48× | 56.55× |
| 6 | 0.054 | 0.020 | 0.111 | 0.185 | 6.83× | 55.54× |
| 8 | 0.066 | 0.019 | 0.136 | 0.221 | 7.27× | 56.29× |
| 10 | 0.106 | 0.019 | 0.194 | 0.318 | 5.95× | 44.86× |
| 12 | 0.118 | 0.019 | 0.226 | 0.363 | 6.45× | 44.50× |
| 14 | 0.133 | 0.019 | 0.265 | 0.419 | 6.51× | 44.18× |
| 16 | 0.146 | 0.019 | 0.307 | 0.474 | 6.85× | 45.13× |
| 18 | 0.295 | 0.027 | 0.792 | 1.114 | 6.44× | 24.82× |
| 20 | 0.918 | 0.060 | 2.281 | 3.259 | 10.21× | 13.51× |
| 22 | 3.889 | 0.224 | 8.534 | 12.649 | 10.85× | 16.19× |
| 24 | 15.522 | 0.862 | 36.496 | 52.879 | 11.40× | 22.34× |
| 26 | 74.471 | 4.377 | 170.144 | 248.993 | 10.44× | 20.49× |
| 28 | 595.605 | 26.768 | 690.957 | 1313.330 | 8.57× | 16.88× |

### SU2-HEA

| q | forward ms | H ms | backward ms | total ms | vs fixed SAD | vs PennyLane |
|--:|--:|--:|--:|--:|--:|--:|
| 4 | 0.051 | 0.019 | 0.127 | 0.199 | 4.51× | 76.42× |
| 6 | 0.065 | 0.019 | 0.178 | 0.262 | 4.75× | 50.56× |
| 8 | 0.087 | 0.019 | 0.245 | 0.350 | 4.52× | 48.53× |
| 10 | 0.135 | 0.019 | 0.347 | 0.504 | 3.97× | 40.07× |
| 12 | 0.150 | 0.019 | 0.399 | 0.568 | 4.34× | 41.08× |
| 14 | 0.167 | 0.019 | 0.449 | 0.635 | 4.67× | 41.86× |
| 16 | 0.190 | 0.019 | 0.522 | 0.732 | 4.75× | 43.05× |
| 18 | 0.438 | 0.035 | 1.514 | 1.987 | 4.33× | 20.14× |
| 20 | 1.482 | 0.096 | 4.884 | 6.461 | 5.62× | 10.21× |
| 22 | 5.971 | 0.379 | 24.048 | 30.480 | 4.91× | 12.09× |
| 24 | 29.794 | 1.679 | 89.626 | 121.278 | 5.41× | 17.31× |
| 26 | 153.367 | 10.148 | 398.454 | 561.969 | 5.09× | 15.37× |
| 28 | 1012.695 | 84.657 | 1610.618 | 2707.970 | 4.54× | 13.87× |

### RZZ-HEA

| q | forward ms | H ms | backward ms | total ms | vs fixed SAD | vs PennyLane |
|--:|--:|--:|--:|--:|--:|--:|
| 4 | 0.057 | 0.019 | 0.153 | 0.229 | 3.72× | 74.37× |
| 6 | 0.076 | 0.019 | 0.219 | 0.314 | 3.80× | 47.70× |
| 8 | 0.110 | 0.019 | 0.317 | 0.446 | 3.47× | 42.17× |
| 10 | 0.160 | 0.019 | 0.446 | 0.628 | 3.15× | 35.53× |
| 12 | 0.177 | 0.019 | 0.514 | 0.711 | 3.52× | 36.43× |
| 14 | 0.194 | 0.019 | 0.572 | 0.784 | 3.87× | 41.62× |
| 16 | 0.220 | 0.019 | 0.662 | 0.902 | 4.29× | 40.05× |
| 18 | 0.520 | 0.036 | 1.539 | 2.094 | 4.21× | 22.38× |
| 20 | 1.746 | 0.093 | 6.110 | 7.949 | 4.62× | 10.24× |
| 22 | 7.321 | 0.375 | 25.174 | 32.866 | 4.57× | 13.71× |
| 24 | 34.014 | 1.689 | 107.864 | 143.729 | 4.51× | 18.76× |
| 26 | 200.856 | 10.127 | 478.968 | 689.951 | 4.05× | 16.40× |
| 28 | 1041.336 | 82.452 | 2072.214 | 3196.003 | 3.81× | 15.88× |

### QAOA

| q | forward ms | H ms | backward ms | total ms | vs PennyLane |
|--:|--:|--:|--:|--:|--:|
| 4 | 0.080 | 0.019 | 0.132 | 0.231 | 37.58× |
| 6 | 0.103 | 0.019 | 0.173 | 0.271 | 40.39× |
| 8 | 0.112 | 0.043 | 0.213 | 0.315 | 40.78× |
| 10 | 0.151 | 0.019 | 0.271 | 0.440 | 35.36× |
| 12 | 0.183 | 0.037 | 0.326 | 0.501 | 37.64× |
| 14 | 0.238 | 0.054 | 0.370 | 0.558 | 37.73× |
| 16 | 0.214 | 0.019 | 0.427 | 0.660 | 37.19× |
| 18 | 0.575 | 0.050 | 1.231 | 1.792 | 18.37× |
| 20 | 2.516 | 0.072 | 5.127 | 7.802 | 7.27× |
| 22 | 8.610 | 0.254 | 20.598 | 29.382 | 10.54× |
| 24 | 40.427 | 0.709 | 90.295 | 132.026 | 13.38× |
| 26 | 252.337 | 3.124 | 481.135 | 736.595 | 10.52× |
| 28 | 1168.374 | 12.032 | 2124.025 | 3304.430 | 10.10× |

### XXZ-HVA

| q | forward ms | H ms | backward ms | total ms | vs PennyLane |
|--:|--:|--:|--:|--:|--:|
| 4 | 0.083 | 0.028 | 0.157 | 0.258 | 63.54× |
| 6 | 0.109 | 0.019 | 0.234 | 0.344 | 45.24× |
| 8 | 0.166 | 0.019 | 0.356 | 0.520 | 38.19× |
| 10 | 0.272 | 0.019 | 0.647 | 0.938 | 24.72× |
| 12 | 0.372 | 0.059 | 0.765 | 1.099 | 24.89× |
| 14 | 0.358 | 0.019 | 0.878 | 1.256 | 25.52× |
| 16 | 0.501 | 0.048 | 1.031 | 1.466 | 26.27× |
| 18 | 1.282 | 0.128 | 3.475 | 4.906 | 10.43× |
| 20 | 5.415 | 0.215 | 13.273 | 19.081 | 4.68× |
| 22 | 18.954 | 0.523 | 60.900 | 80.512 | 6.10× |
| 24 | 91.644 | 1.565 | 264.371 | 357.834 | 7.69× |
| 26 | 478.039 | 8.285 | 1170.906 | 1657.230 | 7.29× |
| 28 | 2138.779 | 43.295 | 5049.400 | 7231.474 | 7.21× |

主要现象：backward 始终是端到端瓶颈；RA 实数特化后这一比例有所下降。XXZ-HVA backward 最重，因为每条 bond 要规约 X/Y/Z 三个 generator，再给 `phi/lambda` 同时做逆演化。18q 左右从 launch/occupancy 主导转入带宽和全状态 pass 主导。26–28q 的 Hamiltonian 也开始可见，但仍远小于 rotation/bond backward。

## 4. 首层直接生成：稳定正优化

从已知输入态出发，首层非纠缠旋转是 product state；后续 RZ/RZZ 是 basis-diagonal，CNOT 是 basis permutation。因此可以在每个输出 basis index 上一次性计算最终 amplitude，不需要：

1. 清零两个完整 state buffer；
2. 写入 `|0...0>` 或 `|+...+>`；
3. 执行一个或多个 RX/RY phase；
4. 单独执行对角门和 CNOT pass。

该变换减少 full-state 读写，且没有新增同步或渐近工作，所以是结构上的正优化。下面是只隔离第一层 forward 的 `legacy / direct-initial` 加速比：

| circuit | 4q | 8q | 12q | 16q | 20q | 24q | 28q |
|---|---:|---:|---:|---:|---:|---:|---:|
| RA | 2.37× | 2.83× | 3.82× | 3.75× | 9.42× | 15.50× | 27.27× |
| SU2 | 2.56× | 3.01× | 3.86× | 3.89× | 10.02× | 16.52× | 27.67× |
| RZZ | 2.29× | 1.90× | 3.22× | 3.26× | 4.13× | 6.82× | 13.60× |
| QAOA（共享角、cost-first） | 1.32× | 1.32× | 1.24× | 1.21× | 1.10× | 1.21× | 1.23× |

前三类 HEA 的 21 个配置全部为正，默认开启。标准 QAOA 不能再利用旧实现中“先对 `|+>` 做 RX 只产生 global phase”的捷径，因为正确门序是 cost 后 mixer。新实现直接写出 `RZZ(gamma_0)|+>`，再执行共享 RX；26q 使用低 live-range 的 split initializer，28q 则把 cost 乘法放进首个 RX phase。这个按规模选择避免了强行使用单一 initializer 时 28q 约 9% 的回退。XXZ-HVA 直接写出 Néel basis state，避免通用清零/置位流程。

## 5. 不同门的耗时与最难路径

### 5.1 完整门级 profile 与 RX/RY 修复

下表给出 24q、complex float64 下的代表性整层耗时。RX/RY 使用最终实现的 128-thread、4-amplitude/thread 受控微基准；其余门保留 Nsight Systems 在 8-layer legacy split 路径上的 profile。RX/RY kernel 一次处理整层 24 个门，RZZ even/odd kernel 各处理 12 个 matching gates，因此表中的 per-gate 均按一次 kernel 实际覆盖的门数均摊。

| 路径 | forward layer ms | forward us/gate | backward layer ms | backward us/gate |
|---|---:|---:|---:|---:|
| RY layer | 4.580 | 191 | 11.623 | 484 |
| RX layer | 4.730 | 197 | 11.093 | 462 |
| RZ layer | 0.640 | 26.7 | 1.811 | 75.5 |
| RZZ matching | 0.714 | 59.5 | 1.617 | 135 |
| ring CNOT permutation | 1.186 | 49.4 | 同一 kernel | 同一 kernel |
| Hamiltonian | 1.73 / evaluation | — | — | — |

RX 和 RY 理论上应当接近：二者都是同样的 2×2 single-qubit rotation，global load/store、shuffle/shared exchange 和 FMA 数量同阶。旧实现的 24q layer profile 中，RY forward/backward 分别为 5.695/14.097 ms，而 RX 为 4.120/9.893 ms；这个显著差距不是门的固有代价，而是实现缺陷。

原因是旧 `rotate_amplitude<RY>` 按输出 bit 写成 `if (bit)`。lane 承载的 5 个 tile bits 会让同一 warp 内两条分支都执行；RY generator 又先构造带符号的临时 complex，进一步延长 live range。RX 的非对角项对两个输出使用相同复数公式，没有该分支。

最终实现做了三项修正：

1. 直接 XOR IEEE sign bit 选择 RY 的 `+s/-s`，不产生控制流分歧；
2. 直接计算 `Y` generator overlap，不构造临时 complex；
3. backward 的 `phi/lambda` 复用一份 shared mailbox。

表中的最终整层结果已经收敛：RY forward 比 RX 快约 3.2%，backward 仅慢约 4.8%。低位 phase 单独测试时 RY 仍可能慢 5%–7%，来自符号数据依赖和略多指令，但整层差距已远小于旧实现的 40%–60%。

### 5.2 RZ、RZZ、CNOT 与 Hamiltonian

- **RZ** 最便宜。整层只需根据 basis bit 组合一个对角 phase factor，不交换配对 amplitude。backward 虽然增加 generator overlap 和规约，仍只需一次顺序读取 `phi/lambda`，实测为 26.7/75.5 us per gate。
- **RZZ** 也是对角门，但每个 eigenvalue 依赖两个 bit，even/odd matching 还要分别构造 lookup code，因此单门约为 RZ 的 1.8–2.2 倍。不过同一 matching 中的全部门共享一次 state pass，仍显著便宜于 RX/RY。
- **ring CNOT** 没有三角函数或梯度。整层被合成为一个可逆 basis permutation；forward 与 backward 使用同一个地址变换 kernel，区别只是选择正向或逆向映射。其主要成本是非连续读取加完整 state 写回，而不是算术。
- **Hamiltonian** 每次 evaluation 同时构造 `H|psi>` 和 energy。ZZ 部分是对角累加，X 部分对每个 qubit 读取一次 bit-flip partner，所以单次成本高于独立 RZ/RZZ layer；但每个训练 step 只执行一次，24q 的 1.73 ms 仍远小于 8 层 rotation backward 总和。

综合来看，最终最难处理的仍是 **RX/RY backward**：它必须读取两条演化、为每个参数规约 generator overlap，并同时逆演化 `phi/lambda`。RZ/RZZ 的优势来自对角性和层内共享，CNOT 是纯 permutation，Hamiltonian 虽然单次读取较多，但调用次数少。

## 6. RX/RY 的内存 hierarchy、tile 与 phase

### 6.1 三层映射

RX/RY kernel 的局部振幅按三层持有：

- lane：warp 的 32 lanes，对应 5 个 tile bits，通过 `shfl_xor` 交换；
- registers：每 thread 持有 `2^r` 个 amplitude，对应 `r` 个 tile bits；
- warp/shared mailbox：CTA 的 `2^w` 个 warps，对应 `w` 个 tile bits。

一个 tile 最多同时处理 `5 + r + w` 个同轴单比特门。`128 threads × 4 amplitudes/thread` 是安全小变体，即 4 warps、9-bit tile。forward 和 backward 分别编译，并由 Python runner 按 circuit/qubits dispatch，因为 backward 必须同时保存两个演化，资源约束不同。

三层对 tile 容量的贡献是乘法关系，而对可同时处理的门数是加法关系：

| 配置 | lane 因子 | thread-local 因子 | warp 因子 | tile amplitudes | 可处理门 slots |
|---|---:|---:|---:|---:|---:|
| 128 threads × 4 amplitudes/thread | 32 | 4 | 4 warps | `32×4×4=512` | `5+2+2=9` |
| 128 threads × 8 amplitudes/thread | 32 | 8 | 4 warps | `32×8×4=1024` | `5+3+2=10` |
| 64 threads × 16 amplitudes/thread | 32 | 16 | 2 warps | `32×16×2=1024` | `5+4+1=10` |

因此 `64×16` 和 `128×8` 都是 10-bit/1024-amplitude tile，但前者把一个 warp bit 换成一个 register bit。二者 global traffic 相同，区别来自 register pressure、occupancy，以及一次 shared-mailbox 交换和一次 thread-local 配对的相对代价。

这里的“32 registers/thread”应准确理解为“每 thread 持有 32 个 complex amplitudes”，不是 ptxas 报告的 32 个硬件寄存器。一个 complex<double> 本身至少占 4 个 32-bit hardware registers；加上索引和中间值后，32 amplitudes/thread 的 forward kernel 实际使用 206–218 个硬件寄存器。

### 6.2 一个低位/高位 phase 的成本

24q forward 单 phase 微基准如下。`high-full` 把完整高位 target 集映射到 lane/register/warp；`high-fixed` 强制 lane 对应物理低 5 位，只在 register/warp slots 放 target。

| threads × registers | low-full ms (gates) | high-full ms (gates) | high-fixed ms (gates) |
|---|---:|---:|---:|
| 128 × 4 | 1.307 (9) | 1.983 (9) | 0.776 (4) |
| 128 × 8 | 1.384 (10) | 1.948 (10) | 0.846 (5) |
| 128 × 16 | 1.387 (11) | 1.973 (11) | 0.917 (6) |
| 256 × 4 | 1.446 (10) | 2.253 (10) | 0.891 (5) |
| 256 × 8 | 1.574 (11) | 2.213 (11) | 1.048 (6) |
| 256 × 16 | 1.746 (12) | 2.697 (12) | 1.224 (7) |

128×4 时，high-full 比 low-full 慢 51.7%；原因是高物理位进入 lane 后，warp 的 global load/store 地址不连续。high-fixed 每门 0.194 ms，优于 high-full 的 0.220 ms，但慢于连续 low-full 的 0.145 ms，而且每 phase 只能做 4 门，因此是否采用仍取决于总 phase 数。

为了更直接地区分三层的动态成本，又用 target mask 做了累计消融。下表是 24q RY 单 phase、100 次 kernel iteration 复测的 median；`base` 仍完成一次 state load/store，但不执行门，后续列依次加入 5 个 lane 门、全部 register 门、全部 warp 门。由于 GPU boost/温度状态与前一批实验不同，这组绝对值只在同表内比较，不与上表的旧批次绝对时间混用：

| direction / 配置 | base | + 5 lane | + registers | + warps = full |
|---|---:|---:|---:|---:|
| forward 128×4，slots `5+2+2` | 0.854 | 0.934 | 1.204 | 1.439 |
| forward 64×16，slots `5+4+1` | 0.859 | 0.888 | 1.179 | 1.344 |
| backward 128×4，slots `5+2+2` | 1.699 | 2.404 | 3.185 | 3.939 |
| backward 64×16，slots `5+4+1` | 1.555 | 1.936 | 3.205 | 3.573 |

以最终常用的 64×16 为例，forward 的三段 marginal 增量是 `0.029/0.291/0.165 ms`，backward 是 `0.381/1.269/0.367 ms`。这不是三种 primitive 的脱离上下文 latency：lane shuffle 的大部分成本被 global-memory latency 隐藏，累计执行后才暴露 register/warp 工作；backward 还包含每个 generator 的 overlap 与 gradient reduction。数据说明两个关键点：低 5 位 lane slots 的边际成本最低；一个 warp slot 通常比一个 register slot 贵，所以在同为 10-bit tile 时，64×16 能用 4 register bits、1 warp bit，常优于 128×8 的 3+2，但它付出 254 hardware registers/thread 和较低的资源余量，不能无条件用于所有方向和规模。

### 6.3 为什么不是 tile 越大越好

增大 tile 并不会立刻劣化。24q 完整 RY layer 中，128×4 为 4.141 ms，64×16 为 3.749 ms；26q 时 32 amplitudes/thread 还能因更大的局部工作集获益。真正的断点出现在 64 amplitudes/thread：kernel 达到 255 hardware registers，RY forward 每 thread 约有 9–10 KiB spill traffic，且 64 KiB shared 只允许 1 CTA/SM。

| forward 配置 | ptxas registers/thread | shared/CTA | active CTA/SM | 24q full layer ms |
|---|---:|---:|---:|---:|
| 64 × 16 amplitudes | 152 | 16 KiB | 5 | 3.749 |
| 64 × 32 amplitudes | 218 | 32 KiB | 3 | 3.844 |
| 64 × 64 amplitudes | 255 + spill | 64 KiB | 1 | 5.179 |
| 512 × 4 amplitudes | 54 | 32 KiB | 2 | 4.876 |
| 512 × 8 amplitudes | 90 | 64 KiB | 1 | 4.289 |

限制因素不只有 shared memory：还有每 SM register file、最大 resident threads/CTA、cooperative grid 的 resident block 数、warp-slot 的同步次数和展开后的 instruction footprint。512 threads 没有“立刻失效”，但 1–2 CTA/SM 和更重的 CTA barrier 使它始终输给 64/128-thread 候选。

backward 原来同时分配完整 `phi_mailbox` 和 `lambda_mailbox`。现在先用 mailbox 完成 phi overlap/逆旋转，block reduction 的 barrier 后覆写为 lambda，再逆旋转 lambda；只增加一次 warp-slot barrier，shared 减半。128×4 complex<double> 从 17,424 B 降到 9,232 B；11-bit backward tile 从超过静态 shared 上限变成可编译。24q fixed-low RY 完整层由安全 128×4 的 10.584 ms 降到最终 64×16 的 8.349 ms。512×4 即使也因此可运行，仍为 12.709 ms。

### 6.4 两种 lane 布局的最终选择

- forward：第一段是 low-full，后续仍用紧凑 high-full；
- RA/SU2 backward，`q >= 18`：第一段是 low-full，后续为 high-fixed。fixed-low 虽有更多 phases，但连续访问带来的收益更大；
- RZZ backward：与 forward 一样使用紧凑 high-full，因为对角融合与 phase ownership 会放大额外 phase 成本；QAOA 的共享 mixer 也用紧凑 high-full，而共享 cost gradient 由独立单 pass 规约。

phase 的最后一段天然允许不等长 target mask，避免无效门；不同 tile 大小是否“刚好少一次 pass”已经在调参数据中体现。

当前固定的是**一次调用内的硬件形状**，不是 phase 门数。runner 先按 circuit/qubits 选择一个编译变体；该次 cooperative kernel 内所有 phase 的 `blockDim` 和 amplitudes/thread 相同，但 target mask 可以不同。例如 24q 的 target 数为：

| tile / 布局 | 各 phase target 数 |
|---|---|
| 9-bit compact | `[9,9,6]` |
| 9-bit fixed-low | `[9,4,4,4,3]` |
| 10-bit compact | `[10,10,4]` |
| 10-bit fixed-low | `[10,5,5,4]` |

### 6.5 fixed-low phase 是否应该变长

这里要区分两个问题：

1. hardware high capacity 不变，只移动短尾 target，例如把 `[9,4,4,4,3]` 改成 `[9,3,4,4,4]`；
2. 根据 q 或 phase 改变 `r+w`，令 high capacity 从 4 变成 5/6，从而可能少一次完整 state pass。

第一种没有收益。24q、128×4、high capacity=4 的 complex RY backward 同轮结果为：

| fixed-low target 分段 | ms |
|---|---:|
| `[9,4,4,4,3]` | **11.393** |
| `[9,3,4,4,4]` | 11.438 |
| `[9,4,3,4,4]` | 11.409 |
| `[9,4,4,3,4]` | 11.410 |

差距最多 0.4%，且当前“前面填满、短尾留最后”略好。因此在 phase 数和 hardware capacity 都相同时，不需要搜索短尾位置。

第二种确实存在离散边界收益。测试的主要形状是：

- high capacity 4：128 threads × 4 amplitudes，9-bit tile；
- high capacity 5：64 threads × 16 amplitudes，10-bit tile；
- high capacity 6：128 threads × 16 amplitudes，11-bit tile。

complex RY backward 的完整 fixed-low layer 如下；`h5/h6` 大于 1 才表示容量 6 更快：

| q | capacity 5 分段 | h5 ms | capacity 6 分段 | h6 ms | h5/h6 |
|---:|---|---:|---|---:|---:|
| 20 | `[10,5,5]` | 0.544 | `[11,6,3]` | 0.564 | 0.97× |
| 21 | `[10,5,5,1]` | 1.176 | `[11,6,4]` | 1.183 | 0.99× |
| 22 | `[10,5,5,2]` | 2.492 | `[11,6,5]` | 2.216 | **1.12×** |
| 23 | `[10,5,5,3]` | 4.703 | `[11,6,6]` | 4.505 | **1.04×** |
| 24 | `[10,5,5,4]` | **9.140** | `[11,6,6,1]` | 10.751 | 0.85× |
| 25 | `[10,5,5,5]` | **18.281** | `[11,6,6,2]` | 21.097 | 0.87× |
| 26 | `[10,5,5,5,1]` | 42.159 | `[11,6,6,3]` | 41.126 | **1.03×** |
| 27 | `[10,5,5,5,2]` | 84.296 | `[11,6,6,4]` | 82.40 | **1.02×** |

规律不是“capacity 越大越好”，而是更大 capacity 是否刚好减少 phase_count。22–23q、26–27q 少一个 state pass，覆盖了更高 register pressure、33 KiB shared 和较低 occupancy；24–25q 的 phase 数不变，capacity 6 反而慢 15%–18%。

真正混用 phase 形状也做了对照。24q 从 capacity 4 的 `[9,4,4,4,3]` 改为“第一段 capacity 4、后三段 capacity 5”的 `[9,5,5,5]`，由 11.149 ms 降至 9.307 ms，确实正优化 16.5%；但整次调用统一使用 capacity 5 的 `[10,5,5,4]` 只需 8.819 ms，又比混合方案快 5.5%。也就是说，异构 phase 可以击败较小固定配置，但目前没有击败该 q 的最佳统一配置。

完整 8-layer 电路也保留了 q/circuit 依赖：RA 的 capacity 6 在 22/26/27/28q 将 backward 分别缩短约 11.6%/5.9%/3.4%/3.1%，total 缩短约 5.3%/4.3%/2.1%/1.6%；SU2 在 22q 的 backward/total 缩短约 7.4%/5.8%，但 26–28q 因 RZ/CNOT 融合和资源压力略为负。23q 收益较小且更接近频率噪声。

结论：**存在按 qubit 数变长的最优方案**，优先级应是按 circuit/q 为整次 backward 选择 high capacity 4/5/6；仅移动同容量的短尾没有价值。逐 phase 混合 threads/registers 需要拆成多个 kernel variant/launch，当前候选尚未胜过每个 q 的最佳统一容量，因此暂不作为默认路径。

### 6.6 forward/backward 分离与按规模 dispatch

单 phase 最优不能代替完整电路最优；最终选择来自 8-layer fused 电路的 total time。小规模保持 128×4，避免展开的大 tile 在有效 state 很小时浪费线程和指令；大规模按下表加载独立共享库变体。

| circuit / q | forward threads × amplitudes | backward threads × amplitudes |
|---|---:|---:|
| 未列出的较小规模 | 128 × 4 | 128 × 4 |
| RA 20–26q | 64 × 16 | 64 × 16 |
| RA 28q | 64 × 8 | 64 × 16 |
| SU2 >=20q | 128 × 8 | 64 × 16 |
| RZZ/QAOA 24q | 64 × 8 | 64 × 16 |
| RZZ/QAOA >=26q | 64 × 16 | 128 × 8 |

相对安全 128×4 变体，选择表在 24/26/28q 的 total speedup 分别为：RA `1.24×/1.31×/1.14×`，SU2 `1.15×/1.34×/1.25×`，RZZ `1.08×/1.05×/1.22×`。标准共享角 QAOA 重新做了正确性与主实验，但没有把旧的非共享 QAOA 调参数字冒充为新语义的消融结论。`SAD_DISABLE_VARIANT_DISPATCH=1` 可回到安全变体做对照。

## 7. RealAmplitude 是否只有实数

是。RA 从实 `|0>` 开始，只含 RY（实矩阵）和 CNOT（实 permutation）。当前 Hamiltonian 的 X、ZZ 也全是实矩阵，因此：

- forward state 始终为实数；
- `lambda = H phi` 始终为实数；
- backward 的两条逆演化仍为实数；
- RY 梯度可直接写成带符号的 `lambda[b] * phi[b xor mask]` 规约，不需要存储 imaginary component。

最终实现把四个 state buffer 的实际元素类型和流量从 complex<T> 降为 T，并使用实数 initial/rotation/H/backward kernel。相对同一优化结构但仍用 complex state 的消融如下：

| q | complex total ms | real total ms | speedup | 单 state 内存 |
|---:|---:|---:|---:|---:|
| 4 | 0.222 | 0.144 | 1.54× | 1/2 |
| 16 | 0.754 | 0.465 | 1.62× | 1/2 |
| 20 | 8.294 | 3.590 | 2.31× | 1/2 |
| 24 | 146.166 | 65.751 | 2.22× | 1/2 |
| 28 | 3404.037 | 1515.938 | 2.25× | 1/2 |

所有规模正收益，默认保留。

## 8. RX/RY phase 与 RZ/ZZ/CNOT 融合

forward 在最后一个 rotation phase 写回时：

- 直接乘 RZ/RZZ lookup factor；
- 若有 ring CNOT，直接 scatter 到 permutation 后的 output index；
- 省掉一到三个完整 state pass。

只比较 `direct-first + split remaining layers` 与 `direct-first + fused remaining layers` 的 forward 加速：

| circuit | 16q | 20q | 24q |
|---|---:|---:|---:|
| RA | 1.04× | 1.09× | 1.17× |
| SU2 | 1.06× | 1.11× | 1.21× |
| RZZ | 1.00× | 1.08× | 1.22× |
| QAOA（共享角，cost-before-RX 候选） | 0.95× | 1.01× | 0.98× |

大规模 RZZ 稳定为正，小规模接近噪声。标准 QAOA 更能说明“少一次 pass”不等于无条件更快：同形状隔离后 16q 慢 4.8%，20q/22q 分别快 1.3%/5.6%；生产变体下 24q 慢 2.4%，26q/28q 分别快 2.8%/5.6%。原因是 cost factor 放进 rotation kernel 后增加 index/lookup live range 和 register pressure。默认只在已复测为正的 20q、22q、26q、28q 启用 cost-before-RX 融合，其余规模保持独立 cost pass。`SAD_QAOA_FUSE_COST_RX=0/1` 可强制 split/fused，复现同一硬件形状的消融。

### phase 方向交替

相邻层交替 high-to-low / low-to-high 可以让一层的最后布局与下一层的第一布局相同，理论上增加 L2 命中。实测只有 20q SU2 约 3% 收益；16q、24q 和 RZZ/RA 均在噪声内或略慢。原因是每层之间还有 diagonal/permutation，且完整 state 很快超过 cache。最终不默认开启，编译宏 `SAD_ALTERNATE_PHASES=1` 保留用于复现。

## 9. RZ/ZZ 的三角函数

若每个 amplitude 在 device 上累加 phase 后调用 double `sincos`，三角函数非常贵。最终实现把每 8 个 generator 的 256 种 eigenvalue code 在 host 端预计算为 complex factor；kernel 只构造 code、读取 lookup 并做少量 complex multiply。

| gate | q | device sincos ms | chunk lookup ms | speedup |
|---|---:|---:|---:|---:|
| RZ | 16 | 0.00881 | 0.00365 | 2.41× |
| RZ | 20 | 0.11974 | 0.02925 | 4.09× |
| RZ | 24 | 2.06783 | 0.70110 | 2.95× |
| RZ | 28 | 34.14548 | 10.95048 | 3.12× |
| ring ZZ | 16 | 0.00884 | 0.00471 | 1.88× |
| ring ZZ | 20 | 0.12042 | 0.04724 | 2.55× |
| ring ZZ | 24 | 2.07889 | 0.75469 | 2.75× |
| ring ZZ | 28 | 34.39739 | 13.96671 | 2.46× |

lookup 构建在 timed region 外，默认保留。继续增大 lookup chunk 会指数增加表大小和 cache 压力；8-bit 是小表与乘法次数之间的稳健点。

### 9.1 thread 持有什么

lookup 表保存在 device global memory；由于单层表通常只有几 KiB 到几十 KiB，并被大量 amplitudes 重复读取，实际主要由 L2/L1 cache 服务。thread 不会长期持有“后一段整张表”。对每个 amplitude，它只保留：

- 一个当前 chunk 的整数 `code`；
- 一个 complex `factor` 累加器；
- 当前从 lookup 读出的一个临时 complex value。

执行顺序是 `构造 code -> load lookup[chunk,code] -> factor *= value`。进入下一 chunk 时只保留累计后的 `factor`；处理下一个 amplitude 时重新构造所有 code，因为 basis pattern 已改变。fused RX/RY kernel 也是每个 register amplitude 临时求 factor，没有把 lookup 表复制进 registers/shared。

### 9.2 预计算与 H2D 成本

对包含 g 个 generators、chunk bits 为 k 的一组门，表项数是

```text
ceil(g/k) * 2^k
```

host 构建复杂度约为 `O(ceil(g/k) * 2^k * k)`，device 端每 amplitude 则做 `ceil(g/k)` 次 lookup 和 complex multiply。当前 k=8 时，28q 单个 RZ 组的构建约 0.035 ms，even+odd ZZ 两组合计约 0.034 ms。因而 8-layer SU2 的全部 RZ lookup 约 0.28 ms；8-layer RZZ-HEA 的 RZ+ZZ lookup 约 0.55 ms。表打包后只做一次 H2D copy，128–256 KiB 的实测/估算量级为几十微秒。

主实验的参数在一次 `energy_and_grad` 调用内不变，lookup 在 timed region 前只构建一次，多个 warmup/timed steps 重用。因此报告中的 kernel 时间不含这 0.3–0.6 ms。真实 optimizer 若每一步更新参数，就应把 lookup rebuild 算进 step；8-bit 下相对 24q/28q 的约 0.1–3 s total 仍很小，但大 chunk 时会变得可见。

### 9.3 chunk bits 扫描

28q 单层 median 如下。RZZ 表示 even/odd 两个 matching 的总表和完整 kernel：

| chunk bits | RZ table | RZ build ms | RZ kernel ms | RZZ table | RZZ build ms | RZZ kernel ms |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1.75 KiB | 0.0105 | 14.457 | 2 KiB | 0.0104 | 19.610 |
| 6 | 5 KiB | 0.0179 | **12.626** | 6 KiB | 0.0181 | 17.411 |
| 8 | 16 KiB | 0.0351 | 12.686 | 16 KiB | 0.0344 | **15.460** |
| 10 | 48 KiB | 0.1124 | 12.709 | 64 KiB | 0.1146 | 16.410 |
| 12 | 192 KiB | 0.4261 | 13.059 | 256 KiB | 0.4833 | 16.565 |

分段有影响，但在合理的 6–10 bits 范围通常不是数量级差异。RZ 28q 的 6/8/10-bit 相差不到 1%；4-bit 因每 amplitude 要做 7 次 lookup/multiply，慢约 14%。RZZ 28q 的 8-bit 明显最好：它把两个 14-gate matching 各压到 2 个 chunks；10/12-bit 没继续减少 chunk 数，只增加表和 cache footprint。

小规模存在局部例外。24q RZZ 用 12-bit 时每个 12-gate matching 恰好只需一个 chunk，isolated kernel 从 8-bit 的 0.869 ms 降到 0.800 ms；但表从 16 KiB 增至 128 KiB，单层构建从 0.034 ms 增至 0.337 ms。8-layer RZZ-HEA 若同时包含 RZ+ZZ，12-bit lookup 约 2 MiB、host 构建约 5.4 ms；在每优化步更新参数的口径下会吞掉相当一部分 kernel 收益。

因此 8-bit 不是每个孤立尺寸的绝对最优，而是 RZ/ZZ、24–28q、forward/backward/fused 路径共享的稳健折中。若追求最后几个百分点，可以按 gate type/q 选择 6/8/12-bit；默认路径继续使用 8-bit，避免为小的 kernel 收益支付指数表和每步 rebuild 成本。

## 10. Backward 的性质与按规模选择

### 10.1 RZ/ZZ 可以只用层端点求全部梯度

同一对角层的 RZ/ZZ 及其 generators 全部对易。对 `D(theta)=prod_j exp(-i theta_j G_j/2)`，每个 `G_j` 与完整 `D` 对易，所以一层的全部

```text
Im(<lambda | G_j | phi>)
```

可以在同一个层端点状态上求出，不需要保存门间状态。实现一次读取 `phi/lambda`，同时规约 RZ 和整圈 ZZ 梯度，并乘合并后的 inverse diagonal factor。

RZZ-HEA backward 对比（ms/layer loop aggregate，8 layers）：

| q | 3-way split | diagonal 单 pass + RX | 融入 RX phases | 选择 |
|---:|---:|---:|---:|---|
| 16 | 0.874 | 0.852 | 0.704 | phase fused |
| 18 | 1.963 | 1.942 | 1.662 | phase fused |
| 20 | 6.648 | 6.439 | 6.422 | phase fused |
| 22 | 25.885 | 24.988 | 26.621 | diagonal single-pass |
| 24 | 118.156 | 113.095 | 119.153 | diagonal single-pass |
| 26 | 565.721 | 554.577 | 520.177 | phase fused |
| 28 | 3573.937 | 3601.114 | 2510.746 | phase fused |

不是“融合越多越好”：22–24q 时，把 2q 个 diagonal partials 塞进每个 RX phase 会增加 shared/reduction 压力，独立单 pass 更好；其他规模 phase 融合减少 state pass 的收益更大。默认按此边界选择。

### 10.2 RX/RY backward

同轴、不同 qubit 的 RX/RY generators 也对易；kernel 已在 tile 内一次 load 后依次完成多个 generator overlap 和两个逆演化，不把中间 state 写回。跨 tile phase 仍必须重新读写全 state。`phi/lambda` 双演化使 backward 的最佳 tile 比 forward 更保守，且 fixed-low 布局只在 RA/SU2 大规模有利。

SU2 的 fused backward 在 18/22/26q 略快，20q 和 28q 反而 split 更快，24q 基本相同；最终同样按规模选择。CNOT backward 与 forward 都是可逆 basis permutation，没有额外 gradient 工作。

## 11. RX 分解与 phased RY/RZ

### 11.1 RX 的两种基变换分解

这里需要区分两个等价恒等式：

```text
RX(theta) = RZ(-pi/2) RY(theta) RZ(+pi/2)
RX(theta) = RY(+pi/2) RZ(theta) RY(-pi/2)
```

你指出的著名“把 RX 变成对角门”策略是第二式：两个恒定角 RY 中间夹一个可训练、对角的 RZ。此前报告只测了第一式，现已补测并纠正。

| q | direct RX ms | fixed-RZ sandwich ms | fixed-RY sandwich ms | fixed-RY slowdown |
|---:|---:|---:|---:|---:|
| 16 | 0.0216 | 0.0287 | 0.0468 | 2.16× |
| 18 | 0.0535 | 0.0728 | 0.1191 | 2.23× |
| 20 | 0.1956 | 0.2710 | 0.4609 | 2.36× |
| 22 | 0.8125 | 1.1067 | 1.8928 | 2.33× |
| 24 | 4.1939 | 5.7085 | 9.2738 | 2.21× |
| 26 | 26.2029 | 31.8499 | 55.3258 | 2.11× |
| 28 | 188.5518 | 204.8813 | 377.3046 | 2.00× |

第二式每层需要两个非对角 RY 全状态 phase 序列和一个对角 pass；直接 RX 只需一个非对角序列，所以始终慢约 2.0×–2.36×。第一式虽然只有一个非对角序列，也仍慢 1.09×–1.39×。生产路径从未依赖该分解，实验源码也已删除，只保留本表作为否定证据。

### 11.2 相邻的可训练 `RY + RZ`

比较两种单 pass 方案：

1. 每个 phase 只做 RY，最后一个 phase 对每 amplitude 乘整层 RZ lookup；
2. 每个 qubit 直接应用 `RZ(phi_i) RY(theta_i)` 的 phased 2×2 matrix。

| q | lookup 路径 ms | phased matrix ms | phased/lookup 结论 |
|---:|---:|---:|---|
| 4 | 0.067 | 0.077 | 慢 15% |
| 16 | 0.241 | 0.267 | 慢 11% |
| 20 | 2.275 | 3.016 | 慢 33% |
| 22 | 9.531 | 12.892 | 慢 35% |
| 24 | 41.066 | 53.976 | 慢 31% |
| 26 | 305.132 | 327.558 | 慢 7% |
| 28 | 1484.7 | 1404.1（多次复测） | 快约 5% |

中等规模时逐门 complex phase 增加算术和 register live range，聚合 lookup 更好；28q 完全带宽主导，省掉最终 per-amplitude lookup/index 组合后 phased matrix 稳定获益约 5%。最终只在 SU2 28q 选择 phased matrix。

这个 phased 矩阵仍然可以在最后 phase 接纠缠层。原层为

```text
CNOT_ring * product_i RZ_i(phi_i) * product_i RY_i(theta_i)
```

不同 qubit 上的算子彼此对易，因此可重排为 `CNOT_ring * product_i[RZ_i RY_i]`，逐 qubit phased matrix 分散到不同 phase 不改变顺序。实现只在所有前置 phase 完成 `grid.sync()` 后，令最后 phase 的写回直接 scatter 到 ring-CNOT 后的 basis index；不能把 CNOT 提前到尚未完成的 phase 之间。`test_trainable_phased_ry_rz_keeps_fused_cnot` 已逐元素对照 legacy energy/gradient。

## 12. QAOA

### 12.1 旧实现审计与最终定义

审计发现旧 QAOA 与 RA/SU2/RZZ-HEA 共用 `-sum ZZ-sum X`，门序是 mixer 后 cost，而且每个 RX/ZZ gate 都有独立参数。它可以作为 TFIM-inspired HVA，但不应在论文中直接称为标准 QAOA。原始 QAOA 的结构是从问题 cost Hamiltonian 和 mixer Hamiltonian 交替演化；PennyLane 的官方实现也把每层写为一个 `gamma_l` cost layer 后接一个 `alpha_l` mixer layer（[Farhi et al.](https://arxiv.org/abs/1411.4028)，[PennyLane QAOA](https://docs.pennylane.ai/en/stable/code/qp_qaoa.html)）。

最终实现选择周期环图 MaxCut：

```text
H on every qubit
repeat L times:
    RZZ(gamma_l) on every ring edge (even matching, then odd matching)
    RX(beta_l) on every qubit
```

每层参数向量严格为 `[beta_l, gamma_l]`，即总参数数 `2L`，不再是 `2qL`。这里 `beta_l` 定义为实际 `RX` gate angle；若采用文献中 `exp(-i beta sum X)` 的 evolution-time 记号，则对应 gate angle 是 `2 beta`，只是参数归一化不同。cost 为

```text
H_C = 1/2 sum_(i,j in ring) (Z_i Z_j - I)
```

其期望是负 cut size，与 PennyLane 的 MaxCut cost 约定一致（[官方 MaxCut 文档](https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.cost.maxcut.html)）。偶/奇 pairing 只是把周期环的全部边拆成两个 disjoint matchings，不改变 cost Hamiltonian；为避免 2-qubit 周期环重复同一无向边，当前要求偶数 `q>=4`。

### 12.2 共享参数 forward/backward

- forward 的所有 RX 从同一个 coefficient 读取，所有 RZZ 从同一个共享角 lookup 读取；
- backward 的每个 RX gate overlap 原子累加到同一个 `dE/dbeta_l`；
- cost backward 一次扫完整 state，用 `sum_edges z_i z_j` 乘端点 overlap，只输出一个 `dE/dgamma_l`，然后同时逆演化 `phi/lambda`；
- 因而 API 返回的梯度长度与优化变量完全一致，不需要在 Python 端再把 per-gate gradient 相加。

首层直接生成 `RZZ(gamma_0)|+>`；后续 cost-before-RX 融合只在 20q/28q 默认开启，因为第 8 节已证明它不是无条件正优化。

主表中标准 QAOA 相对 PennyLane 为 `7.27×–40.78×`；28q total 为 `3.304 s`，PennyLane 为 `33.374 s`。能量最大误差 `7.3e-14`，gradient element 最大误差约 `9.7e-13`。

## 13. XXZ-HVA

Hamiltonian Variational Ansatz 的合理做法是让 ansatz 的局部门对应目标 Hamiltonian 的项。最终加入的周期 XXZ-HVA 使用 Néel 初态与固定 `Delta=0.5`：

```text
H_XXZ = sum_i (X_i X_(i+1) + Y_i Y_(i+1) + 0.5 Z_i Z_(i+1))

for layer l:
    for even bonds, then odd bonds:
        RXX(theta_x[l,left])
        RYY(theta_y[l,left])
        RZZ(theta_z[l,left])
```

这与 HVA 文献对 XXZ/TFIM 的构造方向一致（[Wiersema et al., 2020](https://arxiv.org/abs/2008.02941)）。参数按 layer/axis/edge 独立，总数 `3qL`；偶/奇 matching 保留了相邻非对易 bond layer 的顺序。

### 13.1 RXX/RYY 的 amplitude pairing 与合并

对同一 bond，RXX/RYY/RZZ 两两对易。RXX 与 RYY 都把

```text
00 <-> 11
01 <-> 10
```

配对，但 `YY partner = -(z_i z_j) * XX partner`。设三类门的 half-angle 系数为 `(c_x,s_x)` 等，则一次更新可写为

```text
a = c_x c_y + s_x s_y z
b = s_x c_y - c_x s_y z
psi' = exp(-i theta_z z/2) * (a psi - i b psi_partner)
```

因此 kernel 不执行三次 full-state pass：bond 的两个 qubit 被放到 tile 中相邻 slot，tile amplitude 写入 shared mailbox 一次，thread 读 `local xor pair_mask` 的 partner 后完成合并矩阵。每个 matching 的 bond 数超过 tile 容量时才切到下一 phase；最后 phase 的 `pair_count` 可以自然变短，不做无效门。

backward 在逆演化前同时规约 `XX partner`、`-z*partner` 与 `z*self` 三个 generator overlap，再用负 half-angle 对 `phi/lambda` 做同一个合并逆矩阵。三类门在同 bond 上对易，所以可以用层端状态求梯度；even/odd matching 之间一般不对易，仍按 odd 后 even 逆序处理。

主实验相对 PennyLane 为 `4.68×–63.54×`；28q total 为 `7.231 s`，PennyLane 为 `52.107 s`。所有 4–28q 能量/梯度逐元素对齐，主实验最大 gradient element error 为 `3.53e-15`。

## 14. 结论

1. 最大且最稳定的收益来自减少 full-state pass，而不是单纯增加单 phase 的门数。
2. 首层直接生成是无条件正优化；RA 实数特化也是无条件正优化，两者默认开启。
3. RX/RY 理论代价相近；旧 RY 的显著劣化来自 warp 分支，修复后同条件差距缩到约 0%–7%。二者的 backward 仍是最难路径。
4. 高位 full-tile 的不连续 warp 访问确实更慢，但固定低 5 位会减少每 phase 门数；最终布局必须结合方向、电路与规模选择。
5. RZ/ZZ 不应在 device 内逐 amplitude 算三角函数；host chunk lookup 明显更快。
6. 对角 backward 能用层端点一次求全部梯度；但“融合到 RX phase”与“独立单 pass”仍需按规模选择。
7. `fixed RY + trainable RZ + fixed RY` 的 RX 分解慢 2.0×–2.36×，已丢弃；可训练 phased `RY+RZ` 只在 28q 获益，但其最后 phase 可以继续融合 CNOT。
8. 旧 QAOA 是非共享参数的 TFIM/HVA-like ansatz；最终版本改为 cost-first、每层两个共享参数的环形 MaxCut QAOA，目标 Hamiltonian、参数数和梯度语义均与参考实现对齐。
9. XXZ-HVA 必须按双比特 partner 处理 RXX/RYY；同 bond 的 XX/YY/ZZ 合并能避免三次 state pass，但三个 gradient reduction 使 backward 仍是主要成本。

## 15. 复现

```bash
make -C sad
PYTHONPATH=sad/python:pennylane-lightning/src \
  .venv/bin/python -m pytest sad/tests pennylane-lightning/tests -q

PYTHONPATH=sad/python .venv/bin/python benchmark/benchmark_sad.py
PYTHONPATH=sad/python .venv/bin/python benchmark/benchmark_qaoa_sad.py
PYTHONPATH=pennylane-lightning/src \
  .venv/bin/python benchmark/benchmark_qaoa_xxz_pennylane.py
.venv/bin/python benchmark/summarize_research.py
.venv/bin/python benchmark/benchmark_ablation.py
```

微基准源码：

- `benchmark/microbench_rotation.cu`：tile、低/高 phase、fixed-low；
- `benchmark/microbench_diagonal.cu`：device sincos 与 chunk lookup；

实验执行模式：`legacy`、`initial-only`、`fused-forward`、`phased-forward`、`optimized`、`all-fused`。正式主实验使用 `optimized`。
