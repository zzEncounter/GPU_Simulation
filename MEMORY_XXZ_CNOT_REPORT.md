# L2、XXZ 配对与 ring-CNOT 后续研究

## 1. 结论先行

本报告延续 `ROTATION_OPTIMIZATION_REPORT.md`。硬件仍是 NVIDIA RTX 6000 Ada
Generation、SM 8.9、48 GiB，CUDA 13.2；日期为 2026-08-12，主实验使用
`complex<double>`。

1. **L2 与 VRAM 在地址和正确性上透明，在性能上并不等价。** 本机 L2 为
   96 MiB，但最多只有 66 MiB 可设为 persisting，access-policy window 约
   128 MiB。steady-state 下 16/32 MiB state 的三个 RY phases 几乎不再从
   DRAM 读取；64 MiB state 已发生合计 76.5 MiB DRAM read，128/256 MiB
   state 则接近每 phase 重读完整 state。不能用“state 小于标称 L2”推导
   100% 常驻。
2. **`arg_l2` 分块想法语义正确，但当前原型只有很窄的收益窗口。** 把
   18 个低位旋转按 64 MiB chunk 完成，再做高位 phase，q23/q24 分别快
   5.3%/1.5%；q25/q26 慢 1.2%/0.5%。若严格把 22 个低位全放在 chunk
   内，会多出一次 full-state pass，q24 反而慢 5.6%。L2 persisting hint
   只在 64 MiB state 上快 6.7%，128/256 MiB 上慢 2.7%/4.0%，不应成为
   默认路径。
3. **XX/YY 的 phase tile bit 数不需要是偶数。** 当前 9-bit tile 每 phase
   放 4 个完整 bond，最后 1 bit 是 filler。两个 bond qubit 都在 tile 内时，
   `local ^ ((1<<a)|(1<<b))` 就枚举了 `00↔11`、`01↔10`；pair 跨过
   lane/register/warp slot 也由 whole-tile shared mailbox 正确处理。
4. **偶数 tile 是性能选择，不是正确性要求。** `128×8` 的 10-bit tile
   在 q26 XXZ forward/backward 比当前 9-bit tile 快 14.5%/3.9%；但 q24
   的 `32×8` 8-bit tile 反而是三者中最好。12-bit backward 因静态 shared
   memory 需要 0x10410 bytes、超过 ptxas 的 0xc000 上限而不能编译。因此
   应按方向和 q dispatch，不能统一规定偶数 tile。
5. **同一 bond 的 XX/YY/ZZ 已融合，even/odd matching 尚未融合。** 两层
   一般不对易，不能直接合并；但可保持依赖地做部分融合：一个连续 qubit
   block 内先做 even bonds，再做内部 odd bonds，最后做 block 边界 odd
   bonds。benchmark-only forward 原型把 q24 的 6 次 full-state pass 降到
   4 次并快约 9%，q26 快 15%–21%，q20 数值对照最大误差为 0。它还没有
   backward 实现，暂不应直接进入生产路径。
6. **ring-CNOT 使用双 state buffer 是正确的带宽取舍；当前 standalone
   forward 的 gather 方向却不是最佳。** q24/q26 forward gather 为
   1.067/4.184 ms，scatter 为 0.696/2.977 ms，快 34.7%/28.8%。adjoint
   正相反：当前 gather 为 0.688/3.024 ms，scatter 为 1.696/7.068 ms。
   所以候选改法是 **forward scatter、adjoint gather**，不是删除双缓冲或
   无条件改 scatter。
7. **单 buffer 逐个 CNOT 原地交换不值得。** q24/q26 分别需要
   8.996/39.146 ms，是最佳双 buffer 方向的 12.9×/13.2×。更复杂的原地
   permutation-cycle 算法会失去规则并行性；而 workspace 和融合 backward
   本来就需要 `phi_a/b`、`lambda_a/b`，不能只改一个 kernel 就省掉 buffer。

普通 complex RX/RY 已在 commit `02443c4` 中设为默认；本报告没有继续修改
生产 kernel，只加入可复现的研究基准。

## 2. 数据与方法

新增原始数据：

- `benchmark/results/rotation_l2.csv`：L2 capacity、persisting 和分块计时；
- `benchmark/results/rotation_l2_ncu.csv`：逐 phase L2/DRAM counters；
- `benchmark/results/xxz_tiles.csv`：XXZ 8–11 bit tile forward/backward；
- `benchmark/results/xxz_integrated.csv`：even/odd 部分融合 forward 原型；
- `benchmark/results/cnot_layout.csv`：copy、gather、scatter、dual 和 in-place。

所有计时均 warmup 3 次；每个配置独立运行 5 次，表中取每次平均值的 median。
L2 counters 使用 Nsight Compute application replay、`--cache-control none`，
避免 profiler 默认在 replay 前清空 cache。NVIDIA 也建议对 cache-sensitive
workload 使用这种组合；同时说明禁止清 cache 时 replay 间可能有波动，甚至
ratio 可能短暂超界，所以本报告把 DRAM bytes 与非 profiler wall time 作为
主要证据，而不把 100% 附近的小差异过度解释：
[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。

## 3. L2：实际命中边界

### 3.1 设备上限

`cudaGetDeviceProperties` 和 `cudaDeviceGetLimit` 的实测值：

```text
l2CacheSize                    100663296 B = 96 MiB
persistingL2CacheMaxSize        69206016 B = 66 MiB
accessPolicyMaxWindowSize      134213632 B ≈ 128 MiB - 4 KiB
default persisting set-aside    18874368 B = 18 MiB
```

CUDA 的 access-policy window 只提高指定地址被保留的优先级，不会 pin cache
line，也不改变一致性或地址语义。超过 set-aside 时收益会下降并可能因 thrash
退化；使用结束后还应 reset persisting 状态：
[CUDA Programming Guide: L2 Cache Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html)。

一个 `complex<double>` state 的大小为：

| qubits | state |
|---:|---:|
| 20 | 16 MiB |
| 21 | 32 MiB |
| 22 | 64 MiB |
| 23 | 128 MiB |
| 24 | 256 MiB |
| 25 | 512 MiB |
| 26 | 1 GiB |

当前 9-bit rotation tile 已经在 registers/shared 中一次完成一个 phase 的最多
9 个单 qubit gates。因此 q24 的完整 RY layer 是 3 次 state pass，不是 24
次；`arg_l2` 要减少的是这 3 次 pass 之间的 DRAM traffic。

### 3.2 steady-state counters

下表是 warmup 后连续三个 ordinary RY phases。`DRAM read` 是三个 phase 的
合计，括号内是各 phase 的 read hit rate；接近 100% 的值按“约 100%”处理。

| q | state | DRAM read | 各 phase L2 read hit |
|---:|---:|---:|---|
| 20 | 16 MiB | ≈0 MiB | ≈100%, ≈100%, ≈100% |
| 21 | 32 MiB | 2.0 MiB | 97.8%, 99.2%, 99.9% |
| 22 | 64 MiB | 76.5 MiB | 66.1%, 72.9%, 89.1% |
| 23 | 128 MiB | 345.1 MiB | 15.2%, 50.9%, 20.9% |
| 24 | 256 MiB | 750.3 MiB | 4.1%, 50.9%, 5.2% |

q23 的三次纯 state read 理论量是 384 MiB，q24 是 768 MiB；实测已经很接近。
phase 2 的 hit rate 较高并不代表少读一半 state：该 phase 的 selected mapping
使 L2 request 数不同，DRAM bytes 才是跨 phase 复用的直接量。

“整个 state 小于 96 MiB”仍不能保证常驻，原因包括：

- L2 同时容纳 dirty state lines、写回、coefficients、maps 和其他请求；
- cache 是有限 associativity/replacement 的共享资源，标称容量不是可分配数组；
- 每个 phase 的读写顺序和 physical-to-local bit mapping 不同；
- stores 最终仍可能写回 DRAM，读命中不等于完全没有 DRAM write。

### 3.3 persisting access-policy window

实验把 set-aside 调到 66 MiB；window 覆盖 state（或设备允许的最大范围），
`hitRatio=min(1, 66MiB/window)`，hit 为 `Persisting`、miss 为 `Streaming`。

| q | normal | persisting | 结果 |
|---:|---:|---:|---:|
| 20 | 0.1648 ms | 0.1647 ms | 持平 |
| 21 | 0.3090 ms | 0.3089 ms | 持平 |
| 22 | 0.6553 ms | **0.6143 ms** | 快 6.7% |
| 23 | **1.4957 ms** | 1.5373 ms | 慢 2.7% |
| 24 | **3.0975 ms** | 3.2280 ms | 慢 4.0% |

q20/q21 本来就命中；q22 刚好能放进 66 MiB set-aside，得到有限收益；q23
以后只能概率性保留一部分，提示成本与替换竞争抵消收益。

PTX 还提供 `.cg/.cs`、L2 prefetch、`createpolicy ... evict_last` 等 cache
operator。官方定义明确说 eviction policy 与 prefetch 都只是 performance
hint，可能不被执行，也不改变一致性：
[PTX ISA: Cache Eviction Priority Hints](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-eviction-priority-hints)。
本轮优先测试了可移植且覆盖整个 kernel sequence 的 Runtime access-policy
window。对 SM 8.9 再手写 per-load PTX 不会突破 66 MiB 物理 set-aside；在
Runtime hint 已于 q23+ 退化的证据下，它只值得作为独立微调，不值得先进入
生产 kernel。

## 4. `arg_l2` 分块原型

### 4.1 做法与等价性

原型把 state 按高位 prefix 切成连续 `2^chunk_bits` amplitudes。对每个 chunk，
以该 chunk pointer 为局部 state，对低位 rotation phases 逐一 launch；所有
chunk 完成后，再对剩余高位 targets 做普通 full-state phase。

同轴、不同 qubit 的 RX 或 RY 彼此对易，因此这种重排与原 layer 等价。q16/q20
逐 amplitude 对照的最大误差为 0。

有两个参数必须分开：

- `chunk_bits` 决定 L2 working set；
- `low_targets` 决定在 chunk 内完成多少 gate。

若 `low_targets=22`，9-bit tile 需要 3 个低位 phases，剩余高位还需 1 个 phase，
q24 从原来的 3 passes 增到 4。为保持总 pass 数，主实验使用
`chunk_bits=22, low_targets=18`：64 MiB chunk 内做两个完整 9-target phases，
剩余 6–8 个 targets 用一个 full-state phase。

### 4.2 结果

| q | baseline | 64 MiB chunk / 18 low targets | 结果 |
|---:|---:|---:|---:|
| 22 | **0.6138 ms** | 0.7011 ms | 慢 14.2% |
| 23 | 1.5612 ms | **1.4777 ms** | 快 5.3% |
| 24 | 3.1330 ms | **3.0857 ms** | 快 1.5% |
| 25 | **6.2083 ms** | 6.2826 ms | 慢 1.2% |
| 26 | **13.4326 ms** | 13.4964 ms | 慢 0.5% |

q24 counters 显示 blocked 原型把 DRAM read 从 baseline 的约 750 MiB 降到
586 MiB，说明 L2 复用是真实的；但 kernel launch 数从 3 增到 9，分块 phase
的 mapping/尾部开销和 DRAM writes 抵消了大部分收益。

严格 cutoff 的 q24、64 MiB chunk 结果：

| low targets | total passes 形态 | normal |
|---:|---|---:|
| 18 | 2 chunk-local + 1 global | **3.0857 ms** |
| 19 | 3 chunk-local + 1 global | 3.3583 ms |
| 20 | 3 chunk-local + 1 global | 3.2532 ms |
| 21 | 3 chunk-local + 1 global | 3.3012 ms |
| 22 | 3 chunk-local + 1 global | 3.3073 ms |

这再次说明最重要的约束是 full-state pass/phase 数，而不是尽量让
`arg_l2` 靠近 L2 容量。

### 4.3 建议

当前不应把 `arg_l2` 分块设为默认。若继续研究，顺序应是：

1. 只测试 phase-aligned 的 `low_targets`，禁止因 cutoff 增加 pass；
2. 把 q23/q24 当候选 dispatch 区间，不外推到更大 q；
3. 用 CUDA Graph 降低每 chunk 多 launch 的 host overhead；
4. 评估真实 circuit 中前后 diagonal/CNOT fusion 是否会被分块破坏；
5. 只有端到端仍稳定超过 5% 才考虑生产实现。

## 5. XX/YY 配对与 tile 奇偶性

### 5.1 为什么 9-bit tile 正确

对 bond `(a,b)`，XX 和 YY 都只在下列两对 basis states 间耦合：

```text
00 <-> 11
01 <-> 10
```

这等价于 partner local index：

```text
partner = local ^ (1 << slot_a) ^ (1 << slot_b)
```

只要 `a,b` 两个 logical qubit 都被 selected 到当前 tile，tile 的 `2^t`
个 local indices 自然包含四种组合；`t` 本身无需为偶数。

当前 `build_bond_phase_maps` 的规则是：

```text
pairs_per_phase = tile_bits / 2
bond 0 -> slots (0,1)
bond 1 -> slots (2,3)
...
unused odd slot -> filler qubit
```

默认 `128 threads × 4 amplitudes/thread` 的 slot 划分为：

```text
slots 0..4 : lane bits
slots 5..6 : register bits
slots 7..8 : warp bits
```

四个 pairs 是 `(0,1),(2,3),(4,5),(6,7)`，slot 8 是 filler。第三个 pair
跨 lane/register，第四个跨 register/warp。XXZ kernel 不是分别调用
lane shuffle/register swap，而是先把整个 tile 写入 shared mailbox，再用 XOR
index 取 partner；因此跨边界只影响成本，不影响正确性。一个 warp 恰好 32
threads 也没有遗漏 `00/01/10/11`。

### 5.2 8/9/10-bit tile 实测

表中对 even/odd matching 两个 parity 的 median 再取平均。

| q | direction | `32×8` tile8 | 当前 `128×4` tile9 | `128×8` tile10 |
|---:|---|---:|---:|---:|
| 20 | forward | 0.2643 ms | 0.2751 ms | **0.2538 ms** |
| 20 | backward | **0.8133 ms** | 0.8913 ms | 0.8784 ms |
| 24 | forward | **5.3913 ms** | 5.7250 ms | 5.5177 ms |
| 24 | backward | **15.7117 ms** | 17.1707 ms | 16.2499 ms |
| 26 | forward | 29.9218 ms | 29.7190 ms | **25.3951 ms** |
| 26 | backward | **66.9110 ms** | 71.9259 ms | 69.0920 ms |

tile10 每 phase 可放 5 bonds；q26 matching 从 4 phases 降到 3，所以 forward
收益最大。tile8 仍只有 4 bonds/phase，却用更小 mailbox 和单 warp，在 q24
backward 最好。结果证明 dispatch 要同时看 phase count、mailbox、registers、
active CTAs 与 gradient reduction。

tile12 每 phase 可放 6 bonds，但 backward 的
`mailbox + reduction + tile_base = 0x10410 bytes` 静态 shared，超过 ptxas
在本目标架构接受的 `0xc000`，不能编译。即使 forward 可编译，之前的 isolated
结果也比 tile10 慢；不应继续用“更大偶数 tile”硬推。

### 5.3 even/odd 当前做了什么

同一 bond 上的 RXX、RYY、RZZ 两两对易，当前 `apply_xxz_bond` 已将三者合成
一次 two-amplitude update 和一个 diagonal phase。这部分已经完成。

一个 matching 内的 bonds 不共享 qubit，也被同一个 kernel/phase schedule
处理。完整 layer 仍是：

```text
launch even matching
launch odd matching
```

相邻 bond 的完整 XXZ 生成元一般不对易。例如 `X0X1` 与 `Y1Y2` 在 qubit 1
反对易，因此不能把所有 even/odd gates 当作一个 commuting set。

### 5.4 可保持依赖的部分融合

对连续偶数宽 block，可做：

```text
block 0: all even bonds -> internal odd bonds
block 1: all even bonds -> internal odd bonds
...
final: all boundary odd bonds
```

被提前的 internal odd bond 只跨过其他 block 的 disjoint even bonds，因此与
原 `all even -> all odd` 顺序严格等价。benchmark-only kernel 支持任意
`(slot_a,slot_b,edge)` 顺序，q20 对照误差为 0。

| tile | q | baseline passes | integrated passes | baseline | integrated | speedup |
|---|---:|---:|---:|---:|---:|---:|
| tile9 | 20 | 6 | 4 | **0.5292 ms** | 0.5558 ms | 慢 5.0% |
| tile9 | 24 | 6 | 4 | 11.5809 ms | **10.6569 ms** | 8.7% |
| tile9 | 26 | 8 | 5 | 58.1131 ms | **46.0094 ms** | 26.3% |
| tile10 | 20 | 4 | 3 | 0.5653 ms | **0.4924 ms** | 14.8% |
| tile10 | 24 | 6 | 4 | 10.9604 ms | **9.9784 ms** | 9.8% |
| tile10 | 26 | 6 | 4 | 49.6376 ms | **42.1738 ms** | 17.7% |

这里的 speedup 是 `baseline/integrated - 1`。q20 tile9 pass 很短，通用 schedule
与更多 per-phase metadata 反而贵；大 q 的 DRAM/pass 节省才占主导。

这项是明确的高价值后续方向，但生产落地前必须补：

- backward 逆序 schedule、三轴 gradient 和数值验证；
- wrap-around boundary 与不同 q/tile 宽度的通用 packing；
- 与 ordinary-per-phase XXZ 对照，避免重犯 cooperative persistent 的问题；
- 完整 XXZ-HVA step，而不是只看 forward matching。

## 6. ring-CNOT：双缓冲和连续性

### 6.1 当前 standalone kernel

当前 kernel 按连续 `output_index` 遍历：

```text
forward: output[y] = input[P^-1(y)]
adjoint: output[y] = input[P(y)]
```

所以 output stores 连续，input loads 是 gather。完成后 `StatePair` 交换
`current/scratch`。backward 可在同一个 kernel 中同时移动 phi 和 lambda。

对一般 permutation，一边连续就必然让另一边置换；双 buffer 允许每个 amplitude
只读一次、只写一次，并避免 in-place 的跨 thread race。它是合理的基础结构。

### 6.2 这个 ring permutation 的特殊 coalescing

CUDA global memory 以 32-byte transaction 合并一个 warp 的请求；地址不必按
thread 顺序排列，只要落在尽量少的 32-byte segments 中即可：
[CUDA Programming Guide: Coalesced Global Memory Access](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#coalesced-global-memory-access)。

一个 `complex<double>` 是 16 bytes，理想情况下 32 threads 访问 32 amplitudes
需要 16 个 sectors。对任意 32-aligned warp base，q20–q26 的精确地址枚举为：

| operation | permuted side | 32-byte sectors/warp |
|---|---|---:|
| forward gather | read `P^-1(y)` | 32 |
| forward scatter | write `P(x)` | 16 |
| adjoint gather | read `P(y)` | 16 |
| adjoint scatter | write `P^-1(x)` | 32 |

虽然 `P(x)` 的 destination index 在数值上会跳到 state 两端，但同一 warp 的
32 个 destinations 仍两两落在相同 32-byte sector。因此“scatter store 一定
比 gather load 更差”在这个特定线性 permutation 上不成立。

### 6.3 单 state forward

| q | copy | 当前 gather | scatter | in-place q 个 CNOT |
|---:|---:|---:|---:|---:|
| 22 | 0.1417 ms | 0.1641 ms | **0.1364 ms** | 0.3927 ms |
| 24 | 0.7321 ms | 1.0668 ms | **0.6962 ms** | 8.9959 ms |
| 26 | 2.8765 ms | 4.1838 ms | **2.9769 ms** | 39.1463 ms |

q24 scatter 比 gather 快 34.7%，q26 快 28.8%，并已接近纯 copy。q22 scatter
略快于 copy 属于 cache/timing 波动，不应解释为超过物理 copy；结论只需要它
稳定显著快于 gather。

单 buffer 原型逐个 control launch 一个 race-free swap kernel，每个 CNOT 都要
读写完整 state，因而 ring 需要 q 次 pass。q24/q26 比 scatter 慢 12.9×/13.2×。
用 permutation cycle 让每个 element 只移动一次，必须选 cycle leader、串行搬运
不规则长 cycle，并防止多个 threads 认领同一 cycle；这会用并行性换 memory，
没有证据能胜过近 copy-bandwidth 的双 buffer。

### 6.4 adjoint 与 dual state

| q | phi forward gather | phi forward scatter | phi adjoint gather | phi adjoint scatter |
|---:|---:|---:|---:|---:|
| 22 | 0.1641 | **0.1364** | **0.1442** | 0.2427 |
| 24 | 1.0668 | **0.6962** | **0.6878** | 1.6962 |
| 26 | 4.1838 | **2.9769** | **3.0237** | 7.0685 |

| q | phi+lambda forward gather | forward scatter | adjoint gather | adjoint scatter |
|---:|---:|---:|---:|---:|
| 22 | 0.4341 | **0.3293** | **0.3242** | 0.7475 |
| 24 | 2.1326 | **1.4590** | **1.3999** | 3.4386 |
| 26 | 8.3676 | **5.6492** | **5.7694** | 14.2288 |

方向选择在单/双 state 上一致：

```text
forward U      : contiguous input + scatter P(x)
backward U^dag : gather P(y) + contiguous output
```

### 6.5 生产路径中还有多少 standalone CNOT

- RA-HEA optimized forward：CNOT 已融合到 product initialization 或最后一个
  RY phase 的 scatter store；
- RA-HEA optimized backward：CNOT gather 已融合到第一个 reverse RY phase；
- SU2-HEA optimized forward：普通 fused 或 phased-RY kernel 都在最后 store
  融合 CNOT；
- SU2-HEA optimized backward：q21–q27 使用 fused gather；q20 和 q≥28 回退到
  standalone adjoint CNOT；
- legacy/baseline execution mode：forward/backward 都使用 standalone CNOT；
- RZZ/QAOA/XXZ-HVA 没有 ring-CNOT layer。

因此最高优先级仍是保留现有融合：即使 fused forward 的 scatter store 不如纯
copy 连续，它省掉了一次完整 state read+write。standalone forward scatter
主要改善 baseline、调试路径和未来不能融合的 circuit；standalone adjoint
当前方向已经正确。

workspace 对 complex path 已分配 `phi_a/b` 与 `lambda_a/b`，fused backward
也要从 input pair 写到 output pair。只把 CNOT 改成原地不会让这些 buffer 消失；
要省显存必须重新设计整个 adjoint/fusion 生命周期，代价远超 CNOT kernel。

## 7. 建议的落地顺序

1. 保持 commit `02443c4` 的 ordinary RX/RY 默认。
2. **可低风险落地：** standalone ring-CNOT 在 `adjoint=false` 时改为 scatter，
   `adjoint=true` 保持 gather；先跑 RA/SU2 baseline 和 q≥28 SU2 端到端。
3. **高价值研究：** 实现 XXZ dependency-preserving integrated backward，再做
   XXZ-HVA 完整 step；同时把 tile8/tile10 做成 direction/q dispatch 候选。
4. L2 access-policy 只保留实验开关；不默认 persisting。
5. `arg_l2` 分块只有在 phase 数不增加、真实 circuit 不破坏已有 fusion、且
   q23/q24 端到端稳定超过 5% 时才落地。
6. 不采用单-buffer 逐 CNOT，也不尝试 tile12 backward。

## 8. 复现

```bash
# 设备 L2 属性
nvcc -std=c++17 -O3 benchmark/query_cuda_cache.cu -o /tmp/query_cuda_cache
/tmp/query_cuda_cache

# L2 timing 与逐 phase Nsight counters
.venv/bin/python benchmark/benchmark_rotation_l2.py \
  --repetitions 5 --iterations 10
.venv/bin/python benchmark/profile_rotation_l2.py

# XXZ tile 与 dependency-preserving forward 原型
.venv/bin/python benchmark/benchmark_xxz_tiles.py \
  --repetitions 5 --iterations 10
.venv/bin/python benchmark/benchmark_xxz_integrated.py \
  --repetitions 5 --iterations 10

# ring-CNOT layouts
.venv/bin/python benchmark/benchmark_cnot_layout.py \
  --repetitions 5 --iterations 10

# 生产库与完整回归
make -C sad
PYTHONPATH=sad/python:pennylane-lightning/src \
  .venv/bin/python -m pytest sad/tests pennylane-lightning/tests -q
```
