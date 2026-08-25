# QAOA-BD 搜参方案

## 1. 目标

QAOA-BD 搜索的是 SAD 的编译期执行配置，不是 QAOA 的物理参数。`beta`、
`gamma` 由固定 seed 生成，搜索对象为：

```text
BD cost split/fusion
matching CNOT/RZ kernel 的 block/register shape
ordinary block threads
forward/backward shape
qubit-dependent library dispatch
```

目标优先级：

1. 4–28q 全部 correctness 通过；
2. forward/backward/total time 较低；
3. 显存和 workspace 可控；
4. 多次运行稳定；
5. 不影响原始 `qaoa` 和其他电路。

## 2. BD 专用搜索轴

旧 QAOA 的 `SAD_QAOA_FUSE_COST_RX`、`SAD_QAOA_FUSED_BACKWARD`、compact lookup
和 initial strategy 不适用于 BD。BD 应使用自己的开关：

```text
SAD_QAOA_BD_FUSION=0  # matching CNOT、RZ、CNOT 分开
SAD_QAOA_BD_FUSION=1  # matching 内 fused CNOT-RZ-CNOT
```

当前 P1 matching 结构为：

```text
even matching: CNOT matching -> RZ matching -> CNOT matching
odd matching:  CNOT matching -> RZ matching -> CNOT matching
```

P2 fusion 后每个 matching 只需要一个 cost kernel。不能把 BD 映射到旧的
ring-RZZ compact/fused kernel。

## 3. 固定实验条件

第一轮固定：

```text
qubits       = 4,6,8,10,12,14,16,18,20,22,24,26,28
layers       = 8
precision    = float64
random_seed  = 42
batches      = 1
execution    = optimized
```

建议沿用已有 benchmark schedule：

| qubits | warmup | measured steps |
|---:|---:|---:|
| 4–8 | 5 | 20 |
| 10–14 | 5 | 10 |
| 16–20 | 5 | 5 |
| 22–24 | 1 | 3 |
| 26–28 | 1 | 2 |

每条结果都保存 forward、Hamiltonian、backward、total、显存和 workspace。

## 4. 搜索轴和候选

### 4.1 Fusion

第一轮只搜索：

| variant | flags | 说明 |
|---|---|---|
| `bd-matching-split` | `-DSAD_QAOA_BD_FUSION=0` | 严格 split oracle |
| `bd-matching-fused` | `-DSAD_QAOA_BD_FUSION=1` | matching 内融合 |

两个方向必须独立记录。fused forward 有收益但 backward 退化时，不能用 total
time 掩盖方向性结果。

### 4.2 Block/register shape

RX mixer 仍使用 tiled rotation 路径，因此 forward/backward shape 会影响
QAOA-BD 的完整 energy+gradient。脚本搜索：

```text
SAD_FORWARD_BLOCK_THREADS / SAD_FORWARD_REGISTER_BITS
SAD_BLOCK_THREADS / SAD_REGISTER_BITS
```

线程候选来自 `SHAPE_GRID`，register bits 为 `2,3,4`，并通过
`valid_shape` 过滤。forward 与 backward 独立选择 shape。

### 4.3 Ordinary threads

matching CNOT、matching RZ 和 ordinary kernel 使用
`kOrdinaryBlockThreads`，搜索：

```text
SAD_ORDINARY_BLOCK_THREADS=64,128,256
```

第一轮不加入 512，除非 profile 证明 occupancy 足够。Hamiltonian 占比很低时，
不能用它的局部收益决定完整 variant。

### 4.4 后续 reduction 轴

当前 gamma gradient 使用 atomicAdd。完成 shape 筛选后再搜索：

```text
atomic gamma reduction
partial buffer + second reduction
matching-local reduction
```

不要在第一轮把 reduction、fusion、shape 做全笛卡尔积。

## 5. 分阶段流程

### Stage 0：smoke/correctness

候选：`split + f128r2_b128r2`、`fused + f128r2_b128r2`。

规模：4q/1L、4q/8L、6q/8L、8q/8L。

检查：

- PennyLane energy/full gradient；
- split 与 fused 对齐；
- legacy 与 optimized 对齐；
- wrap-around edge；
- finite difference；
- compute-sanitizer。

失败候选不得进入 timing 搜索。

### Stage 1：fusion A/B

固定 `f128r2_b128r2`，测试 `SAD_QAOA_BD_FUSION=0/1`，规模为：

```text
8q, 16q, 20q, 24q, 28q
```

每个候选至少 5 个独立 batch，记录 median、P95、标准差，判断：

- fused forward 是否获益；
- fused backward 是否获益；
- fused 是否导致 28q 显存或稳定性问题。

### Stage 2：shape screen

固定 fusion winner，分别搜索 forward 和 backward shape；每个阶段按
`forward_ms` 或 `backward_ms` 选出 qubit-specific winner。

### Stage 3：ordinary threads

matching CNOT、matching RZ、pair-CNOT 和状态初始化使用
`SAD_ORDINARY_BLOCK_THREADS`，搜索 `32,64,128`。32 线程（单 warp）是合法
配置，不能预先排除。

### Stage 4：reduction

脚本最后搜索公共 RX mixer reduction 组合
`(legacy_reduction, rotation_warp_atomic) = (1,0), (0,0), (0,1)`，因为该轴
确实影响每层 mixer 的 backward。每个候选仍以完整 energy+gradient median
排序，并通过 correctness gate。

### Stage 5：完整确认

对最终候选运行完整 `4,6,...,28q` sweep，至少 5 个独立 timing batch，确认：

- 4–28q 全部成功；
- median/P95 无异常尖峰；
- 28q 无 OOM；
- energy/gradient 全部通过；
- winner 不因小规模胜出而造成大规模退化。

## 6. 搜索脚本建议

实现脚本为独立的 `benchmark/search_qaoa_bd_parameters.py`，复用
`search_circuit_common.py` 的 candidate library、CSV resume 和 JSON summary，
不修改旧 `qaoa`/`qaoa-ns` 搜参表。阶段顺序为 `fusion`、`forward_shape`、
`backward_shape`、`ordinary_threads`、`reduction`；每个 qubit 的后续阶段读取
前一阶段最快的完整配置。

默认输出：

```text
benchmark/results/qaoa_bd_parameter_search.csv
benchmark/results/qaoa_bd_parameter_search.json
```

CSV 字段至少包括：

```text
circuit, qubits, layers, stage, config, flags
forward_ms, hamiltonian_ms, backward_ms, median_ms, status, error
```

## 7. Correctness gate

每个候选必须先通过 correctness，再计时：

```text
float64 energy absolute error  <= 1e-10
float64 gradient max error     <= 1e-9
float32 energy/gradient error  <= 3e-5
```

必须覆盖：

- CNOT-RZ-CNOT 与 IsingZZ 矩阵等价；
- split/fused 对齐；
- 4q/6q/8q/28q；
- beta、gamma 有限差分；
- odd/even matching backward 逆序。

## 8. Winner 选择

每个 qubit 独立选择：

1. correctness 不通过淘汰；
2. OOM、非法访问、NaN 淘汰；
3. 以 full energy+gradient median 为主排序；
4. forward/backward 方向结果作为约束；
5. median 差距小于 2% 时选择 P95 更低者；
6. P95 接近时选择 workspace 更小者；
7. 至少两个独立 batch 稳定胜出后才写入 production dispatch。

最终 dispatch 必须记录：

```text
qubits, fusion_mode, forward shape
backward shape, ordinary threads, benchmark evidence
```

## 9. 完成标准

- 有可恢复、可追加的 BD 专用搜参脚本；
- split/fused 两种模式都有独立结果；
- 4–28q 全部通过 correctness；
- register bits=2,3,4、线程 32/64/128 的 shape/ordinary 搜索覆盖所有目标 qubit；
- 结果包含 compile flags、library、execution mode 和 error；
- 28q 没有未解释的 error；
- winner 经过重复确认后才进入 BD dispatch；
- 原始 QAOA 的搜参结果和 dispatch 不被修改。
