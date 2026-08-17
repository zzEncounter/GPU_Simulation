# 非共享角 QAOA（qaoa_ns）电路实现计划

## 1. 目标与范围

本计划新增独立的 `qaoa-ns` 电路（代码讨论中简称 `qaoa_ns`），实现环形 MaxCut 上的非共享角 QAOA。现有 `qaoa` 保持语义、参数量、ID、benchmark 和结果不变。

新电路仍使用：

```text
H^n -> for layer in [0, L):
           ring-RZZ(gamma[layer, edge])
           RX(beta[layer, wire])
```

区别是每个物理门都有独立参数：

| 电路 | RZZ 参数 | RX 参数 | 每层参数量 | 总参数量 |
|---|---|---|---:|---:|
| `qaoa` | 全边共享 `gamma_l` | 全 qubit 共享 `beta_l` | 2 | `2L` |
| `qaoa_ns` | 每条边独立 | 每个 qubit 独立 | `2n` | `2nL` |

本计划覆盖 PennyLane、native、SAD C++ executor、CUDA forward/backward、lookup、测试、benchmark 和文档。第一版先完成 correctness，融合 kernel 和专用 variant 放到后续阶段。

## 2. 对齐原则

| 层次 | 当前模式 | `qaoa_ns` 对齐方式 |
|---|---|---|
| PennyLane builder | 每类电路一个 `_circuit_name()` | 只新增 `_qaoa_ns()` |
| 注册 | `CircuitSpec` + 参数量 lambda | 规范名 `qaoa-ns`，别名 `qaoa_ns` |
| C++ 电路 | 一个 layout + executor 特化 | `QaoaNsLayerLayout` + 一个特化 |
| 参数量 | QAOA 在 `expected_parameter_count()` 特判 | 增加 `2 * qubits * layers` 分支 |
| RX | generic rotation kernel | `SharedParameter=false` |
| RZZ | 当前 shared ring kernel | 新增 nonshared ring kernel |
| Hamiltonian | QAOA ring MaxCut 分支 | 与 `qaoa` 复用 |

明确不做：

- 不修改 `qaoa` 的 `[beta_0,gamma_0,...]` 参数布局；
- 不根据参数长度让一个 circuit ID 隐式切换共享模式；
- 不修改 `CircuitSpec`、runner、context 的公共接口；
- 不把所有 ring RZZ 重构成全局 diagonal 框架；
- 不在第一版保留多套重复的 nonshared 顶层 kernel API；
- 不为每条边单独启动一个 kernel 作为最终实现。

## 3. 固定电路定义

### 3.1 Ring topology 与门序

边固定为：

```text
edge i = (i, (i + 1) mod n), i = 0 ... n-1
```

每层语义顺序固定为：

```text
for edge = 0 ... n-1:
    IsingZZ(gamma[layer, edge], edge, edge+1 mod n)
for wire = 0 ... n-1:
    RX(beta[layer, wire], wire)
```

CUDA 可以将边拆成 even/odd matching，但必须保持 cost 在 mixer 之前，并保证 matching 间有 state barrier。

### 3.2 参数布局

为与现有 `QaoaLayerLayout { beta, gamma }` 和 native QAOA 参数索引保持一致，采用 layer-major、beta-first 布局（参数布局不等于执行顺序）：

```text
layer_base(l) = 2 * l * n
beta(l, wire)  = layer_base(l) + wire
gamma(l, edge) = layer_base(l) + n + edge
```

即每层为：

```text
[beta_l_wire_0  ... beta_l_wire_(n-1),
 gamma_l_edge_0 ... gamma_l_edge_(n-1)]
```

总参数量：

```text
parameter_count = 2 * qubits * layers
```

### 3.3 输入约束

沿用现有 ring QAOA 约束：`qubits >= 4`、qubits 为偶数、`layers >= 1`。第一版不扩展 odd-qubit ring；不能只删除 even 检查而不重新设计 overlapping edge 的执行。

## 4. PennyLane reference

修改：

```text
pennylane-lightning/src/pennylane_lightning_baseline/circuits.py
```

### 4.1 Builder

只新增一个 builder，不新增 topology/parameter helper：

```python
def _qaoa_ns(params: object, qubits: int, layers: int) -> None:
    for wire in range(qubits):
        qml.Hadamard(wires=wire)
    for layer in range(layers):
        base = 2 * layer * qubits
        for edge in range(qubits):
            qml.IsingZZ(
                params[base + qubits + edge],
                wires=(edge, (edge + 1) % qubits),
            )
        for wire in range(qubits):
            qml.RX(params[base + wire], wires=wire)
```

### 4.2 注册

```python
register_circuit(
    CircuitSpec(
        name="qaoa-ns",
        aliases=("qaoa_ns", "qaoa-nonshared"),
        parameter_count_fn=lambda qubits, layers: 2 * qubits * layers,
        builder=_qaoa_ns,
        requires_even_qubits=True,
        minimum_qubits=4,
    )
)
```

所有别名必须通过现有 `_normalise_name()` 归一到 `qaoa-ns`，不能因别名改变参数布局。

### 4.3 Hamiltonian

`build_hamiltonian()` 将规范名 `qaoa` 和 `qaoa-ns` 放入同一分支，继续使用：

```text
H_C = 1/2 * sum_i (Z_i Z_{i+1} - I)
```

必须保留 ring wrap-around 边 `(n-1, 0)`。

### 4.4 Reference 测试

在 `pennylane-lightning/tests/test_circuits.py` 增加：

- `parameter_count(4,1)==8`、`parameter_count(6,2)==24`；
- 4q tape 为 4 Hadamard、4 IsingZZ、4 RX；
- 每个 RZZ/RX occurrence 的参数索引唯一；
- 参数顺序是 beta block 后 gamma block（与现有 QAOA 的 beta/gamma layout 对齐）；
- 第二层从 `2 * qubits` 开始，不复用第一层；
- topology 为 `(0,1),...,(n-1,0)`；
- Hamiltonian 与共享 `qaoa` 完全一致。

## 5. Native operation

修改：

```text
pennylane-lightning/src/pennylane_lightning_baseline/native.py
```

新增 `qaoa-ns` 分支。每个 operation 必须引用唯一参数：

```python
for layer in range(layers):
    base = 2 * layer * qubits
    for edge in range(qubits):
        _append_parameterized(
            operations, "IsingZZ",
            (edge, (edge + 1) % qubits), params, base + qubits + edge)
    for wire in range(qubits):
        _append_parameterized(
            operations, "RX", (wire,), params, base + wire)
```

不得复用 shared QAOA 的 `beta=2*layer`、`gamma=beta+1` 逻辑。native backward 输出 shape 必须为 `(2 * qubits * layers,)`，每个参数恰好对应一个 occurrence。

## 6. C API 与 dispatch

### 6.1 Circuit ID

在 `sad/include/sad_api.h` 追加独立 ID，例如：

```cpp
SAD_CIRCUIT_QAOA_NS = 8,
```

已有 `SAD_CIRCUIT_QAOA = 3` 不变。同步修改 `circuit_dispatch.cuh`、Python name-to-ID 映射、CLI 合法名称和枚举测试。

### 6.2 C++ include

在 `sad/src/runtime/circuit_execution.cuh` 与 QAOA 并列加入：

```cpp
#include "../circuits/qaoa_ns.cuh"
```

不复制或修改 shared QAOA executor。

## 7. 参数量、layout 与 executor

### 7.1 参数量特判

在 `expected_parameter_count()` 中增加：

```cpp
if constexpr (Circuit == SAD_CIRCUIT_QAOA_NS) {
    return static_cast<size_t>(2) * qubits * layers;
}
```

该分支必须在通用 `kParametersPerQubitLayer * qubits * layers` 之前。`qaoa_ns.cuh` 的 `kParametersPerQubitLayer` 可设为 0。

### 7.2 Layer layout

新增 `sad/src/circuits/qaoa_ns.cuh`。它的 include 边界应与现有 QAOA 对齐：使用 `qaoa.cuh`（复用 plus-state initializer）、`context.cuh`、`diagonal.cuh`、`rotation.cuh` 和 `lookups.cuh`；由于 plus-state initializer 已属于 `qaoa.cuh`，不得再复制 initializer：

```cpp
struct QaoaNsLayerLayout {
    int beta;
    int gamma;

    static auto at(int layer, int qubits) -> QaoaNsLayerLayout {
        const int base = 2 * layer * qubits;
        return {base, base + qubits};
    }
};
```

edge/wire 参数通过 `gamma + edge`、`beta + wire` 计算，不创建 per-layer topology array；layout 字段顺序和返回值必须与现有 `QaoaLayerLayout` 一致，即 `beta=base`、`gamma=base+qubits`。

### 7.3 Executor 接口

保持现有接口：

```cpp
template <typename T>
struct CircuitExecutor<SAD_CIRCUIT_QAOA_NS, T> {
    static constexpr int kParametersPerQubitLayer = 0;
    static void validate(int qubits);
    static void append_diagonal_lookups(...);
    static auto build_initial_state_lookup(...);
    static void apply_cost(const QaoaNsLayerLayout& layout,
                           const ForwardCircuitContext<T>& context);
    static void apply_mixer(const QaoaNsLayerLayout& layout,
                            const ForwardCircuitContext<T>& context,
                            int layer);
    static void forward_initial(...);
    static void forward_layer(...);
    static void forward_layer_optimized(...);
    static void backward_layer(...);
    static void backward_layer_optimized(...);
    static void backward_layer_fused(...);
};
```

方法集合必须与现有 `qaoa.cuh` 对齐：`apply_cost()` 只负责一层 nonshared ring-RZZ，`apply_mixer()` 只负责一层 nonshared RX，`forward_initial()` 负责 `|+>` 初始化并调用 layer 0 的两个 helper，`forward_layer()`/`forward_layer_optimized()` 负责后续层，反向保留 `backward_layer()`、`backward_layer_optimized()`、`backward_layer_fused()` 三个入口。反向为 inverse RX 再 inverse RZZ，并分别累计 beta/gamma。

`forward_initial()` 应复用现有 `qaoa.cuh` 已定义的 `initialise_plus_state_kernel`，不要在 `qaoa_ns.cuh` 再定义第二个等价的 plus-state initializer。不得为了新电路扩展 context。

## 8. Diagonal lookup 设计

### 8.1 不能复用 shared lookup

当前 QAOA 的 `append_shared_diagonal_lookup_group()` 把一组 RZZ 的 eigenvalue 求和后乘同一个 gamma。非共享版本必须计算：

```text
phase(index) = -1/2 * sum_edge gamma_edge * z_edge(index)
z_edge = +1 if bits[left]==bits[right], else -1
```

只替换函数名而继续使用 shared lookup 会让所有边错误地读取同一个参数。

### 8.2 Host metadata

优先复用 `DiagonalLookupData<T>::factors` 和 `offsets_by_parameter`，为每层每条 edge 记录：

```text
offsets_by_parameter[gamma_offset + edge]
```

如果一次 state pass 需要多个 edge lookup，应使用连续 offset 数组或通用 diagonal offset buffer；不要把 `qaoa_ns` 专用字段加入 forward/backward context。beta 参数不需要 diagonal lookup。

### 8.3 推荐实现路径

第一版沿用现有 shared QAOA 的调度边界：每层 `apply_cost()` 只调用一次 forward ring-RZZ launcher，每层 backward 只调用一次 inverse ring-RZZ launcher。kernel 内部遍历全部 `n` 条 ring edge，并为每条 edge 读取独立 gamma。even/odd matching 拆成两次 launch 只能作为后续性能实验，不能成为 correctness 版本的默认结构。

## 9. CUDA nonshared RZZ kernel

扩展现有 diagonal kernel 文件：

```text
sad/src/kernels/diagonal.cuh
```

当前 shared QAOA 的 `shared_ring_rzz_*` 已位于该文件。非共享版本的 ring-RZZ kernel 也必须放在同一文件，保持“对角门 kernel 按门类型归属”的现有边界；`sad/src/circuits/qaoa_ns.cuh` 只负责 layout、参数 offset 和 executor，不再拥有另一份同类 kernel 文件。

第一版只在 `diagonal.cuh` 中新增一对 forward/backward kernel 和 launcher：

```cpp
nonshared_ring_rzz_forward_kernel
nonshared_ring_rzz_backward_kernel
launch_nonshared_ring_rzz_forward
launch_nonshared_ring_rzz_backward
```

为支持 ring 中任意 edge，kernel 必须同时接收 edge 的全局编号（或 parity + local edge index）。不能直接对所有 edge 重复调用现有 `DiagonalGate::RZZ_EVEN` 的 gate index 0；否则 odd matching 和 wrap-around edge 会错误地作用到 `(0,1)`。如果每条 edge 使用 `append_diagonal_lookup_group(..., gate_count=1)`，lookup 只负责该 edge 的 angle，kernel 负责用全局 edge 计算 eigenvalue。

### 9.1 Forward

对每个 basis index：

```text
phase = -1/2 * sum_edge gamma_edge * z_edge(index)
state[index] *= exp(i * phase)
```

kernel 可以在一个 state pass 内按 even/odd 顺序遍历，但不能每 edge 单独 launch。kernel 必须使用 gamma block 的独立参数/lookup offset。

### 9.2 Backward

backward 逆向应用 RZZ，并分别写入：

```text
dE/dgamma_edge = sum_index
    imag(conj(lambda[index]) * phi[index]) * z_edge(index)
gradients[gamma_offset + edge]
```

forward 的 edge 遍历顺序必须固定；backward 使用其逆向顺序。不能把所有 eigenvalue 求和后只写入一个 `gamma_offset`。

### 9.3 lookup 与 phase 计算

保留 host 参数预计算 convention，不在 amplitude loop 中调用 `sin/cos`。ring edge 的 `z_edge` 可直接通过 bit comparison 计算。实现必须验证 lookup offset、edge index、parameter index 的一一对应。

### 9.4 Kernel 可调参数

`qaoa_ns` 不需要另造一套 circuit-specific 编译宏；它应复用现有 `sad/src/core/cuda_common.cuh` 的全局 kernel 参数。可调参数必须按实际 kernel 使用情况区分：

| 编译宏 | 当前默认值 | 影响的 qaoa_ns 路径 | 第一版状态 |
|---|---:|---|---|
| `SAD_FORWARD_BLOCK_THREADS` | `SAD_BLOCK_THREADS`，默认 128 | RX forward generic kernel | 可调 |
| `SAD_FORWARD_REGISTER_BITS` | `SAD_REGISTER_BITS`，默认 2 | RX forward 的 register amplitudes、tile size | 可调 |
| `SAD_BLOCK_THREADS` | 128 | RX backward generic kernel | 可调 |
| `SAD_REGISTER_BITS` | 2 | RX backward 的 register amplitudes、tile size | 可调 |
| `SAD_ORDINARY_BLOCK_THREADS` | 128 | nonshared ring-RZZ forward/backward 和 QAOA MaxCut Hamiltonian 的 ordinary kernel | 可调 |
| `SAD_DIAGONAL_LOOKUP_BITS` | 8 | RZZ lookup 每个 chunk 的编码宽度和 lookup 大小 | 可调，但需重新检查 lookup 内存与 launch 代价 |
| `SAD_FIXED_LOW_LANES` | 0 | RX backward/forward phase map（通过 generic rotation） | 只有采用 fixed-low-lane map 时有效 |
| `SAD_FORWARD_FIXED_LOW_LANES` | 0 | RX forward phase map | 只有采用 fixed-low-lane map 时有效 |
| `SAD_ALTERNATE_PHASES` | 0 | RX generic kernel 的 phase traversal | 可调，但必须逐 mode 验证 |
| `SAD_QAOA_FUSE_COST_RX` | -1 | 现有 shared QAOA 的 cost/RX fusion | qaoa_ns 第一版不使用，不能直接打开 |

现有 `cuda_common.cuh` 对 threads、register bits、tile bits 和 diagonal lookup bits 有 `static_assert`，合法范围不能凭经验扩大：

```text
block threads            = 64, 128, 256, 512
register bits             = 2 ... 6
forward/backward tile bits <= 12
ordinary block threads    = 64, 128, 256, 512
diagonal lookup bits      = 1 ... 12
```

这些是编译期参数，不是每次 API 调用传入的 QAOA 角度。`beta/gamma` 属于运行时物理参数，不能与 kernel launch tuning 混为一谈。

第一版 correctness kernel 的推荐基线为：

```text
RX forward   : SAD_FORWARD_BLOCK_THREADS=128, SAD_FORWARD_REGISTER_BITS=2
RX backward  : SAD_BLOCK_THREADS=128, SAD_REGISTER_BITS=2
RZZ ordinary : SAD_ORDINARY_BLOCK_THREADS=128
lookup       : SAD_DIAGONAL_LOOKUP_BITS=8
```

第一版的 nonshared ring-RZZ 如果采用普通 state pass 和 block reduction，不应声称 `SAD_REGISTER_BITS` 或 `SAD_FORWARD_REGISTER_BITS` 会直接改变 RZZ kernel 的 tile 形状；它们首先影响 RX。只有将 RZZ 改写为 tile/register kernel 后，才把两组 register 参数纳入 RZZ 的独立 sweep。

`SAD_REAL_AMPLITUDE` 不属于 qaoa_ns 的有效调优参数：当前 real-amplitude 快速路径只对 RA-HEA 生效。`CUDA_ARCH` 是编译目标架构，不是 qaoa_ns kernel shape 参数，应单独记录。

### 9.5 Variant 调优方法

每一个候选 variant 都必须重新编译 `.so`，不能在运行时修改上述宏。建议至少测试：

```text
forward RX : 64x3, 128x2, 128x3
backward RX: 64x3, 128x2, 64x4
ordinary RZZ/Hamiltonian: 64, 128, 256
lookup bits: 6, 8, 10
```

候选组合必须同时通过：

- PennyLane energy 与 full gradient；
- 中心有限差分；
- legacy/optimized execution mode；
- float32/float64；
- `compute-sanitizer`；
- device memory 和 occupancy 检查。

variant label 应在 benchmark CSV 中记录，例如：

```text
f128r2_b128r2_o128_d8
```

不要只记录 `qaoa_ns`，否则不同 kernel 参数下的结果不可复现。

## 10. RX mixer

复用 generic rotation API，但模板参数必须是 nonshared：

```cpp
launch_non_diagonal_forward<T, NonDiagonalGate::RX, false>(
    ..., layout.beta, ...);
```

backward 同样使用 `SharedParameter=false`。第 `wire` 个 RX 必须读取 `parameters[layout.beta + wire]`，不能传 layer base。

## 11. Forward/backward 集成

### 11.1 Forward

第一版固定为：

```text
初始化 |+>
for layer = 0..L-1:
    nonshared ring-RZZ(gamma block)
    nonshared RX(beta block)
```

不启用现有 shared QAOA 的 `SAD_QAOA_FUSE_COST_RX`，因为其 diagonal 输入假设单一 gamma。fused path 必须另行实现并测试。

### 11.2 Backward

每层 reverse order：

```text
inverse RX + accumulate beta[layer, wire]
inverse ring-RZZ + accumulate gamma[layer, edge]
```

梯度 buffer 按完整 `parameter_count` 分配，每次 step 前清零。任何漏写、重复写或错误 offset 都必须由 full-gradient 测试捕获。

## 12. Hamiltonian、workspace 与 runner

`qaoa_ns` 必须与 `qaoa` 复用 ring MaxCut Hamiltonian。runner 分支应将两个 ID 归并到 `qaoa_cost_hamiltonian_kernel`，不能落入默认 TFIM/HEA 分支。

现有 workspace 已按 `parameter_count` 分配 `rotation_coefficients` 和 `gradients`，第一版不修改 `ForwardCircuitContext`、`BackwardCircuitContext` 或 `run_forward/run_backward` 签名。若需要 edge lookup offsets，优先复用已有 diagonal offsets；只有无法表达时才增加通用 metadata buffer，并同步 copy、释放和 memory report。

`append_diagonal_lookups()` 必须为所有 gamma 参数产生有效 offset；beta offset 不使用是允许的。

## 13. Python SAD runner

修改 `sad/python/sad_baseline/runner.py`：

```python
"qaoa-ns": (8, 0),
```

`sad/python/sad_baseline/runner.py` 当前使用 `_normalise_name()` 将下划线转为连字符，因此 `_CIRCUITS` 应新增规范键 `qaoa-ns`。如需命令行别名，应像已有 `ra`、`su2`、`rzz`、`xxz` 一样逐项显式加入并测试；`qaoa_ns` 可作为名称归一规则下的输入别名，不额外强制更多别名。参数量特判：

```python
if circuit_id == SAD_CIRCUIT_QAOA_NS:
    parameter_count = 2 * qubits * layers
```

继续执行现有 qubit、layer、steps、warmup 和 parameter-array shape 校验。library selection 第一版使用默认已验证 variant，不直接套用 shared QAOA 的专用选择表。

## 14. 测试计划

### 14.1 参数与 tape

建议覆盖：

| qubits | layers | 参数量 |
|---:|---:|---:|
| 4 | 1 | 8 |
| 4 | 2 | 16 |
| 6 | 1 | 12 |
| 8 | 2 | 32 |

每项确认 RZZ/RX 数量、每个 occurrence 的参数索引唯一、每层 block 连续、layer 间没有参数复用。

### 14.2 Native/PennyLane 数值

对 4q/6q、1/2 layers 比较 energy 和 full gradient，覆盖 float32/float64。测试参数不能全部相同，否则错误的 shared 实现可能通过；应使用每条边和每个 wire 都不同的确定性向量。

### 14.3 有限差分

对 4q/1 layer float64 检查第一个、中间、最后一个 gamma，以及第一个和最后一个 beta。使用中心有限差分，确认每个独立参数的梯度，而不是只确认梯度总和。

### 14.4 CUDA modes

扩展 `sad/tests/test_sad_runner.py`，至少覆盖 legacy 和 optimized。若 all-fused 尚未实现，应明确测试 unsupported 或回退行为，不能静默调用 shared QAOA fused kernel。所有支持的 mode 都要与 PennyLane 对比 energy 和完整梯度。

### 14.5 共享版本回归

必须确认：

- `qaoa` 参数量仍为 `2 * layers`；
- shared QAOA 的 tape、energy、gradient 和 benchmark 不变；
- 原有 circuit ID 不变；
- 其他电路参数量、Hamiltonian、output shape 不变；
- workspace/context/run signatures 不变。

## 15. Benchmark

新增独立脚本或扩展现有脚本，但不覆盖 shared QAOA 结果：

```text
benchmark/benchmark_qaoa_ns_sad.py
benchmark/benchmark_qaoa_ns_pennylane.py
```

结果文件：

```text
benchmark/results/qaoa_ns_sad_gpu.csv
benchmark/results/qaoa_ns_pennylane_gpu.csv
```

第一轮覆盖 `qubits=4,6,8,10,12,14,16,18,20,22,24`，`layers=1,2,4`，float32/float64。每行记录 parameter_count、forward、backward、Hamiltonian、total 时间和 gradient norm。非共享版本参数量随 qubits 增长，不能把 shared QAOA 的每层 2 参数吞吐指标与其直接比较而不做说明。

## 16. 性能阶段

性能优化必须遵循“先测量、单变量修改、完整 correctness 回归、再保留”的顺序。所有优化都必须与 Phase A 的独立参数实现对照，不能因为速度提升而改变参数索引、梯度语义或 `qaoa` 共享角版本。

### 16.1 统一测量和验收规则

每个候选实现至少记录以下 split：

```text
forward_ms       = state initialization + all QAOA layers
hamiltonian_ms   = qaoa_cost_hamiltonian_kernel
backward_ms      = all inverse layers + gradient accumulation
total_ms         = forward_ms + hamiltonian_ms + backward_ms
```

同时记录：

- qubits、layers、precision、execution mode；
- `parameter_count` 和 kernel variant label；
- RZZ forward/backward 单独耗时；
- RX forward/backward 单独耗时；
- kernel launch 数量；
- device memory、动态 shared memory 和 occupancy；
- energy、gradient norm、最大梯度误差。

每个候选必须通过：

1. 4q/6q/8q 的 PennyLane energy 和 full gradient；
2. 4q/1 layer float64 的逐参数中心有限差分；
3. float32/float64；
4. legacy 与 optimized（以及已实现的 fused mode）；
5. `compute-sanitizer`；
6. 至少一次 20q 以上 smoke test。

性能结论使用预热后的 median，而不是单次最快值。每次实验固定 GPU、CUDA_ARCH、随机参数、steps、warmup 和 execution mode。任何候选若只在一个 qubit 或一种 precision 上变快，不得直接成为默认 variant。

### Phase A：正确性基线和 profile

目标是建立可逐 kernel 对照的 reference，不追求最终吞吐。

实现固定为：

```text
每层：一次 nonshared RZZ forward
      一次 nonshared RX forward
backward：一次 nonshared RX backward
          一次 nonshared RZZ backward
```

具体工作：

- 保留 `diagonal.cuh` 中 ordinary nonshared RZZ kernel；
- RX 继续使用 `SharedParameter=false` 的 generic tile kernel；
- 不启用 cost/RX fusion；
- 在 benchmark 中分别测 RZZ、RX、Hamiltonian 三类时间；
- 用 Nsight Systems 确认 launch 顺序和 kernel 数量；
- 用 Nsight Compute 记录 memory throughput、achieved occupancy、register count 和 branch efficiency。

基线输出必须包含：

```text
qaoa_ns_baseline_f128r2_b128r2_o128_d8
```

该阶段的意义是确认后续优化到底改善了 RZZ、RX、梯度规约还是仅减少了 launch overhead。Phase A kernel 保留为 debug oracle，不随优化删除。

### Phase B：压缩 lookup 和减少 RZZ 内存流量

当前每个独立 RZZ 参数只需要 eigenvalue `+1/-1` 两个 phase factor，但通用 diagonal lookup 可能按 `SAD_DIAGONAL_LOOKUP_BITS` 产生完整 chunk。第一步优先压缩该存储。

推荐实现：

```text
per edge lookup = [phase_for_eigenvalue_+1,
                   phase_for_eigenvalue_-1]
```

需要完成：

1. 新增 qaoa_ns 专用 compact lookup builder，不能改变 shared QAOA 的 lookup layout；
2. forward kernel 按 `edge * 2 + eigenvalue_code` 读取；
3. backward kernel 复用同一 compact lookup；
4. 重新计算 workspace、H2D copy 和 memory report；
5. 比较 compact lookup、完整 lookup 和直接读取 `sin/cos` coefficients。

验收指标：

- energy/full gradient 与 Phase A 一致；
- diagonal lookup device bytes 随 `2 * qubits * layers` 增长，而不是 `SAD_DIAGONAL_LOOKUP_SIZE * qubits * layers`；
- forward/backward global load 数量下降；
- 低 qubit 不因额外间接寻址而回退超过预设阈值。

### Phase C：RZZ forward tile/register 化

Phase B 之后，RZZ 的主要瓶颈预计是每个 amplitude 逐 edge 循环。应参考 `rotation.cuh` 和 `xxz.cuh`，为 RZZ 建立 tile-local 实现。

forward kernel 的目标形状：

```text
一个 block 处理一个或多个 state tile
每个 thread 持有 2^kForwardRegisterBits 个 amplitudes
tile 内复用 basis bit / edge eigenvalue
一次加载、一次写回
```

实现步骤：

1. 用 `kForwardTileBits` 和 `kForwardRegisterAmplitudes` 组织 tile；
2. 对固定 tile 先构造相邻 bit 的 edge eigenvalue pattern；
3. 在 register 中累乘所有独立 gamma 的 phase；
4. 避免每个 edge 重复计算 `(edge + 1) % qubits`；
5. 保留普通 kernel 作为小 qubit 或资源不足时的 fallback；
6. 对 64/128 threads 和 register bits 2/3/4 做单变量 sweep。

重点监控：

- register spill；
- dynamic shared memory；
- occupancy；
- load/store throughput；
- qubits 增长时每个 tile 的有效工作比例。

禁止在第一版 tile kernel 中引入不安全的 global in-place permutation。RZZ 是对角门，state index 不发生置换，应保持单一 state load/compute/store 语义。

### Phase D：RZZ backward 梯度规约

当前 backward 需要为每条 edge 保存 overlap，并在多个 block 中对同一 gamma 参数执行 `atomicAdd`。优化目标是减少 shared memory 和 global atomic contention。

优先方案是两级规约：

```text
kernel 1: 每个 block 计算 qubits 个 gamma partials
kernel 2: 对 block partials 做 edge-wise reduction
```

具体步骤：

1. 为 partial gradient 分配 `grid_size * qubits` 的临时 buffer；
2. kernel 1 只写自己的 block row，避免 global atomic；
3. kernel 2 按 edge 合并 block rows，最终写入 `gradients[gamma_offset + edge]`；
4. 比较两级规约与 atomic 版本在 4q、16q、24q、28q 的表现；
5. 对小 state 保留 atomic 快路径，避免第二个 kernel launch 反而变慢。

如果两级规约的额外 launch 成本在小规模超过收益，应按 state size 设置阈值。partial buffer 必须在 workspace 中正确计入 memory report，并在每次 step 前清零或完整覆盖。

验收重点：

- 所有 gamma 梯度逐元素一致；
- 多 block 下无 race；
- 24q/28q 的 backward occupancy 和 atomic throughput 改善；
- `compute-sanitizer --tool racecheck` 通过。

### Phase E：RZZ + RX forward fusion

当 RZZ tile 和 lookup 稳定后，融合 cost/mixer，减少每层一次 state load/store。

目标流程：

```text
load tile amplitudes
apply nonshared ring-RZZ phase in registers
apply nonshared RX using existing rotation primitive
store tile amplitudes once
```

实现要求：

- 新增 qaoa_ns 专用 fused forward kernel/launcher，不修改 shared QAOA 的 `SAD_QAOA_FUSE_COST_RX` 语义；
- gamma lookup 和 beta rotation coefficients 使用不同 offset；
- 保持 cost 后 mixer 的门序；
- 复用 rotation primitive 的 mailbox，不复制另一套 RX 数学实现；
- 对 fused 与 unfused state 在小规模逐 amplitude 对比。

融合可能增加 register pressure。只有当 global memory traffic 的收益超过 occupancy 损失时才保留。需要分别评估 4q-12q、16q-24q、26q 以上，不应只根据一个规模选择 fused 默认路径。

### Phase F：初始化和 backward fusion

forward 第一层当前是独立的 `|+>` 初始化、RZZ、RX。可以增加：

- plus-state + 第一层 RZZ 的 fused initializer；
- plus-state + RZZ + RX 的完整第一层 initializer；
- 对 `layers=1` 的专用低-launch路径。

这一步优先针对小层数 benchmark，不应影响后续 layer 的通用 kernel。

backward fusion 放在 forward fusion 之后。其难点是同时保存 phi/lambda、RX generator overlap 和每条 gamma 的 overlap。若 fusion 导致 register spill 或 shared memory 超过 occupancy 目标，应保留分离 backward 作为默认路径。

### Phase G：kernel variant 和 dispatch

`sad/python/sad_baseline/runner.py::_select_library()` 当前只为部分已有 circuit ID 选择专用 variant。qaoa_ns 应建立独立候选表，不直接复用 shared QAOA 的经验：

| 方向 | 第一轮候选 |
|---|---|
| RX forward | `64x3`、`128x2`、`128x3` |
| RX backward | `64x3`、`128x2`、`64x4` |
| ordinary RZZ | 64、128、256 threads |
| compact lookup | 2 factors/edge；必要时对照完整 lookup |
| fused | off/on |

dispatch key 至少包含：

```text
circuit_id, qubits, precision, execution_mode, fused_state
```

每个 variant 必须：

- 单独编译并记录宏组合；
- 写入 benchmark CSV 的 `kernel_variant`；
- 通过 energy/full gradient/有限差分；
- 通过 sanitizer；
- 在相邻 qubit 规模上稳定，而不是只对一个规模取最优。

若没有足够 benchmark 数据，qaoa_ns 继续使用默认 `f128r2_b128r2`，不要提前添加 qubit-dependent dispatch。

### Phase H：完整 benchmark 和回退策略

最终 benchmark 至少覆盖：

```text
qubits  = 4, 6, 8, 10, 12, 16, 20, 24, 28
layers  = 1, 2, 4
precision = float32, float64
mode    = legacy, optimized, all-fused（已实现时）
```

每个配置运行固定 warmup 和多个 measured steps，报告 median/p10/p90。对每个优化记录：

- 相对 Phase A 的 forward/backward/total speedup；
- 相对 shared QAOA 的参数量归一化成本；
- memory 增量；
- 数值误差；
- 失败或回退原因。

最终默认路径必须有明确回退：

```text
fused 不稳定       -> unfused tile kernel
tile 资源不足      -> ordinary RZZ kernel
two-level reduction 小规模变慢 -> atomic reduction
variant 未命中     -> 默认 f128r2_b128r2
```

任何回退都必须保留相同的参数布局和梯度结果，不能通过更换路径改变电路语义。

## 17. 实施顺序与验收

### Phase 1：reference 和注册

1. 新增 `_qaoa_ns()`、`CircuitSpec`、aliases。
2. 复用 QAOA Hamiltonian。
3. 完成 tape/parameter-count 测试。
4. 完成 native operation 参数索引测试。

验收：Python/native 生成 `2*n*L` 个独立参数，shared QAOA 全部回归通过。

### Phase 2：SAD 接口与 RX

1. 添加 circuit ID 和 dispatch case。
2. 添加 `expected_parameter_count()` 特判。
3. 添加 `qaoa_ns.cuh` layout/executor。
4. 接入 generic nonshared RX。
5. 接入 Python SAD runner。

验收：RX forward/backward 单独对齐参考实现。

### Phase 3：RZZ correctness

1. 添加 ring-specific lookup/offset 构建。
2. 添加 nonshared RZZ forward kernel。
3. 添加 backward kernel 和逐 edge gamma 梯度。
4. 接入 QAOA Hamiltonian 分支。

验收：4q/6q/8q、1/2 layers 的 energy/full gradient 对齐，有限差分通过。

### Phase 4：完整回归

验证所有支持的 execution modes、float32/float64、完整 Python/native/SAD 测试、compute-sanitizer 和较大 qubit smoke test。要求无非法访问、race、梯度漏写和参数越界。

### Phase 5：性能优化

按第 16 节逐步推进。每次优化保留 correctness 对照，不删除独立参数的参考 kernel。

## 18. 完成标准

- `qaoa` 仍是共享角版本，行为完全不变；
- `qaoa-ns` 是独立 circuit ID 对应的规范名称，`qaoa_ns` 是可归一化输入别名；
- 参数量为 `2 * qubits * layers`；
- 每个 RZZ、RX occurrence 使用独立参数；
- gamma/beta 的 layer-major 布局在 Python/native/CUDA 一致；
- RZZ forward 不读取单一 shared gamma；
- RZZ backward 为每条 edge 写入独立 gamma 梯度；
- RX 使用 `SharedParameter=false`；
- Hamiltonian 仍为环形 MaxCut cost；
- 4q/6q/8q energy、full gradient、有限差分通过；
- 支持的 execution modes 和精度通过；
- 现有电路测试、benchmark 和 ID 不回归；
- 新 benchmark 不覆盖 shared QAOA 结果；
- 文档明确区分 `qaoa` 与 `qaoa_ns`。

## 19. Production 文件与新增符号审计

| 文件 | 允许新增内容 | 约束 |
|---|---|---|
| `pennylane-lightning/.../circuits.py` | 一个 `_qaoa_ns()`、一个注册调用、一个 Hamiltonian 名称分支 | 不新增 topology/parameter helper |
| `pennylane-lightning/.../native.py` | 一个 `qaoa-ns` operation 分支 | 每个 occurrence 独立 source parameter |
| `sad/include/sad_api.h` | 一个 `SAD_CIRCUIT_QAOA_NS` ID | 不修改已有 ID |
| `sad/src/runtime/circuit_dispatch.cuh` | 一个 dispatch case | 复用 compile-time visitor |
| `sad/src/runtime/circuit_execution.cuh` | 一个 include、一个 parameter-count 分支 | 不修改其他公式 |
| `sad/src/circuits/qaoa_ns.cuh` | 一个 layout、一个 executor 特化 | 不复制 shared executor |
| `sad/src/kernels/diagonal.cuh` | 一对 nonshared ring-RZZ kernel/launcher | 与现有 `shared_ring_rzz_*` 同文件、同门类型边界 |
| `sad/src/runtime/lookups.cuh` | ring nonshared lookup 或必要 offset 支持 | 不改变 shared lookup 语义 |
| `sad/src/runtime/runner.cuh` | QAOA/QAOA_NS Hamiltonian 分支扩展 | 不落入默认 Hamiltonian |
| `sad/python/sad_baseline/runner.py` | 注册、ID 参数量/canonical 分支 | 参数量必须为 `2*n*L` |
| tests | tape、参数、数值、梯度、mode 测试 | 与 shared QAOA 分开断言 |
| benchmark | 独立 qaoa_ns 脚本和结果名 | 不覆盖 shared 结果 |

特别禁止：

- 在 `qaoa.cuh` 中依靠 parameter count 隐式切换模式；
- 让 `qaoa_ns` 调用 `launch_shared_ring_rzz_*`；
- 把所有 gamma 梯度归并到一个 offset；
- 用全相同参数验证“非共享”；
- 为每条边建立独立 kernel launch 作为最终实现；
- 没有数值和 sanitizer 证据就启用 fused/variant 优化；
- 修改 shared QAOA 的参数布局、结果文件或 circuit ID。
