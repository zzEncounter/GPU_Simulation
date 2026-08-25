# Equivariant QNN 电路实现计划（严格对齐现有代码结构）

## 1. 目标与结构契约

本文档定义如何把 `(S_n)`-Permutation-Equivariant QNN（下文简称 `EQNN`）加入
当前项目。结构对齐的直接基准是原始五种电路 RA-HEA、SU2-HEA、RZZ-HEA、QAOA、
XXZ-HVA；当前 MERA 作为第六个已实现扩展用于交叉检查。实现必须复用这些电路已经
形成的分层，不重构公共 runner、context、workspace 或 `CircuitSpec`。

强制结构如下：

```text
PennyLane:  circuits.py 中一个 _equivariant_qnn()
Native:     native.py::_build_operations() 中一个 EQNN 分支
SAD C++:    一个 EquivariantQNNLayerLayout
            一个 CircuitExecutor<SAD_CIRCUIT_EQUIVARIANT_QNN,T>
CUDA:       一对 forward/backward kernel
            一对 forward/backward launcher
Hamiltonian: 一个 EQNN Hamiltonian kernel
Python SAD: 现有 _CIRCUITS 和参数量分支中增加一个 ID
```

明确禁止：

```text
第二个 EQNN PennyLane builder
独立的 EQNN runner
EQNN 专用 context/workspace 类型
EQNN host topology registry
legacy/optimized 两套平行 EQNN kernel API
修改现有六个 CircuitExecutor 的方法签名
增加 SAD_EQNN_* 公共调优宏
```

### 1.1 “一个函数”的审计口径

本计划中的“一个”按现有代码的 production 入口层级计算：

```text
circuits.py: 每种电路一个顶层 builder def
native.py:   每种电路在现有 _build_operations() 中一个分支，不新增 def
C++ circuit: 每种电路一个 LayerLayout 和一个 CircuitExecutor 特化
CUDA:        电路专用文件中一对 __global__ kernel 和一对 host launcher
Hamiltonian: 在现有 hamiltonian.cuh 中一个 __global__ kernel
```

同文件 `private __device__` helper 不属于第二套 production API。它只允许拆分无法由
现有公共 primitive 表达的局部计算，例如 round-robin pair 公式、tile-local CX gather
和 tile-local RZ；不得被 host、executor 或其他电路直接调用。该边界对齐
`xxz.cuh`/`mera.cuh` 的“private helper + 一对 kernel/launcher”结构。

因此最终 production 数量固定为：

```text
PennyLane EQNN builder def:                         1
native.py 新增顶层 def:                             0
EquivariantQNNLayerLayout:                          1
CircuitExecutor<SAD_CIRCUIT_EQUIVARIANT_QNN,T>:     1
EQNN matching __global__ kernel:                    2
EQNN matching host launcher:                        2
EQNN Hamiltonian __global__ kernel:                 1
EQNN 专用 runner/context/workspace/lookups 类型:    0
```

## 2. 固定电路定义

### 2.1 参数约定

论文生成元参数可写为 `alpha_l/beta_l/gamma_l`，对应实际门角：

```text
a_l = 2 * alpha_l / n
b_l = 2 * beta_l / n
g_l = 4 * gamma_l / (n * (n - 1))
```

为了严格复用当前 `PreparedWorkspace` 对参数统一生成
`RotationCoefficients(sin(theta/2), cos(theta/2))` 的结构，本项目参数向量直接存放实际
门角：

```text
params[3*l + 0] = a_l
params[3*l + 1] = b_l
params[3*l + 2] = g_l
```

因此不修改 `workspace.cuh`，也不在 kernel 内从三角系数反推原始角度。如果需要使用
论文的原始 `alpha/beta/gamma`，调用方在进入统一 API 前完成上述缩放。

每层三个逻辑参数，总参数量固定为：

```text
parameter_count = 3 * layers
```

### 2.2 RZZ 的唯一表示

本项目不把 RZZ 作为 EQNN 的生产门。所有 RZZ 都固定展开为：

```text
CX(control=i, target=j)
RZ(target=j, angle=g_l)
CX(control=i, target=j)
```

它等价于标准定义：

```text
RZZ(g_l) = exp(-i * g_l * Z_i Z_j / 2)
```

PennyLane、Lightning native 和 SAD 必须使用同一个角度约定，不能一侧使用 `g_l`、
另一侧使用 `2*g_l`。

### 2.3 单个 macro-layer 的门序

第 `l` 层的严格逻辑顺序为：

```text
1. 对 wire 0..n-1 依次执行 RX(a_l)
2. 对 wire 0..n-1 依次执行 RY(b_l)
3. 按 2.4 节 round-robin canonical phase 顺序遍历全部无序 pair (i,j), i<j，
   执行 CX(i,j), RZ(j,g_l), CX(i,j)
```

所有 RX occurrence 共享 `a_l`，所有 RY occurrence 共享 `b_l`，所有 pair block
共享 `g_l`。

每层逻辑门 occurrence 数：

```text
RX = n
RY = n
CX = n * (n - 1)
RZ = n * (n - 1) / 2
```

### 2.4 all-to-all pair 的确定顺序

为兼顾明确语义和 tile kernel，pair 使用确定性的 round-robin matching 顺序，而不是
由 host 预生成 map。

```text
participant_count = n                 if n is even
                  = n + 1             if n is odd  # 最后一个是 dummy
phase_count       = participant_count - 1
```

构造规则：

```text
participants = [0, 1, ..., participant_count - 1]
for phase in 0..phase_count-1:
    pairs = [(participants[k], participants[-1-k])
             for k in 0..participant_count/2-1]
    跳过包含 dummy 的 pair
    每个真实 pair 内令 control=min(a,b), target=max(a,b)
    固定 participants[0]，其余元素循环右移一位
```

同一 phase 内 pair 不共享 wire，所有 phase 合起来恰好覆盖一次全部 `i<j`。
PennyLane、native 和 SAD 都使用这个 phase 顺序，避免 reference 和 CUDA 采用不同的
floating-point accumulation/order。

4 qubits 示例：

```text
phase 0: (0,3), (1,2)
phase 1: (0,2), (1,3)
phase 2: (0,1), (2,3)
```

每个 pair 都展开为 `CX-RZ-CX`。

### 2.5 Observable

固定使用 permutation-invariant observable：

```text
H = (1/n) * sum_i X_i
```

这与电路的 `(S_n)` symmetry 对齐，并与现有 `energy_and_grad()` 的单一 Hamiltonian
分派模式兼容。第一版不同时支持多个 EQNN observable。

### 2.6 输入范围

```text
qubits >= 2
layers >= 1
不要求 qubits 为偶数
不要求 layers = ceil(log2(qubits))
```

奇数 qubits 只在 round-robin schedule 中引入 dummy，dummy 不对应实际 gate。

## 3. PennyLane 实现

修改：

```text
pennylane-lightning/src/pennylane_lightning_baseline/circuits.py
```

### 3.1 只增加一个 builder

只增加：

```python
def _equivariant_qnn(params: object, qubits: int, layers: int) -> None:
    participant_count = qubits if qubits % 2 == 0 else qubits + 1
    dummy = qubits if participant_count != qubits else None

    for layer in range(layers):
        a = params[3 * layer]
        b = params[3 * layer + 1]
        g = params[3 * layer + 2]

        for wire in range(qubits):
            qml.RX(a, wires=wire)
        for wire in range(qubits):
            qml.RY(b, wires=wire)

        participants = list(range(participant_count))
        for _ in range(participant_count - 1):
            for index in range(participant_count // 2):
                first = participants[index]
                second = participants[-1 - index]
                if first == dummy or second == dummy:
                    continue
                control, target = sorted((first, second))
                qml.CNOT(wires=(control, target))
                qml.RZ(g, wires=target)
                qml.CNOT(wires=(control, target))
            participants[1:] = participants[-1:] + participants[1:-1]
```

拓扑代码必须留在 `_equivariant_qnn()` 内部，不能再增加：

```text
build_equivariant_pairs()
equivariant_phase_count()
equivariant_parameter_count()
```

`circuits.py` 的 EQNN production diff 中只允许出现这一个新 `def`。参数量继续写在
`CircuitSpec.parameter_count_fn`，Hamiltonian 继续写在已有 `build_hamiltonian()`
分支中，不能把它们提取成第二、第三个 EQNN 顶层函数。

### 3.2 注册

继续复用一个 `CircuitSpec`：

```python
register_circuit(
    CircuitSpec(
        name="equivariant-qnn",
        aliases=("eqnn",),
        parameter_count_fn=lambda qubits, layers: 3 * layers,
        builder=_equivariant_qnn,
        minimum_qubits=2,
    )
)
```

不修改 `CircuitSpec`、`register_circuit()`、`get_circuit()` 或公共 runner。

### 3.3 Hamiltonian 分支

在现有 `build_hamiltonian()` 中增加一个分支：

```python
if circuit_spec.name == "equivariant-qnn":
    return qml.Hamiltonian(
        [1.0 / qubits] * qubits,
        [qml.PauliX(wire) for wire in range(qubits)],
    )
```

不新增 EQNN 专用 Hamiltonian builder。

## 4. Lightning native baseline

修改：

```text
pennylane-lightning/src/pennylane_lightning_baseline/native.py
```

### 4.1 一个 operation 分支

只在 `_build_operations()` 增加一个 `equivariant-qnn` 分支。它必须生成与
`_equivariant_qnn()` 完全相同的 phase/pair 顺序，并显式生成：

```text
RX occurrences -> source_parameter = 3*l
RY occurrences -> source_parameter = 3*l+1
RZ occurrences -> source_parameter = 3*l+2
CX occurrences -> source_parameter = None
```

多个 occurrence 指向同一个 `source_parameter`，继续复用 native runner 已有的共享
参数梯度归并逻辑，不增加新的 gradient map 类型。

不得增加 `_build_equivariant_operations()`、`_equivariant_pairs()` 或其他 EQNN
顶层 `def`。这与当前五种电路全部内联在同一个 `_build_operations()` 分支中的结构
一致。

### 4.2 一个 Hamiltonian 分支

在 `_build_hamiltonian()` 中使用现有 `named_observable` 和 `hamiltonian` bindings
构造：

```text
coefficients = [1/n, ..., 1/n]
terms        = [X(0), ..., X(n-1)]
```

不增加第二个 native runner 或 EQNN 专用 `OpsData` 路径。

## 5. SAD 注册和参数量

### 5.1 C API enum

修改：

```text
sad/include/sad_api.h
```

在现有 ID 后追加：

```cpp
SAD_CIRCUIT_EQUIVARIANT_QNN = 6,
```

不能重排现有 `0..5`，避免破坏已有 Python/C ABI。

### 5.2 compile-time dispatch

修改：

```text
sad/src/runtime/circuit_dispatch.cuh
```

只增加一个 switch case：

```cpp
case SAD_CIRCUIT_EQUIVARIANT_QNN:
    return function(std::integral_constant<
                    int, SAD_CIRCUIT_EQUIVARIANT_QNN>{});
```

### 5.3 include 和参数量

修改：

```text
sad/src/runtime/circuit_execution.cuh
```

只增加：

```text
#include "../circuits/equivariant_qnn.cuh"
```

并在现有 `expected_parameter_count()` 中增加与 QAOA/MERA 同层级的特判：

```cpp
if constexpr (Circuit == SAD_CIRCUIT_EQUIVARIANT_QNN) {
    return static_cast<size_t>(3) * layers;
}
```

不修改 `run_forward()`、`run_backward()` 或 context 参数列表。

### 5.4 Python SAD registry

修改：

```text
sad/python/sad_baseline/runner.py
```

在 `_CIRCUITS` 中增加：

```python
"equivariant-qnn": (6, 0),
"eqnn": (6, 0),
```

参数量在现有分支中增加：

```python
elif circuit_id == 6:
    parameter_count = 3 * layers
```

canonical name tuple末尾增加 `"equivariant-qnn"`。第一版不增加 ID 6 的 variant
dispatch；使用默认 `f128r2_b128r2`，完成独立 sweep 后再决定。

## 6. SAD circuit 文件

新增：

```text
sad/src/circuits/equivariant_qnn.cuh
```

只定义一个 layout 和一个 executor：

```cpp
struct EquivariantQNNLayerLayout {
    int parameter_offset;
    int phase_count;

    static auto at(int layer, int qubits)
        -> EquivariantQNNLayerLayout {
        const int participant_count = qubits + (qubits & 1);
        return {3 * layer, participant_count - 1};
    }
};

template <typename T>
struct CircuitExecutor<SAD_CIRCUIT_EQUIVARIANT_QNN, T> {
    static constexpr int kParametersPerQubitLayer = 0;
    // 保持现有 executor 方法集合和签名。
};
```

executor 必须与 MERA/QAOA/XXZ 一样实现现有方法集合：

```text
validate
append_diagonal_lookups
build_initial_state_lookup
forward_initial
forward_layer
forward_layer_optimized
backward_layer
backward_layer_optimized
backward_layer_fused
```

除 `EquivariantQNNLayerLayout::at()` 和上述 executor 既有接口方法外，不增加
`build_phases()`、`apply_rx_stage()`、`apply_pair_stage()` 等 host 函数。每个 executor
方法直接调用同一对 launcher；必要的局部拆分只能放在 kernel 文件的 private
`__device__` helper 中。

当前 runtime 的六种 execution mode：

```text
optimized
legacy
all-fused
initial-only
fused-forward
phased-forward
```

必须继续经过现有 `circuit_execution.cuh` 分派。对 EQNN，`forward_layer` 与
`forward_layer_optimized` 调用同一个 forward launcher，`backward_layer`、
`backward_layer_optimized` 与 `backward_layer_fused` 调用同一个 backward launcher；
不得因为 mode 名称不同而复制第二套 EQNN kernel。`phased-forward` 的 SU2 专用分支
不会要求 EQNN 增加 `forward_layer_phased()`。

初始化复用现有 zero-state 初始化方式，不新增 EQNN initialization kernel。

## 7. CUDA kernel 文件

新增：

```text
sad/src/kernels/equivariant_qnn.cuh
```

允许的顶层 production 形状固定为：

```text
private __device__ helpers
equivariant_qnn_forward_kernel
launch_equivariant_qnn_forward
equivariant_qnn_backward_kernel
launch_equivariant_qnn_backward
```

不得同时增加 `legacy`、`optimized`、`shared_gamma` 等平行 kernel。

### 7.1 kernel 内 topology

round-robin participant rotation、dummy 判断、pair 的 `min/max` 必须由
`qubits/phase/pair_index` 在 kernel 内计算。

不增加：

```text
EquivariantTopology
build_equivariant_pair_maps()
workspace.equivariant_selected_maps
context.equivariant_phase_count
```

如果直接用公式求 round-robin pair 较复杂，可以增加同文件 private `__device__`
helper，但不能暴露 host builder 或 public API。

### 7.2 forward

forward kernel 每层执行：

```text
1. shared-a RX on all wires（按 tile wire phases 覆盖）
2. shared-b RY on all wires（按 tile wire phases 覆盖）
3. for phase in forward order:
       kernel 内构造本 phase selected map
       load state tile
       for pair in phase order:
           tile-local CX gather
           tile-local RZ(g) on target
           tile-local CX gather
       store state tile
       grid.sync() before changing selected map
```

当 `qubits > kForwardTileBits` 时，RX/RY 不能假设所有 wire 同时位于一个 tile。
同一个 `equivariant_qnn_forward_kernel` 内按连续 physical wire 分成：

```text
single_wire_phase_count = ceil(qubits / kForwardTileBits)
```

每个 phase 在 kernel 内填充 selected slots、完成一次 tile load/apply/store，并在改变
selected map 前 `grid.sync()`。这只是同一个 kernel 内部 traversal，不增加 RX kernel、
RY kernel 或 launcher。RX/RY 在不同 wire 上对易，但 PennyLane/native 仍按 wire 递增
生成 occurrence；测试必须验证两者的最终 state/gradient 一致。

复用：

```text
apply_tile_gate_forward<T,NonDiagonalGate::RX,Slot>()
apply_tile_gate_forward<T,NonDiagonalGate::RY,Slot>()
scatter_tile_assignment()
scatter_local_assignment()
shared mailbox gather
```

RZ 是对角操作，可在 local amplitude 上按 target bit 乘对应相位。禁止把每个
`CX-RZ-CX` 展开为三次 full-state global pass。

### 7.3 backward

backward 必须严格逆序：

```text
for phase in reverse order:
    load phi/lambda tile
    for pair in reverse phase order:
        CX(phi)
        CX(lambda)
        accumulate contribution to grad[g_l]
        inverse RZ on phi/lambda
        CX(phi)
        CX(lambda)
    store phi/lambda tile

inverse shared-b RY on all wires + accumulate grad[b_l]
inverse shared-a RX on all wires + accumulate grad[a_l]
```

RY/RX 的 single-wire phases 必须分别按 forward traversal 的逆序处理；不能在 backward
中仍按正序切换 selected map。它们继续位于同一个
`equivariant_qnn_backward_kernel`，不增加单独 rotation backward kernel。

所有 pair 的 RZ contribution 写入同一个：

```text
gradients[3 * layer + 2]
```

所有 RY occurrence 写入 `gradients[3*layer+1]`，所有 RX occurrence 写入
`gradients[3*layer]`。第一版继续使用现有 block reduction + `atomicAdd`；只有 profiling
证明 shared-parameter atomic 是瓶颈时，才在独立优化阶段实现多级规约。

### 7.4 shared memory

每个 CTA 只使用一份 amplitude mailbox，并在以下操作间分时复用：

```text
CX phi gather
CX lambda gather
warp-slot RX/RY exchange
```

同时保留：

```text
double reduction[kBlockThreads]
selected[kTileBits]
tile_base
```

launcher 必须用实际 dynamic shared bytes 调用：

```text
cudaFuncSetAttribute()
cudaOccupancyMaxActiveBlocksPerMultiprocessor()
```

grid work unit 是 tile，不是单个 gate。

### 7.5 公共可调参数

不增加 `SAD_EQNN_*`。EQNN 与现有电路复用以下公共编译宏：

| 公共宏 | 源码默认值 | 源码合法范围 | EQNN 第一轮候选 | 对应 EQNN kernel |
|---|---:|---:|---:|---|
| `SAD_FORWARD_BLOCK_THREADS` | 继承 `SAD_BLOCK_THREADS`，最终为 128 | 64/128/256/512 | 64/128 | `equivariant_qnn_forward_kernel` |
| `SAD_FORWARD_REGISTER_BITS` | 继承 `SAD_REGISTER_BITS`，最终为 2 | 2..6，且 forward tile bits <= 12 | 2/3/4 | `equivariant_qnn_forward_kernel` |
| `SAD_BLOCK_THREADS` | 128 | 64/128/256/512 | 64/128 | `equivariant_qnn_backward_kernel` |
| `SAD_REGISTER_BITS` | 2 | 2..6，且 backward tile bits <= 12 | 2/3/4 | `equivariant_qnn_backward_kernel` |
| `SAD_ORDINARY_BLOCK_THREADS` | 128 | 64/128/256/512 | 第一版固定 128；仅在 Hamiltonian profile 后测 64/256/512 | `equivariant_qnn_hamiltonian_kernel` |

派生关系必须直接复用 `cuda_common.cuh`：

```text
kForwardRegisterAmplitudes = 1 << kForwardRegisterBits
kForwardTileBits = 5 + kForwardRegisterBits
                     + log2(kForwardBlockThreads / 32)
kForwardTileAmplitudes = 1 << kForwardTileBits

kRegisterAmplitudes = 1 << kRegisterBits
kTileBits = 5 + kRegisterBits + log2(kBlockThreads / 32)
kTileAmplitudes = 1 << kTileBits
```

默认有效形状：

```text
forward  = 128 threads, register bits 2, tile bits 9
backward = 128 threads, register bits 2, tile bits 9
ordinary = 128 threads
variant  = f128r2_b128r2
```

第一轮只编译当前 runner 已登记的安全 variants：

| Variant | Forward | Backward |
|---|---|---|
| `f128r2_b128r2` | 128 x r2 | 128 x r2 |
| `f64r4_b64r4` | 64 x r4 | 64 x r4 |
| `f64r3_b64r4` | 64 x r3 | 64 x r4 |
| `f128r3_b64r4` | 128 x r3 | 64 x r4 |
| `f64r4_b128r3` | 64 x r4 | 128 x r3 |

这些是编译期参数。切换 variant 表示加载不同预编译 `.so`，不是运行过程中改变 CTA
或 register bits。电路角度 `a_l/b_l/g_l` 是运行时参数，不属于 kernel 调优参数。

下列公共宏在 EQNN 第一版中明确不生效：

```text
SAD_FORWARD_FIXED_LOW_LANES
SAD_FIXED_LOW_LANES
SAD_ALTERNATE_PHASES
SAD_DIAGONAL_LOOKUP_BITS
```

原因是 EQNN selected map 在 kernel 内由 round-robin/continuous-wire 公式构造，不使用
host selected maps、alternate traversal 或 diagonal lookup。它们不得静默改变 EQNN
门序或 topology。

## 8. Hamiltonian kernel

修改：

```text
sad/src/kernels/hamiltonian.cuh
sad/src/runtime/runner.cuh
```

只增加一个：

```cpp
equivariant_qnn_hamiltonian_kernel
```

数学定义：

```text
lambda[index] = (1/n) * sum_i phi[index xor (1 << i)]
energy        = real(<phi|lambda>)
```

kernel 同时构造 `lambda = H|phi>` 并规约 energy，模式与现有 MERA/QAOA/XXZ
Hamiltonian 分支一致。`runner.cuh` 只增加一个 circuit-ID 分支，不修改公共函数签名。

## 9. 不修改的公共文件

以下文件不得因 EQNN 增加字段、buffer 或参数：

```text
sad/src/circuits/context.cuh
sad/src/runtime/lookups.cuh
sad/src/runtime/workspace.cuh
```

`run_forward()`、`run_backward()`、`sad_energy_and_grad()` 的签名保持不变。

EQNN 不使用 host-precomputed diagonal lookup 或 selected-map buffer；已有字段继续服务
原有电路，不能把它们改造成 EQNN 专属结构。

## 10. 测试计划

### 10.1 PennyLane circuit tests

修改：

```text
pennylane-lightning/tests/test_circuits.py
```

覆盖：

```text
2q: 最小 complete graph
3q: odd dummy schedule
4q: 3 个 perfect matching phases
5q: odd dummy 且多 phase
```

检查：

```text
每层只有 3 个 source parameters
RX/RY occurrence 数正确
CX-RZ-CX 顺序正确
每个 i<j 恰好出现一次
所有 pair RZ 使用同一 g_l
parameter_count = 3L
observable = sum X_i / n
```

### 10.2 Native tests

修改：

```text
pennylane-lightning/tests/test_native_runner.py
```

检查 QNode 与 native：

```text
energy
full gradient
shared parameter reduction
float32/float64
```

### 10.3 SAD tests

修改：

```text
sad/tests/test_sad_runner.py
```

关键规模：

```text
2q, 1 layer
3q, 2 layers
4q, 2 layers
5q, 3 layers
8q, 3 layers
12q, 2 layers  # cross-tile/multi-phase
```

比较 SAD、PennyLane QNode 和 Lightning native 的 energy/full gradient。

容差：

```text
float32 absolute = 3e-5
float64 energy   = 1e-10
float64 gradient = 1e-9
```

中心有限差分至少覆盖：

```text
layer 0 a
layer 0 b
layer 0 g
最后一层 g
odd-qubit schedule 下的 g
```

### 10.4 CUDA safety/resource

运行：

```text
compute-sanitizer --tool memcheck
compute-sanitizer --tool racecheck
nvcc --ptxas-options=-v
```

检查：

```text
dummy pair 不产生非法 wire
CX gather 无 shared/global race
最后一个不完整 tile
phase 间 grid barrier
float32/float64 alignment
0 spill 或可接受的已量化 spill
```

### 10.5 原有电路回归

必须运行完整现有测试，确认 RA-HEA、SU2-HEA、RZZ-HEA、QAOA、XXZ-HVA、MERA
均无行为和性能路径回归。

## 11. Benchmark 对齐

修改现有三个 benchmark 的 circuit 配置，不新增 runner：

```text
benchmark/benchmark_sad.py
benchmark/benchmark_pennylane_lightning.py
benchmark/benchmark_lightning_native.py
```

不得新增 `benchmark_eqnn.py` 或 EQNN 专用计时函数。当前 benchmark 已有公共 circuit
循环和 `layers_for_circuit()`；EQNN 直接使用公共 `LAYERS`（或只在该现有函数中增加
一个分支），不能新建 `eqnn_layers()`。

EQNN layers 是用户给定 benchmark depth，不根据 qubits 推导。建议：

```text
qubits    = 4,6,8,...,28
layers    = 1,2,4,8
precision = float64 主 sweep，float32 关键点
seed      = 42
```

当前三个 benchmark 的 `LAYERS` 是单个整数，不是 tuple。上述 `1/2/4/8` 表示分别修改
同一个 `LAYERS` 配置并执行四轮，不能为了 EQNN 新增第二个 layer loop 或专用 runner。

分别保存：

```text
benchmark/results/eqnn_sad_gpu.csv
benchmark/results/eqnn_pennylane_gpu.csv
benchmark/results/eqnn_native_gpu.csv
benchmark/results/eqnn_comparison.csv
```

记录：

```text
forward/Hamiltonian/backward mean/median
total mean/median
energy/full gradient
logical parameter count = 3L
gate occurrence count
kernel variant
workspace bytes
```

第一版不修改 `_select_library()` 的 circuit ID 6 dispatch。只有完整 variant sweep 后，
某个现有公共 variant 在连续 qubit 区间稳定胜出，才增加 measured dispatch。

## 12. 分阶段实施

### Phase A：reference 和注册

1. 增加唯一 `_equivariant_qnn()` 和 `CircuitSpec`。
2. 增加 native operations/Hamiltonian 分支。
3. 固定 round-robin golden topology 和共享参数测试。
4. 增加 C/Python circuit ID 和 `3*layers` 参数量。

验收：PennyLane QNode 与 native energy/full gradient 对齐。

### Phase B：第一版 SAD correctness

1. 增加一个 layout 和一个 executor。
2. 增加一对 forward/backward kernel/launcher。
3. kernel 内计算 pair phase，不增加 workspace metadata。
4. 增加一个 EQNN Hamiltonian kernel。

验收：2/3/4/5/8/12q、float32/float64 全梯度和有限差分通过。

Phase B 使用的 production 名称从一开始就必须是最终的
`equivariant_qnn_forward_kernel/backward_kernel` 和对应 launcher；不得冠以 `legacy`。

### Phase X：闭式全连接 RZZ 相位

Phase X 位于 Phase B 的 correctness kernel 与 Phase C 的 tile/shared-memory 优化之间。
它是一次算法复杂度优化：不改变 executor、context、workspace、launcher 或公共 C API，
只替换 `equivariant_qnn_forward_kernel` 和 `equivariant_qnn_backward_kernel` 中全连接
RZZ 的 pair traversal。Phase X 的目标是消除每个 amplitude 内部枚举
`n * (n - 1) / 2` 个 pair 的双重 `for` 循环；不能把“只减少 kernel launch 次数”误报为
本阶段完成。

1. **固定闭式公式和角度约定。** 对 computational-basis amplitude `x` 定义
   `z_i = +1`（bit `i` 为 0）或 `-1`（bit `i` 为 1），并令
   `z_sum = sum_i z_i = n - 2 * popcount(x)`。所有 pair 的 RZZ 乘积满足：

   ```text
   S(x) = sum_{i < j} z_i * z_j
        = (z_sum * z_sum - n) / 2
   product_{i < j} exp(-i * g_l * z_i * z_j / 2)
        = exp(-i * g_l * S(x) / 2)
   ```

   `g_l` 仍是实际门角，严格沿用第 2.2 节的 `CX-RZ-CX` 定义；不得引入 `2*g_l` 或
   从论文参数 `gamma_l` 在 kernel 内重新缩放。`S(x)` 对合法 `n` 和 `z_sum` 必为整数，
   因此不得用浮点近似计算 pair count 或 `S`。

2. **确定 kernel 内的数据路径。** 在 forward 中，每个线程只读取一次 amplitude，使用
   `popcount`（CUDA `__popc`/`__popcll`，按 state index 类型选择）得到 `z_sum`，再计算
   `S`，最后将 amplitude 乘以 `exp(-i * g_l * S / 2)`。Phase X 不得恢复原始
   `g_l` 角度并调用 `sin/cos` 作为逐 amplitude 的慢路径，也不得再次生成 pair map。
   当前 workspace 只保证 `RotationCoefficients(sin(g_l/2), cos(g_l/2))`；实现必须在不
   修改 workspace 的前提下工作。推荐使用单位复数基底
   `u = cos(g_l/2) - i*sin(g_l/2)` 的整数幂 `u^S`，采用 binary exponentiation，
   对负 `S` 使用共轭基底。这样 pair 计算从每 amplitude 的 `O(n^2)` 降为
   `O(popcount + log |S|)`，不产生额外 full-state pass；若后续允许传递原始角度，才可
   将最后一步替换为一次 `sincos`，但必须单独记录其数值和性能影响。

3. **按同一闭式公式实现 backward。** backward 不再反向遍历 phase/pair，也不执行
   `CX(phi/lambda)`。对当前 amplitude 使用同一个 `S(x)` 和逆相位 `u^{-S}` 更新
   `phi_state` 与 `lambda_state`。共享参数 `g_l` 的导数使用

   ```text
   d/dg_l exp(-i * g_l * S / 2)
       = (-i * S / 2) * exp(-i * g_l * S / 2)
   ```

   在现有 block reduction + `atomicAdd` 框架中累加到 `gradients[3*l + 2]`；贡献的
   共轭/虚部符号必须以当前 `imag_conjugate_product()` 约定为准，并用 Phase B 的
   pair-by-pair kernel 做逐 amplitude 对照，不能凭手工符号假设。RX/RY 的逆序和梯度
   累加保持原设计不变。

4. **保持奇数 qubit 和所有层语义。** 闭式 `S(x)` 不依赖 round-robin phase 顺序，
   但只对“每个无序真实 pair 恰好一次、所有 pair 共用同一个 `g_l`”成立。必须继续
   跳过 dummy，且每层只使用该层的 `parameter_offset + 2`。不得将此公式用于参数不同
   的 pair、非全连接拓扑或尚未证明可交换的门序。

5. **建立分层 correctness 验证。** 至少增加以下测试：

   - CPU/reference 对 `n=2..16` 的每个 basis index 穷举比较 pair product 与闭式相位；
     对 `n=17..28` 覆盖全零、全一、单 bit、交替 bit、各 popcount 边界和固定 seed
     随机样本，避免把 `2^28` 穷举引入常规测试；
   - float32/float64 比较 forward state，覆盖 `g=0`、负 `g`、接近 `pi` 的角度以及
     `S=0`、最大正/负 `S`；
   - backward 的 `g_l` 梯度与 Phase B pair-by-pair backward 逐 amplitude 对齐；
   - 2/3/4/5/8/12/16/20/24/28q 的 energy、完整 gradient 和 finite difference；
   - odd qubits 验证 dummy 不改变闭式结果，且不会发生非法 bit shift 或整数溢出。

   误差阈值沿用现有测试的 float32/float64 阈值，并额外记录最大 state、energy、gradient
   误差。若 binary exponentiation 的累积误差超过阈值，必须先改进复数幂实现（例如按
   `S` 的符号选择共轭、减少不必要的重归一化），不能退回 pair loop 作为默认路径。

6. **分离算法收益和 kernel 调优收益。** Phase X 完成后，使用与 Phase B 相同的
   `layers/qubits/precision/seed/steps`，分别记录 `forward/backward/total` 时间和
   state-vector 访存次数；至少与 pair-by-pair kernel 做两轮重复对比。验收要求是在
   相同数值结果下，RZZ 部分的每-amplitude 算术复杂度不再含 `n^2` pair loop，并在
   16q 以上观察到可重复的 scaling 改善。Phase C 随后只能在 Phase X 的闭式路径上
   增加 tile、mailbox、occupancy 和 shared-parameter reduction 优化，不得重新引入
   pair-by-pair 默认实现。

### Phase C：tile 与 shared-parameter 性能

1. RX/RY tile-local shared-parameter execution。
2. 将 Phase X 的 `popcount -> S(x) -> u^S` 闭式 RZZ 相位合入 tile load/store，
   每个 amplitude 只计算一次闭式相位，不恢复 `CX-RZ-CX` pair traversal。
3. phi/lambda mailbox 分时复用。
4. profile shared-gamma atomic reduction。

验收：相对逐 gate baseline 的 forward/backward 时间下降，sanitizer 通过，无不可接受
spill。

Phase C 只继续替换 Phase X 完成后的同名 kernel/launcher 内部算法。不能保留 Phase B
或 Phase X 前的 kernel 作为第二套路径，也不能修改 executor/context/run signatures。

### Phase D：variant sweep 和 dispatch

1. 编译现有五个公共 variants。
2. 完整 qubit/layer sweep 至少重复两轮。
3. 只有稳定改善达到项目阈值时增加 ID 6 dispatch。
4. 更新 benchmark comparison 和最终结构文档。

## 13. 主文件结构审计

### 13.1 原始五种电路的实际结构基准

以下结论来自当前仓库中的实际主文件，而不是抽象约定：

| 层级 | 原始五种电路的实际结构 | EQNN 必须采用的对应结构 | 不能做的变化 |
|---|---|---|---|
| PennyLane builder | `circuits.py` 中每种电路各一个 `_ra_hea()`/`_su2_hea()`/`_rzz_hea()`/`_qaoa()`/`_xxz_hva()`；`_ring_cnot()` 是已有公共 helper | 只增加一个 `_equivariant_qnn()`；拓扑留在函数体 | 不增加 `_equivariant_pairs()`、`_equivariant_layers()` 等第二/第三个 def |
| PennyLane 注册 | 每种电路一个现有 `register_circuit(CircuitSpec(...))` 调用 | 一个 EQNN `CircuitSpec` 注册 | 不修改 `CircuitSpec` 或注册机制 |
| Hamiltonian | 一个已有 `build_hamiltonian()`，按 `circuit_spec.name` 分支 | 在同一个函数中增加一个 EQNN 分支 | 不新增 `build_equivariant_hamiltonian()` |
| Lightning native | 一个 `_build_operations()`，内部按 circuit name 分支；共享参数通过已有 `source_parameter` 表达 | 在同一个函数中增加一个 EQNN 分支 | 不新增 `_build_equivariant_operations()` 或 EQNN runner |
| C++ circuit | 每个 circuit header 一个 layout（如 `RaLayerLayout`、`QaoaLayerLayout`、`XxzLayerLayout`）和一个 executor 特化 | 一个 `EquivariantQNNLayerLayout` + 一个 executor 特化 | 不增加 `EquivariantQNNExecutorOptimized` 等第二特化 |
| executor 方法 | 公共方法集合为 `validate`、lookup/init、forward 初始/逐层、backward 逐层及已有 mode wrapper；runtime 有六种 mode，SU2 的 phased 方法是既有特殊例外 | 使用同一公共方法集合；EQNN 的各 mode wrapper 仍收敛到同一对 launcher | 不修改公共方法签名，不新增 `forward_layer_phased()` 或 EQNN 专用外层循环 |
| CUDA kernel | RA/SU2/RZZ 复用公共 rotation/diagonal kernel；XXZ/MERA 使用“private helper + 一对 matching kernel/launcher”；QAOA 的 init kernel 是其已有状态初始化例外 | EQNN 采用“private helper + 一对 forward/backward kernel/launcher” | 不保留 legacy/optimized 两套 EQNN matching API，不把 helper提升为 public entry |
| Hamiltonian kernel | `hamiltonian.cuh` 中按 circuit 使用已有专用 kernel 分支 | 只增加一个 EQNN Hamiltonian kernel 和一个 runner 分支 | 不增加多个 EQNN observable kernel |
| 公共 runtime | `context.cuh`、`workspace.cuh`、`lookups.cuh` 为共享边界；现有 circuit-specific metadata 不能随意复制 | 不增加 EQNN 字段、buffer、map builder 或参数 | 不修改 `run_forward`/`run_backward`/C API 签名 |

这里“每个 circuit 一个 kernel”并不是指仓库中所有公共模板只能有一个函数；原始代码
已经有公共 rotation/diagonal primitive 和多个复用 launcher。对齐要求针对 EQNN 新增的
production 入口数量，不能把同文件 private helper 误报为额外电路入口，也不能反过来
用 private helper 名义增加第二条执行路径。

| 文件 | 允许的 EQNN 变化 | 强制数量/边界 |
|---|---|---|
| `circuits.py` | 一个 builder、一个注册、一个 Hamiltonian 分支 | `_equivariant_qnn` 数量为 1 |
| `native.py` | `_build_operations` 一个分支、`_build_hamiltonian` 一个分支 | 不新增 runner/helper topology API |
| `sad_api.h` | 末尾追加 ID 6 | 原 ID 不变 |
| `circuit_dispatch.cuh` | 一个 case | runtime-template 边界不变 |
| `circuit_execution.cuh` | 一个 include、一个参数量特判 | run signatures 不变 |
| `circuits/equivariant_qnn.cuh` | 一个 layout、一个 executor | 方法集合与现有 executor 对齐 |
| `kernels/equivariant_qnn.cuh` | private helpers + 一对 kernel/launcher | 不保留平行实现 |
| `hamiltonian.cuh` | 一个 EQNN Hamiltonian kernel | 不新增第二个 observable variant |
| `runner.cuh` | 一个 Hamiltonian dispatch 分支 | 公共 API 不变 |
| `context.cuh` | 不修改 | 无 EQNN 字段 |
| `lookups.cuh` | 不修改 | 无 EQNN map builder |
| `workspace.cuh` | 不修改 | 无 EQNN buffer/count |
| Python SAD runner | registry、参数量、canonical name | 不新增 EQNN runner |
| 三个 benchmark 主文件 | 现有 circuit tuple/config 和已有 layer 选择函数内分支 | 不新增 EQNN benchmark/计时函数 |

最终 code review checklist：

```text
[ ] PennyLane EQNN builder 恰好 1 个
[ ] circuits.py 因 EQNN 新增的顶层 def 恰好 1 个
[ ] native.py 因 EQNN 新增的顶层 def 恰好 0 个
[ ] 所有 RZZ 均以 CX-RZ-CX 表达
[ ] 每层参数量恰好为 3
[ ] round-robin 覆盖全部 i<j 且无重复
[ ] odd qubits dummy 不产生 gate
[ ] Native shared source_parameter 正确归并
[ ] EquivariantQNNLayerLayout 恰好 1 个
[ ] CircuitExecutor<SAD_CIRCUIT_EQUIVARIANT_QNN,T> 恰好 1 个
[ ] EQNN matching __global__ kernel 恰好 2 个
[ ] EQNN matching launcher 恰好 2 个
[ ] EQNN Hamiltonian kernel 恰好 1 个
[ ] EQNN 专用 host topology/helper 函数恰好 0 个
[ ] EQNN 专用 runner 和 benchmark 函数恰好 0 个
[ ] context/workspace/lookups 无 EQNN 字段或 builder
[ ] run_forward/run_backward/sad_energy_and_grad 签名未变化
[ ] 六种 execution mode 均由现有 runtime 分派且收敛到同一对 EQNN launcher
[ ] 公共调优宏未增加 SAD_EQNN_*
[ ] energy/full gradient/finite difference/sanitizer 通过
[ ] 原有六种电路完整回归通过
```

任何一项不满足，都应视为结构未与当前项目对齐，不能只凭数值正确或性能提升接受。
