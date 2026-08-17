# MERA 电路实现计划（按现有代码结构对齐）

## 1. 对齐原则

本计划只增加 MERA 所需代码，不重构现有五类电路的公共接口。新增代码必须遵循当前项目已经使用的模式：

| 层次 | 当前模式 | MERA 对齐方式 |
|---|---|---|
| PennyLane 电路 | `circuits.py` 中每类电路只有一个 `_circuit_name()` | 只新增一个 `_mera()` |
| PennyLane 注册 | 一个 `register_circuit(CircuitSpec(...))` | 注册处用 lambda 计算参数量 |
| C++ 电路 | 一个 layout struct + 一个 `CircuitExecutor` 特化 | `MeraLayerLayout` + `CircuitExecutor<SAD_CIRCUIT_MERA,T>` |
| 参数量例外 | `expected_parameter_count()` 对 QAOA 特判 | 在同一函数中增加 MERA 特判 |
| Topology | QAOA/ring 直接由 qubits 计算，XXZ 仅为 tile phase 准备 maps | MERA wires 直接由 layer/pair index 公式计算 |
| 专用 kernel | QAOA/XXZ 各有电路相关 kernel | 新增 `kernels/mera.cuh` |
| Hamiltonian 分派 | `runner.cuh` 按 circuit ID 分支 | 增加 MERA 的单 wire Z 分支 |
| Python SAD 注册 | `_CIRCUITS` tuple + 少量 circuit ID 特判 | 增加 ID 5 和 MERA 参数量特判 |

明确不做以下重构：

- 不在 `circuits.py` 增加 `mera_stage_count()`、`build_mera_topology()`、`mera_parameter_count()` 等多个函数；
- 不修改 `CircuitSpec` 数据结构；
- 不统一改造现有五个 `CircuitExecutor::validate()`；
- 不把现有 context 改造成新的多态/view 框架；
- 不重写 QAOA/XXZ 的参数量和 metadata 逻辑；
- 不给 `PreparedWorkspace`、forward/backward context 或 `run_forward/run_backward` 增加 MERA 字段。

## 2. 固定的 MERA 电路定义

### 2.1 Block

`D` disentangler 和 `U` coarse-graining block 使用相同门形状，但各自拥有独立参数：

```text
Block(a,b; theta_a,theta_b)

qa ----RY(theta_a)----o----
                       |
qb ----RY(theta_b)----X----
```

严格门序：

```text
RY(a, theta_a)
RY(b, theta_b)
CX(control=a, target=b)
```

### 2.2 每个 stage 的 pair

当前 active wires 为：

```text
[a0,a1,a2,...,a(m-1)]
```

先执行跨 coarse-graining block 边界的 D matching：

```text
D = [(a1,a2), (a3,a4), ...]
```

再执行 U matching：

```text
U = [(a0,a1), (a2,a3), ...]
```

每个 U 保留右侧 wire。若 active wire 数为奇数，最后一条 wire carry 到下一 stage：

```text
next_active = [a1,a3,...] + [last_wire if m is odd]
```

最后只剩两条 active wires 时没有 D，只执行最终 U。

### 2.3 8-qubit golden topology

```text
stage 0:
    D: (1,2), (3,4), (5,6)
    U: (0,1), (2,3), (4,5), (6,7)
    active -> [1,3,5,7]

stage 1:
    D: (3,5)
    U: (1,3), (5,7)
    active -> [3,7]

stage 2:
    D: none
    U: (3,7)
    active -> [7]

observable: Z(7)
```

### 2.4 非 `2^k` qubit

6-qubit golden topology：

```text
stage 0:
    D: (1,2), (3,4)
    U: (0,1), (2,3), (4,5)
    active -> [1,3,5]

stage 1:
    D: (3,5)
    U: (1,3)
    q5 carry
    active -> [3,5]

stage 2:
    D: none
    U: (3,5)
    active -> [5]

observable: Z(5)
```

carry wire 不参加该 stage 的 U，但可以参加 D，例如上面的 `D(3,5)`。

### 2.5 层数与参数量

MERA stage 数固定为：

```text
stage_count = ceil(log2(qubits)) = (qubits - 1).bit_length()  # Python
```

公共 API 仍保持 `scalability=(qubits,layers)`，但 MERA 要求：

```text
layers == ceil(log2(qubits))
```

每个 stage 的 D/U block 总数为 `active_count - 1`。所有 stage 的总 block 数有闭式表达：

```text
block_count = 2 * (qubits - 1) - popcount(qubits - 1)
parameter_count = 2 * block_count
                = 4 * (qubits - 1) - 2 * popcount(qubits - 1)
```

示例：

| qubits | layers | blocks | parameters |
|---:|---:|---:|---:|
| 4 | 2 | 4 | 8 |
| 6 | 3 | 8 | 16 |
| 8 | 3 | 11 | 22 |
| 10 | 4 | 14 | 28 |

参数顺序固定为：

```text
for stage in forward order:
    all D pairs from left to right, each pair left parameter then right parameter
    all U pairs from left to right, each pair left parameter then right parameter
```

## 3. PennyLane：严格保持“一类电路一个 def”

修改文件：

```text
pennylane-lightning/src/pennylane_lightning_baseline/circuits.py
```

### 3.1 只新增 `_mera()`

与 `_ra_hea()`、`_su2_hea()`、`_qaoa()` 相同，只增加一个电路 builder：

```python
def _mera(params: object, qubits: int, layers: int) -> None:
    active = list(range(qubits))
    cursor = 0
    while len(active) > 1:
        # D matching: cross neighboring U-block boundaries.
        for index in range(1, len(active) - 1, 2):
            left, right = active[index], active[index + 1]
            qml.RY(params[cursor], wires=left)
            qml.RY(params[cursor + 1], wires=right)
            qml.CNOT(wires=(left, right))
            cursor += 2

        # U matching: coarse-grain adjacent active wires.
        next_active = []
        for index in range(0, len(active) - 1, 2):
            left, right = active[index], active[index + 1]
            qml.RY(params[cursor], wires=left)
            qml.RY(params[cursor + 1], wires=right)
            qml.CNOT(wires=(left, right))
            cursor += 2
            next_active.append(right)

        if len(active) % 2:
            next_active.append(active[-1])
        active = next_active
```

这里不新增 topology helper。topology 只在 `_mera()` 内使用局部 `active` list 生成，与现有电路 builder 的自包含风格一致。

### 3.2 注册方式

继续使用现有 `CircuitSpec`，参数量闭式公式直接写在注册处：

```python
register_circuit(
    CircuitSpec(
        name="mera",
        parameter_count_fn=lambda qubits, layers: (
            4 * (qubits - 1) - 2 * (qubits - 1).bit_count()
            if layers == (qubits - 1).bit_length()
            else 0
        ),
        builder=_mera,
        minimum_qubits=2,
    )
)
```

返回 0 会触发现有 `CircuitSpec.parameter_count()` 的 invalid parameter count 校验。`_mera()` 与当前 `_ra_hea()`、`_su2_hea()` 等 builder 一样只负责发门，不重复承担输入校验。

不设置 `requires_even_qubits=True`。

### 3.3 Hamiltonian

在现有 `build_hamiltonian()` 中增加与 QAOA、XXZ 同级的分支：

```python
if circuit_spec.name == "mera":
    return qml.Hamiltonian([1.0], [qml.PauliZ(qubits - 1)])
```

当前“保留右输出 + odd carry”规则保证最终 active wire 始终是 `qubits - 1`，因此不需要额外 topology helper。

### 3.4 PennyLane 测试

修改：

```text
pennylane-lightning/tests/test_circuits.py
pennylane-lightning/tests/test_runner.py
pennylane-lightning/tests/test_native_runner.py
```

增加：

- 4q/2 layers 参数量为 8；
- 6q/3 layers 参数量为 16；
- 8q/3 layers 参数量为 22；
- 6q、8q tape 的完整 gate 顺序和 wires 与 golden topology 一致；
- operation 只含 `RY`、`CNOT`；
- observable 是 `Z(qubits-1)`；
- 错误 layers 被拒绝；
- QNode 与 Lightning native 都能得到有限 energy/gradient。

## 4. SAD 公共注册：沿用当前特判结构

### 4.1 Circuit ID

修改 `sad/include/sad_api.h`：

```cpp
SAD_CIRCUIT_MERA = 5,
```

现有 ID `0..4` 不变。

### 4.2 Compile-time dispatch

修改 `sad/src/runtime/circuit_dispatch.cuh`，在现有 switch 中增加：

```cpp
case SAD_CIRCUIT_MERA:
    return std::forward<Function>(function)(
        std::integral_constant<int, SAD_CIRCUIT_MERA>{});
```

修改 `sad/src/runtime/circuit_execution.cuh`，与其他电路 include 并列加入：

```cpp
#include "../circuits/mera.cuh"
```

### 4.3 参数量

不修改现有五个 executor，也不新增统一 `parameter_count()` 虚拟接口。直接在当前已有 QAOA 特判的 `expected_parameter_count()` 中增加 MERA：

```cpp
template <int Circuit>
inline size_t expected_parameter_count(int qubits, int layers) {
    if constexpr (Circuit == SAD_CIRCUIT_QAOA) {
        return static_cast<size_t>(2) * layers;
    }
    if constexpr (Circuit == SAD_CIRCUIT_MERA) {
        int expected_layers = 0;
        for (int active = qubits; active > 1; active = (active + 1) / 2) {
            ++expected_layers;
        }
        if (layers != expected_layers) {
            throw std::invalid_argument(
                "MERA layers must equal ceil(log2(qubits))");
        }
        const int value = qubits - 1;
        return static_cast<size_t>(4 * value - 2 * __builtin_popcount(value));
    }
    return ...;  // 保持当前固定参数公式
}
```

`CircuitExecutor<SAD_CIRCUIT_MERA,T>::kParametersPerQubitLayer` 可以设为 0，和 QAOA 一样由特判处理。`validate(int qubits)` 保持现有签名，只检查 qubit 范围内的 MERA 特有约束；layers 校验放在已有参数量入口中。

为避免 `__builtin_popcount` 可移植性问题，也可以在 MERA 分支中用小循环计数；不能为此改变其他电路的接口。

## 5. SAD 电路文件：与五个现有 executor 同形

新增：

```text
sad/src/circuits/mera.cuh
```

### 5.1 `MeraLayerLayout`

与 `RaLayerLayout`、`Su2LayerLayout`、`RzzLayerLayout` 一样，在电路文件中定义 layout：

```cpp
struct MeraLayerLayout {
    int active_count;
    int d_pair_count;
    int d_parameter_offset;
    int u_pair_count;
    int u_parameter_offset;

    static auto at(int layer, int qubits) -> MeraLayerLayout;
};
```

`at()` 使用一个不分配内存的小循环，从 `active_count=qubits` 推进到目标 layer，只累计 parameter offset。计算规则：

```text
d_count = floor((active_count - 1) / 2)
u_count = floor(active_count / 2)
stage parameters = 2 * (d_count + u_count)
next active_count = ceil(active_count / 2)
```

pair wire 不存入 workspace。对 stage `layer`，第 `index` 个 active wire 可直接计算：

```cpp
wire = min(((index + 1) << layer) - 1, qubits - 1);
```

因此：

```text
D pair j = (active_wire(layer, 2j+1), active_wire(layer, 2j+2))
U pair j = (active_wire(layer, 2j),   active_wire(layer, 2j+1))
```

该公式对 6q 的 carry 和 8q 的完整二叉树都成立，kernel 不需要 device pair arrays。

### 5.2 `CircuitExecutor`

与其他文件保持相同方法集合：

```cpp
template <typename T>
struct CircuitExecutor<SAD_CIRCUIT_MERA, T> {
    static constexpr int kParametersPerQubitLayer = 0;

    static void validate(int qubits);
    static void append_diagonal_lookups(...);       // no-op
    static auto build_initial_state_lookup(...);   // |0...0>
    static void apply_layer(...);
    static void forward_initial(...);
    static void forward_layer(...);
    static void forward_layer_optimized(...);
    static void backward_layer(...);
    static void backward_layer_optimized(...);
    static void backward_layer_fused(...);
};
```

MERA 不新增初始化 kernel。`forward_initial()` 清零 `phi` 的两个 state buffers 后，复用现有 `hamiltonian.cuh` 中的 `initialise_zero_state_kernel`，再调用 stage 0 的 `apply_layer()`；这与现有 runtime 的 zero-state 约定一致。

行为：

| 方法 | 行为 |
|---|---|
| `apply_layer` | 调用一次 D matching forward，再调用一次 U matching forward |
| `forward_initial` | 初始化 zero state，然后调用 `apply_layer(0, context)` |
| `forward_layer` | layer 0 时初始化，然后调用 `apply_layer()`，与 XXZ 写法一致 |
| `forward_layer_optimized` | 直接调用 `apply_layer()` |
| `backward_layer` | 调用 U matching backward，再调用 D matching backward |
| `backward_layer_optimized` | 直接调用 `backward_layer()` |
| `backward_layer_fused` | 直接调用 `backward_layer()` |

现有 `circuit_execution.cuh` 已经按 `layer=0..layers-1` forward、按反序 backward，因此不新增 MERA 专用外层循环。

## 6. Workspace、context 与 runtime 签名保持不变

MERA topology 完全由 `qubits`、`layer`、matching 类型和 pair index 计算，不需要额外 host/device metadata。因此以下文件不增加 MERA 字段或参数：

```text
sad/src/runtime/lookups.cuh
sad/src/runtime/workspace.cuh
sad/src/circuits/context.cuh
sad/src/runtime/runner.cuh 的 run_forward/run_backward 调用参数
```

MERA executor 直接使用 context 已有字段：

```text
qubits
state_size
rotation_coefficients
gradients
multiprocessors
ordinary_grid
phi / lambda
```

这比照搬 XXZ metadata 更贴合当前结构：XXZ 的 maps 是 tile dispatch 所需，而 MERA 的 wire 编号已有闭式公式。只有以后实测证明 MERA 需要 host-precomputed tile maps 时，才按 XXZ 模式增加，并作为独立优化提交。

## 7. CUDA kernels

新增：

```text
sad/src/kernels/mera.cuh
```

### 7.1 复用与不复用

复用现有基础设施：

- `Complex<T>`；
- `RotationCoefficients<T>`；
- `StatePair<T>` ping-pong state；
- host 端 `sin(theta/2)`/`cos(theta/2)` 参数预计算；
- gradient `double*` 约定；
- CUDA error checking、grid sizing、timing；
- `legacy/optimized/all-fused` execution modes。

不直接复用：

- `ring_cnot_permutation_kernel`：它只表示固定 ring topology；
- XXZ matching kernel：它绑定 RXX/RYY/RZZ algebra；
- RA real-amplitude kernel：MERA 的最终 Z 目标不保证全程能直接套用该存储格式。

### 7.2 Kernel 文件的顶层结构

严格仿照 `kernels/xxz.cuh`，只定义一对 kernel 和一对 launcher：

```cpp
mera_matching_forward_kernel
mera_matching_backward_kernel
launch_mera_matching_forward
launch_mera_matching_backward
```

不再增加另一套 `mera_block_forward/backward`。`legacy`、`optimized`、`all-fused` 第一版都调用同一对 matching launchers，与 XXZ-HVA 当前实现一致；正确性 oracle 是 PennyLane reference 和有限差分。

每次 launch 处理一个 disjoint matching：

```text
一个 D matching launch
一个 U matching launch
```

D 与 U 不能合并为无序集合，因为两个 matching 之间可能共享 wires，必须保留 D 后 U 的 barrier。

kernel 接收 `layer`、`pair_count`、parameter offset 和一个 D/U matching 标记，并用第 5.1 节公式计算每个 pair 的 wires。优化策略参考 `kernels/xxz.cuh`，而不是参考 ring CNOT：

- kernel 由 layer/pair index 计算 wires；
- kernel 根据 wire masks 处理 tile 内 pair；
- tile 外 pair 使用安全的 ping-pong permutation/gather；
- forward/backward 分别调优 threads/register bits；
- 禁止在 global state 上做存在读写竞争的 naive in-place CX。

### 7.3 为什么需要专用 kernel

MERA 没有新 gate algebra，但有现有 kernel 没有表达的 topology：

```text
每个 stage 的 sparse arbitrary-wire matching
pair 数逐层减少
odd active wire carry
D/U 两个有序 matching
```

因此专用 kernel 只负责 matching/topology，状态、adjoint、参数、计时和 runtime 全部复用。这与当前 `xxz_hva.cuh + kernels/xxz.cuh` 的边界一致。

### 7.4 MERA 使用的公共 kernel 宏

MERA 第一版不增加 `SAD_MERA_*` 宏，直接使用 `sad/src/core/cuda_common.cuh` 中现有的公共执行形状。宏、派生常量和 kernel 的对应关系固定如下：

| 公共编译宏 | 源码默认值 | 派生常量 | 对应 MERA kernel | 控制内容 |
|---|---:|---|---|---|
| `SAD_FORWARD_BLOCK_THREADS` | `SAD_BLOCK_THREADS`，最终默认 128 | `kForwardBlockThreads` | `mera_matching_forward_kernel` | forward 每个 CUDA block 的线程数 |
| `SAD_FORWARD_REGISTER_BITS` | `SAD_REGISTER_BITS`，最终默认 2 | `kForwardRegisterBits`、`kForwardRegisterAmplitudes` | `mera_matching_forward_kernel` | forward 每线程持有 `2^bits` 个 amplitudes |
| `SAD_BLOCK_THREADS` | 128 | `kBlockThreads` | `mera_matching_backward_kernel` | backward 每个 CUDA block 的线程数 |
| `SAD_REGISTER_BITS` | 2 | `kRegisterBits`、`kRegisterAmplitudes` | `mera_matching_backward_kernel` | backward 每线程持有 `2^bits` 个 phi/lambda amplitudes |
| `SAD_ORDINARY_BLOCK_THREADS` | 128 | `kOrdinaryBlockThreads` | `mera_hamiltonian_kernel` | 单 wire Z observable kernel 的 block 线程数 |

默认编译没有传入任何 variant flags，因此 MERA 的有效默认形状是：

```text
forward  = 128 threads, 2 register bits, 4 amplitudes/thread
backward = 128 threads, 2 register bits, 4 amplitudes/thread
ordinary = 128 threads
variant label = f128r2_b128r2
```

tile 大小继续使用现有公共公式：

```text
forward_tile_bits = 5 + SAD_FORWARD_REGISTER_BITS
                      + log2(SAD_FORWARD_BLOCK_THREADS / 32)

backward_tile_bits = 5 + SAD_REGISTER_BITS
                       + log2(SAD_BLOCK_THREADS / 32)

tile_amplitudes = 2^tile_bits
```

launcher 必须像 `launch_xxz_matching_forward/backward()` 一样直接使用这些派生常量：

```cpp
// forward launcher
dim3(kForwardBlockThreads)
kForwardRegisterAmplitudes
kForwardTileBits
kForwardTileAmplitudes

// backward launcher
dim3(kBlockThreads)
kRegisterAmplitudes
kTileBits
kTileAmplitudes

// Hamiltonian launch
dim3(kOrdinaryBlockThreads)
```

#### 编译允许范围

公共头文件现有 `static_assert` 允许：

| 参数 | 单项允许范围 | 组合约束 |
|---|---|---|
| forward block threads | `{64,128,256,512}` | 必须是完整 warp，且 warp 数为 2 的幂 |
| backward block threads | `{64,128,256,512}` | 必须是完整 warp，且 warp 数为 2 的幂 |
| forward register bits | `2..6` | `kForwardTileBits <= 12` |
| backward register bits | `2..6` | `kTileBits <= 12` |
| ordinary block threads | `{64,128,256,512}` | 必须是完整 warp |

单项在允许范围内不代表组合一定可编译。例如 threads 增加会增加 warp bits，从而减少 register bits 的可用上限；所有候选必须通过现有 `static_assert` 和实际 shared-memory/occupancy 检查。

#### 第一轮 MERA 候选范围

第一轮只复用项目已经构建和测试过的安全形状：

| 方向 | 候选 `threads x register bits` | 来源 |
|---|---|---|
| forward | `128x2`、`64x3`、`64x4`、`128x3` | 默认库及现有 `f64r3`、`f64r4`、`f128r3` variants |
| backward | `128x2`、`64x4`、`128x3` | 默认库及现有 `b64r4`、`b128r3` variants |
| Hamiltonian | 固定 `128` threads | 首轮不为低占比 observable 单独增加 `.so` variant |

对应现有可组合 library variants：

```text
f128r2_b128r2  # 默认安全基线
f64r4_b64r4
f64r3_b64r4
f128r3_b64r4
f64r4_b128r3
```

MERA 完整 qubit sweep 完成前，`_select_library()` 对 circuit ID 5 返回默认 `f128r2_b128r2`。只有某个现有 variant 在指定 qubit 范围稳定胜出且数值测试通过，才增加 ID 5 的 qubit-dependent dispatch。

#### 第一版明确不使用的公共宏

以下宏虽然属于公共配置，但第一版 MERA matching 没有 phase-map metadata，不应在实现中假装支持：

| 宏 | 第一版状态 | 原因 |
|---|---|---|
| `SAD_FORWARD_FIXED_LOW_LANES` | 不使用 | MERA 第一版不构建 forward selected-map phases |
| `SAD_FIXED_LOW_LANES` | 不使用 | MERA 第一版不构建 backward selected-map phases |
| `SAD_ALTERNATE_PHASES` | 不使用 | D/U 的语义顺序固定，且没有可交替的 phase-map traversal |
| `SAD_DIAGONAL_LOOKUP_BITS` | 不使用 | MERA 没有 RZ/RZZ diagonal lookup |

若后续为 MERA 增加 host-precomputed tile phase maps，必须在单独优化阶段更新本节、测试和 variant dispatch，不能静默让这些宏改变电路语义。

## 8. Observable 与 runner 分派

在 `sad/src/kernels/hamiltonian.cuh` 增加：

```cpp
template <typename T>
__global__ void mera_hamiltonian_kernel(
    const Complex<T>* phi,
    Complex<T>* lambda,
    uint64_t state_size,
    int target_wire,
    double* energy);
```

它与现有 Hamiltonian kernels 一样，同时生成：

```text
energy = <phi|Z_target|phi>
lambda = Z_target|phi>
```

在 `sad/src/runtime/runner.cuh::run_step()` 当前分支链中增加：

```cpp
else if (config.circuit == SAD_CIRCUIT_MERA) {
    mera_hamiltonian_kernel<T><<<...>>>(
        phi.current,
        lambda.current,
        workspace->state_size,
        config.qubits - 1,
        workspace->energy.get());
}
```

不能让 MERA 落入默认 TFIM `hamiltonian_kernel` 分支。

## 9. Python SAD runner：保持现有字典和特判方式

修改 `sad/python/sad_baseline/runner.py`。

### 9.1 注册

在现有 `_CIRCUITS` 中增加：

```python
"mera": (5, 0),
```

第二个值对 MERA 不使用，与 QAOA 类似由下面的 circuit ID 特判计算参数量。

### 9.2 校验和参数量

在现有 QAOA/XXZ 校验旁增加：

```python
if circuit_id == 5:
    expected_layers = (qubits - 1).bit_length()
    if layers != expected_layers:
        raise ValueError(...)
```

把当前参数量表达式扩展为：

```python
if circuit_id == 3:
    parameter_count = 2 * layers
elif circuit_id == 5:
    value = qubits - 1
    parameter_count = 4 * value - 2 * value.bit_count()
else:
    parameter_count = parameters_per_qubit_layer * qubits * layers
```

canonical name tuple末尾增加 `"mera"`。

### 9.3 Library variant

第一版 `_select_library()` 不为 ID 5 增加专用 `.so`，使用默认安全 variant。完成 correctness 与完整性能 sweep 后，才根据数据增加 MERA/qubits dispatch；不能直接复制 RA/SU2/RZZ 的选择表。

## 10. 测试计划

### 10.1 PennyLane topology 测试

在 `test_circuits.py` 中直接检查 tape，不新增 production topology helper：

- 4q、6q、8q 参数量；
- 6q/8q operation names；
- 每个 CNOT wires；
- flat 参数使用顺序；
- D 在 U 之前；
- odd carry 无多余 U；
- 错误 layers；
- MERA Hamiltonian 只有 `Z(qubits-1)`。

### 10.2 Analytic topology 测试

production code 不新增 topology helper 或 pair metadata，因此主要通过 PennyLane tape、layout 公式和 SAD 数值结果交叉验证。建议 golden pairs：

```text
6q D left/right: [(1,2),(3,4),(3,5)]
6q U left/right: [(0,1),(2,3),(4,5),(1,3),(3,5)]

8q D left/right: [(1,2),(3,4),(5,6),(3,5)]
8q U left/right: [(0,1),(2,3),(4,5),(6,7),(1,3),(5,7),(3,7)]
```

测试还要逐 layer 验证 `active_wire(layer,index)` 公式生成上述 pairs，并验证 `MeraLayerLayout::at()` 的 parameter offsets 连续无重叠。

### 10.3 SAD 数值测试

扩展 `sad/tests/test_sad_runner.py`：

| qubits | layers | 目的 |
|---:|---:|---|
| 3 | 2 | 最小 odd carry |
| 4 | 2 | 二幂基础 |
| 6 | 3 | 非二幂 golden topology |
| 8 | 3 | 标准 MERA |
| 12 | 4 | 跨 phase/tile |

每个关键规模检查：

- SAD vs PennyLane energy；
- full gradient element-wise；
- float32/float64；
- legacy vs optimized；
- all-fused vs optimized；
- parameter count；
- split times sum to total。

沿用当前容差：

```text
float32 absolute tolerance: 3e-5
float64 energy: 1e-10
float64 gradient: 1e-9
```

### 10.4 有限差分

对 3q/4q float64 抽查：

- 第一个 D 参数；
- 第一个 U 参数；
- 最后一个 U 参数。

使用中心有限差分，避免 PennyLane 与 SAD 共享同一个门序错误却仍然相互对齐。

### 10.5 原有回归

必须运行现有全部测试，确认：

- 原 circuit ID 不变；
- 原五类参数量不变；
- workspace/context/run signatures 没有变化；
- 原 variant dispatch 不变。

## 11. Benchmark 对齐

现有 benchmark 对 HEA 使用固定 `LAYERS=8`，MERA 不能直接加入该 tuple 后沿用 8 layers。

不修改所有电路 API，只在 benchmark loop 选择实际 layers：

```python
layers = (
    (qubits - 1).bit_length()
    if circuit == "mera"
    else LAYERS
)
```

修改：

```text
benchmark/benchmark_pennylane_lightning.py
benchmark/benchmark_lightning_native.py
benchmark/benchmark_sad.py
benchmark/compare_sad_pennylane.py
```

第一轮建议覆盖：

```text
qubits = 4,6,8,10,12,14,16,18,20,22,24,26,28
precision = float64
```

新结果写入独立 CSV，不覆盖已有五类电路 baseline：

```text
benchmark/results/mera_pennylane_gpu.csv
benchmark/results/mera_native_gpu.csv
benchmark/results/mera_sad_gpu.csv
benchmark/results/mera_comparison.csv
```

## 12. 实施顺序

### Phase 1：PennyLane reference

1. 在 `circuits.py` 只新增 `_mera()`。
2. 注册一个 `CircuitSpec`，使用闭式 lambda 参数量。
3. 在 `build_hamiltonian()` 增加 `Z(qubits-1)`。
4. 完成 6q/8q gate-order golden tests。
5. 完成 QNode/native energy-gradient smoke tests。

验收：`circuits.py` 对 MERA 只有一个新的 `def _mera()`，没有额外 MERA helper。

### Phase 2：SAD matching correctness

1. 添加 circuit ID 5 和 compile-time dispatch。
2. 添加 `MeraLayerLayout` 与 executor。
3. 添加 matching forward/backward kernel 及 launcher。
4. 添加 `mera_hamiltonian_kernel`。
5. 注册 Python SAD runner。

验收：3q/4q/6q/8q 在所有 execution modes 下 energy/full gradient 对齐 PennyLane。

### Phase 3：optimized matching

1. 优化 Phase 2 已有的 matching forward/backward，不新增第二套顶层 kernel API。
2. 复用现有 tile/map 基础设施中确实适用的部分。
3. 保留 PennyLane、有限差分和 execution-mode 对照。
4. 完成 float32/float64 和跨 tile 测试。

验收：optimized/all-fused 与 legacy 对齐，compute-sanitizer 无非法访问或 race。

### Phase 4：性能与 dispatch

1. 运行完整 qubit sweep。
2. 分析 forward/H/backward。
3. 调优 threads/register bits。
4. 只把稳定胜出的组合写入 `_select_library()`。
5. 更新 README 和优化报告。

## 13. 完成标准

- `circuits.py` 只新增一个 MERA builder `_mera()`；
- 使用现有 `CircuitSpec` 注册方式；
- C++ 使用一个 `MeraLayerLayout` 和一个 executor 特化；
- 参数量沿用 `expected_parameter_count()` 的 circuit 特判方式；
- workspace/context/run signatures 保持不变；
- kernel 文件只有一对 matching kernels 和一对 launchers；
- 支持非 `2^k` qubit carry；
- observable 为最终 active wire 的 Z；
- float32/float64 energy 和 full gradient 对齐；
- 有限差分通过；
- 原五类电路全部测试通过；
- benchmark 使用 qubit 推导的 MERA layers；
- 新结果不覆盖已有 CSV。

## 14. Production 文件新增符号审计

下表用于最终 code review。没有列出的 production 文件不应新增 MERA 函数、类型或字段。

| 文件 | 允许新增的符号 | 对齐依据 |
|---|---|---|
| `pennylane-lightning/.../circuits.py` | 一个 `_mera()`；一个现有形式的 `register_circuit(...)` 调用；`build_hamiltonian()` 内一个分支 | RA/SU2/RZZ/QAOA/XXZ builder 和注册方式 |
| `sad/include/sad_api.h` | 一个枚举值 `SAD_CIRCUIT_MERA=5` | 现有 `SadCircuit` |
| `sad/src/runtime/circuit_dispatch.cuh` | switch 中一个 case | 现有五个 case |
| `sad/src/runtime/circuit_execution.cuh` | 一个 include；`expected_parameter_count()` 内一个 `if constexpr` 分支 | QAOA 参数量特判 |
| `sad/src/circuits/mera.cuh` | 一个 `MeraLayerLayout`；一个 `CircuitExecutor` 特化 | `xxz_hva.cuh` 的 layout + executor；`apply_layer()` 也已有对应模式 |
| `sad/src/kernels/mera.cuh` | `mera_matching_forward_kernel`、`launch_mera_matching_forward`、`mera_matching_backward_kernel`、`launch_mera_matching_backward` | `kernels/xxz.cuh` 的一对 kernel + 一对 launcher |
| `sad/src/kernels/hamiltonian.cuh` | 一个 `mera_hamiltonian_kernel` | QAOA/XXZ 各自一个 Hamiltonian kernel |
| `sad/src/runtime/runner.cuh` | `run_step()` 内一个 circuit 分支 | QAOA/XXZ Hamiltonian dispatch |
| `sad/python/sad_baseline/runner.py` | `_CIRCUITS` 一个条目；现有校验/参数量/canonical-name 代码中的 ID 5 分支 | QAOA/XXZ 当前 circuit ID 特判 |
| benchmark scripts | 现有 circuit tuple/loop 中的条目或条件分支 | 当前批量 benchmark 结构 |

特别禁止：

- 在 `circuits.py` 新增第二个 MERA helper `def`；
- 新增 `MeraTopology`、`MeraTopologyView` 或 topology registry；
- 新增 `build_mera_pair_maps()`；
- 给 `PreparedWorkspace` 或 circuit contexts 增加 MERA pair buffers；
- 新增单-block和 matching 两套重复 kernel API；
- 为了 MERA 修改其他五个 executor 的方法签名。
