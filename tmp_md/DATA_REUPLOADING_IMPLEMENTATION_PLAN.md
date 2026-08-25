# Data Re-uploading QNN 电路实现计划（严格对齐现有代码结构）

## 1. 目标与结构契约

Data Re-uploading QNN 必须复用当前 RA-HEA、SU2-HEA、RZZ-HEA、QAOA、XXZ-HVA、MERA
和 EQNN 的代码分层，不重构公共 runner、context、workspace、`CircuitSpec` 或 C API。

```text
PennyLane:   circuits.py 中一个 _data_reuploading_qnn()
Native:      native.py::_build_operations() 中一个分支
SAD C++:     一个 DataReuploadingLayerLayout
             一个 CircuitExecutor<SAD_CIRCUIT_DATA_REUPLOADING,T>
CUDA:        一对 forward/backward kernel 和一对 launcher
Hamiltonian: 现有 hamiltonian.cuh 中一个分支/kernel
Python SAD:  registry、参数量分支、canonical name 增加一个 ID
```

禁止增加第二个 builder、专用 runner、专用 context/workspace/lookups 类型、host
brickwork map、平行 legacy/optimized kernel API、`SAD_DATA_REUPLOADING_*` 宏，或
修改 `run_forward/run_backward/sad_energy_and_grad` 签名。private `__device__`
helper 只能表达 pair/parity 和局部 CZ 相位，不能成为第二个 host API。

生产入口数量固定为：

```text
circuits.py 新增顶层 builder:                    1
native.py 新增顶层 def:                          0
DataReuploadingLayerLayout:                      1
CircuitExecutor<SAD_CIRCUIT_DATA_REUPLOADING,T>: 1
专用 __global__ kernel:                            2
专用 launcher:                                    2
专用 runner/context/workspace/lookups 类型:        0
```

## 2. 固定电路语义

### 2.1 数据与参数

物理角度为：

```text
phi[layer, wire, axis] = theta[layer, wire, axis]
                          + w[wire, axis] * x[wire, axis]
```

第一版固定 `x` 和 `w`，只训练 `theta`。由于现有 `CircuitBuilder` 只有
`(params, qubits, layers)`，调用统一 API 前将 `theta+w*x` 合成为 `params`，SAD
只接收现有 `RotationCoefficients`。参数量为 `3 * qubits * layers`，第一版不训练
`w`，也不把 `dE/dw = x*dE/dphi` 的 classical chain rule 放进 adjoint kernel。

### 2.2 每层门序

```text
wire 0..n-1: RZ(phi[layer, wire, z])
wire 0..n-1: RY(phi[layer, wire, y])
wire 0..n-1: RZ(phi[layer, wire, x])
当前 brickwork parity 的全部 CZ
```

参数为 layer-major、axis-major、wire-minor，保证每个 rotation sweep 的参数连续：

```text
base = 3 * layer * qubits
RZ-z = base + wire
RY-y = base + qubits + wire
RZ-x = base + 2*qubits + wire
```

周期 brickwork 固定为：

```text
even layer: (0,1), (2,3), (4,5), ...
odd layer:  (1,2), (3,4), (5,6), ...
```

第一版只接受偶数 `qubits >= 4`。odd layer 的最后一对固定为 `(n-1,0)`，因此每层
恰好有 `n/2` 个不相交 CZ pair，每条 wire 每层参与一次 CZ。奇数 qubit 会破坏这一
周期 perfect matching，必须在 `CircuitSpec`、native、Python SAD 和 C++ executor
四处拒绝；dummy participant 或 open-boundary 扩展只能作为后续独立版本。

### 2.3 CZ 与 observable

生产路径保留逻辑 `CZ = diag(1, 1, 1, -1)`，不展开为
`H(target) -> CX(control,target) -> H(target)`；该分解只用于测试 reference。
第一版 observable 固定为 `H = Z_0`。

### 2.4 Golden topology

4-qubit、2-layer 的完整拓扑：

```text
layer 0:
    RZ-z: q0 q1 q2 q3
    RY-y: q0 q1 q2 q3
    RZ-x: q0 q1 q2 q3
    CZ: (0,1), (2,3)

layer 1:
    RZ-z: q0 q1 q2 q3
    RY-y: q0 q1 q2 q3
    RZ-x: q0 q1 q2 q3
    CZ: (1,2), (3,0)

observable: Z(0)
```

6-qubit、2-layer：

```text
layer 0 CZ: (0,1), (2,3), (4,5)
layer 1 CZ: (1,2), (3,4), (5,0)
```

8-qubit、2-layer：

```text
layer 0 CZ: (0,1), (2,3), (4,5), (6,7)
layer 1 CZ: (1,2), (3,4), (5,6), (7,0)
```

PennyLane tape、native operation list 和 CUDA analytic pair formula 都必须逐项匹配
这些 golden pairs。pair 内 wire 排序只影响记录形式，不影响 CZ 的对称语义；三端
仍必须使用同一个 canonical 顺序以便 tape 测试和浮点执行顺序稳定。

### 2.5 Gate occurrence 和复杂度

每个 macro-layer：

```text
RZ occurrence = 2*n
RY occurrence = n
CZ occurrence = n/2
trainable physical rotation occurrence = 3*n
logical trainable parameter = 3*n
```

整个电路：

```text
rotation occurrence = 3*n*L
CZ occurrence       = n*L/2
parameter_count     = 3*n*L
asymptotic gate count = O(nL)
```

与 EQNN 不同，本电路没有大量 gate occurrence 共享同一个参数；它主要考察的是
重复数据偏置、交替 brickwork topology、两段 RZ 与一段 RY 的 mixed diagonal/
non-diagonal traversal，以及 CZ diagonal fusion。

### 2.6 确定性数据准备契约

为了让 SAD、PennyLane QNode 和 Lightning native 在各自 runner 中生成完全相同的
实际 gate angles，第一版不依赖额外数据文件。固定 feature 和 weight 为：

```text
x[wire] = 2*(wire + 1)/(qubits + 1) - 1
w[z] = 1.0
w[y] = 0.5
w[x] = -0.5
offset[wire,axis] = x[wire] * w[axis]
```

现有 runner 先按 `random_seed` 生成 `theta`，然后只对 Data Re-uploading 分支在原有
参数数组上按 layer 重复加入同一 `offset[wire,axis]`：

```python
for layer in range(layers):
    for wire in range(qubits):
        feature = 2.0 * (wire + 1) / (qubits + 1) - 1.0
        base = 3 * layer * qubits
        params[base + wire] += feature
        params[base + qubits + wire] += 0.5 * feature
        params[base + 2 * qubits + wire] -= 0.5 * feature
```

这段逻辑只能内联到现有 PennyLane/native/SAD 参数准备分支，不新增公共
`prepare_data_reuploading_parameters()`。同一 wire 的数据在每一层重复出现，才满足
re-uploading 语义；若每层生成不同随机 feature，就变成另一个电路定义。

梯度数组仍解释为对 `theta` 的梯度，因为 `d phi/d theta = 1`。测试必须保存一份
未加 offset 的 `theta`，有限差分时对 `theta` 做扰动，然后重新合成 `phi`。

## 3. PennyLane 与 native

修改 `pennylane-lightning/src/pennylane_lightning_baseline/circuits.py`，只增加：

```python
def _data_reuploading_qnn(params: object, qubits: int, layers: int) -> None:
    for layer in range(layers):
        base = 3 * layer * qubits
        for wire in range(qubits):
            qml.RZ(params[base + wire], wires=wire)
        for wire in range(qubits):
            qml.RY(params[base + qubits + wire], wires=wire)
        for wire in range(qubits):
            qml.RZ(params[base + 2 * qubits + wire], wires=wire)
        parity = layer & 1
        for left in range(parity, qubits, 2):
            right = (left + 1) % qubits
            if left != right:
                qml.CZ(wires=(min(left, right), max(left, right)))
```

数据合成发生在 builder 进入前，不增加 `build_reupload_pairs()`、data encoder 或
第二个顶层 `def`。注册使用：

```python
CircuitSpec(
    name="data-reuploading",
    aliases=("data-reupload", "drqnn"),
    parameter_count_fn=lambda qubits, layers: 3 * qubits * layers,
    builder=_data_reuploading_qnn,
    requires_even_qubits=True,
    minimum_qubits=4,
)
```

在同一个 `build_hamiltonian()` 增加：

```python
if circuit_spec.name == "data-reuploading":
    return qml.Hamiltonian([1.0], [qml.PauliZ(0)])
```

修改 `native.py` 时只在 `_build_operations()` 增加一个分支，`source_parameter` 为：

```text
RZ-z: 3*layer*qubits + wire
RY-y: 3*layer*qubits + qubits + wire
RZ-x: 3*layer*qubits + 2*qubits + wire
CZ:   None
```

native 的 `_build_hamiltonian()` 只增加 `coefficients=[1.0], terms=[Z(0)]` 分支，
不得新增 native runner、operation builder 或 data encoder。

### 3.1 Builder 结构审计

`circuits.py` 的 Data Re-uploading production diff 中只允许出现一个新顶层 `def`。
参数量继续放在 `CircuitSpec.parameter_count_fn`，Hamiltonian 继续放在已有
`build_hamiltonian()`，拓扑和 gate emission 留在 `_data_reuploading_qnn()` 函数体。

`requires_even_qubits=True` 复用当前 `CircuitSpec.validate()` 的偶数校验，不新增
Data Re-uploading validator。`minimum_qubits=4` 与周期两 matching 的定义一致。

### 3.2 Native shared contract

native operation 的实际 `parameters=(float(params[source]),)` 已包含固定 data offset，
`source_parameter` 仍指向同一 index。这样 Lightning adjoint 返回的 operation-level
Jacobian 可以沿用当前归并逻辑还原为长度 `3*n*L` 的 full gradient。

CZ 必须使用 native bindings 已支持的标准 operation name；若目标 Lightning 版本只
支持 CNOT，不得静默在 SAD 生产路径也改成 `H-CX-H`，而应仅在 native reference
分支显式展开并增加等价性测试。

### 3.3 PennyLane/native 测试边界

测试需要直接检查 4q/2l tape 和 native operation list：

```text
operation count per layer = 3*n + n/2
first  n operations = RZ-z in wire order
second n operations = RY-y in wire order
third  n operations = RZ-x in wire order
last n/2 operations = CZ in golden pair order
```

还要检查两个 layer 的同一 wire/axis parameter 相差一个完整 layer stride `3*n`，
并检查同层三个 axis 的 offset 分别为 `0/n/2n`，避免误排成 wire-major 交错布局。

## 4. SAD 注册与 circuit executor

在 `sad/include/sad_api.h` 末尾追加 `SAD_CIRCUIT_DATA_REUPLOADING = 7`，已有
ID `0..6` 不得重排；`circuit_dispatch.cuh` 增加一个 case；`circuit_execution.cuh`
增加 include 和 `3*qubits*layers` 参数量特判。Python SAD registry 增加：

```python
"data-reuploading": (7, 0)
"data-reupload": (7, 0)
"drqnn": (7, 0)
```

canonical name tuple 末尾增加 `data-reuploading`，第一版使用默认
`f128r2_b128r2`，不增加 ID 7 的 measured dispatch。

新增 `sad/src/circuits/data_reuploading.cuh`，只定义：

```cpp
struct DataReuploadingLayerLayout {
    int rz_z_offset;
    int ry_offset;
    int rz_x_offset;
    int parity;
    static auto at(int layer, int qubits)
        -> DataReuploadingLayerLayout {
        const int base = 3 * layer * qubits;
        return {base, base + qubits, base + 2 * qubits, layer & 1};
    }
};

template <typename T>
struct CircuitExecutor<SAD_CIRCUIT_DATA_REUPLOADING, T> {
    static constexpr int kParametersPerQubitLayer = 3;
    // 使用现有 validate、lookup/init、forward/backward 方法集合
};
```

executor 方法集合必须与现有电路对齐：`validate`、`append_diagonal_lookups`、
`build_initial_state_lookup`、`forward_initial`、`forward_layer`、
`forward_layer_optimized`、`backward_layer`、`backward_layer_optimized`、
`backward_layer_fused`。不得新增 `apply_data_encoding()`、`build_brickwork_pairs()`
或 `forward_data_layer()` 等 host executor API。

### 4.1 C++ validation 和参数量

`CircuitExecutor<SAD_CIRCUIT_DATA_REUPLOADING,T>::validate(int qubits)` 只检查：

```text
qubits >= 4
qubits % 2 == 0
```

layers 校验放在现有 `expected_parameter_count()` 的 ID 7 特判旁，不修改公共
`validate_inputs()` 签名：

```cpp
if constexpr (Circuit == SAD_CIRCUIT_DATA_REUPLOADING) {
    return static_cast<size_t>(3) * qubits * layers;
}
```

`kParametersPerQubitLayer=3` 使公共 fallback 公式也与该特判一致，但 ID 7 仍应保留
显式分支，便于未来增加压缩参数布局时不影响其他电路。

### 4.2 Lookup 和初始化

Data Re-uploading 没有 RZZ lookup 或输入态数据 buffer，但两段 RZ 要严格复用现有
`DiagonalGate::RZ` 路径，因此 `append_diagonal_lookups()` 不是 no-op。它在每层追加
两个连续的 per-wire RZ lookup group：

```text
append_diagonal_lookup_group(parameters, layout.rz_z_offset, qubits, data)
append_diagonal_lookup_group(parameters, layout.rz_x_offset, qubits, data)
build_initial_state_lookup: |0...0>
forward_initial: 复用现有 zero-state 初始化路径
```

lookup 的 offset 仍写入已有 `offsets_by_parameter`，executor 使用
`context.diagonal_lookup_at(layout.rz_z_offset/rz_x_offset)` 取得对应表。不要把
feature `x`、weight `w` 或 CZ topology 写入 `DiagonalLookupData`。

### 4.3 Executor method mapping

| 方法 | 规定行为 |
|---|---|
| `forward_initial` | 初始化 zero state，执行 layer 0 |
| `forward_layer` | layer 0 兼容 legacy 初始化，否则执行当前 layer |
| `forward_layer_optimized` | 调用同一 forward rotation/CZ 路径 |
| `backward_layer` | reverse CZ，再 reverse RZ/RY/RZ |
| `backward_layer_optimized` | 调用同一 backward 路径 |
| `backward_layer_fused` | 第一版调用同一 backward 路径 |

runtime 已负责 layer 外层循环；executor 不得再创建第二个 layer loop。

## 5. CUDA kernel 设计

新增 `sad/src/kernels/data_reuploading.cuh`。

### 5.1 通用部分

三类 rotation 直接复用公共 kernel：

```text
RZ-z: generic diagonal/rotation forward/backward
RY-y: generic non-diagonal rotation forward/backward
RZ-x: generic diagonal/rotation forward/backward
RotationCoefficients、tile mapping、gradient reduction 继续复用
```

三个连续参数区间由 `DataReuploadingLayerLayout` 给出；固定 `x/w` 不进入 device
context。这一布局允许直接复用现有按连续 wire 参数工作的 rotation/diagonal API。

### 5.2 专用 brickwork-CZ kernel

production 入口固定为：

```text
data_reuploading_brickwork_forward_kernel
launch_data_reuploading_brickwork_forward
data_reuploading_brickwork_backward_kernel
launch_data_reuploading_brickwork_backward
```

kernel 内根据 `qubits`、`layer & 1` 和 pair index 计算 pair，不使用 host-precomputed
selected map。对每个 basis index 执行：

```text
if bit(control) == 1 and bit(target) == 1:
    amplitude *= -1
```

同一 parity 的全部 CZ 融合到一次 tile traversal；不进行 CNOT permutation、H gate
或额外 amplitude exchange。backward 同时处理 `phi` 和 `lambda`，因为 `CZ† = CZ`。

### 5.3 forward/backward 顺序

```text
forward:
    RZ-z all wires -> RY-y all wires -> RZ-x all wires
    -> fused brickwork CZ(layer & 1)

backward (reverse layers):
    fused brickwork CZ(layer & 1)
    -> inverse RZ-x -> inverse RY-y -> inverse RZ-z
```

### 5.4 公共可调参数

不增加 `SAD_DATA_REUPLOADING_*`。继续复用：

```text
SAD_FORWARD_BLOCK_THREADS, SAD_FORWARD_REGISTER_BITS
SAD_BLOCK_THREADS, SAD_REGISTER_BITS, SAD_ORDINARY_BLOCK_THREADS
```

候选范围与现有 runner 相同：forward/backward block threads 为 `64/128`，register
bits 为 `2/3/4`，ordinary 默认 `128`，默认 variant 为 `f128r2_b128r2`。
`SAD_FORWARD_FIXED_LOW_LANES`、`SAD_FIXED_LOW_LANES`、`SAD_ALTERNATE_PHASES`、
`SAD_DIAGONAL_LOOKUP_BITS` 第一版不得改变 parity 或 gate order。

### 5.5 通用 launcher 的精确调用

每层 forward 依次调用：

```text
launch_diagonal_forward<T, DiagonalGate::RZ>(
    rz_z_lookup, gate_count=qubits)

launch_non_diagonal_forward<T, NonDiagonalGate::RY>(
    parameter_offset=layout.ry_offset)

launch_diagonal_forward<T, DiagonalGate::RZ>(
    rz_x_lookup, gate_count=qubits)

launch_data_reuploading_brickwork_forward(parity=layout.parity)
```

backward 严格逆序调用：

```text
launch_data_reuploading_brickwork_backward(parity)
launch_diagonal_backward<T, DiagonalGate::RZ>(rz_x_offset)
launch_non_diagonal_backward<T, NonDiagonalGate::RY>(ry_offset)
launch_diagonal_backward<T, DiagonalGate::RZ>(rz_z_offset)
```

两个 RZ group 各自有 `qubits` 个独立参数；不能使用 shared-parameter launcher。
RY 同样是 per-wire 参数。所有 gradient 直接写入连续参数区间，不需要额外 gradient
map 或归并 buffer。

### 5.6 CZ pair 的解析公式

专用 kernel 不接收 pair array。对于 `pair_index in [0,n/2)`：

```text
even parity:
    left  = 2 * pair_index
    right = left + 1

odd parity:
    left  = 2 * pair_index + 1
    right = (left + 1) % qubits
```

这在偶数 `n>=4` 上始终生成不相交 perfect matching。kernel 应直接由 parity 和
pair index 得到 masks；不得在 host 创建 `std::vector<pair>` 或写入 workspace。

forward CZ 可以在每个 amplitude 上累计所有 pair 的布尔相位：

```text
phase_parity = 0
for pair in matching:
    phase_parity ^= bit(left) & bit(right)
if phase_parity:
    amplitude = -amplitude
```

由于同一 matching 的 CZ 全部对易，这种融合与逐 pair gate order 完全等价。PennyLane
和 native 仍按 pair index 递增发 gate，以提供确定的 reference tape。

### 5.7 backward correctness

CZ 没有 trainable parameter，不产生 gradient。backward kernel 只对当前 layer 的
`phi` 和 `lambda` 应用同一个 fused sign：

```text
phi[index]    *= sign(index, parity)
lambda[index] *= sign(index, parity)
```

这是原地安全的，因为每个 amplitude 只读写自己的 index。它与需要 permutation/
gather 的 CX 不同，不需要 state ping-pong。launcher 仍接收现有 `StatePair<T>*`，但
不得无意义地 swap `current/scratch`，否则会破坏后续通用 backward launcher 的状态指针。

RZ/RY backward 的 gradient 由现有 kernel 计算。finite difference 必须能单独发现
三段 rotation 顺序、RZ lookup offset 或 CZ parity 的错误。

### 5.8 公共宏、默认值和候选范围

| 公共宏 | 源码默认值 | 合法范围 | 第一轮候选 | 对应路径 |
|---|---:|---:|---:|---|
| `SAD_FORWARD_BLOCK_THREADS` | 继承 `SAD_BLOCK_THREADS`，有效默认 128 | 64/128/256/512 | 64/128 | RY forward、CZ forward |
| `SAD_FORWARD_REGISTER_BITS` | 继承 `SAD_REGISTER_BITS`，有效默认 2 | 2..6 且 tile bits <=12 | 2/3/4 | RY forward、CZ forward |
| `SAD_BLOCK_THREADS` | 128 | 64/128/256/512 | 64/128 | RY backward、CZ backward |
| `SAD_REGISTER_BITS` | 2 | 2..6 且 tile bits <=12 | 2/3/4 | RY backward、CZ backward |
| `SAD_ORDINARY_BLOCK_THREADS` | 128 | 64/128/256/512 | 首轮固定 128 | RZ、Hamiltonian |
| `SAD_DIAGONAL_LOOKUP_BITS` | 8 | 1..12 | 首轮固定 8 | 两个 RZ lookup group |

tile 公式与 `cuda_common.cuh` 完全一致：

```text
forward_tile_bits = 5 + forward_register_bits
                      + log2(forward_block_threads / 32)
backward_tile_bits = 5 + backward_register_bits
                       + log2(backward_block_threads / 32)
```

第一轮只编译当前已有安全 variants：

```text
f128r2_b128r2
f64r4_b64r4
f64r3_b64r4
f128r3_b64r4
f64r4_b128r3
```

这些是编译期参数，不是运行时可随意改变的 layer 参数。实际 gate angles `phi` 是
运行时参数，两者不能混淆。

### 5.9 为什么需要专用 CZ kernel

可选实现比较：

| 方案 | state pass | 额外 gate | 结论 |
|---|---:|---:|---|
| 每个 CZ 展开 `H-CX-H` | 每 pair 多次 | 2H+1CX/pair | 仅测试 reference |
| 每个 CZ 单独 diagonal launch | `n/2` 次/layer | 无 | correctness baseline 可用，性能差 |
| 一层全部 CZ fused | 1 次/layer | 无 | production 目标 |

专用 kernel 的必要性来自 topology fusion，不是新的 gate algebra。它应复用
`Complex<T>`、grid sizing、tile constants、CUDA error checking 和 execution-mode
边界，而不复制 rotation 或 Hamiltonian 基础设施。

## 6. Hamiltonian、测试与回归

在 `hamiltonian.cuh`/`runner.cuh` 增加一个 `Z_0` 分支/kernel：

```text
eigenvalue    = ((index >> 0) & 1) ? -1 : 1
lambda[index] = eigenvalue * phi[index]
energy        = real(<phi|lambda>)
```

不增加多个 observable kernel；context、workspace、lookups 不增加字段或 buffer。

### 6.1 Hamiltonian kernel 形状

```cpp
template <typename T>
__global__ void data_reuploading_hamiltonian_kernel(
    const Complex<T>* phi,
    Complex<T>* lambda,
    uint64_t state_size,
    double* energy);
```

每个 thread 处理若干 basis index，写入 `lambda` 并通过现有
`block_atomic_sum()` 规约 energy。launch 使用 `kOrdinaryBlockThreads` 和
`workspace->ordinary_grid`。

在 `runner.cuh::run_step()` 的 Hamiltonian 分支链中，ID 7 必须在默认 TFIM
`hamiltonian_kernel` 之前命中：

```cpp
else if (config.circuit == SAD_CIRCUIT_DATA_REUPLOADING) {
    data_reuploading_hamiltonian_kernel<T><<<...>>>(
        phi.current, lambda.current, workspace->state_size,
        workspace->energy.get());
}
```

不能让该电路落入默认 Hamiltonian，否则即使 forward 正确，energy 和 adjoint seed
也会错误。

### 6.2 Python SAD 参数准备

`sad/python/sad_baseline/runner.py` 在现有随机参数生成后增加 ID 7 的内联 offset
写入。PennyLane QNode 和 Lightning native runner 必须采用同一公式、dtype 和操作
顺序。offset 合成不计入 CUDA forward timing，和当前随机参数生成一样属于准备阶段。

参数返回字段保持：

```text
result.parameter_count = 3*n*L
result.circuit = "data-reuploading"
result.grad.shape = (3*n*L,)
```

`_select_library()` 第一版对 ID 7 返回默认库；明确的 `SAD_LIBRARY_PATH` 仍优先于
默认 dispatch，与其他电路一致。

### 6.3 公共文件不变清单

以下文件不能因 Data Re-uploading 增加字段、buffer 或函数参数：

```text
sad/src/circuits/context.cuh
sad/src/runtime/workspace.cuh
sad/src/runtime/lookups.cuh 的公共数据结构
sad/src/runtime/circuit_execution.cuh 的 run_forward/run_backward 签名
sad/include/sad_api.h 的 sad_energy_and_grad 参数表
```

允许在现有 `DiagonalLookupData` 中增加由参数自然生成的两组 RZ lookup 内容；这不是
新增字段，也不是 Data Re-uploading 专用 metadata。

测试修改：

```text
pennylane-lightning/tests/test_circuits.py
pennylane-lightning/tests/test_native_runner.py
sad/tests/test_sad_runner.py
```

覆盖 `4q/1l`、`4q/2l`、`6q/3l`、`8q/2l`、`12q/2l`，检查 parameter count、
`RZ->RY->RZ->CZ` 门序、even/odd parity、周期边界、PennyLane/native/SAD energy 和
full gradient，以及 float32/float64 和中心有限差分。测试内部可以用 `H-CX-H`
替换 CZ 做 correctness reference；生产路径仍使用 diagonal CZ。

### 6.4 PennyLane topology tests

直接检查 tape，不新增 production topology helper：

| qubits | layers | parameters | CZ/layer | 目的 |
|---:|---:|---:|---:|---|
| 4 | 1 | 12 | 2 | 最小周期 matching |
| 4 | 2 | 24 | 2 | even/odd golden topology |
| 6 | 2 | 36 | 3 | 更长 wrap pair `(5,0)` |
| 8 | 3 | 72 | 4 | parity 重复与 layer stride |

检查完整 operation name、wire、parameter value 和 source index。还需断言 5 qubit
被 `requires_even_qubits` 拒绝，2 qubit 被 `minimum_qubits` 拒绝。

### 6.5 Native numerical tests

对 4q/2l 和 6q/3l 比较 QNode 与 Lightning native：

```text
energy
full gradient element-wise
parameter_count
float32/float64 dtype
split times sum to total
```

native operation list test必须独立于 QNode 数值测试，防止两端共享相同参数布局错误
后仍然数值对齐。

### 6.6 SAD numerical tests

| qubits | layers | 目的 |
|---:|---:|---|
| 4 | 1 | 最小 production case |
| 4 | 2 | 两种 parity |
| 6 | 3 | 周期 wrap edge |
| 8 | 4 | 多层重复上传 |
| 12 | 2 | 跨 tile/phase |

每个关键规模比较 SAD、PennyLane 和 native。沿用项目当前容差：

```text
float32 absolute tolerance = 3e-5
float64 energy             = 1e-10
float64 gradient           = 1e-9
```

对 `legacy`、`optimized`、`all-fused`、`initial-only`、`fused-forward` 和
`phased-forward` 的既有 runtime mode 做路由检查；Data Re-uploading 的 mode wrapper
必须收敛到同一组通用 rotation launcher 和同一对 CZ launcher，不增加 mode-specific
kernel。

### 6.7 有限差分与 sanitizer

中心有限差分必须直接扰动未编码的 `theta`，再重新加入固定 offset。至少覆盖：

```text
layer 0: q0 RZ-z, q0 RY-y, q0 RZ-x
layer 1: wrap edge (n-1,0) 两端的 rotation 参数
最后一层最后一个参数
```

同时运行：

```text
compute-sanitizer --tool memcheck
compute-sanitizer --tool racecheck
nvcc --ptxas-options=-v
```

检查最后一个 tile、CZ 原地 sign、phi/lambda 同步、RZ lookup offset、float32/float64
alignment 和 spill。

### 6.8 原有回归

完整运行 RA-HEA、SU2-HEA、RZZ-HEA、QAOA、XXZ-HVA、MERA 和 EQNN 测试，确认：

```text
已有 circuit ID 和 canonical name 不变
已有参数量和 validation 不变
workspace/context/run signatures 不变
已有 variant dispatch 不变
现有 benchmark CSV schema 不变
```

## 7. Benchmark 对齐

只修改现有三个 benchmark 的 `CIRCUITS` 配置：

```text
benchmark/benchmark_sad.py
benchmark/benchmark_pennylane_lightning.py
benchmark/benchmark_lightning_native.py
```

继续复用已有 `LAYERS`、`QUBITS`、`steps_for_qubits()`、`layers_for_circuit()` 和
公共计时循环，不新增专用 runner。记录 forward/Hamiltonian/backward time、energy、
full gradient、`3*n*L` logical parameter count、CZ pair count、kernel variant 和
workspace bytes。

### 7.1 layers、输入和输出

Data Re-uploading 不像 MERA 那样由 qubits 推导 layers；`layers` 是用户指定的
benchmark depth。因此已有 benchmark 的 `LAYERS` 和 `layers_for_circuit()` 可以直接
复用。每个 row 必须记录 `layers` 和 `parameter_count=3*n*L`，不能把每次 data
re-upload 当成新的 circuit 或新增 layer loop。

建议第一轮配置：

```text
CIRCUITS = ("data-reuploading",)
QUBITS   = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28)
LAYERS   = 1, 2, 4, 8  # 分别修改同一个配置执行多轮
precision = float64 主 sweep，float32 关键点
random_seed = 42
```

第一轮仍使用当前 benchmark 的单一 `LAYERS` 常量，不为该电路新增第二个 layer
循环；参数 offset 生成也必须使用同一个 `qubits/layers` 配置。

### 7.2 结果文件和指标

为避免覆盖现有结果，推荐使用：

```text
benchmark/results/data_reuploading_sad_gpu.csv
benchmark/results/data_reuploading_pennylane_gpu.csv
benchmark/results/data_reuploading_native_gpu.csv
benchmark/results/data_reuploading_comparison.csv
```

至少记录：

```text
forward_mean/median
hamiltonian_mean/median
backward_mean/median
total_mean/median
energy/full gradient error
parameter_count
rotation occurrence count
CZ count
kernel variant
workspace bytes
```

如果当前 CSV schema 没有 RZ-z/RY/RZ-x/CZ 四个子阶段字段，第一版只保留已有
forward/Hamiltonian/backward 字段，不为该电路新增 benchmark 函数；子阶段拆分属于
后续 profiling 扩展。

## 8. 分阶段实施与结构审计

### Phase A：reference 和注册

1. 在 `circuits.py` 只新增 `_data_reuploading_qnn()`。
2. 注册 `CircuitSpec`，设置 `requires_even_qubits=True`、`minimum_qubits=4`。
3. 在三个现有参数准备路径中内联同一个 `theta+w*x` offset 公式。
4. 在 native operations/Hamiltonian 中增加同层级分支。
5. 增加 C/Python ID 7 和 `3*n*L` 参数量。
6. 固定 4/6/8-qubit golden topology 和参数布局测试。

验收：PennyLane/native energy/full gradient 对齐，奇数或小于 4 qubit 被一致拒绝。

### Phase B：SAD correctness

1. 添加 circuit ID 7 和 compile-time dispatch。
2. 添加 `DataReuploadingLayerLayout` 与唯一 executor 特化。
3. 为每层两个 RZ group 添加现有 diagonal lookup group。
4. 复用 generic RZ/RY/RZ forward/backward launcher。
5. 增加一对 fused brickwork-CZ kernel/launcher。
6. 增加 `Z_0` Hamiltonian kernel 和 runner 分支。
7. 增加 Python SAD registry、canonical name 和参数 offset 准备。

验收：4/6/8/12q、两种 precision、所有已有 execution mode、有限差分通过。

### Phase C：tile matching 与融合

1. 将同一 parity 的 CZ pair 融合到一次 tile traversal。
2. 复用 diagonal mailbox/tile mapping，不创建 CZ map buffer。
3. 评估将 RZ group 与相邻 diagonal traversal 融合的收益；不得改变 RZ/RY/RZ 顺序。
4. 保持 backward 严格逆序，并验证 CZ 对 `phi/lambda` 的 sign 同步。

Phase C 只替换 Phase B 同名 kernel/launcher 的内部实现，不保留 legacy kernel，不改
executor/context/run signatures。

### Phase D：variant sweep

1. 编译现有公共 variants。
2. 扫描 qubits、layers、even/odd parity 和 float32/float64。
3. 至少重复两轮，排除 GPU warmup 和数据准备噪声。
4. 只有某个公共 variant 在连续 qubit 区间稳定胜出且 correctness 通过，才为 ID 7
   增加 measured dispatch。

### 8.1 主文件结构审计

| 文件/层级 | 现有基准结构 | Data Re-uploading 允许新增 | 强制边界 |
|---|---|---|---|
| `circuits.py` | 每种电路一个 builder | 一个 `_data_reuploading_qnn()`、一个注册调用、一个 Hamiltonian 分支 | 不增加第二个 def |
| `native.py` | `_build_operations()` 和 `_build_hamiltonian()` 内部分支 | 各一个 ID 7 分支 | 不新增 native helper/runner |
| `sad_api.h` | circuit enum 末尾追加 | `SAD_CIRCUIT_DATA_REUPLOADING=7` | 不重排 0..6 |
| `circuit_dispatch.cuh` | 一个 ID 对应一个 case | 一个 ID 7 case | runtime-template 边界不变 |
| `circuit_execution.cuh` | include + 参数量特判 | 一个 include、一个 `3*n*L` 特判 | run signatures 不变 |
| `circuits/data_reuploading.cuh` | 一个 layout + executor 特化 | `DataReuploadingLayerLayout` + 一个 executor | 不增加第二特化 |
| `kernels/data_reuploading.cuh` | private helper + 一对 kernel/launcher | CZ matching forward/backward 一对 | 不保留平行 API |
| `hamiltonian.cuh` | 每种专用 observable 一个 kernel | 一个 `Z_0` kernel | 不增加 observable variants |
| `runner.cuh` | Hamiltonian 分支链 | 一个 ID 7 分支 | 不改公共调用签名 |
| `context/workspace/lookups` | 公共共享结构 | 不增加字段；只使用已有 RZ lookup storage | 不加 data/map buffer |
| Python SAD runner | registry、参数量、canonical name | ID 7 条目和现有分支 | 不新增 runner |
| 三个 benchmark | 公共 circuit tuple/loop | 增加配置项和现有层选择分支 | 不新增专用计时函数 |

### 8.2 Production 符号审计

实现完成后用 `rg` 或 clang/nvcc AST 做以下精确检查：

```text
[ ] circuits.py 因该电路新增的顶层 def 恰好 1 个
[ ] native.py 因该电路新增的顶层 def 恰好 0 个
[ ] DataReuploadingLayerLayout 恰好 1 个
[ ] CircuitExecutor<SAD_CIRCUIT_DATA_REUPLOADING,T> 恰好 1 个
[ ] brickwork forward __global__ 恰好 1 个
[ ] brickwork backward __global__ 恰好 1 个
[ ] brickwork forward launcher 恰好 1 个
[ ] brickwork backward launcher 恰好 1 个
[ ] Data Re-uploading Hamiltonian kernel 恰好 1 个
[ ] Data Re-uploading 专用 host topology/map builder 恰好 0 个
[ ] SAD_DATA_REUPLOADING_* 公共宏恰好 0 个
[ ] context/workspace/lookups 无 Data Re-uploading 字段
```

### 8.3 不变量审计

```text
[ ] qubits >= 4 且 qubits 为偶数
[ ] 每层恰好 3*n 个 rotation occurrence 和 n/2 个 CZ occurrence
[ ] 每层每个 wire 恰好参与一次 CZ
[ ] layer parity 只由 layer & 1 决定
[ ] 参数 offset 为 base/base+n/base+2n
[ ] 两个 RZ lookup group 的 offset 连续且不重叠
[ ] forward: RZ-z -> RY-y -> RZ-x -> CZ
[ ] backward: CZ -> inverse RZ-x -> inverse RY-y -> inverse RZ-z
[ ] CZ 不写 gradient
[ ] Hamiltonian 为 Z_0，而不是 X_0 或默认 TFIM
[ ] 数据 offset 在三端完全一致且每层重复
```

### 8.4 当前七个已实现电路的实际结构基准

以下表格按当前仓库实际源码审计，而不是把“一个电路”误解成“文件中只能出现
一个 C++ 方法”：

| 层级 | 当前七个电路的实际结构 | Data Re-uploading 对齐要求 | 明确禁止 |
|---|---|---|---|
| PennyLane builder | `circuits.py` 中 RA、SU2、RZZ、QAOA、XXZ、MERA、EQNN 各一个 builder；`_ring_cnot()` 是已有公共 helper | 只新增一个 `_data_reuploading_qnn()` | 不新增 `_data_reuploading_pairs()`、`_encode_data()` 等顶层 def |
| PennyLane registry | 每个电路一个 `register_circuit(CircuitSpec(...))` | 增加一个 Data Re-uploading `CircuitSpec` | 不修改 `CircuitSpec` 或注册机制 |
| Hamiltonian reference | 一个已有 `build_hamiltonian()`，按 name 分支 | 在同一个函数中增加 `Z_0` 分支 | 不新增 `build_data_reuploading_hamiltonian()` |
| Lightning native | 一个 `_build_operations()` 和一个 `_build_hamiltonian()`，内部按 circuit name 分支 | 各增加一个 ID 7 分支 | 不新增 `_build_data_reuploading_operations()`、runner 或 encoder |
| C++ circuit header | 每个专用电路一个 LayerLayout 和一个 `CircuitExecutor` 特化 | 一个 `DataReuploadingLayerLayout` + 一个 executor 特化 | 不增加 optimized executor 第二特化 |
| C++ executor 方法 | 公共接口方法之外，MERA/XXZ 已有 `apply_layer`，QAOA 已有 `apply_cost/apply_mixer`，SU2 有既有 `forward_layer_phased` 特例；这些是 executor-local static 方法，不是第二 production API | 可将同层 rotation/CZ 调用内联到既有接口；若确需拆分，只能使用同一 executor 内 private static 方法，并在数量审计中标注 | 不新增外层 runner、第二 executor、topology registry 或改变公共方法签名 |
| CUDA | RA/SU2/RZZ 主要复用公共 rotation/diagonal；XXZ/MERA 使用专用 matching kernel；EQNN 有一对专用 kernel | 复用 RZ/RY generic kernel，新增一对 fused brickwork-CZ kernel/launcher | 不增加 legacy/optimized 两套 Data Re-uploading kernel |
| Hamiltonian CUDA | `hamiltonian.cuh` 中已有多个电路专用 observable kernel，由 `runner.cuh` 分支选择 | 增加一个 `Z_0` kernel 和一个 ID 7 分支 | 不新增多个 Data Re-uploading observable kernel |
| Python SAD runner | `_CIRCUITS`、参数量特判、canonical tuple 共用现有函数 | 增加一个 ID 7 条目和现有分支 | 不新增专用 runner |

因此，本计划中的“一个函数/一个 kernel”指新增的 production 入口层级：

```text
circuits.py: 一个新增顶层 builder
native.py:   新增顶层 def 为 0，只增加既有函数分支
C++:         一个 layout + 一个 executor 特化；接口方法和 executor-local helper 不计为第二电路
CUDA:        一对 Data Re-uploading 专用 matching kernel/launcher
```

`DataReuploadingLayerLayout::at()` 属于 layout 的既有模式；`CircuitExecutor` 内的
`forward_initial`、`forward_layer`、`backward_layer` 等属于公共 executor interface。
它们不能被误报为多个 Data Re-uploading 电路函数，也不能借此新增另一条执行路径。

### 8.5 结构差异的最终处理原则

```text
允许：一个 executor 内部的 apply_layer-style private static 拆分
允许：kernel 文件中的 private __device__ pair/parity helper
允许：hamiltonian.cuh 中每个电路一个 observable kernel

不允许：第二个 CircuitExecutor 特化
不允许：第二个 PennyLane builder
不允许：native.py 新增电路专用顶层 def
不允许：Data Re-uploading legacy/optimized 两套 kernel
不允许：host topology map、workspace map 或专用 runner
```

code review 必须同时检查“新增符号数量”和“实际调用路径数量”：即使符号名称只有
一个，也不能通过 private helper 再保留一条未使用或 legacy 执行路径。

最终 checklist：

```text
[ ] PennyLane builder 恰好 1 个；native.py 新增顶层 def 恰好 0 个
[ ] parameter_count = 3*qubits*layers
[ ] 第一版拒绝奇数 qubit 和 qubits < 4
[ ] 三端门序均为 RZ -> RY -> RZ -> brickwork CZ
[ ] even/odd parity 和周期边界完全一致
[ ] CZ 生产路径未展开成 H-CX-H
[ ] LayerLayout 恰好 1 个；CircuitExecutor 特化恰好 1 个
[ ] 专用 forward/backward __global__ kernel 各 1 个，launcher 各 1 个
[ ] Hamiltonian 使用 Z_0 且专用 kernel 恰好 1 个
[ ] context/workspace/lookups 无新增字段或 buffer
[ ] 未增加 SAD_DATA_REUPLOADING_* 公共宏
[ ] energy/full gradient/finite difference/sanitizer 通过
[ ] 原有七种电路完整回归通过
```

任何一项不满足，都不能认为 Data Re-uploading 已与当前项目结构完全对齐。
