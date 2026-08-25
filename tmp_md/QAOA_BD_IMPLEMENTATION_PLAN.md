# QAOA-BD 电路实现计划（按现有代码结构对齐）

## 1. 目标与对齐原则

新增 `qaoa-bd`（basic decomposition）作为独立电路，要求 QAOA cost 层的
每个 RZZ 显式展开为基础门：

```text
RZZ(theta, q0, q1)
    == CNOT(q0, q1) -> RZ(theta, q1) -> CNOT(q0, q1)
```

这不是只为 PennyLane tape 增加一个名称。当前 `qaoa` 的 SAD 路径直接调用
ring-RZZ diagonal、compact lookup 和 cost-RX fused kernel，不会执行 CNOT，
因此 `qaoa-bd` 必须拥有独立的电路 ID、gate builder 和 SAD executor。旧的
`qaoa`、`qaoa-ns` 语义和性能路径保持不变。

实现遵循仓库现有模式：

| 层次 | 当前模式 | QAOA-BD 对齐方式 |
|---|---|---|
| PennyLane builder | `circuits.py` 中每类一个 `_name()` | 只新增 `_qaoa_bd()`，不重构 `CircuitSpec` |
| Python 注册 | 一个 `register_circuit(CircuitSpec(...))` | 新增 canonical `qaoa-bd` 与下划线 alias |
| C++ executor | 一个 circuit header + 一个 `CircuitExecutor` 特化 | 新增 `qaoa_bd.cuh` 和 BD executor |
| C++ dispatch | `SadCircuit` + `visit_circuit()` | 追加新 ID，不移动已有 ID |
| 参数量 | QAOA 在 `expected_parameter_count()` 特判 | BD 同样返回 `2 * layers` |
| Hamiltonian | `runner.cuh` 按 circuit ID 分派 | BD 复用 QAOA MaxCut cost kernel |
| CUDA 状态操作 | 基础门使用独立 launcher，ring CNOT 是整环操作 | 新增指定 `(control,target)` 的 pair-CNOT kernel |
| benchmark | 既有 circuit 名称和结果表 | BD 使用独立名称和 CSV，不污染 qaoa 结果 |

明确不做：

- 不修改已有 QAOA 的 RZZ/fused kernel；
- 不把 `qaoa-bd` 映射到 `SAD_CIRCUIT_QAOA`；
- 不把 `ring_cnot.cuh::launch_cnot()` 当作单边 CNOT 使用；
- 不为每个分解 RZ 创建新的可训练参数；
- 不修改 `PreparedWorkspace`、`ForwardCircuitContext`、`BackwardCircuitContext`
  的公共字段和 `run_forward/run_backward` 签名；
- 不在生产代码中依赖 PennyLane 的隐式 RZZ decomposition；
- 不在第一版加入 cost-RX、CNOT-RZ 或跨层融合，先保证门序列可审计。

## 2. 固定电路定义

### 2.1 初态、层结构和边顺序

对偶数 `n >= 4`、`L >= 1`，初态为 `|+>^n`。现有 PennyLane 使用 Hadamard，
cuQuantum reference 使用等价的 `RY(pi/2)` 生成 `|+>`。每层参数为一个共享
`beta_l` 和一个共享 `gamma_l`：

```text
H(0) ... H(n-1)

for l = 0 ... L-1:
    # even matching
    for left = 0, 2, ..., n-2:
        q0 = left
        q1 = (left + 1) mod n
        CNOT(q0, q1)
        RZ(gamma_l, q1)
        CNOT(q0, q1)

    # odd matching, including wrap-around edge (n-1, 0)
    for left = 1, 3, ..., n-1:
        q0 = left
        q1 = (left + 1) mod n
        CNOT(q0, q1)
        RZ(gamma_l, q1)
        CNOT(q0, q1)

    for wire = 0 ... n-1:
        RX(beta_l, wire)
```

偶匹配与奇匹配的顺序、edge 的 control/target 方向必须固定。特别是最后一
条边必须是 `CNOT(n-1, 0)`，不能因为 `0 < n-1` 而改成反向 CNOT。

同层 cost 边在数学上对易，但 BD 实验需要固定门序，便于逐门比较、反向遍历
和 kernel 计时；不能将所有边任意排序或打包成一次 ring permutation。

### 2.2 参数布局

完全保持共享角 QAOA 的参数 API：

```text
params[2*l]     = beta_l
params[2*l + 1] = gamma_l
parameter_count = 2 * layers
```

CNOT 没有 parameter。每层的 `n` 个 RZ 都引用同一个 `params[2*l+1]`，
不能把一个 gamma 展开成 `n` 个独立参数。梯度输出仍然只有
`[dE/dbeta_0, dE/dgamma_0, ..., dE/dbeta_(L-1), dE/dgamma_(L-1)]`。

门数量（不含初态 H/RY）为：

| 每层门类 | 数量 |
|---|---:|
| CNOT | `2n` |
| RZ | `n` |
| RX | `n` |
| 总数 | `4n` |

### 2.3 RZZ 角度 convention

项目中的 `qml.IsingZZ(theta)` 和 SAD RZZ 约定为：

```text
exp(-i * theta/2 * Z_q0 Z_q1)
```

由于 CNOT 将 `Z` 从 target 共轭为 `Z_control Z_target`，有：

```text
CNOT(q0,q1) RZ(theta,q1) CNOT(q0,q1)
  = exp(-i * theta/2 * Z_q0 Z_q1)
```

因此分解中的 RZ 角度是 `theta`，不是 `theta/2` 或 `2*theta`。第一步必须
用 4x4 矩阵测试锁定此约定，再实现 CUDA kernel。

## 3. PennyLane reference 实现

修改文件：

```text
pennylane-lightning/src/pennylane_lightning_baseline/circuits.py
```

### 3.1 builder

只增加一个 `_qaoa_bd()`；helper 可以是局部逻辑或一个内部 append helper，
但不得调用 `qml.IsingZZ`：

```python
def _qaoa_bd(params: object, qubits: int, layers: int) -> None:
    for wire in range(qubits):
        qml.Hadamard(wires=wire)
    for layer in range(layers):
        beta = params[2 * layer]
        gamma = params[2 * layer + 1]
        for left in range(0, qubits, 2):
            right = (left + 1) % qubits
            qml.CNOT(wires=(left, right))
            qml.RZ(gamma, wires=right)
            qml.CNOT(wires=(left, right))
        for left in range(1, qubits, 2):
            right = (left + 1) % qubits
            qml.CNOT(wires=(left, right))
            qml.RZ(gamma, wires=right)
            qml.CNOT(wires=(left, right))
        for wire in range(qubits):
            qml.RX(beta, wires=wire)
```

builder 只负责追加门，不重复承担 qubit/layer 校验；这些校验继续由
`CircuitSpec.validate()` 完成。

### 3.2 注册与 Hamiltonian

增加一个现有形式的注册项：

```python
register_circuit(
    CircuitSpec(
        name="qaoa-bd",
        aliases=("qaoa_bd",),
        parameter_count_fn=lambda qubits, layers: 2 * layers,
        builder=_qaoa_bd,
        requires_even_qubits=True,
        minimum_qubits=4,
    )
)
```

在 `build_hamiltonian()` 中将 `qaoa-bd` 与 `qaoa`、`qaoa-ns` 放进同一个
MaxCut 分支：

```python
if circuit_spec.name in {"qaoa", "qaoa-bd", "qaoa-ns"}:
    # 0.5 * (Z_i Z_(i+1) - I)
```

门分解不能改变 observable 或 energy 定义。

### 3.3 PennyLane 验证

修改现有 circuit/runner/native 测试的参数化列表，新增：

- `available_circuits()` 含 canonical `qaoa-bd`；
- `qaoa_bd` 与 `qaoa-bd` 指向同一个 `CircuitSpec`；
- 4q/1 layer 的参数量是 2，8q/3 layer 的参数量是 6；
- tape 中不存在 `IsingZZ`，cost 每条 edge 恰为 `CNOT,RZ,CNOT`；
- RZ 的 wire、两个 CNOT 的方向和偶/奇 edge 顺序正确；
- tape 只包含 `Hadamard/CNOT/RZ/RX`；
- `build_hamiltonian(…, "qaoa-bd")` 与 `qaoa` 的系数和 observable 完全相同；
- energy 与 full gradient 在小规模随机参数下对齐 qaoa。

## 4. cuQuantum reference runner

修改：

```text
cuQuantum/python/sad_cuquantum/runner.py
cuQuantum/include/cuquantum_api.h
cuQuantum/src/cuquantum_cuda.cu
```

### 4.1 circuit registry 和参数量

在 `supported_circuits`、`_NATIVE_CIRCUIT`、`expected_parameter_count()`
增加 canonical name 和新 ID。旧 ID `0..8` 不移动，BD 使用下一个未占用值；
公共 `sad_api.h`、cuQuantum native circuit 值和 C++ Hamiltonian 分支必须一致。

`expected_parameter_count("qaoa-bd", q, L)` 返回 `2 * L`，并沿用 QAOA 的
偶数、最小 4 qubit 约束。

### 4.2 gate list

`build_gates()` 的 BD 分支必须生成：

```text
fixed RY(pi/2) per wire       # H equivalent for |0> input
for each layer:
    CNOT(left,right)
    RZ(parameter=gamma,right)
    CNOT(left,right)
    RX(parameter=beta,wire)   # after all cost edges
```

`Gate.kind` 只能出现 `ry/rz/rx/cnot`；BD gate list 中禁止 `rzz`。native gate
中的 CNOT parameter 设为 `-1`。cuStateVec 已支持任意 wire pair 的 CNOT，
不需要把它伪装成 ring CNOT。

### 4.3 Hamiltonian 与 native path

`_hamiltonian()` 将 BD 纳入 QAOA MaxCut 分支。C++ native
`hamiltonian_kernel` 使用 circuit ID 判断时也必须将 BD ID 纳入同一分支；否则
会错误落到默认 TFIM Hamiltonian。native backend 的 CNOT 实现继续使用
`custatevecApplyMatrix` 的 control/target 参数，不调用 RZZ Pauli rotation。

## 5. SAD 公共注册和 runtime dispatch

### 5.1 Circuit ID

修改 `sad/include/sad_api.h`：

```cpp
SAD_CIRCUIT_QAOA_BD = 9,  // 取当前未占用的下一个 ID
```

实际提交前以头文件当前最大值为准；绝对不能重排已有 `0..8`。如果仓库其
他 ABI 消费者固定 enum 值，应在同一 patch 中更新其映射表。

### 5.2 compile-time dispatch

修改 `sad/src/runtime/circuit_dispatch.cuh`，增加一个 `case`，将新 ID 转换
为 `std::integral_constant<int, SAD_CIRCUIT_QAOA_BD>`。修改
`sad/src/runtime/circuit_execution.cuh`：

- include `../circuits/qaoa_bd.cuh`；
- 在 `expected_parameter_count()` 增加 BD 分支，返回 `2 * layers`；
- 不改动其他电路的参数量表达式和 context 构造。

`validate_inputs()` 通过 `visit_circuit()` 调用 BD executor 的
`validate(qubits)`；BD 要求 `qubits >= 4 && even`。

### 5.3 Hamiltonian 分派

修改 `sad/src/runtime/runner.cuh::run_step()` 的 QAOA 条件：

```cpp
config.circuit == SAD_CIRCUIT_QAOA ||
config.circuit == SAD_CIRCUIT_QAOA_NS ||
config.circuit == SAD_CIRCUIT_QAOA_BD
```

BD 与 QAOA 使用相同的 `qaoa_cost_hamiltonian_kernel`。这里不能因为 BD 的
cost 门已分解，就重新把 Hamiltonian 写成一串 observable expectation。

## 6. SAD 电路 executor

新增：

```text
sad/src/circuits/qaoa_bd.cuh
```

### 6.1 layout

定义与 `QaoaLayerLayout` 对齐的 layout：

```cpp
struct QaoaBdLayerLayout {
    int beta;
    int gamma;
    static auto at(int layer) -> QaoaBdLayerLayout {
        return {2 * layer, 2 * layer + 1};
    }
};
```

不新增 edge array；ring edge 由 `left` 和 `qubits` 直接计算，避免 workspace
增加 topology metadata。

### 6.2 executor 方法契约

```cpp
template <typename T>
struct CircuitExecutor<SAD_CIRCUIT_QAOA_BD, T> {
    static constexpr int kParametersPerQubitLayer = 0;
    static void validate(int qubits);
    static void append_diagonal_lookups(...);
    static auto build_initial_state_lookup(...);
    static void forward_initial(...);
    static void forward_layer(...);
    static void forward_layer_optimized(...);
    static void backward_layer(...);
    static void backward_layer_optimized(...);
    static void backward_layer_fused(...);
};
```

行为必须是：

| 方法 | 行为 |
|---|---|
| `forward_initial` | 生成 `|+>^n`，执行 layer 0 的显式 BD cost 与 RX |
| `forward_layer` | 逐 edge 执行 pair-CNOT、RZ、pair-CNOT，再执行 RX |
| `forward_layer_optimized` | 第一版直接调用未融合 `forward_layer` |
| `backward_layer` | 先 RX backward，再按 edge 逆序执行 CNOT、RZ backward、CNOT |
| `backward_layer_optimized` | 第一版直接调用 `backward_layer` |
| `backward_layer_fused` | 第一版不得调用旧 QAOA fused backward，直接 split |

现有 `circuit_execution.cuh` 已负责 forward layer 顺序和 backward 反向 layer
顺序，BD executor 不应再自行倒置 layer loop。

### 6.3 初态

可以复用 `initialise_plus_state_kernel`，因为它只生成均匀实数振幅，不含 cost。
禁止使用 `initialise_plus_cost_state_kernel`；那个 kernel 已隐式施加 RZZ，
会让 gate-level BD 语义失真。`build_initial_state_lookup()` 只返回现有
zero/constant lookup 约定所需的数据，不为 BD 增加 cost lookup。

## 7. pair-CNOT CUDA kernel

### 7.1 为什么不能复用 ring CNOT

`sad/src/kernels/ring_cnot.cuh::launch_cnot()` 的含义是一次性作用整个 ring
的固定 CNOT permutation。若在每一条 RZZ 分解中调用它，会额外执行其他 `n-1`
条边的 CNOT，电路不再等价于 RZZ。

因此新增：

```text
sad/src/kernels/pair_cnot.cuh
```

建议最小 API：

```cpp
template <typename T>
void launch_pair_cnot(StatePair<T>* phi,
                      StatePair<T>* lambda,
                      uint64_t state_size,
                      int control,
                      int target,
                      bool adjoint,
                      int grid_size);
```

### 7.2 kernel 语义

对每个 basis index：

```text
if bit(control) == 1:
    output[index xor (1 << target)] = input[index]
else:
    output[index] = input[index]
```

forward 可以采用 scatter 或 gather；backward 的 CNOT 是自反，但仍需按
`adjoint` 选择正确的 input/output permutation。`phi` 和非空 `lambda` 都要
使用同一 permutation，并在 launcher 完成 ping-pong buffer swap。不能在同一
state buffer 上原地写，避免 self-overwrite 和 race。

第一版优先使用 correctness-first 的普通 permutation kernel；后续再基于
profile 做 tile/register 优化。pair-CNOT 的 control/target 参数应在 launch
时传入，不能硬编码成 ring 全边。

### 7.3 RZ 路径

RZ 可以复用 `launch_diagonal_forward/backward<DiagonalGate::RZ>`，但 lookup
必须按 gamma 参数建立。由于同一个 gamma 在一层被引用 `n` 次：

- parameter offsets 仍只有一个 `gamma` slot；
- 每次 RZ launcher 读取相同 parameter lookup；
- 不建立 RZZ lookup，也不调用 `shared_ring_rzz_factor()`；
- 不调用 `launch_shared_ring_rzz_forward/backward()`。

如果当前普通 diagonal lookup API 只支持“每参数一次”的 group，需扩展重复
引用机制或在 BD executor 中使用参数索引直接生成 RZ 系数，但不能复制 gamma
到新的梯度槽位。

## 8. forward/backward 详细顺序

### 8.1 forward

对每层 `l`：

```text
for left = 0,2,...:
    pair_cnot_forward(left, right)
    rz_forward(gamma_l, right)
    pair_cnot_forward(left, right)
for left = 1,3,...:
    pair_cnot_forward(left, right)
    rz_forward(gamma_l, right)
    pair_cnot_forward(left, right)
rx_layer_forward(beta_l)
```

不能将不同 edge 的 CNOT 合并为一个 permutation，也不能把所有 RZ 合并为
旧的 ring-RZZ diagonal kernel；每条 edge 都是可观察的 BD 实验门。

### 8.2 backward

在 inverse-walk 中，对某层 forward 序列的逆序为：

```text
rx_layer_backward(beta_l)
for left = n-1,n-3,...,1:
    pair_cnot_backward(left, right)
    rz_backward(gamma_l, right)
    pair_cnot_backward(left, right)
for left = n-2,n-4,...,0:
    pair_cnot_backward(left, right)
    rz_backward(gamma_l, right)
    pair_cnot_backward(left, right)
```

每个 RZ backward 必须把对当前 gamma 的贡献累加到同一个 `gradients[gamma]`。
CNOT 没有梯度，但其对 `phi`/`lambda` 的 permutation 不能跳过。注意这里的
edge 顺序是“奇匹配逆序后偶匹配逆序”，而不是重新从左到右执行。

## 9. workspace、context 和编译形状

### 9.1 公共 context 保持不变

BD 的 topology 只由 `qubits`、layer 和 edge index 计算，因此不增加：

```text
ForwardCircuitContext 字段
BackwardCircuitContext 字段
PreparedWorkspace 字段
run_forward/run_backward 参数
```

复用的字段包括：`state_size`、`rotation_coefficients`、`gradients`、
`ordinary_grid`、`multiprocessors`、`phi` 和 `lambda`。

### 9.2 lookup 和 workspace 分配

workspace 初始化必须满足：

- `parameter_count = 2 * layers`；
- BD 不申请 RZZ compact lookup；
- 若复用普通 RZ diagonal lookup，则为每个 gamma 建立一个 group；
- offsets 数组仍按完整 parameter count 分配；
- CNOT 不申请 lookup，只使用 pair wire 和 state ping-pong buffer。

如果第一版 pair-CNOT launcher 不需要 selected maps，则 BD 不应伪造
`xxz_even_selected_maps` 或 `target_masks`。已有 context 中未使用的字段保持
nullptr/空 vector，避免为新电路扩大公共 workspace。

### 9.3 公共编译宏

第一版 pair-CNOT 和 split RZ 直接使用公共安全形状，不增加 `SAD_QAOA_BD_*`
宏。可复用：

| 宏/常量 | 用途 |
|---|---|
| `SAD_FORWARD_BLOCK_THREADS` | pair-CNOT/RZ forward block shape |
| `SAD_FORWARD_REGISTER_BITS` | forward tile/register shape（若 kernel 使用） |
| `SAD_BLOCK_THREADS` | backward block shape |
| `SAD_REGISTER_BITS` | backward tile/register shape |
| `SAD_ORDINARY_BLOCK_THREADS` | MaxCut Hamiltonian kernel |

第一版不使用 `SAD_DIAGONAL_LOOKUP_BITS` 以外的 QAOA compact lookup 宏，也不
使用 `SAD_QAOA_FUSE_COST_RX`、`SAD_QAOA_INITIAL_STRATEGY` 或
`SAD_QAOA_FUSED_BACKWARD`。这些宏控制的是旧专用 QAOA 路径，接入 BD 会违背
门级分解目标。

首轮 variant 只使用默认 `f128r2_b128r2` 安全库；完成正确性和完整 sweep
后，才评估 pair-CNOT 是否需要新增 variant。不能直接把旧 QAOA 的 winner
表复制给 BD。

## 10. Python SAD runner 与 native metadata

修改：

```text
sad/python/sad_baseline/runner.py
```

在 `_CIRCUITS` 增加：

```python
"qaoa-bd": (9, 0),
"qaoa_bd": (9, 0),  # 若 runner 直接保留 alias，则统一 normalise 后注册
```

实际代码应遵循现有 `_normalise_name()`，推荐只保留 canonical/alias 的统一
映射，避免 `_CIRCUITS` 出现重复逻辑。ID 的第二个参数对 BD 不使用；
`expected_parameter_count` 通过 ID 特判返回 `2 * layers`，并检查偶数和最小
qubit。

`_select_library()` 第一版对 BD 使用默认安全 library，不将其误判为旧 qaoa
的策略 winner。结果中的 `circuit`、`kernel_variant`、phase plan 必须保留
`qaoa-bd` 名称，便于独立汇总。

## 11. 测试计划

### 11.1 纯矩阵和 gate-list 测试

先在 Python 中用随机 theta 比较：

```text
qml.matrix(qml.IsingZZ(theta))
vs
matrix(CNOT) @ matrix(RZ(theta)) @ matrix(CNOT)
```

检查最大误差。随后对 4q/6q、1/2 layer 检查：

- 不出现 `IsingZZ`/`rzz`；
- 每条 edge 恰为 CNOT-RZ-CNOT；
- RZ target 与两个 CNOT target 一致；
- 偶 edge 在奇 edge 之前；
- wrap-around edge 为 `(n-1,0)` 方向；
- 每层只有一个 beta parameter 和一个 gamma parameter；
- 门数为 `4n`（不含初态）。

### 11.2 跨 backend 数值等价

对 qubits `4, 6, 8`、layers `1, 2`、固定随机 seed：

- PennyLane `qaoa` vs `qaoa-bd` state/energy/full gradient；
- cuQuantum `qaoa-bd` vs PennyLane BD；
- SAD `qaoa-bd` vs PennyLane BD；
- SAD BD vs SAD 原始 QAOA。

最后一项只比较 energy/gradient，不要求 gate list 相同。应使用
float32/float64 两种精度，并按现有测试容差分别断言。

### 11.3 SAD execution mode 和 kernel 测试

扩展 `sad/tests/test_sad_runner.py` 等现有测试，覆盖：

| 规模 | 目的 |
|---:|---|
| 4q/1L | 最小合法 ring 与单层初始化 |
| 4q/2L | 跨层 phi/lambda 生命周期 |
| 6q/1L | 非二幂不相关但覆盖多 edge |
| 8q/2L | tile/block 边界基础场景 |
| 10q/3L | wrap-around 与较大 state |

每个关键规模至少验证：

- optimized/legacy energy 和 full gradient；
- forward、Hamiltonian、backward split time 都有限；
- parameter count 和错误输入；
- pair-CNOT 单独对 basis state 的 permutation；
- 连续两个相同 CNOT 恢复原 state；
- CNOT-RZ-CNOT 与直接 RZZ 的 state 对齐。

### 11.4 有限差分和 sanitizer

对 4q/1L float64 抽查 beta、gamma 两类参数的中心有限差分，避免 reference
和 SAD 共享同一门序错误。CUDA 环境可用时运行 compute-sanitizer，重点检查：

- pair-CNOT target mask；
- phi/lambda ping-pong swap；
- wrap-around wire 0；
- backward 逆序；
- 重复 gamma lookup 的 offsets。

### 11.5 原有回归

必须运行现有测试，确认旧 ID、旧 parameter count、旧 QAOA fused path 和旧
benchmark strategy descriptor 不变。任何针对 BD 的公共接口扩展都不能改变
旧 circuit 的 ABI 或默认 variant。

## 12. Benchmark 与结果隔离

BD 的 benchmark 目的是量化显式基础门的代价，不是证明它应接近专用 RZZ
QAOA。每层额外执行 `2n` 个 CNOT，并失去旧 cost-RX/RZZ 融合，预期明显更慢。

首轮固定：

```text
qubits    = 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28
layers    = 1, 2, 4, 8（按现有 benchmark 选择）
precision = float64
seed      = 42
```

新增独立结果文件，例如：

```text
benchmark/results/qaoa_bd_pennylane_gpu.csv
benchmark/results/qaoa_bd_native_gpu.csv
benchmark/results/qaoa_bd_sad_gpu.csv
benchmark/results/qaoa_bd_comparison.csv
```

结果中必须同时保存 `gate_count`、energy error、gradient error、forward/
Hamiltonian/backward time、library variant 和 execution mode。不要将 BD 行
写入已有 `qaoa` 或 `qaoa-ns` 的 winner/parameter-selection CSV。

## 13. 分阶段实施顺序

### Phase 1：reference gate decomposition

1. 验证 2-qubit RZZ 分解矩阵和角度 convention；
2. 添加 PennyLane `_qaoa_bd()`、注册和 Hamiltonian 分支；
3. 添加 gate-list、参数布局和 qaoa/qaoa-bd 数值等价测试；
4. 添加 cuQuantum `build_gates()` BD 分支和 native metadata。

验收：所有 reference backend 的 BD gate list 无 RZZ，energy/gradient 对齐。

### Phase 2：SAD ID 与 split executor

1. 添加新 enum ID、compile-time dispatch、参数量特判；
2. 添加 `qaoa_bd.cuh` layout/executor；
3. 实现普通 RZ lookup 使用；
4. 实现最小 pair-CNOT forward/backward kernel；
5. 将 BD 加入 MaxCut Hamiltonian dispatch。

验收：4q/6q/8q 的 legacy 与 optimized（即 split）路径对齐 reference。

### Phase 3：边界与完整正确性

1. 覆盖 odd/even matching、wrap-around、连续 layer；
2. 覆盖 float32/float64、full gradient、有限差分；
3. 覆盖 pair-CNOT 的 phi/lambda permutation 和 execution mode；
4. 运行全量原有回归与 sanitizer。

验收：无旧电路回归，BD 不触发旧 RZZ/fused symbols。

### Phase 4：性能优化

1. 对 pair-CNOT 做 tile/register/block profile；
2. 评估多个 pair launcher 是否可安全批处理，但保留可审计门语义；
3. 只有端到端稳定胜出后才增加 BD-specific variant；
4. 生成独立 benchmark 报告。

## 14. 完成标准

- 三个 runner 都识别 `qaoa-bd`，并正确处理 `qaoa_bd` alias；
- cost 层真实包含 CNOT-RZ-CNOT，不存在隐式/显式 RZZ；
- 参数布局仍为 `[beta_0, gamma_0, ...]`，参数量为 `2 * layers`；
- Hamiltonian 与原 QAOA 的 MaxCut 定义完全相同；
- SAD 使用指定 pair-CNOT，而不是整环 `launch_cnot()`；
- forward/backward 的 edge 和 layer 逆序正确；
- energy、完整 gradient、有限差分在容差内一致；
- workspace/context/run signatures 未被无关扩展；
- 旧 QAOA/RZZ fused path 和 benchmark 结果不受影响；
- 独立 benchmark 能报告 BD 的门数和性能代价。

## 15. Production 文件新增符号审计

没有列出的 production 文件不应新增 QAOA-BD 函数、类型或字段：

| 文件 | 允许新增内容 | 对齐依据 |
|---|---|---|
| `pennylane-lightning/.../circuits.py` | `_qaoa_bd()`、一个注册项、Hamiltonian 一个集合分支 | 现有 circuit builder/registry |
| `cuQuantum/python/.../runner.py` | BD supported name、ID、参数量和 gate-list 分支 | 现有 qaoa/qaoa-ns 分支 |
| `cuQuantum/include/cuquantum_api.h` | 一个 BD native circuit ID（如需要） | native gate API |
| `cuQuantum/src/cuquantum_cuda.cu` | MaxCut ID 集合增加一项 | 现有 Hamiltonian/gate dispatch |
| `sad/include/sad_api.h` | 一个 `SAD_CIRCUIT_QAOA_BD` 枚举值 | `SadCircuit` |
| `sad/src/runtime/circuit_dispatch.cuh` | 一个 switch case | compile-time visitor |
| `sad/src/runtime/circuit_execution.cuh` | 一个 include、一个 parameter-count 分支 | QAOA 特判 |
| `sad/src/circuits/qaoa_bd.cuh` | 一个 layout、一个 executor 特化 | `qaoa.cuh` |
| `sad/src/kernels/pair_cnot.cuh` | pair-CNOT kernel 和 launcher | `ring_cnot.cuh` 的 permutation 模式 |
| `sad/src/kernels/hamiltonian.cuh` | 无新 kernel；复用 `qaoa_cost_hamiltonian_kernel` | QAOA MaxCut observable |
| `sad/src/runtime/runner.cuh` | QAOA Hamiltonian 条件增加一项 | 现有 circuit dispatch |
| `sad/python/sad_baseline/runner.py` | `_CIRCUITS`、ID 参数量和 canonical-name 分支 | 现有 QAOA 特判 |
| benchmark scripts | 独立 BD 条目/结果路径 | 当前 benchmark loop |

特别禁止：

- 新增第二套 QAOA-BD topology registry；
- 给 workspace/context 增加每条 edge 的参数数组；
- 复用旧 ring-RZZ compact lookup 或 cost-RX fused launcher；
- 用一次 ring CNOT 代替每条分解中的 pair-CNOT；
- 为了性能在第一版改变门序或合并不相邻的 CNOT/RZ；
- 用 BD 结果覆盖已有 `qaoa` benchmark 或 parameter-selection 结论。
