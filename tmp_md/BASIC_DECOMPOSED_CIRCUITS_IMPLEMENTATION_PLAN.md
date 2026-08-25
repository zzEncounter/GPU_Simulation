# 基础门分解电路实现方案

## 1. 文档目的

本文设计三种新增电路：

- `qaoa-bd`
- `qaoa-ns-bd`
- `xxz-hva-bd`

`bd` 表示 basic decomposition。新电路显式使用 `RX`、`RY`、`RZ`、`Hadamard` 和 `CNOT`，不直接调用 `IsingZZ`、`IsingXX` 或 `IsingYY`。现有 `qaoa`、`qaoa-ns` 和 `xxz-hva` 保持不变，继续使用当前的专用融合路径。

目标是让 PennyLane、cuQuantum 和 SAD 三种实现具有相同的逻辑门序列，从而可以单独研究“基础门分解”相对于专用二比特门/融合 kernel 的性能影响。

## 2. 不变性原则

### 2.1 旧电路不修改

以下电路的名称、参数布局、门顺序、SAD kernel 和 benchmark 行为均保持不变：

```text
qaoa
qaoa-ns
xxz-hva
```

新增电路使用独立的 canonical name、circuit ID 和 executor，不通过旧名称上的条件分支改变旧行为。

### 2.2 参数数量和参数索引不改变

BD 只展开门，不增加可训练参数。固定的基变换门（例如 `H`、`RX(pi/2)`）不产生梯度。

```text
qaoa-bd:        2 * layers
qaoa-ns-bd:     2 * qubits * layers
xxz-hva-bd:     3 * qubits * layers
```

### 2.3 角度约定

项目中使用的 Pauli rotation 约定为：

```text
R_P(theta) = exp(-i * theta * P / 2)
```

因此所有分解必须使用同一个 `theta`，不能因为某个后端的 API 采用 `exp(-i theta P)` 而直接复用参数。

## 3. 基础门分解规则

### 3.1 RZZ

对二比特 `(q0, q1)`：

```text
RZZ(theta, q0, q1)
    = CNOT(q0, q1)
      RZ(theta, q1)
      CNOT(q0, q1)
```

该规则对应当前 PennyLane 的 `qml.IsingZZ(theta)` 和 SAD 的 `exp(-i theta Z_q0 Z_q1 / 2)`。

### 3.2 RXX

```text
RXX(theta, q0, q1)
    = H(q0) H(q1)
      RZZ(theta, q0, q1)
      H(q0) H(q1)
```

展开后为：

```text
H(q0)
H(q1)
CNOT(q0, q1)
RZ(theta, q1)
CNOT(q0, q1)
H(q0)
H(q1)
```

### 3.3 RYY

```text
RYY(theta, q0, q1)
    = RX(pi/2, q0) RX(pi/2, q1)
      RZZ(theta, q0, q1)
      RX(-pi/2, q0) RX(-pi/2, q1)
```

展开后为：

```text
RX(pi/2, q0)
RX(pi/2, q1)
CNOT(q0, q1)
RZ(theta, q1)
CNOT(q0, q1)
RX(-pi/2, q0)
RX(-pi/2, q1)
```

实现时必须用矩阵/状态向量测试确认 `RYY` 的正负号与当前 `qml.IsingYY` 约定一致。

## 4. qaoa-bd

### 4.1 逻辑电路

初始化和参数顺序与现有 QAOA 相同：

```text
H(q) for q = 0 ... n-1

for layer in 0 ... layers-1:
    beta  = params[2 * layer]
    gamma = params[2 * layer + 1]

    for edge in ring:
        CNOT(left, right)
        RZ(gamma, right)
        CNOT(left, right)

    RX(beta, q) for q = 0 ... n-1
```

环边顺序必须与现有 QAOA 一致：先 `left = 0, 2, ...`，再 `left = 1, 3, ...`。由于同一层 cost 角度共享，所有边使用同一个 `gamma`。

### 4.2 PennyLane

新增 `_qaoa_bd()`，不要调用 `qml.IsingZZ`，而是直接追加：

```python
qml.CNOT(wires=(left, right))
qml.RZ(gamma, wires=right)
qml.CNOT(wires=(left, right))
```

初始化可以继续使用 `qml.Hadamard`，因为 BD 的目标是分解 RZZ；但若要严格限制到 cuQuantum 支持的门集合，应统一改为项目当前 cuQuantum 使用的 `RY(pi/2)` 初态表示，并对能量/梯度做等价性验证。

### 4.3 cuQuantum

`build_gates(..., "qaoa-bd")` 生成的 gate list 只能包含：

```text
RY（或当前 H 的等价初始化表示）
CNOT
RZ
RX
```

cuQuantum native C++ 已支持这些 gate kind，不需要添加新的 `CUQUANTUM_GATE_*` 类型。RZZ 的 `_add_rzz()` 只用于旧电路；BD 电路应新增 `_add_rzz_decomposed()`。

### 4.4 SAD

新增 `SAD_CIRCUIT_QAOA_BD` 和 `CircuitExecutor<SAD_CIRCUIT_QAOA_BD, T>`。该 executor 必须按显式顺序调用：

```text
CNOT -> RZ -> CNOT -> RX
```

不能调用当前 QAOA 的 `shared_ring_rzz_factor()`、compact lookup 或 cost-RX fused kernel，否则执行语义会重新变成专用 RZZ 实现。

可以复用已有的：

- `launch_non_diagonal_forward/backward`
- `launch_cnot`
- 单比特 RZ 的 forward/backward 逻辑

## 5. qaoa-ns-bd

### 5.1 逻辑电路

每层的参数布局保持：

```text
base = 2 * layer * qubits
beta [q]  = params[base + q]
gamma[e]  = params[base + qubits + e]
```

电路为：

```text
H(q) for q = 0 ... n-1

for layer in 0 ... layers-1:
    for edge in ring:
        CNOT(edge, edge + 1)
        RZ(gamma[edge], edge + 1)
        CNOT(edge, edge + 1)

    RX(beta[q], q) for q = 0 ... n-1
```

与共享角 QAOA-BD 的区别只有参数独立性，门序列和分解规则相同。

### 5.2 三种实现

PennyLane 新增 `_qaoa_ns_bd()`；cuQuantum 新增 `qaoa-ns-bd` gate-list 分支；SAD 新增独立 circuit ID。SAD 不能将其映射到当前 `QAOA-NS` executor，因为旧 executor 使用 non-shared RZZ 专用 tile kernel。

BD 版本的 backward 必须分别输出每条边的 `gamma` 梯度，不能将同层边梯度规约成一个共享值。

## 6. xxz-hva-bd

### 6.1 逻辑结构

初态和 matching 顺序与现有 XXZ-HVA 相同：

```text
Néel state

for layer:
    even bonds
    odd bonds
```

每条 bond `(left, right)` 的参数索引保持：

```text
base = layer * 3 * qubits
theta_x = params[base + left]
theta_y = params[base + qubits + left]
theta_z = params[base + 2 * qubits + left]
```

### 6.2 每条 bond 的显式门序列

先展开 RXX：

```text
H(left)
H(right)
CNOT(left, right)
RZ(theta_x, right)
CNOT(left, right)
H(left)
H(right)
```

再展开 RYY：

```text
RX(pi/2, left)
RX(pi/2, right)
CNOT(left, right)
RZ(theta_y, right)
CNOT(left, right)
RX(-pi/2, left)
RX(-pi/2, right)
```

最后展开 RZZ：

```text
CNOT(left, right)
RZ(theta_z, right)
CNOT(left, right)
```

不要改变 `RXX -> RYY -> RZZ` 的顺序。相同 bond 上三者虽然对易，但 BD 电路应保留原始逻辑顺序，便于逐门比较。

### 6.3 PennyLane

新增 `_xxz_hva_bd()`，不再生成：

```python
qml.IsingXX
qml.IsingYY
qml.IsingZZ
```

而是通过内部 helper 追加上述基础门。固定的 `H`、`RX(pi/2)` 和 `RX(-pi/2)` 不应占用参数索引。

### 6.4 cuQuantum

当前 cuQuantum 的 XXZ 路径使用了 `RZZ + RX + RX` 的自定义表示，并没有使用 `theta_y`，不能作为 BD 实现。`xxz-hva-bd` 必须重新生成完整的：

```text
RXX decomposition
RYY decomposition
RZZ decomposition
```

gate list 中只允许出现 `RX`、`RY`、`RZ` 和 `CNOT`。这样 native cuStateVec 层不需要支持 `RXX/RYY` 新 gate kind。

### 6.5 SAD

新增 `SAD_CIRCUIT_XXZ_HVA_BD`。不能调用当前 `apply_xxz_bond()`，因为它会把 RXX/RYY/RZZ 融合为一次 partner update；BD 版本必须按基础门顺序执行。

建议复用现有基础 kernel：

- 单比特 `RX/RY/RZ` forward/backward
- `launch_cnot`
- 固定角度门的 forward/inverse 操作

固定基变换不生成梯度；只有 `theta_x`、`theta_y`、`theta_z` 对应的 `RZ` 产生梯度。若实现通用 inverse-walk，需要在 gate schedule 中保存参数 ID 和固定角度标志。

## 7. 注册和接口改动

### 7.1 PennyLane

修改：

```text
pennylane-lightning/src/pennylane_lightning_baseline/circuits.py
```

新增三个 builder、三个 `CircuitSpec`，并为每个电路增加 aliases（例如 `qaoa_bd`、`qaoa_ns_bd`、`xxz_hva_bd`）。测试中的 canonical name 应统一使用连字符形式。

### 7.2 cuQuantum

修改：

```text
cuQuantum/python/sad_cuquantum/runner.py
cuQuantum/include/cuquantum_api.h
cuQuantum/src/cuquantum_cuda.cu
```

建议新增 circuit enum，而不是复用旧 enum，以便 benchmark 和结果文件能区分 fused 与 BD。native C++ gate kind 无需增加，因为基础门集合已经足够。

### 7.3 SAD

需要同步修改：

```text
sad/include/sad_api.h
sad/python/sad_baseline/runner.py
sad/src/runtime/circuit_dispatch.cuh
sad/src/runtime/circuit_execution.cuh
sad/src/runtime/runner.cuh
sad/src/runtime/workspace.cuh
```

并新增独立 circuit executor/header。所有新 executor 都必须通过 circuit dispatch 使用自己的编译期分支，不能通过运行时字符串偷偷切换旧 executor 的行为。

## 8. 推荐的内部 helper

PennyLane 和 cuQuantum 都建议使用 helper，避免三个电路重复手写：

```text
append_rzz_decomposed(...)
append_rxx_decomposed(...)
append_ryy_decomposed(...)
```

helper 只负责追加门，不负责分配参数。参数索引由调用方传入，固定角度使用 `parameter=None` 或等价标记。

SAD 可以使用结构化 gate descriptor：

```text
kind
wire0
wire1
parameter_index
angle
is_trainable
```

这样 backward 可以跳过固定门，并对可训练 RZ/RX 正确累积梯度。

## 9. 正确性测试

### 9.1 电路级等价性

对相同随机参数比较：

```text
qaoa-bd       == qaoa
qaoa-ns-bd    == qaoa-ns
xxz-hva-bd    == xxz-hva
```

建议覆盖：

```text
qubits: 4, 6, 8
layers: 1, 2, 8
precision: float64
```

误差标准建议：

```text
energy: 1e-12
gradient: 1e-10
```

### 9.2 门序列检查

```text
qaoa-bd       不应出现 IsingZZ
qaoa-ns-bd    不应出现 IsingZZ
xxz-hva-bd    不应出现 IsingXX/IsingYY/IsingZZ
```

同时检查每个 RZZ 都严格对应：

```text
CNOT -> RZ -> CNOT
```

### 9.3 参数梯度检查

固定基变换不得出现在梯度输出中。需要确认：

- QAOA-BD 每层只有 `beta` 和 `gamma` 两个梯度；
- QAOA-NS-BD 每层有 `2 * qubits` 个梯度；
- XXZ-HVA-BD 每层有 `3 * qubits` 个梯度；
- 旧电路的参数数量和梯度顺序不发生变化。

## 10. 性能测试预期

BD 版本预期比旧版本慢，这是设计目标而不是 correctness 问题：

```text
RZZ: 至少 2 次 CNOT + 1 次 RZ
RXX: 额外 H 基变换
RYY: 额外 RX(pi/2) 基变换
```

BD benchmark 应单独输出，不能覆盖旧电路结果。建议结果文件使用：

```text
qaoa_bd_*.csv
qaoa_ns_bd_*.csv
xxz_hva_bd_*.csv
```

比较时至少报告：

```text
旧专用实现时间
BD 实现时间
BD/旧实现时间比
PennyLane BD
cuQuantum BD
SAD BD
```

## 11. 实现顺序

推荐按以下顺序开发：

1. 先在独立 Python helper 中验证 RZZ/RXX/RYY 矩阵等价性；
2. 实现 PennyLane 三个 BD builder 和门序列测试；
3. 修正 cuQuantum XXZ gate-list，新增三个 BD 分支；
4. 用 cuQuantum/PennyLane 做 energy 和 gradient 对照；
5. 增加 SAD circuit ID 和基础门 executor；
6. 完成 SAD forward/backward 正确性测试；
7. 最后运行独立性能 benchmark。

任何阶段如果 BD 与旧电路不等价，应先检查角度符号、qubit endian、CNOT control/target 和固定基变换顺序，不应通过修改 Hamiltonian 或容差掩盖问题。

