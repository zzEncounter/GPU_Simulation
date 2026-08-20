# QAOA-NS 与 XXZ-HVA 基础门分解电路实现文档

## 1. 目标与命名

本文规划新增两种独立电路，仅实现 PennyLane 和 cuQuantum：

```text
qaoa-ns-bd
xxz-hva-bd
```

其中 `bd` 表示 basic decomposition。用户描述中的 `xzz-hva` 按仓库现有 canonical 名称统一为 `xxz-hva`；不新增 `xzz-hva` 这一拼写，避免与现有 circuit registry 不一致。

现有 `qaoa-ns` 和 `xxz-hva` 保持不变。新电路必须使用独立 canonical name、alias、参数计数和 benchmark 输出，不能通过参数长度或环境变量改变旧电路的行为。本文不涉及 SAD executor、SAD kernel 或生产 dispatch。

## 2. 分解约定

所有旋转均采用项目约定：

```text
R_P(theta) = exp(-i * theta * P / 2)
```

### 2.1 RZZ

```text
RZZ(theta, q0, q1)
  = CNOT(q0, q1)
    RZ(theta, q1)
    CNOT(q0, q1)
```

这里的 `RZ` 角度是 `theta`，不是 `theta / 2`。两个 CNOT 必须保持相同的 control/target 方向。

### 2.2 RXX

```text
RXX(theta, q0, q1)
  = H(q0) H(q1)
    RZZ(theta, q0, q1)
    H(q0) H(q1)
```

展开后的门序为：

```text
H(q0), H(q1), CNOT(q0,q1), RZ(theta,q1), CNOT(q0,q1), H(q0), H(q1)
```

PennyLane 可直接追加 `Hadamard`。cuQuantum 若不增加 H gate kind，应使用等价的固定序列 `RZ(pi)` 后 `RY(pi/2)`（仅差全局相位）；该序列的顺序不能反过来。

### 2.3 RYY

```text
RYY(theta, q0, q1)
  = RX(pi/2, q0) RX(pi/2, q1)
    RZZ(theta, q0, q1)
    RX(-pi/2, q0) RX(-pi/2, q1)
```

展开后的门序为：

```text
RX(pi/2,q0), RX(pi/2,q1), CNOT(q0,q1), RZ(theta,q1),
CNOT(q0,q1), RX(-pi/2,q0), RX(-pi/2,q1)
```

实现前必须用 4x4 矩阵测试确认该符号与 `qml.IsingYY` 一致。固定基变换角度不是可训练参数，不出现在梯度向量中。

## 3. 电路不变量

### 3.1 环边顺序

两种电路都沿用现有环拓扑：

```text
edge i = (i, (i + 1) mod qubits), i = 0 ... qubits-1
```

`qaoa-ns-bd` 的每层按 `edge=0 ... qubits-1` 执行；`xxz-hva-bd` 每层先 parity 0，再 parity 1，保持现有 `_xxz_hva` 的 matching 顺序。最后一条环边固定为 `(qubits-1, 0)`，不能改成 `(0, qubits-1)`。

### 3.2 参数布局

`qaoa-ns-bd` 完全复用现有 `qaoa-ns` 布局：

```text
base = 2 * layer * qubits
beta(layer, wire) = params[base + wire]
gamma(layer, edge) = params[base + qubits + edge]
parameter_count = 2 * qubits * layers
```

`xxz-hva-bd` 完全复用现有 `xxz-hva` 布局：

```text
base = 3 * layer * qubits
x(theta, edge) = params[base + edge]
y(theta, edge) = params[base + qubits + edge]
z(theta, edge) = params[base + 2 * qubits + edge]
parameter_count = 3 * qubits * layers
```

每个分解产生的 `RZ` 只引用对应原二比特门的参数；基变换门和 CNOT 不增加参数。

## 4. qaoa-ns-bd

### 4.1 逻辑门序

初态与现有 QAOA-NS 等价，为 `|+>^n`。PennyLane 使用 `Hadamard`；cuQuantum 使用固定 `RY(pi/2)` 制备 `|+>`，避免引入新的 H gate kind。这里只要求初态等价；RXX 的 basis-change H 仍按第 2.2 节用 `RZ(pi)`、`RY(pi/2)` 实现。

```text
H(q) for q = 0 ... n-1

for layer = 0 ... layers-1:
    base = 2 * layer * n
    for edge = 0 ... n-1:
        q0 = edge
        q1 = (edge + 1) mod n
        CNOT(q0, q1)
        RZ(params[base + n + edge], q1)
        CNOT(q0, q1)
    for wire = 0 ... n-1:
        RX(params[base + wire], wire)
```

该电路的 cost 层不允许调用 `qml.IsingZZ` 或 cuQuantum 的 native RZZ gate；目标是让两个后端都实际执行 CNOT-RZ-CNOT。

### 4.2 PennyLane 修改

修改：

```text
pennylane-lightning/src/pennylane_lightning_baseline/circuits.py
```

新增 `_qaoa_ns_bd()`，复制现有 `_qaoa_ns()` 的参数索引和 RX 顺序，只将每个 `qml.IsingZZ` 替换为：

```python
qml.CNOT(wires=(edge, right))
qml.RZ(params[base + qubits + edge], wires=right)
qml.CNOT(wires=(edge, right))
```

注册 canonical `qaoa-ns-bd`，建议 alias 为 `qaoa_ns_bd`。Hamiltonian 与 `qaoa-ns` 完全相同，仍使用环上的 `0.5 * (Z_i Z_(i+1) - I)`。

### 4.3 cuQuantum 修改

修改：

```text
cuQuantum/python/sad_cuquantum/runner.py
```

增加：

- `supported_circuits` 中的 `qaoa-ns-bd`；
- `_NATIVE_CIRCUIT` 的新 ID，不能移动现有 ID；
- `expected_parameter_count()` 返回 `2 * qubits * layers`；
- `build_gates()` 的独立分支。

新增 `_add_rzz_decomposed(gates, left, right, parameter, angle)`，只追加两个 CNOT 和一个参数化 RZ。BD 分支不能复用 `_add_rzz()`，因为 `_add_rzz()` 会产生 native `rzz` gate。固定初始化继续使用现有 `_product_initial(..., "ry", ...)` 或等价固定 RY gate。

cuQuantum C++ 层通常已经支持 RX/RY/RZ/CNOT；如果 gate kind 或 circuit ID 由头文件枚举维护，必须同步更新 `cuQuantum/include/cuquantum_api.h`、native switch 和 Python registry。

## 5. xxz-hva-bd

### 5.1 逻辑门序

初态保持现有 XXZ-HVA 的 Néel state：PennyLane 在奇数 wire 应用 `PauliX`，cuQuantum 继续使用现有固定 `RY(pi)` 等价表示。

每层保留 parity 0、parity 1 的顺序。每条 bond 的原始操作顺序为 RXX、RYY、RZZ，BD 版本逐个展开，不能把三种 Pauli rotation 重排或合并：

```text
for parity in (0, 1):
    for left in range(parity, qubits, 2):
        right = (left + 1) mod qubits
        decompose RXX(params[base + left], left, right)
        decompose RYY(params[base + qubits + left], left, right)
        decompose RZZ(params[base + 2 * qubits + left], left, right)
```

注意：奇数 parity 的 `left` 可取 `qubits-1`，此时 `right=0`；control/target 仍为 `(qubits-1, 0)`。

### 5.2 PennyLane 修改

修改 `pennylane-lightning/src/pennylane_lightning_baseline/circuits.py`，新增 `_xxz_hva_bd()`。保留 `_xxz_hva()` 的初态、layer-major 参数偏移和 parity 循环，只将：

```python
qml.IsingXX(...)
qml.IsingYY(...)
qml.IsingZZ(...)
```

分别替换为第 2 节的显式基础门序列。建议使用三个私有 helper（例如 `_append_rxx_decomposed`），避免在两个电路 builder 中复制分解规则；helper 只追加门，不改变参数索引。

注册 canonical `xxz-hva-bd`，alias 建议为 `xxz_hva_bd`。Hamiltonian 与 `xxz-hva` 完全相同：每条环边的 `XX + YY + 0.5 ZZ`。

### 5.3 cuQuantum 修改

在 `runner.py` 增加 `xxz-hva-bd` 独立 gate-list 分支。该分支必须按 `RXX -> RYY -> RZZ` 顺序追加分解门：

```text
RXX: H,H,CNOT,RZ,CNOT,H,H
RYY: RX(+pi/2),RX(+pi/2),CNOT,RZ,CNOT,RX(-pi/2),RX(-pi/2)
RZZ: CNOT,RZ,CNOT
```

所有参数化 `RZ` 使用各自的原参数索引。固定 H/RY 和 RX(pi/2) 不得占用 parameter slot，也不得出现在 gradient 输出中。不得调用 native RZZ/RXX/RYY gate；如果底层暂时没有 RXX/RYY kind，则无需新增它们，因为 BD 路径已经完全展开。

## 6. 共享实现边界

允许共享：

- Python gate-list 的 `_add_cnot`、单比特 `_add`；
- RZZ/RXX/RYY 的纯 Python append helper；
- cuQuantum 的 RX/RY/RZ/CNOT application 和 adjoint 逻辑；
- circuit name normalization、CSV schema 和随机参数生成。

禁止共享导致语义混淆：

- BD 电路不得进入旧电路的 native `rzz` 分支；
- 旧电路不得因为新增 BD helper 而改变 gate list；
- 不以 `if len(params) == ...` 推断电路类型；
- 不把 BD 的固定基变换作为可训练参数；
- 不将 RXX/RYY/RZZ 跨门融合回一个 native Pauli rotation。

## 7. 测试验收标准

### 7.1 分解正确性

新增纯矩阵测试，覆盖随机 `theta`：

```text
decompose(RZZ(theta)) == qml.IsingZZ(theta)
decompose(RXX(theta)) == qml.IsingXX(theta)
decompose(RYY(theta)) == qml.IsingYY(theta)
```

误差阈值建议为 float64 下 `1e-12`。测试正向矩阵、逆向矩阵和参数导数；重点检查 RYY 的符号。

### 7.2 PennyLane tape 测试

新增 `tests/test_circuits.py` 覆盖：

- canonical name 与 alias 可解析；
- `qaoa-ns-bd` 参数量为 `2*n*layers`；
- `xxz-hva-bd` 参数量为 `3*n*layers`；
- QAOA-NS-BD tape 不含 `IsingZZ`，每条边恰为 `CNOT,RZ,CNOT`；
- XXZ-HVA-BD tape 不含 `IsingXX/IsingYY/IsingZZ`；
- RXX/RYY/RZZ 的参数索引、wire 顺序和 parity 顺序正确；
- 两个 BD 电路的 Hamiltonian 与原电路一致；
- 小规模随机参数的 energy/full gradient 与原电路一致。

### 7.3 cuQuantum gate-list 测试

对 4 qubits、1 layer 检查：

- BD gate list 不含 `rzz`（也不含尚未支持的 `rxx/ryy`）；
- QAOA-NS 每条边有 `CNOT,RZ,CNOT`；
- XXZ-HVA 每条 bond 的门数量为 `7 + 7 + 3 = 17`；
- 参数化 RZ 的 parameter index 与原布局一致；
- 固定初始化和 basis-change rotation 的 parameter index 为 `None`；
- cuQuantum energy/gradient 与 PennyLane reference 对齐。

## 8. Benchmark 与交付物

新增独立输出文件，不覆盖旧结果：

```text
benchmark/results/qaoa_ns_bd_pennylane_gpu.csv
benchmark/results/qaoa_ns_bd_cuquantum.csv
benchmark/results/xxz_hva_bd_pennylane_gpu.csv
benchmark/results/xxz_hva_bd_cuquantum.csv
```

benchmark 固定 `float64`、相同随机种子、相同 qubit 列表、8 layers，以及与原电路一致的 warmup/steps。报告至少同时记录：总 median、forward/energy/backward 时间、energy 误差、gradient 最大绝对误差和 gate count。BD 与原电路的比较必须按同一 qubit、同一 layers、同一参数布局对齐。

完成条件：

1. 两个新 canonical circuit 在 PennyLane 和 cuQuantum 中可独立运行；
2. 所有二比特 Pauli rotation 均由显式基础门组成；
3. energy 与 full gradient 通过 float64 一致性测试；
4. 旧 `qaoa-ns`、`xxz-hva` 测试和结果不发生变化；
5. 新 benchmark CSV 和 gate-list 快照可复现。
