# `/home/rzzhang/sad/cuQuantum` 独立实现计划

## 1. 目标和边界

目标是在 SAD 仓库根目录下新建独立目录：

```text
/home/rzzhang/sad/cuQuantum/
```

在该目录中实现 `inverse_walk_cuQuantum`，复用 SAD 的电路语义和测试/reference 结果，但**不修改**现有：

```text
/home/rzzhang/sad/sad/
```

因此，本方案不是给 SAD 主库增加一个 `ExecutionMode`，而是建立一个独立的 cuQuantum backend。`sad/sad` 只作为：

- 电路定义和参数布局的参考；
- Hamiltonian 语义的参考；
- 测试结果和 PennyLane parity 的参考。

独立 backend 的目标是：给定相同的 `circuit/qubits/layers/params`，输出与 SAD 当前实现数值一致的 energy 和 gradient。

## 2. 为什么采用独立目录

当前 SAD 的入口是一个固定 C ABI：

```text
sad/sad/include/sad_api.h
sad/sad/src/sad_cuda.cu
sad/sad/python/sad_baseline/runner.py
```

如果不修改这些文件，cuQuantum 实现不能通过原来的 `sad_energy_and_grad()` 自动进入，而应提供自己的库和 Python wrapper。这种方式的优点是：

- 不改变 SAD 默认行为；
- 不影响现有 CUDA 编译和 benchmark；
- cuStateVec 不安装时 SAD 仍可正常使用；
- 可以独立迭代 Gate IR 和 cuStateVec API，性能不在本任务范围内；
- 最后再决定是否需要一个极薄的上层适配器。

## 3. 推荐目录结构

第一版只新增以下内容：

```text
cuQuantum/
├── Makefile
├── README.md
├── include/
│   └── cuquantum_api.h
├── src/
│   ├── cuquantum_cuda.cu
│   ├── gate_ir.cuh
│   ├── circuit_adapters.cuh
│   ├── inverse_walk.cuh
│   ├── custatevec_resources.cuh
│   └── hamiltonian.cuh
├── python/
│   └── sad_cuquantum/
│       ├── __init__.py
│       └── runner.py
└── tests/
    ├── test_gate_ir.py
    ├── test_parity.py
    └── test_initial_state_gradients.py
```

如果实现可以直接放入一个 `.cu` 翻译单元，则 `src/` 中的文件可以先合并为：

```text
cuquantum_cuda.cu
cuquantum_runtime.cuh
```

不要为了形式上的模块化创建大量小文件。

## 4. 与 SAD 的接口

### 4.1 独立 C API

`cuQuantum/include/cuquantum_api.h` 提供与 SAD 输入语义相同、但 ABI 名称不同的接口：

```cpp
enum CuQuantumPrecision {
    CUQUANTUM_PRECISION_FLOAT32 = 0,
    CUQUANTUM_PRECISION_FLOAT64 = 1,
};

enum CuQuantumCircuit {
    CUQUANTUM_CIRCUIT_RA_HEA = 0,
    CUQUANTUM_CIRCUIT_SU2_HEA = 1,
    CUQUANTUM_CIRCUIT_RZZ_HEA = 2,
    CUQUANTUM_CIRCUIT_QAOA = 3,
    CUQUANTUM_CIRCUIT_XXZ_HVA = 4,
    CUQUANTUM_CIRCUIT_MERA = 5,
    CUQUANTUM_CIRCUIT_EQUIVARIANT_QNN = 6,
    CUQUANTUM_CIRCUIT_DATA_REUPLOADING = 7,
    CUQUANTUM_CIRCUIT_QAOA_NS = 8,
};

int cuquantum_energy_and_grad(
    int precision, int circuit, int qubits, int layers,
    const void* params, size_t parameter_count,
    double* out_energy, void* out_grad,
    char* error_message, size_t error_message_size);
```

第一版不复制 SAD 的 `steps/warmup/timing/memory` ABI。benchmark 通过 Python 侧计时；不要为了 benchmark 修改 `sad_api.h`。

### 4.2 Python API

`cuQuantum/python/sad_cuquantum/runner.py` 提供：

```python
result = run(
    circuit="su2-hea",
    qubits=8,
    layers=2,
    params=params,
    precision="float64",
)
```

Python 层负责：

- circuit 名称和别名归一化；
- 参数数量校验；
- ctypes 加载 `libcuquantum_sad.so`；
- cuStateVec 不可用时输出清晰错误；
- 返回 `energy` 和 `grad`。

SAD 的 `runner.py` 可以被读取和复制其参数规范，但不修改、不 import 私有实现。

## 5. Gate IR

`cuQuantum/src/gate_ir.cuh` 定义独立、可复用的 gate 表示：

```cpp
enum class GateKind : uint8_t {
    RX, RY, RZ, RZZ, CNOT,
    MATRIX_1Q, MATRIX_2Q,
};

template <typename T>
struct GateOp {
    GateKind kind;
    int wire0;
    int wire1;
    T theta0;
    T theta1;
    size_t parameter0;
    size_t parameter1;
    bool parameterized;
};
```

Gate IR 必须保存参数索引，而不是只保存角度。原因是：

- QAOA 的共享参数会被多个 gate 使用；
- QAOA-NS 的每个物理门有不同参数；
- 参数化初态也必须正确回传 gradient。

参数梯度统一采用：

```text
gradient[p] += 2 Re <lambda | dU_p | current>
current <- U_p^dagger current
lambda  <- U_p^dagger lambda
```

## 6. 从 SAD 电路定义建立 adapter

`cuQuantum/src/circuit_adapters.cuh` 只实现数据转换，不复制 SAD kernel。每个 adapter 读取 SAD 当前已有的门序和参数布局，并输出 Gate IR：

```cpp
template <typename T>
std::vector<GateOp<T>> build_ra_hea_gates(...);

template <typename T>
std::vector<GateOp<T>> build_su2_hea_gates(...);

template <typename T>
std::vector<GateOp<T>> build_rzz_hea_gates(...);

// qaoa, qaoa_ns, xxz_hva, mera,
// equivariant_qnn, data_reuploading
```

adapter 必须固定以下内容：

1. layer 顺序；
2. wire/control-target 顺序；
3. 参数索引；
4. gate 的正向角度符号；
5. 参数化初态；
6. 最终 observable 对应的 Hamiltonian。

### 6.1 电路覆盖表

| 电路 | Gate IR | 初态 | 特殊要求 |
|---|---|---|---|
| RA-HEA | RY/CNOT | 参数化 product state | 初态门也进入逆向遍历 |
| SU2-HEA | RY/RZ/CNOT | 参数化 product state | 与现有 `su2_hea.cuh` 顺序一致 |
| RZZ-HEA | RX/RZ/RZZ | 参数化 product state | 对齐 diagonal kernel 的符号 |
| QAOA | RZZ/RX | `|+>` | beta/gamma 共享参数累加 |
| QAOA-NS | RZZ/RX | `|+>` | 每个物理门独立参数 |
| XXZ-HVA | RX/RXX/RYY/RZZ 或等价矩阵 | Neel state | even/odd matching 顺序不能改变 |
| MERA | MATRIX_1Q/MATRIX_2Q | zero state | block 必须可逆、顺序固定 |
| Equivariant-QNN | RX/RY + 2Q block | zero state | interaction block 需精确分解 |
| Data Reuploading | RZ/RY/CNOT | zero state | 数据门与可训练门区分参数 |

### 6.2 参数化初态

GPU_Simulation 的实现从零态开始，不能直接覆盖 SAD 的所有电路。RA-HEA、SU2-HEA、RZZ-HEA 的参数化 product state 必须表示成初态前置 gate：

```text
|0...0>
-> initial-state gates
-> circuit gates
```

这样初态参数也会经过同一套 inverse-walk derivative。禁止通过有限差分补初态 gradient。

## 7. cuStateVec runtime

### 7.1 资源管理

`custatevec_resources.cuh` 维护：

```cpp
custatevecHandle_t handle;
DeviceBuffer<uint8_t> workspace;
size_t matrix_workspace_size;
bool matrix_workspace_size_cached;
```

使用 RAII，确保 handle 和 workspace 生命周期正确即可。由于本任务不考虑性能，允许每次调用重新创建 handle、重新查询 workspace 或逐 gate 使用临时资源。

### 7.2 forward

```text
allocate/clear state
apply initial-state gates
apply circuit gates in adapter order
```

RX/RY/RZ/RZZ 优先使用 `custatevecApplyPauliRotation`；CNOT 和一般 1Q/2Q gate 使用 `custatevecApplyMatrix`。允许每个 gate 单独调用 cuStateVec，不做 gate fusion；MERA 等复杂 block 使用固定 matrix/decomposition，唯一目标是复现 GPU_Simulation/SAD reference 的逻辑。

### 7.3 Hamiltonian

为避免复制 SAD 现有所有 Hamiltonian kernel，`cuQuantum/src/hamiltonian.cuh` 只需要提供与 SAD 结果等价的 expectation/action：

```text
lambda = H |psi>
energy = Re <psi | lambda>
```

第一版可直接复用数学公式，但不能链接或修改 `sad/sad` 的内部符号。Hamiltonian 实现只要求与 GPU_Simulation/SAD reference 的逻辑和数值一致，不要求高性能 expectation API。

### 7.4 backward

```text
zero gradients
for op in reverse(gates):
    if op.parameterized:
        evaluate analytic dU overlap
        accumulate to op.parameter0/op.parameter1
    apply op dagger to current
    apply op dagger to lambda
copy gradient to host
```

共享参数必须累加，初态参数必须参与逆向遍历，CNOT 等无参数门只执行 dagger。

## 8. 构建系统

`cuQuantum/Makefile` 独立于 `sad/Makefile`：

```make
NVCC ?= /usr/local/cuda/bin/nvcc
CUDA_ARCH ?= native
CUQUANTUM_ROOT ?=
TARGET ?= build/libcuquantum_sad.so
```

支持显式路径：

```bash
make -C /home/rzzhang/sad/cuQuantum \
  CUQUANTUM_ROOT=/path/to/cuquantum \
  CUDA_ARCH=sm_XX
```

链接时必须包含 `custatevec`，并设置运行时可发现的 rpath 或在 Python 加载前检查库路径。没有 cuStateVec 时：

- `/home/rzzhang/sad/sad` 仍可正常构建；
- `cuQuantum` 构建应明确提示缺失依赖；
- Python 调用不能静默回退到 SAD optimized。

## 9. 测试方案

### 9.1 Gate primitive

验证 RX/RY/RZ/RZZ/CNOT 的 forward、dagger 和 derivative。有限差分只作为测试 oracle，不进入实现。

### 9.2 电路 parity

每个电路至少测试最小合法 qubit/layer 和固定随机 seed，并比较：

1. SAD 当前实现；
2. `cuQuantum` backend；
3. PennyLane/native reference（已有时）。

建议初始容差：

```text
float64 energy: atol=1e-9, rtol=1e-8
float64 grad:   atol=1e-8, rtol=1e-7
float32 energy: atol=1e-5, rtol=1e-5
float32 grad:   atol=1e-4, rtol=1e-4
```

### 9.3 必须单独覆盖

- 参数化初态：RA-HEA、SU2-HEA、RZZ-HEA；
- 共享参数：QAOA；
- 非共享参数：QAOA-NS；
- even/odd matching：XXZ-HVA；
- 非幂次 qubit：MERA；
- 复杂 2Q block：equivariant-QNN；
- 数据门和训练参数：Data Reuploading。

测试代码只放在 `cuQuantum/tests`，不修改 SAD 原测试；测试通过 ctypes 或 Python wrapper 调用独立 `.so`。

## 10. 实施阶段

### 阶段 A：独立库骨架

- 建立 `cuQuantum/` 目录；
- 完成 Makefile、API header、Python loader；
- 完成 cuStateVec handle/workspace 的正确生命周期；
- 实现单门 primitive。

验收：可构建、可加载、默认 SAD 无变化。

### 阶段 B：SU2-HEA

优先实现 SU2-HEA，因为它覆盖 RY、RZ、CNOT 和参数化初态。完成 energy、全部 gradient 和 float32/float64 parity 后再扩展。

### 阶段 C：RA-HEA、RZZ-HEA、QAOA、QAOA-NS、Data Reuploading

重点验证初态梯度、RZZ 符号、共享参数和参数布局。

### 阶段 D：XXZ-HVA、MERA、Equivariant-QNN

先使用明确 gate decomposition/matrix apply，允许逐 gate、非融合执行；禁止为了性能改变 GPU_Simulation 的门序、逆向顺序或导数公式。

### 阶段 E：仓库 benchmark 入口

在现有目录中新增：

```text
/home/rzzhang/sad/benchmark/benchmark_cuQuantum.py
```

该脚本只依赖独立的 `cuQuantum/python/sad_cuquantum` wrapper，不修改 `sad/sad` 或现有 benchmark 脚本。

脚本职责：

1. 枚举全部电路：`ra-hea`、`su2-hea`、`rzz-hea`、`qaoa`、`qaoa-ns`、`xxz-hva`、`mera`、`equivariant-qnn`、`data-reuploading`；
2. 支持 `--circuits`、`--qubits`、`--layers`、`--precision`、`--seed`、`--repeats`；
3. 按 SAD 参数布局生成 deterministic 参数；
4. 调用 cuQuantum API，获得 energy 和 gradient；
5. 可选加载 SAD/PennyLane reference 做 correctness 对照；
6. 输出表格，并支持 `--output-json PATH`。

推荐命令：

```bash
python /home/rzzhang/sad/benchmark/benchmark_cuQuantum.py \
  --circuits all --qubits 4 6 8 --layers 1 \
  --precision float64 --repeats 1 --check-reference \
  --output-json /tmp/cuquantum_results.json
```

`--repeats` 默认值为 1。脚本可以记录 wall time 作为诊断信息，但不得将吞吐量、workspace 大小、kernel launch 数或 speedup 作为验收条件。核心输出为：

```text
circuit, qubits, layers, precision,
energy, max_abs_grad, reference_energy_error, reference_grad_error, status
```

不满足某电路合法 qubits/layers 约束时记录 `skipped` 和原因，不中止整个矩阵。

## 11. 最终验收标准

1. `cuQuantum` 目录可单独构建和测试；
2. `sad/sad` 源码没有被修改；
3. `/home/rzzhang/sad/benchmark/benchmark_cuQuantum.py` 可以枚举并运行全部 9 类电路；
4. 参数化初态、共享参数和复杂 block 的 gradient 都通过 parity；
5. float32/float64 结果在约定容差内一致；
6. cuStateVec 缺失时错误清晰，且不影响 SAD 原项目；
7. inverse-walk 的 forward 门序、反向门序、导数公式和 Hamiltonian 逻辑与 GPU_Simulation 一致；
8. README 说明独立安装、构建、调用和依赖。

明确不作为验收条件的项目：

- 执行时间；
- kernel launch 次数；
- cuStateVec workspace 大小；
- 显存峰值；
- 相对 SAD optimized 的 speedup；
- gate fusion、batching 或 stream overlap。

## 12. 不采用的方案

- 不把实现放入 `sad/sad/src`、`sad/sad/include` 或 `sad/sad/python`；
- 不修改 SAD 的 `ExecutionMode` 或 `sad_energy_and_grad()` ABI；
- 不复制 GPU_Simulation 的 Ring-Ising 专用文件作为通用实现；
- 不为每种电路复制一套完整 inverse-walk runtime；
- 不用有限差分替代 analytic inverse walk；
- 不在 cuStateVec 缺失时静默回退到 SAD optimized；
- 不为了降低 wall time 合并 gate、改写 matching、缓存跨调用状态或改变同步位置。

## 13. 结论

在不修改现有 `sad/sad` 的前提下，最小可行方案是建立一个独立的 `cuQuantum` backend，并在 `/home/rzzhang/sad/benchmark/benchmark_cuQuantum.py` 提供统一运行入口。它复用 SAD 的电路规范和结果作为 reference，通过 adapter 生成统一 Gate IR，再按 GPU_Simulation 的逐门 forward、Hamiltonian action 和 inverse-walk backward 逻辑执行。性能、融合和资源复用均不在本任务范围内，唯一验收目标是逻辑一致和数值 parity。
