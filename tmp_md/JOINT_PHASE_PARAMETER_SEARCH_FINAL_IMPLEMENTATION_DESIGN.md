# 联合 Phase 参数搜索脚本最终实现设计

## 1. 目标与范围

本文是最终收敛后的实现设计。脚本只搜索所有目标电路共同适用的三类参数，并对含 RZZ 门的电路增加一个二值策略参数：

```text
execution_profile × shape_pair × partition_level × mailbox_chunks
（含 RZZ 的优化 profile 再乘以 rzz_strategy）
```

正式搜索电路为：

```text
ra-hea
su2-hea
rzz-hea
qaoa
xxz-hva
equivariant-qnn
data-reuploading
qaoa-ns
```

明确排除：

```text
mera
qaoa-bd
```

本设计不搜索 ordinary/diagonal threads、lookup bits、reduction、persistent 等执行参数。搜索保留两个执行 profile：`optimized` 使用当前 SAD 优化路径，`legacy-generic` 使用 legacy/非 real-amplitude 路径作为粗粒度 baseline，以覆盖从粗到精的运行时间范围。对于包含 RZZ 门的优化 profile，`rzz_strategy` 取 `merged` 或 `split`。XXZ 使用历史 `microbench_xxz.cu` 路径执行 pair-count phase，不走 SAD Python E2E ABI。

## 2. 输出布局

每个电路独立输出一个 CSV 和一个 JSON，不允许所有电路共用同一结果文件：

```text
benchmark/results/joint_phase/
  ra-hea.csv
  ra-hea.json
  su2-hea.csv
  su2-hea.json
  rzz-hea.csv
  rzz-hea.json
  qaoa.csv
  qaoa.json
  xxz-hva.csv
  xxz-hva.json
  equivariant-qnn.csv
  equivariant-qnn.json
  data-reuploading.csv
  data-reuploading.json
  qaoa-ns.csv
  qaoa-ns.json
```

脚本文件：

```text
benchmark/search_joint_phase_parameters.py
```

如需保存全局运行信息，另写一个不参与候选统计的 metadata 文件：

```text
benchmark/results/joint_phase/run_metadata.json
```

## 3. 固定实验常量

```python
LAYERS = 8
PRECISION = "float64"
RANDOM_SEED = 42
PHASE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
MAILBOX_CHUNKS = (1, 2)
```

qubit 默认列表：

```python
QUBITS = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28)
```

shape pair 固定为：

`ShapePair` 是有方向的有序 pair，forward 和 backward 可以不同，不能在候选生成器中强制二者相等。当前代表性 grid 为：

```python
SHAPE_PAIRS = (
    ShapePair(32, 2, 32, 2),
    ShapePair(64, 2, 64, 2),
    ShapePair(128, 2, 128, 2),
    ShapePair(32, 2, 128, 2),
    ShapePair(128, 2, 32, 2),
)
```

最后两个 pair 是显式异构方向候选。后续扩展 shape grid 时也必须使用 `forward_shapes × backward_shapes` 的有序笛卡尔积，并分别执行 forward/backward 的 `valid_shape()` 和资源检查。

不含 RZZ 策略时，每个 `(circuit, qubits)` 的优化 profile 理论候选数为：

```text
5 shape pairs × 5 phase levels × 2 mailbox values = 50
```

每个非 XXZ 电路另外增加 5 个 `legacy-generic` baseline（每个 shape 一个，不重复 phase/mailbox，因为 legacy 不消费 phase map）；因此非 RZZ 电路为 55，含 RZZ 电路为 105。XXZ 保持 50 个 microbenchmark 候选。实际候选数可能因电路 qubit 约束、shape 合法性和资源预检出现失败状态，但失败候选必须保留。

## 4. Phase 语义

### 4.1 统一 level 映射

对每个电路、qubit、方向和 shape 独立计算：

```text
K_min = 当前 shape 下的最少合法 phase 数
K_max = 当前执行 kernel 支持的最细合法 phase 数
```

再使用：

```python
K(level) = round(K_min + level * (K_max - K_min))
```

并强制：

```text
level 0.0 -> K_min
level 1.0 -> K_max
```

中间 level 如果映射到相同 `K`，仍保留独立候选行和独立 `candidate_key`；可以复用相同 plan，但不能丢掉 level 记录。

phase 内的 unit 使用连续、尽可能均匀的分区：

```python
base, remainder = divmod(unit_count, phase_count)
counts = [base + 1] * remainder + [base] * (phase_count - remainder)
```

实际生成时必须再次通过 shape capacity、family 约束和顺序 validator；不能仅依赖数学分配。

### 4.2 Rotation 电路

适用电路：

```text
ra-hea
su2-hea
rzz-hea 的 RX 部分
qaoa/qaoa-ns 的 RX mixer 部分
equivariant-qnn
data-reuploading
```

phase 只作用于 RX/RY rotation target，不对 RZZ、RZ 或其他 diagonal gate 生成 phase level。

运行时编码使用当前 SAD 支持的形式：

```text
compact:LxRyWz-...
fixed:LxRyWz-...
```

每个方向独立生成：

```python
build_rotation_plan(circuit, qubits, shape, level, direction)
```

`partition_level=0` 表示该 shape 下最少合法 rotation target phase，不保证字面上只有一个 phase；`partition_level=1` 表示最细合法 rotation-target phase，不宣称每个原始电路门独立执行。

### 4.3 RZZ 电路

`rzz-hea`、`qaoa` 和 `qaoa-ns` 中的 RZZ 不参与 `partition_level` 搜索，但增加一个独立的：

```text
rzz_strategy ∈ {merged, split}
```

该参数只对含 RZZ 门的电路生效；其他电路写入 `not_applicable`。不得把 RZZ bond phase 编码成 rotation `LxRyWz`。

### 4.4 XXZ-HVA

`xxz-hva` 保持历史搜索逻辑。它使用 bond/matching pair-count partition，而不是普通 rotation plan。

对 even/odd matching 和 forward/backward 分别生成 pair counts：

```text
forward_even_pair_counts
forward_odd_pair_counts
backward_even_pair_counts
backward_odd_pair_counts
```

例如：

```text
4+4+4
3+3+2+2+2
1+1+...+1
```

`partition_level=0` 对应 shape 下最少合法 matching phase，`partition_level=1` 对应每个合法 bond/matching unit 一个 phase。XXZ 的既有 matching policy 按当前生产配置固定，不作为搜索轴。

## 5. 代码模块设计

### 5.1 `joint_phase_search_common.py`

新增公共模块，包含：

```python
ShapePair
PhasePlan
Candidate
normalize_circuit()
steps_for_qubits()
build_rotation_plan()
build_xxz_matching_plan()
validate_plan()
candidate_key()
write_json_summary()
```

`PhasePlan` 至少包含：

```python
family                 # rotation 或 xxz-matching
partition_level
phase_count
unit_count
encoded                # SAD runtime 字符串或 pair-count 字符串
coverage_digest
```

### 5.2 `search_joint_phase_parameters.py`

负责：

1. CLI 参数解析；
2. 电路和 qubit 合法性检查；
3. 完整候选生成；
4. 固定随机种子打乱候选顺序；
5. 调用 SAD runner；
6. correctness 检查；
7. 逐候选 CSV/JSON 写入；
8. 中断与恢复。

不得调用旧脚本的 `best_config()`，不得根据已有结果淘汰 shape、level 或 mailbox。

## 6. Candidate 数据结构

```python
@dataclass(frozen=True)
class Candidate:
    candidate_key: str
    circuit: str
    qubits: int
    layers: int
    precision: str
    random_seed: int
    shape: ShapePair
    partition_level: float
    forward_plan: PhasePlan
    backward_plan: PhasePlan
    mailbox_chunks: int
    rzz_strategy: str
    execution_profile: str
    fixed_execution_flags: tuple[tuple[str, int], ...]
    fixed_structure_policy: str
    steps: int
    warmup_steps: int
```

`candidate_key` 使用排序后的完整配置 JSON 生成 SHA-256 摘要，至少包含：

```text
circuit, qubits, layers, precision, seed
shape pair
partition level
forward/backward plan
mailbox
fixed flags
fixed structure policy
steps/warmup_steps
```

不同 phase plan、方向 shape、mailbox 或 level 不得共享 key。

## 7. 测量协议

### 7.1 Steps schedule

```python
def repetitions_for_qubits(qubits: int) -> int:
    if qubits <= 8:
        return 20
    if qubits <= 14:
        return 10
    if qubits <= 20:
        return 5
    if qubits <= 24:
        return 3
    return 2

warmup_steps = 3
steps = repetitions_for_qubits(qubits)
```

每个候选的重复次数由 qubit 数决定，所有候选的 `warmup_steps` 固定为 3。`repetitions`、`steps` 和 `warmup_steps` 都必须写入 CSV/JSON metadata，不能使用旧搜索脚本的固定 `3/10` 默认值。

### 7.2 完整执行

除 `xxz-hva` 外，每个候选调用完整：

```text
forward + Hamiltonian + backward
```

`xxz-hva` 采用历史方案 B：编译并调用 `benchmark/microbench_xxz.cu`，分别测量 even/odd parity 的 forward/backward matching kernel，再将 forward 与 backward 的平均时间写入同一候选行。XXZ 的时间口径是 matching microbenchmark，不等同于其他电路的完整 energy-and-gradient E2E 时间；输出中的 `kernel_variant` 字段留空，不使用特殊标识。

调用 `sad_baseline.energy_and_grad()` 时显式传入：

```python
circuit=candidate.circuit
scalability=(candidate.qubits, 8)
precision="float64"
steps=candidate.steps
warmup_steps=candidate.warmup_steps
forward_phase_plan=candidate.forward_plan.encoded
backward_phase_plan=candidate.backward_plan.encoded
```

每个候选至少重复 3--5 次。保存每次样本，并计算：

```text
median_ms
mean_ms
min_ms
max_ms
std_ms
forward_ms
hamiltonian_ms
backward_ms
```

## 8. CSV 设计

每个电路 CSV 使用 append-only 写入，至少包含：

```text
timestamp_utc
run_id
candidate_key
status
error
circuit
qubits
layers
precision
random_seed
forward_threads
forward_register_bits
backward_threads
backward_register_bits
mailbox_chunks
partition_level
phase_family_forward
phase_family_backward
phase_count_forward
phase_count_backward
forward_phase_plan
backward_phase_plan
forward_pair_counts
backward_pair_counts
fixed_execution_flags
fixed_structure_policy
steps
warmup_steps
repetitions
rzz_strategy
median_ms
mean_ms
min_ms
max_ms
std_ms
forward_ms
hamiltonian_ms
backward_ms
energy_abs_error
gradient_max_abs_error
kernel_variant
library_path
```

XXZ 不适用的 `forward_phase_plan` 字段写空字符串，pair counts 写入专用字段；rotation 电路反之。

状态统一为：

```text
invalid_shape
invalid_phase
resource_limit
compile_failed
runtime_failed
correctness_failed
ok
```

所有状态都写 CSV。只有 `ok` 且 correctness 通过的行参与运行时间统计。

## 9. JSON 实时汇总

每个电路拥有独立 JSON。每写入一条 CSV 后，立即从该电路完整 CSV 重建 JSON，并使用临时文件原子替换：

```text
ra-hea.csv -> ra-hea.json
```

JSON 至少包含：

```json
{
  "schema_version": 1,
  "circuit": "ra-hea",
  "updated_utc": "...",
  "expected_candidates": "按 qubit/circuit 约束计算",
  "attempted_candidates": 12,
  "completed_ok": 8,
  "status_counts": {
    "ok": 8,
    "resource_limit": 2,
    "runtime_failed": 2
  },
  "runtime_ms": {
    "best": 0.0,
    "worst": 0.0,
    "arithmetic_mean": 0.0,
    "median": 0.0
  },
  "best_candidate": {},
  "worst_candidate": {},
  "by_qubits": {},
  "by_partition_level": {}
}
```

其中：

- `best`：所有 `status=ok` 候选的最小 `median_ms`；
- `worst`：所有 `status=ok` 候选的最大 `median_ms`；
- `arithmetic_mean`：所有有效候选 median 的算术平均；
- `median`：所有有效候选 median 的中位数。

JSON 还应按 `qubits` 提供局部统计。每个 qubit 项至少包含：

```json
{
  "completed_ok": 50,
  "runtime_ms": {
    "best": 0.0,
    "worst": 0.0,
    "arithmetic_mean": 0.0,
    "median": 0.0
  },
  "best_candidate": {},
  "worst_candidate": {}
}
```

如需绘制 phase 曲线，再按 `(qubits, partition_level)` 从 CSV 派生统计；JSON 的强制汇总粒度是每个 qubit。

如果当前没有有效候选，四个时间字段写 `null`，不能写 0 伪造结果。

## 10. 实时写入与中断恢复

### 10.1 写入顺序

每个候选严格按以下顺序提交：

1. 执行候选；
2. 生成完整 row；
3. `writer.writerow(row)`；
4. `flush()`；
5. `os.fsync()`；
6. 从 CSV 重建 JSON；
7. JSON 写入 `.tmp` 后 `os.replace()`。

因此只要候选执行完成，CSV 和 JSON 都会立即反映结果。

### 10.2 中断处理

捕获 `SIGINT`、`SIGTERM` 和 `KeyboardInterrupt`：

- 不再提交新的候选；
- 等待当前 runner 调用返回；
- flush/fsync 当前 CSV；
- 重建并原子替换 JSON；
- 退出时保留已完成结果。

如果进程在 CUDA 调用中被强制 kill，当前候选可能没有结果行，但之前已经提交的行不会丢失。

### 10.3 恢复

启动时读取对应电路 CSV 的所有 `candidate_key`：

- 已存在的 `ok` 或明确失败状态默认跳过；
- 没有记录的候选继续执行；
- `--retry-failed` 可重新执行 `compile_failed`、`runtime_failed` 和 `correctness_failed`；
- 重试追加新行，不覆盖旧行，并生成新的 `run_id`。

CSV 半行或损坏行必须在启动时报告，不能静默当成成功候选。

## 11. CLI

```text
python benchmark/search_joint_phase_parameters.py \
  --circuits ra-hea,su2-hea,rzz-hea,qaoa,xxz-hva,equivariant-qnn,data-reuploading,qaoa-ns \
  --qubits 4,6,8,10,12,14,16,18,20,22,24,26,28 \
  --output-dir benchmark/results/joint_phase \
  --shuffle-seed 42 \
  --resume \
  --retry-failed
```

建议选项：

```text
--dry-run              只生成候选计数和 manifest，不调用 CUDA
--max-candidates       仅测试/调试使用，正式实验禁止依赖它缩减空间
--no-shuffle           调试时保持确定性顺序
--device               CUDA device id
```

默认每个电路单独处理；如果指定多个电路，完成一个候选后立即更新该电路自己的 CSV/JSON。

## 12. 正确性验证

每个有效候选都检查：

```text
energy_abs_error <= tolerance
gradient_max_abs_error <= tolerance
```

phase plan 额外检查：

### Rotation

- 所有 RX/RY target 恰好覆盖一次；
- phase 顺序稳定；
- phase count 与 level 映射一致；
- forward/backward plan 方向正确；
- plan 满足 tile/register/fixed family 约束。

### XXZ

- even/odd pair counts 总和正确；
- 每个 bond 恰好出现一次；
- 每个 phase 不超过 pair capacity；
- backward matching 顺序正确；
- matching policy 与当前固定策略一致。

正确性失败的候选必须保留 CSV 行，但不进入 JSON runtime 统计。

## 13. 测试计划

纯 Python 测试至少覆盖：

1. 电路 alias 规范化；
2. `mera`、`qaoa-bd` 被拒绝；
3. 非 RZZ 电路 50、含 RZZ 电路 100 候选/`(circuit, qubits)` 计数；
4. `K_min/K_max` 和五个 level 映射；
5. rotation plan 均匀分区和 coverage；
6. XXZ pair-count plan 与旧 `candidate_partitions()` 语义一致；
7. candidate key 对 shape/level/mailbox/plan 变化敏感；
8. CSV append、JSON 原子更新和重复 key 检查；
9. best/worst/mean/median 统计公式；
10. 中断后恢复不重复已记录候选。

GPU smoke test 至少覆盖：

```text
ra-hea 4q level=0.0
ra-hea 4q level=1.0
rzz-hea 4q RX phase
xxz-hva 4q pair-count phase
```

XXZ smoke test 必须实际编译 `microbench_xxz.cu`，验证自定义 pair counts 被传入并执行；不能再把 XXZ pair-count candidate 标记为 `invalid_phase`。

## 14. 实施顺序

1. 实现公共 dataclass、固定 grids 和候选计数；
2. 实现 rotation/XXZ 两套 phase generator；
3. 实现 candidate key、CSV schema 和 JSON 汇总；
4. 实现 append/fsync/atomic replace；
5. 实现恢复、retry 和信号处理；
6. 接入 SAD runner 的完整 energy-and-gradient 调用；
7. 完成纯 Python 测试；
8. 完成 GPU smoke test；
9. 先运行单电路小 qubit 校准，再启动完整搜索。

最终实验结论只基于每个电路自己的有效候选集合。JSON 中的 best、worst、arithmetic mean 和 median 是实际已完成有效候选的统计值；程序中断时这些字段只代表截至中断时已完成的数据，并通过 `attempted_candidates` 和 `updated_utc` 明确标识。
