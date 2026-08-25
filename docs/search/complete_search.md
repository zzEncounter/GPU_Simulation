# 完整电路执行粒度参数搜索方案

## 1. 目标与边界

目标是覆盖从“每个原始门单独启动一个 CUDA kernel”到当前 SAD 生产最佳实现之间的全部合法执行粒度，并搜索中间的融合、分组、kernel 几何和运行策略。

这里的“完整”定义为覆盖每一种合法的 kernel 启动粒度和融合级别，而不是枚举所有任意门排列。对有 `G` 个可排序门的层，任意连续分区有 `2^(G-1)` 种，按字面完全枚举不可执行。因此中间方案采用保持门顺序的连续、均匀分组，并枚举每一种合法 group count。

当前仅调节 rotation phase 不能达到该目标，因为它只控制 RX/RY target 的分组，RZ、RZZ、CNOT 和电路专用 kernel 仍然可能批量或融合执行。

## 2. 两个明确端点

### 2.1 Gatewise 最差端点

对每个原始逻辑门分别启动 kernel：

```text
每个 RX 一个 kernel
每个 RY 一个 kernel
每个 RZ 一个 kernel
每个 RZZ bond 一个 kernel
每个 CNOT 一个 kernel
```

Forward 和 backward 都必须支持 gatewise schedule。Backward 应按逆序执行逆门；参数门的 backward kernel 同时完成逆演化和该门的梯度累积，非参数门只执行逆传播。

### 2.2 Production 最优端点

直接使用当前 SAD production dispatch 作为最优基线，包括自动 shape、自动 phase plan、专用电路融合、RA real-amplitude、RZZ 策略和 QAOA 专用 backward。不要手工复制生产策略，以免基线与生产实现漂移。

## 3. 统一 GateSchedule 接口

当前 `forward_phase_plan`/`backward_phase_plan` 只能描述 rotation target 的 phase，必须增加能够描述整个电路执行图的 schedule 接口。

示意格式：

```text
forward_schedule =
  layer0:init
  layer0:group(RY:0-3)
  layer0:group(CNOT:0-3)
  layer0:group(RZ:0-3)
  layer0:group(RZZ_EVEN:0-3)
  layer0:group(RZZ_ODD:0-3)
```

每个 schedule group 至少包含：

```text
gate_family       RX / RY / RZ / RZZ_EVEN / RZZ_ODD / CNOT / INIT
gate_indices      该组包含的逻辑门或 target
kernel_variant    generic / specialized / fused
fusion_boundary   是否与相邻 group 融合
launch_mode       ordinary / persistent
```

Forward 和 backward 必须分别保存 schedule，不能假设二者的分组、shape 或融合方式相同。

## 4. 中间粒度的规范化枚举

对每个 gate family 单独枚举 group count。例如一层有 28 个 RX target 时，枚举：

```text
group_count = 1, 2, ..., 28
```

使用连续、尽可能均匀的分组：

```text
28 个 target，4 组 -> 7+7+7+7
28 个 target，10 组 -> 3+3+3+3+3+3+3+3+2+2
```

这样可以覆盖从一个 family 一个 kernel 到每个 target 一个 kernel 的所有启动次数，同时避免枚举性能等价的任意排列。

应分别搜索：

```text
rx_group_count
ry_group_count
rz_group_count
rzz_even_group_count
rzz_odd_group_count
cnot_group_count
```

电路中不存在的 gate family 使用 `not_applicable`。

## 5. Fusion 与 launch 搜索轴

### 5.1 Fusion mask

对合法的相邻 gate family 搜索是否融合，例如：

```text
RX + RZ
RX + RZZ
RX + CNOT
RZ + RZZ
RZZ + CNOT
```

融合必须经过电路依赖和状态读写检查，不能强行合并存在顺序约束的门。

### 5.2 Launch mode

```text
ordinary-per-group
persistent-per-schedule
```

当前宏的语义为：

```text
SAD_ROTATION_PERSISTENT=0  -> 每个 Phase/group 单独 launch
SAD_ROTATION_PERSISTENT=1  -> 所有 Phase 在一个 cooperative kernel 内执行
```

RA real-amplitude 和 SU2 phased RY 也分别有对应的 persistent 宏。

## 6. 可直接使用的现有编译宏

以下宏已经存在，可以作为候选搜索轴加入 `candidate_library(flags)`。

### RA-HEA

```text
SAD_REAL_AMPLITUDE=0/1
SAD_RA_FORWARD_FUSED=0/1
SAD_RA_BACKWARD_FUSED=0/1
SAD_REAL_PERSISTENT=0/1
```

### SU2-HEA

```text
SAD_SU2_FORWARD_STRATEGY=0/1/2
SAD_SU2_BACKWARD_STRATEGY=0/1/2
SAD_PHASED_RY_PERSISTENT=0/1
SAD_SU2_PHASED_BACKWARD=0/1
```

### RZZ-HEA

```text
SAD_RZZ_FORWARD_FUSED=0/1
SAD_RZZ_BACKWARD_STRATEGY=0/1/2
```

### QAOA/QAOA-NS

```text
SAD_QAOA_FUSE_COST_RX=0/1
SAD_QAOA_FUSED_BACKWARD=0/1
SAD_QAOA_COMPACT_LOOKUP=0/1
SAD_QAOA_INITIAL_STRATEGY=0/1/2
```

### 通用 rotation、资源和 mailbox

```text
SAD_ROTATION_PERSISTENT
SAD_BLOCK_THREADS
SAD_FORWARD_BLOCK_THREADS
SAD_REGISTER_BITS
SAD_FORWARD_REGISTER_BITS
SAD_ORDINARY_BLOCK_THREADS
SAD_DIAGONAL_BLOCK_THREADS
SAD_SHARED_DIAGONAL_BLOCK_THREADS
SAD_DIAGONAL_LOOKUP_BITS
SAD_MAILBOX_CHUNKS
SAD_RY_SCALAR_MAILBOX
SAD_ROTATION_WARP_ATOMIC
SAD_DIAGONAL_WARP_ATOMIC
SAD_LEGACY_BLOCK_REDUCTION
SAD_FIXED_LOW_LANES
SAD_FORWARD_FIXED_LOW_LANES
SAD_ALTERNATE_PHASES
```

所有组合必须通过 tile capacity、shared memory、register、occupancy 和编译检查；非法候选仍需写入结果并标记失败原因。

## 7. 现有宏无法完成的部分

以下能力没有现成宏，必须增加 GateSchedule runtime 和新的 CUDA launch wrapper/kernel：

```text
每个 RZ 门一个 kernel
每个 RZZ bond 一个 kernel
每个 CNOT 一个 kernel
每个 gate 的 backward inverse kernel
逐门参数梯度累积
Equivariant-QNN brickwork 逐门拆分
Data-Reuploading brickwork 逐门拆分
通用 GateSchedule ABI
```

现有 diagonal launcher 会批量处理一个 RZ/RZZ family；仅改变 block 或 fusion 宏不会把它拆成每个门一次 launch。

## 8. Candidate 定义

完整 candidate 至少应包含：

```text
circuit, qubits, layers
forward_shape, backward_shape
forward_gate_schedule, backward_gate_schedule
forward_fusion_mask, backward_fusion_mask
forward_launch_mode, backward_launch_mode
rotation_kernel_variant, diagonal_kernel_variant, cnot_kernel_variant
mailbox_chunks, register_bits, forward_register_bits
ordinary_threads, diagonal_threads, shared_diagonal_threads, lookup_bits
real_amplitude, legacy_reduction
rotation_warp_atomic, diagonal_warp_atomic
fixed_low_lanes, alternate_phases
rzz_strategy, qaoa_strategy, su2_strategy, ra_strategy
```

`candidate_key` 必须包含上述所有会影响执行的字段，防止不同 schedule 被错误去重。

## 9. 推荐搜索流程

不建议把所有轴直接做一个巨大笛卡尔积。建议使用三层搜索，同时保留每层的完整 CSV/JSON 结果。

### 第一层：执行结构搜索

固定中等 kernel shape，搜索：

```text
gatewise
familywise
phasewise
fused
production
```

目标是建立从逐门到融合的结构性能曲线。

### 第二层：完整 schedule 粒度搜索

对每种结构搜索：

```text
RX/RY/RZ/RZZ/CNOT 的 group_count
forward/backward 独立配置
fusion mask
ordinary/persistent launch
```

### 第三层：kernel 参数搜索

对第二层的每种 schedule 搜索：

```text
forward/backward shape
register bits
block threads
mailbox
lookup bits
atomic/reduction
```

如果要求严格全穷举，shape 应使用所有合法的：

```text
all_valid_forward_shapes × all_valid_backward_shapes
```

不能只保留当前的 5 个代表性 shape pair。

## 10. 输出与统计

每个电路继续写一个 CSV 和一个 JSON，并在每个候选完成后 flush、fsync 和原子更新 JSON。JSON 除了全局和按 qubits 的最好、最差、平均、中位数，还应记录：

```text
total_kernel_launches_forward
total_kernel_launches_backward
kernel_launches_per_layer
gatewise_fraction
fused_group_count
forward_fusion_mask
backward_fusion_mask
```

同时保存 cuQuantum 基准时间和：

```text
cuQuantum / SAD-best
cuQuantum / SAD-worst
```

双方必须统一 precision、layers、warmup、steps，以及是否包含 gradient。

## 11. 最终结论

真正完整的搜索方案是：

```text
通用 GateSchedule ABI
+ 每个 gate family 的 group_count 搜索
+ forward/backward 独立 schedule
+ fusion mask 搜索
+ ordinary/persistent launch 搜索
+ 现有编译宏搜索
+ 所有合法 forward/backward shape 搜索
+ gatewise 与 production 两个明确端点
```

仅修改当前 `partition_level` 或增加 rotation 宏，最多只能得到：

```text
RX/RY 逐 target
+ RZ/RZZ/CNOT 按 family 批量
```

要真正覆盖“每个原始门一个 kernel”到“当前最佳实现”，必须新增 RZ、RZZ、CNOT 以及 backward gatewise 的执行路径。

## 12. SAD 代码改造方案

### 12.1 数据模型

在 C API 和 Python API 之间增加版本化 schedule 描述。建议使用结构化 JSON 作为外部格式，在 workspace 创建阶段解析一次，转换为紧凑的 C++ 数据结构；执行 step 时不得重复解析字符串。

核心结构建议如下：

```cpp
enum class GateFamily {
    RX,
    RY,
    RZ,
    RZZ,
    CNOT,
    INITIALIZE
};

enum class LaunchMode {
    ORDINARY_PER_GROUP,
    PERSISTENT_PER_SCHEDULE
};

struct GateGroup {
    GateFamily family;
    int first_gate;
    int gate_count;
    int parameter_offset;
    int fusion_group;
};

struct DirectionSchedule {
    LaunchMode launch_mode;
    std::vector<GateGroup> groups;
};
```

每个电路必须先生成规范化的逻辑 gate 列表，再由 schedule 将连续 gate 映射到 kernel group。Schedule 不得改变原始门顺序或参数编号。

### 12.2 API 扩展

建议新增 API 版本，而不是修改现有函数的行为：

```text
sad_energy_and_grad_v2(...,
                       forward_schedule_json,
                       backward_schedule_json,
                       ...)
```

兼容规则：

```text
未传 GateSchedule -> 保持当前 production dispatch
只传 phase plan -> 保持当前 rotation phase 行为
传 GateSchedule -> 由新 schedule runtime 执行
```

必须拒绝同时传入互相冲突的 phase plan 和 GateSchedule，避免实际执行方案与结果字段不一致。

### 12.3 Gatewise kernel

需要为以下操作增加带范围参数的 launch wrapper：

```text
launch_rx_group(first, count)
launch_ry_group(first, count)
launch_rz_group(first, count)
launch_rzz_group(first_bond, count)
launch_cnot_group(first_edge, count)
```

当 `count=1` 时即为逐门端点；当 `count=family_size` 时为 familywise 端点。中间值由同一个 wrapper 支持，避免为每种粒度维护不同 kernel。

Backward 需要对应的 group kernel：

```text
launch_rx_backward_group
launch_ry_backward_group
launch_rz_backward_group
launch_rzz_backward_group
launch_cnot_backward_group
```

参数门必须只写入本 group 对应的梯度范围，并保证多个 group 的累积结果与生产实现一致。

### 12.4 运行时 dispatch

执行路径建议分为：

```text
PRODUCTION       当前生产实现
SCHEDULED        新 GateSchedule runtime
LEGACY           当前 legacy 实现，仅作为兼容基线
```

`SCHEDULED` 路径逐层读取预处理后的 group 数组，并按 forward 或 backward 顺序调用相应 launcher。禁止在 step 循环中分配 host/device 内存。

## 13. 各电路的 gate family 映射

每个电路需要显式声明一层中的 gate family、顺序和可融合边界。

| 电路 | 主要 gate family | 必须支持的拆分 |
|---|---|---|
| RA-HEA | RY、CNOT | RY target、CNOT edge |
| SU2-HEA | RY、RZ、CNOT | RY target、RZ target、CNOT edge |
| RZZ-HEA | RX、RZ、RZZ-even、RZZ-odd | RX/RZ target、每个 RZZ bond |
| QAOA | RZZ cost、RX mixer | 每个 cost bond、每个 mixer target |
| QAOA-NS | RZZ cost、RX mixer | 每个 cost bond、每个 mixer target |
| Equivariant-QNN | RX、RY、brickwork entangler | rotation target、每条 entangler edge |
| Data Re-uploading | RZ、RY、RZ、brickwork entangler | rotation target、每条 entangler edge |

第一层初始化也必须明确：production 可以保留 product-state 初始化 kernel；严格 gatewise 端点则应由零态开始，按原始初始化门序列执行。

MERA 和 QAOA-BD 按当前范围继续排除。XXZ-HVA 若未来纳入完整 E2E 搜索，应先把当前 matching microbenchmark 替换为统一 GateSchedule E2E 路径，不能与其他电路直接混用时间口径。

## 14. 合法融合规则

融合搜索不能使用任意 bit mask。应由每个电路提供合法 fusion rule，候选生成器只枚举规则允许的边。

每条 fusion rule 至少验证：

```text
两个 group 在原始门序列中相邻
不存在未包含在 group 中的依赖门
状态输入和输出 buffer 兼容
forward/backward 的门顺序正确
梯度写入范围不冲突
shared memory 和 register 使用合法
persistent kernel 满足 cooperative launch 限制
```

融合后的 group count 必须重新计算。结果文件应同时保存逻辑 gate 数、schedule group 数和实际 kernel launch 数。

## 15. 候选空间控制

### 15.1 完整粒度集合

对 unit count 为 `N` 的 family，严格完整的 group-count 集合为：

```text
K = {1, 2, ..., N}
```

这与当前只选 5 个 `partition_level` 不同。候选中直接保存 `group_count` 和分组 counts，`partition_level` 只作为派生显示值：

```text
partition_level = (group_count - K_min) / (K_max - K_min)
```

### 15.2 联合搜索规模

即使每个 family 只枚举 group count，多 family 的完整笛卡尔积仍可能非常大。例如 RX、RZ、CNOT 都有 `N` 种粒度时，仅分组轴就有 `N^3` 个组合；再乘 forward/backward、fusion 和 shape 后不可直接运行。

因此必须区分：

```text
完整轴覆盖：每个合法参数值至少被测量
完整笛卡尔积：所有参数值彼此组合
```

建议保证完整轴覆盖，但使用分阶段的条件笛卡尔积：

1. 固定 kernel 参数，完整搜索 schedule 粒度和 fusion。
2. 对每种结构类别保留最好、最差和若干分位候选。
3. 对保留候选完整搜索 kernel 参数。
4. 最后对各阶段赢家做交叉组合验证。

如果研究要求真正的完整笛卡尔积，脚本必须先生成 manifest 并报告候选总数、预计编译数、预计 GPU 时间和磁盘占用，得到明确确认后再执行。

### 15.3 去重

以下情况必须规范化去重：

```text
不同 level 映射到相同 group counts
关闭 fusion 后无效的 fusion 子参数
非 persistent 路径中的 persistent-only 参数
不含某 gate family 的电路参数
不同描述生成完全相同的 schedule
```

去重依据应是规范化 schedule 与全部有效编译宏的哈希，而不是用户输入文本。

## 16. 搜索脚本架构

建议拆分为以下模块：

```text
benchmark/complete_search/
  circuit_ir.py          电路逻辑门 IR
  schedule_models.py     Candidate 和 GateSchedule 数据模型
  schedule_generator.py  group count、fusion、方向组合生成
  schedule_validator.py  依赖、容量和资源合法性检查
  build_cache.py         编译宏归一化和动态库缓存
  runner.py              warmup、重复测量和超时控制
  persistence.py         CSV append、fsync、JSON 原子更新
  summarize.py           按电路/qubits/结构聚合
  search.py              CLI 和阶段编排
```

编译缓存 key 与运行 candidate key 必须分离：多个 schedule 可以共享同一动态库，但不能共享运行结果。

```text
build_key     = hash(有效编译宏 + 源码版本)
candidate_key = hash(build_key + circuit + qubits + schedules + measurement config)
```

## 17. 搜索 CLI

建议 CLI 至少支持：

```bash
python benchmark/complete_search/search.py \
  --circuits ra-hea,su2-hea,rzz-hea,qaoa,qaoa-ns,equivariant-qnn,data-reuploading \
  --qubits 4,6,8,10,12,14,16,18,20,22,24,26,28 \
  --layers 8 \
  --stages structure,schedule,kernel,cross-validation \
  --output-dir benchmark/results/complete_search \
  --resume
```

辅助选项：

```text
--dry-run                 只生成 manifest 和预计成本
--candidate-manifest PATH 从固定 manifest 执行
--retry-failed            重试失败候选
--timeout-seconds N       单候选超时
--max-builds N            限制不同动态库数量
--max-candidates N        调试时限制候选数量
--verify-correctness      每个候选与 reference 对比
--profile-kernel-counts   使用 profiler 验证 launch 数
```

## 18. 测量规则

沿用按 qubits 设置重复次数的规则：

| qubits | steps |
|---:|---:|
| 4-8 | 20 |
| 10-14 | 10 |
| 16-20 | 5 |
| 22-24 | 3 |
| 26-28 | 2 |

所有候选使用 `warmup_steps=3`。此外应增加：

```text
同一 qubits 内候选随机化执行顺序
记录 GPU 型号、driver、CUDA、时钟和源码 commit
每个候选执行前后检查 CUDA error
超时和 OOM 独立记录
不得把编译时间计入运行时间
```

最好、最差、平均和中位数继续基于每个候选的 `median_ms` 聚合。

## 19. 正确性验证

性能候选必须先通过正确性门槛：

```text
energy_abs_error <= tolerance
gradient_max_abs_error <= tolerance
所有参数梯度均被写入一次且仅一次
forward 最终状态与 reference 一致
backward 后 phi/lambda 状态与 reference 一致
```

至少覆盖以下测试：

1. 每个 gate family 的 `count=1` 和 `count=N`。
2. 奇偶 qubits 和 ring 边界。
3. forward/backward 使用不同 group count。
4. 每一种合法 fusion rule。
5. ordinary 与 persistent 输出一致。
6. 逐门端点和 production 端点均与 reference 一致。
7. 断点恢复不会重复或漏掉 candidate。

错误候选必须保留在 CSV 中，但不得进入性能统计。

## 20. Kernel launch 数验证

不能仅根据 schedule 推算 launch 数。实现完成后应使用 CUPTI、Nsight Systems 或 CUDA profiler hook 验证实际 launch：

```text
expected_kernel_launches
observed_kernel_launches
kernel_name_histogram
```

Gatewise 端点只有在每个目标门都能对应到预期 launch，且不存在意外融合或 CUDA Graph 合并时，才能标记为 `gatewise_verified=true`。

## 21. 持久化与中断恢复

每个电路一个 CSV 和一个 JSON：

```text
benchmark/results/complete_search/ra-hea.csv
benchmark/results/complete_search/ra-hea.json
...
```

每个候选结束后：

1. 向 CSV append 一行。
2. 调用 `flush()` 和 `fsync()`。
3. 从兼容 schema 的 reader 读取有效行。
4. 将 JSON 写入同目录临时文件。
5. `fsync()` 后使用原子 rename 替换正式 JSON。

JSON 必须按每个 qubits 保存：

```text
expected_candidates
attempted_candidates
completed_ok
status_counts
best/worst/arithmetic_mean/median
best_candidate/worst_candidate
gatewise_endpoint
production_endpoint
cuquantum_baseline
```

Resume 以 `candidate_key` 跳过成功候选；`--retry-failed` 只重试失败候选。Manifest 必须单独保存，保证中断前后候选空间不因代码枚举顺序变化而改变。

## 22. 实施顺序与验收标准

### 阶段 A：IR 和 schedule runtime

实现 Gate IR、schedule parser/validator、RX/RY group 路径和 API v2。验收条件是 RA-HEA 能从一个 RY group 连续搜索到每个 RY/CNOT 一个 group。

### 阶段 B：对角门与双比特门

实现 RZ、RZZ、CNOT group forward/backward。验收条件是 SU2-HEA、RZZ-HEA 和 QAOA 都有经过 profiler 验证的 gatewise 端点。

### 阶段 C：其余电路和融合图

实现 Equivariant-QNN、Data Re-uploading 的 entangler group，并为所有电路建立合法 fusion rule。

### 阶段 D：完整搜索器

实现 manifest、编译缓存、分阶段搜索、实时 CSV/JSON、resume、失败重试和统计报告。

最终验收必须同时满足：

```text
每个电路存在 gatewise_verified 端点
每个电路存在 production 端点
所有 group count 值均被覆盖
forward/backward 可以独立配置
结果通过 energy/gradient reference
实际 kernel launch 数与 schedule 一致
中断后 CSV/JSON 可读且可恢复
cuQuantum 与 SAD 使用相同 benchmark 口径
```
