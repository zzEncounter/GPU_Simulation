# 文档索引

## 当前实验

- [`experiments/贡献.md`](experiments/贡献.md)：面向 reviewer 的论文写作底稿；按 motivation、design、作用机制、实验证据和结论边界系统说明各项贡献。
- [`experiments/主实验.md`](experiments/主实验.md)：2026-08-17 重新实测的五类电路最终结果，只报告 SAD 相对 Lightning GPU native 的外部比较。
- [`experiments/参数选择.md`](experiments/参数选择.md)：唯一参数选择报告；包含全固定→部分选择→最佳、快速结构开关、方向独立 shape、bit-position、异构 phase 与 mailbox 验证。
- [`experiments/NATIVE_BASELINE_COMPARISON.md`](experiments/NATIVE_BASELINE_COMPARISON.md)：早期 SAD、Lightning native 与 PennyLane QNode 三方基线研究，作为历史方法说明保留。

## 阶段研究

- [`research/OPTIMIZATION_REPORT.md`](research/OPTIMIZATION_REPORT.md)：最初的 CUDA adjoint 优化、消融与端到端实验。
- [`research/ROTATION_OPTIMIZATION_REPORT.md`](research/ROTATION_OPTIMIZATION_REPORT.md)：RX/RY tile、mailbox、reduction 与 backward 深入研究。
- [`research/MEMORY_XXZ_CNOT_REPORT.md`](research/MEMORY_XXZ_CNOT_REPORT.md)：L2、XXZ 配对和 ring-CNOT 后续研究；包含不采用显式 L2 hit 优化的最终结论。
- [`research/DIAGONAL_FUSION_REDUCTION_REPORT.md`](research/DIAGONAL_FUSION_REDUCTION_REPORT.md)：对角门融合、reduction、cross-matching 和 launch 优化。
- [`research/EXECUTION_STRATEGY_REPORT.md`](research/EXECUTION_STRATEGY_REPORT.md)：phase、kernel shape、mailbox、XXZ 分区及电路融合的搜索总结。

最终外部结果以《主实验》为准，内部参数与消融结论以《参数选择》为准；阶段研究报告只用于解释历史候选来源。
