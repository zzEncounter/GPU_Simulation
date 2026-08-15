# 文档索引

## 当前实验

- [`experiments/主实验.md`](experiments/主实验.md)：2026-08-16 重新实测的五类电路主实验，以及按电路/规模选择参数相对统一缺省参数的配对实验。
- [`experiments/NATIVE_BASELINE_COMPARISON.md`](experiments/NATIVE_BASELINE_COMPARISON.md)：早期 SAD、Lightning native 与 PennyLane QNode 三方基线研究，作为历史方法说明保留。

## 阶段研究

- [`research/OPTIMIZATION_REPORT.md`](research/OPTIMIZATION_REPORT.md)：最初的 CUDA adjoint 优化、消融与端到端实验。
- [`research/ROTATION_OPTIMIZATION_REPORT.md`](research/ROTATION_OPTIMIZATION_REPORT.md)：RX/RY tile、mailbox、reduction 与 backward 深入研究。
- [`research/MEMORY_XXZ_CNOT_REPORT.md`](research/MEMORY_XXZ_CNOT_REPORT.md)：L2、XXZ 配对和 ring-CNOT 后续研究；包含不采用显式 L2 hit 优化的最终结论。
- [`research/DIAGONAL_FUSION_REDUCTION_REPORT.md`](research/DIAGONAL_FUSION_REDUCTION_REPORT.md)：对角门融合、reduction、cross-matching 和 launch 优化。
- [`research/EXECUTION_STRATEGY_REPORT.md`](research/EXECUTION_STRATEGY_REPORT.md)：phase、kernel shape、mailbox、XXZ 分区及电路融合的搜索总结。

当前结果以《主实验》为准；阶段研究报告用于解释候选方案、消融过程和最终 dispatch 的来源。
