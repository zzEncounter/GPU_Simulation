import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── 读取数据 ──────────────────────────────────────────────────────────────────
DIR = "/home/user225/GPU_Simulation/ring_cx_pipeline_vs_original_structed_vs_inversecuQuantum"

df_pipeline = pd.read_csv(f"{DIR}/ring_cnot_pipeline_vs_cuquantum_8to24q_2layers_merged.csv")
df_origin   = pd.read_csv(f"{DIR}/originstructed_vs_cuquantum_8to24q_2layers_summary_merged.csv")

# ── 提取三条曲线 ──────────────────────────────────────────────────────────────
# df_pipeline 包含：structured_adjoint (ring_cx_pipeline版) + inverse_walk_cuQuantum
# df_origin   包含：structured_adjoint (original版)         + inverse_walk_cuQuantum

def extract(df, method):
    sub = df[df['method'] == method].sort_values('num_qubits')
    return sub['num_qubits'].values, sub['avg_step_ms'].values

q_pipe_sa,  ms_pipe_sa  = extract(df_pipeline, 'structured_adjoint')
q_pipe_cq,  ms_pipe_cq  = extract(df_pipeline, 'inverse_walk_cuQuantum')
q_orig_sa,  ms_orig_sa  = extract(df_origin,   'structured_adjoint')
q_orig_cq,  ms_orig_cq  = extract(df_origin,   'inverse_walk_cuQuantum')

# inverse_walk_cuQuantum 在两个文件里基本一致，取 origin 版本作为参考
# 三条主线：
#   1. original structured_adjoint  (df_origin)
#   2. ring_cx_pipeline structured_adjoint (df_pipeline)
#   3. inverse_walk_cuQuantum (df_origin)

# ── 绘图 ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "Performance Comparison: original structured_adjoint vs ring_cx_pipeline vs inverse_walk_cuQuantum\n"
    "(2 layers, 20 steps, RTX 6000 Ada)",
    fontsize=13, fontweight='bold', y=1.01
)

colors = {
    'orig_sa':   '#1f77b4',   # 蓝
    'pipe_sa':   '#2ca02c',   # 绿
    'cuquantum': '#d62728',   # 红
}
markers = {'orig_sa': 'o', 'pipe_sa': 's', 'cuquantum': '^'}

# ── 子图 1：avg_step_ms（线性 Y 轴）─────────────────────────────────────────
ax1 = axes[0]
ax1.plot(q_orig_sa, ms_orig_sa,
         color=colors['orig_sa'], marker=markers['orig_sa'], linewidth=2,
         label='original structured_adjoint')
ax1.plot(q_pipe_sa, ms_pipe_sa,
         color=colors['pipe_sa'], marker=markers['pipe_sa'], linewidth=2,
         label='ring_cx_pipeline structured_adjoint')
ax1.plot(q_orig_cq, ms_orig_cq,
         color=colors['cuquantum'], marker=markers['cuquantum'], linewidth=2,
         label='inverse_walk_cuQuantum')

ax1.set_xlabel('Number of Qubits', fontsize=12)
ax1.set_ylabel('Avg Step Time (ms)', fontsize=12)
ax1.set_title('Avg Step Time (linear scale)', fontsize=12)
ax1.set_xticks(q_orig_sa)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# ── 子图 2：avg_step_ms（对数 Y 轴，更清晰地看小 qubit 差异）────────────────
ax2 = axes[1]
ax2.semilogy(q_orig_sa, ms_orig_sa,
             color=colors['orig_sa'], marker=markers['orig_sa'], linewidth=2,
             label='original structured_adjoint')
ax2.semilogy(q_pipe_sa, ms_pipe_sa,
             color=colors['pipe_sa'], marker=markers['pipe_sa'], linewidth=2,
             label='ring_cx_pipeline structured_adjoint')
ax2.semilogy(q_orig_cq, ms_orig_cq,
             color=colors['cuquantum'], marker=markers['cuquantum'], linewidth=2,
             label='inverse_walk_cuQuantum')

ax2.set_xlabel('Number of Qubits', fontsize=12)
ax2.set_ylabel('Avg Step Time (ms, log scale)', fontsize=12)
ax2.set_title('Avg Step Time (log scale)', fontsize=12)
ax2.set_xticks(q_orig_sa)
ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax2.grid(True, which='both', alpha=0.3)
ax2.legend(fontsize=10)

# ── 在每条曲线上标注数值（仅对数图，避免拥挤）────────────────────────────────
for q, ms, col in [
    (q_orig_sa, ms_orig_sa, colors['orig_sa']),
    (q_pipe_sa, ms_pipe_sa, colors['pipe_sa']),
    (q_orig_cq, ms_orig_cq, colors['cuquantum']),
]:
    for x, y in zip(q, ms):
        if x in [8, 12, 16, 20, 24]:   # 只标注部分点，避免拥挤
            ax2.annotate(f'{y:.2f}', xy=(x, y),
                         xytext=(3, 4), textcoords='offset points',
                         fontsize=7, color=col)

plt.tight_layout()
out_path = f"{DIR}/comparison_avg_step_ms.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"图片已保存到: {out_path}")

# ── 额外：加速比图 ─────────────────────────────────────────────────────────────
fig2, ax3 = plt.subplots(figsize=(10, 5))
fig2.suptitle(
    "Speedup vs inverse_walk_cuQuantum\n(2 layers, 20 steps, RTX 6000 Ada)",
    fontsize=13, fontweight='bold'
)

# 以 inverse_walk_cuQuantum 为基准，计算加速比
# 对齐 qubit 列表
common_q = np.intersect1d(q_orig_sa, q_orig_cq)
ms_cq_aligned  = [ms_orig_cq[list(q_orig_cq).index(q)]  for q in common_q]
ms_orig_aligned = [ms_orig_sa[list(q_orig_sa).index(q)] for q in common_q]
ms_pipe_aligned = [ms_pipe_sa[list(q_pipe_sa).index(q)] for q in common_q if q in q_pipe_sa]
common_q_pipe   = [q for q in common_q if q in q_pipe_sa]

speedup_orig = [cq / sa for cq, sa in zip(ms_cq_aligned, ms_orig_aligned)]
speedup_pipe = [ms_orig_cq[list(q_orig_cq).index(q)] / ms_pipe_sa[list(q_pipe_sa).index(q)]
                for q in common_q_pipe]

ax3.plot(common_q, speedup_orig,
         color=colors['orig_sa'], marker=markers['orig_sa'], linewidth=2,
         label='original structured_adjoint speedup')
ax3.plot(common_q_pipe, speedup_pipe,
         color=colors['pipe_sa'], marker=markers['pipe_sa'], linewidth=2,
         label='ring_cx_pipeline structured_adjoint speedup')
ax3.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='baseline (cuQuantum)')

ax3.set_xlabel('Number of Qubits', fontsize=12)
ax3.set_ylabel('Speedup (× faster than inverse_walk_cuQuantum)', fontsize=12)
ax3.set_title('Speedup over inverse_walk_cuQuantum', fontsize=12)
ax3.set_xticks(common_q)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)

plt.tight_layout()
out_path2 = f"{DIR}/comparison_speedup.png"
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
print(f"加速比图已保存到: {out_path2}")
