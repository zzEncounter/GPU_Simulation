# Lightning-GPU native 基线对比

## 方法

所有结果使用 RTX 6000 Ada、float64、seed 42，以及完全相同的参数和 Hamiltonian。原五种电路使用 8 layers，MERA 使用 ceil(log2(qubits))。`SAD native` 是优化后 CUDA/C++ 的三阶段同步时间之和；`Lightning native` 预构造 circuit、observable、OpsData、state vector 和 adjoint 对象，再通过 `lightning_gpu_ops` 测量同步后的 reset + forward + Hamiltonian + adjoint-gradient；`PennyLane QNode` 保留原 `qml.grad(qnode)` 端到端 wall time。

为消除大规模配置释放显存后回到小规模时的瞬态抖动，4–20q 使用 5 次 warmup，22–28q 使用 1 次 warmup；正式测量次数仍采用 20/10/5/3/2 的规模调度。

这个 packaged native binding 的 forward 仍然每个 gate 跨越一次 Python/nanobind 边界；它移除了 QNode、Autograd、transforms 和逐 step 序列化，但不是独立的纯 C++ Lightning 可执行程序。

## 汇总

| Circuit | SAD/native 范围 | SAD/QNode 范围 | 4q QNode/native | 28q QNode/native | native 最大梯度误差 |
|:--|--:|--:|--:|--:|--:|
| qaoa-ns | 5.20–25.94× | 5.14–36.53× | 1.37× | 0.91× | 6.66e-16 |

## 结论

- 4q 去掉 QNode/Autograd/序列化后，Lightning 缩短了 `1.37–1.37×`；但 SAD 相对低层 native 仍有 `25.91–25.91×`。因此原小规模高加速比只有一部分来自 PennyLane 框架。
- 4q 的 Lightning native backward 占总时间 `89.0%–89.0%`，固定开销主要位于通用 adjoint 路径，而不是 forward 或 Hamiltonian。
- 22q 以后 QNode/native 为 `0.91–0.95×`；两者已经同量级，少量反转来自分批运行、三阶段额外同步和测量波动，不应解释为低层接口在大规模上系统性更慢。

## qaoa-ns

| q | SAD ms | Lightning native ms | native fwd ms | native H ms | native bwd ms | vs native | QNode ms | vs QNode | QNode/native |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 0.230 | 5.970 | 0.377 | 0.283 | 5.348 | 25.91× | 8.187 | 35.53× | 1.37× |
| 6 | 0.286 | 7.354 | 0.539 | 0.325 | 6.519 | 25.67× | 10.244 | 35.76× | 1.39× |
| 8 | 0.343 | 8.901 | 0.746 | 0.374 | 7.833 | 25.94× | 12.537 | 36.53× | 1.41× |
| 10 | 0.474 | 10.307 | 0.936 | 0.419 | 9.001 | 21.74× | 14.639 | 30.88× | 1.42× |
| 12 | 0.534 | 11.782 | 1.119 | 0.466 | 10.267 | 22.05× | 16.894 | 31.61× | 1.43× |
| 14 | 0.612 | 13.496 | 1.364 | 0.520 | 11.748 | 22.04× | 19.424 | 31.72× | 1.44× |
| 16 | 0.943 | 16.535 | 1.653 | 0.735 | 14.176 | 17.53× | 23.285 | 24.69× | 1.41× |
| 18 | 2.449 | 23.404 | 2.663 | 1.133 | 19.690 | 9.55× | 30.114 | 12.29× | 1.29× |
| 20 | 9.057 | 47.056 | 6.335 | 2.619 | 38.212 | 5.20× | 53.131 | 5.87× | 1.13× |
| 22 | 37.264 | 305.689 | 21.224 | 21.303 | 263.324 | 8.20× | 288.919 | 7.75× | 0.95× |
| 24 | 159.900 | 1807.596 | 276.081 | 118.034 | 1413.185 | 11.30× | 1656.723 | 10.36× | 0.92× |
| 26 | 994.210 | 7742.902 | 1192.979 | 502.148 | 6047.774 | 7.79× | 7083.551 | 7.12× | 0.91× |
| 28 | 5862.718 | 33030.865 | 5096.243 | 2134.938 | 25799.684 | 5.63× | 30142.438 | 5.14× | 0.91× |

## 数值校验

- Native/QNode 最大 energy absolute error：`3.553e-15`。
- Native/QNode 最大 gradient-element absolute error：`6.661e-16`。
- SAD/native 最大 energy absolute error：`6.928e-14`。
- SAD/native 最大 gradient-element absolute error：`1.715e-14`。

机器可读数据：`benchmark/results/native_baseline_comparison.csv`。
