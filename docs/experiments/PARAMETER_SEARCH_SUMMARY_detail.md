# 七个电路参数搜索与基准对比

> 生成时间：`2026-08-18T16:00:35.442092+00:00`

## 口径

本报告使用四个参数搜索 CSV/JSON 中每个 qubit 的**当前最快已完成组合**。搜索时间为完整 `forward + Hamiltonian + backward` 调用的 `median_ms`。
每个 qubit 同时保留该搜索结果的最快、最慢和平均搜索时间。最优候选的 `forward_phase_plan` 和 `backward_phase_plan` 取自原始 CSV。

加速比定义为：

```text
PennyLane 加速比 = PennyLane median time / 搜索最优 SAD median time
cuQuantum 加速比 = cuQuantum median time / 搜索最优 SAD median time
```

PennyLane 和 cuQuantum 时间来自 `benchmark/results/native_baseline_comparison_merged_cuquantum.csv`，时间单位统一为 ms。搜索 JSON 中的最快组合是候选筛选结果，不等同于已经写入生产 dispatch 的最终配置。

## 总览

| 电路 | 已完成 | 失败 | 搜索最优几何均值(ms) | PennyLane/SAD 几何均值 | cuQuantum/SAD 几何均值 |
|---|---:|---:|---:|---:|---:|
| MERA | 364 | 19 | 1.503 | 21.469x | 19.365x |
| Equivariant-QNN | 456 | 13 | 5.250 | 101.446x | 18.561x |
| Data Re-uploading | 547 | 39 | 4.070 | 31.555x | 24.254x |
| QAOA-NS | 458 | 13 | 3.761 | 23.424x | 19.694x |
| QAOA-BD | 13（SAD 非 BD） | — | 2.800 | 40.582x | 27.119x |
| QAOA-NS-BD | 13（SAD 非 BD） | — | 3.761 | 34.180x | 21.431x |
| XXV-HEV-BD (`xxz-hva-bd`) | 13（SAD 非 BD） | — | 6.754 | 55.872x | 36.999x |

## MERA

| qubits | 最快 ms | 平均 ms | 最慢 ms | 最快阶段 | PennyLane ms | PL/SAD | cuQuantum ms | cuQ/SAD |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 4 | 0.069 | 0.090 | 0.131 | `reduction` | 4.590 | 66.370x | 9.484 | 137.132x |
| 6 | 0.103 | 0.137 | 0.212 | `reduction` | 5.296 | 51.215x | 1.686 | 16.301x |
| 8 | 0.112 | 0.164 | 0.267 | `reduction` | 5.876 | 52.250x | 1.743 | 15.502x |
| 10 | 0.156 | 0.232 | 0.385 | `reduction` | 6.792 | 43.425x | 1.917 | 12.256x |
| 12 | 0.156 | 0.259 | 0.452 | `reduction` | 7.314 | 46.798x | 2.173 | 13.906x |
| 14 | 0.178 | 0.291 | 0.531 | `reduction` | 8.075 | 45.302x | 3.252 | 18.241x |
| 16 | 0.264 | 0.347 | 0.601 | `ordinary_threads` | 9.200 | 34.844x | 7.299 | 27.644x |
| 18 | 0.726 | 0.806 | 1.016 | `backward_shape` | 11.806 | 16.271x | 29.405 | 40.527x |
| 20 | 2.419 | 2.673 | 3.365 | `forward_shape` | 18.073 | 7.471x | 114.587 | 47.367x |
| 22 | 10.424 | 11.133 | 13.185 | `reduction` | 74.757 | 7.172x | 181.984 | 17.458x |
| 24 | 50.232 | 53.212 | 60.375 | `reduction` | 395.551 | 7.874x | 502.755 | 10.009x |
| 26 | 231.147 | 248.952 | 275.199 | `reduction` | 1714.363 | 7.417x | 1784.926 | 7.722x |
| 28 | 1012.003 | 1113.448 | 1310.221 | `ordinary_threads` | 7183.024 | 7.098x | 7067.472 | 6.984x |

最优配置（JSON 中的当前最快候选）：

| qubits | candidate key | 配置 |
|---:|---|---|
| 4 | `reduction:4:251ed2dd0266ac09` | `backward_register_bits=2, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=32, legacy_reduction=1, lookup_bits=8, mailbox_chunks=1, ordinary_threads=64, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 6 | `reduction:6:901433acac5ab3db` | `backward_register_bits=2, backward_threads=64, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=32, legacy_reduction=0, lookup_bits=8, mailbox_chunks=1, ordinary_threads=64, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 8 | `reduction:8:e5d15b61df59cc1d` | `backward_register_bits=2, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=64, legacy_reduction=0, lookup_bits=8, mailbox_chunks=1, ordinary_threads=64, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 10 | `reduction:10:68443d463d1d9ca4` | `backward_register_bits=2, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=64, legacy_reduction=0, lookup_bits=8, mailbox_chunks=1, ordinary_threads=64, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 12 | `reduction:12:6d7a5bb01345c3a7` | `backward_register_bits=2, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=32, legacy_reduction=0, lookup_bits=8, mailbox_chunks=1, ordinary_threads=64, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 14 | `reduction:14:b204e76d1e61be94` | `backward_register_bits=2, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=32, legacy_reduction=0, lookup_bits=8, mailbox_chunks=1, ordinary_threads=64, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 16 | `ordinary_threads:16:f6ded9e1b0049f26` | `backward_register_bits=2, backward_threads=64, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=64, legacy_reduction=1, lookup_bits=8, mailbox_chunks=1, ordinary_threads=128, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 18 | `backward_shape:18:a270924c9907f080` | `backward_register_bits=2, backward_threads=128, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=64, legacy_reduction=1, lookup_bits=8, mailbox_chunks=1, ordinary_threads=128, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 20 | `forward_shape:20:8348c5688d68645f` | `backward_register_bits=2, backward_threads=128, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=128, legacy_reduction=1, lookup_bits=8, mailbox_chunks=1, ordinary_threads=128, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 22 | `reduction:22:e8bda7da77296f09` | `backward_register_bits=2, backward_threads=64, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=64, legacy_reduction=0, lookup_bits=8, mailbox_chunks=1, ordinary_threads=128, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 24 | `reduction:24:fb4a300223f7d80c` | `backward_register_bits=3, backward_threads=64, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=256, legacy_reduction=0, lookup_bits=8, mailbox_chunks=1, ordinary_threads=128, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 26 | `reduction:26:1a804fb99a21bd21` | `backward_register_bits=3, backward_threads=64, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=256, legacy_reduction=0, lookup_bits=8, mailbox_chunks=1, ordinary_threads=64, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 28 | `ordinary_threads:28:7105f15eb1d2f22b` | `backward_register_bits=4, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=4, forward_threads=128, legacy_reduction=1, lookup_bits=8, mailbox_chunks=1, ordinary_threads=64, rotation_warp_atomic=0, shared_diagonal_threads=128` |

最优候选的 phase plan（原始 CSV）：

| qubits | forward/backward execution and phase plan |
|---:|---|
| 4 | `F:t32r2m1-C[] / B:t32r2m1-C[]` |
| 6 | `F:t32r2m1-C[] / B:t64r2m1-C[]` |
| 8 | `F:t64r2m1-C[] / B:t32r2m1-C[]` |
| 10 | `F:t64r2m1-C[] / B:t32r2m1-C[]` |
| 12 | `F:t32r2m1-C[] / B:t32r2m1-C[]` |
| 14 | `F:t32r2m1-C[] / B:t32r2m1-C[]` |
| 16 | `F:t64r2m1-C[] / B:t64r2m1-C[]` |
| 18 | `F:t64r2m1-C[] / B:t128r2m1-C[]` |
| 20 | `F:t128r2m1-C[] / B:t128r2m1-C[]` |
| 22 | `F:t64r2m1-C[] / B:t64r2m1-C[]` |
| 24 | `F:t256r2m1-C[] / B:t64r3m1-C[]` |
| 26 | `F:t256r2m1-C[] / B:t64r3m1-C[]` |
| 28 | `F:t128r4m1-C[] / B:t32r4m1-C[]` |

## Equivariant-QNN

| qubits | 最快 ms | 平均 ms | 最慢 ms | 最快阶段 | PennyLane ms | PL/SAD | cuQuantum ms | cuQ/SAD |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 4 | 0.218 | 0.330 | 0.611 | `reduction` | 19.006 | 87.156x | 9.484 | 43.492x |
| 6 | 0.263 | 0.420 | 0.810 | `reduction` | 26.690 | 101.616x | 2.734 | 10.410x |
| 8 | 0.333 | 0.526 | 0.949 | `reduction` | 43.438 | 130.413x | 3.349 | 10.053x |
| 10 | 0.476 | 0.692 | 1.245 | `reduction` | 64.453 | 135.357x | 4.143 | 8.701x |
| 12 | 0.540 | 0.840 | 1.579 | `reduction` | 90.030 | 166.678x | 5.839 | 10.810x |
| 14 | 0.592 | 0.947 | 1.831 | `backward_shape` | 133.225 | 224.981x | 13.028 | 22.001x |
| 16 | 1.065 | 1.328 | 2.254 | `backward_shape` | 188.824 | 177.242x | 41.617 | 39.065x |
| 18 | 2.974 | 3.206 | 3.851 | `backward_shape` | 262.657 | 88.329x | 170.634 | 57.383x |
| 20 | 10.880 | 11.679 | 13.951 | `forward_phase` | 447.624 | 41.141x | 739.626 | 67.979x |
| 22 | 42.886 | 45.806 | 53.518 | `forward_phase` | 2201.396 | 51.331x | 971.633 | 22.656x |
| 24 | 188.932 | 205.549 | 244.354 | `ordinary_threads` | 15161.129 | 80.247x | 2256.785 | 11.945x |
| 26 | 788.936 | 880.173 | 1022.659 | `forward_phase` | 69524.918 | 88.125x | 7494.541 | 9.500x |
| 28 | 3597.894 | 4056.981 | 4667.517 | `reduction` | 316341.466 | 87.924x | 30261.973 | 8.411x |

最优配置（JSON 中的当前最快候选）：

| qubits | candidate key | 配置 |
|---:|---|---|
| 4 | `reduction:4:4c30b10d3c87672b` | `backward_register_bits=2, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=32, legacy_reduction=0, lookup_bits=8, mailbox_chunks=1, ordinary_threads=64, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 6 | `reduction:6:daf41452e4f67830` | `backward_register_bits=2, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=32, legacy_reduction=0, lookup_bits=8, mailbox_chunks=2, ordinary_threads=64, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 8 | `reduction:8:255202c2e45e28e6` | `backward_register_bits=2, backward_threads=64, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=64, legacy_reduction=0, lookup_bits=8, mailbox_chunks=2, ordinary_threads=64, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 10 | `reduction:10:241a92df91e3a8bc` | `backward_register_bits=2, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=32, legacy_reduction=0, lookup_bits=8, mailbox_chunks=1, ordinary_threads=64, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 12 | `reduction:12:f90f70e5e7b81321` | `backward_register_bits=2, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=32, legacy_reduction=0, lookup_bits=8, mailbox_chunks=2, ordinary_threads=64, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 14 | `backward_shape:14:ef2a353af8272811` | `backward_register_bits=2, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=32, legacy_reduction=1, lookup_bits=8, mailbox_chunks=1, ordinary_threads=128, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 16 | `backward_shape:16:86e3d98bb78998bf` | `backward_register_bits=2, backward_threads=64, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=64, legacy_reduction=1, lookup_bits=8, mailbox_chunks=1, ordinary_threads=128, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 18 | `backward_shape:18:97fa6c41136e5245` | `backward_register_bits=2, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=128, legacy_reduction=1, lookup_bits=8, mailbox_chunks=1, ordinary_threads=128, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 20 | `forward_phase:20:41eb001a16bd3eb2` | `backward_register_bits=2, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=3, forward_threads=32, legacy_reduction=1, lookup_bits=8, mailbox_chunks=1, ordinary_threads=128, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 22 | `forward_phase:22:76befb10c17b2285` | `backward_register_bits=3, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=3, forward_threads=64, legacy_reduction=1, lookup_bits=8, mailbox_chunks=1, ordinary_threads=128, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 24 | `ordinary_threads:24:c23010263bce021c` | `backward_register_bits=4, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=3, forward_threads=32, legacy_reduction=1, lookup_bits=8, mailbox_chunks=2, ordinary_threads=64, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 26 | `forward_phase:26:39e3665d92d1ace7` | `backward_register_bits=4, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=4, forward_threads=32, legacy_reduction=1, lookup_bits=8, mailbox_chunks=1, ordinary_threads=128, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 28 | `reduction:28:a8d32fc9fa083778` | `backward_register_bits=3, backward_threads=128, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=3, forward_threads=128, legacy_reduction=0, lookup_bits=8, mailbox_chunks=2, ordinary_threads=64, rotation_warp_atomic=1, shared_diagonal_threads=128` |

最优候选的 phase plan（原始 CSV）：

| qubits | forward/backward execution and phase plan |
|---:|---|
| 4 | `F:t32r2m1-C[] / B:t32r2m1-C[]` |
| 6 | `F:t32r2m2-C[] / B:t32r2m2-C[]` |
| 8 | `F:t64r2m2-C[] / B:t64r2m2-C[]` |
| 10 | `F:t32r2m1-C[L5R2W0-L3R0W0] / B:t32r2m1-C[L5R2W0-L3R0W0]` |
| 12 | `F:t32r2m2-C[L5R2W0-L5R0W0] / B:t32r2m2-C[L5R2W0-L5R0W0]` |
| 14 | `F:t32r2m1-C[] / B:t32r2m1-C[]` |
| 16 | `F:t64r2m1-C[] / B:t64r2m1-C[]` |
| 18 | `F:t128r2m1-C[] / B:t32r2m1-C[]` |
| 20 | `F:t32r3m1-C[L5R3W0-L5R3W0-L4R0W0] / B:t32r2m1-C[]` |
| 22 | `F:t64r3m1-C[L5R3W1-L5R3W1-L4R0W0] / B:t32r3m1-C[]` |
| 24 | `F:t32r3m2-C[] / B:t32r4m2-C[]` |
| 26 | `F:t32r4m1-C[] / B:t32r4m1-C[]` |
| 28 | `F:t128r3m2-C[] / B:t128r3m2-C[]` |

## Data Re-uploading

| qubits | 最快 ms | 平均 ms | 最慢 ms | 最快阶段 | PennyLane ms | PL/SAD | cuQuantum ms | cuQ/SAD |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 4 | 0.212 | 0.260 | 0.391 | `diagonal_reduction` | 11.054 | 52.058x | 9.484 | 44.662x |
| 6 | 0.243 | 0.328 | 0.535 | `diagonal_reduction` | 14.554 | 59.888x | 2.709 | 11.149x |
| 8 | 0.304 | 0.410 | 0.639 | `diagonal_reduction` | 18.341 | 60.324x | 3.366 | 11.070x |
| 10 | 0.377 | 0.532 | 0.878 | `diagonal_reduction` | 21.843 | 57.966x | 4.052 | 10.754x |
| 12 | 0.427 | 0.597 | 1.000 | `diagonal_reduction` | 25.396 | 59.491x | 6.033 | 14.132x |
| 14 | 0.510 | 0.679 | 1.172 | `rotation_reduction` | 29.314 | 57.438x | 13.056 | 25.582x |
| 16 | 0.874 | 1.015 | 1.488 | `backward_shape` | 34.836 | 39.839x | 41.286 | 47.215x |
| 18 | 2.080 | 2.223 | 2.511 | `backward_shape` | 45.076 | 21.673x | 170.072 | 81.774x |
| 20 | 6.781 | 7.289 | 7.935 | `rotation_reduction` | 74.818 | 11.033x | 737.153 | 108.702x |
| 22 | 29.746 | 30.961 | 34.476 | `lookup_bits` | 412.149 | 13.856x | 981.864 | 33.009x |
| 24 | 132.002 | 139.411 | 160.329 | `ordinary_threads` | 2426.496 | 18.382x | 2355.426 | 17.844x |
| 26 | 550.000 | 599.393 | 691.085 | `ordinary_threads` | 10375.900 | 18.865x | 7949.217 | 14.453x |
| 28 | 2451.255 | 2657.534 | 2965.586 | `rotation_reduction` | 44212.000 | 18.036x | 32170.375 | 13.124x |

最优配置（JSON 中的当前最快候选）：

| qubits | candidate key | 配置 |
|---:|---|---|
| 4 | `diagonal_reduction:4:05f6567ba62ead1d` | `backward_register_bits=2, backward_threads=32, diagonal_threads=128, diagonal_warp_atomic=1, forward_register_bits=2, forward_threads=32, legacy_reduction=0, lookup_bits=6, mailbox_chunks=2, ordinary_threads=64, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 6 | `diagonal_reduction:6:79072408cc491324` | `backward_register_bits=2, backward_threads=32, diagonal_threads=128, diagonal_warp_atomic=1, forward_register_bits=2, forward_threads=32, legacy_reduction=0, lookup_bits=8, mailbox_chunks=2, ordinary_threads=128, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 8 | `diagonal_reduction:8:146c82b4ac6f4434` | `backward_register_bits=2, backward_threads=64, diagonal_threads=128, diagonal_warp_atomic=1, forward_register_bits=2, forward_threads=64, legacy_reduction=0, lookup_bits=8, mailbox_chunks=1, ordinary_threads=64, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 10 | `diagonal_reduction:10:f11d1bc76b1bbd55` | `backward_register_bits=2, backward_threads=32, diagonal_threads=128, diagonal_warp_atomic=1, forward_register_bits=2, forward_threads=32, legacy_reduction=0, lookup_bits=10, mailbox_chunks=2, ordinary_threads=64, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 12 | `diagonal_reduction:12:30c73f4018b0a8e6` | `backward_register_bits=2, backward_threads=32, diagonal_threads=128, diagonal_warp_atomic=1, forward_register_bits=2, forward_threads=32, legacy_reduction=0, lookup_bits=8, mailbox_chunks=2, ordinary_threads=128, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 14 | `rotation_reduction:14:ab0aba9d58f02010` | `backward_register_bits=2, backward_threads=32, diagonal_threads=128, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=32, legacy_reduction=0, lookup_bits=10, mailbox_chunks=2, ordinary_threads=128, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 16 | `backward_shape:16:86e3d98bb78998bf` | `backward_register_bits=2, backward_threads=64, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=64, legacy_reduction=1, lookup_bits=8, mailbox_chunks=1, ordinary_threads=128, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 18 | `backward_shape:18:97fa6c41136e5245` | `backward_register_bits=2, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=128, legacy_reduction=1, lookup_bits=8, mailbox_chunks=1, ordinary_threads=128, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 20 | `rotation_reduction:20:3e8dacbb7a834252` | `backward_register_bits=2, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=64, legacy_reduction=1, lookup_bits=10, mailbox_chunks=1, ordinary_threads=128, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 22 | `lookup_bits:22:746a4a162ccb9571` | `backward_register_bits=3, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=3, forward_threads=64, legacy_reduction=1, lookup_bits=8, mailbox_chunks=1, ordinary_threads=128, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 24 | `ordinary_threads:24:83a67d1338b84f27` | `backward_register_bits=4, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=3, forward_threads=32, legacy_reduction=1, lookup_bits=10, mailbox_chunks=2, ordinary_threads=64, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 26 | `ordinary_threads:26:a5f8234455a9104c` | `backward_register_bits=4, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=4, forward_threads=32, legacy_reduction=1, lookup_bits=10, mailbox_chunks=1, ordinary_threads=64, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 28 | `rotation_reduction:28:d78f536a1bd6ecac` | `backward_register_bits=3, backward_threads=128, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=3, forward_threads=128, legacy_reduction=0, lookup_bits=10, mailbox_chunks=2, ordinary_threads=64, rotation_warp_atomic=1, shared_diagonal_threads=128` |

最优候选的 phase plan（原始 CSV）：

| qubits | forward/backward execution and phase plan |
|---:|---|
| 4 | `F:t32r2m2-C[] / B:t32r2m2-C[]` |
| 6 | `F:t32r2m2-C[] / B:t32r2m2-C[]` |
| 8 | `F:t64r2m1-C[] / B:t64r2m1-C[]` |
| 10 | `F:t32r2m2-C[] / B:t32r2m2-C[]` |
| 12 | `F:t32r2m2-C[] / B:t32r2m2-C[]` |
| 14 | `F:t32r2m2-C[L5R2W0-L5R2W0] / B:t32r2m2-C[L5R2W0-L5R2W0]` |
| 16 | `F:t64r2m1-C[] / B:t64r2m1-C[]` |
| 18 | `F:t128r2m1-C[] / B:t32r2m1-C[]` |
| 20 | `F:t64r2m1-C[] / B:t32r2m1-X[L5R2W0-L0R2W0-L0R2W0-L0R2W0-L0R2W0-L0R2W0-L0R2W0-L0R1W0]` |
| 22 | `F:t64r3m1-C[L5R3W1-L5R3W1-L4R0W0] / B:t32r3m1-C[L5R3W0-L5R3W0-L5R1W0]` |
| 24 | `F:t32r3m2-C[] / B:t32r4m2-C[]` |
| 26 | `F:t32r4m1-C[] / B:t32r4m1-C[]` |
| 28 | `F:t128r3m2-C[] / B:t128r3m2-C[]` |

## QAOA-NS

| qubits | 最快 ms | 平均 ms | 最慢 ms | 最快阶段 | PennyLane ms | PL/SAD | cuQuantum ms | cuQ/SAD |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 4 | 0.159 | 0.200 | 0.314 | `backward_shape` | 8.187 | 51.635x | 9.483 | 59.807x |
| 6 | 0.197 | 0.269 | 0.451 | `reduction` | 10.244 | 52.046x | 2.383 | 12.107x |
| 8 | 0.259 | 0.335 | 0.574 | `reduction` | 12.537 | 48.337x | 2.872 | 11.075x |
| 10 | 0.347 | 0.457 | 0.733 | `reduction` | 14.639 | 42.194x | 3.389 | 9.767x |
| 12 | 0.381 | 0.541 | 0.880 | `backward_shape` | 16.894 | 44.337x | 4.637 | 12.170x |
| 14 | 0.444 | 0.646 | 1.467 | `reduction` | 19.424 | 43.752x | 9.367 | 21.100x |
| 16 | 0.788 | 0.999 | 1.741 | `reduction` | 23.285 | 29.549x | 29.905 | 37.950x |
| 18 | 2.147 | 2.329 | 2.861 | `reduction` | 30.114 | 14.028x | 114.773 | 53.465x |
| 20 | 7.148 | 8.027 | 8.921 | `ordinary_threads` | 53.131 | 7.433x | 494.210 | 69.139x |
| 22 | 29.366 | 31.766 | 35.926 | `reduction` | 288.919 | 9.839x | 669.742 | 22.807x |
| 24 | 128.700 | 137.060 | 156.304 | `ordinary_threads` | 1656.723 | 12.873x | 1594.003 | 12.385x |
| 26 | 554.474 | 601.069 | 676.643 | `ordinary_threads` | 7083.551 | 12.775x | 5360.638 | 9.668x |
| 28 | 2499.321 | 2703.351 | 2998.348 | `forward_phase` | 30142.438 | 12.060x | 21754.932 | 8.704x |

最优配置（JSON 中的当前最快候选）：

| qubits | candidate key | 配置 |
|---:|---|---|
| 4 | `backward_shape:4:441cac79afd5021b` | `backward_register_bits=2, backward_threads=64, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=32, legacy_reduction=1, lookup_bits=8, mailbox_chunks=1, ordinary_threads=128, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 6 | `reduction:6:5c86136e09d9cca4` | `backward_register_bits=2, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=32, legacy_reduction=0, lookup_bits=8, mailbox_chunks=2, ordinary_threads=128, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 8 | `reduction:8:283c78474e1fe897` | `backward_register_bits=2, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=64, legacy_reduction=0, lookup_bits=8, mailbox_chunks=2, ordinary_threads=64, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 10 | `reduction:10:e0a2e0146bb8a59b` | `backward_register_bits=2, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=32, legacy_reduction=0, lookup_bits=8, mailbox_chunks=2, ordinary_threads=64, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 12 | `backward_shape:12:ec98b1357c4936e6` | `backward_register_bits=2, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=32, legacy_reduction=1, lookup_bits=8, mailbox_chunks=1, ordinary_threads=128, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 14 | `reduction:14:90f91a6b0c54c2fe` | `backward_register_bits=2, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=32, legacy_reduction=0, lookup_bits=8, mailbox_chunks=2, ordinary_threads=128, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 16 | `reduction:16:20640feb9b2aec88` | `backward_register_bits=3, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=64, legacy_reduction=0, lookup_bits=8, mailbox_chunks=2, ordinary_threads=64, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 18 | `reduction:18:8d125c4f4b661532` | `backward_register_bits=2, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=256, legacy_reduction=0, lookup_bits=8, mailbox_chunks=1, ordinary_threads=64, rotation_warp_atomic=1, shared_diagonal_threads=128` |
| 20 | `ordinary_threads:20:9cb5954f1eb099b8` | `backward_register_bits=2, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=64, legacy_reduction=1, lookup_bits=8, mailbox_chunks=1, ordinary_threads=64, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 22 | `reduction:22:4a30d9337f397983` | `backward_register_bits=3, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=2, forward_threads=128, legacy_reduction=1, lookup_bits=8, mailbox_chunks=1, ordinary_threads=64, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 24 | `ordinary_threads:24:6793fd25a1b89f4e` | `backward_register_bits=4, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=3, forward_threads=32, legacy_reduction=1, lookup_bits=8, mailbox_chunks=1, ordinary_threads=64, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 26 | `ordinary_threads:26:e735f5b6a4d84f58` | `backward_register_bits=4, backward_threads=32, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=4, forward_threads=32, legacy_reduction=1, lookup_bits=8, mailbox_chunks=1, ordinary_threads=64, rotation_warp_atomic=0, shared_diagonal_threads=128` |
| 28 | `forward_phase:28:ed4b4f5bc2916047` | `backward_register_bits=3, backward_threads=128, diagonal_threads=64, diagonal_warp_atomic=0, forward_register_bits=3, forward_threads=128, legacy_reduction=1, lookup_bits=8, mailbox_chunks=1, ordinary_threads=128, rotation_warp_atomic=0, shared_diagonal_threads=128` |

最优候选的 phase plan（原始 CSV）：

| qubits | forward/backward execution and phase plan |
|---:|---|
| 4 | `F:t32r2m1-C[] / B:t64r2m1-C[]` |
| 6 | `F:t32r2m2-C[] / B:t32r2m2-C[]` |
| 8 | `F:t64r2m2-C[] / B:t32r2m2-C[L5R2W0-L1R0W0]` |
| 10 | `F:t32r2m2-C[] / B:t32r2m2-C[]` |
| 12 | `F:t32r2m1-C[] / B:t32r2m1-C[]` |
| 14 | `F:t32r2m2-C[L5R2W0-L5R2W0] / B:t32r2m2-C[L5R2W0-L5R2W0]` |
| 16 | `F:t64r2m2-C[L5R2W1-L5R2W1] / B:t32r3m2-C[]` |
| 18 | `F:t256r2m1-C[] / B:t32r2m1-C[L5R2W0-L5R2W0-L4R0W0]` |
| 20 | `F:t64r2m1-C[L5R2W1-L5R2W1-L4R0W0] / B:t32r2m1-C[]` |
| 22 | `F:t128r2m1-C[L5R2W2-L5R2W2-L4R0W0] / B:t32r3m1-C[]` |
| 24 | `F:t32r3m1-C[] / B:t32r4m1-C[]` |
| 26 | `F:t32r4m1-C[] / B:t32r4m1-C[]` |
| 28 | `F:t128r3m1-C[] / B:t128r3m1-C[]` |

## QAOA-BD

BD 的 PennyLane/cuQuantum 时间来自 BD benchmark；SAD 分母沿用标准 QAOA 的非 BD 参数搜索最优结果（`parameter_search_experiment.csv`）。

| qubits | SAD 最快 ms | SAD 平均 ms | SAD 最差 ms | PennyLane BD ms | PL/SAD | cuQuantum BD ms | cuQ/SAD |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.116679 | 0.154171 | 0.189591 | 10.280 | 88.105x | 2.390 | 20.488x |
| 6 | 0.139045 | 0.192209 | 0.241476 | 13.352 | 96.026x | 2.859 | 20.563x |
| 8 | 0.190413 | 0.242655 | 0.304479 | 16.649 | 87.439x | 3.483 | 18.293x |
| 10 | 0.250435 | 0.333838 | 0.393694 | 19.817 | 79.131x | 4.230 | 16.890x |
| 12 | 0.277477 | 0.382866 | 0.455946 | 22.935 | 82.657x | 5.582 | 20.116x |
| 14 | 0.310935 | 0.433453 | 0.517022 | 26.469 | 85.126x | 9.957 | 32.024x |
| 16 | 0.513046 | 0.566115 | 0.639762 | 31.146 | 60.708x | 29.542 | 57.582x |
| 18 | 1.360837 | 1.476269 | 1.622996 | 40.379 | 29.672x | 116.057 | 85.283x |
| 20 | 5.137266 | 5.517345 | 6.380971 | 67.288 | 13.098x | 496.027 | 96.555x |
| 22 | 20.683093 | 23.066809 | 26.671612 | 336.492 | 16.269x | 711.614 | 34.406x |
| 24 | 96.010000 | 105.319945 | 120.232843 | 2063.726 | 21.495x | 2004.362 | 20.877x |
| 26 | 404.283501 | 454.332817 | 511.502473 | 8769.759 | 21.692x | 7103.134 | 17.570x |
| 28 | 1754.224946 | 2139.237178 | 2476.285594 | 37204.773 | 21.209x | 28967.513 | 16.513x |

表中 SAD 最快、平均、最差分别按每个 qubit 的非 BD 搜参成功候选计算；来源为 `parameter_selection_stages_raw.csv`。

非 BD SAD 最优 phase plan：4q `F:t32r2m1-C[4] / B:t32r2m1-C[4]`；6q `F:t32r2m1-C[6] / B:t32r2m1-C[6]`；8q `F:t64r2m1-C[8] / B:t64r2m1-C[8]`；10q `F:t32r2m1-C[L2R2W0-L4R2W0] / B:t32r2m1-C[L4R2W0-L2R2W0]`；12q `F:t32r2m1-C[7+5] / B:t32r2m1-C[7+5]`；14q `F:t32r2m1-C[7+7] / B:t32r2m1-C[7+7]`；16q `F:t64r2m1-C[8+8] / B:t64r2m1-C[8+8]`；18q `F:t128r2m1-C[9+9] / B:t32r2m1-C[7+7+4]`；20q `F:t128r2m1-C[9+9+2] / B:t128r2m1-C[9+9+2]`；22q `F:t128r2m1-C[9+9+4] / B:t128r2m1-C[9+9+4]`；24q `F:t128r2m1-C[9+9+6] / B:t128r2m1-C[9+9+6]`；26q `F:t128r2m1-C[9+9+8] / B:t128r3m1-C[10+10+6]`；28q `F:t64r4m1-C[10+10+8] / B:t128r3m1-C[10+10+8]`。

## QAOA-NS-BD

BD 的 PennyLane/cuQuantum 时间来自 `pennylane-cuQuantum-qaoa-ns-bd.csv`；SAD 分母沿用 QAOA-NS 非 BD 参数搜索最优结果（`qaoa_ns_parameter_search.json`）。

| qubits | SAD 最快 ms | SAD 平均 ms | SAD 最差 ms | PennyLane BD ms | PL/SAD | cuQuantum BD ms | cuQ/SAD |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.158557 | 0.200253 | 0.314266 | 11.597 | 73.141x | 2.651 | 16.721x |
| 6 | 0.196820 | 0.269055 | 0.451018 | 15.314 | 77.810x | 3.027 | 15.382x |
| 8 | 0.259367 | 0.335371 | 0.573557 | 18.984 | 73.195x | 3.600 | 13.879x |
| 10 | 0.346953 | 0.457376 | 0.733151 | 23.228 | 66.949x | 4.372 | 12.600x |
| 12 | 0.381030 | 0.540965 | 0.879637 | 26.499 | 69.546x | 5.598 | 14.692x |
| 14 | 0.443954 | 0.646017 | 1.467445 | 30.616 | 68.963x | 10.579 | 23.828x |
| 16 | 0.788017 | 0.998848 | 1.740860 | 36.323 | 46.094x | 31.415 | 39.866x |
| 18 | 2.146693 | 2.328691 | 2.861047 | 47.321 | 22.044x | 123.162 | 57.373x |
| 20 | 7.148092 | 8.027088 | 8.920647 | 75.934 | 10.623x | 528.697 | 73.963x |
| 22 | 29.365608 | 31.766497 | 35.925837 | 364.274 | 12.405x | 769.435 | 26.202x |
| 24 | 128.699919 | 137.060121 | 156.303605 | 2163.367 | 16.809x | 2144.732 | 16.665x |
| 26 | 554.473542 | 601.068623 | 676.643027 | 9438.589 | 17.023x | 7626.885 | 13.755x |
| 28 | 2499.320507 | 2703.350855 | 2998.347739 | 42381.135 | 16.957x | 31426.512 | 12.574x |

表中 SAD 最快、平均、最差分别按每个 qubit 的 458 个非 BD 搜参成功候选计算；来源为 `qaoa_ns_parameter_search.csv`。

非 BD SAD phase plan 与上面的 QAOA-NS 章节相同；BD 仅替换外部 PennyLane/cuQuantum 基准实现。

## XXV-HEV-BD

原始电路名为 `xxz-hva-bd`。BD 的 PennyLane/cuQuantum 时间来自 `pennylane-cuQuantum-xxz-hva-bd.csv`；SAD 分母沿用 `xxz-hva` 非 BD 参数搜索最优结果（`parameter_search_experiment.csv`）。

| qubits | SAD 最快 ms | SAD 平均 ms | SAD 最差 ms | PennyLane BD ms | PL/SAD | cuQuantum BD ms | cuQ/SAD |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.178357 | 0.199321 | 0.211989 | 26.922 | 150.947x | 6.338 | 35.538x |
| 6 | 0.247893 | 0.280540 | 0.298032 | 38.597 | 155.702x | 8.719 | 35.173x |
| 8 | 0.435216 | 0.464535 | 0.483364 | 51.287 | 117.842x | 12.173 | 27.970x |
| 10 | 0.506906 | 0.758006 | 0.915994 | 62.755 | 123.801x | 15.271 | 30.126x |
| 12 | 0.575432 | 0.897179 | 1.131954 | 73.839 | 128.319x | 19.677 | 34.195x |
| 14 | 0.736891 | 1.046418 | 1.240530 | 87.289 | 118.456x | 29.277 | 39.730x |
| 16 | 1.239013 | 1.400671 | 1.631391 | 103.781 | 83.761x | 65.191 | 52.615x |
| 18 | 3.882914 | 4.232557 | 4.489726 | 135.717 | 34.952x | 225.194 | 57.996x |
| 20 | 16.001269 | 16.799409 | 18.856042 | 253.105 | 15.818x | 912.741 | 57.042x |
| 22 | 62.779950 | 67.590443 | 76.359450 | 1283.797 | 20.449x | 2373.518 | 37.807x |
| 24 | 270.453102 | 305.021570 | 332.992942 | 7283.260 | 26.930x | 10347.041 | 38.258x |
| 26 | 1186.831244 | 1342.168238 | 1462.695482 | 31926.437 | 26.901x | 42331.277 | 35.667x |
| 28 | 5154.060212 | 5769.290021 | 6218.633554 | 137180.232 | 26.616x | 179781.920 | 34.882x |

表中 SAD 最快、平均、最差分别按每个 qubit 的非 BD 搜参成功候选计算；来源为 `parameter_selection_stages_raw.csv`。

非 BD SAD phase plan：4q `F:t32r2m1-Zx2 / B:t32r2m1-Zx2`；6q `F:t32r2m1-Zsep / B:t32r2m1-Zsep`；8q `F:t64r2m1-Zx2 / B:t64r2m1-Zx2`；10q `F:t32r2m1-Zx3 / B:t32r2m1-Zx3`；12q `F:t32r2m1-Zx3 / B:t32r2m1-Zx3`；14q `F:t32r2m1-Zx4 / B:t32r2m1-Zx4`；16q `F:t64r2m1-Zx3 / B:t64r2m1-Zx3`；18q `F:t32r2m1-Zx4 / B:t32r2m1-Zx4`；20q `F:t128r2m1-Zx4 / B:t128r2m1-Zx4`；22–24q `F:t128r3m1-Zx4 / B:t32r3m1-Zx4`；26–28q `F:t128r3m1-Zx4 / B:t32r3m1-Zx5`。

## 注意事项

- 搜索 JSON 中的 `failed_rows` 是编译资源限制或运行失败的候选，不参与最快、最慢和平均时间统计。
- phase plan 表按原始 CSV 中每个 qubit 的最低 `median_ms` 且 `status=ok` 的候选生成。`C[]` 表示对应 CSV 字段为空，运行时通过 `build_phase_maps(...)` 自动生成规范 phase 映射，并非不执行 phase 映射。
- 组合编码为 `F:t<threads>r<register_bits>m<mailbox_chunks>-C[<phase_plan>] / B:t<threads>r<register_bits>m<mailbox_chunks>-C[<phase_plan>]`；`F`/`B` 分别是 forward/backward，`C` 表示 compact、`X` 表示 fixed-low-5，`L`、`R`、`W` 分别表示 phase 中的 lane、register、warp target 数量。
- PennyLane、cuQuantum 与 SAD 的 benchmark 均使用 float64 和相同 qubit 列表；除 MERA 使用其拓扑要求的 `ceil(log2(qubits))` layers 外，其余电路使用 8 layers。具体 warmup/steps 以各自 CSV 字段为准。
- 加速比大于 `1x` 表示搜索最优 SAD 更快；小于 `1x` 表示对应基准更快。
- 本报告只做时间比较，不重新验证能量/梯度正确性；正确性字段仍保留在原始 benchmark CSV 中。
