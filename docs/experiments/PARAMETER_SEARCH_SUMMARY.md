# 四个电路参数搜索与基准对比

> 生成时间：`2026-08-18T16:00:35.442092+00:00`

## 口径

本报告使用四个参数搜索 JSON 中每个 qubit 的**当前最快已完成组合**。搜索时间为完整 `forward + Hamiltonian + backward` 调用的 `median_ms`。
每个 qubit 同时保留该 JSON 的最快、最慢和平均搜索时间。

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

## 注意事项

- 搜索 JSON 中的 `failed_rows` 是编译资源限制或运行失败的候选，不参与最快、最慢和平均时间统计。
- PennyLane、cuQuantum 与 SAD 的 benchmark 均使用 float64 和相同 qubit 列表；除 MERA 使用其拓扑要求的 `ceil(log2(qubits))` layers 外，其余电路使用 8 layers。具体 warmup/steps 以各自 CSV 字段为准。
- 加速比大于 `1x` 表示搜索最优 SAD 更快；小于 `1x` 表示对应基准更快。
- 本报告只做时间比较，不重新验证能量/梯度正确性；正确性字段仍保留在原始 benchmark CSV 中。
