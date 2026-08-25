# Diagonal, fusion, reduction, and launch follow-up

Date: 2026-08-12. Hardware: RTX 6000 Ada, CUDA `sm_89`, complex float64
unless stated otherwise. Timings are medians and include native CUDA launch
cost in the end-to-end tables.

## 1. Decisions landed

- RX/RY, real-amplitude RA, phased SU2, and XXZ multi-phase kernels now use
  ordinary stream-ordered launches by default. Cooperative implementations
  remain behind `SAD_*_PERSISTENT=1` for reproducible ablation.
- A device only needs cooperative-launch support when one of those opt-in
  paths is compiled. The default executable no longer rejects other CUDA
  devices for that reason.
- Independent RZ/RZZ backward uses 64-thread CTAs; QAOA's single shared-gamma
  reduction remains 128 threads. Per-warp global atomics and hierarchical
  CTA reduction did not become defaults.
- Standalone ring CNOT uses forward scatter and adjoint gather.
- SU2 phased `RZ*RY` backward is retained as a disabled experiment because it
  is correct but 1.47--1.56x slower.
- XXZ now uses a dependency-preserving cross-matching schedule and the
  circuit-specific 10-bit forward / 8-bit backward tile at q>=20.
- Shared-angle QAOA uses a `q/2+1` domain-wall lookup rather than two chunk-code
  lookups. At q>=24, backward folds shared-gamma overlap and inverse cost into
  the final mixer-adjoint phase.

Raw/reproducible programs are in `benchmark/benchmark_persistent_gates.py`,
`benchmark/benchmark_diagonal_backward.py`,
`benchmark/benchmark_su2_phased_backward.py`,
`benchmark/benchmark_qaoa_compact_lookup.py`,
`benchmark/benchmark_cnot_end_to_end.py`, and
`benchmark/benchmark_xxz_full_circuit.py`.

## 2. Persistent kernels

Ordinary versus cooperative, 8-layer optimized full step:

| circuit | q | ordinary total improvement |
|---|---:|---:|
| RA | 20 / 24 / 26 / 28 | 3.0% / 10.4% / 9.7% / 23.2% |
| phased SU2 | 28 | 10.8% |
| XXZ | 20 / 24 / 26 / 28 | 0.8% / 9.8% / 12.7% / 16.1% |

The cooperative grid limits launched CTAs to simultaneous residency and pays
grid-wide synchronization. At large state sizes, allowing the scheduler to
run the full ordinary grid and using the kernel boundary for global order is
better. All energy differences were <=1.1e-16 and gradient differences were
<=4.7e-15.

## 3. RZ/RZZ forward and backward

### Forward

An RZ group, or one even/odd RZZ matching, is already one full-state pass.
Each amplitude builds an eigenvalue code for chunks of eight generators and
multiplies a 256-entry lookup factor. A complete RZZ layer can also be applied
as even+odd in one pass where scheduling permits.

Device `sincos` is not competitive. Existing isolated results at q24--28 show
the lookup path 2.5--3.1x faster. Host construction for q28 is about 0.035 ms
for a complete RZ group and 0.034 ms for both RZZ matchings. With optimizer
parameters changing every step this rebuild must be charged, but it remains
small next to the state pass. Chunk sizes 6--10 are close; eight bits is the
stable default. A 12-bit q24 RZZ exception saves about 0.07 ms in the kernel,
but grows an 8-layer RZ+RZZ table to about 2 MiB and host construction to
about 5.4 ms, so it is not the general choice.

### Backward and adjoint order

For `U(theta)=exp(-i theta G/2)`, the code accumulates

```text
dE/dtheta = Im <lambda | G | phi>
```

at the layer endpoint and applies `U^dagger` to both `phi` and `lambda`.
For a product of commuting diagonal RZ/ZZ gates, all generators commute with
the whole layer. Therefore all overlaps may be evaluated at the same endpoint
and the combined inverse factor applied once. The implementation loads one
`phi/lambda` pair, computes one base `Im(conj(lambda)*phi)`, signs it for every
Z/ZZ eigenvalue, accumulates all gradients, and inverse-evolves both states.
Generator overlap is thus fully applicable; diagonal generators need no
partner-amplitude load.

The useful choices are state-pass fusion and CTA shape, not one kernel per
gradient. Reading `phi/lambda` again per parameter would multiply memory
traffic by O(q).

## 4. Gradient reduction

The retained pattern is a thread-local sum over grid-stride amplitudes, then a
CTA reduction, then one `atomicAdd` per CTA/gradient. Results:

- independent multi-gradient RZ: 64 threads is 15--30% faster than 128 at
  q24--28;
- shared QAOA gamma: 128 threads is 18--22% faster than 64;
- combined `RZ + all ZZ` is insensitive because its fixed 64-thread kernel is
  dominated by evaluating 2q generators;
- per-warp diagonal atomics are within noise;
- replacing the CTA tree with warp-shuffle hierarchy slows XXZ backward by
  2--5% and RA backward by up to 11%;
- the earlier RX/RY sweep likewise found only a one-off ~1% result, not a
  general replacement.

The larger optimization is batching commuting generators in one state pass,
or fusing a reduction into an already-required pass. Micro-optimizing the
last CTA reduction rarely moves end-to-end time.

## 5. Cross-operation fusion

### RX/RY with diagonal and CNOT

Forward is valid when layer order is preserved: the final rotation phase can
multiply the complete RZ/RZZ factor and scatter directly to the CNOT output.
Backward gathers CNOT on the load, accumulates diagonal overlaps, applies the
inverse diagonal factor, then computes/inverts RX/RY phases. Each ZZ gradient
has an owner phase so it is accumulated exactly once. This is implemented in
the fused HEA paths.

Fusion is not universally faster: at q22--24 RZZ, a standalone combined
diagonal backward pass is better than carrying 2q diagonal partials through
every RX phase. The runtime therefore keeps the measured size boundary.

### Phased SU2 backward

The experimental kernel performs, for each qubit, RZ overlap -> inverse RZ ->
RY overlap -> inverse RY. Operators on different qubits commute, so distributing
these pairs across tile phases is algebraically correct. Full-gradient errors
were <=2.5e-15, but backward was 1.54x/1.47x/1.47x/1.56x slower at
q20/24/26/28. Per-qubit RZ reduction barriers and longer live ranges outweigh
removing the standalone diagonal pass. `SAD_SU2_PHASED_BACKWARD=0` remains the
default.

### XXZ same-bond and cross-bond

For one bond, XX, YY, and ZZ commute and are already one two-amplitude update;
splitting ZZ would add a state pass. Across bonds, overlapping even/odd gates
do not commute. The safe schedule is:

```text
per contiguous tile block: its even bonds, then its internal odd bonds
final phase: every odd bond crossing block boundaries
```

An internal odd bond only crosses disjoint even bonds from other blocks, so
this preserves the original `all even -> all odd` dependency order. Backward
reverses that schedule. With the tuned tile, cross matching adds 1.4--3.5%
over separate matchings; tile plus schedule improves full XXZ total time by
7.7--10.7% at q20--28. Energy errors are <=9e-17 and gradient errors <=8e-15.

### Shared-angle QAOA

There are two parameters per layer, not one: shared mixer beta and shared cost
gamma. All RX overlaps reduce into one beta scalar. The cost generator is
`sum_edges ZZ`; one amplitude contributes its base overlap times
`q - 2*domain_walls`, so backward produces one gamma scalar in one reduction.

The phase still depends on theta, so eliminating every lookup would require a
device `sincos`. Instead, the new table has only `q/2+1` reachable factors
indexed by half the domain-wall count (a closed binary ring always has an even
transition count). It saves about 60 KiB across eight layers and improves total
time over chunk lookup by 22.4%/2.7%/7.1%/6.5% for split compact execution at
q20/24/26/28. Folding the gamma reduction and inverse cost into the final
mixer-adjoint phase is selected at q>=24; it adds another ~1.7% at q24 and
6.3--6.7% at q26/28. At q20 the separate cost pass stays faster.

## 6. Directional ring CNOT

The ring's forward map preserves complex-amplitude pairs in 32-byte sectors:
consecutive input reads plus scatter writes are better coalesced than forward
gather. The inverse direction has the opposite property, so adjoint remains a
gather.

In complete 8-layer legacy paths, forward scatter improves RA forward by
5.7--6.7%, q24 SU2 by 5.0%, and q28 SU2 by 3.9%. Backward timing is unchanged
apart from measurement noise because it still uses gather. Optimized SU2
already fuses forward CNOT into the final RY write and gathers it in fused or
standalone backward, so no extra standalone launch remains there.

## 7. Parameter selection

The established HEA/RZZ/QAOA dispatch remains. XXZ q>=20 now selects
`f128r3_b32r3`: 128x8 forward (10-bit tile) and 32x8 backward (8-bit tile).
This separates XXZ's best geometry from the generic safe 128x4 library.

New compile-time A/B controls:

```text
SAD_{ROTATION,REAL,PHASED_RY,XXZ}_PERSISTENT=0
SAD_CNOT_FORWARD_SCATTER=1
SAD_DIAGONAL_BLOCK_THREADS=64
SAD_SHARED_DIAGONAL_BLOCK_THREADS=128
SAD_SU2_PHASED_BACKWARD=0
SAD_XXZ_CROSS_MATCHING=1
SAD_QAOA_COMPACT_LOOKUP=1
SAD_QAOA_FUSED_BACKWARD=1
```

The defaults are empirical choices for the measured Ada GPU, while each
semantic alternative remains available for architecture-specific retuning.
