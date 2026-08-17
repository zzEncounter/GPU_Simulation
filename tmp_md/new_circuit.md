可以。针对你这个 **adjoint differentiation 性能 benchmark**，我会优先选下面 3 类，而且我建议的优先级就是：

**MERA > (S_n)-equivariant QNN > Data Re-uploading QNN**。

关键不是它们的应用名字不同，而是它们分别给 adjoint 带来了三种真正不同的计算结构：

| Circuit                   | 与 HEA 最大区别             |                        参数结构 | Adjoint 适配度 |      我建议 |
| ------------------------- | ---------------------- | --------------------------: | ----------: | -------: |
| **MERA / TTN**            | 层次化、active qubits 逐层减少 |                      (O(n)) |       ★★★★★ | **强烈加入** |
| **(S_n)-Equivariant QNN** | 参数共享 + all-to-all ZZ   | 很少的逻辑参数，大量 gate occurrences |       ★★★★★ | **强烈加入** |
| **Data Re-uploading**     | 数据反复插入每层               |                     (O(nL)) |       ★★★★★ |     推荐加入 |

这里“adjoint 适配”指的是：它们都可以保持为纯 unitary 演化并最终计算 expectation value，因此可以直接放进 statevector adjoint 的

[
|\psi\rangle=U(\theta)|0\rangle,\qquad
f(\theta)=\langle\psi|H|\psi\rangle
]

框架。PennyLane `lightning.gpu` 本身目前也直接支持 adjoint differentiation。需要注意的是：**我不是说这些论文原本就是用 adjoint differentiation 做训练，而是说这些 circuit 对 adjoint 是天然兼容的。** ([PennyLane][1])

---

## 1. MERA：我最推荐加入的“非 HEA”结构

我甚至会把 **MERA 而不是 TTN** 放进最终 benchmark。

Grant 等人的 *Hierarchical quantum classifiers* 明确研究了两种层次结构：

[
\mathrm{TTN}
\quad\text{和}\quad
\mathrm{MERA}.
]

TTN 每一层把相邻 qubit 两两经过一个二比特 unitary，然后只保留其中一个 qubit 继续进入下一层，因此 active qubits 是

[
n\rightarrow \frac n2\rightarrow \frac n4
\rightarrow\cdots\rightarrow1.
]

MERA 则在每次 TTN coarse-graining 之前，再加入跨相邻 branch 的二比特 **disentangler (D_i)**。原论文最后也是在唯一剩余的 qubit 上计算一个 observable 的 expectation value。([Nature][2])

### 一个 8-qubit TTN 可以理解成

```text
q0 ──U──x
q1 ──┘────────U──x
               │
q2 ──U──x      │
q3 ──┘─────────┘────────U──x
                         │
q4 ──U──x                │
q5 ──┘────────U──x       │
               │         │
q6 ──U──x      │         │
q7 ──┘─────────┘─────────┘── <Z>
```

`x` 不是 measurement，而是“这个 wire 后面不再参与”。

MERA 就是在这些 coarse-graining block 前再加入：

```text
       D            D
──────■────────────■────
      │            │
──────■────U───────■────U── ...
```

把相邻 branch 先 disentangle，再压缩。

### 二比特 block 怎么实现？

Grant 等人的简单参数化恰好非常适合你的 simulator：

```text
q_i ──RY(θ0)──●────
              │
q_j ──RY(θ1)──X────
```

即两个任意 single-qubit rotation 加一个固定 CNOT；对于 real-valued 参数化，他们直接使用 (R_Y)。论文也测试了更一般的 (SU(4)) 二比特 unitary。([Nature][2])

因此我建议第一版就用：

[
U_{ij}(\theta_a,\theta_b)
=========================

\operatorname{CNOT}_{i,j}
R_Y^{(j)}(\theta_b)
R_Y^{(i)}(\theta_a).
]

`D` block 也使用同样的二参数 block 即可。

### 为什么它特别适合你的 adjoint benchmark？

这个很重要。原论文描述 TTN 时说“discard”一个 output qubit，但**statevector simulator 根本不需要真的 partial trace**。

你只需要：

```text
forward:
    U01 U23 U45 U67
    U13 U57
    U37

backward:
    U37†
    U57† U13†
    U67† U45† U23† U01†
```

那些“discarded” qubits 只是不再参与后续 gates。因为最后 observable 在 active qubit 上、inactive wires 上等价于 identity，所以 expectation 与显式 trace-out 完全一致。

这意味着它依然是一个干净的

[
\langle\psi|
U^\dagger M U
|\psi\rangle,
]

实际上原论文自己的 classifier 就直接写成了这个形式。([Nature][2])

所以它对你的工作特别有价值：**HEA 每层都扫全部 qubit；MERA 的 backward 却会出现逐层变化的 active-wire topology。** 这是完全不同的 GPU workload。

### 怎么引用？

最直接引用：

> E. Grant et al., “Hierarchical quantum classifiers,” *npj Quantum Information*, vol. 4, Art. 65, 2018. ([Nature][3])

```bibtex
@article{Grant2018Hierarchical,
  title   = {Hierarchical quantum classifiers},
  author  = {Grant, Edward and Benedetti, Marcello and Cao, Shuxiang
             and Hallam, Andrew and Lockhart, Joshua and Stojevic, Vid
             and Green, Andrew G. and Severini, Simone},
  journal = {npj Quantum Information},
  volume  = {4},
  pages   = {65},
  year    = {2018},
  doi     = {10.1038/s41534-018-0116-9}
}
```

论文中你就可以写：

> We include a MERA-based quantum classifier following the hierarchical circuit architecture of Grant et al. [X].

---

# 2. (S_n)-Permutation-Equivariant QNN：对 adjoint 尤其有意思

这个是我觉得**从系统论文角度甚至比 Data Re-uploading 更值得加**的。

Schatzki 等人在 2024 年 npj Quantum Information 的工作中构造了 permutation-equivariant QNN。它不是通常的

[
RY(\theta_{i})-CNOT-RY(\theta_{i+1})
]

模式，而是从三个 **collective generators** 出发：

[
H_X=\frac1n\sum_iX_i,
]

[
H_Y=\frac1n\sum_iY_i,
]

[
H_{ZZ}
======

\frac{2}{n(n-1)}
\sum_{i<j} Z_iZ_j.
]

每个 QNN layer 是其中一个 generator 的 exponentiation。([Nature][4])

从实际 gate 看，例如

[
e^{-i\theta H_X}
]

就是：

```text
q0 ─ RX(α) ─
q1 ─ RX(α) ─
q2 ─ RX(α) ─
q3 ─ RX(α) ─
      ↑
  同一个参数
```

而 (H_{ZZ}) 是：

```text
q0 ─■────■────■──
     ZZ   ZZ   ZZ
q1 ─■────┼────┼──
          │    │
q2 ───────■────┼──
               │
q3 ────────────■──
```

实际上是所有

[
(i,j),\quad i<j
]

pair 都做 ZZ interaction。

而最重要的一点是：

> **同一 collective layer 里的所有 gates 共享同一个 trainable parameter。**

原论文的 Fig. 2 就明确这样定义。([Nature][4])

---

## 我建议你的 benchmark 怎么定义

定义一个 macro-layer：

[
U_l=
e^{-i\gamma_lH_{ZZ}}
e^{-i\beta_lH_Y}
e^{-i\alpha_lH_X}.
]

然后：

[
U(\theta)=U_LU_{L-1}\cdots U_1.
]

也就是说：

```text
repeat L:
    RX(shared α_l) on all qubits
    RY(shared β_l) on all qubits
    RZZ(shared γ_l) on ALL pairs
```

严格来说，原论文把每次 generator exponentiation 称为一个 layer；这里我是为了 benchmark 参数方便，把 `X + Y + ZZ` 三个原论文 layer 合成一个 **macro-layer**。([Nature][4])

如果按照常用

[
RX(\phi)=e^{-i\phi X/2}
]

定义，那么归一化系数可以直接折进 gate angle：

[
RX\left(\frac{2\alpha_l}{n}\right),
]

[
RY\left(\frac{2\beta_l}{n}\right),
]

以及

[
RZZ\left(
\frac{4\gamma_l}{n(n-1)}
\right)
]

作用于所有 (i<j)。

---

## 它为什么对 adjoint 特别有研究价值？

因为这里出现了一个你现有 HEA 几乎没有的问题：

### parameter 数量和 parameterized gate 数量脱钩。

假设有 (L) 个 macro layers：

[
N_{\mathrm{logical-param}}=3L.
]

但一个 layer 实际执行：

[
n+n+\frac{n(n-1)}2
]

个 parameterized gates。

所以：

[
N_{\mathrm{gate}}
=================

O(Ln^2),
]

但

[
N_{\mathrm{param}}
==================

O(L).
]

例如 (n=32,L=8)：

[
N_\mathrm{param}=24,
]

而 parameterized gate occurrences 已经有几千个。

对于 adjoint：

[
\frac{\partial E}{\partial \gamma_l}
]

不是一个 gate contribution，而是**同一个 parameter 在大量 RZZ gate 上产生的 gradient contribution 的累加**。

这会真正考察：

* shared-parameter gradient accumulation；
* 多 gate contribution reduction；
* diagonal RZZ 的特殊优化；
* gate count 和 gradient dimension 不成比例的情况。

这比“再弄一个 RX/RY/CNOT HEA”提供的信息丰富得多。

而且这篇论文自己就专门分析了 loss partial derivative 的 variance、gradient scaling 和 barren plateau，因此作为“我们为什么选这个 trainable QNN”的引用也很漂亮。([arXiv][5])

### Observable

论文给出的 symmetry-compatible observable 包括：

[
\frac1n\sum_i X_i
]

以及

[
\frac{2}{n(n-1)}
\sum_{i<j}X_iX_j
]

等 permutation-invariant Pauli observables。([Nature][4])

所以你完全可以把

[
H=\frac1n\sum_i X_i
]

直接塞进你的统一 `energy_and_grad()`。

这比人为给它配一个 Hamiltonian 更自然。

### 怎么引用？

> L. Schatzki, M. Larocca, Q. T. Nguyen, F. Sauvage, and M. Cerezo, “Theoretical guarantees for permutation-equivariant quantum neural networks,” *npj Quantum Information*, vol. 10, Art. 12, 2024. ([Nature][6])

```bibtex
@article{Schatzki2024Permutation,
  title   = {Theoretical guarantees for permutation-equivariant quantum neural networks},
  author  = {Schatzki, Louis and Larocca, Mart{\'i}n
             and Nguyen, Quynh T. and Sauvage, Fr{\'e}d{\'e}ric
             and Cerezo, M.},
  journal = {npj Quantum Information},
  volume  = {10},
  pages   = {12},
  year    = {2024},
  doi     = {10.1038/s41534-024-00804-1}
}
```

我会把它称为 **(S_n)-equivariant QNN (EQNN)**，不要泛泛地只写 “equivariant circuit”。

---

# 3. Data Re-uploading QNN：不是拓扑最怪，但 QML 意义非常强

这个结构在 gate topology 上没有 MERA 那么“异形”，但是在 **QML semantics** 上与普通 HEA 差异非常明显。

普通 QML 常见：

[
U_{\rm train}(\theta)U_{\rm encode}(x)|0\rangle.
]

也就是：

```text
encode x
   ↓
trainable circuit
   ↓
measurement
```

Data re-uploading 则变成：

[
U(\phi_N)U(x)\cdots
U(\phi_2)U(x)
U(\phi_1)U(x)|0\rangle.
]

也就是：

```text
x → θ1 → x → θ2 → x → θ3 → ... → θL
```

数据 **每一层重新进入 circuit**。这正是 Pérez-Salinas 等人的原始定义。([ar5iv][7])

更有意思的是他们给出了 compressed layer：

[
L_l
===

U\left(
\boldsymbol{\theta}_l+
\mathbf w_l\circ\mathbf x
\right),
]

其中

[
\mathbf w_l\circ\mathbf x
=========================

(w_l^1x^1,w_l^2x^2,w_l^3x^3).
]

也就是 rotation angle 本身就是

[
\theta+w x.
]

([ar5iv][7])

所以一个 qubit 可以写成：

```text
── RZ(θz + wz*xz)
── RY(θy + wy*xy)
── RZ(θx + wx*xx)
```

然后下一层再把同一个 (x) re-upload 一遍。

---

## 多 qubit 版本也有明确出处

原论文并不只做 1 qubit。

他们的 multi-qubit classifier 是：

```text
q0 ──L0(1)──●────L0(2)────●── ...
            │             │
q1 ──L1(1)──●────L1(2)────●── ...
```

也就是：

> parallel data-reuploading rotations + CZ entanglers。

四 qubit 情况下，他们交替使用

[
(0,1),(2,3)
]

与

[
(1,2),(3,0)
]

这样的 CZ pairing。([ar5iv][7])

这正好可以自然推广成你需要的任意 (n)：

```text
Layer 0:
    Rot(x, θ) × n
    CZ(0,1), CZ(2,3), ...

Layer 1:
    Rot(x, θ) × n
    CZ(1,2), CZ(3,4), ...

Layer 2:
    Rot(x, θ) × n
    CZ(0,1), CZ(2,3), ...
```

也就是 brickwork CZ。

这里后面的任意-(n)推广是**我们的 benchmark extension**；原论文明确展示的是 2/4-qubit 版本。([ar5iv][7])

---

## 对你的 adjoint 实验，我会做一个重要简化

不要一开始同时 train

[
\theta,\quad w.
]

否则一个 physical gate angle：

[
\phi=\theta+wx
]

对应两个 logical parameters：

[
\frac{\partial E}{\partial\theta}
=================================

\frac{\partial E}{\partial\phi},
\qquad
\frac{\partial E}{\partial w}
=============================

x\frac{\partial E}{\partial\phi}.
]

这就把“quantum adjoint kernel 性能”和 classical chain rule 混在一起了。

第一版 benchmark 我会固定：

[
x\sim\text{seeded dataset},\qquad
w=\text{fixed},
]

**只把 (\theta) 当 trainable parameter。**

那么 adjoint 看到的就是普通 parameterized rotations，而电路仍然具有完整的数据 re-uploading 结构。

输出也没必要完全复现分类 loss。为了统一你的 `energy_and_grad()` API，直接用

[
E=\langle Z_0\rangle
]

或者一个固定 Pauli-sum observable 即可。

原论文实际使用的是最终状态与 class state 的 fidelity；这也可以写成 projector expectation，但为了你的 simulator benchmark，我更推荐统一用 Pauli observable。原始 classifier 的确以最终量子态的 fidelity 构造 cost。([ar5iv][7])

### 怎么引用？

> A. Pérez-Salinas, A. Cervera-Lierta, E. Gil-Fuster, and J. I. Latorre, “Data re-uploading for a universal quantum classifier,” *Quantum*, vol. 4, p. 226, 2020. ([量子期刊][8])

```bibtex
@article{PerezSalinas2020DataReuploading,
  title   = {Data re-uploading for a universal quantum classifier},
  author  = {P{\'e}rez-Salinas, Adri{\'a}n
             and Cervera-Lierta, Alba
             and Gil-Fuster, Elies
             and Latorre, Jos{\'e} I.},
  journal = {Quantum},
  volume  = {4},
  pages   = {226},
  year    = {2020},
  doi     = {10.22331/q-2020-02-06-226}
}
```

---

## 如果是我来定你最终的 benchmark suite

我现在反而会比较明确地做成：

[
\boxed{
\text{HEA}
+
\text{QAOA/HVA}
+
\text{MERA}
+
S_n\text{-EQNN}
+
\text{Data Re-uploading}
}
]

它们可以分别代表：

```text
HEA
 └─ regular topology
    + independent parameters

QAOA / HVA
 └─ physics-informed
    + commuting / diagonal operators
    + parameter sharing

MERA
 └─ hierarchical topology
    + shrinking active set

Sn-EQNN
 └─ symmetry-constrained
    + massive parameter sharing
    + all-to-all ZZ

Data Re-uploading
 └─ repeated data encoding
    + deep QML-specific structure
```

这样你的 benchmark 不再是“三四种不同名字的 `Rotation → CNOT`”，而是真的覆盖了 **regular、physics-informed、hierarchical、symmetry-aware、data-encoding-heavy** 五种不同结构。

尤其从你的 **GPU adjoint optimization** 角度，我认为 **MERA + (S_n)-EQNN 是这次最值得补进去的两个**：前者改变 backward traversal 的 topology，后者改变“parameter ↔ gate occurrence”的关系。这两个对系统设计的压力与普通 HEA 是本质不同的。([Nature][2])

[1]: https://docs.pennylane.ai/projects/lightning/en/latest/lightning_gpu/device.html?utm_source=chatgpt.com "Lightning GPU device — Lightning 0.46.0-dev14 documentation"
[2]: https://www.nature.com/articles/s41534-018-0116-9 "Hierarchical quantum classifiers | npj Quantum Information"
[3]: https://www.nature.com/articles/s41534-018-0116-9?utm_source=chatgpt.com "Hierarchical quantum classifiers | npj Quantum Information"
[4]: https://www.nature.com/articles/s41534-024-00804-1 "Theoretical guarantees for permutation-equivariant quantum neural networks | npj Quantum Information"
[5]: https://arxiv.org/abs/2210.09974 "Theoretical Guarantees for Permutation-Equivariant Quantum Neural Networks"
[6]: https://www.nature.com/articles/s41534-024-00804-1?utm_source=chatgpt.com "Theoretical guarantees for permutation-equivariant quantum neural networks | npj Quantum Information"
[7]: https://ar5iv.labs.arxiv.org/html/1907.02085 "[1907.02085] Data re-uploading for a universal quantum classifier"
[8]: https://quantum-journal.org/papers/q-2020-02-06-226/?utm_source=chatgpt.com "Data re-uploading for a universal quantum classifier – Quantum"
