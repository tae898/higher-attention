# **Hierarchical Attention, Kernel Perspectives, and Higher-Order Similarity in Transformers**

## Motivation

Modern Transformers merge the outputs of multiple attention heads via _concatenation +
linear mixing_. We ask four questions:

1. **RQ1 – Is concatenation sufficient?**
2. **RQ2 – Can we replace concatenation with _attention-over-heads_ and obtain a
   genuinely hierarchical mechanism?**
3. **RQ3 – How does the self-attention matrix relate to kernel methods and Multiple
   Kernel Learning (MKL)?**
4. **RQ4 – What do higher-order (≥3-way) similarity functions buy us in NLP?**

We review the mathematics, catalogue existing work, and weigh expressivity against
computational cost.

---

## 1 Background

### 1.1 Standard Multi-Head Attention

For $M$ heads and $N$ tokens $X\!\in\!\mathbb{R}^{N\times d}$:

$$
\text{head}_m(X)\;=\;\operatorname{softmax}\!\Bigl(\tfrac{Q_mK_m^\top}{\sqrt{d_k}}\Bigr)V_m
\;\;\in\mathbb{R}^{N\times d_v}. $$

Heads are **concatenated** and projected,

$$
\text{MHA}(X)\;=\;\operatorname{Concat}\!\bigl[\text{head}_1,\dots,\text{head}_M\bigr]W^O,
$$

yielding $N$ output tokens. Concatenation is _flat_: heads do not explicitly interact.

### 1.2 Self-Attention as Kernels

Ignoring softmax, each head forms a Gram matrix  
$K^{(m)}_{ij}=\langle\phi_m(x_i),\phi_m(x_j)\rangle$,  
so multi-head attention implements **multiple learned kernels**.

---

## 2 RQ1 – Is Concatenation Enough?

Empirically many heads are redundant; pruning leaves accuracy intact without performance
degradation. This suggests that standard concatenation may be suboptimal. As Cordonnier
et al. (2020) demonstrate, the flat fusion approach forces $W^O$ to learn cross-head
coordination in one step, which fundamentally limits representation sharing and
parameter efficiency. Their collaborative attention mechanism shows that moving beyond
simple concatenation can maintain accuracy while reducing parameters, directly
challenging the sufficiency of traditional head fusion.

---

## 3 RQ2 – Hierarchical _Attention-over-Heads_

### 3.1 Mechanism

For token $i$ let $h_{i,m}$ be the $m$-th head vector.  
Introduce **second-level attention**:

$$ \tilde{h}_i \;=\;\sum_{m=1}^{M}\beta_{i,m}\,W_m\,h_{i,m},\quad
\beta_{i,m}=\frac{\exp(q_i^\top k_m)}{\sum_{m'}\exp(q_i^\top k_{m'})}. $$

Now fusion is **input-dependent** and **hierarchical** (token→token then head→head).

### 3.2 Existing Evidence

The hierarchical attention paradigm has strong empirical support from three key works:

**Collaborative Multi-Head Attention** (Cordonnier et al., 2020) fundamentally
challenges concatenation by introducing shared key/query projections with linear mixing.
This approach achieves the same accuracy with fewer parameters, proving that head
collaboration can be more efficient than independent processing followed by
concatenation.

**Talking-Heads Attention** (Shazeer et al., 2020) implements the core idea of
attention-over-heads through linear projections before and after softmax operations. By
enabling explicit head-wise interactions, they achieve better perplexity on masked
language modeling tasks, demonstrating that heads can productively "talk" to each other.

**Hierarchical Attention Transformers** (Chalkidis et al., 2022) show practical benefits
in long document processing, where hierarchical attention (intra-segment then
cross-segment) yields both accuracy improvements and memory efficiency gains.

#### Benefits of Hierarchical Attention

- Dynamic head selection per token (input-dependent fusion)
- Parameter sharing across heads (efficiency gains)
- Closer alignment to deep kernel learning principles (see §4)

#### Implementation Considerations

- Slightly higher latency due to additional softmax operations
- Requires careful regularization to prevent over-concentration on single heads

---

## 4 RQ3 – Kernel View: From MKL to Hierarchical Kernels

### 4.1 Multiple Kernel Learning (flat)

$$ K(x,x')=\sum_{m=1}^{M}\beta_m K^{(m)}(x,x'),\quad \beta_m\ge0,\;\sum_{m}\beta_m=1. $$

Weights $\beta_m$ are **global** ⇒ same for all samples.

### 4.2 Hierarchical Attention as Deep, Input-Conditional MKL

The connection between hierarchical attention and kernel methods becomes clear when we
examine how attention-over-heads operates. As shown by Cordonnier et al. (2020) and
Shazeer et al. (2020), hierarchical fusion mechanisms realize:

$$ K_{ij}=\sum_{m}\beta_{i,m}\,\beta_{j,m}\,K^{(m)}_{ij}, $$

where the **combination weights depend on each token** – this is far richer than
classical MKL where weights are global. This input-conditional weighting is precisely
what makes talking-heads attention so effective: the model learns to dynamically
emphasize different kernel combinations based on the specific input context.

---

## 5 RQ4 – Higher-Order (≥3-Way) Similarity

### 5.1 From Pairwise to Multi-Way Attention

Standard self-attention computes **pairwise similarity** between tokens $i$ and $j$:

$$A_{ij} = \text{softmax}\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right)$$

This limits us to modeling relationships between token pairs. But what if we need to
capture **three-way or higher-order interactions** directly in the attention matrix?

### 5.2 Higher-Order Attention Mechanisms

Instead of pairwise attention $A_{ij}$, consider **trilinear attention** that computes
similarity across three tokens simultaneously:

$$ A_{ijk} = \text{softmax}\left(\sum_{a,b,c} T_{abc} \, q_i^{(a)} k_j^{(b)}
k_k^{(c)}\right) $$

where $T \in \mathbb{R}^{d \times d \times d}$ is a learned tensor capturing 3-way
interactions.

The attended output becomes: $\text{output}_i = \sum_{j,k} A_{ijk} \, V_{jk}$

where $V_{jk}$ represents some combination of value vectors from positions $j$ and $k$.

### 5.3 Why Higher-Order Attention in NLP?

Moving beyond pairwise attention matrices enables richer relational modeling:

| Use-case            | Gain from multi-way attention                                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **QA co-attention** | Jointly attend to (question, context, answer) triplets – BiDAF's trilinear attention (Seo et al., 2016) scores query-context-candidate interactions |
| **Compositional reasoning** | Capture (subject, relation, object) dependencies in knowledge-intensive tasks |
| **Discourse modeling** | Model (event, participant, temporal slot) interactions in narrative understanding |
| **Multi-document QA** | Attend across (query, doc1, doc2) simultaneously for cross-document reasoning |

### 5.4 Computational Reality Check

The attention perspective reveals why higher-order similarity is rarely used:

- **Storage**: $O(N^3)$ attention tensors vs. $O(N^2)$ matrices
- **Computation**: $O(N^3 d)$ vs. $O(N^2 d)$ for standard attention  
- **Memory**: Cubic scaling makes this prohibitive for long sequences

Practical implementations use **decompositions** (e.g., DistMult's diagonal $T$) or
**approximate methods** to make 3-way attention tractable.

---

## 6 Discussion

The evidence from recent work strongly supports moving beyond simple concatenation:

- **Concatenation** is computationally simple but empirically suboptimal, as Cordonnier
  et al. (2020) demonstrate through their parameter-efficient collaborative attention
  mechanism.

- **Hierarchical attention** introduces adaptive kernel fusion that aligns Transformers
  with learned deep kernels. The success of talking-heads attention (Shazeer et al.,
  2020) validates this theoretical connection between attention mechanisms and kernel
  methods.

- **Higher-order similarity** proves powerful yet costly; the success of BiDAF's
  trilinear attention shows these methods should be used when tasks are inherently
  multi-relational (complex QA scenarios requiring three-way reasoning).

---

## 7 Research Questions Revisited

- **RQ1**: Concatenation suffices in practice but is demonstrably suboptimal. Cordonnier
  et al. (2020) prove that collaborative mechanisms can achieve equivalent accuracy with
  fewer parameters, exposing the inefficiency of head concatenation.

- **RQ2**: Attention-over-heads yields both parameter savings and richer fusion
  capabilities. The empirical success of talking-heads attention (Shazeer et al., 2020)
  and hierarchical attention transformers (Chalkidis et al., 2022) validates this
  approach across multiple domains.

- **RQ3**: Viewing attention as kernels clarifies why hierarchical fusion approximates
  deep Multiple Kernel Learning. This theoretical connection, supported by the practical
  success of collaborative and talking-heads variants, provides a principled foundation
  for future attention mechanisms.

- **RQ4**: Higher-order attention mechanisms can capture richer multi-way token
  interactions than standard pairwise attention matrices, as demonstrated by BiDAF's
  trilinear attention for QA tasks. However, the cubic computational and memory costs
  ($O(N^3)$) severely limit their practical applicability, making them viable only for
  specialized tasks where multi-way reasoning is essential and sequence lengths are
  manageable.

---

## References

1. **Cordonnier, J-B., Loukas, A., & Jaggi, M.** (2020). _Multi-Head Attention:
   Collaborate Instead of Concatenate._ arXiv preprint arXiv:2006.16362.
   [https://arxiv.org/abs/2006.16362](https://arxiv.org/abs/2006.16362)

2. **Shazeer, N., Lan, Z., Cheng, Y., Ding, N., & Hou, L.** (2020). _Talking-Heads
   Attention._ arXiv preprint arXiv:2003.02436.
   [https://arxiv.org/abs/2003.02436](https://arxiv.org/abs/2003.02436)

3. **Chalkidis, I., Dai, X., Fergadiotis, M., Malakasiotis, P., & Elliott, D.** (2022).
   _An Exploration of Hierarchical Attention Transformers for Efficient Long Document
   Classification._ arXiv preprint arXiv:2210.05529.
   [https://arxiv.org/abs/2210.05529](https://arxiv.org/abs/2210.05529)

4. **Seo, M., Kembhavi, A., Farhadi, A., & Hajishirzi, H.** (2016). _Bidirectional
   Attention Flow for Machine Comprehension._ arXiv preprint arXiv:1611.01603.
   [https://arxiv.org/abs/1611.01603](https://arxiv.org/abs/1611.01603)
