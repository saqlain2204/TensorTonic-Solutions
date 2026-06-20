# Transformer Encoder Block

The encoder block is the fundamental repeating unit of the Transformer encoder. Each block contains two sub-layers -- Multi-Head Self-Attention (MHA) and a position-wise Feed-Forward Network (FFN) -- each wrapped with a residual connection and followed by Layer Normalization. Stacking $N$ identical blocks produces the full encoder.

Introduced in "Attention Is All You Need" (Vaswani et al., 2017), the encoder block replaced recurrence entirely with attention-based token mixing and per-position feed-forward computation. The design remains the backbone of BERT, ViT, and many encoder-based architectures.

---

## What It Is

An encoder block is one complete repeating unit of the Transformer encoder. It takes a sequence of vectors $X \in \mathbb{R}^{T \times d_{\text{model}}}$ and produces an output of the same shape. The block does not change the sequence length or the hidden dimension -- it refines the representation in place.

Internally, the block has exactly two sub-layers arranged in a fixed order:

* **Sub-layer 1 -- Multi-Head Self-Attention (MHA):** Every token attends to every other token in the sequence. This is the "token mixing" step. Because this is self-attention, the queries, keys, and values all come from the same input.
* **Sub-layer 2 -- Position-wise Feed-Forward Network (FFN):** A two-layer MLP applied independently to each position. This is the "channel mixing" step, where each token's representation is transformed without any interaction with other tokens.

Each sub-layer is wrapped with a residual connection and followed by Layer Normalization. The output of one block becomes the input to the next, and the final block's output is the encoder's representation.

---

## Key Equations

The encoder block computes two sub-layer operations sequentially. Let $X$ be the input to the block.

**Sub-layer 1 -- Self-Attention with residual and LayerNorm:**

$$
X' = \text{LayerNorm}(X + \text{MHA}(X, X, X))
$$

Here $\text{MHA}(X, X, X)$ means the query, key, and value matrices are all derived from the same input $X$. The residual adds the original $X$ to the attention output, then LayerNorm is applied to the sum.

**Sub-layer 2 -- FFN with residual and LayerNorm:**

$$
\text{output} = \text{LayerNorm}(X' + \text{FFN}(X'))
$$

The FFN itself is defined as:

$$
\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2
$$

where $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$, and $d_{ff} = 2048$ in the original paper (4x expansion from $d_{\text{model}} = 512$).

The general pattern for both sub-layers can be written uniformly:

$$
\text{SubLayerOutput} = \text{LayerNorm}(x + \text{SubLayer}(x))
$$

This uniform wrapping is what the paper describes as "a residual connection around each of the two sub-layers, followed by layer normalization."

---

## The Two Sub-Layers

The two sub-layers serve fundamentally different purposes, and understanding the division of labor is key to understanding the encoder block.

**Sub-layer 1: Multi-Head Self-Attention (token mixing).** This sub-layer allows every token to gather information from every other token. Given input $X$, it computes:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

$Q$, $K$, and $V$ are all projected from the same $X$ -- this is what makes it "self"-attention. In the multi-head variant, $h$ separate attention computations run in parallel on subspaces of dimension $d_k = d_{\text{model}} / h$, then outputs are concatenated and projected. With $h = 8$, each head uses $d_k = 64$.

The attention weights form a $T \times T$ matrix where each row sums to 1, representing how much each token attends to every other token. The encoder uses full bidirectional attention with no causal mask.

**Sub-layer 2: Position-wise FFN (channel mixing).** This sub-layer processes each token independently through the same two-layer MLP. "Position-wise" means the network is applied to each position separately with shared weights -- no information flows between positions.

The FFN expands from $d_{\text{model}}$ to $d_{ff}$, applies ReLU, then projects back. The expansion to $d_{ff} = 4 \times d_{\text{model}}$ gives capacity to learn complex per-token transformations. Think of attention as "which tokens should talk to each other" and FFN as "what to do with the information each token has gathered."

---

## Post-Norm Architecture

The original Transformer uses what is now called the "Post-Norm" configuration: LayerNorm is applied **after** the residual addition. This is distinct from the "Pre-Norm" variant used in GPT-2, LLaMA, and most modern architectures.

**Post-Norm (original Transformer):**

$$
\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))
$$

**Pre-Norm (modern variant):**

$$
\text{output} = x + \text{SubLayer}(\text{LayerNorm}(x))
$$

The difference matters for training stability:

* **Post-Norm** normalizes the combined signal (residual + sub-layer output). The sub-layer receives unnormalized input, which can have growing magnitude in deeper stacks. This makes training sensitive to learning rate and often requires warmup. However, post-norm can produce better final performance because the full representational capacity flows through the residual path.
* **Pre-Norm** normalizes the input before the sub-layer, ensuring stable inputs regardless of depth. This eliminates the need for careful warmup, but some studies report slightly lower final quality compared to well-tuned post-norm.

When implementing the original Transformer encoder block, it is essential to use post-norm. Placing LayerNorm in the wrong position changes the gradient flow entirely.

---

## Residual Connections

Each sub-layer is wrapped with a residual (skip) connection. Instead of computing $y = f(x)$, the block computes $y = x + f(x)$. This simple addition has profound consequences.

**Gradient highways.** During backpropagation, the gradient of $y = x + f(x)$ with respect to $x$ is:

$$
\frac{\partial y}{\partial x} = I + \frac{\partial f(x)}{\partial x}
$$

The identity matrix $I$ ensures gradients always have a direct path through the residual connection. Even if $\frac{\partial f}{\partial x}$ vanishes, the gradient through the identity path is exactly 1. This prevents the vanishing gradient problem.

**Enabling deep stacking.** The original Transformer uses $N = 6$ encoder blocks, meaning 12 sub-layers total. The skip connections allow each sub-layer to learn a small refinement rather than a complete transformation -- a much easier optimization target.

**Dimension constraint.** For $x + f(x)$ to work, the sub-layer output must match the input dimension. This is why $d_{\text{model}}$ remains constant throughout the encoder. The paper notes: "all sub-layers in the model produce outputs of dimension $d_{\text{model}} = 512$" to facilitate these residual connections.

---

## Why This Specific Order

The encoder block always runs attention first, then FFN. This ordering is not arbitrary.

**Step 1: Attention captures dependencies.** Self-attention allows each token to read from all other tokens. After this step, each position's representation has been enriched with contextual information from the entire sequence. A token like "bank" can now carry information about whether "river" or "money" appeared nearby.

**Step 2: FFN processes each position.** The feed-forward sub-layer takes the context-enriched representation and applies a nonlinear transformation independently at each position. This is where the model "processes" what each token means given its context. Research has shown FFN layers act as key-value memories storing factual associations.

**Step 3: LayerNorm stabilizes.** After each sub-layer (with its residual addition), LayerNorm re-centers and re-scales the activations, preventing drift to extreme magnitudes.

If the order were reversed (FFN first, then attention), each token would be transformed in isolation before seeing its context. The attention layer would then mix already-transformed representations without contextual guidance. The attention-then-FFN order ensures contextual gathering happens before per-position reasoning.

---

## Paper Context

The encoder block was introduced in "Attention Is All You Need" (Vaswani et al., 2017). The paper's encoder configuration:

* **Number of blocks ($N$):** 6 identical encoder layers stacked sequentially
* **Model dimension ($d_{\text{model}}$):** 512
* **Attention heads ($h$):** 8, each with $d_k = d_v = 64$
* **FFN inner dimension ($d_{ff}$):** 2048 (4x expansion)
* **Normalization:** Post-LayerNorm after residual addition
* **Dropout:** Applied to sub-layer output before residual addition, $P_{\text{drop}} = 0.1$

The paper states: "Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. We employ a residual connection around each of the two sub-layers, followed by layer normalization."

The encoder was designed for the source side of machine translation. Unlike the decoder, it has no causal mask -- every token attends to every other token bidirectionally. This is what BERT later exploited for masked language modeling. With dropout, the full computation is $\text{LayerNorm}(x + \text{Dropout}(\text{SubLayer}(x)))$.

---

## Numerical Example

Let us trace a concrete input through one encoder block with $d_{\text{model}} = 4$, $T = 2$, one attention head, and $d_{ff} = 8$.

**Input (after embedding + positional encoding):**

$$
X = \begin{bmatrix} 1.0 & 0.5 & -0.5 & 0.2 \\ 0.3 & -0.1 & 0.8 & -0.4 \end{bmatrix}
$$

**Sub-layer 1: Self-Attention.** Assume the single-head attention produces:

$$
\text{MHA}(X, X, X) = \begin{bmatrix} 0.12 & -0.08 & 0.15 & -0.03 \\ 0.09 & -0.05 & 0.11 & -0.02 \end{bmatrix}
$$

**Residual addition:**

$$
X + \text{MHA} = \begin{bmatrix} 1.12 & 0.42 & -0.35 & 0.17 \\ 0.39 & -0.15 & 0.91 & -0.42 \end{bmatrix}
$$

**LayerNorm (post-norm).** For token 1: $[1.12, 0.42, -0.35, 0.17]$, mean $\mu = 0.34$, std $\sigma = 0.507$. After normalizing (with $\gamma = 1, \beta = 0$):

$$
X' = \begin{bmatrix} 1.54 & 0.16 & -1.36 & -0.34 \\ 0.40 & -0.52 & 1.38 & -1.26 \end{bmatrix}
$$

**Sub-layer 2: FFN.** Suppose $\text{ReLU}(X' W_1 + b_1) W_2 + b_2$ produces:

$$
\text{FFN}(X') = \begin{bmatrix} 0.08 & -0.12 & 0.05 & 0.10 \\ -0.06 & 0.09 & -0.03 & 0.07 \end{bmatrix}
$$

**Residual addition:**

$$
X' + \text{FFN}(X') = \begin{bmatrix} 1.62 & 0.04 & -1.31 & -0.24 \\ 0.34 & -0.43 & 1.35 & -1.19 \end{bmatrix}
$$

**LayerNorm (post-norm).** Normalize again per token:

$$
\text{output} = \begin{bmatrix} 1.56 & 0.10 & -1.21 & -0.18 \\ 0.38 & -0.46 & 1.40 & -1.22 \end{bmatrix}
$$

Output shape matches input: $(2, 4)$. Each sub-layer contributed a small refinement via its residual, and LayerNorm kept activations well-scaled. This output flows to the next block or, if final, to the decoder's cross-attention.

---

## The Encoder Stack

The full encoder consists of $N = 6$ identical blocks stacked sequentially. "Identical" means same architecture (two sub-layers with residual + post-LayerNorm), but each block has its own learned parameters.

**Progressive refinement.** Early blocks tend to capture local syntactic patterns (subject-verb agreement, phrase structure). Middle blocks build semantic relationships (coreference, entity types). Later blocks produce task-ready representations encoding global relationships.

**Information flow.** Because encoder self-attention has no causal mask, every token can attend to every other token at every layer. By stacking 6 blocks, the model builds increasingly abstract and compositional representations.

**Output to decoder.** The final block's output $\in \mathbb{R}^{T \times d_{\text{model}}}$ is used by decoder cross-attention as the key and value source. For encoder-only models like BERT, this output feeds directly into task-specific heads.

**Why 6 blocks?** $N = 6$ balances capacity and training cost for translation. BERT-Base uses 12, BERT-Large uses 24. Each block adds roughly $4 d_{\text{model}}^2 + 2 d_{\text{model}} d_{ff}$ parameters plus LayerNorm parameters.

---

## Common Pitfalls

When implementing the encoder block, several mistakes commonly arise:

* **Wrong LayerNorm placement (post-norm vs. pre-norm).** The original Transformer uses post-norm: $\text{LN}(x + f(x))$. Many modern implementations default to pre-norm: $x + f(\text{LN}(x))$. Using the wrong one changes the gradient flow and training dynamics entirely.
* **Incorrect residual connections.** The residual must add the sub-layer's own input to its output. A common mistake is adding the block's original input $X$ to both sub-layers. Correct: sub-layer 1 uses $X$ as residual, producing $X'$. Sub-layer 2 uses $X'$ as residual. Each sub-layer has its own residual from its own input.
* **Forgetting self-attention uses the same input for Q, K, and V.** In the encoder, Q, K, and V all come from the same $X$. This differs from decoder cross-attention where Q comes from the decoder and K, V from the encoder. Using different sources produces cross-attention, not self-attention.
* **Missing one of the two sub-layers.** Both sub-layers are essential. Without FFN, the model has only token mixing and no per-position processing. Without attention, tokens cannot interact at all.
* **Applying a causal mask in the encoder.** The encoder uses full bidirectional attention. Accidentally applying a causal (triangular) mask prevents tokens from attending to future positions, destroying the encoder's bidirectional nature.
* **Forgetting dropout before residual addition.** The original architecture applies dropout to the sub-layer output before adding the residual: $\text{LN}(x + \text{Dropout}(\text{SubLayer}(x)))$. Placing dropout elsewhere changes the regularization effect.