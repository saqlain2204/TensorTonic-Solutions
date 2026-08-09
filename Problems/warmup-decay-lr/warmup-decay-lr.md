## Why Training Starts Badly

At the very beginning of training, everything is random:

- **Weights are random**: initialized from a random distribution (Xavier, He, etc.)
- **Activations are random**: since inputs pass through random weights, the activations are essentially noise
- **Gradients are unreliable**: computed from random activations, these gradients have high variance and may point in misleading directions
- **Optimizer state is empty**: Adam's moment estimates ($m$ and $v$) are all zeros, making the first few updates poorly calibrated

If you apply a full-strength learning rate to these unreliable gradients, the results can be catastrophic:

- A large, noisy gradient can push weights far into a bad region
- Once the model is in a bad region, it may take thousands of steps to recover
- In extreme cases, the loss diverges to infinity or becomes NaN
- Even if training eventually stabilizes, the early damage can lead to a suboptimal final result

---

## Warmup: Starting Gently

**Warmup** addresses this by starting with a very small (or zero) learning rate and gradually increasing it over the first several hundred or thousand steps.

The linear warmup formula for step $t  T$)

Learning rate stays fixed at 0 (or $\eta_{\min}$ if specified).

---

## The Shape

If you plot $\eta$ vs. step, the schedule looks like a **triangle** (or a ramp up followed by a ramp down):

- Rises linearly during warmup
- Peaks at step $W$
- Falls linearly during decay
- Flat at 0 after step $T$

The peak is always at the warmup boundary $t = W$. The warmup is typically much shorter than the total training, so the triangle is asymmetric: a short rise followed by a long decline.

---

## Edge Cases

**Zero warmup** ($W = 0$):
- The schedule starts at $\eta_0$ and decays linearly to 0
- No warmup phase at all
- This is just linear decay

**Warmup equals total steps** ($W = T$):
- The schedule ramps up from 0 to $\eta_0$ and immediately starts decaying
- The peak lasts only one step
- Not a useful configuration in practice

**Step beyond total steps** ($t > T$):
- The learning rate is clamped at 0 (or $\eta_{\min}$)
- No negative learning rates

---

## Warmup + Linear Decay vs. Other Schedules

- **Warmup + linear decay**: triangular shape. Simple, effective. The default in Hugging Face Transformers (`get_linear_schedule_with_warmup`).
- **Warmup + cosine decay**: warmup followed by a cosine curve. Slightly smoother. Often marginally better than linear decay. Used in many LLM training recipes.
- **Step decay (no warmup)**: learning rate drops by a fixed factor (e.g., 10x) at specific epochs. The traditional approach for CNNs trained with SGD. No warmup phase.
- **Constant (no schedule)**: $\eta$ never changes. Simple but suboptimal for most tasks.
- **Exponential decay**: $\eta$ is multiplied by a constant factor each step. Similar to linear in practice but with a different curve shape.

---

## Where This Schedule Shows Up

- **BERT pretraining**: the original BERT paper used exactly this: linear warmup + linear decay. This is what popularized the approach.
- **BERT fine-tuning**: the standard recipe for fine-tuning BERT on downstream tasks uses warmup (5-10% of steps) + linear decay.
- **Hugging Face Transformers**: the default scheduler is `get_linear_schedule_with_warmup`, which implements exactly this three-phase schedule.
- **GPT-2/GPT-3 training**: used warmup + cosine decay (a close relative).
- **Any Transformer training**: warmup + some form of decay has become the standard practice. If you are training or fine-tuning a Transformer and not using warmup, you are likely leaving performance on the table.
- **General best practice**: even outside of Transformers, warmup + decay is a safe default for any deep learning training run.