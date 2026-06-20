## What Is Optimization?

In machine learning, **optimization** means finding the set of parameters that makes a model perform as well as possible. We measure performance with a **loss function** (also called a cost function or objective function). The loss tells you how wrong your model is:

- High loss = bad predictions
- Low loss = good predictions
- Zero loss = perfect predictions (rarely achievable)

The goal is to find the parameter values that **minimize** the loss function. For a simple 1D quadratic:

$$
f(x) = ax^2 + bx + c
$$

the "parameter" is just $x$, and we want to find the value of $x$ that makes $f(x)$ as small as possible.

When $a > 0$, this is a parabola opening upward, with a single minimum. You could find it analytically: set the derivative to zero and solve. The minimum is at $x^* = -\frac{b}{2a}$.

But in real machine learning:

- The loss function has **millions of parameters** (not just one $x$)
- There is **no closed-form solution** for the minimum
- The function is too complex to solve analytically
- We need an **iterative method** that takes small steps toward the minimum

That iterative method is gradient descent.

---

## The Derivative Points Uphill

The derivative of a function at a point tells you two things:

- **Direction**: whether the function is increasing or decreasing at that point
- **Magnitude**: how steeply it is changing

For our quadratic:

$$
f'(x) = 2ax + b
$$

- When $f'(x) > 0$: the function is increasing (going uphill). Moving **left** (decreasing $x$) would decrease the function.
- When $f'(x)  \frac{1}{a}$, the algorithm **oscillates** back and forth across the minimum
- If $\eta > \frac{2}{a}$, the oscillations grow and the algorithm **diverges** to infinity
- Example: with $f(x) = x^2$ and $\eta = 1.5$, starting at $x = 1$: $x_1 = 1 - 1.5(2) = -2$, $x_2 = -2 - 1.5(-4) = 4$, $x_3 = 4 - 1.5(8) = -8$. It is blowing up.

**Too small** ($\eta$ too tiny):
- Each step barely moves
- Convergence takes thousands or millions of iterations
- The algorithm works correctly but wastes computation
- Example: with $\eta = 0.001$, it takes about 2000 steps to get close to the minimum

**Just right**:
- For the quadratic $ax^2 + bx + c$, the optimal learning rate is $\eta = \frac{1}{2a}$
- With this rate, gradient descent reaches the exact minimum in **one step**
- $x_1 = x_0 - \frac{1}{2a}(2ax_0 + b) = x_0 - x_0 - \frac{b}{2a} = -\frac{b}{2a} = x^*$

In practice (with complex, non-quadratic loss functions), finding the "just right" learning rate requires experimentation. Common starting values are 0.01, 0.001, or 0.0001.

---

## Convergence: How Fast Does It Get There?

For a convex quadratic, gradient descent converges **exponentially** (also called linear convergence):

- The error at step $t$ is: $|x_t - x^*| = |1 - 2a\eta|^t \cdot |x_0 - x^*|$
- The factor $|1 - 2a\eta|$ is the **contraction rate**
- When this factor is between 0 and 1, the error shrinks each step
- Smaller factor = faster convergence

For $a = 1$ and $\eta = 0.1$:
- Contraction rate: $|1 - 0.2| = 0.8$
- After 10 steps: error is $0.8^{10} \approx 0.107$ of the original (about 10x smaller)
- After 50 steps: error is $0.8^{50} \approx 0.000014$ (about 70,000x smaller)

This exponential decay on a log scale looks like a straight line going down, which is why it is called "linear convergence."

---

## From 1D Quadratic to Real Machine Learning

This 1D quadratic is the training ground for understanding optimization. Every concept here scales up:

- **One parameter** becomes **millions of parameters** (weights, biases)
- **The derivative** becomes **the gradient** (a vector of partial derivatives, one per parameter)
- **The parabola** becomes a **high-dimensional loss surface** with hills, valleys, saddle points, and flat regions
- **One exact gradient** becomes a **noisy mini-batch estimate** (stochastic gradient descent)
- **Simple update rule** gets enhanced with **momentum** (SGD + momentum), **adaptive rates** (Adam, RMSProp), and **learning rate schedules**

But the core operation never changes: compute the gradient, take a step in the opposite direction.