## What Is a Geometric Distribution?

The Geometric distribution models the **number of trials until the first success** in a sequence of independent Bernoulli trials.

It answers the question: "How many times must I repeat an experiment before I succeed for the first time?"

---

## Two Conventions

There are two common ways to define the Geometric distribution:

**Convention 1: Number of trials until first success**

$X \in \{1, 2, 3, ...\}$

Includes the successful trial in the count.

**Convention 2: Number of failures before first success**

$Y \in \{0, 1, 2, ...\}$

Counts only failures, not the success.

These are related by $X = Y + 1$. We will primarily use Convention 1.

---

## Real-World Examples

Many waiting-time scenarios follow a Geometric distribution:

- Number of coin flips until the first heads
- Number of job applications until the first offer
- Number of customers until the first purchase
- Number of attempts until passing an exam
- Number of rolls until rolling a six
- Number of server requests until the first failure

---

## The Parameter $p$

The Geometric distribution has a single parameter:

$$
p = P(\text{success on each trial})
$$

**Constraints:**
- $0  k) = 1 - F(k) = (1-p)^k
$$

This is the probability of $k$ consecutive failures.

**Example:** For $p = 0.3$, probability of still waiting after 5 trials:
$$
P(X > 5) = (0.7)^5 = 0.168 = 16.8\%
$$

---

## Memoryless Property

The Geometric distribution is **memoryless**:

$$
P(X > s + t | X > s) = P(X > t)
$$

If you have already failed $s$ times, the probability of needing at least $t$ more trials is the same as starting fresh.

**Intuition:** Past failures do not affect future success probability. Each trial is independent.

**Example:** If you have flipped 10 tails in a row, the expected number of additional flips to get heads is still $1/p$, not less.

---

## Proof of Memorylessness

$$
P(X > s + t | X > s) = \frac{P(X > s + t \text{ and } X > s)}{P(X > s)}
$$

Since $X > s + t$ implies $X > s$:

$$
= \frac{P(X > s + t)}{P(X > s)} = \frac{(1-p)^{s+t}}{(1-p)^s} = (1-p)^t = P(X > t)
$$

The Geometric is the **only** discrete memoryless distribution.

---

## Alternative PMF (Convention 2)

For $Y$ = number of failures before first success:

$$
P(Y = k) = (1-p)^k p
$$

where $k \in \{0, 1, 2, ...\}$.

**Mean:**
$$
E[Y] = \frac{1-p}{p}
$$

**Variance:**
$$
\text{Var}(Y) = \frac{1-p}{p^2}
$$

Note: $Y = X - 1$, so $E[Y] = E[X] - 1 = \frac{1}{p} - 1 = \frac{1-p}{p}$

---

## Relationship to Other Distributions

**Negative Binomial:**

The Geometric is a special case of Negative Binomial with $r = 1$ success.

**Exponential:**

The Geometric is the discrete analog of the Exponential distribution. Both are memoryless.

**Bernoulli:**

Each trial in a Geometric sequence is a Bernoulli trial.

---

## Sum of Geometrics

If $X_1, X_2, ..., X_r$ are independent $\text{Geometric}(p)$ random variables:

$$
\sum_{i=1}^{r} X_i \sim \text{Negative Binomial}(r, p)
$$

This is the number of trials to get $r$ successes.

---

## Maximum Likelihood Estimation

Given observations $x_1, x_2, ..., x_n$ from $\text{Geometric}(p)$:

$$
\hat{p} = \frac{n}{\sum_{i=1}^{n} x_i} = \frac{1}{\bar{x}}
$$

The MLE is the reciprocal of the sample mean.

---

## Mode of the Distribution

The mode (most likely value) is always:

$$
\text{mode} = 1
$$

The probability decreases as $k$ increases, so the first trial is always most likely to be the success.

---

## Applications in Machine Learning

**Waiting time models:**
Expected number of iterations until convergence.

**Reliability:**
Number of uses until first failure.

**Sampling:**
Number of random samples until finding a specific type.

**Coupon collector:**
Related to collecting all items in a set.