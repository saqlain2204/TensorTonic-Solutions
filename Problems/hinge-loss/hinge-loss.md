## The Origins of Hinge Loss

Hinge loss is the loss function behind Support Vector Machines (SVMs). It was designed with a specific goal: maximize the margin between classes.

For binary classification with labels $y \in \{-1, +1\}$ and raw model output $f(x)$ (not a probability):

$$
L = \max(0, 1 - y \cdot f(x))
$$

This deceptively simple formula encodes a powerful idea: the model should not just classify correctly, but classify with confidence.

---

## Breaking Down the Formula

The term $y \cdot f(x)$ is called the **functional margin**:
- If y = +1 and f(x) > 0: correct classification, positive margin
- If y = -1 and f(x)  0: wrong classification, negative margin

The loss max(0, 1 - y * f(x)) then says:
- If margin >= 1: loss is 0 (correct and confident)
- If margin = 1
- Has a "hinge" (sharp corner) at f(x) = 1

This is where the name "hinge loss" comes from.

---

## The Gradient

$$
\frac{\partial L}{\partial f(x)} = \begin{cases} 0 & \text{if } y \cdot f(x) \geq 1 \\ -y & \text{if } y \cdot f(x)  1
- Piecewise linear (non-smooth at the hinge)

**Cross-entropy:**
- Outputs are probabilities (via softmax/sigmoid)
- Never zero loss (always some gradient)
- Continuously pushes toward higher confidence
- Smooth everywhere

Hinge loss is "satisfied" once the margin is large enough. Cross-entropy always wants higher confidence, even for samples already correctly classified with high confidence.

---

## Multi-Class Hinge Loss

For $C$ classes, the multi-class hinge loss is:

$$
L = \sum_{j \neq y} \max(0, 1 + f_j(x) - f_y(x))
$$

Where:
- $y$ is the true class
- $f_y(x)$ is the score for the true class
- $f_j(x)$ is the score for each incorrect class

Interpretation: for each wrong class, penalize if its score is within 1 of the true class score.

---

## Squared Hinge Loss

A variant that penalizes misclassifications more heavily:

$$
L = \max(0, 1 - y \cdot f(x))^2
$$

Differences from standard hinge:
- Squared penalty for violations
- Smooth gradient at the hinge point
- Larger penalty for confident wrong predictions
- Sometimes used when you want to more heavily penalize errors

---

## Where Hinge Loss Is Used

- **Support Vector Machines**: the original and primary use case
- **Maximum-margin classifiers**: any model aiming for large margins
- **Structured prediction**: variants like structured hinge loss for sequence labeling
- **Neural networks**: sometimes used as an alternative to cross-entropy, especially for binary classification
- **Ranking problems**: pairwise hinge loss for learning to rank