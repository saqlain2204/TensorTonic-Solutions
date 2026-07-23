## What Is Vector Normalization?

Vector normalization is the process of converting a vector into a **unit vector** (a vector with length 1) that points in the same direction as the original.

A normalized vector is also called a **unit vector** or **direction vector**.

$$
\hat{\mathbf{v}} = \frac{\mathbf{v}}{||\mathbf{v}||}
$$

---

## Why Normalize Vectors?

**1. Direction without magnitude:**

When you only care about direction, not length. Example: surface normals for lighting.

**2. Simplified calculations:**

Many formulas simplify when vectors have unit length:
- Dot product becomes just $\cos(\theta)$
- No need to divide by magnitudes repeatedly

**3. Numerical stability:**

Keeping vectors normalized prevents values from growing unboundedly.

**4. Standard representation:**

Allows fair comparison between vectors of different original magnitudes.

---

## The Normalization Formula

For a 3D vector $\mathbf{v} = (v_x, v_y, v_z)$:

**Step 1:** Compute the magnitude (length):
$$
||\mathbf{v}|| = \sqrt{v_x^2 + v_y^2 + v_z^2}
$$

**Step 2:** Divide each component by the magnitude:
$$
\hat{\mathbf{v}} = \left( \frac{v_x}{||\mathbf{v}||}, \frac{v_y}{||\mathbf{v}||}, \frac{v_z}{||\mathbf{v}||} \right)
$$

---

## Properties of Normalized Vectors

**Unit length:**
$$
||\hat{\mathbf{v}}|| = 1
$$

**Same direction:**
$$
\hat{\mathbf{v}} \parallel \mathbf{v}
$$

**Scaling relationship:**
$$
\mathbf{v} = ||\mathbf{v}|| \cdot \hat{\mathbf{v}}
$$

Any vector equals its magnitude times its unit vector.

---

## Worked Example

**Vector:** $\mathbf{v} = (3, 4, 0)$

**Step 1: Compute magnitude**
$$
||\mathbf{v}|| = \sqrt{3^2 + 4^2 + 0^2} = \sqrt{9 + 16 + 0} = \sqrt{25} = 5
$$

**Step 2: Divide components**
$$
\hat{\mathbf{v}} = \left( \frac{3}{5}, \frac{4}{5}, \frac{0}{5} \right) = (0.6, 0.8, 0)
$$

**Verification:**
$$
||\hat{\mathbf{v}}|| = \sqrt{0.6^2 + 0.8^2 + 0^2} = \sqrt{0.36 + 0.64} = \sqrt{1} = 1 \checkmark
$$

---

## Another Example

**Vector:** $\mathbf{v} = (1, 2, 2)$

**Magnitude:**
$$
||\mathbf{v}|| = \sqrt{1^2 + 2^2 + 2^2} = \sqrt{1 + 4 + 4} = \sqrt{9} = 3
$$

**Normalized vector:**
$$
\hat{\mathbf{v}} = \left( \frac{1}{3}, \frac{2}{3}, \frac{2}{3} \right) \approx (0.333, 0.667, 0.667)
$$

**Verification:**
$$
||\hat{\mathbf{v}}|| = \sqrt{\frac{1}{9} + \frac{4}{9} + \frac{4}{9}} = \sqrt{\frac{9}{9}} = 1 \checkmark
$$

---

## Example with Non-Integer Components

**Vector:** $\mathbf{v} = (1, 1, 1)$

**Magnitude:**
$$
||\mathbf{v}|| = \sqrt{1 + 1 + 1} = \sqrt{3} \approx 1.732
$$

**Normalized vector:**
$$
\hat{\mathbf{v}} = \left( \frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}} \right) \approx (0.577, 0.577, 0.577)
$$

This is the unit vector pointing equally in all three positive axis directions.

---

## The Zero Vector Problem

**Critical issue:** The zero vector $\mathbf{0} = (0, 0, 0)$ cannot be normalized.

$$
||\mathbf{0}|| = 0
$$

Division by zero is undefined. There is no "direction" for the zero vector.

**Handling:**
- Check if $||\mathbf{v}|| = 0$ (or $||\mathbf{v}||  \epsilon$ before dividing.

**2. Normalizing in-place incorrectly:**

Must compute magnitude first, then divide all components. Do not divide $v_x$, then use modified $v_x$ for magnitude.

**3. Assuming already normalized:**

Do not assume input vectors are normalized unless explicitly documented.

**4. Accumulating numerical errors:**

Re-normalize periodically in iterative algorithms.