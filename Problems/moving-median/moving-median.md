## What Is a Moving Median?

A moving median is the median of the most recent $w$ observations in a time series. It is a robust smoothing technique that resists outliers better than moving averages.

$$
\text{MM}_t(w) = \text{median}(y_{t-w+1}, y_{t-w+2}, ..., y_t)
$$

where $w$ is the window size.

---

## The Formula

For a time series $y_1, y_2, ..., y_T$, the moving median at time $t$ with window $w$ is:

$$
\text{MM}_t = \text{median}\{y_{t-w+1}, y_{t-w+2}, ..., y_t\}
$$

**Odd window size:** Median is the middle value.

**Even window size:** Median is the average of two middle values.

---

## Worked Example

**Time series:** [10, 15, 12, 100, 14, 13, 16]

**Window size:** $w = 3$

**Calculations:**

**$t=3$:** Median of {10, 15, 12}

Sorted: [10, 12, 15] → Median = 12

**$t=4$:** Median of {15, 12, 100}

Sorted: [12, 15, 100] → Median = 15

**$t=5$:** Median of {12, 100, 14}

Sorted: [12, 14, 100] → Median = 14

**$t=6$:** Median of {100, 14, 13}

Sorted: [13, 14, 100] → Median = 14

**$t=7$:** Median of {14, 13, 16}

Sorted: [13, 14, 16] → Median = 14

**Result:** [NaN, NaN, 12, 15, 14, 14, 14]

**Note:** Outlier 100 has minimal impact on median values.

---

## Comparison with Moving Average

**Same data with moving average ($w=3$):**

**$t=3$:** $(10+15+12)/3 = 12.33$

**$t=4$:** $(15+12+100)/3 = 42.33$ (heavily influenced by outlier)

**$t=5$:** $(12+100+14)/3 = 42.00$

**$t=6$:** $(100+14+13)/3 = 42.33$

**$t=7$:** $(14+13+16)/3 = 14.33$

**Observation:** Moving average distorted by outlier for multiple periods. Moving median remains stable.

---

## Robustness to Outliers

**Breakdown point:** Proportion of outliers that can be tolerated.

**Moving median:** Approximately 50% breakdown point.

Up to half of values in window can be outliers without severely affecting result.

**Moving average:** 0% breakdown point.

Single extreme outlier can arbitrarily distort average.

**Application:** Use moving median when data contains outliers, spikes, or measurement errors.

---

## Computational Complexity

**Naive approach:**

For each position, extract window, sort, find median.

Time: $O(w \log w)$ per position, $O(Tw \log w)$ total.

**Efficient approach:**

Maintain sorted structure (balanced tree or specialized median heap).

Time: $O(\log w)$ per update, $O(T \log w)$ total.

**Comparison to moving average:** $O(T)$ for moving average vs $O(T \log w)$ for moving median.

**Trade-off:** Slower but more robust.

---

## Window Size Selection

**Small window (e.g., $w=3$):**

- Responsive to changes
- Follows data closely
- Less smoothing

**Large window (e.g., $w=25$):**

- More smoothing
- Better outlier resistance
- Less responsive (lag)

**General guidance:**

- For outlier removal: $w \geq 5$
- For trend extraction: $w \geq 10$
- Odd $w$ preferred (unique median without averaging)

---

## Edge Handling

**At the start ($t  k \cdot \text{MAD}_t
$$

Typical: $k \approx 3$ for anomaly flagging.

---

## Seasonality and Moving Median

**Seasonal adjustment:**

Use window size equal to seasonal period.

**Monthly data with yearly seasonality:** $w = 12$

**Centered moving median:**

$$
\text{CMM}_t = \text{median}(y_{t-6}, ..., y_t, ..., y_{t+6})
$$

**Effect:** Removes seasonal pattern, preserves trend.

**Note:** Requires future values (non-causal), suitable for historical analysis only.

---

## Quantile Extensions

**Moving percentiles:**

Generalization of moving median to other percentiles.

**25th percentile (Q1):**

$$
Q1_t = \text{P}_{25}(y_{t-w+1}, ..., y_t)
$$

**75th percentile (Q3):**

$$
Q3_t = \text{P}_{75}(y_{t-w+1}, ..., y_t)
$$

**Interquartile range:**

$$
\text{IQR}_t = Q3_t - Q1_t
$$

**Application:** Robust confidence bands, volatility estimation.

---

## Mode vs Median

**Mode:** Most frequent value in window.

**Median:** Middle value in sorted window.

**For continuous data:** Mode often undefined (no repeated values).

**For discrete data:** Mode can be useful.

**Example:** [1, 2, 2, 3, 2, 5] → Mode = 2, Median = 2

**Moving mode:** Less common than moving median, but useful for categorical time series.

---

## Weighted Median

**Standard median:** Equal weight to all observations.

**Weighted median:** Assign weights, find value where cumulative weight = 50%.

$$
\text{WM}_t = \arg\min_m \sum_{i=t-w+1}^{t} w_i |y_i - m|
$$

**Application:** Give more importance to recent observations.

**Example weights:** Exponentially decaying weights $w_i = \alpha^{t-i}$.

---

## Residual Analysis

**Detrended series:**

$$
r_t = y_t - \text{MM}_t
$$

**Analysis:**

- Plot residuals to check for patterns
- Residuals should be approximately symmetric around zero
- Large residuals indicate outliers or structural breaks

**Use case:** Identify anomalies after removing baseline trend.

---

## Comparison to Percentile Filters

**Minimum filter:** 0th percentile (moving minimum).

**Maximum filter:** 100th percentile (moving maximum).

**Median filter:** 50th percentile (moving median).

**Range filter:** Difference between max and min.

**Applications:**

- Minimum: Lower envelope
- Maximum: Upper envelope
- Range: Local volatility measure

---

## Non-Uniform Weighting

**Time-weighted median:**

Recent observations weighted more heavily.

**Implementation:**

Replicate recent values in window before computing median.

**Example:** Weight pattern [1, 2, 3] with $w=6$

Replicate: [value1, value2, value2, value3, value3, value3]

Compute median of replicated window.

**Effect:** Biases median toward recent values.

---

## Edge Preservation

**Key advantage:** Preserves sharp transitions.

**Example:** Step function $y_t = \begin{cases} 0 & t  slow, short when fast < slow.

---

## Seasonal Decomposition

**Trend extraction with moving median:**

$$
\text{Trend}_t = \text{MM}_t(w = s)
$$

where $s$ is seasonal period.

**Detrended:**

$$
D_t = y_t - \text{Trend}_t
$$

**Seasonal component:** Average $D_t$ for each season.

**Remainder:** $y_t - \text{Trend}_t - \text{Seasonal}_t$

**Robust STL decomposition:** Uses moving median instead of moving average for robustness.

---

## Multi-Scale Analysis

**Pyramid approach:**

Apply moving median at multiple window sizes.

**Small window:** Captures high-frequency variations.

**Medium window:** Captures mid-frequency trends.

**Large window:** Captures low-frequency baseline.

**Interpretation:** Decompose signal into components at different scales.

**Application:** Multi-resolution analysis, wavelet-like decomposition using medians.

---

## Practical Considerations

**Data type:** Works with any orderable data (numeric, ordinal categorical).

**Window size:** Odd preferred for unique median, even requires averaging.

**Computational cost:** Slower than moving average but faster than many robust alternatives.

**Outlier definition:** Depends on context. Median identifies values different from local neighborhood.

**Implementation:** Most libraries provide built-in functions (pandas, scipy, R).