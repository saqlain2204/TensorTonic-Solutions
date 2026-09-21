## What Is an Exponential Moving Average?

An exponential moving average (EMA) is a weighted average that gives more weight to recent observations. Weights decrease exponentially as observations get older.

$$
\text{EMA}_t = \alpha y_t + (1-\alpha) \text{EMA}_{t-1}
$$

where $0  0
$$

**Flat forecast:** EMA assumes no trend, predicts constant future.

**Use case:** Appropriate for stationary series with no trend.

**Limitation:** Poor for trending data. Use double exponential smoothing instead.

---

## Connection to Simple Exponential Smoothing

EMA is equivalent to simple exponential smoothing (SES):

**SES forecast:**

$$
\hat{y}_{t+1} = \alpha y_t + (1-\alpha)\hat{y}_t
$$

**Substituting $\hat{y}_t = \text{EMA}_{t-1}$:**

$$
\hat{y}_{t+1} = \alpha y_t + (1-\alpha)\text{EMA}_{t-1} = \text{EMA}_t
$$

**Interpretation:** EMA is both a smoothing technique and a forecasting method.

---

## Optimal Alpha Selection

**Minimize forecast error:**

$$
\alpha^* = \arg\min_{\alpha} \sum_{t=1}^{T} (y_t - \text{EMA}_{t-1})^2
$$

**Grid search:** Try $\alpha \in \{0.01, 0.02, ..., 0.99\}$, select best.

**Closed-form:** No analytical solution. Numerical optimization required.

**Cross-validation:** Split data, optimize on training set, validate on test set.

**Typical result:** $\alpha \in [0.1, 0.3]$ for most applications.

---

## Adaptive Exponential Smoothing

**Problem:** Fixed $\alpha$ may not suit all periods.

**Solution:** Let $\alpha$ vary over time.

**Example adaptive rule:**

$$
\alpha_t = |\frac{e_t}{y_t}|
$$

where $e_t = y_t - \text{EMA}_{t-1}$

**Interpretation:** Increase $\alpha$ when forecast errors are large (adapt quickly).

**Constraints:** Bound $\alpha_t$ to $[\alpha_{\min}, \alpha_{\max}]$ to prevent instability.

---

## Centered vs Trailing EMA

**Trailing EMA (standard):**

$$
\text{EMA}_t = \alpha y_t + (1-\alpha)\text{EMA}_{t-1}
$$

Uses only past values (causal).

**Centered EMA:**

Apply EMA forward and backward, then average.

$$
\text{EMA}_{\text{centered},t} = \frac{\text{EMA}_{\text{forward},t} + \text{EMA}_{\text{backward},t}}{2}
$$

**Advantage:** No lag, better smoothing.

**Disadvantage:** Requires future values (non-causal). Cannot forecast.

**Use:** Historical analysis, not real-time forecasting.

---

## Variance of EMA

For white noise input with variance $\sigma^2$:

$$
\text{Var}(\text{EMA}_t) = \frac{\alpha}{2-\alpha} \sigma^2
$$

**Higher $\alpha$:** Higher variance (less smoothing).

**Lower $\alpha$:** Lower variance (more smoothing).

**Variance reduction factor:**

$$
\frac{\text{Var}(\text{EMA}_t)}{\text{Var}(y_t)} = \frac{\alpha}{2-\alpha}
$$

**Example:** $\alpha = 0.2$ gives variance reduction to $\frac{0.2}{1.8} = 0.111$ (11.1% of original).

---

## Signal-to-Noise Ratio

**Goal:** Maximize signal retention while minimizing noise.

**Trade-off:** Low $\alpha$ removes more noise but also removes signal.

**Optimal choice depends on:**

- Signal frequency
- Noise characteristics
- Application requirements

**High SNR data:** Use higher $\alpha$ (less smoothing needed).

**Low SNR data:** Use lower $\alpha$ (more smoothing needed).

---

## EMA for Trend Detection

**EMA slope:**

$$
\text{Slope}_t = \text{EMA}_t - \text{EMA}_{t-1} = \alpha(y_t - \text{EMA}_{t-1})
$$

**Positive slope:** Upward trend.

**Negative slope:** Downward trend.

**Acceleration:**

$$
\text{Acceleration}_t = \text{Slope}_t - \text{Slope}_{t-1}
$$

**Monitoring:** Track slope changes for early trend reversal detection.

---

## Bollinger Bands with EMA

**Middle band:** EMA of price.

**Upper band:** $\text{EMA}_t + k \cdot \sigma_t$

**Lower band:** $\text{EMA}_t - k \cdot \sigma_t$

where $\sigma_t$ is rolling standard deviation and $k=2$ typically.

**Interpretation:**

- Price near upper band: Overbought
- Price near lower band: Oversold
- Squeeze (bands narrow): Low volatility, potential breakout

---

## Exponential Weighting in Other Contexts

**Exponentially weighted variance:**

$$
\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda) r_t^2
$$

Used in GARCH models for volatility.

**Exponentially weighted covariance:**

$$
\text{Cov}_t(x, y) = \lambda \text{Cov}_{t-1}(x, y) + (1-\lambda) x_t y_t
$$

**RiskMetrics:** $\lambda = 0.94$ for daily data.

**Principle:** Recent data more relevant for current risk.

---

## Double Exponential Smoothing Extension

**Problem:** EMA assumes no trend.

**Solution:** Add trend component (Holt's method).

$$
\ell_t = \alpha y_t + (1-\alpha)(\ell_{t-1} + b_{t-1})
$$

$$
b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)b_{t-1}
$$

**Forecast:**

$$
\hat{y}_{t+h} = \ell_t + h \cdot b_t
$$

Linear trend instead of flat.

---

## Triple Exponential Smoothing

**Holt-Winters method:**

Adds seasonal component to level and trend.

**Three equations:**

1. Level: $\ell_t = \alpha(y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + b_{t-1})$
2. Trend: $b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)b_{t-1}$
3. Seasonal: $s_t = \gamma(y_t - \ell_t) + (1-\gamma)s_{t-m}$

**Forecast:**

$$
\hat{y}_{t+h} = \ell_t + hb_t + s_{t+h-m}
$$

Handles trend and seasonality.

---

## Residual Analysis

**One-step forecast errors:**

$$
e_t = y_t - \text{EMA}_{t-1}
$$

**Good fit:** Residuals should be white noise.

**Diagnostic plots:**

1. Time plot of residuals (no patterns)
2. ACF of residuals (no significant autocorrelation)
3. Histogram of residuals (approximately normal)

**Ljung-Box test:** Test for autocorrelation in residuals.

**Heteroscedasticity:** Check if residual variance is constant.

---

## EMA in Volatility Modeling

**EWMA volatility:**

$$
\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda) r_t^2
$$

where $r_t$ is return at time $t$.

**RiskMetrics approach:** $\lambda = 0.94$ for daily returns.

**Interpretation:** Variance forecast adapts to recent squared returns.

**Application:** Value at Risk (VaR) calculations, portfolio risk management.

---

## Filter Interpretation

EMA is a discrete-time first-order IIR (infinite impulse response) filter:

**Transfer function:**

$$
H(z) = \frac{\alpha}{1 - (1-\alpha)z^{-1}}
$$

**Frequency response:**

Low-pass filter. Attenuates high-frequency components (noise).

**Cutoff frequency:** Depends on $\alpha$. Higher $\alpha$ has higher cutoff (less filtering).

---

## Outlier Robustness

**Problem:** Single outlier disproportionately affects EMA.

**Example:** Outlier $y_t = 1000$ when typical values are 100.

$$
\text{EMA}_t = 0.2(1000) + 0.8(100) = 280
$$

Large jump. Persists for several periods.

**Robust alternatives:**

1. Median-based smoothing
2. Trimmed mean exponential smoothing
3. Winsorization before applying EMA

**Trade-off:** Robustness vs sensitivity to legitimate changes.

---

## EMA for Time Series Decomposition

**Trend extraction:**

$$
\text{Trend}_t = \text{EMA}_t
$$

**Detrended series:**

$$
\text{Detrended}_t = y_t - \text{EMA}_t
$$

**Seasonal component:** Apply EMA to detrended series at seasonal lag.

**Residual:**

$$
\text{Residual}_t = y_t - \text{Trend}_t - \text{Seasonal}_t
$$

**Application:** Decompose into components for separate analysis.

---

## Practical Considerations

**Choice of $\alpha$:**

- Financial trading: $\alpha \in [0.1, 0.2]$ (smooth trends)
- Web analytics: $\alpha \in [0.2, 0.4]$ (responsive to changes)
- Sensor data: $\alpha \in [0.05, 0.15]$ (heavy smoothing)

**Initialization:** Use first observation or average of first 5-10 observations.

**Outliers:** Consider robust preprocessing before applying EMA.

**Trend:** If data has strong trend, use double exponential smoothing instead.

**Seasonality:** If seasonal patterns exist, use Holt-Winters method.