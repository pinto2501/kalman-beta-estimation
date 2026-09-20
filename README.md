# Intraday Beta Estimation Strategy Using a Kalman Filter

Standard CAPM assumes an equity's beta is constant. This project estimates beta dynamically against a benchmark index using a state-space model and a Kalman filter, updating the estimate recursively as new data arrives. The goal is a real-time beta (with uncertainty) for hedging, risk measurement, and regime-shift detection.

## Approach

Alpha and beta are treated as latent states inferred from excess returns:

$$r_{i,t} = \alpha_t + \beta_t\, r_{m,t} + \epsilon_t, \qquad \mathbf{x}_{t+1} = \Phi\,\mathbf{x}_t + \mathbf{w}_t, \quad \mathbf{x}_t = [\alpha_t,\ \beta_t]^\top$$

- **Transition:** random walk ($\Phi = I$) or mean-reverting ($\Phi < 1$)
- **Initialization:** $\alpha_0, \beta_0$ from OLS on historical returns
- **Tuning:** process noise $Q$ and measurement noise $R$ chosen by sequential cross-validation (e.g., 6-month train / 1-month validation), then stress-tested with $\pm 10\%$ to $\pm 50\%$ perturbations

## Data

| Source | Data |
|---|---|
| Polygon.io | Adjusted OHLCV for individual equities (daily/hourly) |
| Polygon.io / Yahoo Finance | S&P 500 returns, VIX |
| Treasury yields | Risk-free rate for excess returns |

Returns are log differences of adjusted closes, with timestamps aligned across securities and the benchmark.

## Evaluation

- **Benchmarks:** static full-sample CAPM beta; rolling-window OLS beta (60/90 days)
- **Metrics:** RMSE and predictive $R^2$
- **Stress test:** 2020 COVID-19 crash (airlines, hospitality, tech), comparing effects on VaR, Expected Shortfall, and hedge ratios
