# Mathematics, Statistics and Probability

A complete account of every quantitative method used in this project, why it was
chosen, and the reasoning behind the way it is applied.

The system detects latent *market regimes* from a price series and trades them.
Three modelling paths exist: a Gaussian Hidden Markov Model, a Kalman Filter, and
a combined Kalman → HMM pipeline. Everything downstream — position sizing,
backtesting, performance measurement — is shared.

---

## 1. Preprocessing: Log Returns

`data/csv_loader.py`, `data/api_loader.py`

```python
data['Log_Return'] = np.log(prices / prices.shift(1))
```

$$r_t = \ln\!\left(\frac{P_t}{P_{t-1}}\right)$$

Prices are chosen as `Adj Close` when available, otherwise `Close`. Adjusted
close folds dividends and splits back into the series, so a 2-for-1 split does
not appear as a −69% return.

**Why log returns and not simple returns:**

1. **Time-additivity.** $\ln(P_T/P_0) = \sum_t r_t$. Multi-period aggregation
   becomes a sum, which is why every equity curve in the codebase is
   `np.exp(series.cumsum())` — one cheap cumulative sum instead of a compounding
   product.
2. **Approximate stationarity.** Prices are non-stationary (they trend, and
   their variance grows with level). Differencing in log space removes the unit
   root and produces a series whose mean and variance are roughly stable — the
   precondition for fitting a stationary HMM or a linear-Gaussian state-space
   model.
3. **Symmetry.** A +10% move followed by a −10% move is not a round trip in
   simple returns but is exactly zero in log space. This eliminates a
   directional bias in mean estimation.
4. **Closer to Gaussian.** Simple returns are bounded below by −1 and unbounded
   above, so they are right-skewed. Log returns are unbounded in both directions.
   Both models here assume Gaussian emissions, so this matters directly for
   likelihood correctness.

The first observation is `NaN` by construction (no prior price) and is dropped.

---

## 2. The Gaussian Hidden Markov Model

`models/hmm.py`

### 2.1 The probability model

Let $z_t \in \{1,\dots,K\}$ be the unobserved regime at time $t$ and $r_t$ the
observed log return. The model asserts:

$$P(z_t \mid z_{t-1}, z_{t-2}, \dots) = P(z_t \mid z_{t-1}) = A_{z_{t-1}, z_t}$$

$$r_t \mid z_t = k \;\sim\; \mathcal{N}(\mu_k,\ \sigma_k^2)$$

with initial distribution $\pi_k = P(z_1 = k)$. The joint likelihood is

$$P(r_{1:T}, z_{1:T}) = \pi_{z_1}\, \mathcal{N}(r_1 \mid \mu_{z_1}, \sigma^2_{z_1}) \prod_{t=2}^{T} A_{z_{t-1},z_t}\, \mathcal{N}(r_t \mid \mu_{z_t}, \sigma^2_{z_t})$$

**Why this model for markets.** Financial returns exhibit volatility clustering:
calm periods and turbulent periods arrive in runs, not independently. A single
Gaussian cannot represent this — it would fit one average volatility and be wrong
in both regimes. A mixture of Gaussians captures the fat-tailed unconditional
distribution but assumes independence across time, which throws away the
clustering. The Markov chain is precisely the minimal addition that encodes
*persistence*: the diagonal of $A$ measures how sticky each regime is, and
expected duration in state $k$ is $1/(1 - A_{kk})$.

The latent variable is not a mathematical convenience — "bull market" and "crisis"
are genuinely unobserved. We only ever see returns, never the regime label.

### 2.2 Fitting: Baum–Welch (EM)

```python
X = data.values.reshape(-1, 1)
self.model.fit(X)                          # Baum-Welch
self.log_likelihood = self.model.score(X)  # forward algorithm
```

The regimes are unobserved, so we cannot maximise the complete-data likelihood
directly. Baum–Welch is Expectation-Maximisation specialised to HMMs:

- **E-step**: run forward–backward to compute the posterior state occupancies
  $\gamma_t(k) = P(z_t = k \mid r_{1:T})$ and pairwise transition posteriors
  $\xi_t(j,k) = P(z_t=j, z_{t+1}=k \mid r_{1:T})$.
- **M-step**: re-estimate parameters as posterior-weighted moments —
  $\mu_k = \sum_t \gamma_t(k) r_t / \sum_t \gamma_t(k)$, and analogously for
  $\sigma_k^2$, with $A_{jk} \propto \sum_t \xi_t(j,k)$.

EM guarantees the observed-data log-likelihood is **monotonically
non-decreasing** at every iteration. That guarantee is what makes the convergence
plot in the dashboard meaningful: `self.history = self.model.monitor_.history`
records the log-likelihood trace, and a curve that has visibly flattened is
evidence that `n_iter` was large enough. A curve still rising at the last
iteration means the fit was truncated.

EM converges only to a **local** optimum. `random_state=42` fixes the
initialisation so results are reproducible; the tuning routine deliberately
varies `random_state=i` across candidates, which doubles as a crude multi-start
search over the likelihood surface.

`score(X)` evaluates $\ln P(r_{1:T})$ by the forward algorithm, which marginalises
over all $K^T$ possible state paths in $O(TK^2)$ via dynamic programming.

### 2.3 Decoding: Viterbi vs. posteriors

Two different questions, two different algorithms, both exposed:

```python
def predict_regimes(self, data):        # Viterbi
    return self.model.predict(X)

def get_state_probabilities(self, data): # forward-backward
    return self.model.predict_proba(X)
```

- **Viterbi** (`predict`) returns $\arg\max_{z_{1:T}} P(z_{1:T} \mid r_{1:T})$ —
  the single most likely *sequence*. It respects transition probabilities
  globally, so it will not emit a one-day regime flip that the transition matrix
  considers implausible. This is what drives trading, because a trading system
  needs one committed label per day.

- **Forward–backward posteriors** (`predict_proba`) give the marginal
  $P(z_t = k \mid r_{1:T})$ at each $t$ independently. These are used for
  visualising *confidence*. A day where the top posterior is 0.95 and a day where
  it is 0.40 both produce a hard Viterbi label, but they mean very different
  things about how much to trust it.

The distinction matters: the sequence of per-timestep marginal argmaxes is not
generally the same as the Viterbi path, and can even be a path of probability
zero if it traverses a forbidden transition.

### 2.4 Parameter count

```python
n_params = self.n_components - 1 \
         + self.n_components * (self.n_components - 1) \
         + 2 * self.n_components
```

$$k = \underbrace{(K-1)}_{\text{initial }\pi} + \underbrace{K(K-1)}_{\text{transitions}} + \underbrace{2K}_{\text{emissions}}$$

Each term counts *free* parameters after constraints:

- $\pi$ has $K$ entries but sums to 1 → $K-1$ free.
- $A$ has $K^2$ entries but **each row** sums to 1 → $K(K-1)$ free.
- Emissions are univariate Gaussians: one $\mu_k$ and one $\sigma_k^2$ per state
  → $2K$.

For $K=3$: $2 + 6 + 6 = 14$ parameters. This count feeds AIC/BIC (§5) and is the
only reason those criteria are comparable across different $K$.

Note the $2K$ term hard-codes the univariate case. It is correct for the single
`Log_Return` feature used here, but adding a second observed feature would make
it wrong (a $d$-dimensional full-covariance Gaussian has $d + d(d+1)/2$
parameters per state).

---

## 3. The Kalman Filter

`models/kalman.py`

### 3.1 The state-space model

The defaults define a **local level model** (random walk plus noise):

```python
transition_matrices    = ones((1,1))         # A = 1
observation_matrices   = ones((1,1))         # H = 1
transition_covariance  = ones((1,1)) * 0.01  # Q
observation_covariance = ones((1,1))         # R
```

$$x_t = A x_{t-1} + w_t, \qquad w_t \sim \mathcal{N}(0, Q)$$
$$r_t = H x_t + v_t, \qquad\;\; v_t \sim \mathcal{N}(0, R)$$

With $A = H = 1$ this says: *there is a latent "true" drift $x_t$ that wanders as
a random walk, and the return we observe is that drift plus noise.* The filter's
job is to recover $x_t$ from the contaminated $r_t$.

**Why a random walk and not a mean-reverting AR(1)?** Setting $A=1$ is an
explicitly agnostic prior — it assumes no known long-run level to revert to. For
market drift, which is not stably anchored to any particular value, this is the
honest default. It also makes the filter an adaptive exponentially-weighted mean
rather than something that pulls toward a possibly-wrong constant.

### 3.2 The recursion

Two half-steps per observation:

**Predict** (propagate belief forward without data):
$$\hat{x}_{t|t-1} = A\hat{x}_{t-1|t-1}, \qquad P_{t|t-1} = A P_{t-1|t-1} A^\top + Q$$

**Update** (fold in the new observation):
$$\tilde{y}_t = r_t - H\hat{x}_{t|t-1} \quad \text{(innovation)}$$
$$S_t = H P_{t|t-1} H^\top + R \quad \text{(innovation covariance)}$$
$$K_t = P_{t|t-1} H^\top S_t^{-1} \quad \text{(Kalman gain)}$$
$$\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t \tilde{y}_t, \qquad P_{t|t} = (I - K_t H) P_{t|t-1}$$

This is exact recursive Bayesian inference. Because the model is linear and all
noise is Gaussian, the posterior stays Gaussian forever and is fully described by
$(\hat{x}, P)$ — no approximation, no particle cloud, $O(1)$ memory per step.

Under the Gaussian assumption the Kalman Filter is the **minimum mean-squared-error
estimator**, not merely the best linear one.

### 3.3 The Q/R ratio is the only tuning knob that matters

The gain $K_t$ interpolates between two extremes:

- $Q/R \to 0$: the state is believed nearly constant, observations are believed
  noisy → $K_t \to 0$ → heavy smoothing, slow adaptation.
- $Q/R \to \infty$: the state is believed volatile, observations trusted →
  $K_t \to 1$ → output tracks the raw input, no smoothing.

The default $Q/R = 0.01/1.0 = 0.01$ is deliberately at the smoothing end. Daily
log returns have an extremely low signal-to-noise ratio — the day-to-day drift
signal is tiny relative to the noise — so aggressive smoothing is the correct
prior. The tuning grid explores `transition_covariance ∈ {0.01, 0.05, 0.1}`
against `observation_covariance ∈ {0.5, 1.0, 2.0}`, i.e. $Q/R$ from 0.005 to 0.2,
staying entirely within the smoothing regime.

### 3.4 Filtering, not smoothing — a deliberate choice

```python
def filter(self, data):
    filtered_state_means, filtered_state_covariances = self.model.filter(X)
```

pykalman offers both `filter()` (forward pass, $\hat{x}_{t|t}$, uses data up to
$t$) and `smooth()` (forward–backward, $\hat{x}_{t|T}$, uses *all* data
including the future). The code uses `filter()` exclusively.

**This is the single most important line in the file for backtest validity.** The
smoother produces a visibly nicer curve, but $\hat{x}_{t|T}$ at any historical
date incorporates information that did not exist on that date. Trading on it
would manufacture returns that are unobtainable in reality. Filtering is causal;
smoothing is not.

### 3.5 Likelihood and online updating

`loglikelihood(X)` uses the prediction-error decomposition: the innovations
$\tilde{y}_t$ are, under a correctly specified model, independent
$\mathcal{N}(0, S_t)$, so

$$\ln P(r_{1:T}) = -\tfrac{1}{2}\sum_{t=1}^{T}\left(\ln 2\pi + \ln|S_t| + \tilde{y}_t^\top S_t^{-1}\tilde{y}_t\right)$$

The `em(X, n_iter=20)` call maximises this over $Q$, $R$ and the initial
conditions.

```python
def update_online(self, new_data, current_mean, current_cov):
    return self.model.filter_update(...)
```

`filter_update` performs exactly one predict+update pair. This exists because the
recursion is genuinely sequential: incorporating tomorrow's price costs $O(1)$ and
requires only $(\hat{x}_t, P_t)$, never the historical series. This is the
structural advantage the Kalman Filter has over the HMM, whose `update_online`
correspondingly raises `NotImplementedError` — batch EM has no cheap incremental
form.

`n_params = 6` is hard-coded for AIC/BIC. It is an approximation; it does not
provably match the number of quantities `em` actually re-estimates.

---

## 4. The Combined Model: Kalman → HMM

`app.py`

```python
kalman.fit(train_data['Log_Return'])
filtered_means, _ = kalman.filter(clean_data['Log_Return'])
smoothed_returns = pd.Series(filtered_means.flatten(), index=clean_data.index)
hmm.fit(smoothed_returns.iloc[:split_idx])
```

**The logic.** The HMM's Gaussian emissions must explain *all* variation in the
input. Fed raw returns, a large share of the fitted variance is microstructure
noise, and regime boundaries get placed on noise excursions. Running the Kalman
Filter first strips the high-frequency component, so the HMM segments the
persistent drift instead. It is a denoise-then-segment cascade: continuous state
estimation followed by discrete state classification.

### 4.1 The variance-shrinkage hazard

```python
# Regimes come from the smoothed series, but position decisions must be
# based on RAW returns: smoothing shrinks volatility and would inflate
# the t-statistic, defeating the quality filters.
regime_positions, regime_decisions = build_regime_positions(
    returns=train_data['Log_Return'],       # RAW, not smoothed
    regimes=regimes[:split_idx], ...)
```

This separation is essential and easy to get wrong. Kalman filtering reduces the
variance of the series by construction. If the smoothed series were passed to the
statistical quality gates, then:

$$t = \frac{\bar{r}}{\hat\sigma/\sqrt{n}}$$

would have a deflated denominator, inflating $|t|$ and letting regimes pass the
significance test on the strength of an artefact of the smoother. The filter is
allowed to decide *where* regime boundaries fall; it is not allowed to influence
*whether* a regime is tradeable. Raw returns are also what the strategy actually
earns, so they are the only correct basis for expected-return estimates.

---

## 5. Model Selection: AIC and BIC

`utils/evaluation.py`

```python
aic = 2 * n_params - 2 * log_likelihood
bic = n_params * np.log(n_samples) - 2 * log_likelihood
```

$$\mathrm{AIC} = 2k - 2\ln\hat{L}, \qquad \mathrm{BIC} = k\ln n - 2\ln\hat{L}$$

Log-likelihood alone cannot select $K$: adding states can only increase $\ln\hat L$,
so unpenalised maximisation always picks the most complex model, which overfits.
Both criteria add a complexity penalty; they differ in what they are trying to do.

- **AIC** ($2k$) estimates out-of-sample predictive loss (Kullback–Leibler
  divergence to the truth). It is asymptotically efficient but not consistent —
  it tends to over-select as $n$ grows.
- **BIC** ($k \ln n$) approximates the Bayesian model evidence and is consistent:
  given enough data it selects the true model if it is in the candidate set.

For any $n > e^2 \approx 7.4$ — i.e. always, for daily financial data — BIC's
penalty is strictly harsher. With ~2500 observations, $\ln n \approx 7.8$, so BIC
charges nearly four times as much per parameter as AIC.

**Why HMM tuning minimises BIC:**

```python
if bic < best_bic:
    best_bic = bic
    best_params = params
```

The tuning grid varies `n_components ∈ {3,4,5}`, so $k$ genuinely varies across
candidates and the penalty does real work. The harsher penalty is the right
choice here because regimes are only useful if they are *interpretable and
persistent*. A 5-state model can always fit training data better than a 3-state
model, but the extra states usually turn out to be transient splinters that do
not recur out of sample. BIC's bias toward parsimony directly serves the
objective.

**Why Kalman tuning maximises raw log-likelihood instead** — an asymmetry worth
naming:

```python
ll = temp_model.loglikelihood(X)
if ll > best_ll: ...
```

The Kalman grid varies only $Q$, $R$ and initial conditions; the model structure,
and therefore $k = 6$, is constant across every candidate. When $k$ is fixed, both
penalty terms are constant, so ranking by $\ln\hat L$, AIC or BIC produces
*identical* orderings. The asymmetry is cosmetic, not a defect — but it would
become one if `n_components`-style structural variation were ever added to that
grid.

---

## 6. From Regimes to Positions

`utils/evaluation.py::build_regime_positions`

### 6.1 The label-switching problem

HMM state indices are arbitrary. "Regime 0" carries no inherent meaning; a refit
with a different seed may relabel the same economic state as "Regime 2". Therefore
the regime → position map must be **learned from training data and frozen**, never
assumed. This function is what learns it.

Critically, it is called with training data only:

```python
returns=train_data['Log_Return'], regimes=regimes[:split_idx]
```

If it saw test data, the sign of every position would have been chosen with
knowledge of the outcome it is about to be evaluated on — the most severe form of
look-ahead bias possible in this design.

### 6.2 Per-regime statistics

For each regime $k$, take the subset of training returns where $z_t = k$:

$$\hat\mu_k^{ann} = \bar{r}_k \times 252, \qquad
  \hat\sigma_k^{ann} = s_k \times \sqrt{252}, \qquad
  \mathrm{Sharpe}_k = \frac{\hat\mu_k^{ann}}{\hat\sigma_k^{ann}}$$

The $\sqrt{252}$ on volatility follows from variance being additive over
independent periods: $\mathrm{Var}(\sum_{i=1}^{252} r_i) = 252\sigma^2$, so the
standard deviation scales as the square root. The mean scales linearly. 252 is the
conventional count of US trading days per year.

Because the numerator scales by $252$ and the denominator by $\sqrt{252}$, the
Sharpe ratio scales by $\sqrt{252}$ — this is why the annualised Sharpe is
$\sqrt{252} \cdot (\text{daily mean}/\text{daily std})$.

### 6.3 The three quality gates

A regime is traded directionally **only if it passes all three tests**; otherwise
it is held in cash. The default in a noisy domain should be *no position*, and
that is what the code does.

```python
if abs(ann_return) < min_abs_ann_return:  → Cash, 'Mean return too close to zero'
elif ann_vol > max_ann_volatility:        → Cash, 'Volatility above limit'
elif abs(tstat) < min_abs_tstat:          → Cash, 'Mean not statistically distinct from zero'
else:                                     → Long if ann_return > 0 else Short
```

Plus a guard: a regime observed fewer than twice in training gets no position at
all, since the sample standard deviation is undefined for $n < 2$.

**Gate 1 — economic significance.** A regime whose annualised mean is +0.5% is
not worth the transaction costs of trading, regardless of its statistics.

**Gate 2 — risk tolerance.** A regime may have a genuinely positive mean and still
be untradeable if its volatility implies unacceptable drawdowns at the chosen
leverage.

**Gate 3 — statistical significance.** This is the critical one. The one-sample
t-statistic of the regime mean against zero:

$$t_k = \frac{\bar{r}_k}{s_k / \sqrt{n_k}}$$

testing $H_0: \mu_k = 0$ against $H_1: \mu_k \neq 0$.

The source comments state the rationale directly: *"a genuinely flat regime can
easily show a large sample mean by chance, and |t| accounts for sample size in a
way that the Sharpe ratio alone does not."*

This is the key insight of the whole position-sizing layer. Consider a regime
occupying 15 days with an annualised mean of +40% and volatility of 25% — a
Sharpe of 1.6, which looks superb. But
$t = 1.6 \times \sqrt{15/252} \approx 0.39$: entirely consistent with pure noise.
Sharpe is a *ratio of population parameters* and says nothing about how precisely
they were estimated. The t-statistic divides by the **standard error**
$s/\sqrt{n}$ rather than the standard deviation $s$, and is therefore the only one
of the two that penalises small samples.

The relationship is exact:
$$t_k = \mathrm{Sharpe}_k \times \sqrt{n_k / 252}$$

A high Sharpe over a short window and a modest Sharpe over a long window can yield
the same $t$ — and the t-statistic is right to treat them as equally credible.

The default threshold is `min_abs_tstat = 2.0`, matching the UI caption: *"|t| ≈ 2
means the regime's mean return is distinguishable from zero at roughly 95%
confidence."* For large $n$ the t-distribution converges to the standard normal,
whose two-sided 5% critical value is 1.96.

One honest caveat: applying this test to $K$ regimes simultaneously is multiple
hypothesis testing. With $K=5$ and $\alpha=0.05$ per test, the family-wise
probability of at least one false positive is $1 - 0.95^5 \approx 23\%$. Raising
the threshold above 2.0 is the practical mitigation, and the slider allows it.

The final sign is taken from the sign of the training mean, subject to the
`trading_direction` constraint ("Long & Short" / "Long Only" / "Short Only"), and
scaled by `leverage`.

Every decision is recorded with a human-readable `Reason` in the returned
diagnostics table. The system is designed to explain *why* it is flat, not merely
that it is.

---

## 7. The Regime Confirmation Filter

`utils/evaluation.py::apply_regime_filter`

```python
if min_days <= 1:
    return regimes.copy()
```

A new regime label is adopted only after it has been predicted for `min_days`
consecutive observations.

**Why.** Viterbi output near a regime boundary can flicker. Each flicker is a
round-trip trade. With a proportional cost $c$, a long→short reversal costs $2c$
in turnover, so a handful of spurious flips can consume the entire edge. The
filter is a hysteresis band around the regime decision.

**The trade-off is explicit and unavoidable:** the filter introduces up to
`min_days` of recognition lag. Entering a genuine crash three days late has a real
cost. `min_days` is exposed as a slider (1–10) precisely because the optimum
depends on the transaction cost level — the two parameters must be tuned together.

Implementation detail that matters: the filter is **causal**. It scans forward,
tracking a `candidate` label and a `streak` counter, and only writes the confirmed
`current` label. At no point does it consult `regimes[i+1]`. A centred smoother
would have been simpler and would have destroyed the backtest.

---

## 8. Backtest Mathematics

`utils/evaluation.py::backtest_regime_strategy`

### 8.1 Look-ahead prevention

```python
df['Position'] = df['Position'].shift(1).fillna(0)
```

The regime for day $t$ is inferred from day $t$'s return. Trading that return
would mean acting on information revealed by the very bar being traded — a
guaranteed-profit artefact. The one-period shift enforces:

$$\text{position}_t = f(\text{information available at } t-1)$$

### 8.2 Transaction costs

```python
df['Trade'] = df['Position'].diff().abs().fillna(0)
df['TC'] = df['Trade'] * transaction_cost
```

Turnover is $|\Delta\text{position}_t|$. This correctly charges a long→short
reversal at 2× leverage as $|{-2} - 2| = 4$ units, whereas a naive "1 unit per
trade" model would charge 1. The UI expresses cost in basis points:
`transaction_cost = bps / 10000.0`.

### 8.3 Leverage requires leaving log space

```python
simple_return = np.exp(df['Log_Return']) - 1
leveraged_simple_return = df['Position'] * simple_return - df['TC']
leveraged_simple_return = leveraged_simple_return.clip(lower=-0.9999)
df['Strategy_Return'] = np.log1p(leveraged_simple_return)
```

This three-step conversion is mathematically necessary and a common source of
error in backtests.

**Log returns are not proportionally scalable.** For leverage $L$,
$L \cdot r^{\log} \neq \ln(1 + L(e^{r^{\log}} - 1))$ except at $L = 1$. Simply
multiplying the log return by 2 does *not* describe a 2× leveraged position — it
describes squaring the gross return, which is a different and more favourable
thing. It systematically understates leveraged losses.

Leverage is a proportional claim on the *simple* return, so:

1. $r^{simp} = e^{r^{\log}} - 1$
2. $r^{strat,simp} = L \cdot r^{simp} - \text{cost}$
3. $r^{strat,\log} = \ln(1 + r^{strat,simp})$

`np.log1p(x)` computes $\ln(1+x)$ with better numerical precision than
`np.log(1+x)` for small $x$ — relevant here since daily returns are typically
$O(10^{-3})$.

**The clip.** At $L = 5$, a −25% day gives $r^{strat,simp} = -1.25$, i.e. equity
below zero. $\ln$ of a non-positive number is undefined. Clipping at −0.9999
represents a −99.99% day: effectively total ruin, which is the honest outcome, but
numerically representable so the equity curve survives to display it. The clip is
a numerical guard, not a rescue — it does not hide the wipeout, it just lets the
chart render.

### 8.4 Equity curves

```python
df['Buy_Hold_Equity'] = np.exp(df['Log_Return'].cumsum())
df['Strategy_Equity'] = np.exp(df['Strategy_Return'].cumsum())
```

Direct use of time-additivity: $\prod_t (1 + r_t^{simp}) = \exp\left(\sum_t r_t^{\log}\right)$.
Both curves start at ~1.0, making them directly comparable. Buy & Hold is the
benchmark; a strategy that fails to beat it after costs has produced nothing but
risk.

---

## 9. Performance Metrics

`utils/evaluation.py`

### 9.1 Sharpe ratio

```python
excess_returns = returns - (risk_free_rate / 252)
return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
```

$$S = \sqrt{252}\;\frac{\overline{r - r_f/252}}{s_{r - r_f/252}}$$

Return per unit of volatility. The annual risk-free rate is de-annualised to daily
by dividing by 252 (simple, not geometric — an acceptable approximation at these
magnitudes). The $\sqrt{252}$ factor follows from §6.2.

Guard: a zero-variance series (always flat, never traded) returns 0.0 rather than
dividing by zero.

**Its limitation, which is why it is not used alone.** Sharpe penalises upside and
downside volatility identically, and assumes returns are roughly Gaussian and
i.i.d. Regime strategies violate the i.i.d. assumption by construction — that is
the entire premise. This is why the dashboard reports drawdown alongside it.

### 9.2 Maximum drawdown

```python
rolling_max = equity_curve.cummax()
drawdowns = (equity_curve - rolling_max) / rolling_max
return drawdowns.min()
```

$$\mathrm{MDD} = \min_t \frac{V_t - \max_{s \le t} V_s}{\max_{s \le t} V_s}$$

The worst peak-to-trough decline. The running maximum is the *causal* high-water
mark — computed from history only, so it is the drawdown an investor actually
experienced in real time.

Drawdown captures the risk Sharpe misses: path dependence. Two strategies with
identical mean and variance can have very different worst losses depending on
whether their bad days cluster. And clustering is exactly what regimes are about.
For a leveraged strategy this is the binding constraint — a −60% drawdown may end
the strategy regardless of its long-run expectation.

### 9.3 Annualised return (CAGR)

```python
total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
years = len(equity_curve) / 252.0
if total_return <= 0:
    return -1.0
return (total_return ** (1 / years)) - 1
```

$$\mathrm{CAGR} = \left(\frac{V_T}{V_0}\right)^{1/y} - 1, \qquad y = T/252$$

The geometric mean, which is the correct aggregation for compounded returns. The
arithmetic mean would overstate performance: +50% followed by −50% averages to 0%
arithmetically but is −13.4% annualised — and −13.4% is what the account balance
shows.

The guard returns −1.0 (total loss) for a non-positive final equity, since a
fractional power of a negative number is not real.

---

## 10. Experimental Design and Look-Ahead Control

`app.py`

The most consequential statistical decisions in this project are not formulas.
They concern *what information is available when*.

### 10.1 Chronological split

```python
split_idx = int(len(clean_data) * (train_size_pct / 100.0))
train_data = clean_data.iloc[:split_idx]
test_data  = clean_data.iloc[split_idx:]
```

Never randomised. Random k-fold splitting on a time series leaks the future into
the past in the most direct way possible — training on 2023 to predict 2021 is not
a forecast. The split is a hard chronological wall.

### 10.2 Split Viterbi decoding

```python
# Predict train/test separately to avoid Viterbi look-ahead bias
train_regimes = model.predict_regimes(train_data['Log_Return'])
if train_size_pct < 100:
    test_regimes = model.predict_regimes(test_data['Log_Return'])
    regimes = np.concatenate([train_regimes, test_regimes])
```

A subtle and important point. Viterbi is a *global* algorithm — it optimises over
the entire sequence, so decoding train and test together would let test-period
returns influence the labels assigned to training days, and vice versa. Decoding
the two blocks separately breaks that channel.

**Stated honestly:** this is block-causal, not strictly online. Decoding the test
block still uses the whole test block, so the label for the first test day is
informed by the last. A fully realistic simulation would re-decode with an
expanding window at every step. This is a deliberate cost/realism trade-off, and
the residual optimism it introduces should be assumed to exist.

### 10.3 The full information hygiene chain

Four independent barriers, each blocking a different leak:

| Barrier | Mechanism | Leak prevented |
|---|---|---|
| Chronological split | `iloc[:split_idx]` | Training on the future |
| Model fitting on train only | `model.fit(train_data)` | Parameters shaped by test data |
| Position map from train only | `build_regime_positions(train_data...)` | Regime signs chosen with hindsight |
| Position shift | `.shift(1)` | Trading the bar that generated the signal |
| Kalman `filter` not `smooth` | forward pass only | State estimates using future prices |

Evaluation is then restricted to the out-of-sample block:

```python
eval_data = test_data if train_size_pct < 100 else clean_data
eval_regimes = regimes[split_idx:] if train_size_pct < 100 else regimes
```

Setting `train_size_pct = 100` evaluates in-sample. That mode exists for
diagnostics and model inspection; its performance numbers are not evidence of
predictive ability.

---

## 11. Known Limitations

Stated plainly, because a quantitative document that only lists strengths is not
a quantitative document.

**Gaussian emissions understate tail risk.** Real returns are leptokurtic. The
regime mixture recovers some excess kurtosis unconditionally, but *within* a
regime the model still assumes normality, so extreme single-day moves are
assigned lower probability than they deserve. A Student-t emission would be more
faithful.

**Parameter counts are approximate.** The HMM's $2K$ emission term hard-codes
univariate observations. The Kalman's `n_params = 6` is a fixed guess. AIC/BIC
comparisons remain valid *within* a model family — where the error is either
constant or correct — but not across families.

**`covariance_type` is a no-op here.** The tuning grid varies it between `"full"`
and `"diag"`, but with one-dimensional observations a covariance matrix is a
single scalar and the two are identical. That grid dimension does nothing but
consume search iterations.

**Kalman tuning grid inconsistencies.** Candidate models are constructed without
`transition_matrices` / `observation_matrices`, relying on pykalman's defaults
rather than the class's own, and the grid supplies covariances as `[1.0]` where
the constructor elsewhere uses `[[1.0]]`. The tuner works, but it is not exploring
exactly the same parameterisation the fitted model uses.

**EM finds local optima.** Both models can converge to a poor solution depending
on initialisation. The HMM tuner's varying `random_state` mitigates this
incidentally; the Kalman path does not.

**Regime persistence is assumed stable.** A first-order Markov chain has
geometrically-distributed state durations. Real regimes may have duration
distributions the model cannot represent (a semi-Markov / HSMM formulation would
handle this).

**Multiple testing across regimes.** As noted in §6.3, testing $K$ regimes at 95%
confidence each does not yield 95% family-wise confidence.

**Backtests are optimistic by nature.** No slippage model, no market impact, no
borrow costs for shorts, no financing costs for leverage, and perfect fills at the
close are all assumed. Transaction cost is the only friction modelled.

---

## Notation Summary

| Symbol | Meaning |
|---|---|
| $P_t$ | Price at time $t$ (adjusted close where available) |
| $r_t$ | Log return, $\ln(P_t/P_{t-1})$ |
| $z_t$ | Latent regime label at time $t$ |
| $K$ | Number of regimes (`n_components`) |
| $A$ | HMM transition matrix, $A_{jk} = P(z_t = k \mid z_{t-1} = j)$ |
| $\pi$ | HMM initial state distribution |
| $\mu_k, \sigma_k^2$ | Emission mean and variance for regime $k$ |
| $x_t$ | Kalman latent state (filtered level) |
| $Q, R$ | Process and observation noise covariance |
| $K_t$ | Kalman gain |
| $\tilde{y}_t, S_t$ | Innovation and its covariance |
| $\hat{L}$ | Maximised likelihood |
| $k$ | Free parameter count |
| $n, T$ | Number of observations |
| 252 | Trading days per year (annualisation constant) |
