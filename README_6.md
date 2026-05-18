# JPMorgan Chase Quantitative Research Portfolio

Quantitative modeling projects completed as part of JPMorgan Chase's virtual experience program. This portfolio demonstrates end-to-end quantitative research methodology: from data quality assessment through model development to business impact analysis.

## Executive Summary

Three interconnected projects addressing core quantitative finance challenges:
1. **Credit Risk**: Default probability modeling with economic decision optimization
2. **Score Quantization**: Optimal bucketing algorithms with statistical validation
3. **Commodities Trading**: Price forecasting and reinforcement learning for trading strategy

Central theme: rigorous methodology over impressive metrics. Each project includes honest assessment of data limitations and focuses on demonstrating sound quantitative reasoning.

---

## Project 1: Credit Risk Modeling

### Objective

Build a probability-of-default model on the Forage retail loan
dataset, then translate the probabilities into a credit-approval
threshold under an asymmetric cost matrix. Report the value of the
threshold relocation on a held-out test fold (not on the fold used
to pick the threshold).

### Data Assessment

The full dataset is trivially separable:

- Simple logistic regression achieves 1.0000 AUC on all features.
- `credit_lines_outstanding` correlates 0.86 with default.
- Real-world credit models typically achieve 0.65 to 0.75 AUC.

This is consistent with a synthetic dataset designed for an
educational program. The notebook documents the diagnosis and
restricts the feature set to `income`, `years_employed`,
`fico_score`, `loan_amt_outstanding` to obtain a more realistic
modelling regime.

### Methodology

**Model comparison with paired t-test on CV folds**

- Logistic Regression: 0.783 AUC (+/- 0.013)
- Random Forest: 0.729 AUC (+/- 0.010)
- XGBoost: 0.740 AUC (+/- 0.012)

A paired t-test on the 5-fold AUC differences (LR vs XGBoost)
gives p = 0.0004. Logistic regression wins on this restricted
feature set.

**Calibration (Brier)**

Brier score on the LR model: 0.126. The calibration curve tracks
the diagonal with slight underconfidence in the 0.3 to 0.5 region.

**Explainability (SHAP)**

LinearExplainer on the LR model. FICO score has the largest
average absolute SHAP value; loan amount has the next largest.
Directional effects are economically coherent (higher FICO -> lower
risk).

**Economic decision optimisation: three-way split**

The original analysis swept thresholds on the same test fold that
produced the headline profit number, which selects the empirically
best operating point and reports its in-sample profit. The
corrected protocol uses a three-way stratified split:

- Train (60%, n = 6000): fit the logistic regression.
- Threshold-select (20%, n = 2000): sweep thresholds and pick the
  one that maximises profit on this fold alone.
- Final test (20%, n = 2000): report profit at the chosen
  threshold. No threshold tuning touches this fold.

The split, profit function, threshold optimiser, and bootstrap CIs
live under `src/credit/` with unit tests at `tests/test_credit_*.py`.

**Cost convention (single source of truth, used by `src.credit.eval`)**

```
loan_amount           = $10,000
profit_margin         = 0.15
loss_given_default    = 0.90

TN (good approved)     -> + loan * margin
TP (default rejected)  ->   0
FP (good rejected)     -> - loan * margin       (lost margin)
FN (default approved)  -> - loan * LGD
```

### Results (corrected protocol)

| Quantity                              | Value           |
| ------------------------------------- | --------------- |
| Optimal threshold t* (chosen on selection fold) | 0.25  |
| Test profit at t*                     | $0              |
| Test profit at t = 0.5 (default)      | -$363,000       |
| **Improvement on test fold**          | **$363,000**    |
| Per-loan improvement                  | $181.50 / loan  |
| 95% bootstrap CI on improvement       | $159,000 to $579,000 |
| Test rejection rate at t*             | 24.3%           |
| False rejections at t* (good loans rejected) | 290     |
| Approved loan volume at t*            | $15.15M         |
| Default rate on approved book at t*   | 11.6%           |

Sensitivity to cost assumptions (chosen on selection fold,
evaluated on test fold):

| Scenario     | margin | LGD  | t*   | test profit  | 95% CI                 |
| ------------ | ------ | ---- | ---- | -----------  | ---------------------  |
| Conservative | 10%    | 95%  | 0.19 | -$310,500    | [-$543,038, -$99,938]  |
| Base         | 15%    | 90%  | 0.25 |       $0     | [-$265,537, $250,500]  |
| Aggressive   | 20%    | 80%  | 0.34 |  $776,000    | [$495,950, $1,032,100] |

### Key findings

- A held-out threshold-selection fold reduces the headline number
  from the in-sample $489,000 reported by the original protocol to
  $363,000 (95% CI $159K to $579K) on data the threshold sweep did
  not see.
- The 95% CI on the *improvement* is tighter than the CI on the
  raw test profit because the same bootstrap sample fixes both
  terms in the difference.
- Threshold relocation under the base cost regime moves the
  decision rule from losing $363K (at t = 0.5) to breaking even
  (at t = 0.25), not from break-even to a $489K profit.
- All numbers are computed on the synthetic Forage dataset and
  inherit its limitations.


---

## Project 2: FICO Score Bucketing

### Objective
Optimally discretize continuous FICO scores (300-850) into categorical buckets for downstream ML models while preserving credit risk signal.

### Mathematical Framework

**Log-Likelihood Maximization**
```
LL(b₁, ..., bₖ₋₁) = Σᵢ [kᵢ ln pᵢ + (nᵢ - kᵢ) ln(1 - pᵢ)]
```
Where:
- bᵢ = bucket boundaries
- nᵢ = records in bucket i
- kᵢ = defaults in bucket i
- pᵢ = kᵢ/nᵢ = default probability

### Methodology

**Greedy vs Dynamic Programming**

Greedy optimization:
- Initialize with quantile boundaries
- Iteratively adjust to maximize LL
- Computationally efficient: O(iterations × buckets × deltas)

Dynamic programming (exact solution):
- Precompute LL for all possible intervals
- Build optimal solution via recurrence
- Guarantees global optimum: O(n² × k)

Results for 7 buckets:
- Greedy LL: -4242.74
- DP LL: -4229.60
- Gap: 13.14 (greedy found local optimum)

**Bootstrap Confidence Intervals**

50 bootstrap samples reveal boundary instability:
- Boundary 1: 513 [493, 522]
- Boundary 3: 583 [561, 611] (±25 points)
- Boundary 4: 614 [586, 648] (±31 points)

Middle boundaries have substantial uncertainty, suggesting optimal configuration is sensitive to sample composition.

**Monotonicity Analysis**

7 buckets: Monotonic (66% → 46% → 34% → 24% → 16% → 9% → 3%)

10 buckets: Violates monotonicity
- Bucket 8: 1.65% default rate
- Bucket 9: 37.50% default rate (small sample anomaly)

Conclusion: More granularity introduces instability without improving discrimination.

**Model Selection (AIC/BIC)**

Both AIC and BIC select 7 buckets as optimal, balancing fit against complexity.

**Information Value**

IV = 0.77 (strong predictive power, >0.3 threshold)

Weight of Evidence progression: -2.1 -> -1.3 -> -0.8 -> -0.4 -> +0.2 -> +0.8 -> +2.0

Monotonic WoE confirms good risk separation for scorecard development.

### Key Findings
- Exact optimization (DP) outperforms heuristics
- Boundary stability matters for production deployment
- Monotonicity constraints are economically necessary
- 7 buckets balances granularity against statistical stability

---

## Project 3: Natural Gas Storage Contracts

### Objective

Value storage contracts allowing seasonal arbitrage (inject during
low-price summer, withdraw during high-price winter) and evaluate
whether a tabular Q-learning agent can improve on simple heuristic
trading rules.

### Price Forecasting

**Model**

```
f(t) = a + b*t + c*t^2 + d*sin(2 pi t / 12) + e*cos(2 pi t / 12)
```

Components: linear trend, quadratic curvature, and a 12-month
seasonal sinusoid. Fitted on the 48 monthly observations in
`Nat_Gas.csv` (October 2020 through September 2024). Used by the
`get_price_for_date` helper for interpolation on arbitrary days.

### Contract Valuation

Standard NPV with strict volume constraints:

```
Contract Value = sum(withdrawal_revenue) - sum(injection_cost) - storage_fee
```

The `volume_checker` function enforces `0 <= inventory <= capacity`
at every transaction date and rejects schedules that would breach
either bound.

### Reinforcement Learning for Storage Trading

**Code under `src/gas/`.** Three modules with unit tests under
`tests/test_gas_*.py`:

- `src.gas.baselines` defines `buy_and_hold_profit` and
  `seasonal_swing_profit`, plus the shared `DEFAULT_STORAGE_COST`
  ($0.05 per unit per month). This constant is the single source of
  truth used by both heuristics and by the env, so a comparison
  cannot quietly use a different cost on each side.
- `src.gas.env.GasStorageEnv` is the MDP. Illegal actions
  (`withdraw` at zero inventory, `inject` at full capacity) become
  no-op `hold` actions and return `illegal=True` in the info dict
  rather than silently passing through.
- `src.gas.qlearning` separates training and evaluation. `train`
  takes a price series, an env config, and a `QLearningConfig`
  (deterministic seed, linear epsilon decay 0.3 -> 0.05 over
  episodes). `evaluate` rolls the greedy policy on a separate
  price series so out-of-sample reporting is possible.

**Evaluation protocol**

Chronological 24/24 split on the 48-month series:

- Train window: October 2020 through September 2022.
- Test window: October 2022 through September 2024.

Both windows cover a full annual cycle so the seasonal-swing baseline
is well defined on each side.

**Results (per 10-unit position, storage cost included)**

| Window               | Buy-and-hold | Seasonal swing | Q-learning |
| -------------------- | ------------ | -------------- | ---------- |
| Train (in-sample)    | -$4.50       | $15.10         | $27.10     |
| Test (held out)      | -$3.50       | $16.00         | $1.50      |

Buy-and-hold is negative on both windows because the modest 24-month
price drift does not cover the per-month carrying cost on 10 units.
On the training window the Q-learning agent outperforms the seasonal
swing by roughly 1.8x. On the held-out window the advantage
disappears: the seasonal swing produces $16.00 while the trained
agent produces $1.50.

**Interpretation**

The in-sample improvement does not generalise. A tabular Q with
time as part of the state can memorise "do X at time t" on the
training series, and that memorised policy is brittle when the next
year's price path differs. This is the expected failure mode of the
formulation, and it is what the held-out window measures. Phase 5
revisits the formulation (state without raw time index, sliding-window
evaluation, deep Q-networks for continuous control) as an explicit
methodology improvement; this section documents the baseline result
before any of those changes.

### Findings

- A chronological train/test split is necessary for any non-trivial
  claim about a trading agent; without one, the headline number is
  in-sample.
- The simplest defensible baseline (one annual seasonal swing) is a
  high bar on a small dataset.
- The corrected per-unit storage cost convention is shared between
  the env and the baselines so the comparison is on equal footing.


---

## Repository Structure

```
├── Risk_Estimation.ipynb           # Credit default prediction
│   ├── Data diagnostics
│   ├── Model comparison
│   ├── Calibration analysis
│   ├── SHAP explainability
│   └── Profit optimization
│
├── Bucket_FICO_scores.ipynb        # Score quantization
│   ├── Log-likelihood framework
│   ├── Greedy vs DP optimization
│   ├── Bootstrap confidence intervals
│   ├── Monotonicity enforcement
│   └── Information value analysis
│
├── JPMC_Gas_Contracts.ipynb        # Commodities pricing + RL
│   ├── Time series forecasting
│   ├── Contract valuation
│   ├── Q-learning implementation
│   ├── Policy visualization
│   └── Strategy comparison
│
└── README_6.md                      # This file (renamed to README.md in Phase 3)
```

---

## Limitations and Honest Assessment

### Data Limitations
- **Credit Risk**: Synthetic dataset with unrealistic feature relationships. Results demonstrate methodology, not production performance.
- **Gas Pricing**: Only 48 months of data limits forecast reliability. Extrapolation uncertainty not quantified.
- **FICO Bucketing**: Same synthetic credit data. Boundaries would differ on real population.

### Model Limitations
- **No temporal validation**: Credit model uses random split, not time-ordered. Real deployment requires out-of-time testing.
- **Deterministic RL**: Gas trading assumes perfect price knowledge. Production would need stochastic price modeling.
- **No transaction costs**: Gas RL ignores bid-ask spreads, market impact, and operational constraints.
- **Simplified state space**: Discrete inventory levels lose granularity of continuous control.

### What These Projects Don't Show
- Real-time data integration
- Model monitoring and drift detection
- Regulatory compliance (Basel III capital calculations)
- Stress testing under adverse scenarios
- Production deployment considerations

---

## Learnings

### Technical Skills
1. **Data Quality First**: Diagnosing synthetic data before trusting results
2. **Statistical Rigor**: Significance testing, not just point estimates
3. **Exact vs Heuristic**: When to use DP over greedy algorithms
4. **Explainability**: SHAP for understanding model decisions
5. **RL Fundamentals**: Q-learning for sequential decision problems

### Quantitative Finance Domain
1. **Credit Risk**: PD-LGD-EAD framework, calibration importance
2. **Scorecard Development**: WoE, IV, monotonicity constraints
3. **Commodities**: Seasonal patterns, storage economics, arbitrage
4. **Decision Theory**: Asymmetric costs, threshold optimization

---

## Future Extensions

### Credit Risk
- Deep learning architectures (neural networks)
- Survival analysis for time-to-default
- Economic stress testing scenarios
- Fairness and bias analysis
- Real credit bureau data

### FICO Bucketing
- Online learning for boundary updates
- Multi-objective optimization (LL + monotonicity + stability)
- Comparison with isotonic regression
- Transfer learning across portfolios

### Gas Trading
- GARCH models for volatility forecasting
- Monte Carlo simulation for contract value distributions
- Continuous action spaces (not discrete inject/withdraw)
- Deep Q-Networks for larger state spaces
- Multi-commodity portfolio optimization

---

## About This Project

This portfolio was completed as part of JPMorgan Chase's Quantitative Research virtual experience program. The program provides specifications and synthetic datasets; the implementation, analysis, and extensions are original work.

---

*Last Updated: November 2024*

*Virtual experience program - datasets provided were synthetic/simplified for educational purposes*
