[![CI](https://github.com/thylinao1/JPMC-Quantitative-Researcher/actions/workflows/ci.yml/badge.svg)](https://github.com/thylinao1/JPMC-Quantitative-Researcher/actions/workflows/ci.yml)

# JPMorgan Chase Quantitative Research (JPMorgan Chase & Forage program)

Three modelling projects built against the synthetic datasets and task specifications from the JPMorgan Chase Quantitative Research virtual experience program on Forage. The program supplies the data and the prompts; the modelling, evaluation protocol, and code in this repository are independent work.

**Source:** [J.P. Morgan Quantitative Research on Forage](https://www.theforage.com/simulations/jpmorgan/quantitative-research-11oc)

## Summary

1. **Credit Risk**: probability-of-default model on a synthetic retail loan dataset with held-out threshold selection and a bootstrap CI on the test-fold improvement.
2. **Score Quantisation**: optimal FICO bucketing by log-likelihood maximisation (greedy vs dynamic programming) with bootstrap CIs on the chosen boundaries.
3. **Commodities Trading**: a seasonal price model plus tabular Q-learning for storage trading, with a chronological train/test split and explicit baselines (buy-and-hold, seasonal swing). The agent's in-sample advantage does not transfer out of sample; this is reported as the result.

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/thylinao1/JPMC-Quantitative-Researcher.git
cd JPMC-Quantitative-Researcher
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Reproduce

```bash
# Run the test suite (47 tests).
pytest

# Execute every notebook end-to-end against the committed CSVs.
for nb in notebooks/*.ipynb; do
    jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

The notebooks discover the repo root automatically (via a small
`pyproject.toml`-lookup in the first cell), so they run from any
working directory.


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
- XGBoost: 0.730 AUC (+/- 0.011)

A paired t-test on the 5-fold AUC differences (LR vs XGBoost)
gives p = 0.0003. Logistic regression wins on this restricted
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

Threshold selection on the same data that produces the headline
profit number selects the empirically best operating point on that
data and reports its in-sample value. The protocol used here is a
three-way stratified split:

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

### Results

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

- On the held-out test fold the threshold chosen on the selection
  fold yields a $363,000 improvement over the default 0.5 cutoff
  (95% CI $159K to $579K).
- The 95% CI on the improvement is tighter than the CI on the raw
  test profit because the same bootstrap sample fixes both terms
  in the difference.
- Under the base cost regime the decision rule moves from a
  $363K expected loss at t = 0.5 to roughly break-even at t = 0.25.
- All numbers are computed on the synthetic Forage dataset and
  inherit its limitations.

### Additional methodology

Three additions strengthen the soundness of the headline numbers
without changing the underlying model. Code lives under
`src/credit/{metrics,calibration,generalisation}.py` with unit
tests at `tests/test_credit_*.py`.

**Bootstrap 95% CIs on the headline metrics (test fold, 2000 resamples):**

| Metric    | Point  | 95% CI             |
| --------- | ------ | ------------------ |
| AUC       | 0.7827 | [0.7575, 0.8062]   |
| Brier     | 0.1259 | [0.1164, 0.1358]   |
| Recall    | 0.5270 | [0.4774, 0.5803]   |
| Precision | 0.4021 | [0.3600, 0.4475]   |
| F1        | 0.4561 | [0.4147, 0.4994]   |

**Calibration: raw LR vs isotonic (test fold):**

| Probability series | ECE    | Brier  |
| ------------------ | ------ | ------ |
| Raw LR             | 0.0216 | 0.1259 |
| Isotonic (CV=5)    | 0.0282 | 0.1262 |

The base logistic regression is already well calibrated on the
restricted feature set. Post-hoc isotonic fitting on small CV
folds adds variance and does not reduce miscalibration on this
data: a useful negative result.

**Cohort generalisation (tenure-based split):**

The dataset has no time column, so the split uses `years_employed`
as a monotone proxy. Train on customers with
`years_employed >= 3` (n = 9,116, default rate 16.5%); evaluate on
customers with `years_employed < 3` (n = 884, default rate 39.1%).

| Cohort               | t*   | Profit       | Per loan   |
| -------------------- | ---- | ------------ | ---------- |
| Long-tenure (train)  | 0.21 | $1,393,500   |  $152.86   |
| Short-tenure (test)  | 0.21 | -$639,000    | -$722.85   |

The model trained on long-tenure customers does not transfer:
per-loan profit drops by $876 and total profit on the short-tenure
cohort is -$639,000. The short-tenure cohort has a 2.4x higher base
default rate; the model fit on the long-tenure population cannot
recover this through the same t* and the operating point breaks
down (90.0% rejection rate, 458 false rejections out of 796
rejections).


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

**Bootstrap confidence intervals**

Bootstrap (50 resamples) on the 7-bucket optimal boundaries (these are the numbers printed by the notebook):

| Boundary | Point | 95% CI       |
| -------- | ----- | ------------ |
| 1        | 513   | [494, 535]   |
| 2        | 553   | [526, 580]   |
| 3        | 585   | [553, 611]   |
| 4        | 617   | [608, 644]   |
| 5        | 655   | [638, 694]   |
| 6        | 715   | [690, 740]   |

Middle boundaries have substantial uncertainty, suggesting the optimal configuration is sensitive to sample composition.

**Monotonicity Analysis**

7 buckets: Monotonic (66% → 46% → 34% → 24% → 16% → 9% → 3%)

10 buckets: Violates monotonicity
- Bucket 8: 1.65% default rate
- Bucket 9: 37.50% default rate (small sample anomaly)

Conclusion: More granularity introduces instability without improving discrimination.

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
formulation, and it is what the held-out window measures.
Reasonable next steps include dropping the raw time index from the
state, sliding-window evaluation, and moving to a deep Q-network
for continuous control of inventory.

### Findings

- A chronological train/test split is necessary for any non-trivial
  claim about a trading agent; without one, the headline number is
  in-sample.
- The simplest defensible baseline (one annual seasonal swing) is a
  high bar on a small dataset.
- The per-unit storage cost convention is shared between the env
  and the baselines so the comparison is on equal footing.


---

## Repository Structure

```
.
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .github/workflows/ci.yml
├── data/
│   ├── README.md            # describes both CSVs
│   ├── Nat_Gas.csv          # monthly gas price series (Project 3)
│   └── loan_data.csv        # synthetic loan portfolio (Projects 1, 2)
├── notebooks/
│   ├── 01_credit_risk.ipynb
│   ├── 02_fico_bucketing.ipynb
│   └── 03_gas_storage.ipynb
├── src/
│   ├── credit/              # loader, profit/threshold eval, operational profile
│   └── gas/                 # loader, baselines, env, q-learning
└── tests/                   # 47 pytest tests covering src/
```

---

## Limitations

### Data
- The loan dataset is synthetic. Headline dollar amounts come from
  this synthetic data and inherit its limitations; they are not
  estimates of production performance.
- The 48-month gas series limits any forecast or trading claim to its
  observed sample.
- The FICO bucketing inherits the same synthetic credit data; the
  optimal boundaries would differ on a real population.

### Model and evaluation
- The credit model uses a stratified random split rather than a time
  ordered one; real deployment requires out-of-time testing.
- The Q-learning state includes the raw time index, which makes the
  agent vulnerable to memorising the training trajectory. The
  out-of-sample number reported in Project 3 is this exact failure
  mode and is the reason the in-sample improvement does not transfer.
- The gas RL formulation assumes perfect price knowledge during
  rollouts and ignores transaction costs, bid-ask spreads, and
  operational constraints.

### Out of scope
- Real-time data integration, model monitoring, regulatory capital
  (Basel III), stress testing, fairness and bias analysis, production
  deployment.

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

JPMorgan Chase Quantitative Research virtual experience program on Forage. The program supplies the task specifications and the synthetic datasets. The modelling code, evaluation protocol, and analysis text in this repository are this project's contribution.

---

*Last updated: 2026-05-18.*
