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
Build probability of default (PD) model for retail loan portfolio and translate ML predictions into business decisions.

### Data Assessment
Initial analysis revealed critical issue: dataset is synthetic and trivially separable.
- Simple logistic regression achieves 1.0000 AUC (perfect discrimination)
- `credit_lines_outstanding` correlates 0.86 with default (unrealistic)
- Real-world credit models typically achieve 0.65-0.75 AUC

Rather than ignoring this, the analysis proceeds on two tracks:
1. Full features: demonstrates synthetic nature
2. Restricted features (income, FICO, employment, loan amount): yields realistic 0.78 AUC

### Methodology

**Model Comparison with Statistical Testing**
- Logistic Regression: 0.783 AUC (±0.013)
- Random Forest: 0.729 AUC (±0.010)
- XGBoost: 0.740 AUC (±0.012)

Paired t-test confirms logistic regression superiority (p=0.0004). Simpler model wins when relationships are linear and features are limited.

**Calibration Analysis**
- Brier score: 0.126
- Calibration curve tracks diagonal well
- Slight underconfidence in 0.3-0.5 probability range
- Critical for credit risk: poorly calibrated models lead to mispriced loans

**Explainability (SHAP)**
- FICO score: highest impact
- Years employed: second highest
- All feature effects economically coherent (higher FICO → lower risk)

**Economic Decision Optimization**

Key insight: ML optimization (AUC) ≠ business optimization (profit)

Cost assumptions:
- Loan amount: $10,000
- Profit margin on good loan: 15%
- Loss given default: 90%

Results:
- Default threshold (0.50): -$354,000 expected profit
- Optimal threshold (0.19): +$135,000 expected profit
- Improvement: $489,000

Sensitivity analysis shows optimal threshold ranges from 0.18 to 0.34 depending on cost assumptions.

### Key Findings
- Simpler models can outperform complex ones on limited features
- Classification threshold optimization is as important as model selection
- Calibration matters as much as discrimination for risk pricing

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
Value storage contracts allowing seasonal arbitrage (inject during low-price summer, withdraw during high-price winter) and optimize trading strategy.

### Price Forecasting

**Model Specification**
```
f(t) = a + bt + ct² + d·sin(2πt/12) + e·cos(2πt/12)
```

Components:
- Linear trend: captures long-term price movement
- Quadratic term: allows for acceleration/deceleration
- Sine/cosine: seasonal pattern with 12-month period

Fitted on 48 months of historical data (Oct 2020 - Sep 2024).

**Residual Analysis**
- RMSE calculated for model fit
- Residuals examined for patterns
- Model captures seasonal structure (winter peaks, summer troughs)

### Contract Valuation

Standard NPV calculation:
```
Contract Value = Σ(Withdrawal Revenue) - Σ(Injection Costs) - Storage Fees
```

Volume constraints enforced:
- Inventory ≥ 0 at all times
- Inventory ≤ maximum capacity
- Validation prevents over-injection or impossible withdrawals

### Reinforcement Learning for Optimal Timing

**Problem Formulation**

State space: (inventory_level, time_period)
- Inventory: 0 to 100 units (discretized by 10)
- Time: 48 monthly periods

Action space: {hold, inject, withdraw}

Reward function:
- Inject: -price × volume (cost)
- Withdraw: +price × volume (revenue)
- Storage cost: -inventory × rate per period
- Terminal: liquidate remaining inventory

**Q-Learning Implementation**

Parameters:
- Learning rate (α): 0.1
- Discount factor (γ): 0.99
- Exploration rate (ε): 0.3
- Episodes: 5,000

Algorithm learns state-action value function Q(s, a) through temporal difference updates.

**Results**

Naive strategy (single seasonal cycle):
- Buy in summer (low price): $9.84/MMBtu
- Sell in winter (high price): $10.30/MMBtu
- Profit: $2.60 per 10 units

RL optimal strategy:
- Exploits multiple price cycles
- Builds inventory before uptrends (months 20-25)
- Liquidates after peaks
- Total profit: $101.93

**Improvement: 39x over naive approach**

The agent discovers that multiple small arbitrage opportunities compound to far exceed single seasonal trade.

### Findings
- Sequential optimization outperforms static rules
- RL naturally discovers multi-cycle trading patterns
- Price forecasting + decision optimization is more powerful than forecasting alone

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
└── README.md                        # This file
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
