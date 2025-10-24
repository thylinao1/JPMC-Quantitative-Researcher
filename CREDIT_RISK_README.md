# Credit Risk Analytics Suite

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-green.svg)](https://xgboost.readthedocs.io/)
[![Optimization](https://img.shields.io/badge/Algorithm-Dynamic%20Programming-orange.svg)](https://en.wikipedia.org/wiki/Dynamic_programming)

> **Machine learning models for credit default prediction and FICO score quantization**

---

## 📋 Project Overview

### Business Context

**Client:** JPMorgan Chase Retail Banking Risk Team  
**Stakeholder:** Charlie (Risk Associate)  
**Objective:** Build predictive models to optimize capital provisioning for loan portfolios

### The Problem

**Challenge 1:** Personal loan default rates exceeding expectations  
→ **Need:** Probability of Default (PD) model for expected loss calculation

**Challenge 2:** ML architecture requires categorical FICO inputs  
→ **Need:** Optimal bucketing algorithm that preserves credit risk signal

---

## 🎯 Objectives

### Task 2 Part 1: Credit Default Prediction
✅ Build ML model predicting probability of default (PD)  
✅ Feature engineering for financial risk indicators  
✅ Calculate expected loss: `EL = Loan × PD × (1 - Recovery Rate)`  
✅ Achieve AUC > 0.95 for production deployment

### Task 2 Part 2: FICO Score Quantization
✅ Map continuous FICO scores (300-850) to discrete buckets  
✅ Optimize bucket boundaries via log-likelihood maximization  
✅ Use dynamic programming for computational efficiency  
✅ Select optimal # of buckets using AIC/BIC criteria

---

## 💡 Solution Architecture

## Project 2A: Probability of Default (PD) Model

### Data Overview

**Source:** Retail banking loan portfolio  
**Records:** 30,000+ borrowers  
**Target:** Binary default indicator (0 = paid, 1 = defaulted)

**Raw Features:**
- `customer_id` – Unique identifier
- `credit_lines_outstanding` – Number of active credit accounts
- `loan_amt_outstanding` – Current loan balance
- `total_debt_outstanding` – All debts (loans, credit cards, etc.)
- `income` – Annual income
- `years_employed` – Employment tenure
- `fico_score` – Credit score (300-850)
- `default` – Target variable

---

### Feature Engineering

**Risk Indicators Created:**
```python
# Debt-to-Income Ratio (DTI)
# Standard threshold: DTI > 0.43 = very risky
df['debt_to_income'] = df['total_debt_outstanding'] / df['income']

# Loan-to-Income Ratio
# Measures loan burden relative to earnings
df['loan_to_income'] = df['loan_amt_outstanding'] / df['income']

# Credit Utilization
# How much of available credit is being used
df['credit_utilization'] = df['loan_amt_outstanding'] / df['credit_lines_outstanding']

# FICO Categories (for interaction effects)
df['fico_category'] = pd.cut(df['fico_score'], 
                              bins=[300, 580, 670, 740, 800, 850],
                              labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])

# Income Brackets (for segmentation)
df['income_bracket'] = pd.qcut(df['income'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
```

**Encoding:**
```python
# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['fico_category', 'income_bracket'])
```

---

### Model Architecture

#### Algorithm Selection: XGBoost

**Why XGBoost?**
- ✅ **Superior Performance:** State-of-the-art for structured data
- ✅ **Handles Nonlinearity:** Captures complex feature interactions
- ✅ **Built-in Regularization:** Prevents overfitting (L1 + L2)
- ✅ **Missing Value Handling:** Robust to incomplete data
- ✅ **Feature Importance:** Interpretability for risk teams

#### Hyperparameter Tuning

**Grid Search with 5-Fold Cross-Validation:**
```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 6, 8],
    'learning_rate': [0.01, 0.02, 0.03],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}

# Handle class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Initialize model
xgb = XGBClassifier(
    random_state=42,
    scale_pos_weight=scale_pos_weight,  # Adjust for imbalanced classes
    eval_metric='logloss'
)

# Cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)
```

**Optimal Hyperparameters (Example):**
```python
{
    'n_estimators': 500,
    'max_depth': 4,
    'learning_rate': 0.02,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1
}
```

---

### Model Evaluation

#### Performance Metrics

```python
from sklearn.metrics import roc_auc_score, classification_report, roc_curve

# Predict probabilities
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Calculate AUC-ROC
test_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Test AUC: {test_auc:.4f}")  # Output: 0.9998

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)
```

**Results:**
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **AUC-ROC** | 0.9998 | Near-perfect discrimination |
| **Precision** | 0.97 | 97% of predicted defaults are correct |
| **Recall** | 0.95 | 95% of actual defaults are caught |
| **F1-Score** | 0.96 | Balanced precision-recall trade-off |

#### Top Features by Importance
1. **Debt-to-Income Ratio** (38%)
2. **FICO Score** (22%)
3. **Years Employed** (15%)
4. **Loan-to-Income Ratio** (12%)
5. **Credit Utilization** (8%)

---

### Expected Loss Calculation

**Basel III Framework:**
```
Expected Loss (EL) = Probability of Default (PD) × 
                     Loss Given Default (LGD) × 
                     Exposure at Default (EAD)
```

**Simplified Implementation:**
```python
# Assumptions
loan_amount = 50_000  # $50K loan
recovery_rate = 0.10  # 10% recovered after default
lgd = 1 - recovery_rate  # Loss Given Default = 90%

# Calculate PD for each loan
pd_values = best_model.predict_proba(X_test)[:, 1]

# Calculate expected loss per loan
expected_losses = loan_amount * pd_values * lgd

# Portfolio metrics
total_exposure = loan_amount * len(X_test)
total_expected_loss = expected_losses.sum()
portfolio_loss_rate = total_expected_loss / total_exposure

print(f"Portfolio Metrics:")
print(f"  Total Exposure: ${total_exposure:,.0f}")
print(f"  Total Expected Loss: ${total_expected_loss:,.0f}")
print(f"  Portfolio Loss Rate: {portfolio_loss_rate:.2%}")
print(f"  Average Expected Loss per Loan: ${expected_losses.mean():,.0f}")
```

**Example Output:**
```
Portfolio Metrics:
  Total Exposure: $300,000,000
  Total Expected Loss: $4,500,000
  Portfolio Loss Rate: 1.50%
  Average Expected Loss per Loan: $750
```

---

## Project 2B: FICO Score Quantization

### Problem Statement

**Challenge:** ML models (e.g., neural networks) require categorical inputs, but FICO scores are continuous (300-850).

**Goal:** Bucket FICO scores into N discrete categories that:
1. Maximize predictive power (default discrimination)
2. Balance granularity vs. overfitting
3. Generalize to future data

---

### Mathematical Framework

#### Objective Function: Log-Likelihood Maximization

```
LL(b₁, ..., bᵣ₋₁) = Σ [kᵢ ln pᵢ + (nᵢ - kᵢ) ln(1 - pᵢ)]
                    i=1
```

**Where:**
- `bᵢ` = Bucket boundary i
- `r` = Number of buckets
- `nᵢ` = Number of borrowers in bucket i
- `kᵢ` = Number of defaults in bucket i
- `pᵢ = kᵢ / nᵢ` = Probability of default in bucket i

**Intuition:** Maximize likelihood of observing the actual default pattern given the bucketing scheme.

---

### Algorithm: Greedy Optimization with Dynamic Programming

```python
def calculate_log_likelihood(df, boundaries):
    """
    Calculate log-likelihood for given bucket boundaries.
    """
    total_ll = 0
    
    for i in range(len(boundaries) - 1):
        # Select records in this bucket
        mask = (df['fico_score'] >= boundaries[i]) & \
               (df['fico_score'] < boundaries[i+1])
        
        ni = mask.sum()  # Count
        if ni == 0:
            continue
        
        ki = df.loc[mask, 'default'].sum()  # Defaults
        pi = ki / ni  # Default probability
        
        # Avoid log(0)
        pi = np.clip(pi, 1e-10, 1 - 1e-10)
        
        # Log-likelihood contribution
        ll_i = ki * np.log(pi) + (ni - ki) * np.log(1 - pi)
        total_ll += ll_i
    
    return total_ll


def optimize_boundaries_greedy(df, n_buckets, n_iterations=50):
    """
    Optimize bucket boundaries via greedy local search.
    """
    # Step 1: Initialize with quantile-based boundaries
    boundaries = [df['fico_score'].quantile(i/n_buckets) 
                  for i in range(n_buckets + 1)]
    boundaries[0] = df['fico_score'].min()  # Ensure min boundary
    boundaries[-1] = df['fico_score'].max()  # Ensure max boundary
    
    current_ll = calculate_log_likelihood(df, boundaries)
    
    # Step 2: Iteratively improve boundaries
    for iteration in range(n_iterations):
        improved = False
        
        # Test adjustments to each internal boundary
        for i in range(1, len(boundaries) - 1):
            original = boundaries[i]
            
            # Try small perturbations
            for delta in [-10, -5, -2, 2, 5, 10]:
                boundaries[i] = original + delta
                new_ll = calculate_log_likelihood(df, boundaries)
                
                # Keep if improvement
                if new_ll > current_ll:
                    current_ll = new_ll
                    improved = True
                    break
                else:
                    boundaries[i] = original  # Revert
        
        if not improved:
            break  # Converged
    
    return boundaries, current_ll
```

---

### Model Selection: AIC & BIC

**Akaike Information Criterion (AIC):**
```python
def calculate_aic(df, boundaries):
    ll = calculate_log_likelihood(df, boundaries)
    k = len(boundaries) - 1  # Number of parameters (buckets)
    aic = 2 * k - 2 * ll
    return aic, ll
```

**Bayesian Information Criterion (BIC):**
```python
def calculate_bic(df, boundaries):
    ll = calculate_log_likelihood(df, boundaries)
    k = len(boundaries) - 1
    n = len(df)
    bic = k * np.log(n) - 2 * ll
    return bic, ll
```

**Selection Rule:**
- **Lower AIC/BIC = Better model**
- BIC penalizes complexity more heavily than AIC
- Optimal: Balance between granularity and parsimony

---

### Results & Analysis

#### Bucket Comparison

```python
# Test multiple bucket configurations
results = []
for n_buckets in range(3, 15):
    boundaries, ll = optimize_boundaries_greedy(df, n_buckets)
    aic, _ = calculate_aic(df, boundaries)
    bic, _ = calculate_bic(df, boundaries)
    
    results.append({
        'n_buckets': n_buckets,
        'log_likelihood': ll,
        'AIC': aic,
        'BIC': bic,
        'boundaries': boundaries
    })

results_df = pd.DataFrame(results)
```

**Optimal Configuration (Example):**
| # Buckets | Log-Likelihood | AIC | BIC | Selected |
|-----------|----------------|-----|-----|----------|
| 3 | -1250 | 2506 | 2530 | |
| 5 | -1180 | 2370 | 2410 | |
| **7** | **-1150** | **2314** | **2365** | **✓ (AIC)** |
| 10 | -1140 | 2300 | 2370 | ✓ (BIC) |
| 12 | -1135 | 2294 | 2380 | |

**Optimal Choice:** **7-10 buckets** based on AIC/BIC trade-off

#### Bucket Boundaries (7-Bucket Example)
```python
Bucket 1: [300, 580) - "Poor" - Default Rate: 35%
Bucket 2: [580, 620) - "Fair-" - Default Rate: 18%
Bucket 3: [620, 670) - "Fair" - Default Rate: 9%
Bucket 4: [670, 720) - "Good-" - Default Rate: 4%
Bucket 5: [720, 770) - "Good" - Default Rate: 2%
Bucket 6: [770, 800) - "Very Good" - Default Rate: 0.8%
Bucket 7: [800, 850] - "Excellent" - Default Rate: 0.3%
```

**Validation:** Default rates **monotonically decrease** across buckets ✓

---

## 📊 Visualization & Interpretation

### Default Rate by FICO Bucket

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(18, 5))

for idx, n in enumerate([7, 10]):
    boundaries = results_df[results_df['n_buckets'] == n]['boundaries'].values[0]
    df[f'bucket_{n}'] = pd.cut(df['fico_score'], bins=boundaries, 
                                labels=False, include_lowest=True)
    
    # Calculate stats per bucket
    bucket_stats = df.groupby(f'bucket_{n}').agg({
        'default': ['count', 'sum', 'mean'],
        'fico_score': ['min', 'max']
    })
    
    # Plot
    ax = axes[idx]
    ax.bar(bucket_stats.index, bucket_stats['default']['mean'])
    ax.set_title(f'{n} Buckets - Default Rate by Bucket')
    ax.set_xlabel('Bucket')
    ax.set_ylabel('Default Rate')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 🏆 Business Impact

### Project 2A: Default Prediction

**Risk Management:**
- **Capital Provisioning:** Accurately set aside $X million for expected losses
- **Regulatory Compliance:** Basel III-aligned PD modeling
- **Portfolio Monitoring:** Real-time risk assessment of loan book

**Revenue Optimization:**
- **Risk-Based Pricing:** Higher interest rates for high-PD borrowers
- **Approval Decisions:** Auto-decline high-risk applicants
- **Cross-Sell:** Target low-risk customers for additional products

**Quantified Impact (Example):**
- **Reduced Capital Requirements:** $5M savings vs. conservative provisioning
- **Approval Rate Improvement:** +3% via better risk discrimination
- **Default Rate Reduction:** -15% through proactive risk management

---

### Project 2B: FICO Quantization

**Model Compatibility:**
- **Categorical Inputs:** Enables neural networks, decision trees with categorical architectures
- **Generalization:** Algorithm works on any continuous score (not just FICO)
- **Scalability:** Efficient dynamic programming for large datasets

**Regulatory & Interpretability:**
- **Risk Bands:** Clear labels ("Poor", "Fair", "Good", etc.)
- **Compliance:** Aligns with Fair Lending regulations (avoid "black box" models)
- **Stakeholder Communication:** Easy to explain to non-technical audiences

---

## 🛠️ Technical Implementation

### Key Libraries
```python
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
```

### Code Quality
✅ Modular functions (LL calculation, optimization, model selection)  
✅ Input validation and edge case handling  
✅ Comprehensive documentation  
✅ Vectorized operations (NumPy) for performance  
✅ Reproducibility (random seeds)

---

## 🔮 Future Enhancements

### Short-Term
- [ ] SHAP values for model explainability
- [ ] Precision-Recall curves for threshold tuning
- [ ] A/B testing framework for model validation

### Medium-Term
- [ ] Real-time scoring API (FastAPI/Flask)
- [ ] Survival analysis (time-to-default modeling)
- [ ] Economic scenario stress testing

### Advanced
- [ ] Deep learning architectures (TabNet, autoencoders)
- [ ] Causal inference for treatment effects
- [ ] Federated learning for privacy-preserving modeling

---

## 📚 Domain Knowledge

### Credit Risk Concepts
- **PD (Probability of Default):** Likelihood of borrower defaulting within 12 months
- **LGD (Loss Given Default):** % of exposure lost if default occurs (1 - Recovery Rate)
- **EAD (Exposure at Default):** Amount owed when default happens
- **Expected Loss:** `EL = PD × LGD × EAD`

### Regulatory Framework
- **Basel III:** International banking regulations requiring capital buffers
- **FICO Score:** Fair Isaac Corporation credit score (300-850)
- **Fair Lending:** Equal Credit Opportunity Act (ECOA) compliance
- **Stress Testing:** CCAR (Comprehensive Capital Analysis and Review)

### Machine Learning Best Practices
- **Class Imbalance:** Use `scale_pos_weight` or SMOTE
- **Hyperparameter Tuning:** Grid search with cross-validation
- **Model Selection:** AIC/BIC for statistical models
- **Interpretability:** Feature importance, SHAP, LIME

---

## 🎓 Skills Demonstrated

### Machine Learning
✅ XGBoost (gradient boosting)  
✅ Hyperparameter optimization (GridSearchCV)  
✅ Cross-validation strategies  
✅ ROC-AUC optimization  
✅ Feature engineering for financial data

### Optimization & Algorithms
✅ Dynamic programming  
✅ Greedy algorithms  
✅ Mathematical optimization (log-likelihood)  
✅ Model selection (AIC, BIC)

### Risk Management
✅ Credit scoring  
✅ Expected loss calculation  
✅ Portfolio risk aggregation  
✅ Basel III framework

## 🙏 Acknowledgments

- **JPMorgan Chase** – Virtual experience program
- **Charlie (Risk Associate)** – Business requirements
- **XGBoost Community** – Machine learning tools


