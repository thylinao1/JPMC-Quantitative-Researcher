# Data

Both files in this directory are the synthetic datasets supplied by the
JPMorgan Chase Quantitative Research virtual experience program on
Forage. They are committed here so that `git clone` + the install
instructions in the top-level README are enough to run every notebook
end-to-end.

## `loan_data.csv`

Retail loan portfolio used by `notebooks/01_credit_risk.ipynb` and
`notebooks/02_fico_bucketing.ipynb`.

- 10,000 rows, one per loan / customer.
- Columns: `customer_id`, `credit_lines_outstanding`,
  `loan_amt_outstanding`, `total_debt_outstanding`, `income`,
  `years_employed`, `fico_score`, `default`.
- The target column is `default` (1 = defaulted, 0 = repaid). Base
  rate is 18.5%.
- The dataset is synthetic and trivially separable on the full
  feature set; the analysis restricts to a weaker subset to obtain a
  more realistic regime (see Project 1 in the top-level README).

## `Nat_Gas.csv`

Monthly Henry Hub-style price series used by
`notebooks/03_gas_storage.ipynb`.

- 48 rows, monthly observations from 2020-10-31 to 2024-09-30.
- Columns: `Dates` (date), `Prices` (USD per MMBtu, approximate).
