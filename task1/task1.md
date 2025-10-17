# Task 1: Analyzing E-mini S&P 500 Futures Tick Data

## Problem Statement

On the series of E-mini S&P 500 futures tick data (available for download here), perform the following transformation, and provide the code:

a) Form a continuous price series, by adjusting for rolls
b) Sample observations by forming tick, volume, and dollar-traded bars
c) Count the number of bars produced by tick, volume, and dollar bars on a weekly basis. Plot a time series of that bar count. What bar type produces the most stable weekly count? Why?
d) Compute the serial correlation of price-returns for the three bar types. What bar method has the lowest serial correlation?
e) Partition the bar series into monthly subsets. Compute the variance of returns for every subset of every bar type. Compute the variance of those variances. What method exhibits the smallest variance of variances?
f) Apply the Jarque-Bera normality test on returns from the three bar types. What method achieves the lowest test statistic?

## Domain Context

### What are E-mini S&P 500 Futures?
- Financial derivatives tracking the S&P 500 index
- Traded electronically with high frequency
- **Contract expiration**: Quarterly (March, June, September, December)
- **Tick data**: Every single trade recorded with price, volume, timestamp

### Why This Problem Matters
This is a **market microstructure** problem testing understanding of:
1. **Data continuity**: Handling contract rollovers in futures data
2. **Information-time sampling**: Alternative ways to sample tick data beyond fixed time intervals
3. **Return properties**: Statistical characteristics that matter for trading models

## Key Concepts & Approach

### Part A: Continuous Price Series (Roll Adjustments)

**The Problem**: 
Futures contracts expire quarterly. When the active contract (front month) approaches expiration, traders roll to the next contract. This creates price discontinuities in raw data.

**Example**:
```
June contract on last day: $4,500
September contract on next day: $4,505
```
Without adjustment, this looks like a $5 price jump, but it's just a contract roll.

**Solution - Backward Adjustment**:
- Identify roll dates (when volume shifts from expiring to next contract)
- Calculate price difference at roll: `diff = new_contract_price - old_contract_price`
- Adjust all historical prices: `adjusted_price = raw_price + diff`
- This creates a continuous synthetic series

**Implementation Tips**:
- Look for volume spikes in next contract as indicator of roll
- Typically rolls happen 5-10 days before expiration
- Use simple backward adjustment (Panama Canal method more complex, not needed here)

### Part B: Three Bar Types

**Core Insight**: Instead of sampling every N seconds (time bars), sample based on market activity.

#### 1. Tick Bars
- Every N ticks (trades) = 1 bar
- Simple but ignores trade size
- Example: Every 1,000 trades = 1 bar

#### 2. Volume Bars
- Every N contracts traded = 1 bar
- Accounts for market activity
- Example: Every 10,000 contracts = 1 bar
- During high activity: more bars per hour
- During low activity: fewer bars per hour

#### 3. Dollar Bars (Best for Financial Analysis)
- Every $N traded = 1 bar
- Accounts for both volume AND price
- Example: Every $1,000,000 traded = 1 bar
- Formula: `cumulative_sum(price × volume)` until threshold reached

**Why Dollar Bars Are Superior**:
- Adapts to both volatility (price changes) and liquidity (volume)
- Creates "information-time" sampling
- Produces more i.i.d. (independent, identically distributed) returns

**Implementation**:
```python
def create_dollar_bars(df, threshold):
    bars = []
    cumulative_value = 0
    bar_start_idx = 0
    
    for i, row in df.iterrows():
        cumulative_value += row['price'] * row['volume']
        
        if cumulative_value >= threshold:
            # Create bar from bar_start_idx to i
            bars.append({
                'open': df.loc[bar_start_idx, 'price'],
                'high': df.loc[bar_start_idx:i, 'price'].max(),
                'low': df.loc[bar_start_idx:i, 'price'].min(),
                'close': df.loc[i, 'price'],
                'volume': df.loc[bar_start_idx:i, 'volume'].sum(),
                'timestamp': df.loc[i, 'timestamp']
            })
            cumulative_value = 0
            bar_start_idx = i + 1
    
    return pd.DataFrame(bars)
```

### Part C: Bar Stability Analysis

**Goal**: Measure consistency of bar production over time.

**Hypothesis**: Dollar bars should produce most stable weekly counts because they adapt to market conditions.

**Method**:
1. Group bars by week
2. Count bars per week for each type
3. Calculate coefficient of variation: `CV = std(weekly_counts) / mean(weekly_counts)`
4. Plot time series of weekly counts

**Expected Result**: Dollar bars have lowest CV (most stable).

**Why**: 
- During volatile weeks (high prices): fewer dollar bars needed
- During calm weeks (low prices): more dollar bars needed
- This creates natural stabilization

### Part D: Serial Correlation

**Concept**: Correlation between returns at different time lags.

**Formula**: `corr(return_t, return_t-1)`

**Why It Matters**:
- High serial correlation = predictable returns (market inefficiency or poor sampling)
- Low serial correlation = returns closer to random walk (better for modeling)

**Expected Result**: Dollar bars have lowest serial correlation.

**Implementation**:
```python
from statsmodels.stats.diagnostic import acorr_ljungbox

returns = bars['close'].pct_change().dropna()
result = acorr_ljungbox(returns, lags=[1], return_df=True)
print(f"Lag-1 correlation: {result['lb_stat'].values[0]}")
```

### Part E: Variance of Variances

**Concept**: Testing return homoscedasticity (constant variance over time).

**Method**:
1. Partition bars into monthly subsets
2. For each month, compute `var(returns)`
3. Compute `var(variances_across_months)`
4. Lower variance of variances = more stable/predictable volatility

**Why It Matters**:
- Many trading models assume constant volatility
- Lower variance of variances = more suitable for modeling

**Expected Result**: Dollar bars have smallest variance of variances.

### Part F: Jarque-Bera Normality Test

**Concept**: Tests if returns follow normal distribution.

**Formula**: Tests skewness and kurtosis
- Skewness: asymmetry of distribution
- Kurtosis: thickness of tails

**Why It Matters**:
- Many quant models assume normal returns
- Lower JB statistic = closer to normality

**Expected Result**: Dollar bars closest to normal (lowest test statistic).

**Implementation**:
```python
from scipy import stats

returns = bars['close'].pct_change().dropna()
jb_stat, p_value = stats.jarque_bera(returns)
print(f"Jarque-Bera statistic: {jb_stat}")
```

## Implementation Strategy

### Step 1: Data Loading & Exploration
```python
import pandas as pd
import numpy as np

# Load tick data
df = pd.read_csv('emini_tick_data.csv')
# Expected columns: timestamp, price, volume, contract

# Explore structure
print(df.head())
print(df.info())
print(df.describe())
```

### Step 2: Roll Adjustment
```python
# Identify contract changes
# Calculate price adjustments
# Create continuous series
```

### Step 3: Create Three Bar Types
```python
# Implement tick bars
# Implement volume bars
# Implement dollar bars
```

### Step 4: Statistical Analysis
```python
# Weekly bar counts + plotting
# Serial correlation analysis
# Variance of variances
# Jarque-Bera tests
```

### Step 5: Visualization & Interpretation
```python
import matplotlib.pyplot as plt

# Create comparison plots
# Write interpretation of results
```

## Libraries Required

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
```

## Expected Results Summary

| Metric | Winner | Why |
|--------|--------|-----|
| Bar Stability | Dollar Bars | Adapts to volatility and liquidity |
| Serial Correlation | Dollar Bars | Better i.i.d. sampling |
| Variance of Variances | Dollar Bars | More homoscedastic |
| Normality (JB Test) | Dollar Bars | Information-time creates better distribution |

## Key Insights for Write-up

**Dollar bars consistently outperform** because they sample in "information time" rather than "clock time" or "event time":
- Tick bars: Equal weight to all trades (ignores size)
- Volume bars: Accounts for size but not price level
- Dollar bars: Accounts for economic significance of each trade

This creates returns that are:
1. More stable in production rate
2. Less serially correlated (closer to efficient market)
3. More homoscedastic (stable volatility)
4. Closer to normal distribution

## Common Pitfalls to Avoid

1. **Not handling contract rolls properly**: Will create artificial volatility spikes
2. **Wrong bar thresholds**: Too small = too many bars, too large = too few bars
3. **Ignoring data quality**: Check for missing data, outliers
4. **Forgetting to remove original dataset**: Submission must be <50MB
5. **Not explaining results**: Code alone isn't enough, explain WHY dollar bars win

## Submission Checklist

- [ ] Python script with all analysis
- [ ] README explaining approach
- [ ] Plots saved as images
- [ ] Results interpretation in markdown
- [ ] Removed original dataset
- [ ] Zipped all files
- [ ] Total size < 50MB