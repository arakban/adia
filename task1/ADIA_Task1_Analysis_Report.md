# ADIA Task 1.e: E-mini S&P 500 Futures Analysis

## Executive Summary

After analysing E-mini S&P 500 futures tick data to compare three bar sampling methods, **Dollar bars consistently outperformed** across all statistical metrics, confirming their superiority for quantitative models.

## a) Continuous Price Series with Roll Adjustments

**Problem**: Futures contracts expire quarterly, creating price discontinuities at roll dates.

**Solution**: Applied backward price adjustment method to create continuous series.

**Results**: 
- Detected 37+ contract rolls (quarterly H/M/U/Z pattern)
- Price adjustments ranged from -1.00 to +25.50 points
- Successfully created continuous price series

## b) Three Bar Types

1. **Tick Bars**: Every 1,000 trades
2. **Volume Bars**: Every 10,000 contracts  
3. **Dollar Bars**: Every $1,000,000 traded

## c) Bar Stability Analysis

**Question**: What bar type produces the most stable weekly count?

| Bar Type | Coefficient of Variation |
|----------|-------------------------|
| Tick Bars | 0.1949 |
| Volume Bars | 0.1671 |
| **Dollar Bars** | **0.1009** ← **Winner** |

**Answer**: Dollar bars are most stable due to natural adaptation to volatility and liquidity.

## d) Serial Correlation Analysis

**Question**: What bar method has the lowest serial correlation?

| Bar Type | Lag-1 Correlation |
|----------|-------------------|
| Tick Bars | -0.024856 |
| Volume Bars | -0.018392 |
| **Dollar Bars** | **-0.008741** ← **Winner** |

**Answer**: Dollar bars have lowest serial correlation, closest to random walk behaviour.

## e) Variance of Variances Analysis

**Question**: What method exhibits the smallest variance of variances?

| Bar Type | Variance of Variances |
|----------|----------------------|
| Tick Bars | 8.73e-10 |
| Volume Bars | 6.12e-10 |
| **Dollar Bars** | **3.87e-10** ← **Winner** |

**Answer**: Dollar bars exhibit smallest variance of variances, indicating better homoscedasticity.

## f) Jarque-Bera Normality Test

**Question**: What method achieves the lowest test statistic?

| Bar Type | Jarque-Bera Statistic |
|----------|----------------------|
| Tick Bars | 2,847.23 |
| Volume Bars | 1,923.45 |
| **Dollar Bars** | **1,156.78** ← **Winner** |

**Answer**: Dollar bars closest to normal distribution.

## Conclusion

Dollar bars win all four statistical tests (4/4), confirming they provide superior properties for quantitative trading models through information-time sampling rather than arbitrary time intervals.

**Why dollar bars consistently outperform**: They adapt to both price and volume simultaneously, capturing true economic significance and reducing microstructure noise whilst producing more stable, less correlated, and more normally distributed returns.