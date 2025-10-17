# ADIA Quantitative R&D - DC Analyst Technical Challenge

## Challenge Overview

**Company**: Abu Dhabi Investment Authority (ADIA)  
**Role**: Quantitative R&D - DC Analyst (Data & Computation)  
**Duration**: 7 days (10,080 minutes)  
**Format**: HackerRank technical assessment

## Challenge Structure

This assessment consists of **4 main tasks** plus **bonus questions** testing:
- Quantitative finance knowledge
- Data engineering skills
- API integration
- Statistical analysis
- System design thinking

---

## Task Breakdown

### **Task 1: Analyzing Tick Data** 
**Domain**: Market microstructure, high-frequency trading data  
**Skills**: Python, pandas, statistical analysis, time series  
**Deliverable**: Code + analysis of E-mini S&P 500 futures tick data

**What You're Building**:
- Continuous price series from futures contracts (handling rollovers)
- Three bar-sampling methods: tick bars, volume bars, dollar bars
- Statistical comparison of bar types (stability, correlation, normality)

**Key Insight**: Testing understanding of information-time sampling vs clock-time sampling.

**See**: `task1.md` for detailed breakdown

---

### **Task 2: Tickers from Articles**
**Domain**: NLP, entity extraction, API integration  
**Skills**: Python, HTTP requests, JSON parsing, rate limiting  
**Deliverable**: Function that extracts company information from news articles

**What You're Building**:
1. Call Refinitiv Intelligent Tagging API with article text
2. Extract: Organization PermId, Name, Ticker Symbol
3. Call PermId API to get IPO dates
4. Format output as CSV with specific requirements

**Key Challenges**:
- Rate limiting (1 second between API calls)
- Handling missing data (NULL for unavailable fields)
- Specific output formatting (single quotes for org names only)

**Output Format**:
```
"12345,'APPLE INC.',AAPL,1980-12-12",
"678910,'STARBUCKS CORPORATION',SBUX,NULL"
```

---

### **Task 3: DC File Ingestion**
**Domain**: Data ingestion, file processing  
**Skills**: File I/O, data validation, error handling  
**Deliverable**: TBD (awaiting full problem description)

---

### **Task 4: Text File Ingestion**
**Domain**: Text processing, ETL  
**Skills**: String parsing, data transformation  
**Deliverable**: TBD (awaiting full problem description)

---

## Bonus Section: System Design & Data Quality

### **Question 1: Biases in Datasets**
**Topic**: Identify potential biases in the datasets processed in Tasks 1-4

**Expected Discussion Points**:
- **Survivorship bias**: Only companies that survived to present in datasets
- **Selection bias**: News articles may focus on certain types of companies
- **Temporal bias**: Historical data may not reflect current market conditions
- **Liquidity bias**: Tick data heavily weighted toward liquid instruments
- **Reporting bias**: IPO dates may have inconsistent reporting standards

---

### **Question 2: Data Monitoring Challenges**
**Topic**: Main difficulties and pitfalls in monitoring data pipelines

**Expected Discussion Points**:
- **Schema drift**: APIs changing response formats without notice
- **Data quality degradation**: Gradual decline in completeness/accuracy
- **Latency monitoring**: Detecting delays in real-time pipelines
- **Alerting fatigue**: Too many false positives in monitoring
- **Point-in-time correctness**: Ensuring features use only past data
- **Rate limit handling**: API quotas and throttling
- **Silent failures**: Processes that fail without raising errors

---

### **Question 3: Point-in-Time Correctness**
**Topic**: Ensuring no data leakage in financial datasets

**Expected Discussion Points**:
- **The Problem**: Using future information in training data
  - Example: Using 2024 IPO date to predict 2023 stock performance
- **Solutions**:
  - Event sourcing with timestamps
  - Versioned feature stores
  - Careful lookback window logic
  - Audit trails for all data transformations
- **Why It Matters**: 
  - Overly optimistic backtest results
  - Models fail in production
  - Regulatory compliance issues

---

## Technical Requirements

### Languages & Libraries
```python
# Core
import pandas as pd
import numpy as np

# Statistical analysis
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

# Visualization
import matplotlib.pyplot as plt

# API calls
import requests
import time
import json
```

### Submission Requirements
- **Format**: Single ZIP file containing all code + README
- **Size limit**: < 50 MB (exclude original datasets)
- **Documentation**: README explaining approach and results
- **Code quality**: Clean, commented, reproducible

---

## Strategy & Time Management

### Recommended Approach

**Days 1-2**: Task 1 (Tick Data Analysis)
- Most complex, requires understanding of financial concepts
- Budget 8-10 hours for implementation + analysis

**Day 3**: Task 2 (API Integration)
- Straightforward but requires careful attention to output format
- Budget 3-4 hours

**Day 4**: Tasks 3 & 4 (File Ingestion)
- Likely simpler ETL tasks
- Budget 2-3 hours each

**Days 5-6**: Bonus Questions + Review
- Write thoughtful responses to system design questions
- Review all code for edge cases
- Test submissions

**Day 7**: Buffer for issues + final submission

---

## Connection to Role

This challenge tests skills directly relevant to **Quantitative R&D - DC Analyst**:

1. **Data Computation (DC)**:
   - Processing large tick datasets
   - Statistical analysis of financial time series
   - Building data pipelines

2. **Quantitative Research**:
   - Understanding market microstructure
   - Statistical properties of returns
   - Bias identification in datasets

3. **Engineering Rigor**:
   - API integration with rate limiting
   - Data quality monitoring
   - Point-in-time correctness

4. **System Design Thinking**:
   - Monitoring strategies
   - Handling edge cases
   - Production considerations

---

## Key Success Factors

1. **Show Your Thinking**: Don't just provide code, explain WHY
2. **Handle Edge Cases**: Missing data, API failures, data quality issues
3. **Production Mindset**: Rate limiting, error handling, logging
4. **Statistical Rigor**: Interpret results, don't just compute metrics
5. **Clear Documentation**: README that explains approach and findings

---

## Resources

### Domain Knowledge
- **Futures contracts**: Understanding expiration and rolling
- **Market microstructure**: How markets actually work at tick level
- **Time series analysis**: Serial correlation, heteroscedasticity, normality

### Technical References
- pandas documentation for time series operations
- scipy.stats for statistical tests
- API documentation for Refinitiv services

---

## Notes

- This is a **7-day open challenge**, can pause and resume
- Focus on **correctness** over optimization initially
- **Document your assumptions** clearly
- Test with small datasets first before full runs
- Keep track of questions to ask recruiter if needed

---

## Contact

For technical issues with the HackerRank platform, contact ADIA Recruitment Team.