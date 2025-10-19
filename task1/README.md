# ADIA Task 1: E-mini S&P 500 Futures Analysis

Complete solution for ADIA Hackerrank Task 1 analyzing E-mini S&P 500 futures tick data.

## 📋 Task Overview

**Task 1a**: Form continuous price series by adjusting for rolls  
**Task 1b**: Sample observations by forming tick, volume, and dollar bars  
**Task 1c**: Analyze weekly bar count stability  
**Task 1d**: Compute serial correlation of returns  
**Task 1e**: Calculate variance of variances  
**Task 1f**: Apply Jarque-Bera normality test  

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Usage
```bash
python3 task1.py
```

**Choose from 4 options:**
1. Quick test with sample data (50K records)
2. Process full dataset (903M records) 
3. Save continuous series sample (1M records)
4. Run statistical analyses (Tasks 1c-1f)

## 📁 File Structure

```
task1/
├── task1.py                           # Main implementation
├── ES.h5                             # Input data (E-mini S&P 500 tick data)
├── requirements.txt                   # Python dependencies
├── ADIA_Task1_Analysis_Report.md     # Complete analysis report
├── README.md                         # This file
└── Generated Output Files:
    ├── tick_bars.csv                 # Tick bar OHLC data
    ├── volume_bars.csv               # Volume bar OHLC data  
    ├── dollar_bars.csv               # Dollar bar OHLC data
    ├── task_1c_bar_stability.csv     # Stability analysis results
    ├── task_1d_serial_correlation.csv # Correlation analysis results
    ├── task_1e_variance_of_variances.csv # Variance analysis results
    ├── task_1f_normality_tests.csv   # Normality test results
    ├── final_analysis_summary.csv    # Executive summary
    └── Charts:
        ├── task_1c_weekly_bar_counts.png
        ├── task_1d_serial_correlation.png
        ├── task_1e_variance_analysis.png
        └── task_1f_normality_analysis.png
```

## 🎯 Key Results

**Winner: Dollar Bars across all metrics**

| Task | Metric | Winner | Score |
|------|--------|--------|-------|
| 1c | Bar Stability (CV) | Dollar Bars | 0.1009 |
| 1d | Serial Correlation | Dollar Bars | 0.008741 |
| 1e | Variance of Variances | Dollar Bars | 3.87e-10 |
| 1f | Normality (JB stat) | Dollar Bars | 1,156.78 |

## 🏗️ Implementation Features

- **Memory Efficient**: Processes 2GB+ files in chunks
- **Checkpoint System**: Resume from interruptions
- **Production Ready**: Error handling and progress tracking
- **Comprehensive Analysis**: Statistical tests and visualizations
- **Clean Code**: Object-oriented design with clear documentation

## 📊 Analysis Methods

### Task 1a: Roll Adjustments
- Backward adjustment for contract rollovers
- Quarterly futures pattern (H/M/U/Z)
- Continuous price series generation

### Task 1b: Bar Construction
- **Tick Bars**: Every N trades
- **Volume Bars**: Every N contracts  
- **Dollar Bars**: Every $N traded

### Tasks 1c-1f: Statistical Analysis
- **Stability**: Coefficient of variation
- **Efficiency**: Serial correlation tests
- **Homoscedasticity**: Variance of variances
- **Normality**: Jarque-Bera and other tests

## 💡 Key Insights

**Why Dollar Bars Win:**
1. **Information-time sampling** vs clock-time
2. **Dual adaptation** to price AND volume
3. **Market microstructure** noise reduction
4. **Statistical properties** ideal for modeling

## 🔧 Technical Details

**Languages**: Python 3.9+  
**Key Libraries**: pandas, numpy, scipy, statsmodels, matplotlib  
**Data Format**: HDF5 input, CSV outputs  
**Processing**: Streaming/chunked for memory efficiency  
**Testing**: Statistical hypothesis testing  

## 📈 Expected Runtime

- **Sample (50K)**: ~2 minutes
- **Full dataset (903M)**: ~30-60 minutes  
- **Analysis only**: ~5 minutes

