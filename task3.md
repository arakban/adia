Here's the transcription for Task 3:

```markdown
# Task 3: DC File Ingestion

## Problem Statement

Build a pipeline to process stock loan data from several prime brokers where the datasets consist of columnar data (date, stockid, broker, measure_one, measure_two). The zip file from the broker is called 'dc-test-file-ingestion.zip' and you can assume it is in the FILEPATH (confirm by os.listdir() if you'd like). 

Design a data schema for storing this data, including any raw and derived tables you would like to use. Clean as many data errors as you can find so that the table is useful for machine learning.

Document your assumptions in the code. Your answer should in the form of a python function.

---

## Production Robustness Discussion

Finally, in words, describe a robust ingestion pipeline that can handle such messy data files, the infrastructure you would need to run this pipeline. 

What are the possible "real world" scenarios that you need to handle which are outside of the "academic" scenario where clean data comes daily? 

How would you detect these cases and handle them? Does this list match the data issues you see in the provided sample data? 

Include this as a comment block in your code.

---

## Key Requirements

1. **Data Processing**:
   - Extract data from zip file
   - Parse columnar data (date, stockid, broker, measure_one, measure_two)
   - Design schema for raw and derived tables
   - Clean data errors for ML readiness

2. **Code Structure**:
   - Implement as Python function
   - Document assumptions in comments
   - Include production pipeline discussion as comment block

3. **Production Considerations** (to discuss in comments):
   - Infrastructure requirements
   - Handling messy/malformed data
   - Data quality monitoring
   - Error detection and handling
   - Real-world scenarios beyond clean daily data

---

## Expected Deliverable

- Python function that:
  - Reads from 'dc-test-file-ingestion.zip'
  - Processes and cleans stock loan data
  - Creates appropriate data schema
  - Handles data quality issues
- Comment block describing production-grade pipeline design
- Documentation of assumptions and data quality decisions

---

## Domain Context

**Stock Loan Data**: 
- Financial data from prime brokers
- Used for securities lending analysis
- Requires high data quality for ML applications
- Common issues: missing values, format inconsistencies, duplicates, outliers

## Implementation Approach

### Step 1: Data Exploration
```python
import zipfile
import pandas as pd
import os

# Confirm file exists
print(os.listdir('.'))

# Extract and explore
with zipfile.ZipFile('dc-test-file-ingestion.zip', 'r') as z:
    z.extractall('data/')
    
# Load and inspect
df = pd.read_csv('data/...')
print(df.head())
print(df.info())
print(df.describe())
```

### Step 2: Data Quality Checks
- Missing values
- Data type mismatches
- Date format consistency
- Duplicate records
- Outliers in measures
- Invalid stockids/broker names

### Step 3: Schema Design
- **Raw table**: Preserve original data
- **Cleaned table**: Post-validation data
- **Derived table**: Aggregations/features for ML

### Step 4: Production Pipeline Discussion
Consider:
- Late-arriving data
- Schema changes
- Data volume spikes
- Corrupted files
- Partial data
- Broker-specific quirks
```

This captures the complete Task 3 requirements with emphasis on both the coding task and the system design discussion component.