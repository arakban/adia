# ADIA Task 3: DC File Ingestion - Solution

## Implementation Overview

Comprehensive stock loan data processing pipeline with production-grade data quality management and ML feature engineering.

## Core Function

**`process_stock_loan_data(filepath: str) -> Dict[str, pd.DataFrame]`**

### Data Schema Design

**Three-Table Architecture:**

1. **`raw_data`** - Preserves original data integrity
2. **`cleaned_data`** - Post-validation, ML-ready core data  
3. **`derived_data`** - Feature-engineered table with aggregations

### Data Quality Pipeline

**Input Schema:** `date, stockid, broker, measure_one, measure_two`

**Assumptions:**
- `measure_one`: Loan volume (shares)
- `measure_two`: Loan rate (percentage)  
- Data represents daily stock lending positions
- Missing brokers indicate transmission issues
- Extreme outliers are data entry errors

## Data Cleaning Operations

### 1. Date Standardization
- Multi-format parsing (`%Y-%m-%d`, `%m/%d/%Y`, etc.)
- Remove future dates and dates >10 years old
- Handle various international formats

### 2. Stock ID Validation  
- Uppercase normalization
- Remove invalid patterns (`NULL`, `ERROR`, etc.)
- Length validation (1-10 characters)

### 3. Broker Name Standardization
- Map common variations (`GS` → `GOLDMAN SACHS`)
- Standardize naming conventions
- Remove invalid broker entries

### 4. Measure Validation
- **Volume (measure_one)**: Remove negatives, cap at 99.9th percentile
- **Rate (measure_two)**: Constrain to reasonable range (-1% to 50%)
- Convert to numeric with error handling

### 5. Duplicate Handling
- Remove duplicates by `(date, stockid, broker)`
- Keep most recent record when duplicates exist

## ML Feature Engineering

### Time-Based Features
- `year`, `month`, `day_of_week`, `quarter`
- Weekend/holiday flags
- Rolling 7-day averages for volume and rates

### Lag Features  
- 1-day lag for volume and rates
- Change calculations (current - previous)

### Aggregation Features
- Stock-level statistics (mean, std, count)
- Broker-level patterns
- Cross-sectional rankings

### Categorical Encoding
- Label encoding for broker and stock identifiers
- Frequency-based encoding options

## Database Schema

**SQLite Implementation:**

```sql
-- Raw data preservation
CREATE TABLE raw_stock_loan (...);

-- Cleaned, validated data  
CREATE TABLE cleaned_stock_loan (...);
CREATE INDEX idx_cleaned_date_stock ON cleaned_stock_loan(date, stockid);

-- ML-ready features
CREATE TABLE ml_features (...);
CREATE INDEX idx_ml_date_stock ON ml_features(date, stockid);
```

## Production Pipeline Design

### Infrastructure Requirements

**Data Ingestion:**
- SFTP/FTP servers for broker file drops
- S3/Azure Blob with versioning
- Apache Airflow for orchestration
- Apache Kafka for real-time streaming

**Processing:**
- Apache Spark for distributed processing
- Docker/Kubernetes for scaling
- Redis for caching

**Storage:**
- Data Lake for raw archive
- Data Warehouse for structured data
- Time-series DB for high-frequency data

**Monitoring:**
- Prometheus/Grafana metrics
- ELK Stack logging
- Data quality dashboards

### Real-World Scenarios

**Beyond Clean Daily Data:**

1. **Late-Arriving Data** - Backfill procedures, SLA monitoring
2. **Schema Evolution** - Schema registry, compatibility checks  
3. **Volume Spikes** - Auto-scaling, partitioning
4. **Corrupted Files** - Checksum validation, atomic operations
5. **Broker Quirks** - Custom parsers, configuration management
6. **Regulatory Changes** - Flexible schema, audit trails
7. **System Outages** - Circuit breakers, retry logic
8. **Data Privacy** - PII masking, encryption

### Error Handling Strategy

**Four-Layer Validation:**
1. **File-level**: Size, format, checksum
2. **Schema-level**: Columns, data types  
3. **Business-level**: Value ranges, logic
4. **Cross-reference**: Historical patterns

**Quarantine System:**
- Isolate bad records for review
- Continue processing clean data
- Alert data stewards

**Quality Scoring:**
- Assign confidence scores
- Track quality trends
- Use in ML model weighting

## Mock Data Capability

When `dc-test-file-ingestion.zip` unavailable:
- Generates realistic stock loan data
- Includes intentional quality issues (5% bad data)
- Demonstrates full pipeline functionality
- Uses realistic distributions for volumes/rates

## Key Features

- **Robust File Handling**: Multiple formats, encodings, separators
- **Comprehensive Cleaning**: 8 validation layers
- **ML-Ready Output**: 15+ engineered features  
- **Production Design**: Scalable, monitored, fault-tolerant
- **Quality Monitoring**: Metrics, alerting, quarantine
- **Documentation**: Extensive assumptions and design decisions

## Usage

```python
# Process stock loan data
tables = process_stock_loan_data('/path/to/data')

# Access cleaned data
clean_df = tables['cleaned_data']
ml_df = tables['derived_data']

# Database automatically created as stock_loan_data.db
```

## Testing

Run with: `python3 task3.py`

Includes comprehensive testing with mock data generation and pipeline validation.