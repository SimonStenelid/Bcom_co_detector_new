# Bot Pattern Analyzer

A comprehensive batch pattern miner for detecting automated/bot traffic in flight search logs using machine learning and rule-based scoring.

## Features

- **Cohort-level Analysis**: Groups traffic by site×partner×IP×user_agent×day
- **Advanced Feature Engineering**: 40+ engineered features including timing patterns, signature repetition, and client behavior
- **Dual Scoring System**: Combines rule-based scoring with Isolation Forest anomaly detection
- **Comprehensive Reporting**: Generates CSV, JSON, and Markdown reports with evidence
- **Historical Analysis**: EWMA-based trend analysis and cross-day comparisons
- **Robust Data Handling**: Graceful handling of missing data and edge cases

## Quick Start

### Installation

```bash
# Install dependencies
pip install -e .

# Or using uv (recommended)
uv pip install -e .
```

### Basic Usage

#### For Real Booking Data (Batch Processing - Recommended)
```bash
# Process ALL data files in data/raw folder automatically
python process_all_data.py 2025-09-05

# Or use CLI batch mode directly
python -m botminer.cli pipeline 2025-09-05 --batch --verbose
```

#### For Single File Processing
```bash
# Process single booking data file with IP addresses
python -m botminer.cli pipeline 2025-09-05 --csv-path booking_data.csv --ip-csv ip_addresses.csv

# Or use the convenience script
python process_data.py booking_data.csv ip_addresses.csv 2025-09-05
```

#### For Example Data (Single CSV)
```bash
# Run the complete pipeline
python -m botminer.cli pipeline example_data.csv 2025-01-15

# Or run individual steps
python -m botminer.cli ingest example_data.csv 2025-01-15
python -m botminer.cli features 2025-01-15
python -m botminer.cli score 2025-01-15
python -m botminer.cli report 2025-01-15
```

### User Agent Analysis

The system automatically enriches user agent IDs with detailed browser information:

```bash
# Analyze user agent distribution
python analyze_user_agents.py 2025-09-05

# Analyze order patterns and conversion rates
python analyze_orders.py 2025-09-05
```

**Sample Analysis Results:**
- **Browser Distribution**: Chrome (64.4%), Safari (34.2%), Firefox (0.7%)
- **Device Types**: PC (49.7%), iPhone (32.8%), Android Phone (16.3%)
- **OS Distribution**: Windows (46.7%), iOS (33.7%), Android (16.6%)
- **User Agent Types**: Web Browser (56.7%), Application (43.2%)
- **Order Analysis**: 0.0% conversion rate, 2 total orders from 6,658 events

### Example Output

After running the pipeline, you'll find:

- `data/features/2025-01-15.parquet` - Engineered features with botness scores
- `reports/2025-01-15/cohorts.csv` - Detailed cohort data
- `reports/2025-01-15/cohorts.json` - Structured results
- `reports/2025-01-15/summary.md` - Human-readable summary

## Architecture

### Core Modules

- **`ingest.py`** - CSV processing and data normalization
- **`features.py`** - Cohort-level feature engineering
- **`scoring.py`** - Rule-based and anomaly detection scoring
- **`report.py`** - Report generation (CSV/JSON/Markdown)
- **`utils/`** - Time, signature, and statistical utilities

### Feature Categories

1. **Time/Cadence**: Hourly patterns, night activity, schedule regularity
2. **Repetition**: Signature diversity, exact repeats, ladder sweeps
3. **Client Hygiene**: Cookie rates, referrer rates, HTTP versions
4. **Volume**: Event counts, session ratios
5. **Outcomes**: Status code distributions, error bursts
6. **Historical**: EWMA trends, cross-day comparisons

### Scoring Methodology

1. **Rule-based Score**: Weighted combination of behavioral indicators
2. **Anomaly Score**: Isolation Forest on normalized features
3. **Combined Score**: `α × rule_score + (1-α) × anomaly_score`
4. **Risk Bands**: High/Medium/Low based on configurable thresholds

## Configuration

Edit `config.yaml` to customize:

- Scoring weights and thresholds
- Feature engineering parameters
- File paths and output formats
- Model hyperparameters

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=botminer tests/

# Run specific test file
pytest tests/test_utils.py -v
```

## Input Data Format

### Real Booking Data (Two CSV Files + User Agent Mapping)

**Main CSV (booking data):**
- `Created Time`, `Site`, `Partner Domain`
- `OriginCity`, `DestinationCity`, `JourneyType`
- `Out Date`, `Return Date`
- `Num Adt`, `Num Cnn`, `Num Inf` (passenger counts)
- `Cabin Classes`, `Total Stops`
- `MarketingCarriers`, `Sessions ID`
- `ID` (10-digit click ID for merging)

**IP CSV (IP addresses):**
- `mdc.addr` (IP address)
- `mdc.click` (10-digit click ID - matches main CSV ID)
- `mdc.request.id`

**User Agent Excel File (browser mapping):**
- `User Agent Major Details List.xlsx` - Maps 4-digit user agent IDs to browser information
- Contains 4,574+ user agent mappings with browser type, device type, OS type, etc.

### Example Data (Single CSV)
- `timestamp` (ISO format)
- `site`, `partner`
- `origin`, `destination`
- `journey_type` (OW/RT)
- `dep_date`, `ret_date`
- `pax_config`, `cabin`, `stops`
- `ip`, `user_agent`
- `session_id`, `referrer`
- `cookie_present`, `http_version`
- `status_code`, `request_id`

## Output Reports

### CSV Report
Detailed cohort data with all features and scores.

### JSON Report
Structured data with metadata, statistics, and top risk cohorts.

### Markdown Summary
Human-readable report with:
- Executive summary
- Risk distribution
- Score statistics
- Top risk cohorts table
- Methodology explanation

## Key Indicators

The system detects bots based on:

- **High Night Activity**: Automated traffic during off-peak hours
- **Signature Repetition**: Repeated search patterns
- **Low Client Hygiene**: Missing cookies, referrers, or unusual HTTP patterns
- **Rapid Requests**: Very short time gaps between requests
- **Ladder Sweeps**: Systematic date progression patterns
- **Anomalous Behavior**: Unusual patterns detected by machine learning

## Performance

- Processes thousands of cohorts per minute
- Memory-efficient with Parquet storage
- Parallel processing for large datasets
- Configurable batch sizes and thresholds

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
