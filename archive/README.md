# Archive Directory

This directory contains deprecated and legacy scripts that are no longer part of the main project pipeline. These files are retained for reference and historical purposes.

## Archived Files

### `populate_data.py`

**Status**: Deprecated

**Purpose**: Legacy lightweight SEC data downloader for 2022 quarters only.

**Description**: 
- Simple script that downloads SEC XBRL data for Q1-Q4 of 2022
- Uses a basic `download_and_unzip()` function to fetch data from SEC.gov
- Downloads directly to hardcoded `data/2022q1/`, `data/2022q2/`, etc. directories
- Minimal error handling and progress reporting

**Why Deprecated**: 
- Replaced by `download_sec.py` which provides:
  - Comprehensive multi-year data download (2015-2025)
  - Automatic discovery of available quarters
  - Retry logic and better error handling
  - Progress reporting and status updates
  - Flexible configuration

**Usage** (for reference only):
```bash
python archive/populate_data.py
```

**Note**: This script may not work correctly with current project structure as it uses hardcoded paths.

---

### `download_sec_yf.py`

**Status**: Deprecated

**Purpose**: Legacy Yahoo Finance stock data downloader.

**Description**:
- Downloads stock price data from Yahoo Finance API (via `yfinance` library)
- Processes featurized data to map CIKs to ticker symbols
- Downloads historical stock prices in batches with rate limiting
- Generates month-end prices and trend labels
- Supports downloading missed tickers from previous runs

**Key Functions**:
- `collect_cik_ticker_pairs()`: Maps CIKs to ticker symbols from featurized data
- `download_stock_data()`: Downloads stock data in batches with rate limiting
- `download_missed_tickers()`: Downloads only tickers that were missed in previous runs
- `month_end_price()`: Extracts month-end closing prices
- `price_trend()`: Generates up-or-down trend labels with look-ahead horizon
- `closing_price_single_ticker()`: Fetches closing prices for individual tickers
- `closing_price_batch()`: Processes a batch of tickers and saves results

**Why Deprecated**:
- Replaced by `stock_stooq.py` which uses the Stooq dataset instead
- Stooq provides more reliable and comprehensive historical data
- Better handling of delisted companies and OTC transitions
- More consistent data format across exchanges

**Limitations**:
- Yahoo Finance API rate limiting and occasional data gaps
- Requires internet connection and API availability
- Hardcoded paths that may not match current project structure
- Less reliable for historical data

**Usage** (for reference only):
```bash
python archive/download_sec_yf.py
```

**Note**: This script contains hardcoded paths and may not work with the current project structure. Many of its utility functions (like `price_trend()`) have been moved to `utility_data.py` and are still actively used.

---

## Migration Guide

### From `populate_data.py` to `download_sec.py`

**Old approach**:
```python
# populate_data.py - downloads only 2022 quarters
download_and_unzip("https://www.sec.gov/files/.../2022q1.zip", extract_to="data/2022q1")
```

**New approach**:
```bash
# download_sec.py - downloads all available quarters (2015-2025)
python download_sec.py
```

### From `download_sec_yf.py` to `stock_stooq.py`

**Old approach**:
- Download stock data from Yahoo Finance API
- Process in batches with API rate limiting
- Generate trends from Yahoo Finance data

**New approach**:
1. Manually download Stooq data from https://stooq.com/db/h/ (requires authentication)
2. Save zip file to `data/explore_stooq/d_us_txt.zip`
3. Run `stock_stooq.py` to process the data

---

## Related Archived Data

The archived data directories are located in `data/archive/`:
- **`data/archive/featurized_2022/`** - Featurized data from 2022-only processing
- **`data/archive/SEC_raw_2022/`** - Raw SEC XBRL data for 2022 quarters
- **`data/archive/stock_202001_to_202507_yf/`** - Stock price data downloaded from Yahoo Finance

These directories contain data generated using the deprecated scripts and are kept for reference and comparison purposes.

---

## Notes

- These scripts are **not maintained** and may contain bugs or outdated dependencies
- Paths and configurations may not match the current project structure
- For new development, always use the current scripts in the root directory
- If you need functionality from these scripts, check if it has been migrated to the current codebase

---

## Contributing

If you find these archived scripts useful or need to reference their functionality, please check:
1. The main project scripts (`download_sec.py`, `stock_stooq.py`) first
2. Utility modules (`utility_data.py`, `utility_binary_classifier.py`)
3. The main project README.md for current best practices

