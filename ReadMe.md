# SEC Data Exploration and Stock Price Analysis

This repository contains Python tools for downloading, processing, and analyzing SEC XBRL financial data combined with stock price information from Stooq. The project enables comprehensive financial analysis by correlating SEC filings with stock price trends to build predictive models.

## Overview

The project processes SEC XBRL data from 2015-2025 to extract financial features and correlates them with stock price movements to build predictive models. It includes data download, feature engineering, stock price analysis, and machine learning components for predicting stock price trends.

## Project Structure

### Data Directories

- **`data/SEC_raw_2015_to_2025/`** - Raw SEC XBRL data files organized by quarters (2015q1 through 2025q2). Each quarter directory contains XBRL submission files (num.txt, pre.txt, sub.txt, tag.txt).

- **`data/featurized_2015_to_2025/`** - Processed and featurized financial data. Contains:
  - Quarterly featurized files (e.g., `2015q1_featurized.csv`)
  - Combined files: `featurized_all_quarters.csv`, `featurized_simplified.csv`
  - Feature completeness ranking: `feature_completeness_ranking.csv`

- **`data/stock_Stooq_daily_US/`** - Stock price data from Stooq, organized by exchange (NASDAQ, NYSE, NYSEMKT) and instrument type (stocks, ETFs). Contains:
  - Exchange-specific directories (e.g., `nasdaq_stocks_1/`, `nyse_etfs/`)
  - `derived_data/` - Processed stock data:
    - `month_end_price_stooq.csv` - Monthly end-of-period prices
    - `price_trends_1month.csv` - 1-month price trend labels
    - `price_trends_3month.csv` - 3-month price trend labels

- **`data/archive/`** - Archived older datasets (2022-only featurized data, legacy SEC data, Yahoo Finance stock data)

- **`data/explore_stooq/`** - Temporary directory for Stooq zip file downloads

- **`notebooks/`** - Jupyter notebooks for data exploration and analysis

### Python Files

#### Data Download and Population

- **`download_sec.py`** - Comprehensive SEC XBRL data downloader. Discovers and downloads multiple quarters/years (2015-2025) with retry logic, progress reporting, and automatic directory structure creation. Outputs to `data/SEC_raw_2015_to_2025/`.

For deprecated/archived scripts, see `archive/README.md`.

#### Data Exploration and Analysis

- **`exploreTags.py`** - Explores and analyzes XBRL tags within the SEC data. Helps identify the most relevant financial metrics and tags for feature engineering. Generates tag statistics and rankings.

- **`featurize.py`** - Processes top tags from 10-Q and 10-K reports to create numerical features. Performs featurization quarter-by-quarter without stitching data across quarters. Includes gradient features (quarter-over-quarter changes) and feature completeness filtering. Outputs to `data/featurized_2015_to_2025/`.

- **`validate_featurized_data.py`** - Debugging and validation tool for the featurization process. Ensures data quality and correctness of the feature engineering pipeline.

- **`spotcheck.py`** - Spot-check tool to compare XBRL tags used by different companies in their 10-Q reports. Identifies common tags, unique tags, and finds closest matching tags for key financial metrics across companies. Useful for understanding tag consistency in SEC filings.

#### Stock Price Analysis

- **`stock_stooq.py`** - Main stock data processor for Stooq dataset. Handles:
  - Preparation of directory structure from downloaded zip file
  - Processing stock data from NASDAQ, NYSE, and NYSEMKT exchanges
  - CIK-to-ticker mapping and standardization
  - Calculation of price trends with configurable look-ahead horizons (1-month and 3-month)
  - Generation of derived data files (month-end prices, trend labels)
  - Outputs to `data/stock_Stooq_daily_US/derived_data/`

#### Machine Learning

- **`baseline_model.py`** - Implements baseline machine learning models (XGBoost, LightGBM) for predicting stock price trends using featurized financial data. Includes:
  - Feature importance ranking
  - Model performance evaluation
  - Investment simulation and backtesting
  - Long-term gain analysis

#### Utility Files

- **`utility_data.py`** - Core utility functions for data processing:
  - SEC XBRL data loading and joining
  - CIK-to-ticker mapping
  - Stock price trend calculation
  - Data filtering (by date range, price range, continuity)

- **`utility_binary_classifier.py`** - Utility functions for binary classification tasks:
  - Train/validation splitting strategies
  - Baseline binary classifier implementations

- **`config.py`** - Centralized configuration file containing:
  - Directory paths and file names
  - Featurization parameters
  - Machine learning parameters
  - Data filtering thresholds

## Requirements

Install required packages:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- pandas, numpy (data processing)
- scikit-learn, xgboost, lightgbm (machine learning)
- matplotlib, seaborn, plotly (visualization)
- requests (data downloading)
- jupyter (notebooks)

See `requirements.txt` for complete list with version specifications.

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd sec_report_predict
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Download SEC Data

Download SEC XBRL data for all available quarters (2015-2025):

```bash
python download_sec.py
```

This will download data to `data/SEC_raw_2015_to_2025/` organized by quarter.

### 2. Explore and Featurize Data

Explore XBRL tags and create features:

```bash
# Explore XBRL tags and generate statistics
python exploreTags.py

# Create features from financial data
# This processes all quarters in data/SEC_raw_2015_to_2025/
python featurize.py

# Validate featurization results
python validate_featurized_data.py
```

### 3. Prepare and Process Stock Data

**Prerequisites**: Download US daily stock data from [Stooq](https://stooq.com/db/h/) (requires authentication). Save the zip file as `d_us_txt.zip` in `data/explore_stooq/`.

Process stock data and calculate trends:

```bash
# Process Stooq stock data and compute trend labels
# This will:
# - Unzip and reorganize directory structure if needed
# - Process stock files from all exchanges
# - Generate month-end prices and trend labels
python stock_stooq.py
```

### 4. Spot Check SEC Reports (Optional)

Compare XBRL tags across companies:

```bash
# Compare tags used by different companies in their 10-Q reports
python spotcheck.py
```

### 5. Build Machine Learning Models

Train baseline models:

```bash
# Train baseline models and evaluate performance
python baseline_model.py
```

## Data Flow

1. **Raw Data Collection**: SEC XBRL data is downloaded via `download_sec.py` and stored in `data/SEC_raw_2015_to_2025/` organized by quarters.

2. **Feature Engineering**: `featurize.py` processes XBRL tags to extract financial metrics, creates gradient features (quarter-over-quarter changes), and filters by completeness. Output stored in `data/featurized_2015_to_2025/`.

3. **Stock Price Analysis**: `stock_stooq.py` processes raw stock price files, standardizes tickers, maps CIKs to tickers, and computes price trends with configurable horizons. Output stored in `data/stock_Stooq_daily_US/derived_data/`.

4. **Model Training**: `baseline_model.py` loads featurized data and stock trends, trains ML models (XGBoost, LightGBM), evaluates performance, and runs investment simulations.

## Configuration

Most configuration parameters are centralized in `config.py`, including:
- Directory paths (DATA_DIR, SAVE_DIR, STOCK_DIR)
- Feature engineering parameters (DEFAULT_K_TOP_TAGS, DEFAULT_MIN_COMPLETENESS)
- ML parameters (TOP_K_FEATURES, TREND_HORIZON_IN_MONTHS, COMPLETENESS_THRESHOLD)
- Investment simulation parameters (INVEST_EXP_START_MONTH_STR, INVEST_EXP_END_MONTH_STR)

## Key Features

- **Comprehensive Data Pipeline**: End-to-end processing from raw SEC data (2015-2025) to ML-ready features
- **Stock Price Integration**: Correlates financial filings with actual stock performance from Stooq
- **Trend Analysis**: Computes price trends with configurable look-ahead horizons (1-month, 3-month)
- **Feature Engineering**: Includes gradient features (quarter-over-quarter changes) and completeness filtering
- **Data Quality Control**: Validation tools and handling of missing/incomplete financial data
- **Scalable Processing**: Handles large datasets with efficient batch processing
- **Tag Analysis**: Tools to explore and compare XBRL tags across companies and quarters

## Notes

- The project includes automatic download and retry logic for SEC data to handle network issues
- Data processing is designed to handle missing or incomplete financial data gracefully
- Stock price analysis includes handling of delisted companies and OTC market transitions
- All intermediate results are saved for reproducibility and debugging
- Stooq data requires manual download from their website (authentication required)

## Additional Resources

- **`recommendation.md`** - Guidance on reading financial reports and key metrics to focus on
- **`notebooks/`** - Jupyter notebooks for interactive data exploration
- **`TODO.md`** - Project roadmap and future improvements

## Contributing

Suggestions and improvements are welcome. Please ensure any changes maintain data quality and processing efficiency.
