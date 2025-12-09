#!/usr/bin/env python3
"""
Stock Data Processing from Stooq Directories

This module handles both preparation and processing of Stooq stock data:
1. Prepares directory structure from downloaded zip file
2. Processes stock data from Stooq directories, reading closing prices
   from NASDAQ, NYSE, and NYSEMKT stock files and standardizing ticker symbols

Prerequisites:
- Download the stock data from Stooq site https://stooq.com/db/h/
- Save the zip file as d_us_txt.zip in data/explore_stooq/

Functions:
- prepare_stooq_data(): Prepare directory structure from zip file
- load_stooq_stock_data(): Main function to load and process all stock data
- process_stock_directory(): Helper function to process a specific stock directory
"""

import pandas as pd
import numpy as np
import os
import glob
import zipfile
import shutil
import re
from pathlib import Path
from utility_data import get_cik_ticker_mapping, price_trend
from utility_data import remove_cik_w_missing_month, filter_by_date_continuity, filter_by_date_range, filter_by_price_range
from config import DATA_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================

# Preparation paths
ZIP_FILE_PATH = os.path.join(DATA_DIR, 'explore_stooq', 'd_us_txt.zip')
EXTRACT_DIR = os.path.join(DATA_DIR, 'explore_stooq', 'extracted')
STOOQ_DATA_DIR = os.path.join(EXTRACT_DIR, 'data', 'daily', 'us')

# Base directory containing Stooq stock data (after preparation)
STOOQ_BASE_DIR = os.path.join(DATA_DIR, 'stock_Stooq_daily_US')
STOOQ_SAVE_DIR = os.path.join(STOOQ_BASE_DIR, 'derived_data')

# Stock exchange directories and their corresponding output variable names
STOCK_EXCHANGES = {
    'nasdaq_stock*': 'df_nasdaq',
    'nyse_stock*': 'df_nyse', 
    'nysemkt_stock*': 'df_nysemkt'
}

# =============================================================================
# PREPARATION FUNCTIONS (from prepare_stooq.py)
# =============================================================================

def unzip_stooq_data():
    """
    Unzip the Stooq data file.
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("=" * 80)
    print("Step 1: Unzipping Stooq data file")
    print("=" * 80)
    
    if not os.path.exists(ZIP_FILE_PATH):
        print(f"❌ Error: Zip file not found at {ZIP_FILE_PATH}")
        return False
    
    print(f"📦 Found zip file: {ZIP_FILE_PATH}")
    
    # Create extraction directory if it doesn't exist
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    
    try:
        with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
            print(f"📂 Extracting to: {EXTRACT_DIR}")
            zip_ref.extractall(EXTRACT_DIR)
        print("✅ Successfully unzipped file")
        return True
    except Exception as e:
        print(f"❌ Error unzipping file: {e}")
        return False


def normalize_directory_name(name):
    """
    Normalize directory name by removing spaces and converting to lowercase.
    
    Args:
        name (str): Original directory name (e.g., "nyse stocks")
        
    Returns:
        str: Normalized name (e.g., "nyse_stocks")
    """
    # Remove spaces and replace with underscores
    normalized = name.replace(' ', '_').lower()
    return normalized


def flatten_stocks_directory(stocks_dir_path, exchange_name):
    """
    Flatten a stocks directory structure by merging numbered subdirectories.
    
    If the directory contains subdirectories "1", "2", "3", etc., it will:
    - Move contents from each numbered subdirectory up
    - Create new directories like "nyse_stocks_1", "nyse_stocks_2", etc.
    
    Args:
        stocks_dir_path (str): Path to the stocks directory (e.g., "daily/us/nyse stocks")
        exchange_name (str): Exchange name (e.g., "nyse", "nasdaq", "nysemkt")
        
    Returns:
        list: List of created directory paths
    """
    created_dirs = []
    stocks_dir = Path(stocks_dir_path)
    
    if not stocks_dir.exists():
        print(f"⚠️  Directory does not exist: {stocks_dir_path}")
        return created_dirs
    
    # Check if there are numbered subdirectories (1, 2, 3, etc.)
    numbered_subdirs = []
    for item in stocks_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            numbered_subdirs.append(item)
    
    numbered_subdirs.sort(key=lambda x: int(x.name))  # Sort by number
    
    if numbered_subdirs:
        # Flatten structure: create new directories like "nyse_stocks_1"
        print(f"  📁 Found {len(numbered_subdirs)} numbered subdirectories")
        
        for subdir in numbered_subdirs:
            number = subdir.name
            new_dir_name = f"{exchange_name}_stocks_{number}"
            new_dir_path = Path(STOOQ_BASE_DIR) / new_dir_name
            
            # Create the new directory
            new_dir_path.mkdir(parents=True, exist_ok=True)
            
            # Move all files from numbered subdirectory to new directory
            files_moved = 0
            for file_path in subdir.rglob('*'):
                if file_path.is_file():
                    dest_path = new_dir_path / file_path.name
                    if not dest_path.exists():  # Avoid overwriting
                        shutil.copy2(file_path, dest_path)
                        files_moved += 1
            
            print(f"    ✅ Created {new_dir_name} with {files_moved} files")
            created_dirs.append(str(new_dir_path))
    else:
        # No numbered subdirectories - just copy all files to a single directory
        normalized_name = normalize_directory_name(f"{exchange_name} stocks")
        new_dir_path = Path(STOOQ_BASE_DIR) / normalized_name
        new_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all files from stocks directory
        files_moved = 0
        for file_path in stocks_dir.rglob('*'):
            if file_path.is_file():
                dest_path = new_dir_path / file_path.name
                if not dest_path.exists():  # Avoid overwriting
                    shutil.copy2(file_path, dest_path)
                    files_moved += 1
        
        print(f"    ✅ Created {normalized_name} with {files_moved} files")
        created_dirs.append(str(new_dir_path))
    
    return created_dirs


def process_etfs_directory(etfs_dir_path, exchange_name):
    """
    Process an ETFs directory (just rename to remove spaces).
    
    Args:
        etfs_dir_path (str): Path to the ETFs directory
        exchange_name (str): Exchange name (e.g., "nyse", "nasdaq", "nysemkt")
        
    Returns:
        str: Path to created directory, or None if failed
    """
    etfs_dir = Path(etfs_dir_path)
    
    if not etfs_dir.exists():
        return None
    
    normalized_name = normalize_directory_name(f"{exchange_name} etfs")
    new_dir_path = Path(STOOQ_BASE_DIR) / normalized_name
    new_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Copy all files from ETFs directory
    files_moved = 0
    for file_path in etfs_dir.rglob('*'):
        if file_path.is_file():
            dest_path = new_dir_path / file_path.name
            if not dest_path.exists():  # Avoid overwriting
                shutil.copy2(file_path, dest_path)
                files_moved += 1
    
    print(f"    ✅ Created {normalized_name} with {files_moved} files")
    return str(new_dir_path)


def reorganize_directories():
    """
    Reorganize the Stooq directory structure according to requirements.
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "=" * 80)
    print("Step 2: Reorganizing directory structure")
    print("=" * 80)
    
    if not os.path.exists(STOOQ_DATA_DIR):
        print(f"❌ Error: Stooq data directory not found at {STOOQ_DATA_DIR}")
        print("   Make sure the zip file was extracted correctly.")
        return False
    
    print(f"📂 Processing directories in: {STOOQ_DATA_DIR}")
    
    # Create final base directory
    os.makedirs(STOOQ_BASE_DIR, exist_ok=True)
    
    # Process each directory in daily/us/
    for item in Path(STOOQ_DATA_DIR).iterdir():
        if not item.is_dir():
            continue
        
        dir_name = item.name.lower()
        print(f"\n📁 Processing: {item.name}")
        
        # Extract exchange name and type (stocks or etfs)
        parts = dir_name.split()
        if len(parts) < 2:
            print(f"  ⚠️  Skipping directory with unexpected name format: {item.name}")
            continue
        
        exchange_name = parts[0]  # nasdaq, nyse, or nysemkt
        dir_type = parts[1]  # stocks or etfs
        
        if dir_type == 'stocks':
            # Process stocks directory (may have numbered subdirectories)
            created_dirs = flatten_stocks_directory(str(item), exchange_name)
            if not created_dirs:
                print(f"  ⚠️  No directories created for {item.name}")
        elif dir_type == 'etfs':
            # Process ETFs directory (just rename)
            created_dir = process_etfs_directory(str(item), exchange_name)
            if not created_dir:
                print(f"  ⚠️  Failed to process ETFs directory: {item.name}")
        else:
            print(f"  ⚠️  Unknown directory type: {dir_type} (expected 'stocks' or 'etfs')")
    
    print("\n✅ Directory reorganization complete!")
    return True


def verify_directory_structure():
    """
    Verify that the final directory structure matches expected patterns.
    
    Checks for directories matching the patterns:
    - {exchange}_etfs (e.g., nasdaq_etfs, nyse_etfs)
    - {exchange}_stocks_{number} (e.g., nasdaq_stocks_1, nyse_stocks_2)
    - {exchange}_stocks (e.g., nysemkt_stocks, if no numbered subdirectories)
    
    Returns:
        bool: True if structure is correct, False otherwise
    """
    print("\n" + "=" * 80)
    print("Step 3: Verifying directory structure")
    print("=" * 80)
    
    if not os.path.exists(STOOQ_BASE_DIR):
        print(f"❌ Final base directory does not exist: {STOOQ_BASE_DIR}")
        return False
    
    # Get actual directories
    actual_dirs = [d.name for d in Path(STOOQ_BASE_DIR).iterdir() if d.is_dir()]
    actual_dirs.sort()
    
    # Expected patterns
    exchange_patterns = ['nasdaq', 'nyse', 'nysemkt']
    etf_pattern = re.compile(r'^({})_etfs$'.format('|'.join(exchange_patterns)))
    stocks_numbered_pattern = re.compile(r'^({})_stocks_(\d+)$'.format('|'.join(exchange_patterns)))
    stocks_unnumbered_pattern = re.compile(r'^({})_stocks$'.format('|'.join(exchange_patterns)))
    
    # Categorize directories
    etf_dirs = []
    stocks_numbered_dirs = []
    stocks_unnumbered_dirs = []
    unexpected_dirs = []
    
    for dir_name in actual_dirs:
        if etf_pattern.match(dir_name):
            etf_dirs.append(dir_name)
        elif stocks_numbered_pattern.match(dir_name):
            stocks_numbered_dirs.append(dir_name)
        elif stocks_unnumbered_pattern.match(dir_name):
            stocks_unnumbered_dirs.append(dir_name)
        else:
            unexpected_dirs.append(dir_name)
    
    # Sort stocks_numbered_dirs by exchange and number
    def sort_key(name):
        match = stocks_numbered_pattern.match(name)
        if match:
            exchange, number = match.groups()
            exchange_order = {'nasdaq': 0, 'nyse': 1, 'nysemkt': 2}
            return (exchange_order.get(exchange, 99), int(number))
        return (99, 0)
    
    stocks_numbered_dirs.sort(key=sort_key)
    
    print(f"📋 Total directories found: {len(actual_dirs)}")
    
    # Report ETFs directories
    if etf_dirs:
        print(f"\n📁 ETFs directories ({len(etf_dirs)}):")
        for dir_name in sorted(etf_dirs):
            dir_path = Path(STOOQ_BASE_DIR) / dir_name
            file_count = len(list(dir_path.rglob('*.txt'))) + len(list(dir_path.rglob('*.csv')))
            print(f"  • {dir_name:25s} ({file_count:,} files)")
    
    # Report numbered stocks directories
    if stocks_numbered_dirs:
        print(f"\n📁 Numbered stocks directories ({len(stocks_numbered_dirs)}):")
        for dir_name in stocks_numbered_dirs:
            dir_path = Path(STOOQ_BASE_DIR) / dir_name
            file_count = len(list(dir_path.rglob('*.txt'))) + len(list(dir_path.rglob('*.csv')))
            print(f"  • {dir_name:25s} ({file_count:,} files)")
    
    # Report unnumbered stocks directories
    if stocks_unnumbered_dirs:
        print(f"\n📁 Unnumbered stocks directories ({len(stocks_unnumbered_dirs)}):")
        for dir_name in sorted(stocks_unnumbered_dirs):
            dir_path = Path(STOOQ_BASE_DIR) / dir_name
            file_count = len(list(dir_path.rglob('*.txt'))) + len(list(dir_path.rglob('*.csv')))
            print(f"  • {dir_name:25s} ({file_count:,} files)")
    
    # Report unexpected directories
    if unexpected_dirs:
        print(f"\n⚠️  Unexpected directories ({len(unexpected_dirs)}):")
        for dir_name in sorted(unexpected_dirs):
            dir_path = Path(STOOQ_BASE_DIR) / dir_name
            file_count = len(list(dir_path.rglob('*.txt'))) + len(list(dir_path.rglob('*.csv')))
            print(f"  • {dir_name:25s} ({file_count:,} files)")
        print("   (These don't match expected naming patterns)")
    
    # Summary by exchange
    print("\n📊 Summary by exchange:")
    for exchange in exchange_patterns:
        exchange_etfs = [d for d in etf_dirs if d.startswith(f'{exchange}_etfs')]
        exchange_stocks_num = [d for d in stocks_numbered_dirs if d.startswith(f'{exchange}_stocks_')]
        exchange_stocks_un = [d for d in stocks_unnumbered_dirs if d.startswith(f'{exchange}_stocks')]
        
        total = len(exchange_etfs) + len(exchange_stocks_num) + len(exchange_stocks_un)
        if total > 0:
            parts = []
            if exchange_etfs:
                parts.append(f"{len(exchange_etfs)} ETFs")
            if exchange_stocks_num:
                parts.append(f"{len(exchange_stocks_num)} numbered stocks")
            if exchange_stocks_un:
                parts.append(f"{len(exchange_stocks_un)} unnumbered stocks")
            print(f"  • {exchange:10s}: {', '.join(parts)}")
    
    print("\n✅ Verification complete!")
    return True


def prepare_stooq_data():
    """
    Main function to prepare Stooq data directory structure from zip file.
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("=" * 80)
    print("Stooq Data Preparation")
    print("=" * 80)
    print(f"Zip file: {ZIP_FILE_PATH}")
    print(f"Extract to: {EXTRACT_DIR}")
    print(f"Final directory: {STOOQ_BASE_DIR}")
    print("=" * 80)
    
    # Step 1: Unzip
    if not unzip_stooq_data():
        print("\n❌ Failed to unzip data. Exiting.")
        return False
    
    # Step 2: Reorganize directories
    if not reorganize_directories():
        print("\n❌ Failed to reorganize directories. Exiting.")
        return False
    
    # Step 3: Verify structure
    verify_directory_structure()
    
    # Step 4: Clean up temporary extraction directory
    print("\n" + "=" * 80)
    print("Step 4: Cleaning up temporary extraction directory")
    print("=" * 80)
    
    if os.path.exists(EXTRACT_DIR):
        try:
            print(f"🗑️  Removing temporary directory: {EXTRACT_DIR}")
            shutil.rmtree(EXTRACT_DIR)
            print("✅ Successfully removed temporary extraction directory")
        except Exception as e:
            print(f"⚠️  Warning: Could not remove temporary directory: {e}")
            print("   You may need to manually delete it later.")
    else:
        print("✅ Temporary extraction directory already removed or doesn't exist")
    
    print("\n" + "=" * 80)
    print("🎉 Stooq data preparation complete!")
    print("=" * 80)
    print(f"📂 Prepared data is available at: {STOOQ_BASE_DIR}")
    print("=" * 80)
    
    return True

# =============================================================================
# STOCK PROCESSING FUNCTIONS
# =============================================================================

def process_stock_directory(directory_pattern: str, base_dir: str = STOOQ_BASE_DIR) -> pd.DataFrame:
    """
    Process stock data from a specific directory pattern.
    
    Args:
        directory_pattern (str): Pattern to match directories (e.g., 'nasdaq_stock*')
        base_dir (str): Base directory path
        
    Returns:
        pd.DataFrame: Processed stock data with standardized ticker symbols
    """
    print(f"🔍 Processing {directory_pattern}...")
    
    # Find all matching directories
    search_pattern = os.path.join(base_dir, directory_pattern)
    directories = glob.glob(search_pattern)
    
    if not directories:
        print(f"⚠️  No directories found matching pattern: {search_pattern}")
        return pd.DataFrame()
    
    print(f"📁 Found {len(directories)} directories: {[os.path.basename(d) for d in directories]}")
    
    all_stock_data = []
    
    for directory in directories:
        print(f"  📂 Processing directory: {os.path.basename(directory)}")
        
        # Find all data files in the directory (both .csv and .txt)
        csv_files = glob.glob(os.path.join(directory, "*.csv"))
        txt_files = glob.glob(os.path.join(directory, "*.txt"))
        data_files = csv_files + txt_files
        
        if not data_files:
            print(f"    ⚠️  No data files found in {directory}")
            continue
        
        # Count preferred stocks (files with "_" or "-" in filename)
        preferred_stocks = [f for f in data_files if '_' in os.path.basename(f) or '-' in os.path.basename(f)]
        regular_stocks = [f for f in data_files if '_' not in os.path.basename(f) and '-' not in os.path.basename(f)]
        
        print(f"    📄 Found {len(data_files)} data files ({len(csv_files)} CSV, {len(txt_files)} TXT)")
        print(f"    🚫 Skipping {len(preferred_stocks)} preferred stocks (with '_' or '-')")
        print(f"    ✅ Processing {len(regular_stocks)} regular stocks")
        
        for data_file in data_files:
            try:
                # Skip preferred stocks (files with "_" or "-" in filename)
                filename = os.path.basename(data_file)
                if '_' in filename or '-' in filename:
                    continue
                
                # Check if file is empty
                if os.path.getsize(data_file) == 0:
                    continue
                
                # Read the data file (handle both CSV and TXT)
                if data_file.endswith('.csv'):
                    df = pd.read_csv(data_file)
                else:  # .txt file - Stooq format is comma-separated
                    df = pd.read_csv(data_file, sep=',')
                
                # Skip if dataframe is empty
                if df.empty:
                    continue
                
                # Check if the file has the expected columns (Stooq format uses <CLOSE>)
                if '<CLOSE>' not in df.columns and 'Close' not in df.columns:
                    continue
                
                # Extract ticker from filename (remove extension)
                ticker = os.path.splitext(os.path.basename(data_file))[0]
                
                # Remove '.US' suffix from ticker if present
                if ticker.endswith('.US'):
                    ticker = ticker[:-3]  # Remove last 3 characters (.US)
                
                # Handle different column name formats
                if '<CLOSE>' in df.columns:
                    # Stooq format
                    close_col = '<CLOSE>'
                    date_col = '<DATE>'
                else:
                    # Standard format
                    close_col = 'Close'
                    date_col = 'Date'
                
                # Create a new dataframe with ticker and closing prices
                stock_df = df[[date_col, close_col]].copy()
                stock_df['ticker'] = ticker
                stock_df['exchange'] = os.path.basename(directory)
                
                # Rename columns for consistency
                stock_df = stock_df.rename(columns={date_col: 'date', close_col: 'close_price'})
                
                # Reorder columns
                stock_df = stock_df[['ticker', 'exchange', 'date', 'close_price']]
                
                all_stock_data.append(stock_df)
                
            except Exception as e:
                print(f"    ❌ Error processing {os.path.basename(data_file)}: {str(e)}")
                continue
    
    if not all_stock_data:
        print(f"⚠️  No valid stock data found for {directory_pattern}")
        return pd.DataFrame()
    
    # Combine all stock data
    combined_df = pd.concat(all_stock_data, ignore_index=True)
    
    # Convert date column to datetime (handle YYYYMMDD format)
    combined_df['date'] = pd.to_datetime(combined_df['date'], format='%Y%m%d', errors='coerce')
    
    # Sort by ticker and date
    combined_df = combined_df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    print(f"✅ Processed {len(combined_df)} records for {len(combined_df['ticker'].unique())} unique tickers")
    
    return combined_df


def load_stooq_stock_data() -> pd.DataFrame:
    """
    Load and process stock data from all Stooq directories.
    
    Returns:
        pd.DataFrame: Combined stock data with columns:
            - ticker: Stock ticker (uppercase, no .US suffix)
            - exchange: Exchange name (nasdaq_stocks_1, nyse_stocks_1, etc.)
            - date: Trading date
            - close_price: Closing price
            - cik: Central Index Key (mapped from ticker)
    """
    print("🚀 Starting Stooq stock data processing...")
    print(f"📁 Base directory: {STOOQ_BASE_DIR}")
    
    # Check if base directory exists
    if not os.path.exists(STOOQ_BASE_DIR):
        print(f"❌ Base directory does not exist: {STOOQ_BASE_DIR}")
        print(f"💡 Tip: Run prepare_stooq_data() first to prepare the directory structure")
        return pd.DataFrame()
    
    # Process each stock exchange and collect all data
    all_dataframes = []
    
    for directory_pattern, output_name in STOCK_EXCHANGES.items():
        print(f"\n{'='*60}")
        print(f"Processing {directory_pattern} -> {output_name}")
        print(f"{'='*60}")
        
        df = process_stock_directory(directory_pattern)
        if not df.empty:
            all_dataframes.append(df)
    
    # Combine all dataframes
    if not all_dataframes:
        print("❌ No data found in any exchange")
        return pd.DataFrame()
    
    df_combined = pd.concat(all_dataframes, ignore_index=True)
    
    # Process ticker names: convert to uppercase and remove .US suffix
    print(f"\n🔧 Processing ticker names...")
    df_combined['ticker'] = df_combined['ticker'].str.upper()
    df_combined['ticker'] = df_combined['ticker'].str.replace('.US', '', regex=False)
    
    # Add CIK column using ticker mapping
    print(f"🔗 Adding CIK mappings...")
    cik_to_ticker, ticker_to_cik = get_cik_ticker_mapping() 
    
    df_combined['cik'] = df_combined['ticker'].map(ticker_to_cik)
    
    # Drop records with null CIK values
    initial_count = len(df_combined)
    df_combined = df_combined.dropna(subset=['cik'])
    dropped_count = initial_count - len(df_combined)
    print(f"🗑️  Dropped {dropped_count:,} records with null CIK values")
    
    # Report mapping statistics
    mapped_count = df_combined['cik'].notna().sum()
    total_count = len(df_combined)
    print(f"✅ Mapped {mapped_count:,} out of {total_count:,} records ({mapped_count/total_count*100:.1f}%)")

    # Reorder columns
    df_combined = df_combined[['ticker', 'cik', 'exchange', 'date', 'close_price']]
    
    # Sort by ticker and date
    df_combined = df_combined.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total records: {len(df_combined):,}")
    print(f"Unique tickers: {len(df_combined['ticker'].unique()):,}")
    print(f"Unique exchanges: {len(df_combined['exchange'].unique()):,}")
    print(f"Date range: {df_combined['date'].min()} to {df_combined['date'].max()}")
    
    return df_combined


def month_end_price_stooq(df_combined):
    """
    Extract month-end price from the combined stock data.
    
    Args:
        df_combined (pd.DataFrame): Combined stock data with date and close_price columns
        
    Returns:
        pd.DataFrame: Month-end price data with one record per (cik, ticker, year_month)
    """
    df_combined['year_month'] = df_combined['date'].dt.to_period('M')
    
    # Group by (cik, ticker, year_month) and find the record with the largest Date in each group
    # This gives us the last trading day of each month for each company
    month_end_df = df_combined.loc[df_combined.groupby(['cik', 'ticker', 'year_month'])['date'].idxmax()].copy()
    
    # Rename columns for clarity
    month_end_df = month_end_df.rename(columns={'date': 'month_end_date'})
    
    # Select only the columns we need
    month_end_df = month_end_df[['cik', 'ticker', 'month_end_date', 'close_price', 'year_month']].reset_index(drop=True)
    return month_end_df

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Ensure save directory exists
    os.makedirs(STOOQ_SAVE_DIR, exist_ok=True)
    
    month_end_price_file = os.path.join(STOOQ_SAVE_DIR, 'month_end_price_stooq.csv')

    # Check if data needs to be prepared first
    if not os.path.exists(STOOQ_BASE_DIR) or not any(Path(STOOQ_BASE_DIR).iterdir()):
        print("📦 Stooq data directory not found or empty. Preparing data first...")
        if not prepare_stooq_data():
            print("\n❌ Failed to prepare Stooq data. Exiting.")
            exit(1)
        print("\n" + "=" * 80)
    else:
        # Clean up extract directory if it still exists from a previous run
        if os.path.exists(EXTRACT_DIR):
            print("🧹 Cleaning up leftover extraction directory...")
            try:
                shutil.rmtree(EXTRACT_DIR)
                print(f"✅ Removed temporary directory: {EXTRACT_DIR}")
            except Exception as e:
                print(f"⚠️  Warning: Could not remove temporary directory: {e}")
                print("   You may need to manually delete it later.")
    
    # Always regenerate month_end_price from source data to ensure it's up-to-date
    print("🔄 Loading stock data and generating month-end prices...")
    df_combined = load_stooq_stock_data()
        
    if df_combined.empty:
        print("\n❌ No stock data loaded. Exiting.")
        exit(1)
    
    if not os.path.exists(month_end_price_file):
        # remove records with date outside of 2000-01-01 to 2025-11-01
        df_combined = filter_by_date_range(df_combined, 'date', start_date='2000-01-01', end_date='2025-11-01')

        # # remove records with large gaps in date
        # df_combined, removed_ticker_info = filter_by_date_continuity(df_combined, 'date', gap_in_days=7)
        # print('debugging trace: removed tickers with max gap > 7 days:', removed_ticker_info[:5])

        month_end_df = month_end_price_stooq(df_combined)

        # Apply filters before saving
        month_end_df = remove_cik_w_missing_month(month_end_df)
        month_end_df = filter_by_price_range(month_end_df, 'close_price', min_price=1, max_price=1000)
        
        # Save the filtered month-end price data
        month_end_df.to_csv(month_end_price_file, index=False)
        month_end_df['year_month'] = month_end_df['year_month'].apply(lambda x: pd.to_datetime(x).to_period('M'))
        print(f"\n💾 Month-end prices saved to: {month_end_price_file}")
    else:
        month_end_df = pd.read_csv(month_end_price_file)
        # Convert year_month back to Period object (CSV loads it as string)
        month_end_df['year_month'] = pd.to_datetime(month_end_df['year_month']).dt.to_period('M')

    # Calculate trends
    for horizon in [12, 3, 1]:
        trend_df = price_trend(month_end_df, trend_horizon_in_months=horizon)
        if len(trend_df) > 0: 
            # Save results to CSV
            output_file = os.path.join(STOOQ_SAVE_DIR, f'price_trends_{horizon}month.csv')
            trend_df.to_csv(output_file, index=False)
            print(f"\n💾 {horizon}-month trends saved to: {output_file}")

    print("\n" + "=" * 80)
    print("🎉 Stock data processing complete!")
    print("=" * 80)
