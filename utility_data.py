import pandas as pd
import numpy as np
import requests
import os
import json
from config import DATA_DIR, STOCK_DIR


def clean_cik_dup_submissions(submissions):
    """
    Removes companies that filed multiple submissions for the same (form, period) combination.
    
    A company (cik) should file only one submission (adsh) for any given (form, period) tuple.
    If multiple submissions are found for the same (cik, form, period), ALL submissions 
    from that combination are removed to maintain data quality.
    For instance, in 2022q4, we have 2 submissions submissions from
     (cik==1590364) & (period==20220930)]
    This breaks the segments feature. 
    
    Args:
        submissions (DataFrame): DataFrame containing submission data with columns 
                               ['adsh', 'cik', 'form', 'period', ...]
    Returns:
        DataFrame: Cleaned submissions dataframe with problematic combinations removed
    """
    initial_submission_count = len(submissions)
    
    # Identify companies with multiple submissions for the same (form, period)
    submission_counts = submissions.groupby(['cik', 'form', 'period']).size()
    multiple_submissions = submission_counts[submission_counts > 1]
    
    if len(multiple_submissions) > 0:
        print(f"  Found {len(multiple_submissions)} cases of multiple submissions for same (cik, form, period)")
        
        # Get the (cik, form, period) combinations that have multiple submissions
        problematic_combinations = multiple_submissions.index.tolist()
        
        # Create a mask to exclude these problematic submissions
        mask = True
        for cik, form, period in problematic_combinations:
            current_mask = ~((submissions['cik'] == cik) & 
                           (submissions['form'] == form) & 
                           (submissions['period'] == period))
            mask = mask & current_mask
        
        # Apply the mask to filter out problematic submissions
        submissions_cleaned = submissions[mask].copy()
        
        removed_count = initial_submission_count - len(submissions_cleaned)
        print(f"  Removed {removed_count} submissions from {len(problematic_combinations)} problematic (cik, form, period) combinations")
        
        return submissions_cleaned
    else:
        # No problematic submissions found, return original dataframe
        return submissions.copy()


def load_and_join_sec_xbrl_data(data_dir):
    """
    Loads and joins SEC XBRL data from multiple quarters.
    
    Args:
        data_dir (list): List of directory paths containing SEC data files
        
    Returns:
        pd.DataFrame: Joined dataframe with SEC XBRL data from all quarters
    """ 
    df_joined = pd.DataFrame()
    
    for crt_dir in data_dir:
        print(f"Processing {crt_dir}...")
        # Read submission data (company information)
        submissions = pd.read_csv(f'{crt_dir}/sub.txt', sep='\t', low_memory=False, keep_default_na=False )
        submissions = submissions[
            ['adsh', 'cik', 'name', 'sic', 'fye', 'fy', 'fp', 'form', 'period']
        ]
        
        # Clean duplicate submissions for same (cik, form, period) combinations
        submissions = clean_cik_dup_submissions(submissions)
        
        # Read numeric facts data (financial metrics with tags)
        numericFacts = pd.read_csv( f'{crt_dir}/num.txt',  sep='\t', low_memory=False )
        # Join numeric facts with submission data on 'adsh' (accession number) - inner join
        joined = numericFacts.join(submissions.set_index('adsh'), on='adsh', how='inner')
        
        # Append to the main dataframe
        df_joined = pd.concat([df_joined, joined], ignore_index=True)
    
    df_joined['custom_tag'] = 1.0 * (df_joined['version'] == df_joined['adsh'])

    print(f"Total records loaded: {len(df_joined):,}")
    print(f"Total unique tags: {df_joined['tag'].nunique():,}")
    print(f"Total unique companies (CIK): {df_joined['cik'].nunique():,}")
    
    return df_joined


def top_tags(df_joined):
    """
    Utility function to explore SEC filing data: 
    1. Group by tag and count occurrences and distinct companies (cik)
    2. Display top 100 tags by occurrence count
    """
    # Group by tag and calculate statistics
    tag_stats = df_joined.groupby('tag').agg({
        'adsh': 'count',  # Count of occurrences
        'cik': 'nunique'  # Count of distinct companies
    }).rename(columns={
        'adsh': 'occurrences',
        'cik': 'distinct_companies'
    })
    
    # Sort by occurrences in descending order
    # NOTE: we sort by num of companies because some companies have ~6K segments 
    tag_stats_sorted = tag_stats.sort_values('distinct_companies', ascending=False)    
    tag_stats_sorted.to_csv('tag_stats_sorted.csv', index=True, header=True)


def read_tags_to_featurize(K_top_tags=50):
    """
    Reads tag statistics from CSV and creates a dataframe with top K tags to be used for featurization.
    
    Args:
        K_top_tags (int, default=50): Number of top tags to select
    
    Returns:
        DataFrame: DataFrame with columns ['rank', 'tag'] specifying which tags to featurize
    """
    print(f"Creating tags to featurize dataframe for top {K_top_tags} tags...")
    
    # Read the top tags from tag_stats_sorted.csv
    tag_stats = pd.read_csv(os.path.join(DATA_DIR, 'tag_stats_sorted.csv'), index_col=0)
    df_tags_to_featurize = pd.DataFrame({
        'rank': range(1, K_top_tags + 1),
        'tag': tag_stats.head(K_top_tags).index.tolist()
    })
    
    print(f"Selected {len(df_tags_to_featurize)} tags for featurization:")
    print(f"Top 5 tags: {df_tags_to_featurize['tag'].head().tolist()}")
    
    return df_tags_to_featurize


def print_featurization(df_featurized, cik, period, tag, qtrs):
    """
    Utility function to print all non-null features for a specific (cik, period, tag, qtrs) combination.
    
    Args:
        df_featurized (DataFrame): The featurized dataframe from organize_feature_dataframe()
        cik (int): Company identifier
        period (str/int): Period identifier
        tag (str): Tag name (e.g., 'Assets', 'Revenues')
        qtrs (int): Number of quarters
    
    Returns:
        None: Prints the features to console
    """
    
    # Filter for the specific (cik, period) combination
    row_data = df_featurized[
        (df_featurized['cik'] == cik) & 
        (df_featurized['period'] == period)
    ]
    
    if len(row_data) == 0:
        print(f"No data found for CIK: {cik}, Period: {period}")
        return
    
    # Get the first (and should be only) matching row
    row = row_data.iloc[0]
    
    # Create the tag_qtrs pattern to look for
    tag_qtrs_pattern = f"{tag}_{qtrs}qtrs"
    
    # Find all columns that match this tag_qtrs pattern
    matching_columns = [col for col in df_featurized.columns if col.startswith(tag_qtrs_pattern)]
    
    if not matching_columns:
        print(f"No features found for Tag: {tag}, Qtrs: {qtrs}")
        print(f"Looking for pattern: {tag_qtrs_pattern}")
        return
    
    # Print header
    print(f"\n{'='*60}")
    print(f"FEATURES FOR CIK: {cik}, PERIOD: {period}, TAG: {tag}, QTRS: {qtrs}")
    print(f"{'='*60}")
    
    # Print non-null features
    found_features = False
    for col in sorted(matching_columns):
        value = row[col]
        if pd.notna(value):  # Check if value is not NaN
            feature_type = col.replace(tag_qtrs_pattern + '_', '')
            print(f"{feature_type:15}: {value:12.6f}")
            found_features = True
    
    if not found_features:
        print(f"All features for {tag_qtrs_pattern} are null/NaN")
    
    print(f"{'='*60}")


def get_cik_ticker_mapping():
    """
    Get CIK to ticker symbol mapping from SEC's company tickers JSON file.
    
    Returns:
        tuple: (cik_to_ticker, ticker_to_cik) dictionaries
            - cik_to_ticker: Mapping of CIK (str) to ticker symbol (str)
            - ticker_to_cik: Mapping of ticker symbol (str) to CIK (str)
    """
    try:
        # SEC's official company tickers JSON file
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {
            'User-Agent': 'SEC Data Analysis (your-email@domain.com)',  # SEC requires user agent
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to both CIK -> ticker and ticker -> CIK mappings
        cik_to_ticker = {}
        ticker_to_cik = {}
        for entry in data.values():
            cik = str(entry['cik_str']).zfill(10)  # Pad CIK to 10 digits
            ticker = entry['ticker']
            cik_to_ticker[cik] = ticker
            ticker_to_cik[ticker] = cik
            
        # print(f"✅ Loaded {len(cik_to_ticker)} CIK->ticker mappings from SEC")
        # print(f"✅ Loaded {len(ticker_to_cik)} ticker->CIK mappings from SEC")
        return cik_to_ticker, ticker_to_cik
        
    except Exception as e:
        print(f"❌ Error loading CIK->ticker mapping: {e}")
        return {}, {}



def load_company_tickers_exchange_mappings():
    """
    Load SEC ticker and exchange mappings, downloaded from 
    https://www.sec.gov/files/company_tickers_exchange.json
    
    Returns:
        tuple: (ticker_mapping, exchange_mapping) dictionaries
    Usage:
        ticker_mapping, exchange_mapping = load_company_tickers_exchange_mappings() 
    """
    sec_file = 'data/company_tickers_exchange.json'
    
    if not os.path.exists(sec_file):
        print(f"❌ SEC file not found: {sec_file}")
        return {}, {}
    
    with open(sec_file, 'r') as f:
        data = json.load(f)
    
    # Create mappings
    records = data['data']
    ticker_mapping = {record[2]: record[1] for record in records}  # ticker -> name
    exchange_mapping = {record[2]: record[3] for record in records}  # ticker -> exchange
    
    print(f"✅ Loaded SEC mappings for {len(ticker_mapping)} companies")
    return ticker_mapping, exchange_mapping


def remove_cik_w_missing_month(month_end_df):
    """
    Remove (cik, ticker) tuples that have inconsecutive year_month records.
    
    This function identifies (cik, ticker) pairs with missing months in their data
    and removes all records for those pairs from the DataFrame. 
    Missing months may happen if a company gets delisted or goes onto the
    OTC (over the counter) market. 
    
    Args:
        month_end_df (pd.DataFrame): DataFrame with columns 
        ['cik', 'ticker', 'month_end_date', 'close_price', 'year_month']
        
    Returns:
        pd.DataFrame: Filtered DataFrame with only (cik, ticker) pairs that 
        have consecutive months
    """
    if month_end_df.empty:
        print("❌ No data provided for processing")
        return month_end_df
    
    # Check required columns
    required_cols = ['cik', 'ticker', 'year_month']
    if not all(col in month_end_df.columns for col in required_cols):
        print(f"❌ Missing required columns. Need: {required_cols}")
        return month_end_df
    
    violations = []
    
    # Group by (cik, ticker) and check each pair
    for (cik, ticker), group in month_end_df.groupby(['cik', 'ticker']):
        # Sort by year_month
        group = group.sort_values('year_month')
        months = group['year_month'].tolist()
        
        # Check if months are consecutive
        is_consecutive = True
        missing_months = []
        
        if len(months) > 1:
            # Convert periods to integers for easier comparison
            month_ints = [int(str(month).replace('-', '')) for month in months]
            
            for i in range(len(month_ints) - 1):
                current_month = month_ints[i]
                next_month = month_ints[i + 1]
                
                # Calculate expected next month
                if current_month % 100 == 12:  # December
                    expected_next = (current_month // 100 + 1) * 100 + 1  # Next year, January
                else:
                    expected_next = current_month + 1
                
                if next_month != expected_next:
                    is_consecutive = False
                    # Find missing months between current and next
                    missing_start = current_month
                    missing_end = next_month
                    
                    # Generate missing months
                    temp_month = missing_start
                    while temp_month < missing_end:
                        if temp_month % 100 == 12:
                            temp_month = (temp_month // 100 + 1) * 100 + 1
                        else:
                            temp_month += 1
                        if temp_month < missing_end:
                            missing_months.append(f"{temp_month//100:04d}-{temp_month%100:02d}")
        
        if not is_consecutive:
            violations.append((cik, ticker, missing_months))
    
    # Remove records for (cik, ticker) pairs with missing months
    if violations:
        print(f"Removing {len(violations)} (cik, ticker) pairs with missing months:")
        for cik, ticker, missing_months in violations:
            print(f"  CIK: {cik}, Ticker: {ticker}")
            if missing_months:
                print(f"    Missing months: {missing_months}")
        
        # Create mask to keep only records not in violations
        violation_pairs = [(cik, ticker) for cik, ticker, _ in violations]
        mask = ~month_end_df.apply(lambda row: (row['cik'], row['ticker']) in violation_pairs, axis=1)
        filtered_df = month_end_df[mask].copy()
        
        print(f"Removed {len(month_end_df) - len(filtered_df)} records")
        return filtered_df
    else:
        print("No (cik, ticker) pairs with missing months found")
        return month_end_df


def price_trend(month_end_df, trend_horizon_in_months):
    """
    Generate up-or-down trend labels for month-end prices with look-ahead horizon.
    
    This function takes month-end price data and calculates trend labels by looking
    ahead for a specified number of months to determine if prices go up or down.
    
    Args:
        month_end_df (pd.DataFrame): DataFrame with columns (cik, ticker, month_end_date, close_price, year_month)
        trend_horizon_in_months (int): Number of months to look ahead for trend calculation
        
    Returns:
        pd.DataFrame: DataFrame with columns (cik, ticker, month_end_date, trend_up_or_down, trend_5per_up, price_change)
                     where trend_up_or_down is 1 for price going up, 0 for price going down,
                     trend_5per_up is 1 for price going up more than 5%, 0 otherwise,
                     and price_change is the ratio of future_close_price to close_price
    """
    print("📈 Computing price trends...")
    print("=" * 50)
    print(f"🔍 Trend horizon: {trend_horizon_in_months} months")
    
    if month_end_df.empty:
        print("❌ No month-end data provided")
        return pd.DataFrame()
    
    # Calculate trends using DataFrame join approach
    print(f"\n📊 Calculating trends with {trend_horizon_in_months} month horizon...")
    
    # Add year_month_horizon column and create future price lookup
    month_end_df = month_end_df.copy()
    month_end_df['year_month_horizon'] = month_end_df['year_month'] + trend_horizon_in_months
    
    # Create future price lookup table
    future_df = month_end_df[['cik', 'ticker', 'year_month', 'close_price']].rename(columns={
        'year_month': 'year_month_horizon',
        'close_price': 'future_close_price'
    })
    
    # Join to get future prices and calculate trends
    trend_df = (month_end_df.merge(future_df, on=['cik', 'ticker', 'year_month_horizon'], how='left')
                .dropna(subset=['future_close_price'])
                .assign(
                    trend_up_or_down=lambda x: (x['future_close_price'] > x['close_price']).astype(int),
                    trend_5per_up=lambda x: (x['future_close_price'] > x['close_price'] * 1.05).astype(int),
                    price_return=lambda x: x['future_close_price'] / x['close_price']
                )
                [['cik', 'ticker', 'month_end_date', 'trend_up_or_down', 'trend_5per_up', 
                'price_return', 'close_price', 'future_close_price']])
    print(f"📊 Trend records: {len(trend_df)}")

    return trend_df


def filter_by_date_range(df, date_col, start_date='2000-01-01', end_date='2025-07-01'): 
    return df[(df[date_col] >= pd.to_datetime(start_date)) & 
              (df[date_col] <= pd.to_datetime(end_date))]

def filter_by_price_range(df, price_col, min_price, max_price):
    return df[(df[price_col] >= min_price) & (df[price_col] <= max_price)]


def filter_by_date_continuity(df, date_col, stop_date='2025-01-01', gap_in_days=7): 
    """
    Filter records by date continuity. 
    The stock data is supposed to be continuous, with price record for every trading day, 
    and thus we should observe only small gaps (holidays, weekends, etc.). 
    If a ticker has large gaps (exceeding gap_in_days), remove the records. 

    Args:
        df (DataFrame): DataFrame with date column
        date_col (str): Name of the date column
        gap_in_days (int): Maximum gap in days

    Returns:
        df_filtered (DataFrame): Filtered DataFrame
        removed_ticker (list): List of dictionaries with detailed information about removed tickers.
                              Each dictionary contains:
                              - 'ticker': ticker symbol
                              - 'max_gap_days': maximum gap in days
                              - 'gap_start_date': date when the gap starts (YYYY-MM-DD)
                              - 'gap_end_date': date when the gap ends (YYYY-MM-DD)
    """
    if df.empty:
        print("❌ No data provided for filtering")
        return df, []
    
    # Check if required columns exist
    if date_col not in df.columns:
        print(f"❌ Date column '{date_col}' not found in DataFrame")
        return df, []
    
    # Check if ticker column exists (assuming it's named 'ticker')
    if 'ticker' not in df.columns:
        print("❌ Ticker column not found in DataFrame")
        return df, []
    
    print(f"🔍 Filtering by date continuity (max gap: {gap_in_days} days)...")
    
    # Make a copy to avoid modifying original
    df_work = df.copy()
    
    # Ensure date column is datetime
    df_work[date_col] = pd.to_datetime(df_work[date_col])
    
    # Sort by ticker and date
    df_work = df_work.sort_values(['ticker', date_col]).reset_index(drop=True)
    
    # Calculate date differences within each ticker group using shift
    df_work['date_diff'] = df_work.groupby('ticker')[date_col].diff()
    
    # Convert to days (first row of each ticker will be NaN, which is expected)
    df_work['gap_days'] = df_work['date_diff'].dt.days
    
    # Find tickers with gaps exceeding the threshold
    # We need to check if ANY gap in a ticker's data exceeds the threshold
    ticker_max_gaps = df_work.groupby('ticker')['gap_days'].max()
    problematic_tickers = ticker_max_gaps[ticker_max_gaps > gap_in_days].index.tolist()
    
    if problematic_tickers:
        print(f"🗑️  Found {len(problematic_tickers)} tickers with gaps > {gap_in_days} days:")
        
        # Create a DataFrame with problematic tickers and their max gaps using vectorized operations
        problematic_df = pd.DataFrame({
            'ticker': problematic_tickers,
            'max_gap_days': ticker_max_gaps[problematic_tickers].values
        })
        
        # Find records where the max gap occurs for each ticker using vectorized operations
        # First, create a mask for records that have the max gap for their ticker
        max_gap_mask = df_work.groupby('ticker')['gap_days'].transform('max') == df_work['gap_days']
        max_gap_records = df_work[max_gap_mask & df_work['ticker'].isin(problematic_tickers)]
        
        # Merge with problematic_df to get the max gap info
        gap_info = max_gap_records.merge(problematic_df, on='ticker', how='right')
        
        # Calculate gap start dates vectorized
        gap_info['gap_start_date'] = gap_info[date_col] - pd.to_timedelta(gap_info['max_gap_days'], unit='D')
        
        # Create the detailed information list using vectorized operations
        removed_ticker_info = gap_info.apply(lambda row: {
            'ticker': row['ticker'],
            'max_gap_days': int(row['max_gap_days']),
            'gap_start_date': row['gap_start_date'].strftime('%Y-%m-%d'),
            'gap_end_date': row[date_col].strftime('%Y-%m-%d')
        }, axis=1).tolist()

        # Filter out problematic tickers
        df_filtered = df_work[~df_work['ticker'].isin(problematic_tickers)].copy()
        
        # Remove helper columns
        df_filtered = df_filtered.drop(['date_diff', 'gap_days'], axis=1)
        
        removed_count = len(df) - len(df_filtered)
        print(f"📊 Removed {removed_count:,} records from {len(problematic_tickers)} tickers")
        
        return df_filtered, removed_ticker_info
    else:
        print(f"✅ All tickers have gaps ≤ {gap_in_days} days")
        # Remove helper columns
        df_work = df_work.drop(['date_diff', 'gap_days'], axis=1)
        return df_work, []


