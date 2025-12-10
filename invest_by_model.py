#!/usr/bin/env python3
"""
Investment and Growth Analysis Functions

This module contains functions for analyzing investment strategies based on ML model predictions
and computing growth/gain metrics.
"""

import os

import numpy as np
import pandas as pd
import xgboost as xgb

from baseline_model import (
    collect_column_names_w_suffix,
    filter_features_by_importance,
    prep_data_feature_label,
)
from config import (
    MODEL_DIR,
    QUARTER_GRADIENTS,
    Y_LABEL,
    FEATURE_IMPORTANCE_RANKING_FLAG,
    STOCK_DIR,
    SAVE_DIR,
    DATA_BASE_DIR,
    DEFAULT_K_TOP_TAGS,
    DEFAULT_MIN_COMPLETENESS,
    DEFAULT_N_QUARTERS_HISTORY_COMP,
)
from featurize import (
    featurize_multi_qtrs,
)
from utility_data import read_tags_to_featurize

# =============================================================================
# INVESTMENT EXPERIMENT CONFIGURATION
# =============================================================================
INVEST_EXP_START_MONTH_STR = '2020-01'
INVEST_EXP_END_MONTH_STR = '2025-04' 
LONG_TERM_TRENDS_FILE = os.path.join(STOCK_DIR, 'price_trends_12month.csv')



def top_candidate_w_return(df_companies, strategy):
    """
    Filter top candidates based on strategy and return DataFrame with selected companies.
    
    Args: 
        df_companies: The dataframe containing the companies, 
           must contain columns: ['year_month', 'cik', 'ticker', 'price_return', 'y_pred_proba']
        strategy (dict): Strategy for selecting top candidates with 'method' and 'param'
        
    Returns:
        pd.DataFrame: DataFrame with columns ['year_month', 'cik', 'ticker', 'y_pred_proba', 'rank', 'price_return']
            Contains only the companies that meet the strategy criteria, sorted by y_pred_proba descending.
            Returns empty DataFrame if no companies meet the criteria.
    """
    if len(df_companies) == 0:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['year_month', 'cik', 'ticker', 'y_pred_proba', 'rank', 'price_return'])
    
    # Assert that all year_month values are the same
    unique_year_months = df_companies['year_month'].unique()
    assert len(unique_year_months) == 1, \
        f"All rows in df_companies must have the same year_month. Found {len(unique_year_months)} different values: {unique_year_months}"
    
    # Sort by prediction probability descending
    df_sorted = df_companies.sort_values(by='y_pred_proba', ascending=False).copy()
    
    # Determine threshold based on strategy
    if strategy['method'] == 'top_k':
        K = strategy['param']
        effective_k = min(K, len(df_sorted))
        top_candidates = df_sorted[:effective_k].copy()
    elif strategy['method'] == 'top_proba':
        threshold = strategy['param']
        top_candidates = df_sorted[df_sorted['y_pred_proba'] >= threshold].copy()
    elif strategy['method'] == 'proba_range': 
        top_candidates = df_sorted[(df_sorted['y_pred_proba'] >= strategy['param'][0])
                                 & (df_sorted['y_pred_proba'] <= strategy['param'][1])
                                 ].copy()
    elif strategy['method'] == 'mixed':
        top_candidates = df_sorted[(df_sorted['y_pred_proba'] >= strategy['param'][0])
                            & (df_sorted['y_pred_proba'] <= strategy['param'][1])
                            ].iloc[:strategy['param'][2]].copy()
    else: 
        raise ValueError(f"Invalid strategy: {strategy}")
    
    # Filter companies above threshold
    
    if len(top_candidates) == 0:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['year_month', 'cik', 'ticker', 'y_pred_proba', 'rank', 'price_return'])
    
    # Add rank (1, 2, 3, ... based on y_pred_proba order, already sorted descending)
    top_candidates['rank'] = range(1, len(top_candidates) + 1)
    
    # Select columns (year_month and cik are always present since df_companies comes from prep_data_feature_label)
    result_df = top_candidates[['year_month', 'cik', 'ticker', 'y_pred_proba', 'rank', 'price_return']].copy()
    
    return result_df



def invest_monthly_retro_performance():
    """
    Test investment strategies using ML model predictions with time-based train/test splits.
    
    Iterates over months from 2024-01 to 2025-04, where each month serves as test data
    and all previous months serve as training data.
    
    Returns:
        pd.DataFrame: Investment record with columns ['year_month', 'rank', 'ticker']
    """   

    strategies = {
        'top_5': {'method': 'top_k', 'param': 5},
        'top_10': {'method': 'top_k', 'param': 10},
        'proba_0.8': {'method': 'top_proba', 'param': 0.8},
        'proba_0.85': {'method': 'top_proba', 'param': 0.85},
        'proba_0.75_to_0.85': {'method': 'proba_range', 'param': [0.75, 0.85]}, 
        'mixed': {'method': 'mixed', 'param': [0.75, 0.85, 10]} 
    } 

    # Define the range of months to test
    start_month = pd.Period(INVEST_EXP_START_MONTH_STR, freq='M')
    end_month = pd.Period(INVEST_EXP_END_MONTH_STR, freq='M')
    
    # Generate list of months to iterate over
    current_month = start_month
    months_to_test = []
    while current_month <= end_month:
        months_to_test.append(current_month)
        current_month += 1
    
    print(f"\n📊 Testing investment strategy with time-based splits...")
    print(f"📅 Testing months: {len(months_to_test)} months from {start_month} to {end_month}")
    
 
    
    # Prepare data (you can change split_strategy to 'date' for time-based splitting)
    df = prep_data_feature_label(quarters_for_gradient_comp=QUARTER_GRADIENTS)
    if FEATURE_IMPORTANCE_RANKING_FLAG:
        feature_importance_ranking = pd.read_csv(os.path.join(MODEL_DIR, 'feature_importance_ranking.csv'))
        df = filter_features_by_importance(df, feature_importance_ranking) 
    
    # Get feature columns from the full dataset
    suffix_cols = collect_column_names_w_suffix(df.columns, feature_suffixes= ['_current'])
    feature_cols = suffix_cols + [f for f in df.columns if '_change' in f and f not in suffix_cols]

    # Initialize strategy outcome tracking
    strategy_outcome = {}
    for strategy_name in strategies:
        strategy_outcome[strategy_name] = {
            'monthly_invest_record': []  # Use list to collect DataFrames, concatenate later
        }
    
    # Iterate over each month
    for current_month in months_to_test:
        print(f"\n" + "="*60)
        print(f"📅 Testing performance for {current_month}")
        
        # train/test split
        df_test = df[df['year_month'] == current_month].copy()
        df_train = df[df['year_month'] < current_month].copy()
        print(f"📊 Training data: {len(df_train)} samples", f"📊 Test data: {len(df_test)} samples")
        
        # Skip if no test data for this month
        if len(df_test) == 0:
            print(f"⚠️  No test data available for {current_month}, skipping...")
            continue
        # Skip if insufficient training data
        if len(df_train) < 100:  # Minimum threshold for training
            print(f"⚠️  Insufficient training data ({len(df_train)} samples) for {current_month}, skipping...")
            continue
        
        # Train model on df_train
        X_train = df_train[feature_cols].copy()
        y_train = df_train[Y_LABEL].copy()
        model = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1, eval_metric='logloss'
            )
        model.fit(X_train, y_train)
        
        # Run top_candidate_w_return on df_test
        X_test = df_test[feature_cols].copy()  
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        df_test['y_pred_proba'] = y_pred_proba
        
        # Get market average return for this month
        avg_return = df_test['price_return'].mean()

        for strategy_name, strategy in strategies.items(): 
            df_top_candidates = top_candidate_w_return(df_test, strategy)
            
            if len(df_top_candidates) > 0:
                strategy_outcome[strategy_name]['monthly_invest_record'].append(df_top_candidates)

                # print out the results for the current month
                top_candidate_return = df_top_candidates['price_return'].mean()
                num_tickers = len(df_top_candidates)
                ticker_str = ','.join(df_top_candidates['ticker'].tolist())
                print(f"  📊 Strategy: {strategy_name}", f"Average return (market): {avg_return:.4f}")
                print(f"     Selected tickers ({num_tickers}): {ticker_str}", 
                      f"return from selected: {top_candidate_return:.4f}")
            else:
                print(f"  📊 Strategy: {strategy_name}, No candidates selected")
    
 
    # Convert lists of DataFrames to single DataFrames
    for strategy_name in strategies:
        monthly_records_list = strategy_outcome[strategy_name]['monthly_invest_record']
        if len(monthly_records_list) > 0:
            strategy_outcome[strategy_name]['monthly_invest_record'] = pd.concat(
                monthly_records_list, ignore_index=True
            )
        else:
            strategy_outcome[strategy_name]['monthly_invest_record'] = pd.DataFrame(
                columns=['year_month', 'cik', 'ticker', 'y_pred_proba', 'rank', 'price_return']
            )
    
    print(f"\n" + "="*30 + "Overall summary" + "="*30)
    # print out the results
    for strategy_name in strategies:
        if len(strategy_outcome[strategy_name]['monthly_invest_record']) == 0:
            print(f"📊 Strategy: {strategy_name}, No investment record")
        else:
            invest_record = strategy_outcome[strategy_name]['monthly_invest_record'].copy()
            invest_record = augment_invest_record_w_long_term_return(invest_record)   
            print(f"📊 Strategy: {strategy_name}", 
                f"\n\t{len(invest_record)} selected, ", 
                f"Short-term return: {invest_record['price_return'].mean():.4f}", 
                f"Long-term return: {invest_record['price_return_long_term'].mean():.4f}"
            ) 
            # print proba stats 
            print(f"     Proba stats: {invest_record['y_pred_proba'].describe()}") 
 

def augment_invest_record_w_long_term_return(df_invest_record):
    """
    Augment investment records with pre-computed long-term returns.
    
    This function joins investment records with pre-computed long-term price trends
    from LONG_TERM_TRENDS_FILE to get long-term returns.
    
    Args:
        df_invest_record (pd.DataFrame): DataFrame with columns ['year_month', 'cik', 'ticker', ...] 
                                       containing investment decisions
    
    Returns:
        pd.DataFrame: Investment records augmented with price_return_long_term column
    """
    # Read pre-computed long-term trends from CSV
    price_trend_df = pd.read_csv(LONG_TERM_TRENDS_FILE)
    price_trend_df['year_month'] = pd.to_datetime(price_trend_df['month_end_date']).dt.to_period('M')
    price_trend_df.rename(columns={'price_return': 'price_return_long_term'}, inplace=True)
    
    # Convert df_invest_record year_month to Period if it's not already
    if df_invest_record['year_month'].dtype == 'object':
        df_invest_record = df_invest_record.copy()
        df_invest_record['year_month'] = pd.to_datetime(df_invest_record['year_month']).dt.to_period('M')
    
    # Join on (cik, year_month)
    result_df = price_trend_df.merge(
        df_invest_record,
        on=['cik', 'year_month'],
        how='inner'
    )

    return result_df 
    

def invest_monthly_w_holdback():
    """
    Test investment strategies using a holdback approach:
    - Train on data up to 2024-06-30 (inclusive)
    - Test on data from 2024-07-01 onwards
    
    Logic:
    1. Featurize data up to 2024-06-30 (inclusive) -> featurized_up_to_2024q2.csv
    2. Featurize data from 2024-07-01 onwards -> featurized_2024q3_onwards.csv
    3. Filter both datasets to keep only features >15% populated
    4. Prepare data for training and testing using prep_data_feature_label()
    5. Train model and generate predictions (similar to invest_monthly_retro_performance lines 168-180)
    6. Generate investment results (similar to invest_monthly_retro_performance lines 182-200)
    7. Join with 6-month price trends from price_trends_6month.csv
    """
    import re
    
    # Holdback date: 2024-06-30 (inclusive)
    holdback_date = pd.Period('2024-06', freq='M')
    
    print(f"\n📊 Investment Strategy with Holdback")
    print(f"📅 Train on data up to: {holdback_date}")
    print(f"📅 Test on data from: {holdback_date + 1}")
    print("="*60)
    
    # Step 1: Find all quarter directories
    quarter_directories = []
    if os.path.exists(DATA_BASE_DIR):
        quarter_pattern = re.compile(r'^[0-9]{4}[qQ][1-4]$')
        for item in os.listdir(DATA_BASE_DIR):
            item_path = os.path.join(DATA_BASE_DIR, item)
            if os.path.isdir(item_path) and quarter_pattern.match(item):
                quarter_directories.append(item_path)
    
    quarter_directories.sort()
    
    # Separate quarters into train (up to 2024q2) and test (2024q3 onwards)
    train_quarter_dirs = []
    test_quarter_dirs = []
    
    for qtr_dir in quarter_directories:
        quarter_name = qtr_dir.split('/')[-1]
        # Parse quarter name (e.g., "2024q2" -> year=2024, quarter=2)
        match = re.match(r'^(\d{4})[qQ](\d)$', quarter_name)
        if match:
            year = int(match.group(1))
            quarter = int(match.group(2))
            # Convert to Period for comparison (quarter end month)
            quarter_end_month = pd.Period(f'{year}-{quarter*3:02d}', freq='M')
            
            if quarter_end_month <= holdback_date:
                train_quarter_dirs.append(qtr_dir)
            else:
                test_quarter_dirs.append(qtr_dir)
    
    print(f"\n📊 Found {len(train_quarter_dirs)} training quarters and {len(test_quarter_dirs)} test quarters")
    
    # Step 2: Get tags to featurize
    df_tags_to_featurize = read_tags_to_featurize(K_top_tags=DEFAULT_K_TOP_TAGS)
    
    # Step 3: Featurize training data (up to 2024-06-30)
    print(f"\n{'='*60}")
    print("Step 1: Featurizing training data (up to 2024-06-30)...")
    train_file = os.path.join(SAVE_DIR, 'featurized_up_to_2024q2.csv')
    if not os.path.exists(train_file):
        df_train_featurized = featurize_multi_qtrs(
            train_quarter_dirs,
            df_tags_to_featurize,
            N_qtrs_history_comp=DEFAULT_N_QUARTERS_HISTORY_COMP,
            save_file_name=train_file
        )
    else:
        df_train_featurized = pd.read_csv(train_file)
        print(f"📊 Training data already featurized: {df_train_featurized.shape}")
    
    # Step 4: Featurize test data (from 2024-07-01 onwards)
    print(f"\n{'='*60}")
    print("Step 2: Featurizing test data (from 2024-07-01 onwards)...")
    test_file = os.path.join(SAVE_DIR, 'featurized_2024q3_onwards.csv')
    if not os.path.exists(test_file):
        df_test_featurized = featurize_multi_qtrs(
            test_quarter_dirs,
            df_tags_to_featurize,
            N_qtrs_history_comp=DEFAULT_N_QUARTERS_HISTORY_COMP,
            save_file_name=test_file
        )
    else:
        df_test_featurized = pd.read_csv(test_file)
        print(f"📊 Test data already featurized: {df_test_featurized.shape}")
        
    # Step 3: Prepare data for training and testing using prep_data_feature_label
    # Skip feature completeness filtering - use entire feature set
    print(f"\n{'='*60}")
    print("Step 3: Preparing data for ML model...")
    df_train = prep_data_feature_label(
        featurized_data_file=train_file,
        quarters_for_gradient_comp=QUARTER_GRADIENTS
    )
    
    df_test = prep_data_feature_label(
        featurized_data_file=test_file,
        quarters_for_gradient_comp=QUARTER_GRADIENTS
    )
    
    # Apply feature importance filtering if enabled
    if FEATURE_IMPORTANCE_RANKING_FLAG:
        feature_importance_ranking = pd.read_csv(os.path.join(MODEL_DIR, 'feature_importance_ranking.csv'))
        df_train = filter_features_by_importance(df_train, feature_importance_ranking)
        df_test = filter_features_by_importance(df_test, feature_importance_ranking)
    
    # Get feature columns from training data (we'll align df_test to match)
    suffix_cols = collect_column_names_w_suffix(df_train.columns, feature_suffixes=['_current'])
    feature_cols = suffix_cols + [f for f in df_train.columns if '_change' in f and f not in suffix_cols]
    
    # Identify columns that are in df_train but missing in df_test
    missing_cols = [col for col in feature_cols if col not in df_test.columns]
    if missing_cols:
        print(f"\n⚠️  Columns in training data but missing in test data ({len(missing_cols)} columns):")
        for col in missing_cols[:20]:  # Print first 20
            print(f"   - {col}")
        if len(missing_cols) > 20:
            print(f"   ... and {len(missing_cols) - 20} more")
        print(f"\n   Adding missing columns to df_test with NaN values...")
    
    # Add missing columns to df_test with NaN values
    for col in missing_cols:
        df_test[col] = np.nan
    
    # Also identify columns that are in df_test but not in df_train (for information only)
    extra_cols = [col for col in df_test.columns if col in feature_cols and col not in df_train.columns]
    if extra_cols:
        print(f"\nℹ️  Columns in test data but not in training data ({len(extra_cols)} columns):")
        for col in extra_cols[:20]:  # Print first 20
            print(f"   - {col}")
        if len(extra_cols) > 20:
            print(f"   ... and {len(extra_cols) - 20} more")
        print(f"   These columns will be ignored during prediction.")
    
    print(f"\n📊 Training data: {len(df_train)} samples, {len(feature_cols)} features")
    print(f"📊 Test data: {len(df_test)} samples, aligned to {len(feature_cols)} features")
    
    # Step 4: Train model (similar to invest_monthly_retro_performance lines 168-175)
    print(f"\n{'='*60}")
    print("Step 4: Training model...")
    X_train = df_train[feature_cols].copy()
    y_train = df_train[Y_LABEL].copy()
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, n_jobs=-1, eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # Step 5: Generate predictions for test data (similar to invest_monthly_retro_performance lines 177-180)
    print(f"\n{'='*60}")
    print("Step 5: Generating predictions...")
    X_test = df_test[feature_cols].copy()
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    df_test['y_pred_proba'] = y_pred_proba
    
    # Step 6: Define strategies and generate investment results (similar to invest_monthly_retro_performance lines 182-200)
    print(f"\n{'='*60}")
    print("Step 6: Generating investment results...")
    strategies = {
        'top_5': {'method': 'top_k', 'param': 5},
        'top_10': {'method': 'top_k', 'param': 10},
        'proba_0.8': {'method': 'top_proba', 'param': 0.8},
        'proba_0.85': {'method': 'top_proba', 'param': 0.85},
        'proba_0.75_to_0.85': {'method': 'proba_range', 'param': [0.75, 0.85]}, 
        'mixed': {'method': 'mixed', 'param': [0.75, 0.85, 10]} 
    } 
    
    strategy_outcome = {}
    for strategy_name, strategy in strategies.items():
        # Group by month and apply strategy
        monthly_records = []
        for year_month in df_test['year_month'].unique():
            df_month = df_test[df_test['year_month'] == year_month].copy()
            df_top_candidates = top_candidate_w_return(df_month, strategy)
            if len(df_top_candidates) > 0:
                monthly_records.append(df_top_candidates)
        
        if len(monthly_records) > 0:
            strategy_outcome[strategy_name] = pd.concat(monthly_records, ignore_index=True)
        else:
            strategy_outcome[strategy_name] = pd.DataFrame(
                columns=['year_month', 'cik', 'ticker', 'y_pred_proba', 'rank', 'price_return']
            )
    
    # Step 7: Augment with 6-month price trends
    print(f"\n{'='*60}")
    print("Step 9: Augmenting with 6-month price trends...")
    price_trends_6month_file = os.path.join(STOCK_DIR, 'price_trends_6month.csv')
    if not os.path.exists(price_trends_6month_file):
        raise FileNotFoundError(f"6-month price trends file not found: {price_trends_6month_file}")
    
    price_trend_df = pd.read_csv(price_trends_6month_file)
    price_trend_df['year_month'] = pd.to_datetime(price_trend_df['month_end_date']).dt.to_period('M')
    price_trend_df.rename(columns={'price_return': 'price_return_6month'}, inplace=True)
    
    # Step 8: Print results
    print(f"\n{'='*60}")
    print("📊 INVESTMENT RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for strategy_name in strategies:
        if len(strategy_outcome[strategy_name]) == 0:
            print(f"📊 Strategy: {strategy_name}, No investment record")
        else:
            invest_record = strategy_outcome[strategy_name].copy()
            
            # Join with 6-month price trends
            invest_record['year_month'] = invest_record['year_month'].astype(str)
            price_trend_df['year_month'] = price_trend_df['year_month'].astype(str)
            invest_record = invest_record.merge(
                price_trend_df[['cik', 'year_month', 'price_return_6month']],
                on=['cik', 'year_month'],
                how='left'
            )
            
            print(f"📊 Strategy: {strategy_name}")
            print(f"   {len(invest_record)} investments selected")
            print(f"   Short-term return: {invest_record['price_return'].mean():.4f}")
            if invest_record['price_return_6month'].notna().any():
                print(f"   6-month return: {invest_record['price_return_6month'].mean():.4f}")
            print(f"   Proba stats: {invest_record['y_pred_proba'].describe()}")

    # benchmark 
    # Ensure year_month types match for merging
    df_test_benchmark = df_test.copy()
    if df_test_benchmark['year_month'].dtype == 'object':
        df_test_benchmark['year_month'] = pd.to_datetime(df_test_benchmark['year_month']).dt.to_period('M')
    
    price_trend_df_benchmark = price_trend_df.copy()
    if price_trend_df_benchmark['year_month'].dtype == 'object':
        price_trend_df_benchmark['year_month'] = pd.to_datetime(price_trend_df_benchmark['year_month']).dt.to_period('M')
    
    # Convert to string for consistent merging
    df_test_benchmark['year_month'] = df_test_benchmark['year_month'].astype(str)
    price_trend_df_benchmark['year_month'] = price_trend_df_benchmark['year_month'].astype(str)
    
    df_benchmark = df_test_benchmark.merge(
                price_trend_df_benchmark[['cik', 'year_month', 'price_return_6month']],
                on=['cik', 'year_month'],
                how='left'
            )
    print(f"   benchmark -- market average short-term return: {df_benchmark['price_return'].mean():.4f}", 
                  f"6-month return: {df_benchmark['price_return_6month'].mean():.4f}")



if __name__ == "__main__":
    # invest_monthly_retro_performance() 
    invest_monthly_w_holdback()