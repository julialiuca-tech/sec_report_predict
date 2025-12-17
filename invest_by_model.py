#!/usr/bin/env python3
"""
Investment and Growth Analysis Functions

This module contains functions for analyzing investment strategies based on ML model predictions
and computing growth/gain metrics.
"""

import os
import re

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
    DEFAULT_N_QUARTERS_HISTORY_COMP,
    FEATURIZED_ALL_QUARTERS_FILE,
    STOCK_TREND_DATA_FILE,
)
from utility_data import read_tags_to_featurize, standardize_df_to_reference
from featurize import featurize_multi_qtrs
from utility_binary_classifier import baseline_binary_classifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
# from sklearn.metrics import precision_score, recall_score, roc_curve, auc, accuracy_score, roc_auc_score

# =============================================================================
# INVESTMENT EXPERIMENT CONFIGURATION
# =============================================================================
INVEST_EXP_START_MONTH_STR = '2023-01'
INVEST_EXP_END_MONTH_STR = '2025-07' 
LONG_TERM_TRENDS_FILE = os.path.join(STOCK_DIR, 'price_trends_12month.csv')
MODEL_TYPE = 'rf'
DEBUG_PRINT = True

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
    # Load featurized data and stock trends
    from config import FEATURIZED_ALL_QUARTERS_FILE, STOCK_TREND_DATA_FILE
    df_features = pd.read_csv(FEATURIZED_ALL_QUARTERS_FILE)
    df_trends = pd.read_csv(STOCK_TREND_DATA_FILE)
    df = prep_data_feature_label(df_featurized_data=df_features, 
                                  df_stock_trend=df_trends,
                                  quarters_for_gradient_comp=QUARTER_GRADIENTS)

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
        
        # Train model on df_train
        X_train = df_train[feature_cols].copy()
        y_train = df_train[Y_LABEL].copy()
        X_test = df_test[feature_cols].copy()  
        y_test = df_test[Y_LABEL].copy()
        model_perf_records = baseline_binary_classifier(X_train, X_test, y_train, y_test, 
                                                        model_name=MODEL_TYPE)
 
        print(f"📊 Model performance -- accuracy: {model_perf_records['accuracy']:.4f}", \
            f"precision: {model_perf_records['precision']:.4f}", \
            f"recall: {model_perf_records['recall']:.4f}", \
            f"roc_auc: {model_perf_records['roc_auc']:.4f}") 
        
        # Run top_candidate_w_return on df_test
        df_test['y_pred_proba'] = model_perf_records['y_pred_proba']

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
    
    # benchmark performance: market average short-term return and 6-month return
    # Use the full dataset as benchmark (comparing overall market performance)
    if len(df) > 0:
        print(f"\n{'='*60}")
        print("📊 BENCHMARK: Overall Market Performance")
        print(f"{'='*60}")
        benchmark_performance(df[(df['year_month'] > INVEST_EXP_START_MONTH_STR) 
                               & (df['year_month'] <= INVEST_EXP_END_MONTH_STR)
                               ], num_months=6)
    
def benchmark_performance(df_test, num_months):
    """
    Calculate benchmark performance metrics for the test dataset.
    
    Computes market average returns for both short-term (1-month) and long-term (num_months) horizons.
    This provides a baseline against which investment strategies can be compared.
    
    Args:
        df_test (pd.DataFrame): Test dataset with columns ['year_month', 'cik', 'price_return', ...]
        num_months (int): Number of months to evaluate long-term gain (e.g., 6 for 6-month return)
    
    Returns:
        dict: Dictionary containing benchmark metrics:
            - 'short_term_return': Average 1-month return
            - 'long_term_return': Average num_months-month return (if available)
            - 'num_records': Number of records in df_test
            - 'num_with_long_term': Number of records with long-term return data
    """
    if len(df_test) == 0:
        print("⚠️  Benchmark: No test data available")
        return {
            'short_term_return': np.nan,
            'long_term_return': np.nan,
            'num_records': 0,
            'num_with_long_term': 0
        }
    
    # Calculate short-term (1-month) average return
    short_term_return = df_test['price_return'].mean()
    
    # Load long-term price trends file
    long_term_trends_file = os.path.join(STOCK_DIR, f'price_trends_{num_months}month.csv')
    
    if not os.path.exists(long_term_trends_file):
        print(f"⚠️  Benchmark: Long-term trends file not found: {long_term_trends_file}")
        print(f"   Showing short-term benchmark only")
        print(f"   📊 Benchmark - Market average short-term return: {short_term_return:.4f}")
        return {
            'short_term_return': short_term_return,
            'long_term_return': np.nan,
            'num_records': len(df_test),
            'num_with_long_term': 0
        }
    
    # Load and prepare long-term trends
    price_trend_df = pd.read_csv(long_term_trends_file)
    price_trend_df['year_month'] = pd.to_datetime(price_trend_df['month_end_date']).dt.to_period('M')
    price_trend_df.rename(columns={'price_return': f'price_return_{num_months}month'}, inplace=True)
    
    # Prepare df_test for merging
    df_test_copy = df_test.copy()
    if df_test_copy['year_month'].dtype == 'object':
        df_test_copy['year_month'] = pd.to_datetime(df_test_copy['year_month']).dt.to_period('M')
    
    # Convert both to string for consistent merging
    df_test_copy['year_month'] = df_test_copy['year_month'].astype(str)
    price_trend_df['year_month'] = price_trend_df['year_month'].astype(str)
    
    # Merge with long-term trends
    df_benchmark = df_test_copy.merge(
        price_trend_df[['cik', 'year_month', f'price_return_{num_months}month']],
        on=['cik', 'year_month'],
        how='left'
    )
    
    # Calculate long-term average return (only for records with data)
    long_term_return = df_benchmark[f'price_return_{num_months}month'].mean()
    num_with_long_term = df_benchmark[f'price_return_{num_months}month'].notna().sum()
    
    # Print results
    print(f"\n📊 BENCHMARK PERFORMANCE ({num_months}-month evaluation)")
    print(f"   Market average short-term return: {short_term_return:.4f}")
    if num_with_long_term > 0:
        print(f"   Market average {num_months}-month return: {long_term_return:.4f}")
        print(f"   Records with {num_months}-month data: {num_with_long_term:,} / {len(df_benchmark):,}")
    else:
        print(f"   ⚠️  No {num_months}-month return data available")
    
    return {
        'short_term_return': short_term_return,
        'long_term_return': long_term_return,
        'num_records': len(df_test),
        'num_with_long_term': num_with_long_term
    }

 

 

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
    Test investment strategies using ML model predictions with time-based train/test splits.
    Similar to invest_monthly_retro_performance(), but recomputes train/test datasets every month
    instead of slicing from a single batch.
    
    Iterates over months from INVEST_EXP_START_MONTH_STR to INVEST_EXP_END_MONTH_STR, where each 
    month serves as test data and all previous months serve as training data. For each month,
    we recompute the featurized data from scratch.
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
    
    print(f"\n📊 Testing investment strategy with time-based splits (recomputing data each month)...")
    print(f"📅 Testing months: {len(months_to_test)} months from {start_month} to {end_month}")
    
    # Load stock trends once (used for all months)
    df_stock_trends = pd.read_csv(STOCK_TREND_DATA_FILE)
    
    # Load full featurized data (will be filtered by month in the loop)
    df_featurized_batch = pd.read_csv(FEATURIZED_ALL_QUARTERS_FILE)
    # Convert period to year_month if not already present
    if 'year_month' not in df_featurized_batch.columns:
        df_featurized_batch['year_month'] = pd.to_datetime(df_featurized_batch['period'], format='%Y%m%d').dt.to_period('M')
    else:
        df_featurized_batch['year_month'] = pd.to_datetime(df_featurized_batch['year_month']).dt.to_period('M')
    
    # Initialize strategy outcome tracking
    strategy_outcome = {}
    for strategy_name in strategies:
        strategy_outcome[strategy_name] = {
            'monthly_invest_record': []  # Use list to collect DataFrames, concatenate later
        }
    
    # Track all test data for benchmarking
    all_test_data = []
    
    # Iterate over each month
    for current_month in months_to_test:
        print(f"\n" + "="*60)
        print(f"📅 Testing performance for {current_month}")
        
        # Split featurized data by month
        df_train_featurized_month = df_featurized_batch[df_featurized_batch['year_month'] < current_month].copy()
        df_test_featurized_month = df_featurized_batch[df_featurized_batch['year_month'] == current_month].copy()
        
        # Skip if no test data for this month
        if len(df_test_featurized_month) == 0:
            print(f"⚠️  No test data available for {current_month}, skipping...")
            continue
        
        # Skip if insufficient training data
        if len(df_train_featurized_month) < 100:  # Minimum threshold for training
            print(f"⚠️  Insufficient training data ({len(df_train_featurized_month)} samples) for {current_month}, skipping...")
            continue
        
        # Prepare data with labels for training and testing
        df_train = prep_data_feature_label(
            df_featurized_data=df_train_featurized_month,
            df_stock_trend=df_stock_trends,
            quarters_for_gradient_comp=QUARTER_GRADIENTS
        )
        
        df_test = prep_data_feature_label(
            df_featurized_data=df_test_featurized_month,
            df_history_data=df_train_featurized_month,
            df_stock_trend=df_stock_trends,
            quarters_for_gradient_comp=QUARTER_GRADIENTS
        )
        
        # Apply feature importance filtering if enabled
        if FEATURE_IMPORTANCE_RANKING_FLAG:
            feature_importance_ranking = pd.read_csv(os.path.join(MODEL_DIR, 'feature_importance_ranking.csv'))
            df_train = filter_features_by_importance(df_train, feature_importance_ranking)
            df_test = filter_features_by_importance(df_test, feature_importance_ranking)
        
        # Get feature columns from training data
        suffix_cols = collect_column_names_w_suffix(df_train.columns, feature_suffixes=['_current'])
        feature_cols = suffix_cols + [f for f in df_train.columns if '_change' in f and f not in suffix_cols]
        
        train_min_month = df_train['year_month'].min()
        train_max_month = df_train['year_month'].max()
        test_min_month = df_test['year_month'].min()
        test_max_month = df_test['year_month'].max()
        print(f"📊 Training data: {len(df_train)} samples, {len(feature_cols)} features, year_month range: {train_min_month} to {train_max_month}")
        print(f"📊 Test data: {len(df_test)} samples, year_month range: {test_min_month} to {test_max_month}")
        
        # Train model on df_train
        X_train = df_train[feature_cols].copy()
        y_train = df_train[Y_LABEL].copy()
        X_test = df_test[feature_cols].copy()
        y_test = df_test[Y_LABEL].copy()
        model_perf_records = baseline_binary_classifier(X_train, X_test, y_train, y_test, 
                                                        model_name=MODEL_TYPE)
        print(f"📊 Model performance -- accuracy: {model_perf_records['accuracy']:.4f}", \
            f"precision: {model_perf_records['precision']:.4f}", \
            f"recall: {model_perf_records['recall']:.4f}", \
            f"roc_auc: {model_perf_records['roc_auc']:.4f}") 
        
        # Run top_candidate_w_return on df_test
        df_test['y_pred_proba'] = model_perf_records['y_pred_proba']
        
        # Get market average return for this month
        avg_return = df_test['price_return'].mean()
        
        # Track test data for benchmarking
        all_test_data.append(df_test.copy())
        
        for strategy_name, strategy in strategies.items(): 
            df_top_candidates = top_candidate_w_return(df_test, strategy)
            
            if len(df_top_candidates) > 0:
                strategy_outcome[strategy_name]['monthly_invest_record'].append(df_top_candidates)
                
                # Print out the results for the current month
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
    # Print out the results
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
            # Print proba stats 
            print(f"     Proba stats: {invest_record['y_pred_proba'].describe()}") 
    
    # Benchmark performance: market average short-term return and 6-month return
    if len(all_test_data) > 0:
        df_all_test = pd.concat(all_test_data, ignore_index=True)
        print(f"\n{'='*60}")
        print("📊 BENCHMARK: Overall Market Performance")
        print(f"{'='*60}")
        benchmark_performance(df_all_test, num_months=6)



if __name__ == "__main__":
    # invest_monthly_retro_performance() 
    invest_monthly_w_holdback()