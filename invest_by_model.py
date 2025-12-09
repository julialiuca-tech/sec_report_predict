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
)

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
        threshold = df_sorted.iloc[effective_k - 1]['y_pred_proba']
    elif strategy['method'] == 'top_proba':
        threshold = strategy['param']
    else:
        raise ValueError(f"Invalid strategy: {strategy}")
    
    # Filter companies above threshold
    top_candidates = df_sorted[df_sorted['y_pred_proba'] >= threshold].copy()
    
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
        'proba_0.85': {'method': 'top_proba', 'param': 0.85}
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
                f" {len(invest_record)} selected", 
                f"Short-term return: {invest_record['price_return'].mean():.4f}", 
                f"Long-term return: {invest_record['price_return_long_term'].mean():.4f}"
            ) 
 

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
    


if __name__ == "__main__":
    invest_monthly_retro_performance() 