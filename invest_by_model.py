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
    INVEST_EXP_END_MONTH_STR,
    INVEST_EXP_START_MONTH_STR,
    MODEL_DIR,
    MONTH_END_PRICE_FILE,
    QUARTER_GRADIENTS,
    Y_LABEL,
    FEATURE_IMPORTANCE_RANKING_FLAG,
)
from utility_data import price_trend



def top_candidate_w_return(df_companies, strategy_param, year_month):
    """
    Filter top candidates based on strategy and return DataFrame with selected companies.
    
    Args: 
        df_companies: The dataframe containing the companies, 
           must contain columns: ['ticker', 'price_return', 'y_pred_proba']
        strategy_param (dict): Strategy for selecting top candidates with 'name' and 'param'
        year_month: Period or string representing the year-month
        
    Returns:
        pd.DataFrame: DataFrame with columns ['year_month', 'ticker', 'y_pred_proba', 'rank', 'price_return']
            Contains only the companies that meet the strategy criteria, sorted by y_pred_proba descending.
            Returns empty DataFrame if no companies meet the criteria.
    """
    if len(df_companies) == 0:
        return pd.DataFrame(columns=['year_month', 'ticker', 'y_pred_proba', 'rank', 'price_return'])
    
    # Sort by prediction probability descending
    df_sorted = df_companies.sort_values(by='y_pred_proba', ascending=False).copy()
    
    # Determine threshold based on strategy
    if strategy_param['name'] == 'top_k':
        K = strategy_param['param']
        effective_k = min(K, len(df_sorted))
        threshold = df_sorted.iloc[effective_k - 1]['y_pred_proba']
    elif strategy_param['name'] == 'top_proba':
        threshold = strategy_param['param']
    else:
        raise ValueError(f"Invalid strategy: {strategy_param['name']}")
    
    # Filter companies above threshold
    top_candidates = df_sorted[df_sorted['y_pred_proba'] >= threshold].copy()
    
    if len(top_candidates) == 0:
        return pd.DataFrame(columns=['year_month', 'ticker', 'y_pred_proba', 'rank', 'price_return'])
    
    # Add rank (1, 2, 3, ... based on y_pred_proba order, already sorted descending)
    top_candidates['rank'] = range(1, len(top_candidates) + 1)
    top_candidates['year_month'] = year_month
    result_df = top_candidates[['year_month', 'ticker', 'y_pred_proba', 'rank', 'price_return']].copy()
    
    return result_df



def invest_monthly_retro_performance():
    """
    Test investment strategies using ML model predictions with time-based train/test splits.
    
    Iterates over months from 2024-01 to 2025-04, where each month serves as test data
    and all previous months serve as training data.
    
    Returns:
        pd.DataFrame: Investment record with columns ['year_month', 'rank', 'ticker']
    """   

    strategies = {}
    strategies['top_k'] = {'param': 10}
    strategies['top_proba'] = {'param': 0.8} 

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
            'monthly_invest_record': pd.DataFrame(columns=['year_month', 'rank', 'ticker'])
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

        for strategy_name in strategies:
            param = {'name': strategy_name, 'param': strategies[strategy_name]['param']}
            df_top_candidates = top_candidate_w_return(df_test, param, current_month)
            
            if len(df_top_candidates) > 0:
                strategy_outcome[strategy_name]['monthly_invest_record'] = pd.concat(
                    [strategy_outcome[strategy_name]['monthly_invest_record'], df_top_candidates], 
                    ignore_index=True
                )

                # print out the results for the current month
                top_candidate_return = df_top_candidates['price_return'].mean()
                num_tickers = len(df_top_candidates)
                ticker_str = ','.join(df_top_candidates['ticker'].tolist())
                print(f"  📊 Strategy: {strategy_name}", f"Average return (market): {avg_return:.4f}")
                print(f"     Selected tickers ({num_tickers}): {ticker_str}", 
                      f"return from selected: {top_candidate_return:.4f}")
            else:
                print(f"  📊 Strategy: {strategy_name}, No candidates selected")
    
 
    print(f"\n" + "="*30 + "Overall summary" + "="*30)
    # print out the results
    for strategy_name in strategies:
        if len(strategy_outcome[strategy_name]['monthly_invest_record']) == 0:
            print(f"📊 Strategy: {strategy_name}, No investment record")
        else:
            invest_record = strategy_outcome[strategy_name]['monthly_invest_record'].copy()
            invest_record = augment_invest_record_w_long_term_return(invest_record, num_months_horizon=12)   
            print(f"📊 Strategy: {strategy_name}", 
                f"Short-term return: {invest_record['price_return'].mean():.4f}", 
                f"Long-term return: {invest_record['price_return_long_term'].mean():.4f}"
            ) 
 

def augment_invest_record_w_long_term_return(df_invest_record, num_months_horizon):
    """
    Although the model is trained on short-term (1-mo or 3-mo) up/down trends, 
    the model may be useful for picking stocks with long-term potential.
    This function computes the long-term gain from investment records for a specified time horizon.
    
    Args:
        df_invest_record (pd.DataFrame): DataFrame with columns ['year_month', 'ticker'] 
                                       containing investment decisions
        num_months_horizon (int): Number of months to look ahead for return calculation
    
    Returns:
        pd.DataFrame: Result from price_trend() function containing trend analysis
    """
    # Read month_end_price from config file
    df_month_end = pd.read_csv(MONTH_END_PRICE_FILE)
    
    # Convert year_month back to Period objects (CSV loads them as strings)
    df_month_end['year_month'] = pd.to_datetime(df_month_end['year_month']).dt.to_period('M')
    
    # Filter for tickers relevant to our input dataframe
    relevant_tickers = df_invest_record['ticker'].unique()
    df_work = df_month_end[df_month_end['ticker'].isin(relevant_tickers)].copy() 
    price_trend_df = price_trend(df_work, num_months_horizon) 
    price_trend_df['year_month'] = pd.to_datetime(price_trend_df['month_end_date']).dt.to_period('M')
    price_trend_df.rename(columns={'price_return': 'price_return_long_term'}, inplace=True)
    
    # Filter to only include records where investment was made
    # Join on ticker and year_month matches investment dates
    result_df = price_trend_df.merge(
        df_invest_record,
        on=['ticker', 'year_month'],
        how='inner'
    ) 

    return result_df 
    


if __name__ == "__main__":
    invest_monthly_retro_performance() 