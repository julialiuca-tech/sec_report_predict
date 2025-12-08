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
    FEATURE_SUFFIXES,
    INVEST_EXP_END_MONTH_STR,
    INVEST_EXP_START_MONTH_STR,
    MODEL_DIR,
    MONTH_END_PRICE_FILE,
    QUARTER_GRADIENTS,
    Y_LABEL,
    FEATURE_IMPORTANCE_RANKING_FLAG,
)
from utility_data import price_trend



def top_candidate_w_return(df_companies, strategy_param):
    """
    Compute the performance of the model on the top K companies.
    
    Args: 
        df_companies: The dataframe containing the companies, 
           must contain columns: ['ticker', 'price_return', 'y_pred_proba']
        strategy_param (int):  strategy for selecting top candidates
        
    Returns:
        tuple: (top_k_cumulative_return, market_avg_return, top_k_tickers)
            - top_k_cumulative_return: Cumulative average return from top K candidates
            - market_avg_return: Average return of the market (all companies)
            - top_k_tickers: List of top K candidates' tickers
    """
    df_companies.sort_values(by= 'y_pred_proba', ascending=False, inplace=True)

    # Get market average return (all companies), used for benchmark comparison
    market_avg_return = df_companies['price_return'].mean()

    # Calculate cumulative average of invest_val as we slide down the sorted dataframe
    df_companies['cumulative_avg_return'] = df_companies['price_return'].expanding().mean()
    df_companies['cumulative_count'] = range(1, len(df_companies) + 1)
    
    if strategy_param['name'] == 'top_k':
        K = strategy_param['param']
        # Get the cumulative return for top K candidates
        if len(df_companies) >= K:
            top_candidate_cumulative_return = df_companies.iloc[K-1]['cumulative_avg_return']
            top_candidate_tickers = df_companies.iloc[:K]['ticker'].tolist()
        else:
            top_candidate_cumulative_return = df_companies['cumulative_avg_return'].iloc[-1]
            top_candidate_tickers = df_companies['ticker'].tolist()
    elif strategy_param['name'] == 'top_proba': 
        threshold = strategy_param['param']
        top_candidate_companies = df_companies[df_companies['y_pred_proba'] >= threshold]
        if len(top_candidate_companies) == 0:
            top_candidate_cumulative_return = np.nan
            top_candidate_tickers = []
        else:
            top_candidate_cumulative_return = top_candidate_companies['cumulative_avg_return'].iloc[-1]
            top_candidate_tickers = top_candidate_companies['ticker'].tolist()
    else:
        raise ValueError(f"Invalid strategy: {strategy_param['name']}")
 
    return top_candidate_cumulative_return, market_avg_return, top_candidate_tickers



def invest_top_candidate_monthly_retro_performance():
    """
    Test investment strategy using ML model predictions with time-based train/test splits.
    
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
            'result_returns': pd.DataFrame(columns=['year_month', 'top_candidate_return', 'avg_return', 'top_candidate_return_cgt_adjusted']),
            'monthly_invest_record': pd.DataFrame(columns=['year_month', 'rank', 'ticker'])
        }
    
    # Iterate over each month
    for current_month in months_to_test:
        print(f"\n" + "="*60)
        print(f"📅 Testing performance for {current_month}")
        
        # train/test split
        df_test = df[df['year_month'] == current_month].copy()
        df_train = df[df['year_month'] < current_month].copy()
        print(f"📊 Training data: {len(df_train)} samples")
        print(f"📊 Test data: {len(df_test)} samples")
        
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
        df_test['y_pred_proba']= y_pred_proba

        for strategy_name in strategies:
            param = {'name': strategy_name, 'param': strategies[strategy_name]['param']}
            top_candidate_return, avg_return, top_candidate_tickers = top_candidate_w_return(df_test, param)
            if len(top_candidate_tickers) > 0 and not pd.isna(top_candidate_return):
                top_candidate_return_cgt_adjusted = 1+0.8*(top_candidate_return-1) if top_candidate_return > 1 else top_candidate_return
                current_month_returns = pd.DataFrame({
                    'year_month': [current_month],
                    'top_candidate_return': [top_candidate_return],
                    'avg_return': [avg_return],
                    'top_candidate_return_cgt_adjusted': [top_candidate_return_cgt_adjusted]
                })
                 
                monthly_invest_record_rows = pd.DataFrame({
                    'year_month': [current_month] * len(top_candidate_tickers),
                    'rank': range(1, len(top_candidate_tickers) + 1),
                    'ticker': top_candidate_tickers
                })

                # save data to strategy_outcome
                strategy_outcome[strategy_name]['result_returns'] = pd.concat(  
                    [strategy_outcome[strategy_name]['result_returns'], current_month_returns], 
                    ignore_index=True
                )
                strategy_outcome[strategy_name]['monthly_invest_record'] = pd.concat(
                    [strategy_outcome[strategy_name]['monthly_invest_record'], monthly_invest_record_rows], 
                    ignore_index=True
                )
            # print trace 
            return_str = f"{top_candidate_return:.4f}" if not pd.isna(top_candidate_return) else "NaN"
            ticker_str = ','.join(top_candidate_tickers) if len(top_candidate_tickers) > 0 else "None"
            print(f"  📊 Strategy: {strategy_name}, Top candidate return: {return_str}", 
                  f" Average return: {avg_return:.4f}", 
                  f" Top candidate tickers: {len(top_candidate_tickers)}: {ticker_str}")

    # print out the results
    for strategy_name in strategies:
        result_returns = strategy_outcome[strategy_name]['result_returns']
        result_returns.to_csv(os.path.join(MODEL_DIR, "returns_" + strategy_name + ".csv"), index=False)
        # print out the results
        print(f"\n" + "="*60)
        print(f"📊 Strategy: {strategy_name}")
        print(f"Monthly invest record shape: {strategy_outcome[strategy_name]['monthly_invest_record'].shape}")
        strategy_outcome[strategy_name]['monthly_invest_record'].to_csv(
            os.path.join(MODEL_DIR, "invest_record_" + strategy_name + ".csv"), 
            index=False
        )
   
        # though the model is trained on short-term (1-mo or 3-mo) up/down trends, 
        # the model may be useful for picking stocks with long-term potential.  
        # calculate the long-term gain from the investment record for 12 months horizon
        if len(strategy_outcome[strategy_name]['monthly_invest_record']) > 0:
            df_long_term_gain_by_month = long_term_gain_from_invest_record(
                strategy_outcome[strategy_name]['monthly_invest_record'], 
                num_months_horizon=12
            )  

 

def long_term_gain_from_invest_record(df_invest_record, num_months_horizon):
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
    
    # Run price_trend() on df_work with num_months_horizon as parameter
    price_trend_df = price_trend(df_work, num_months_horizon)
    
    # Generate year_month from month_end_date
    price_trend_df['year_month'] = pd.to_datetime(price_trend_df['month_end_date']).dt.to_period('M')
    
    # Filter to only include records where investment was made
    # Join on ticker and year_month matches investment dates
    result_df = price_trend_df.merge(
        df_invest_record,
        on=['ticker', 'year_month'],
        how='inner'
    )
    result_df.sort_values(by=['year_month', 'rank'], ascending=True, inplace=True)
    result_df.to_csv('invest_record_12month_gain.csv', index=False)
    
    # aggregate by year_month to get the average return for each month
    result_df_agg_by_month = result_df.groupby('year_month').agg({'price_return': 'mean'}).reset_index()
    print(result_df_agg_by_month.sort_values(by='year_month', ascending=True))
    print('average return: ', result_df_agg_by_month['price_return'].mean())
    result_df_agg_by_month['price_return_cgt_adjusted'] = result_df_agg_by_month['price_return'].apply(
        lambda x: 1+0.8*(x-1) if x > 1 else x
    )
    print('return cgt adjusted: ', result_df_agg_by_month['price_return_cgt_adjusted'].mean())
    
    return result_df_agg_by_month


if __name__ == "__main__":
    invest_top_candidate_monthly_retro_performance() 