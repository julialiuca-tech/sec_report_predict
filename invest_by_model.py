#!/usr/bin/env python3
"""
Investment and Growth Analysis Functions

This module contains functions for analyzing investment strategies based on ML model predictions
and computing growth/gain metrics.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

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
    FEATURIZED_ALL_QUARTERS_FILE,
    STOCK_TREND_DATA_FILE,
)
from utility_data import get_cik_ticker_mapping
from utility_binary_classifier import baseline_binary_classifier

# =============================================================================
# INVESTMENT EXPERIMENT CONFIGURATION
# =============================================================================
INVEST_EXP_START_MONTH_STR = '2023-01'
INVEST_EXP_END_MONTH_STR = '2025-07' 
LONG_TERM_TRENDS_FILE = os.path.join(STOCK_DIR, 'price_trends_12month.csv')
MODEL_TYPE = 'rf'
STRATEGIES = {
    'top_5': {'method': 'top_k', 'param': 5},
    'top_10': {'method': 'top_k', 'param': 10},
    'proba_0.8': {'method': 'top_proba', 'param': 0.8},
    'proba_0.85': {'method': 'top_proba', 'param': 0.85},
    'proba_0.75_to_0.85': {'method': 'proba_range', 'param': [0.75, 0.85]}, 
    'mixed': {'method': 'mixed', 'param': [0.75, 0.85, 10]} 
}

# Investment simulation constants
MONTHLY_INVESTMENT = 10000  # $10,000 per month
DEFAULT_BENCHMARK_TICKER = 'VOO'  # Default benchmark ticker


# =============================================================================
# PORTFOLIO SIMULATION HELPERS
# =============================================================================

def get_month_end_price_for_ticker(prices_df: pd.DataFrame, ticker: str, year_month: pd.Period) -> float:
    """
    Get month-end price for a specific ticker and month.
    
    Args:
        prices_df: DataFrame with price data (columns: ticker, year_month, close_price)
        ticker: Ticker symbol
        year_month: Period object representing the month
    
    Returns:
        Price as float, or None if not found
    """
    year_month_str = str(year_month)
    mask = (prices_df['ticker'] == ticker) & (prices_df['year_month'].astype(str) == year_month_str)
    matches = prices_df[mask]
    
    if len(matches) > 0:
        return float(matches['close_price'].iloc[0])
    else:
        return None


def invest_simple(
    prediction_scores: pd.DataFrame,
    stock_prices: pd.DataFrame,
    monthly_investment: float = MONTHLY_INVESTMENT,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Simple buy-and-hold investment strategy.
    
    For each month:
    - Select top K stocks from current month's prediction scores based on y_pred_proba
    - Invest monthly_investment divided evenly among top K stocks
    - Hold all stocks (never sell)
    - Track cumulative portfolio value over time
    
    Args:
        prediction_scores: DataFrame with prediction scores (columns: year_month, cik, ticker, y_pred_proba)
        stock_prices: DataFrame with historical stock prices
        monthly_investment: Amount to invest each month (default: $10,000)
        top_k: Number of top stocks to select each month (default: 10)
    
    Returns:
        DataFrame with columns: ['year_month', 'portfolio_value', 'total_invested', 'cumulative_return', 'num_stocks']
    """
    print(f"\n📈 Simulating simple buy-and-hold strategy...")
    print(f"   Parameters: top_k={top_k}, monthly_investment=${monthly_investment:,.0f}")
    
    # Portfolio holdings: {ticker: shares_owned}
    portfolio = {}
    
    # Track portfolio values over time
    portfolio_history = []
    
    # Get unique months sorted
    unique_months = sorted(prediction_scores['year_month'].unique())
    
    total_invested = 0.0
    
    for month in unique_months:
        # Get current month's prediction scores
        month_scores = prediction_scores[prediction_scores['year_month'] == month].copy()
        
        if len(month_scores) == 0:
            print(f"  ⚠️  No prediction scores for {month}, skipping...")
            continue
        
        # Select top K stocks for this month
        top_k_scores = month_scores.nlargest(top_k, 'y_pred_proba')
        
        # Filter to only stocks with available prices
        investable_tickers = []
        for _, row in top_k_scores.iterrows():
            ticker = row['ticker']
            price = get_month_end_price_for_ticker(stock_prices, ticker, month)
            if price is not None and price > 0:
                investable_tickers.append(ticker)
            else:
                print(f"  ⚠️  No price available for {ticker} in {month}, skipping purchase")
        
        if len(investable_tickers) == 0:
            print(f"  ⚠️  No investable stocks for {month}, skipping...")
            # Still calculate portfolio value with existing holdings
        else:
            # Invest monthly_investment divided evenly among investable stocks
            investment_per_stock = monthly_investment / len(investable_tickers)
            
            for ticker in investable_tickers:
                price = get_month_end_price_for_ticker(stock_prices, ticker, month)
                shares_to_buy = investment_per_stock / price
                
                # Add to portfolio (accumulate shares if already owned)
                if ticker in portfolio:
                    portfolio[ticker] += shares_to_buy
                else:
                    portfolio[ticker] = shares_to_buy
            
            total_invested += monthly_investment
        
        # Calculate current portfolio value (at month-end prices)
        portfolio_value = 0.0
        active_stocks = 0
        for ticker, shares in portfolio.items():
            price = get_month_end_price_for_ticker(stock_prices, ticker, month)
            if price is not None:
                portfolio_value += shares * price
                active_stocks += 1
        
        portfolio_history.append({
            'year_month': month,
            'portfolio_value': portfolio_value,
            'total_invested': total_invested,
            'cumulative_return': (portfolio_value / total_invested - 1) * 100 if total_invested > 0 else 0,
            'num_stocks': active_stocks
        })
        
        if len(portfolio_history) % 10 == 0:
            print(f"   Processed {len(portfolio_history)} months... (portfolio: {active_stocks} stocks, value: ${portfolio_value:,.2f})")
    
    result_df = pd.DataFrame(portfolio_history)
    print(f"   ✅ Portfolio simulation complete: {len(result_df)} months tracked")
    print(f"   Final portfolio value: ${result_df['portfolio_value'].iloc[-1]:,.2f}")
    print(f"   Total invested: ${result_df['total_invested'].iloc[-1]:,.2f}")
    print(f"   Cumulative return: {result_df['cumulative_return'].iloc[-1]:.2f}%")
    print(f"   Final number of stocks in portfolio: {result_df['num_stocks'].iloc[-1]}")
    
    return result_df


def simulate_benchmark_portfolio(
    benchmark_ticker: str,
    start_month: pd.Period,
    end_month: pd.Period,
    monthly_investment: float = MONTHLY_INVESTMENT
) -> pd.DataFrame:
    """
    Simulate benchmark ticker portfolio growth using dollar-cost averaging.
    
    Args:
        benchmark_ticker: Ticker symbol for the benchmark (e.g., 'VOO', 'SPY')
        start_month: Starting month (Period)
        end_month: Ending month (Period)
        monthly_investment: Amount to invest each month (default: $10,000)
    
    Returns:
        DataFrame with columns: ['year_month', 'portfolio_value', 'total_invested', 'cumulative_return']
    """
    print(f"\n📈 Simulating {benchmark_ticker} benchmark portfolio...")
    
    # Generate list of months
    current_month = start_month
    months = []
    while current_month <= end_month:
        months.append(current_month)
        current_month += 1
    
    start_date_str = str(start_month.start_time.date())
    end_date_str = str(end_month.end_time.date())
    
    # Load benchmark prices
    try:
        ticker_obj = yf.Ticker(benchmark_ticker)
        hist = ticker_obj.history(start=start_date_str, end=end_date_str)
        
        if hist.empty:
            print(f"   ❌ Error: No price data available for {benchmark_ticker}")
            return pd.DataFrame()
        
        # Get month-end prices
        hist_month_end = hist.resample('ME').last()
        benchmark_prices_df = pd.DataFrame({
            'date': hist_month_end.index,
            'close_price': hist_month_end['Close'].values
        })
        benchmark_prices_df['date'] = pd.to_datetime(benchmark_prices_df['date'])
        if benchmark_prices_df['date'].dt.tz is not None:
            benchmark_prices_df['date'] = benchmark_prices_df['date'].dt.tz_localize(None)
        benchmark_prices_df['year_month'] = benchmark_prices_df['date'].dt.to_period('M')
        benchmark_prices = dict(zip(benchmark_prices_df['year_month'], benchmark_prices_df['close_price']))
        
    except Exception as e:
        print(f"   ❌ Error loading {benchmark_ticker}: {e}")
        return pd.DataFrame()
    
    # Simulate portfolio
    shares_owned = 0.0
    portfolio_history = []
    total_invested = 0.0
    
    for month in months:
        price = benchmark_prices.get(month, None)
        
        if price is not None and price > 0:
            # Buy shares with monthly investment
            shares_to_buy = monthly_investment / price
            shares_owned += shares_to_buy
            total_invested += monthly_investment
        
        # Calculate portfolio value
        if price is not None:
            portfolio_value = shares_owned * price
        else:
            # Use last known price if current month price unavailable
            portfolio_value = portfolio_history[-1]['portfolio_value'] if portfolio_history else 0.0
        
        portfolio_history.append({
            'year_month': month,
            'portfolio_value': portfolio_value,
            'total_invested': total_invested,
            'cumulative_return': (portfolio_value / total_invested - 1) * 100 if total_invested > 0 else 0
        })
    
    result_df = pd.DataFrame(portfolio_history)
    print(f"   ✅ Benchmark simulation complete: {len(result_df)} months tracked")
    print(f"   Final portfolio value: ${result_df['portfolio_value'].iloc[-1]:,.2f}")
    print(f"   Total invested: ${result_df['total_invested'].iloc[-1]:,.2f}")
    print(f"   Cumulative return: {result_df['cumulative_return'].iloc[-1]:.2f}%")
    
    return result_df


def plot_portfolio_comparison(
    strategy_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    strategy_name: str = "Strategy",
    benchmark_ticker: str = "VOO",
    output_file: str = None
):
    """
    Plot portfolio value and cumulative return comparison between strategy and benchmark.
    
    Args:
        strategy_df: DataFrame with strategy portfolio history (columns: year_month, portfolio_value, cumulative_return)
        benchmark_df: DataFrame with benchmark portfolio history (columns: year_month, portfolio_value, cumulative_return)
        strategy_name: Name of the strategy for the plot title
        benchmark_ticker: Ticker symbol for the benchmark
        output_file: Path to save the plot (optional)
    """
    # Convert year_month to datetime for plotting
    strategy_df = strategy_df.copy()
    benchmark_df = benchmark_df.copy()
    
    strategy_df['date'] = strategy_df['year_month'].dt.to_timestamp('M')
    benchmark_df['date'] = benchmark_df['year_month'].dt.to_timestamp('M')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Portfolio Values
    ax1.plot(strategy_df['date'], strategy_df['portfolio_value'], 
             label=f'{strategy_name}', linewidth=2, marker='o', markersize=4)
    ax1.plot(benchmark_df['date'], benchmark_df['portfolio_value'], 
             label=f'{benchmark_ticker}', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.set_title(f'Portfolio Value Comparison: {strategy_name} vs {benchmark_ticker}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')
    
    # Plot 2: Cumulative Returns
    ax2.plot(strategy_df['date'], strategy_df['cumulative_return'], 
             label=f'{strategy_name}', linewidth=2, marker='o', markersize=4)
    ax2.plot(benchmark_df['date'], benchmark_df['cumulative_return'], 
             label=f'{benchmark_ticker}', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax2.set_title(f'Cumulative Return Comparison: {strategy_name} vs {benchmark_ticker}', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   📊 Plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def invest_by_score_and_strategy(df_test, strategies, strategy_outcome, verbose=True):
    """
    Apply investment strategies to test data based on prediction scores.
    
    This function takes a test DataFrame with prediction scores and applies multiple
    investment strategies, updating the strategy_outcome dictionary with results.
    
    Args:
        df_test (pd.DataFrame): Test DataFrame with columns including:
            - 'year_month', 'cik', 'ticker', 'price_return', 'y_pred_proba'
        strategies (dict): Dictionary of strategies, e.g.:
            {'top_5': {'method': 'top_k', 'param': 5}, ...}
        strategy_outcome (dict): Dictionary to update with results (modified in-place).
            Should have structure: {strategy_name: {'monthly_invest_record': []}}
        verbose (bool): If True, print strategy results for current month
    
    Returns:
        float: Market average return for this month (for reference)
    """
    if len(df_test) == 0:
        if verbose:
            print("  ⚠️  No test data available for strategy evaluation")
        return np.nan
    
    # Get market average return for this month
    avg_return = df_test['price_return'].mean()
    
    # Apply each strategy
    for strategy_name, strategy in strategies.items():
        df_top_candidates = top_candidate_w_return(df_test, strategy)
        
        if len(df_top_candidates) > 0:
            strategy_outcome[strategy_name]['monthly_invest_record'].append(df_top_candidates)
            
            if verbose:
                # Print out the results for the current month
                top_candidate_return = df_top_candidates['price_return'].mean()
                num_tickers = len(df_top_candidates)
                ticker_str = ','.join(df_top_candidates['ticker'].tolist())
                print(f"  📊 Strategy: {strategy_name}", f"Average return (market): {avg_return:.4f}")
                print(f"     Selected tickers ({num_tickers}): {ticker_str}", 
                      f"return from selected: {top_candidate_return:.4f}")
        else:
            if verbose:
                print(f"  📊 Strategy: {strategy_name}, No candidates selected")
    
    return avg_return


def finalize_strategy_outcomes(strategy_outcome, strategies, file_prefix_suffix='', save_to_csv=True):
    """
    Finalize strategy outcomes by concatenating monthly records, printing summary, and saving to CSV.
    
    This function handles the post-processing of strategy outcomes after all months have been processed:
    - Converts lists of DataFrames to single DataFrames
    - Augments with long-term returns
    - Prints summary statistics
    - Optionally saves to CSV files
    
    Args:
        strategy_outcome (dict): Dictionary with structure: {strategy_name: {'monthly_invest_record': []}}
        strategies (dict): Dictionary of strategies (used for iteration)
        file_prefix_suffix (str): Optional suffix to add to filename (e.g., '_holdback')
        save_to_csv (bool): If True, save investment records to CSV files
    """
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
    
    # Print summary
    print(f"\n" + "="*30 + "Overall summary" + "="*30)
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
    
    # Save investment records to CSV files
    if save_to_csv:
        print(f"\n{'='*60}")
        print("💾 Saving investment records to CSV files...")
        print(f"{'='*60}")
        
        # Create output directory for investment records
        invest_records_dir = os.path.join(MODEL_DIR, 'investment_records')
        os.makedirs(invest_records_dir, exist_ok=True)
        
        # Generate filename with date range
        start_str = INVEST_EXP_START_MONTH_STR.replace('-', '')
        end_str = INVEST_EXP_END_MONTH_STR.replace('-', '')
        file_prefix = f"invest_records_{start_str}_to_{end_str}{file_prefix_suffix}"
        
        # Save individual strategy files
        all_records_list = []
        for strategy_name in strategies:
            if len(strategy_outcome[strategy_name]['monthly_invest_record']) > 0:
                invest_record = strategy_outcome[strategy_name]['monthly_invest_record'].copy()
                
                # Drop price_return columns before saving
                columns_to_drop = ['price_return', 'price_return_long_term']
                columns_to_drop = [col for col in columns_to_drop if col in invest_record.columns]
                if columns_to_drop:
                    invest_record = invest_record.drop(columns=columns_to_drop)
                
                # Save individual strategy file
                strategy_file = os.path.join(invest_records_dir, f"{file_prefix}_{strategy_name}.csv")
                invest_record.to_csv(strategy_file, index=False)
                print(f"   ✅ Saved {strategy_name}: {len(invest_record)} records -> {strategy_file}")
                
                # Add strategy name column for combined file
                invest_record['strategy_name'] = strategy_name
                all_records_list.append(invest_record)
        
        # Save combined file with all strategies
        if len(all_records_list) > 0:
            all_records_df = pd.concat(all_records_list, ignore_index=True)
            combined_file = os.path.join(invest_records_dir, f"{file_prefix}_all_strategies.csv")
            all_records_df.to_csv(combined_file, index=False)
            print(f"   ✅ Saved combined file: {len(all_records_df)} records -> {combined_file}")


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
    strategies = STRATEGIES.copy()

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
    
    # Initialize list to accumulate scores across all months
    df_score_list = []
    
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
        
        # Add prediction scores to df_test
        df_test['y_pred_proba'] = model_perf_records['y_pred_proba']

        # Create df_score DataFrame with (year_month, cik, period, y_pred_proba)
        df_score_month = df_test[['year_month', 'cik', 'period', 'y_pred_proba']].copy()
        df_score_list.append(df_score_month)

        # Apply investment strategies
        invest_by_score_and_strategy(df_test, strategies, strategy_outcome, verbose=True)
    
    # Concatenate all scores and save to CSV
    if len(df_score_list) > 0:
        df_score = pd.concat(df_score_list, ignore_index=True)
        
        # Save df_score to CSV
        invest_records_dir = os.path.join(MODEL_DIR, 'investment_records')
        os.makedirs(invest_records_dir, exist_ok=True)
        start_str = INVEST_EXP_START_MONTH_STR.replace('-', '')
        end_str = INVEST_EXP_END_MONTH_STR.replace('-', '')
        score_file = os.path.join(invest_records_dir, f"prediction_scores_{start_str}_to_{end_str}.csv")
        df_score.to_csv(score_file, index=False)
        print(f"\n💾 Saved prediction scores: {len(df_score)} records -> {score_file}")
    
    # Finalize strategy outcomes (concatenate, print summary, save to CSV)
    finalize_strategy_outcomes(strategy_outcome, strategies, file_prefix_suffix='', save_to_csv=True)
    
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
    
    strategies = STRATEGIES.copy()
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
    
    # Initialize list to accumulate scores across all months
    df_score_list = []
    
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
        
        # Add prediction scores to df_test
        df_test['y_pred_proba'] = model_perf_records['y_pred_proba']
        
        # Create df_score DataFrame with (year_month, cik, period, y_pred_proba)
        df_score_month = df_test[['year_month', 'cik', 'period', 'y_pred_proba']].copy()
        df_score_list.append(df_score_month)
        
        # Track test data for benchmarking
        all_test_data.append(df_test.copy())
        
        # Apply investment strategies
        invest_by_score_and_strategy(df_test, strategies, strategy_outcome, verbose=True)
    
    # Concatenate all scores and save to CSV
    if len(df_score_list) > 0:
        df_score = pd.concat(df_score_list, ignore_index=True)
        
        # Save df_score to CSV
        invest_records_dir = os.path.join(MODEL_DIR, 'investment_records')
        os.makedirs(invest_records_dir, exist_ok=True)
        start_str = INVEST_EXP_START_MONTH_STR.replace('-', '')
        end_str = INVEST_EXP_END_MONTH_STR.replace('-', '')
        score_file = os.path.join(invest_records_dir, f"prediction_scores_{start_str}_to_{end_str}_holdback.csv")
        df_score.to_csv(score_file, index=False)
        print(f"\n💾 Saved prediction scores: {len(df_score)} records -> {score_file}")
    
    # Finalize strategy outcomes (concatenate, print summary, save to CSV)
    finalize_strategy_outcomes(strategy_outcome, strategies, file_prefix_suffix='_holdback', save_to_csv=True)
    
    # Benchmark performance: market average short-term return and 6-month return
    if len(all_test_data) > 0:
        df_all_test = pd.concat(all_test_data, ignore_index=True)
        print(f"\n{'='*60}")
        print("📊 BENCHMARK: Overall Market Performance")
        print(f"{'='*60}")
        benchmark_performance(df_all_test, num_months=6)



def load_prediction_scores_and_mark_tickers(prediction_scores_file: str) -> pd.DataFrame:
    """
    Load prediction scores from CSV and map CIK to ticker if needed.
    
    Args:
        prediction_scores_file: Path to prediction scores CSV file
    
    Returns:
        DataFrame with prediction scores including 'ticker' column
    """
    prediction_scores = pd.read_csv(prediction_scores_file)
    
    # Ensure year_month is in Period format
    if prediction_scores['year_month'].dtype == 'object':
        dates = pd.to_datetime(prediction_scores['year_month'])
        if dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)
        prediction_scores['year_month'] = dates.dt.to_period('M')
    else:
        prediction_scores['year_month'] = pd.to_period(prediction_scores['year_month'])
    
    # Check if 'ticker' column exists, if not, map from CIK
    if 'ticker' not in prediction_scores.columns:
        if 'cik' not in prediction_scores.columns:
            raise ValueError("Prediction scores file must contain either 'ticker' or 'cik' column")
        
        print("   ℹ️  'ticker' column not found, mapping from CIK...")
        cik_to_ticker, _ = get_cik_ticker_mapping()
        
        if len(cik_to_ticker) == 0:
            raise ValueError("CIK to ticker mapping is empty")
        
        # Convert CIK to string format matching the mapping (zero-padded to 10 digits)
        prediction_scores['cik_str'] = prediction_scores['cik'].astype(str).str.zfill(10)
        prediction_scores['ticker'] = prediction_scores['cik_str'].map(cik_to_ticker)
        
        # Check how many successfully mapped
        mapped_count = prediction_scores['ticker'].notna().sum()
        total_count = len(prediction_scores)
        
        print(f"   ✅ Mapped {mapped_count}/{total_count} records to tickers")
        
        if mapped_count == 0:
            raise ValueError("Could not map any CIKs to tickers")
        
        # Drop rows where ticker mapping failed
        if mapped_count < total_count:
            print(f"   ⚠️  Dropping {total_count - mapped_count} records without ticker mapping")
            prediction_scores = prediction_scores[prediction_scores['ticker'].notna()].copy()
        
        # Drop temporary cik_str column
        if 'cik_str' in prediction_scores.columns:
            prediction_scores = prediction_scores.drop(columns=['cik_str'])
    
    return prediction_scores


def load_stock_prices_for_tickers(unique_tickers: list, start_date_str: str, end_date_str: str) -> pd.DataFrame:
    """
    Load stock prices for a list of tickers using yfinance.
    
    Args:
        unique_tickers: List of ticker symbols
        start_date_str: Start date in 'YYYY-MM-DD' format
        end_date_str: End date in 'YYYY-MM-DD' format
    
    Returns:
        DataFrame with columns: ['ticker', 'year_month', 'close_price']
    """
    print(f"\n📊 Loading stock prices for {len(unique_tickers)} tickers...")
    print("   (This may take a few minutes...)")
    
    all_prices = []
    for idx, ticker in enumerate(unique_tickers, 1):
        try:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(start=start_date_str, end=end_date_str)
            
            if not hist.empty:
                # Get month-end prices
                hist_month_end = hist.resample('ME').last()
                df = pd.DataFrame({
                    'date': hist_month_end.index,
                    'ticker': ticker,
                    'close_price': hist_month_end['Close'].values
                })
                # Convert date to year_month Period
                df['date'] = pd.to_datetime(df['date'])
                if df['date'].dt.tz is not None:
                    df['date'] = df['date'].dt.tz_localize(None)
                df['year_month'] = df['date'].dt.to_period('M')
                all_prices.append(df[['ticker', 'year_month', 'close_price']])
            
            if idx % 50 == 0:
                print(f"   Processed {idx}/{len(unique_tickers)} tickers...")
        except Exception as e:
            print(f"   ⚠️  Error loading {ticker}: {e}")
            continue
    
    if len(all_prices) == 0:
        raise ValueError("No stock prices loaded")
    
    stock_prices = pd.concat(all_prices, ignore_index=True)
    print(f"   ✅ Loaded prices for {stock_prices['ticker'].nunique()} tickers")
    print(f"   Total price records: {len(stock_prices):,}")
    
    return stock_prices


def eval_strategy_against_index(
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
    top_k: int = 10,
    monthly_investment: float = MONTHLY_INVESTMENT,
    skip_strategy: bool = False,
    prediction_scores_file: str = None
) -> int:
    """
    Evaluate investment strategy against a benchmark index.
    
    This function:
    1. Optionally runs invest_monthly_retro_performance() to generate prediction scores
    2. Loads prediction scores (either newly generated or from existing file)
    3. Runs a simple buy-and-hold investment strategy
    4. Simulates benchmark portfolio performance
    5. Compares strategy vs benchmark and saves results
    
    Args:
        benchmark_ticker: Benchmark ticker to compare against (default: VOO, e.g., SPY, QQQ)
        top_k: Number of top stocks for buy-and-hold strategy (default: 10)
        monthly_investment: Amount to invest each month (default: $10,000)
        skip_strategy: If True, skip running invest_monthly_retro_performance, use existing prediction scores
        prediction_scores_file: Path to prediction scores CSV file (if skip_strategy is True)
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    # Run investment strategy to generate prediction scores if requested
    if not skip_strategy:
        print("="*70)
        print("STEP 1: Running investment strategy to generate prediction scores")
        print("="*70)
        invest_monthly_retro_performance()
    
    # Determine prediction scores file path
    if prediction_scores_file is None:
        # Default to the file that would have been generated
        start_str = INVEST_EXP_START_MONTH_STR.replace('-', '')
        end_str = INVEST_EXP_END_MONTH_STR.replace('-', '')
        prediction_scores_file = os.path.join(MODEL_DIR, 'investment_records', 
                                             f"prediction_scores_{start_str}_to_{end_str}.csv")
    
    # Check if file exists when skip_strategy is True
    if skip_strategy and not os.path.exists(prediction_scores_file):
        print(f"❌ Error: Prediction scores file not found: {prediction_scores_file}")
        print(f"   Please provide a valid path using prediction_scores_file parameter")
        return 1
    
    # Compare against benchmark using simple buy-and-hold strategy
    print("\n" + "="*70)
    print("STEP 2: Comparing investment strategy against benchmark")
    print("="*70)
    
    # Load prediction scores
    print(f"\n📂 Loading prediction scores from: {prediction_scores_file}")
    try:
        prediction_scores = load_prediction_scores_and_mark_tickers(prediction_scores_file)
    except Exception as e:
        print(f"❌ Error loading prediction scores: {e}")
        return 1
    
    print(f"   ✅ Loaded {len(prediction_scores)} prediction scores")
    print(f"   Date range: {prediction_scores['year_month'].min()} to {prediction_scores['year_month'].max()}")
    print(f"   Unique companies: {prediction_scores['ticker'].nunique()}")
    
    # Get date range
    start_month = prediction_scores['year_month'].min()
    end_month = prediction_scores['year_month'].max()
    start_date_str = str(start_month.start_time.date())
    end_date_str = str(end_month.end_time.date())
    
    # Get unique tickers that will be used (top K for at least one month)
    print(f"\n📊 Identifying tickers that will be used (top {top_k} in at least one month)...")
    unique_months = sorted(prediction_scores['year_month'].unique())
    tickers_to_use = set()
    
    for month in unique_months:
        month_scores = prediction_scores[prediction_scores['year_month'] == month].copy()
        if len(month_scores) > 0:
            top_k_scores = month_scores.nlargest(top_k, 'y_pred_proba')
            tickers_to_use.update(top_k_scores['ticker'].dropna().tolist())
    
    unique_tickers = sorted([t for t in tickers_to_use if pd.notna(t)])
    print(f"   Unique tickers in prediction scores: {prediction_scores['ticker'].nunique()}")
    print(f"   Tickers that will be used: {len(unique_tickers)}")
    
    # Load stock prices
    try:
        stock_prices = load_stock_prices_for_tickers(unique_tickers, start_date_str, end_date_str)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    
    # Run investment simulation
    strategy_name = f"Top_{top_k}_BuyHold"
    result_df = invest_simple(
        prediction_scores=prediction_scores,
        stock_prices=stock_prices,
        monthly_investment=monthly_investment,
        top_k=top_k
    )
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS - STRATEGY")
    print(f"{'='*70}")
    print(f"Total invested: ${result_df['total_invested'].iloc[-1]:,.2f}")
    print(f"Final portfolio value: ${result_df['portfolio_value'].iloc[-1]:,.2f}")
    print(f"Cumulative return: {result_df['cumulative_return'].iloc[-1]:.2f}%")
    print(f"Final number of stocks: {result_df['num_stocks'].iloc[-1]}")
    
    # Simulate benchmark portfolio
    benchmark_df = simulate_benchmark_portfolio(
        benchmark_ticker=benchmark_ticker,
        start_month=start_month,
        end_month=end_month,
        monthly_investment=monthly_investment
    )
    
    if not benchmark_df.empty:
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        strategy_value = result_df['portfolio_value'].iloc[-1]
        benchmark_value = benchmark_df['portfolio_value'].iloc[-1]
        strategy_return = result_df['cumulative_return'].iloc[-1]
        benchmark_return = benchmark_df['cumulative_return'].iloc[-1]
        
        print(f"\nFinal Portfolio Values:")
        print(f"  Strategy ({strategy_name}): ${strategy_value:,.2f}")
        print(f"  {benchmark_ticker}:         ${benchmark_value:,.2f}")
        print(f"  Difference:                     ${strategy_value - benchmark_value:,.2f}")
        print(f"  Outperformance:                 {((strategy_value / benchmark_value - 1) * 100):.2f}%")
        
        print(f"\nCumulative Returns:")
        print(f"  Strategy ({strategy_name}): {strategy_return:.2f}%")
        print(f"  {benchmark_ticker}:         {benchmark_return:.2f}%")
        print(f"  Difference:                     {strategy_return - benchmark_return:.2f} percentage points")
        
        # Save results
        invest_records_dir = os.path.join(MODEL_DIR, 'investment_records')
        os.makedirs(invest_records_dir, exist_ok=True)
        
        output_file = os.path.join(invest_records_dir, f'simple_buy_hold_top_{top_k}.csv')
        result_df.to_csv(output_file, index=False)
        print(f"\n💾 Strategy results saved to: {output_file}")
        
        benchmark_file = os.path.join(invest_records_dir, f'benchmark_{benchmark_ticker}.csv')
        benchmark_df.to_csv(benchmark_file, index=False)
        print(f"💾 Benchmark results saved to: {benchmark_file}")
        
        # Plot comparison
        plot_output = os.path.join(invest_records_dir, f'comparison_top_{top_k}_vs_{benchmark_ticker}.png')
        plot_portfolio_comparison(result_df, benchmark_df, strategy_name, benchmark_ticker, plot_output)
    else:
        # Save strategy results even if benchmark fails
        invest_records_dir = os.path.join(MODEL_DIR, 'investment_records')
        os.makedirs(invest_records_dir, exist_ok=True)
        output_file = os.path.join(invest_records_dir, f'simple_buy_hold_top_{top_k}.csv')
        result_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        print(f"⚠️  Benchmark comparison skipped due to errors")
    
    return 0


def main():
    """
    Main entry point.
    
    First runs invest_monthly_retro_performance() to generate prediction scores,
    then evaluates the strategy against a benchmark index.
    """
    # Step 1: Generate prediction scores
    print("="*70)
    print("STEP 1: Running investment strategy to generate prediction scores")
    print("="*70)
    invest_monthly_retro_performance()
    
    # Step 2: Evaluate strategy against benchmark (skip strategy since we just ran it)
    return eval_strategy_against_index(skip_strategy=True)


if __name__ == "__main__":
    exit(main())