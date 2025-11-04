#!/usr/bin/env python3
"""
Baseline Machine Learning Model for SEC Data

This script serves as the starting point for building a baseline ML model
using the featurized SEC financial data.

Current functionality:
- Loads simplified featurized data
- Ready for ML model development

Future ML pipeline components:
- Feature engineering and selection
- Model training and validation
- Performance evaluation
- Hyperparameter tuning
"""

import pandas as pd

# Import utility functions
from utility_binary_classifier import split_train_val_by_column, baseline_binary_classifier
from utility_data import price_trend
from config import COMPLETENESS_THRESHOLD, Y_LABEL, SPLIT_STRATEGY, QUARTER_GRADIENTS
from config import TOP_K_FEATURES, FEATURE_IMPORTANCE_RANKING_FLAG
from config import TREND_HORIZON_IN_MONTHS, INVEST_EXP_START_MONTH_STR, INVEST_EXP_END_MONTH_STR
from config import FEATURIZED_ALL_QUARTERS_FILE, STOCK_TREND_DATA_FILE, MONTH_END_PRICE_FILE
import xgboost as xgb
from featurize import enhance_tags_w_gradient, summarize_feature_completeness




TEST_CUTOFF_DATE = 20241231

def prep_data_feature_label(featurized_data_file=FEATURIZED_ALL_QUARTERS_FILE, 
                             stock_trend_data_file=STOCK_TREND_DATA_FILE, 
                             quarters_for_gradient_comp=None):
    """
    Load and join featurized data with stock trends.
    
    Args:
        featurized_data_file (str): Path to featurized data file
        stock_trend_data_file (str): Path to stock trend data file
        quarters_for_gradient_comp (list, optional): List of quarters to compute gradients from. 
                                                    If None, no gradient features are computed.
    
    Returns:
        pd.DataFrame: Joined dataset with features and labels
    """
    
    # Load simplified featurized data 
    df_features = pd.read_csv(featurized_data_file)
    print(f"Features loaded: {df_features.shape}")
    
    # Deduplicate features to ensure clean data from the start
    initial_count = len(df_features)
    df_features = df_features.drop_duplicates(subset=['cik', 'period'])
    final_count = len(df_features)
    if initial_count != final_count:
        print(f"🧹 Removed {initial_count - final_count:,} duplicate records (cik, period)")
        print(f"Features after deduplication: {df_features.shape}")

    if quarters_for_gradient_comp is not None:
        df_features = enhance_tags_w_gradient(df_features, quarters_for_gradient_comp)
        print(f"Gradient features loaded: {df_features.shape}")
        print("columns: ", df_features.columns)
    
    # Load ground truth -- stock price trends
    df_trends = pd.read_csv(stock_trend_data_file)
    print(f"Trends loaded: {df_trends.shape}")

    # Join features and trends on cik and year_month resolution
    # Convert period and month_end_date to year_month for proper joining
    # Period is in YYYYMMDD format, so parse it correctly
    df_features['year_month'] = pd.to_datetime(df_features['period'], format='%Y%m%d').dt.to_period('M')
    # Handle timezone-aware dates by converting to naive datetime first
    df_trends['year_month'] = pd.to_datetime(df_trends['month_end_date']).dt.tz_localize(None).dt.to_period('M')
    # Inner join on cik and year_month
    df = df_features.merge(df_trends, on=['cik', 'year_month'], how='inner')
    print(f"Joined data: {df.shape}")

    return df


def split_data_for_train_val(df_work, train_val_split_prop=0.7, train_val_split_strategy=SPLIT_STRATEGY):
    """
    Split data into train/val sets.
    
    Args:
        df_work (pd.DataFrame): Work dataset to split (already filtered for train/val)
        train_val_split_prop (float): Proportion of work data to use for training (0.0 to 1.0)
        train_val_split_strategy (dict): Dictionary with one key-value pair for splitting strategy
                                       (e.g., {'cik': 'random'}, {'period': 'top'})
    
    Returns:
        tuple: (df_train, df_val) - Training, validation DataFrames
               - df_train: Training data (from work set)
               - df_val: Validation data (from work set) 
    """
    

    # # Perform correlation analysis
    # print(f"\n" + "="*60)
    # print("Feature Correlation Analysis...")
    # correlations = correlation_analysis(df_work, Y_LABEL)

    # Use the split_strategy parameter - expect a dictionary with one key
    try:    
        by_column = list(train_val_split_strategy.keys())[0]
        split_for_training = train_val_split_strategy[by_column]
        print(f"Splitting data by {by_column} using {split_for_training} strategy")
        df_train, df_val = split_train_val_by_column(df_work, train_val_split_prop, by_column, split_for_training)
    except Exception as e:
        print(f"❌ Error with split_strategy: {str(e)}")
        print("🔄 Falling back to random splitting...")
        df_train, df_val = split_train_val_by_column(df_work, train_val_split_prop, None, 'random')

    return df_train, df_val


def select_feature_cols(df, strategy='all'):
    """
    Select feature columns based on strategy (all, completeness, current, change).
    
    Args:
        df (pd.DataFrame): DataFrame containing features
        strategy (str): Selection strategy
        
    Returns:
        list: Selected feature column names
    """
    # Identify feature columns
    feature_cols = [col for col in df.columns if '_current' in col or '_change' in col]
    if len(feature_cols) == 0:
        print("❌ No feature columns found for feature selection.")
        return []
    
    if strategy == 'all':
        return feature_cols 

    if strategy == 'completeness': 
        threshold = COMPLETENESS_THRESHOLD
        completeness = df[feature_cols].notna().mean()
        filtered_features = completeness[completeness >= threshold].index.tolist()
        return filtered_features
    
    if strategy == 'current': 
        return [col for col in feature_cols if '_current' in col]
    
    if strategy == 'change':
        return [col for col in feature_cols if '_change' in col]


def apply_imputation(X_train, X_val, imputation_strategy='none'): 
    """
    Apply imputation strategy to handle missing values.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_val (pd.DataFrame): Validation features
        imputation_strategy (str): Strategy ('none' or 'median')
        
    Returns:
        tuple: (X_train_imputed, X_val_imputed)
    """
    if imputation_strategy == 'median':
        # Simple median imputation (baseline approach)
        X_train_imputed = X_train.fillna(X_train.median())
        X_val_imputed = X_val.fillna(X_train.median())  # Use training median for validation
        return X_train_imputed, X_val_imputed

    else: 
        return X_train.copy(), X_val.copy()



def build_baseline_model(X_train, X_val, y_train, y_val, feature_cols):
    """
    Build and evaluate baseline models with different feature selection and imputation strategies.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_val (pd.DataFrame): Validation features
        y_train (pd.Series): Training target labels
        y_val (pd.Series): Validation target labels
        feature_cols (list): List of feature column names
        
    Returns:
        tuple: (model_perf_records, feature_importance_ranking)
            - model_perf_records: Dictionary with performance metrics for each model configuration
            - feature_importance_ranking: DataFrame with features ranked by importance (descending order)
    """
    print(f"\n" + "="*60)
    print("Testing Different Missing Value Handling Approaches...")

    model_perf_records = {}
    feature_selection_strategy_list = ['completeness', 'current', 'change', 'all']
    imputation_strategy_list = [ 'none', 'median']
    model_type_list = ['rf', 'xgb']
    X_train_orig, X_val_orig = X_train.copy(), X_val.copy()

    for selection in feature_selection_strategy_list:
         print(f"\n🔍 Testing strategy: {selection}")
         feature_cols = select_feature_cols(X_train_orig, selection)
         print(f"   Selected {len(feature_cols)} features")
         
         if len(feature_cols) == 0:
             print(f"❌ No features found for strategy '{selection}'. Skipping...")
             continue
         
         X_train_filtered = X_train_orig[feature_cols]
         X_val_filtered = X_val_orig[feature_cols]
         print(f"   Training data shape: {X_train_filtered.shape}")
         print(f"   Validation data shape: {X_val_filtered.shape}")
         
         for imputation in imputation_strategy_list:
             X_train_imputed, X_val_imputed = apply_imputation(X_train_filtered, X_val_filtered, imputation)
             for model_name in model_type_list:
                 try:
                     model_perf = baseline_binary_classifier(X_train_imputed, X_val_imputed, y_train, y_val, model_name)
                     model_perf_records[f"{selection}_{imputation}_{model_name}"] = model_perf
                 except Exception as e:
                     print(f"❌ Error with {selection}_{imputation}_{model_name}: {str(e)}")
                     model_perf_records[f"{selection}_{imputation}_{model_name}"] = None

    
    # 4. Compare results and recommend best approach
    print(f"\n" + "="*80)
    print("Model Comparison Results:")
    print("="*80)
    
    # Create comparison table
    comparison_data = []
    for approach_name, model_perf in model_perf_records.items():
        if model_perf is not None:
            comparison_data.append({
                'Approach': approach_name,
                'Accuracy': f"{model_perf['accuracy']:.4f}",
                'Precision': f"{model_perf['precision']:.4f}",
                'Recall': f"{model_perf['recall']:.4f}",
                'ROC-AUC': f"{model_perf['roc_auc']:.4f}"
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Find best approach by ROC-AUC
        best_approach = max([k for k, v in model_perf_records.items() if v is not None], 
                          key=lambda k: model_perf_records[k]['roc_auc'])
        best_model_perf = model_perf_records[best_approach]
        
        print(f"\n🏆 Best Approach: {best_approach}")
        print(f"   Accuracy: {best_model_perf['accuracy']:.4f}")
        print(f"   Precision: {best_model_perf['precision']:.4f}")
        print(f"   Recall: {best_model_perf['recall']:.4f}")
        print(f"   ROC-AUC: {best_model_perf['roc_auc']:.4f}")
        
        # Show top 10 features
        print(f"\n🔍 Top 10 Most Important Features:")
        print("=" * 50)
        for i, row in best_model_perf['feature_importance'].head(10).iterrows():
            print(f"  {i:2d}. {row['feature']:<30} {row['importance']:.4f}")
        
    IF FEATURE_IMPORTANCE_RANKING_FLAG:
        # Get feature importance ranking (descending order)
        FEATURE_IMPORTANCE_RANKING_FLAG = True
        feature_importance_ranking = best_model_perf['feature_importance'].sort_values('importance', ascending=False)
        top_k_features = feature_importance_ranking.head(TOP_K_FEATURES)['feature'].tolist()
        # Filter training and validation data to only include top K features
        X_train_topk = X_train[top_k_features].copy()
        X_val_topk = X_val[top_k_features].copy()
        
        # Determine the best model type from the approach name
        best_model_type = best_approach.split('_')[-1]  # Extract model type (rf or xgb) 
        retrained_model_perf = baseline_binary_classifier(X_train_topk, X_val_topk, y_train, y_val, best_model_type)
        print(f"\n🏆 Retrained Model Performance (Top {TOP_K_FEATURES} features):")
        print(f"   Model: {best_model_type.upper()}")
        print(f"   Accuracy: {retrained_model_perf['accuracy']:.4f}")
        print(f"   Precision: {retrained_model_perf['precision']:.4f}")
        print(f"   Recall: {retrained_model_perf['recall']:.4f}")
        print(f"   ROC-AUC: {retrained_model_perf['roc_auc']:.4f}")
        
    else:
        feature_importance_ranking = None

    
    
    return model_perf_records, feature_importance_ranking




def top_candidate_w_return(model, df_companies, feature_cols, K=10):
    """
    Compute the performance of the model on the top K companies.
    
    Args:
        model: The trained model
        df_companies: The dataframe containing the companies
        feature_cols: The columns to use as features
        K (int): Number of top candidates to select (default: 10)
        
    Returns:
        tuple: (top_k_cumulative_return, market_avg_return, top_k_tickers)
            - top_k_cumulative_return: Cumulative average return from top K candidates
            - market_avg_return: Average return of the market (all companies)
            - top_k_tickers: List of top K candidates' tickers
    """
    X_companies = df_companies[feature_cols].copy()
    y_companies = df_companies[Y_LABEL].copy() 

    y_pred = model.predict(X_companies)
    y_pred_proba = model.predict_proba(X_companies)[:, 1]

    df_companies['y_pred_proba']= y_pred_proba
    df_companies.sort_values(by= 'y_pred_proba', ascending=False, inplace=True)

    # Calculate cumulative average of invest_val as we slide down the sorted dataframe
    df_companies['cumulative_avg_return'] = df_companies['price_return'].expanding().mean()
    df_companies['cumulative_count'] = range(1, len(df_companies) + 1)
    
    # Calculate analysis metrics
    # Get the cumulative return for top K candidates
    if len(df_companies) >= K:
        top_k_cumulative_return = df_companies.iloc[K-1]['cumulative_avg_return']
        top_k_companies = df_companies.iloc[:K]
    else:
        top_k_cumulative_return = df_companies['cumulative_avg_return'].iloc[-1]
        top_k_companies = df_companies
    
    # Get market average return (all companies)
    market_avg_return = df_companies['price_return'].mean()
    
    # Get list of top K tickers
    top_k_tickers = top_k_companies['ticker'].tolist()
    
    return top_k_cumulative_return, market_avg_return, top_k_tickers


def invest_top10_per_month(df):
    """
    Test investment strategy using ML model predictions with time-based train/test splits.
    
    Iterates over months from 2024-01 to 2025-04, where each month serves as test data
    and all previous months serve as training data.
    
    Returns:
        None -- prints performance analysis for each month
    """  

    # Get feature columns from the full dataset
    feature_cols = [f for f in df.columns if '_current' in f or '_change' in f]
    
    # Define the range of months to test
    import pandas as pd
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
    
    # save the records 
    result_returns = [] 
    df_invest_record = pd.DataFrame(columns=['year_month', 'rank', 'ticker'])  
    # Iterate over each month
    for current_month in months_to_test:
        print(f"\n" + "="*60)
        print(f"📅 Testing performance for {current_month}")
        
        # Get test data (current month)
        df_test = df[df['year_month'] == current_month].copy()
        
        # Get training data (all months before current month)
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
        top10_return, avg_return, top10_tickers = top_candidate_w_return(model, df_test, feature_cols, K=10)
        print(f"  📊 Top 10 return: {top10_return:.4f}", f"  📊 Average return: {avg_return:.4f}")
        print(f"  📊 Top 10 tickers: {top10_tickers}")
        top10_return_cgt_adjusted = 1+0.8*(top10_return-1) if top10_return > 1 else top10_return
        result_returns.append([current_month, top10_return, avg_return, top10_return_cgt_adjusted])
        
        # Append all top tickers at once using vectorized operation
        new_rows = pd.DataFrame({
            'year_month': [current_month] * len(top10_tickers),
            'rank': range(1, len(top10_tickers) + 1),
            'ticker': top10_tickers
        })
        df_invest_record = pd.concat([df_invest_record, new_rows], ignore_index=True)

    # print out the 
    result_returns = pd.DataFrame(result_returns, columns=['year_month', 'top10_return', 'avg_return', 'top10_return_cgt_adjusted'])
   
    print(f"\n" + "="*60)
    print("Aggregated Investment Performance:")
    print(f"   Avg market return: {result_returns['avg_return'].mean():.4f}")
    print(f"   Top 10 return: {result_returns['top10_return'].mean():.4f}")
    print(f"   Top 10 return cgt adjusted: {result_returns['top10_return_cgt_adjusted'].mean():.4f}")
    print(f"="*60)
   
    return df_invest_record


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
    
    result_df_agg_by_month = result_df.groupby('year_month').agg({'price_return': 'mean'}).reset_index()
    print(result_df_agg_by_month.sort_values(by='year_month', ascending=True))
    print('average return: ', result_df_agg_by_month['price_return'].mean())
    result_df_agg_by_month['price_return_cgt_adjusted'] = result_df_agg_by_month['price_return'].apply(
        lambda x: 1+0.8*(x-1) if x > 1 else x
    )
    print('return cgt adjusted: ', result_df_agg_by_month['price_return_cgt_adjusted'].mean())
    
    result_df.sort_values(by=['year_month', 'rank'], ascending=True, inplace=True)
    result_df.to_csv('result_df.csv', index=False)
    return result_df


def main():
    """
    Run the complete SEC data analysis and ML pipeline.
    
    Returns:
        tuple: (model_perf_records, feature_importance_ranking, df_long_term_gain)
            - model_perf_records: Dictionary with performance metrics for each model configuration
            - feature_importance_ranking: DataFrame with features ranked by importance (descending order)
            - df_long_term_gain: DataFrame with long-term investment performance analysis
    """
    print("🚀 Starting SEC Data Analysis and ML Pipeline")
    print("=" * 60)
    
    
    # Prepare data (you can change split_strategy to 'date' for time-based splitting)
    # Load and join data
    df = prep_data_feature_label(quarters_for_gradient_comp=QUARTER_GRADIENTS)
    
    # Split data into train/val sets
    df_train, df_val = split_data_for_train_val(df, 
                                                train_val_split_prop=0.7,    
                                                train_val_split_strategy=SPLIT_STRATEGY)
    
    # Prepare features and target
    feature_cols = [f for f in df_train.columns if '_current' in f or '_change' in f]
    X_train = df_train[feature_cols].copy()
    X_val = df_val[feature_cols].copy()
    y_train = df_train[Y_LABEL].copy()
    y_val = df_val[Y_LABEL].copy()
 
    # Build and compare models
    print(f"\n" + "="*60)
    print("Building and Comparing ML Models...")
    model_perf_records, feature_importance_ranking = \
         build_baseline_model(X_train, X_val, y_train, y_val, feature_cols)
  
    if FEATURE_IMPORTANCE_RANKING_FLAG:
        # Get top K features
        top_k_features = feature_importance_ranking.head(TOP_K_FEATURES)['feature'].tolist()
        # Keep non-feature columns (cik, period, year_month, ticker, price_return, etc.)
        non_feature_cols = [col for col in df.columns if '_current' not in col and '_change' not in col]
        df_filtered = df[non_feature_cols + top_k_features].copy()
        print(f"📊 Original dataframe shape: {df.shape}", f"📊 Filtered dataframe shape: {df_filtered.shape}")
        df = df_filtered 
    
    # test the investment performance
    print("="*60)
    print("Testing Investment Performance...")
    print("="*60)
    df_invest_record = invest_top10_per_month(df)
    df_long_term_gain = long_term_gain_from_invest_record(df_invest_record, num_months_horizon=12)
    print(df_long_term_gain)
    


if __name__ == "__main__":
    main()

