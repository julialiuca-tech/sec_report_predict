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

import os

import numpy as np
import pandas as pd
import xgboost as xgb

from config import (
    COMPLETENESS_THRESHOLD,
    FEATURIZED_ALL_QUARTERS_FILE,
    FEATURE_IMPORTANCE_RANKING_FLAG,
    FEATURE_SUFFIXES,
    FILTER_OUTLIERS_FROM_RATIOS,
    MODEL_DIR,
    QUARTER_GRADIENTS,
    SPLIT_STRATEGY,
    STOCK_TREND_DATA_FILE,
    SUFFIXES_TO_ENHANCE_W_GRADIENT,
    TOP_K_FEATURES,
    USE_RATIO_FEATURES,
    Y_LABEL,
)
from feature_augment import compute_ratio_features, flag_outliers_by_hard_limits
from featurize import enhance_tags_w_gradient
from utility_data import standardize_df_to_reference
from utility_binary_classifier import baseline_binary_classifier, split_train_val_by_column
from config import SAVE_DIR


def collect_column_names_w_suffix(cols, feature_suffixes=['_current']):
    """
    Collect column names that end with any suffix in the suffixes list.
    
    Args:
        cols (list): List of column names to check
        feature_suffixes (list): List of suffixes to match against
        
    Returns:
        list: List of column names that end with any suffix in the suffixes list
    """
    return [col for col in cols if any(col.endswith(suffix) for suffix in feature_suffixes)]

def prep_data_feature_label(df_featurized_data, 
                            df_stock_trend,
                            df_history_data=None, 
                            quarters_for_gradient_comp=None):
    """
    Process and join featurized data with stock trends.
    
    Args:
        df_featurized_data (pd.DataFrame): Featurized data DataFrame
        df_stock_trend (pd.DataFrame): Stock trend data DataFrame (required)
        df_history_data (pd.DataFrame, optional): Historical data DataFrame for gradient computation.
                                                   If provided, will be used to standardize columns and 
                                                   compute gradient features.
        quarters_for_gradient_comp (list, optional): List of quarters to compute gradients from. 
                                                    If None, no gradient features are computed.
    
    Returns:
        pd.DataFrame: Joined dataset with features and labels
    """
    
    # Use provided featurized data
    df_features = df_featurized_data.copy()
    print(f"Features loaded: {df_features.shape}")

    # Prepare history data for gradient computation if provided
    df_history_for_gradient = None
    if df_history_data is not None:
        df_history = df_history_data.copy()
        print(f"History loaded: {df_history.shape}")
        df_features = standardize_df_to_reference(df_features, df_history)
        # Also prepare history data for gradient computation
        # Apply same processing steps to history data
        if USE_RATIO_FEATURES:
            df_history_for_gradient = compute_ratio_features(df_history.copy())
        else:
            df_history_for_gradient = df_history.copy()
    
    # Deduplicate features to ensure clean data from the start
    initial_count = len(df_features)
    df_features = df_features.drop_duplicates(subset=['cik', 'period'])
    final_count = len(df_features)
    if initial_count != final_count:
        print(f"🧹 Removed {initial_count - final_count:,} duplicate records (cik, period)")
        print(f"Features after deduplication: {df_features.shape}")

    # augment with ratio features 
    if USE_RATIO_FEATURES:
        df_features = compute_ratio_features(df_features)
        print(f"Ratio features computed: {df_features.shape}") 

    # enhance with gradient features
    if quarters_for_gradient_comp is not None:
        df_features = enhance_tags_w_gradient(df_features, 
                                              df_extra_history_for_gradient=df_history_for_gradient,
                                              quarters_for_gradient_comp=quarters_for_gradient_comp, 
                                              suffixes_to_enhance=SUFFIXES_TO_ENHANCE_W_GRADIENT)
        print(f"Gradient features loaded: {df_features.shape}") 

    # filter outliers from ratio features
    if FILTER_OUTLIERS_FROM_RATIOS:
        df_features = flag_outliers_by_hard_limits(df_features)
        df_features = df_features[df_features['flag_outlier'] == False]
        df_features.drop(columns=['flag_outlier'], inplace=True)
        print(f"Outliers filtered from ratio features: {df_features.shape} records remaining") 
    
    # Use provided stock trends
    df_trends = df_stock_trend.copy()
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

    Strategy: 
    - all: select all features
    - completeness: select features with completeness >= COMPLETENESS_THRESHOLD
    - current: select features with suffixes in FEATURE_SUFFIXES
    - change: select features with _change
    """

    # Identify feature columns (any suffix in FEATURE_SUFFIXES or contains '_change')
    suffix_cols = collect_column_names_w_suffix(df.columns, FEATURE_SUFFIXES)
    feature_cols = suffix_cols + [col for col in df.columns if '_change' in col and col not in suffix_cols]
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
        return collect_column_names_w_suffix(feature_cols, FEATURE_SUFFIXES)
    
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

    Strategy:
    - none: do not impute
    - median: impute with median of the training data
    """

    if imputation_strategy == 'median':
        # Simple median imputation (baseline approach)
        X_train_imputed = X_train.fillna(X_train.median())
        X_val_imputed = X_val.fillna(X_train.median())  # Use training median for validation
        return X_train_imputed, X_val_imputed

    else: 
        return X_train.copy(), X_val.copy()


def build_baseline_model(df_train, df_val, feature_cols):
    """
    Build and evaluate baseline models with different feature selection and imputation strategies.
    
    Args:
        df_train (pd.DataFrame): Training dataframe with features and labels
        df_val (pd.DataFrame): Validation dataframe with features and labels
        feature_cols (list): List of feature column names to use
        
    Returns:
        tuple: (model_perf_records, feature_importance_ranking)
            - model_perf_records: Dictionary with performance metrics for each model configuration
            - feature_importance_ranking: DataFrame with features ranked by importance (descending order)
    """
    print(f"\n" + "="*60)
    print("Testing Different Missing Value Handling Approaches...")

    # Extract features and labels from dataframes
    X_train = df_train[feature_cols].copy()
    X_val = df_val[feature_cols].copy()
    y_train = df_train[Y_LABEL].copy()
    y_val = df_val[Y_LABEL].copy()

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
                    # get additional performance metric: correlation between confidence and growth
                    y_pred_proba = model_perf['trained_model'].predict_proba(X_val_imputed)[:, 1]
                    corr_confidence_w_growth = np.corrcoef(y_pred_proba, df_val['price_return'])[0, 1]
                    model_perf['corr_confidence_w_growth'] = corr_confidence_w_growth
                    # collect performance records 
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
                'ROC-AUC': f"{model_perf['roc_auc']:.4f}",
                'Corr_Confidence_Growth': f"{model_perf['corr_confidence_w_growth']:.4f}"
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
        print(f"   Corr_Confidence_Growth: {best_model_perf['corr_confidence_w_growth']:.4f}")
        
        # Show top 20 features with correlation to label
        print(f"\n🔍 Top 20 Most Important Features (with correlation to label):")
        print("=" * 80)
        top_features_df = best_model_perf['feature_importance'].head(20)
        for i, row in top_features_df.iterrows():
            feature_name = row['feature']
            importance = row['importance']
            # Calculate correlation between feature and label
            if feature_name in X_train.columns:
                correlation = X_train[feature_name].corr(y_train)
            else:
                correlation = float('nan')
            print(f"  {i:2d}. {feature_name:<30} Importance: {importance:.4f}  Correlation: {correlation:.4f}")
       
        feature_importance_ranking = best_model_perf['feature_importance'].sort_values('importance', ascending=False)
        if FEATURE_IMPORTANCE_RANKING_FLAG: # If True, retrain model with top K features based on importance
            # Get feature importance ranking (descending order)
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
            print(f"   Corr_Confidence_Growth: {retrained_model_perf['corr_confidence_w_growth']:.4f}")
    
    return model_perf_records, feature_importance_ranking


def filter_features_by_importance(df, feature_importance_ranking, top_k_features=None):
    """
    Filter dataframe to keep only top K features based on feature importance ranking.
    
    Args:
        df (pd.DataFrame): Input dataframe with features and non-feature columns
        feature_importance_ranking (pd.DataFrame): DataFrame with feature importance ranking,
                                                   must have 'feature' column
        top_k_features (int, optional): Number of top features to keep. If None, uses TOP_K_FEATURES from config.
        
    Returns:
        pd.DataFrame: Filtered dataframe with non-feature columns + top K features
    """
    if top_k_features is None:
        top_k_features = TOP_K_FEATURES
    
    # Get top K features
    top_k_features_list = feature_importance_ranking.head(top_k_features)['feature'].tolist()
    
    # Keep non-feature columns (cik, period, year_month, ticker, price_return, etc.)
    suffix_cols = collect_column_names_w_suffix(df.columns, FEATURE_SUFFIXES)
    non_feature_cols = [col for col in df.columns if col not in suffix_cols and '_change' not in col]
    
    # Filter dataframe to only include non-feature columns + top K features
    df_filtered = df[non_feature_cols + top_k_features_list].copy()
    
    print(f"📊 Original dataframe shape: {df.shape}")
    print(f"📊 Filtered dataframe shape: {df_filtered.shape}")
    print(f"📊 Kept {len(non_feature_cols)} non-feature columns and {len(top_k_features_list)} top features")
    
    return df_filtered


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
    # Load featurized data and stock trends
    df_features = pd.read_csv(FEATURIZED_ALL_QUARTERS_FILE)
    df_trends = pd.read_csv(STOCK_TREND_DATA_FILE)
    df = prep_data_feature_label(df_featurized_data=df_features, 
                                  df_stock_trend=df_trends,
                                  quarters_for_gradient_comp=QUARTER_GRADIENTS)
    pickle_file = os.path.join(SAVE_DIR, 'df_all.pkl')
    df.to_pickle(pickle_file)
    print(f"📊 All data pickled: {df.shape}")
    
    # Split data into train/val sets
    df_train, df_val = split_data_for_train_val(df, 
                                                train_val_split_prop=0.7,    
                                                train_val_split_strategy=SPLIT_STRATEGY)
    
    # Prepare feature columns with 
    suffix_cols = collect_column_names_w_suffix(df_train.columns, FEATURE_SUFFIXES)
    feature_cols = suffix_cols + [f for f in df_train.columns if '_change' in f and f not in suffix_cols]
 
    # Build and compare model
    print(f"\n" + "="*60)
    print("Building and Comparing ML Models...")
    model_perf_records, feature_importance_ranking = build_baseline_model(df_train, df_val, feature_cols)
    # Save feature importance ranking to file for future use
    if feature_importance_ranking is not None:  
        feature_importance_ranking_file = os.path.join(MODEL_DIR, 'feature_importance_ranking.csv')
        feature_importance_ranking.to_csv(feature_importance_ranking_file, index=False)
        print(f"💾 Saved feature importance ranking to: {feature_importance_ranking_file}")

    # for comparison, build a model without _augment features 
    print(f"\n" + "="*60)
    print("Building and Comparing ML Models without _augment features...")
    suffix_cols = collect_column_names_w_suffix(df_train.columns, ['_current'])
    feature_cols = suffix_cols + [f for f in df_train.columns if '_change' in f and f not in suffix_cols]
    model_perf_records_no_augment, feature_importance_ranking_no_augment = build_baseline_model(df_train, df_val, feature_cols)
    if feature_importance_ranking_no_augment is not None:  
        feature_importance_ranking_no_augment_file = os.path.join(MODEL_DIR, 'feature_importance_ranking_no_augment.csv')
        feature_importance_ranking_no_augment.to_csv(feature_importance_ranking_no_augment_file, index=False)
        print(f"💾 Saved feature importance ranking to: {feature_importance_ranking_no_augment_file}")


if __name__ == "__main__":
    main()

