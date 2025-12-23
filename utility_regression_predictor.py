#!/usr/bin/env python3
"""
Regression Predictor for Stock Price Returns

This module provides regression models to predict stock price returns (continuous target variable).
It uses the same feature preparation as the contrastive classifier and supports multiple regression
algorithms (XGBoost, Random Forest, etc.) for comparison.
"""

import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
from scipy.stats import pearsonr
import xgboost as xgb
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from config import (
    FEATURIZED_ALL_QUARTERS_FILE,
    STOCK_TREND_DATA_FILE,
    QUARTER_GRADIENTS,
    Y_LABEL,
    FEATURE_SUFFIXES,
    MODEL_DIR,
    FEATURE_IMPORTANCE_RANKING_FLAG,
)
from baseline_model import (
    prep_data_feature_label,
    collect_column_names_w_suffix,
    apply_imputation,
    filter_features_by_importance,
)


class RegressionStockPredictor:
    """
    Wrapper class for regression models to predict stock price returns.
    
    Supports multiple regression algorithms for easy comparison.
    """
    
    def __init__(self, model_type='xgboost', **model_params):
        """
        Initialize regression predictor.
        
        Args:
            model_type (str): Type of regression model. Supported types:
                - 'xgboost': XGBoost gradient boosting (handles missing values natively)
                - 'random_forest': Random Forest (Note: sklearn's RandomForestRegressor does NOT handle 
                  missing values natively, same as RandomForestClassifier. Despite some comments suggesting 
                  otherwise, both require imputation or will fail with NaN values.)
                - 'lightgbm': LightGBM gradient boosting (handles missing values natively, if available)
                - 'gradient_boosting': sklearn Gradient Boosting (requires imputation)
            **model_params: Additional parameters to pass to the underlying model
        """
        self.model_type = model_type
        self.model_params = model_params
        self.model_ = None
        self.feature_cols_ = None
        
        # Initialize model based on type
        if model_type == 'xgboost':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'random_state': 42,
            }
            default_params.update(model_params)
            self.model_ = xgb.XGBRegressor(**default_params)
        elif model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1,  # Use all available CPU cores for parallel processing
            }
            default_params.update(model_params)
            self.model_ = RandomForestRegressor(**default_params)
        elif model_type == 'lightgbm':
            if not HAS_LIGHTGBM:
                raise ImportError("LightGBM is not installed. Install it with: pip install lightgbm")
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'random_state': 42,
                'verbosity': -1,  # Suppress LightGBM output
            }
            default_params.update(model_params)
            self.model_ = lgb.LGBMRegressor(**default_params)
        elif model_type == 'gradient_boosting':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'random_state': 42,
                # Note: sklearn's GradientBoostingRegressor does not support parallel processing (n_jobs)
                # It trains trees sequentially, which is why it can be slower than Random Forest
            }
            default_params.update(model_params)
            self.model_ = GradientBoostingRegressor(**default_params)
        else:
            supported = ['xgboost', 'random_forest', 'lightgbm', 'gradient_boosting']
            raise ValueError(f"Unknown model_type: {model_type}. Supported: {supported}")
    
    def fit(self, X, y):
        """
        Train the regression model.
        
        Args:
            X (pd.DataFrame or np.array): Training features
            y (pd.Series or np.array): Training target (price_return)
        
        Returns:
            self: Returns self for method chaining
        """
        # Store feature columns if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_cols_ = X.columns.tolist()
            X_train = X.values
        else:
            X_train = X
        
        y_train = np.asarray(y)
        
        # Train model
        self.model_.fit(X_train, y_train)
        
        return self
    
    def predict(self, X):
        """
        Predict price returns for samples.
        
        Args:
            X (pd.DataFrame or np.array): Test features
        
        Returns:
            np.array: Predicted price returns
        """
        if isinstance(X, pd.DataFrame):
            # Ensure columns match training data
            if self.feature_cols_ is not None:
                X_test = X[self.feature_cols_].values
            else:
                X_test = X.values
        else:
            X_test = X
        
        return self.model_.predict(X_test)
    
    def score(self, X, y):
        """
        Return R² score on the given test data and labels.
        
        Args:
            X (pd.DataFrame or np.array): Test samples
            y (pd.Series or np.array): True price returns
        
        Returns:
            float: R² score
        """
        y_pred = self.predict(X)
        y_true = np.asarray(y)
        return r2_score(y_true, y_pred)


def evaluate_regression_predictor(model, X_test, y_test, df_test=None, print_report=True):
    """
    Evaluate the regression predictor and print performance metrics.
    
    Args:
        model: Trained RegressionStockPredictor
        X_test: Test features (DataFrame or array)
        y_test: Test target values (price_return)
        df_test (pd.DataFrame, optional): Full test dataframe. If provided and contains Y_LABEL column
                                          (from config), will compute correlation with predicted values.
        print_report (bool): Whether to print evaluation report
    
    Returns:
        dict: Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_test = np.asarray(y_test)
    
    # Compute error metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Compute correlation with true target
    correlation_target, p_value_target = pearsonr(y_test, y_pred)
    
    results = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'correlation_target': correlation_target,
        'correlation_target_p_value': p_value_target,
        'predictions': y_pred,
    }
    
    # Compute correlation with Y_LABEL (trend_up_or_down) if df_test is provided
    correlation_trend_coef = None
    correlation_trend_p_value = None
    roc_auc = None
    if df_test is not None and Y_LABEL in df_test.columns:
        # Align indices: if X_test is a DataFrame, use its index; otherwise use range
        if isinstance(X_test, pd.DataFrame):
            y_label_values = df_test.loc[X_test.index, Y_LABEL].values
        else:
            # If X_test is array, assume it matches df_test row order
            y_label_values = df_test[Y_LABEL].values[:len(y_pred)]
        
        # Remove any NaN values for correlation computation
        valid_mask = ~np.isnan(y_label_values)
        if valid_mask.sum() > 0:
            correlation_trend_coef, correlation_trend_p_value = pearsonr(
                y_label_values[valid_mask], 
                y_pred[valid_mask]
            )
            results['correlation_trend_coef'] = correlation_trend_coef
            results['correlation_trend_p_value'] = correlation_trend_p_value
            
            # Compute ROC-AUC using predicted price_return as scores and trend_up_or_down as truth
            # Valid mask already excludes NaN values in y_label_values
            # Also check for NaN in y_pred
            valid_mask_roc = valid_mask & ~np.isnan(y_pred)
            if valid_mask_roc.sum() > 0:
                y_label_binary = y_label_values[valid_mask_roc]
                y_pred_scores = y_pred[valid_mask_roc]
                
                # Ensure binary labels (should be 0/1, but check for edge cases)
                unique_labels = np.unique(y_label_binary)
                if len(unique_labels) == 2 and set(unique_labels).issubset({0, 1}):
                    try:
                        roc_auc = roc_auc_score(y_label_binary, y_pred_scores)
                        results['roc_auc'] = roc_auc
                    except ValueError as e:
                        # Handle edge cases (e.g., only one class present after filtering)
                        roc_auc = None
                        results['roc_auc'] = None
    
    if print_report:
        print("\n" + "="*60)
        print("REGRESSION PREDICTOR EVALUATION")
        print("="*60)
        print(f"\nModel Type: {model.model_type}")
        print(f"\nError Metrics:")
        print(f"  Mean Squared Error (MSE): {mse:.6f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.6f}")
        print(f"  Mean Absolute Error (MAE): {mae:.6f}")
        print(f"  R² Score: {r2:.6f}")
        print(f"\nCorrelation Metrics:")
        print(f"  Correlation (predicted vs actual price_return): {correlation_target:.4f} (p-value: {p_value_target:.4e})")
        
        if correlation_trend_coef is not None:
            print(f"  Correlation (predicted price_return vs {Y_LABEL}): {correlation_trend_coef:.4f} (p-value: {correlation_trend_p_value:.4e})")
        
        if roc_auc is not None:
            print(f"  ROC-AUC (predicted price_return as scores vs {Y_LABEL} as truth): {roc_auc:.4f}")
        elif df_test is not None and Y_LABEL in df_test.columns:
            print(f"  ROC-AUC (predicted price_return vs {Y_LABEL}): Not available (insufficient data or non-binary labels)")
        
        print("="*60)
    
    return results


def compare_regression_models(X_train, X_test, y_train, y_test, df_test=None, models_to_compare=None):
    """
    Compare multiple regression models and print their performance.
    
    Args:
        X_train: Training features (DataFrame or array)
        X_test: Test features (DataFrame or array)
        y_train: Training target (price_return)
        y_test: Test target (price_return)
        df_test (pd.DataFrame, optional): Full test dataframe for trend correlation
        models_to_compare (list, optional): List of model types to compare. 
                                           Default: ['xgboost', 'random_forest']
    
    Returns:
        dict: Dictionary mapping model_type to evaluation results
    """
    if models_to_compare is None:
        models_to_compare = ['xgboost', 'random_forest']
    
    results = {}
    
    print("\n" + "="*70)
    print("COMPARING REGRESSION MODELS")
    print("="*70)
    
    for model_type in models_to_compare:
        print(f"\n{'='*70}")
        print(f"Training and evaluating {model_type} model...")
        print(f"{'='*70}")
        
        # Initialize and train model
        model = RegressionStockPredictor(model_type=model_type)
        model.fit(X_train, y_train)
        
        # Evaluate model
        model_results = evaluate_regression_predictor(
            model, X_test, y_test, df_test=df_test, print_report=True
        )
        
        results[model_type] = {
            'model': model,
            'results': model_results
        }
    
    # Print comparison summary
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Model':<20} {'RMSE':<12} {'MAE':<12} {'R²':<12} {'Corr (target)':<15} {'Corr (trend)':<15} {'ROC-AUC':<12}")
    print("-" * 70)
    
    for model_type, model_data in results.items():
        r = model_data['results']
        corr_trend = r.get('correlation_trend_coef', np.nan)
        corr_trend_str = f"{corr_trend:.4f}" if not np.isnan(corr_trend) else "N/A"
        roc_auc = r.get('roc_auc', None)
        roc_auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
        print(f"{model_type:<20} {r['rmse']:<12.6f} {r['mae']:<12.6f} {r['r2']:<12.4f} "
              f"{r['correlation_target']:<15.4f} {corr_trend_str:<15} {roc_auc_str:<12}")
    
    print("="*70)
    
    return results


def main():
    """
    Demonstration of regression models using real project data.
    
    This function follows the same data preparation process as utility_contrastive_classifier.py:main_real_data().
    """
    print("="*70)
    print("REGRESSION STOCK PREDICTOR - REAL DATA DEMONSTRATION")
    print("="*70)
    
    # ============================================================================
    # STEP 0: Prepare data (same as contrastive classifier)
    # ============================================================================
    print("\n📊 Loading and preparing real project data...")
    
    # Load featurized data and stock trends
    df_features = pd.read_csv(FEATURIZED_ALL_QUARTERS_FILE)
    df_trends = pd.read_csv(STOCK_TREND_DATA_FILE)
    df = prep_data_feature_label(
        df_featurized_data=df_features, 
        df_stock_trend=df_trends,
        quarters_for_gradient_comp=QUARTER_GRADIENTS
    )
    
    # Prepare feature columns (same as contrastive classifier)
    suffix_cols = collect_column_names_w_suffix(df.columns, feature_suffixes=FEATURE_SUFFIXES)
    feature_cols = suffix_cols + [f for f in df.columns if '_change' in f and f not in suffix_cols]
    
    # Extract features and target
    X = df[feature_cols].copy()
    y = df['price_return'].copy()
    
    # Handle infinite values first (replace with NaN)
    print(f"\n   Data cleaning...")
    X_cleaned = X.replace([np.inf, -np.inf], np.nan)
    
    # Remove zero-variance columns
    X_var = X_cleaned.var()
    zero_var_cols = X_var[X_var == 0].index.tolist()
    if len(zero_var_cols) > 0:
        print(f"   ⚠️  Dropping {len(zero_var_cols)} zero-variance columns.")
        X_cleaned = X_cleaned.drop(columns=zero_var_cols)
        feature_cols = [c for c in feature_cols if c not in zero_var_cols]
    
    # Filter out rows with NaN target values
    valid_mask = ~y.isna()
    X_cleaned = X_cleaned[valid_mask]
    y_cleaned = y[valid_mask]
    df_cleaned = df[valid_mask].reset_index(drop=True)
    X_cleaned = X_cleaned.reset_index(drop=True)
    y_cleaned = y_cleaned.reset_index(drop=True)
    
    print(f"   ✅ Data preparation complete: {X_cleaned.shape[0]} samples, {X_cleaned.shape[1]} features")
    
    # ============================================================================
    # STEP 1: Split data into train/test
    # ============================================================================
    from baseline_model import split_data_for_train_val
    from config import SPLIT_STRATEGY
    
    print("\n" + "="*70)
    print("STEP 1: Splitting data into train/test sets")
    print("="*70)
    
    # Combine features back into dataframe for splitting (using same columns)
    df_split = df_cleaned.copy()
    for col in X_cleaned.columns:
        df_split[col] = X_cleaned[col].values
    
    # Split using the same strategy as contrastive classifier
    df_train, df_test = split_data_for_train_val(
        df_split,
        train_val_split_prop=0.7,
        train_val_split_strategy=SPLIT_STRATEGY
    )
    
    # Extract train/test splits
    X_train = df_train[feature_cols].reset_index(drop=True)
    X_test = df_test[feature_cols].reset_index(drop=True)
    y_train = df_train['price_return'].reset_index(drop=True)
    y_test = df_test['price_return'].reset_index(drop=True)
    df_train_clean = df_train.reset_index(drop=True)
    df_test_clean = df_test.reset_index(drop=True)
    
    # Note: XGBoost and LightGBM can handle missing values natively.
    # Random Forest and sklearn Gradient Boosting do NOT handle missing values natively
    # (neither RandomForestClassifier nor RandomForestRegressor handle NaN values - 
    #  both will fail if given NaN values, so imputation is required).
    # We apply imputation here for consistency across all models.
    # If you want to leverage native missing value handling for XGBoost/LightGBM,
    # you could skip imputation for those models specifically.
    X_train, X_test = apply_imputation(X_train, X_test, imputation_strategy='median')
    
    # Additional check: handle columns that are all NaN (median would be NaN)
    # Fill any remaining NaN values (in case median was NaN for all-NaN columns) with 0
    train_median = X_train.median()
    all_nan_cols = train_median.isna().sum()
    if all_nan_cols > 0:
        print(f"   ⚠️  Found {all_nan_cols} columns with all NaN values in training data. Filling with 0.")
        train_median = train_median.fillna(0)
        X_train = X_train.fillna(train_median)
        X_test = X_test.fillna(train_median)
    
    # Final check: ensure no NaN values remain
    if X_train.isna().any().any():
        print(f"   ⚠️  Warning: Some NaN values remain in X_train after imputation. Filling with 0.")
        X_train = X_train.fillna(0)
    if X_test.isna().any().any():
        print(f"   ⚠️  Warning: Some NaN values remain in X_test after imputation. Filling with 0.")
        X_test = X_test.fillna(0)
    
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    
    # ============================================================================
    # STEP 2: Compare multiple regression models
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 2: Training and comparing regression models")
    print("="*70)
    
    # Compare multiple models (comment out models that are not needed)
    models_to_compare = [
        'xgboost',           # XGBoost - powerful gradient boosting
        'lightgbm',          # LightGBM - fast gradient boosting (if available)
        'gradient_boosting', # sklearn Gradient Boosting
        'random_forest',     # Random Forest - ensemble of trees
    ]
    
    # Filter out lightgbm if not available
    if not HAS_LIGHTGBM:
        models_to_compare = [m for m in models_to_compare if m != 'lightgbm']
        print("⚠️  LightGBM not available, skipping...")
    
    results = compare_regression_models(
        X_train, X_test, y_train, y_test, df_test=df_test_clean,
        models_to_compare=models_to_compare
    )
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    
    return results


def invest_monthly_retro_w_regression(INVEST_EXP_START_MONTH_STR='2023-01', INVEST_EXP_END_MONTH_STR='2025-07'):
    """
    Test investment strategies using LightGBM regression model predictions with time-based train/test splits.
    
    This function follows the same process as invest_monthly_retro_w_contrastive() in utility_contrastive_classifier.py,
    but uses LightGBM regression to predict price_return directly instead of 3-class probabilities.
    The predicted price_return values are converted to a [0, 1] score for compatibility with investment strategies.
    
    Iterates over months from INVEST_EXP_START_MONTH_STR to INVEST_EXP_END_MONTH_STR, where each 
    month serves as test data and all previous months serve as training data.
    
    Args:
        INVEST_EXP_START_MONTH_STR (str): Start month for investment experiment (default: '2023-01')
        INVEST_EXP_END_MONTH_STR (str): End month for investment experiment (default: '2025-07')
    
    Returns:
        None (prints results)
    """   
    if not HAS_LIGHTGBM:
        raise ImportError("LightGBM is not installed. Install it with: pip install lightgbm")
    
    # Import helper functions from invest_by_model
    from invest_by_model import top_candidate_w_return, augment_invest_record_w_long_term_return, benchmark_performance
    
    strategies = {
        'top_5': {'method': 'top_k', 'param': 5},
        'top_10': {'method': 'top_k', 'param': 10},
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
    
    print(f"\n📊 Testing investment strategy with LightGBM regression (time-based splits)...")
    print(f"📅 Testing months: {len(months_to_test)} months from {start_month} to {end_month}")
    
    # Prepare data - Load featurized data and stock trends
    df_features = pd.read_csv(FEATURIZED_ALL_QUARTERS_FILE)
    df_trends = pd.read_csv(STOCK_TREND_DATA_FILE)
    df = prep_data_feature_label(df_featurized_data=df_features, 
                                  df_stock_trend=df_trends,
                                  quarters_for_gradient_comp=QUARTER_GRADIENTS)

    if FEATURE_IMPORTANCE_RANKING_FLAG:
        feature_importance_ranking = pd.read_csv(os.path.join(MODEL_DIR, 'feature_importance_ranking.csv'))
        df = filter_features_by_importance(df, feature_importance_ranking) 
    
    # Get feature columns from the full dataset
    suffix_cols = collect_column_names_w_suffix(df.columns, feature_suffixes=FEATURE_SUFFIXES)
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
        
        # Skip if insufficient data
        if len(df_train) < 100:
            print(f"⚠️  Insufficient training data ({len(df_train)} samples), skipping...")
            continue
        if len(df_test) == 0:
            print(f"⚠️  No test data for {current_month}, skipping...")
            continue
        
        # Prepare features and target
        X_train = df_train[feature_cols].copy()
        y_train = df_train['price_return'].copy()
        X_test = df_test[feature_cols].copy()
        y_test = df_test['price_return'].copy()
        
        # Handle infinite values
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        
        # Apply imputation
        X_train, X_test = apply_imputation(X_train, X_test, imputation_strategy='median')
        
        # Additional check: handle columns that are all NaN (median would be NaN)
        train_median = X_train.median()
        all_nan_cols = train_median.isna().sum()
        if all_nan_cols > 0:
            train_median = train_median.fillna(0)
            X_train = X_train.fillna(train_median)
            X_test = X_test.fillna(train_median)
        
        # Final check: ensure no NaN values remain
        if X_train.isna().any().any():
            X_train = X_train.fillna(0)
        if X_test.isna().any().any():
            X_test = X_test.fillna(0)
        
        # Remove zero-variance columns
        train_var = X_train.var()
        zero_var_cols = train_var[train_var == 0].index.tolist()
        if len(zero_var_cols) > 0:
            X_train = X_train.drop(columns=zero_var_cols)
            X_test = X_test.drop(columns=zero_var_cols)
        
        # Filter out rows with NaN target values
        valid_mask_train = ~y_train.isna()
        valid_mask_test = ~y_test.isna()
        X_train = X_train[valid_mask_train].reset_index(drop=True)
        y_train = y_train[valid_mask_train].reset_index(drop=True)
        X_test = X_test[valid_mask_test].reset_index(drop=True)
        y_test = y_test[valid_mask_test].reset_index(drop=True)
        df_test = df_test[valid_mask_test].reset_index(drop=True)
        
        # Train LightGBM regression model
        model = RegressionStockPredictor(model_type='lightgbm')
        
        print(f"   Training LightGBM regression model...")
        model.fit(X_train, y_train)
        
        # Predict price_return
        y_pred_price_return = model.predict(X_test)
        
        # Evaluate model performance
        results = evaluate_regression_predictor(model, X_test, y_test, df_test=df_test, print_report=False)
        
        rmse = results['rmse']
        mae = results['mae']
        r2 = results['r2']
        correlation_target = results['correlation_target']
        print(f"📊 Model performance -- RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}, Correlation: {correlation_target:.4f}")
        
        if results.get('correlation_trend_coef') is not None:
            print(f"📊 Model performance -- Correlation (predicted price_return vs {Y_LABEL}): {results['correlation_trend_coef']:.4f}")
        
        if results.get('roc_auc') is not None:
            print(f"📊 Model performance -- ROC-AUC (predicted price_return vs {Y_LABEL}): {results['roc_auc']:.4f}")
        
        # Convert predicted price_return to [0, 1] score for compatibility with top_candidate_w_return
        # Normalize based on training data statistics to handle different scales
        # Use min-max normalization: (pred - min) / (max - min)
        # But we need to use training statistics to avoid data leakage
        train_price_return_min = y_train.min()
        train_price_return_max = y_train.max()
        price_return_range = train_price_return_max - train_price_return_min
        
        if price_return_range > 0:
            # Normalize to [0, 1] using training statistics
            y_pred_proba = (y_pred_price_return - train_price_return_min) / price_return_range
            # Clip to [0, 1] in case predictions are outside training range
            y_pred_proba = np.clip(y_pred_proba, 0, 1)
        else:
            # If all training returns are the same, use a simple transformation
            y_pred_proba = (y_pred_price_return - train_price_return_min) + 0.5
            y_pred_proba = np.clip(y_pred_proba, 0, 1)
        
        # Add predictions to test dataframe
        df_test = df_test.copy()
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
            # print predicted price_return stats 
            print(f"     Predicted price_return stats: {invest_record['y_pred_proba'].describe()}") 
    
    # benchmark performance: market average short-term return and 6-month return
    # Use the full dataset as benchmark (comparing overall market performance)
    if len(df) > 0:
        print(f"\n{'='*60}")
        print("📊 BENCHMARK: Overall Market Performance")
        print(f"{'='*60}")
        benchmark_performance(df[(df['year_month'] > INVEST_EXP_START_MONTH_STR) 
                               & (df['year_month'] <= INVEST_EXP_END_MONTH_STR)
                               ], num_months=6)


if __name__ == "__main__":
    main()
    # invest_monthly_retro_w_regression()

