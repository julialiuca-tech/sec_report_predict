#!/usr/bin/env python3
"""
Centralized configuration file for SEC data exploration project.

This file contains all global constants, default values, and configuration parameters
used across the project to ensure consistency and easy maintenance.

Note: Module-specific constants (like Yahoo Finance API parameters and quarter mappings)
are defined in their respective modules rather than here.
"""

import os

# =============================================================================
# PROJECT PATHS AND DIRECTORIES
# =============================================================================

# Base project directory (automatically detected)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
SAVE_DIR = os.path.join(DATA_DIR, 'featurized_2015_to_2025')
DATA_BASE_DIR = os.path.join(DATA_DIR, 'SEC_raw_2015_to_2025')

# stock data 
STOCK_DIR = os.path.join(DATA_DIR, 'stock_Stooq_daily_US', 'derived_data')

# model related data
MODEL_DIR = os.path.join(DATA_DIR, 'model_related') 

# =============================================================================
# FILE NAMES AND PATTERNS
# =============================================================================

# Featurized data files
FEATURIZED_ALL_QUARTERS_FILE = os.path.join(SAVE_DIR, 'featurized_all_quarters.csv')
FEATURE_COMPLETENESS_RANKING_FILE = os.path.join(SAVE_DIR, 'feature_completeness_ranking.csv')
FEATURIZED_SIMPLIFIED_FILE = os.path.join(SAVE_DIR, 'featurized_simplified.csv')

# Quarter file naming pattern
QUARTER_FEATURIZED_PATTERN = os.path.join(SAVE_DIR, '{}_featurized.csv')  # Format: 2022q1_featurized.csv

# SEC data files
COMPANY_TICKERS_EXCHANGE_FILE = os.path.join(DATA_DIR, 'company_tickers_exchange.json')

# =============================================================================
# FEATURIZATION PARAMETERS
# =============================================================================

# Default parameters for featurization
DEFAULT_K_TOP_TAGS = 250
DEFAULT_MIN_COMPLETENESS = 15.0
DEFAULT_DEBUG_FLAG = True
DEFAULT_N_QUARTERS_HISTORY_COMP = 0  # 0 means no history comparisons, 4 for past 1 year comparison


TREND_HORIZON_IN_MONTHS= 3
STOCK_TREND_DATA_FILE = os.path.join(STOCK_DIR, f'price_trends_{TREND_HORIZON_IN_MONTHS}month.csv')

# parameters for feature augmentation  

# =============================================================================
# MACHINE LEARNING PARAMETERS
# =============================================================================
USE_RATIO_FEATURES = True
FILTER_OUTLIERS_FROM_RATIOS = True
SUFFIXES_TO_ENHANCE_W_GRADIENT = ['_current', '_augment']
# Feature suffixes to use for ML model training (based on USE_RATIO_FEATURES flag)
FEATURE_SUFFIXES = ['_current', '_augment'] if USE_RATIO_FEATURES else ['_current']
QUARTER_GRADIENTS = [1, 2, 4]
TOP_K_FEATURES = 200
FEATURE_IMPORTANCE_RANKING_FLAG = False  # If True, retrain model with top K features based on importance

# Model training parameters
COMPLETENESS_THRESHOLD = 0.2
Y_LABEL = 'trend_up_or_down'  # can be 'trend_up_or_down' or 'trend_5per_up'
SPLIT_STRATEGY = {'period':'bottom'} 
# SPLIT_STRATEGY = None
 

# =============================================================================
# VALIDATION AND DEBUGGING
# =============================================================================

def validate_config():
    """
    Validate that all required directories exist and create them if needed.
    """
    directories_to_check = [DATA_DIR, SAVE_DIR, DATA_BASE_DIR, STOCK_DIR, MODEL_DIR]
    
    for directory in directories_to_check:
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)
    
    print("✅ Configuration validated successfully!")

if __name__ == "__main__":
    validate_config()
