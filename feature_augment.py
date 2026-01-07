"""
Feature augmentation module to compute ratio features from base financial metrics.

This module adds computed ratio features (profitability, liquidity, leverage, efficiency)
to featurized dataframes, as outlined in recommendation.md.

All generated features use the "_augment" suffix to distinguish them from original features.

Typical/Normal Ranges for Financial Ratios:
==========================================

PROFITABILITY RATIOS:
--------------------
- GrossMargin_1qtrs: Varies by industry (typically 20-50% for most industries)
  * Retail: 30-40%, Technology: 50-70%, Manufacturing: 20-35%
  
- OperatingMargin_1qtrs: Varies by industry (typically 5-15% for healthy companies)
  * 10-15% is considered good for many industries
  * Can be negative (operating losses)
  
- NetMargin_1qtrs: Typically 5-15% for healthy companies
  * 10% or higher is considered strong
  * Can be negative (net losses)
  
- ROE_1qtrs (Return on Equity): 15-20% is considered strong
  * 10-15% is acceptable for many industries
  * Can be negative (losses or negative equity)
  
- ROA_1qtrs (Return on Assets): 5% or higher is generally considered good
  * Can be negative (losses)

LIQUIDITY RATIOS:
----------------
- CurrentRatio_0qtrs: Typical range 1.5 to 3.0
  * Below 1.0 indicates potential liquidity issues
  * Above 3.0 may suggest inefficient use of assets
  
- QuickRatio_0qtrs: Typical range 1.0 to 2.0
  * Below 1.0 might indicate potential liquidity problems
  * Below 0.5 may indicate risk of running out of working capital
  
- CashRatio_0qtrs: Typically 0.2 to 0.5 for healthy companies
  * Lower than quick ratio (excludes receivables)
  * Higher values indicate stronger cash position

LEVERAGE RATIOS:
---------------
- DebtToEquity_0qtrs: Below 2.0 is generally preferred
  * Varies significantly by industry (capital-intensive industries may have higher acceptable ratios)
  * Can be negative if equity is negative
  
- DebtToAssets_0qtrs: Lower is better, typically 0.3 to 0.6 for healthy companies
  * Higher ratios indicate greater financial risk
  * Should be between 0 and 1 (debt cannot exceed assets)
  
- InterestCoverage_1qtrs: Higher is better, typically 2.0 or higher
  * Below 1.0 indicates inability to cover interest payments
  * Can be negative (operating losses)

EFFICIENCY RATIOS:
-----------------
- AssetTurnover_1qtrs: Greater than 0.45 (45%) is considered strong
  * 0.30-0.45 is in the caution range
  * Varies significantly by industry
  
- InventoryTurnover_1qtrs: Varies significantly by industry
  * Retail: 4-6 times per year, Manufacturing: 6-12 times per year
  * Higher ratios generally indicate efficient inventory management
  
- ReceivablesTurnover_1qtrs: Typically 10-12 times per year (30-36 days)
  * Higher ratios indicate faster collection of receivables
  * Varies by industry and payment terms

Note: All ranges are general guidelines and can vary significantly by industry, 
company size, and economic conditions. Industry-specific benchmarks should be 
consulted for more accurate assessments.
"""

import pandas as pd
import numpy as np
import os
from typing import Optional
from config import FEATURIZED_SIMPLIFIED_FILE


def _find_feature_column(df: pd.DataFrame, tag: str, qtrs: int, suffix: str = '_current') -> Optional[str]:
    """
    Find the column name matching a tag and quarter pattern.
    
    Args:
        df: DataFrame with feature columns
        tag: Tag name (e.g., 'GrossProfit', 'Revenues')
        qtrs: Number of quarters (e.g., 0, 1, 4)
        suffix: Column suffix (default: '_current')
    
    Returns:
        Column name if found, None otherwise
    """
    qtrs_str = f'{qtrs}qtrs'
    expected_col = f'{tag}_{qtrs_str}{suffix}'
    
    # Try exact match first
    if expected_col in df.columns:
        return expected_col
    
    # Try case-insensitive match
    for col in df.columns:
        if col.lower() == expected_col.lower():
            return col
    
    return None


def _find_feature_column_flexible(df: pd.DataFrame, tag: str, suffix: str = '_current') -> Optional[str]:
    """
    Find a feature column for a tag by trying common qtrs values (1, 0).
    
    Args:
        df: DataFrame with feature columns
        tag: Tag name (e.g., 'GrossProfit', 'Revenues')
        suffix: Column suffix (default: '_current')
    
    Returns:
        Column name if found, None otherwise
    """
    # Try common qtrs values in order of likelihood (1qtrs for quarterly, 0qtrs for point-in-time)
    for qtrs in [1, 0]:
        col = _find_feature_column(df, tag, qtrs, suffix)
        if col:
            return col
    return None


def _safe_divide(numerator: pd.Series, denominator: pd.Series, fill_value: np.float64 = np.nan) -> pd.Series:
    """
    Safely divide two series, handling division by zero and missing values.
    
    Args:
        numerator: Numerator series
        denominator: Denominator series
        fill_value: Value to use for division by zero or invalid results
    
    Returns:
        Series with division results
    """
    # Handle zero and negative denominators
    mask = (denominator != 0) & denominator.notna() & numerator.notna()
    result = pd.Series(fill_value, index=numerator.index)
    result[mask] = numerator[mask] / denominator[mask]
    
    # Handle negative denominators by taking absolute value or keeping as NaN
    # For now, we'll keep negative denominators as NaN (can be customized)
    
    return result


def compute_ratio_features(df: pd.DataFrame, debug_print: bool = False) -> pd.DataFrame:
    """
    Compute ratio features from base financial metrics in the dataframe.
    
    This function computes profitability, liquidity, leverage, and efficiency ratios
    as outlined in recommendation.md. All generated features use the "_augment" suffix.
    
    Args:
        df: Featurized dataframe with columns like 'TagName_qtrs_current'
        debug_print: If True, print debug information about missing features
    
    Returns:
        DataFrame with original columns plus new ratio features (suffixed with '_augment')
    
    Ratio Features Computed:
        Profitability:
        - GrossMargin_1qtrs_augment
        - OperatingMargin_1qtrs_augment
        - NetMargin_1qtrs_augment
        - ROE_1qtrs_augment
        - ROA_1qtrs_augment
        
        Liquidity:
        - CurrentRatio_0qtrs_augment
        - QuickRatio_0qtrs_augment
        - CashRatio_0qtrs_augment
        
        Leverage:
        - DebtToEquity_0qtrs_augment
        - DebtToAssets_0qtrs_augment
        - InterestCoverage_1qtrs_augment
        
        Efficiency:
        - AssetTurnover_1qtrs_augment
        - InventoryTurnover_1qtrs_augment
        - ReceivablesTurnover_1qtrs_augment
    """
    df_result = df.copy()
    
    if debug_print:
        print("=" * 80)
        print("COMPUTING RATIO FEATURES (AUGMENTED)")
        print("=" * 80)
        print(f"Input dataframe shape: {df_result.shape}")
    
    # =============================================================================
    # PROFITABILITY RATIOS
    # =============================================================================
    if debug_print:
        print("\n📊 Computing Profitability Ratios...")
    
    # GrossMargin_1qtrs = GrossProfit_1qtrs_current / Revenues_1qtrs_current
    gross_profit_col = _find_feature_column(df_result, 'GrossProfit', 1)
    revenues_col = _find_feature_column(df_result, 'Revenues', 1)
    if gross_profit_col and revenues_col:
        df_result['GrossMargin_1qtrs_augment'] = _safe_divide(
            df_result[gross_profit_col], df_result[revenues_col]
        )
    elif debug_print:
        print(f"  ⚠️  Skipping GrossMargin: missing {gross_profit_col or 'GrossProfit_1qtrs_current'} or {revenues_col or 'Revenues_1qtrs_current'}")
    
    # OperatingMargin_1qtrs = OperatingIncomeLoss_1qtrs_current / Revenues_1qtrs_current
    operating_income_col = _find_feature_column(df_result, 'OperatingIncomeLoss', 1)
    if operating_income_col and revenues_col:
        df_result['OperatingMargin_1qtrs_augment'] = _safe_divide(
            df_result[operating_income_col], df_result[revenues_col]
        )
    elif debug_print:
        print(f"  ⚠️  Skipping OperatingMargin: missing {operating_income_col or 'OperatingIncomeLoss_1qtrs_current'} or {revenues_col or 'Revenues_1qtrs_current'}")
    
    # NetMargin_1qtrs = NetIncomeLoss_1qtrs_current / Revenues_1qtrs_current
    net_income_col = _find_feature_column(df_result, 'NetIncomeLoss', 1)
    if net_income_col and revenues_col:
        df_result['NetMargin_1qtrs_augment'] = _safe_divide(
            df_result[net_income_col], df_result[revenues_col]
        )
    elif debug_print:
        print(f"  ⚠️  Skipping NetMargin: missing {net_income_col or 'NetIncomeLoss_1qtrs_current'} or {revenues_col or 'Revenues_1qtrs_current'}")
    
    # ROE_1qtrs = NetIncomeLoss_1qtrs_current / StockholdersEquity_0qtrs_current
    equity_col = _find_feature_column(df_result, 'StockholdersEquity', 0)
    if net_income_col and equity_col:
        df_result['ROE_1qtrs_augment'] = _safe_divide(
            df_result[net_income_col], df_result[equity_col]
        )
    elif debug_print:
        print(f"  ⚠️  Skipping ROE: missing {net_income_col or 'NetIncomeLoss_1qtrs_current'} or {equity_col or 'StockholdersEquity_0qtrs_current'}")
    
    # ROA_1qtrs = NetIncomeLoss_1qtrs_current / Assets_0qtrs_current
    assets_col = _find_feature_column(df_result, 'Assets', 0)
    if net_income_col and assets_col:
        df_result['ROA_1qtrs_augment'] = _safe_divide(
            df_result[net_income_col], df_result[assets_col]
        )
    elif debug_print:
        print(f"  ⚠️  Skipping ROA: missing {net_income_col or 'NetIncomeLoss_1qtrs_current'} or {assets_col or 'Assets_0qtrs_current'}")
    
    # =============================================================================
    # LIQUIDITY RATIOS
    # =============================================================================
    if debug_print:
        print("\n💧 Computing Liquidity Ratios...")
    
    # CurrentRatio_0qtrs = AssetsCurrent_0qtrs_current / LiabilitiesCurrent_0qtrs_current
    assets_current_col = _find_feature_column(df_result, 'AssetsCurrent', 0)
    liabilities_current_col = _find_feature_column(df_result, 'LiabilitiesCurrent', 0)
    if assets_current_col and liabilities_current_col:
        df_result['CurrentRatio_0qtrs_augment'] = _safe_divide(
            df_result[assets_current_col], df_result[liabilities_current_col]
        )
    elif debug_print:
        print(f"  ⚠️  Skipping CurrentRatio: missing {assets_current_col or 'AssetsCurrent_0qtrs_current'} or {liabilities_current_col or 'LiabilitiesCurrent_0qtrs_current'}")
    
    # QuickRatio_0qtrs = (AssetsCurrent_0qtrs_current - InventoryNet_0qtrs_current) / LiabilitiesCurrent_0qtrs_current
    inventory_col = _find_feature_column(df_result, 'InventoryNet', 0)
    if assets_current_col and inventory_col and liabilities_current_col:
        numerator = df_result[assets_current_col] - df_result[inventory_col]
        df_result['QuickRatio_0qtrs_augment'] = _safe_divide(
            numerator, df_result[liabilities_current_col]
        )
    elif debug_print:
        print(f"  ⚠️  Skipping QuickRatio: missing required columns")
    
    # CashRatio_0qtrs = CashAndCashEquivalentsAtCarryingValue_0qtrs_current / LiabilitiesCurrent_0qtrs_current
    cash_col = _find_feature_column(df_result, 'CashAndCashEquivalentsAtCarryingValue', 0)
    if cash_col and liabilities_current_col:
        df_result['CashRatio_0qtrs_augment'] = _safe_divide(
            df_result[cash_col], df_result[liabilities_current_col]
        )
    elif debug_print:
        print(f"  ⚠️  Skipping CashRatio: missing {cash_col or 'CashAndCashEquivalentsAtCarryingValue_0qtrs_current'} or {liabilities_current_col or 'LiabilitiesCurrent_0qtrs_current'}")
    
    # =============================================================================
    # LEVERAGE RATIOS
    # =============================================================================
    if debug_print:
        print("\n⚖️  Computing Leverage Ratios...")
    
    # DebtToEquity_0qtrs = (LongTermDebtNoncurrent_0qtrs_current + LiabilitiesCurrent_0qtrs_current) / StockholdersEquity_0qtrs_current
    long_term_debt_col = _find_feature_column(df_result, 'LongTermDebtNoncurrent', 0)
    if long_term_debt_col and liabilities_current_col and equity_col:
        total_debt = df_result[long_term_debt_col] + df_result[liabilities_current_col]
        df_result['DebtToEquity_0qtrs_augment'] = _safe_divide(
            total_debt, df_result[equity_col]
        )
    elif debug_print:
        print(f"  ⚠️  Skipping DebtToEquity: missing required columns")
    
    # DebtToAssets_0qtrs = (LongTermDebtNoncurrent_0qtrs_current + LiabilitiesCurrent_0qtrs_current) / Assets_0qtrs_current
    if long_term_debt_col and liabilities_current_col and assets_col:
        total_debt = df_result[long_term_debt_col] + df_result[liabilities_current_col]
        df_result['DebtToAssets_0qtrs_augment'] = _safe_divide(
            total_debt, df_result[assets_col]
        )
    elif debug_print:
        print(f"  ⚠️  Skipping DebtToAssets: missing required columns")
    
    # InterestCoverage_1qtrs = OperatingIncomeLoss_1qtrs_current / InterestExpense_1qtrs_current
    interest_expense_col = _find_feature_column(df_result, 'InterestExpense', 1)
    if operating_income_col and interest_expense_col:
        df_result['InterestCoverage_1qtrs_augment'] = _safe_divide(
            df_result[operating_income_col], df_result[interest_expense_col]
        )
    elif debug_print:
        print(f"  ⚠️  Skipping InterestCoverage: missing {operating_income_col or 'OperatingIncomeLoss_1qtrs_current'} or {interest_expense_col or 'InterestExpense_1qtrs_current'}")
    
    # =============================================================================
    # EFFICIENCY RATIOS
    # =============================================================================
    if debug_print:
        print("\n🔄 Computing Efficiency Ratios...")
    
    # AssetTurnover_1qtrs = Revenues_1qtrs_current / Assets_0qtrs_current
    if revenues_col and assets_col:
        df_result['AssetTurnover_1qtrs_augment'] = _safe_divide(
            df_result[revenues_col], df_result[assets_col]
        )
    elif debug_print:
        print(f"  ⚠️  Skipping AssetTurnover: missing {revenues_col or 'Revenues_1qtrs_current'} or {assets_col or 'Assets_0qtrs_current'}")
    
    # InventoryTurnover_1qtrs = CostOfGoodsAndServicesSold_1qtrs_current / InventoryNet_0qtrs_current
    cogs_col = _find_feature_column(df_result, 'CostOfGoodsAndServicesSold', 1)
    inventory_net_col = _find_feature_column(df_result, 'InventoryNet', 0)
    if cogs_col and inventory_net_col:
        df_result['InventoryTurnover_1qtrs_augment'] = _safe_divide(
            df_result[cogs_col], df_result[inventory_net_col]
        )
    elif debug_print:
        print(f"  ⚠️  Skipping InventoryTurnover: missing {cogs_col or 'CostOfGoodsAndServicesSold_1qtrs_current'} or {inventory_net_col or 'InventoryNet_0qtrs_current'}")
    
    # ReceivablesTurnover_1qtrs = Revenues_1qtrs_current / AccountsReceivableNetCurrent_0qtrs_current
    receivables_col = _find_feature_column(df_result, 'AccountsReceivableNetCurrent', 0)
    if revenues_col and receivables_col:
        df_result['ReceivablesTurnover_1qtrs_augment'] = _safe_divide(
            df_result[revenues_col], df_result[receivables_col]
        )
    elif debug_print:
        print(f"  ⚠️  Skipping ReceivablesTurnover: missing {revenues_col or 'Revenues_1qtrs_current'} or {receivables_col or 'AccountsReceivableNetCurrent_0qtrs_current'}")
    
    # Count augmented features created
    augment_features = [col for col in df_result.columns if col.endswith('_augment')]
    
    if debug_print:
        print(f"\n✅ Created {len(augment_features)} augmented ratio features")
        print(f"Output dataframe shape: {df_result.shape}")
        print("\n📋 Augmented features created:")
        for feat in sorted(augment_features):
            non_null_count = df_result[feat].notna().sum()
            non_null_pct = (non_null_count / len(df_result)) * 100.0
            print(f"  - {feat}: {non_null_count:,} non-null values ({non_null_pct:.1f}%)")
            
            # Print statistics and sample values
            if non_null_count > 0:
                mean_val = df_result[feat].mean()
                median_val = df_result[feat].median()
                std_val = df_result[feat].std()
                sample_vals = df_result[feat].dropna().head(3).tolist()
                sample_str = ", ".join([f"{v:.4f}" for v in sample_vals])
                print(f"      Mean: {mean_val:.4f}, Median: {median_val:.4f}, Std: {std_val:.4f}")
                print(f"      Sample: [{sample_str}]")
    
    return df_result


def compute_ratio_features_on_ratio_definitions(df: pd.DataFrame, ratio_definitions: dict, 
                                                  debug_print: bool = False) -> pd.DataFrame:
    """
    Compute ratio features dynamically from ratio definitions.
    
    This function takes a ratio_definitions dictionary and computes ratios accordingly.
    Unlike compute_ratio_features(), this function is not hard-wired and can handle
    any ratio definitions provided.
    
    Args:
        df: Featurized dataframe with columns like 'TagName_qtrs_current'
        ratio_definitions: Dictionary mapping ratio names to (numerator, denominator) tuples
                         where numerator can be a single tag string or tuple of tag strings,
                         and denominator is a single tag string
                         Example: {'GrossMargin': ('GrossProfit', 'Revenues'),
                                  'QuickRatio': (('AssetsCurrent', 'InventoryNet'), 'LiabilitiesCurrent')}
        debug_print: If True, print debug information about missing features
    
    Returns:
        DataFrame with original columns plus new ratio features (suffixed with '_augment')
    """
    df_result = df.copy()
    
    if debug_print:
        print("=" * 80)
        print("COMPUTING RATIO FEATURES FROM RATIO DEFINITIONS (AUGMENTED)")
        print("=" * 80)
        print(f"Input dataframe shape: {df_result.shape}")
        print(f"Number of ratios to compute: {len(ratio_definitions)}")
    
    ratios_computed = []
    ratios_skipped = []
    
    for ratio_name, (num_component, denom_component) in ratio_definitions.items():
        # Find denominator column
        denom_col = _find_feature_column_flexible(df_result, denom_component)
        if not denom_col:
            if debug_print:
                print(f"  ⚠️  Skipping {ratio_name}: denominator '{denom_component}' not found")
            ratios_skipped.append(ratio_name)
            continue
        
        # Handle numerator component (can be single tag or tuple)
        if isinstance(num_component, tuple):
            # Tuple numerator (e.g., QuickRatio: (AssetsCurrent - InventoryNet))
            num_cols = []
            for comp_tag in num_component:
                comp_col = _find_feature_column_flexible(df_result, comp_tag)
                if not comp_col:
                    if debug_print:
                        print(f"  ⚠️  Skipping {ratio_name}: numerator component '{comp_tag}' not found")
                    ratios_skipped.append(ratio_name)
                    break
                num_cols.append(comp_col)
            else:
                # All numerator components found, compute numerator
                if len(num_cols) == 1:
                    numerator = df_result[num_cols[0]]
                else:
                    # Multiple components: subtract them (e.g., AssetsCurrent - InventoryNet)
                    numerator = df_result[num_cols[0]]
                    for col in num_cols[1:]:
                        numerator = numerator - df_result[col]
                
                # Compute ratio
                ratio_col_name = f'{ratio_name}_augment'
                df_result[ratio_col_name] = _safe_divide(numerator, df_result[denom_col])
                ratios_computed.append(ratio_name)
        else:
            # Single numerator component
            num_col = _find_feature_column_flexible(df_result, num_component)
            if not num_col:
                if debug_print:
                    print(f"  ⚠️  Skipping {ratio_name}: numerator '{num_component}' not found")
                ratios_skipped.append(ratio_name)
                continue
            
            # Compute ratio
            ratio_col_name = f'{ratio_name}_augment'
            df_result[ratio_col_name] = _safe_divide(df_result[num_col], df_result[denom_col])
            ratios_computed.append(ratio_name)
    
    # Count augmented features created
    augment_features = [col for col in df_result.columns if col.endswith('_augment')]
    
    if debug_print:
        print(f"\n✅ Computed {len(ratios_computed)} ratios successfully")
        if ratios_skipped:
            print(f"⚠️  Skipped {len(ratios_skipped)} ratios due to missing components")
        print(f"Output dataframe shape: {df_result.shape}")
        print(f"\n📋 Ratio computation summary:")
        print(f"  - Computed: {', '.join(ratios_computed)}")
        if ratios_skipped:
            print(f"  - Skipped: {', '.join(ratios_skipped)}")
    
    return df_result


def flag_outliers_by_hard_limits(df_feature: pd.DataFrame) -> pd.DataFrame:
    """
    Check ratio bounds and add a single boolean column documenting any limit violations.
    
    This function adds a validation column to identify ratio values that violate
    expected bounds based on financial logic:
    - Ratios that should be ≤ 1.0: GrossMargin, OperatingMargin (if positive), 
      NetMargin (if positive), DebtToAssets
    - Ratios that should be ≥ 0.0: All liquidity and efficiency ratios, plus
      GrossMargin and DebtToAssets
    
    Args:
        df_feature: Featurized dataframe with ratio features (ending in '_augment')
    
    Returns:
        DataFrame with additional boolean column:
        - `flag_outlier`: True if ANY ratio violates its bound limits
    """
    df_result = df_feature.copy()
    
    # Initialize flag_outlier column
    df_result['flag_outlier'] = False
    
    # Ratios that should be ≤ 1.0
    ratios_max_1 = [
        'GrossMargin_1qtrs_augment',
        'OperatingMargin_1qtrs_augment',
        'NetMargin_1qtrs_augment',
        'DebtToAssets_0qtrs_augment',
    ]
    
    # Ratios that should be ≥ 0.0
    ratios_min_0 = [
        'GrossMargin_1qtrs_augment',
        'CurrentRatio_0qtrs_augment',
        'QuickRatio_0qtrs_augment',
        'CashRatio_0qtrs_augment',
        'DebtToAssets_0qtrs_augment',
        'AssetTurnover_1qtrs_augment',
        'InventoryTurnover_1qtrs_augment',
        'ReceivablesTurnover_1qtrs_augment',
    ]
    
    # Check for violations and mark in the flag_outlier column
    for ratio_col in ratios_max_1:
        if ratio_col in df_result.columns:
            # Check if exceeds 1.0
            exceeds_1_mask = df_result[ratio_col] > 1.0
            df_result.loc[exceeds_1_mask, 'flag_outlier'] = True
    
    for ratio_col in ratios_min_0:
        if ratio_col in df_result.columns:
            # Check if negative
            negative_mask = df_result[ratio_col] < 0.0
            df_result.loc[negative_mask, 'flag_outlier'] = True
    
    # Print violation summary
    violation_count = df_result['flag_outlier'].sum()
    if violation_count > 0:
        violation_pct = (violation_count / len(df_result)) * 100.0
        print(f"\n🔍 Ratio bound violations:")
        print(f"   - Records with any bound violation: {violation_count:,} ({violation_pct:.2f}%)")
    
    return df_result


def flag_outliers_by_expected_range(df_feature: pd.DataFrame, debug_print: bool = True) -> pd.DataFrame:
    """
    Check ratio values against expected normal ranges and identify anomalies.
    
    This function analyzes each ratio feature and identifies:
    - Values outside expected typical ranges
    - Extreme outliers (using IQR method)
    - Statistical anomalies (mean vs median discrepancies)
    
    Args:
        df_feature: Featurized dataframe with ratio features (ending in '_augment')
        debug_print: If True, print detailed anomaly report
    
    Returns:
        DataFrame with anomaly flags and summary statistics
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    augment_features = [col for col in df_feature.columns if col.endswith('_augment')]
    
    if not augment_features:
        if debug_print:
            print("⚠️  No augmented ratio features found to analyze")
        return df_feature
    
    if debug_print:
        print("\n" + "=" * 80)
        print("RATIO ANOMALY ANALYSIS")
        print("=" * 80)
    
    # Define expected ranges for each ratio (based on RATIO_NORMAL_RANGES.md)
    expected_ranges = {
        # Profitability ratios (as decimals)
        'GrossMargin_1qtrs_augment': {'typical_min': 0.20, 'typical_max': 0.70, 'must_be_leq': 1.0},
        'OperatingMargin_1qtrs_augment': {'typical_min': 0.05, 'typical_max': 0.15, 'can_be_negative': True},
        'NetMargin_1qtrs_augment': {'typical_min': 0.05, 'typical_max': 0.15, 'can_be_negative': True},
        'ROE_1qtrs_augment': {'typical_min': 0.10, 'typical_max': 0.20, 'can_be_negative': True},
        'ROA_1qtrs_augment': {'typical_min': 0.05, 'typical_max': None, 'can_be_negative': True},
        
        # Liquidity ratios
        'CurrentRatio_0qtrs_augment': {'typical_min': 1.5, 'typical_max': 3.0, 'must_be_ge': 0.0},
        'QuickRatio_0qtrs_augment': {'typical_min': 1.0, 'typical_max': 2.0, 'must_be_ge': 0.0},
        'CashRatio_0qtrs_augment': {'typical_min': 0.2, 'typical_max': 0.5, 'must_be_ge': 0.0},
        
        # Leverage ratios
        'DebtToEquity_0qtrs_augment': {'typical_max': 2.0, 'can_be_negative': True},
        'DebtToAssets_0qtrs_augment': {'typical_min': 0.3, 'typical_max': 0.6, 'must_be_leq': 1.0, 'must_be_ge': 0.0},
        'InterestCoverage_1qtrs_augment': {'typical_min': 2.0, 'typical_max': None, 'can_be_negative': True},
        
        # Efficiency ratios
        'AssetTurnover_1qtrs_augment': {'typical_min': 0.30, 'typical_max': None, 'must_be_ge': 0.0},
        'InventoryTurnover_1qtrs_augment': {'typical_min': 4.0, 'typical_max': 12.0, 'must_be_ge': 0.0},
        'ReceivablesTurnover_1qtrs_augment': {'typical_min': 10.0, 'typical_max': 12.0, 'must_be_ge': 0.0},
    }
    
    # Initialize result dataframe
    df_result = df_feature.copy()
    if 'flag_outlier' not in df_result.columns:
        df_result['flag_outlier'] = False
    
    # STEP 1: Generate individual outlier flag columns for each dimension
    individual_flag_columns = []
    
    for feat in sorted(augment_features):
        if feat not in df_result.columns:
            continue
        
        expected = expected_ranges.get(feat, {})
        
        # Dimension 1: Below typical minimum
        if 'typical_min' in expected:
            col_name = f'{feat}_outlier_below_typical_min'
            df_result[col_name] = df_result[feat] < expected['typical_min']
            individual_flag_columns.append(col_name)
        
        # Dimension 2: Above typical maximum
        if 'typical_max' in expected and expected['typical_max'] is not None:
            col_name = f'{feat}_outlier_above_typical_max'
            df_result[col_name] = df_result[feat] > expected['typical_max']
            individual_flag_columns.append(col_name)
        
        # Dimension 3: Exceeds upper logical bound (must_be_leq)
        if 'must_be_leq' in expected:
            col_name = f'{feat}_outlier_exceeds_upper_bound'
            df_result[col_name] = df_result[feat] > expected['must_be_leq']
            individual_flag_columns.append(col_name)
        
        # Dimension 4: Below lower logical bound (must_be_ge)
        if 'must_be_ge' in expected:
            col_name = f'{feat}_outlier_below_lower_bound'
            df_result[col_name] = df_result[feat] < expected['must_be_ge']
            individual_flag_columns.append(col_name)
        
        # Dimension 5: Extreme outliers using IQR
        non_null_mask = df_result[feat].notna()
        if non_null_mask.sum() > 0:
            non_null_values = df_result.loc[non_null_mask, feat]
            Q1 = non_null_values.quantile(0.25)
            Q3 = non_null_values.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                col_name = f'{feat}_outlier_extreme_iqr'
                df_result[col_name] = (df_result[feat] < lower_bound) | (df_result[feat] > upper_bound)
                individual_flag_columns.append(col_name)
    
    # STEP 2: Generate final flag_outlier column (ANY of individual flags)
    if individual_flag_columns:
        df_result['flag_outlier'] = df_result[individual_flag_columns].any(axis=1) | df_result.get('flag_outlier', False)
    
    # STEP 3: Print outlier stats for each dimension
    if debug_print:
        print(f"\n{'='*80}")
        print("RATIO ANOMALY ANALYSIS")
        print(f"{'='*80}")
        
        for feat in sorted(augment_features):
            if feat not in df_result.columns:
                continue
            
            values = df_result[feat].dropna()
            if len(values) == 0:
                continue
            
            mean_val = values.mean()
            median_val = values.median()
            expected = expected_ranges.get(feat, {})
            
            anomalies = []
            
            # Report stats from individual flag columns
            if f'{feat}_outlier_below_typical_min' in df_result.columns:
                count = df_result[f'{feat}_outlier_below_typical_min'].sum()
                if count > 0:
                    pct = (count / len(values)) * 100
                    anomalies.append(f"{pct:.1f}% of values below typical minimum ({expected.get('typical_min')})")
            
            if f'{feat}_outlier_above_typical_max' in df_result.columns:
                count = df_result[f'{feat}_outlier_above_typical_max'].sum()
                if count > 0:
                    pct = (count / len(values)) * 100
                    anomalies.append(f"{pct:.1f}% of values above typical maximum ({expected.get('typical_max')})")
            
            if f'{feat}_outlier_exceeds_upper_bound' in df_result.columns:
                count = df_result[f'{feat}_outlier_exceeds_upper_bound'].sum()
                if count > 0:
                    pct = (count / len(values)) * 100
                    anomalies.append(f"⚠️  {pct:.1f}% exceed logical bound of {expected.get('must_be_leq')} ({count} values)")
            
            if f'{feat}_outlier_below_lower_bound' in df_result.columns:
                count = df_result[f'{feat}_outlier_below_lower_bound'].sum()
                if count > 0:
                    pct = (count / len(values)) * 100
                    anomalies.append(f"⚠️  {pct:.1f}% below logical bound of {expected.get('must_be_ge')} ({count} values)")
            
            if f'{feat}_outlier_extreme_iqr' in df_result.columns:
                count = df_result[f'{feat}_outlier_extreme_iqr'].sum()
                if count > 0:
                    # Get outlier values directly from flag column
                    outlier_mask = df_result[f'{feat}_outlier_extreme_iqr']
                    outlier_values = df_result.loc[outlier_mask, feat].dropna()
                    pct = (count / len(values)) * 100
                    max_outlier = outlier_values.max() if len(outlier_values) > 0 else None
                    min_outlier = outlier_values.min() if len(outlier_values) > 0 else None
                    anomalies.append(f"⚠️  {pct:.1f}% extreme outliers (using 3x IQR): min={min_outlier:.2f}, max={max_outlier:.2f}")
            
            # Mean vs Median discrepancy (for reporting only)
            if abs(mean_val - median_val) > 0.1 * abs(median_val) if median_val != 0 else False:
                mean_median_ratio = abs(mean_val / median_val) if median_val != 0 else float('inf')
                if mean_median_ratio > 10 or mean_median_ratio < 0.1:
                    anomalies.append(f"Mean ({mean_val:.2f}) differs significantly from median ({median_val:.4f}) - ratio: {mean_median_ratio:.1f}x")
            
            if anomalies:
                print(f"\n🔍 {feat}:")
                print(f"   Non-null: {len(values):,}, Mean: {mean_val:.4f}, Median: {median_val:.4f}")
                for anomaly in anomalies:
                    print(f"   - {anomaly}")
        
        # Summary statistics
        features_with_outliers = [feat for feat in augment_features 
                                 if feat in df_result.columns and 
                                 any(col.startswith(f'{feat}_outlier_') and df_result[col].any() 
                                     for col in df_result.columns if col.startswith(f'{feat}_outlier_'))]
        
        total_features = len([f for f in augment_features if f in df_result.columns])
        print(f"\n{'='*80}")
        print(f"SUMMARY: {len(features_with_outliers)} out of {total_features} ratios have anomalies")
        print(f"{'='*80}\n")
    
    # STEP 4: Remove individual flag columns, keep only final flag_outlier
    df_result = df_result.drop(columns=individual_flag_columns, errors='ignore')
    
    return df_result


def flag_outliers_by_extreme_stats(df_feature: pd.DataFrame, debug_print: bool = True) -> pd.DataFrame:
    """
    Analyze the impact of removing extreme outliers across all ratio features.
    
    This function identifies extreme outliers for each ratio (using 3x IQR method),
    then calculates how many records remain after removing ALL outliers (accounting
    for overlaps, since a single record may be an outlier in multiple ratios).
    
    Args:
        df_feature: Featurized dataframe with ratio features (ending in '_augment')
        debug_print: If True, print detailed analysis
    
    Returns:
        DataFrame with outlier flags and summary statistics
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    augment_features = [col for col in df_feature.columns if col.endswith('_augment')]
    
    if not augment_features:
        if debug_print:
            print("⚠️  No augmented ratio features found to analyze")
        return df_feature
    
    # Initialize result dataframe
    df_result = df_feature.copy()
    if 'flag_outlier' not in df_result.columns:
        df_result['flag_outlier'] = False
    
    # STEP 1: Generate individual outlier flag columns (one per feature for extreme IQR outliers)
    individual_flag_columns = []
    
    for feat in sorted(augment_features):
        if feat not in df_result.columns:
            continue
        
        non_null_mask = df_result[feat].notna()
        if non_null_mask.sum() == 0:
            continue
        
        values = df_result.loc[non_null_mask, feat]
        
        # Calculate IQR for extreme outlier detection (3x IQR)
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            col_name = f'{feat}_outlier_extreme_iqr'
            df_result[col_name] = (df_result[feat] < lower_bound) | (df_result[feat] > upper_bound)
            individual_flag_columns.append(col_name)
    
    # STEP 2: Generate final flag_outlier column (ANY of individual flags)
    if individual_flag_columns:
        df_result['flag_outlier'] = df_result[individual_flag_columns].any(axis=1) | df_result.get('flag_outlier', False)
    
    # STEP 3: Print outlier stats for each dimension
    if debug_print:
        print("\n" + "=" * 80)
        print("EXTREME OUTLIER REMOVAL IMPACT ANALYSIS")
        print("=" * 80)
        print(f"Total records: {len(df_result):,}")
        print(f"Analyzing {len(augment_features)} ratio features")
        
        # Compute statistics from individual flag columns
        feature_outlier_stats = []
        all_outlier_indices = set(df_result[df_result['flag_outlier']].index)
        
        for feat in sorted(augment_features):
            col_name = f'{feat}_outlier_extreme_iqr'
            if col_name not in df_result.columns:
                continue
            
            non_null_mask = df_result[feat].notna()
            if non_null_mask.sum() == 0:
                continue
            
            non_null_indices = df_result[non_null_mask].index
            values = df_result.loc[non_null_indices, feat]
            
            # Get outlier count from flag column
            outlier_count = df_result[col_name].sum()
            outlier_indices = set(df_result[df_result[col_name]].index)
            
            if outlier_count > 0:
                # Calculate bounds for reporting
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                feature_outlier_stats.append({
                    'feature': feat,
                    'count': outlier_count,
                    'percentage': (outlier_count / len(non_null_indices)) * 100,
                    'non_null_count': len(non_null_indices),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outlier_indices': outlier_indices
                })
        
        # Print summary by feature
        print(f"\n{'='*80}")
        print("OUTLIER SUMMARY BY FEATURE")
        print(f"{'='*80}")
        
        # Sort by outlier count (descending)
        feature_outlier_stats.sort(key=lambda x: x['count'], reverse=True)
        
        for stats in feature_outlier_stats:
            if stats['count'] > 0:
                print(f"\n{stats['feature']}:")
                print(f"  - Outliers: {stats['count']:,} ({stats['percentage']:.2f}% of non-null)")
                print(f"  - Non-null values: {stats['non_null_count']:,}")
                print(f"  - Bounds: [{stats['lower_bound']:.4f}, {stats['upper_bound']:.4f}]")
        
        # Calculate overall impact
        total_records = len(df_result)
        total_outlier_records = len(all_outlier_indices)
        records_remaining = total_records - total_outlier_records
        pct_remaining = (records_remaining / total_records) * 100.0 if total_records > 0 else 0
        pct_removed = (total_outlier_records / total_records) * 100.0 if total_records > 0 else 0
        
        print(f"\n{'='*80}")
        print("OVERALL IMPACT (ACCOUNTING FOR OVERLAPS)")
        print(f"{'='*80}")
        print(f"Total records in dataset: {total_records:,}")
        print(f"Records with extreme outliers (any ratio): {total_outlier_records:,} ({pct_removed:.2f}%)")
        print(f"Records remaining after removal: {records_remaining:,} ({pct_remaining:.2f}%)")
        
        # Calculate overlap statistics
        if feature_outlier_stats:
            total_if_no_overlap = sum(stats['count'] for stats in feature_outlier_stats)
            overlap_count = total_if_no_overlap - total_outlier_records
            if total_if_no_overlap > 0:
                overlap_pct = (overlap_count / total_if_no_overlap) * 100.0
                print(f"\nOverlap analysis:")
                print(f"  - Total outliers if no overlap: {total_if_no_overlap:,}")
                print(f"  - Actual unique outlier records: {total_outlier_records:,}")
                print(f"  - Overlap (records that are outliers in multiple ratios): {overlap_count:,} ({overlap_pct:.1f}%)")
        
        print(f"\n{'='*80}\n")
    
    # STEP 4: Remove individual flag columns, keep only final flag_outlier
    df_result = df_result.drop(columns=individual_flag_columns, errors='ignore')
    
    return df_result


def main():
    """
    Main function to test the feature augmentation module.
    
    This function:
    1. Loads a sample of featurized data
    2. Computes ratio features
    3. Shows before/after comparison
    4. Validates the results
    """
    # Load featurized data
    df_input = pd.read_csv(FEATURIZED_SIMPLIFIED_FILE, low_memory=False)
    if df_input is None:
        print("❌ No featurized data found. Generate it by running featurize.py")
        return
    
    # Compute ratio features
    try:
        df_output = compute_ratio_features(df_input, debug_print=True)
        augment_cols = [c for c in df_output.columns if c.endswith('_augment')]
        print(f"\n✅ Created {len(augment_cols)} augmented features")
        
        # Check bounds and violations
        df_with_validation = flag_outliers_by_hard_limits(df_output)
        
        # Calculate percentage of rows with no violations
        if 'flag_outlier' in df_with_validation.columns:
            no_violation_count = (~df_with_validation['flag_outlier']).sum()
            no_violation_pct = (no_violation_count / len(df_with_validation)) * 100.0
            print(f"\n✅ Rows with no violations: {no_violation_count:,} / {len(df_with_validation):,} ({no_violation_pct:.2f}%)")
        else:
            print("\n⚠️  No flag_outlier column found (no ratio features to validate)")
        
        # Check for anomalies against expected ranges
        flag_outliers_by_expected_range(df_with_validation, debug_print=True)
        
        # Analyze impact of removing extreme outliers
        flag_outliers_by_extreme_stats(df_with_validation, debug_print=True)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

