#!/usr/bin/env python3
"""
Collect relevant information for a given company (ticker).

This module gathers various data points about a company to facilitate
easy visual inspection and analysis.
"""

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import os
import io
import sys
from contextlib import redirect_stdout
import numpy as np
from feature_augment import (
    compute_ratio_features, 
    compute_ratio_features_on_ratio_definitions, 
    _find_feature_column_flexible,
    _find_feature_column
)
from config import (
    DATA_DIR, SAVE_DIR, MODEL_DIR, DEFAULT_K_TOP_TAGS, 
    DEFAULT_N_QUARTERS_HISTORY_COMP, FEATURIZED_ALL_QUARTERS_FILE
)
from config_metrics import RATIO_DEFINITIONS, KEY_METRICS
from get_company_filings_XBRL_API import (
    SECCompanyFilings,
    frame_to_qtrs,
    convert_to_bulk_schema
)
from utility_data import load_sic_codes_from_raw_data, read_tags_to_featurize
from featurize import featurize_df
from spotcheck_tag_synonyms import find_tag_matches
GRAPH_SAVE_DIR = os.path.join(DATA_DIR, 'company_graphs')


def collect_stock_price(ticker: str, start_date: str = None, end_date: str = None) -> dict:
    """
    Collect stock price data for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format (default: 5 year ago)
        end_date: End date in 'YYYY-MM-DD' format (default: today)
    
    Returns:
        dict: Dictionary with 'stock_price' key containing (date, price) time series
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        start_dt = datetime.now() - timedelta(days=365*5)
        start_date = start_dt.strftime('%Y-%m-%d')
    
    ticker_obj = yf.Ticker(ticker)
    hist = ticker_obj.history(start=start_date, end=end_date)
    
    if hist.empty:
        return {'stock_price': {}}
    
    # Create (year_month_day, price) time series using Close price
    stock_price_series = {}
    for date, row in hist.iterrows():
        date_str = date.strftime('%Y-%m-%d')
        stock_price_series[date_str] = float(row['Close'])
    
    return {'stock_price': stock_price_series}


def collect_sec_filing(ticker: str = None, client: SECCompanyFilings = None) -> dict:
    """
    Collect most recent SEC filing (10-Q only) data.
    
    Only retrieves 10-Q filings to ensure consistent quarterly comparisons.
    Ignores 10-K filings.
    
    Args:
        ticker: Stock ticker symbol (required if client not provided)
        client: SECCompanyFilings instance (optional, will create if not provided)
    
    Returns:
        dict: Dictionary with 'filing_url', 'filing_data' (XBRL DataFrame), 
              'bulk_data' (bulk schema DataFrame), 'form_type', 'report_date'
    """
    if client is None:
        if ticker is None:
            raise ValueError("Either ticker or client must be provided")
        client = SECCompanyFilings(ticker=ticker)
    
    # Get only 10-Q filings (ignore 10-K)
    df_xbrl = client.get_xbrl_data_dataframe(form_type='10-Q')
    form_type = '10-Q'
    
    if df_xbrl.empty:
        return {
            'filing_url': None,
            'filing_data': pd.DataFrame(),
            'bulk_data': pd.DataFrame(),
            'form_type': None,
            'report_date': None
        }
    
    report_date = df_xbrl['report_date'].max()
    
    if df_xbrl is None or df_xbrl.empty:
        return {
            'filing_url': None,
            'filing_data': pd.DataFrame(),
            'bulk_data': pd.DataFrame(),
            'form_type': None,
            'report_date': None
        }
    
    # Get filing URL
    accn = df_xbrl['accn'].iloc[0]
    cik = client.cik
    filing_url = f"https://www.sec.gov/cgi-bin/viewer?action=view&cik={cik}&accession_number={accn}&xbrl_type=v"
    
    # Convert to bulk schema
    df_bulk = convert_to_bulk_schema(df_xbrl, taxonomy='us-gaap')
    
    return {
        'filing_url': filing_url,
        'filing_data': df_xbrl,
        'bulk_data': df_bulk,
        'form_type': form_type,
        'report_date': report_date
    }


def prep_data_featurize_ratio(bulk_data: pd.DataFrame, cik: str, form_type: str,
                               df_tags_to_featurize: pd.DataFrame = None,
                               N_qtrs_history_comp: int = None,
                               ratio_definitions: dict = None) -> pd.DataFrame:
    """
    Prepare data and featurize to generate a dataframe with ratio features.
    
    Converts bulk data to the expected format, runs featurization pipeline,
    and computes ratio features using compute_ratio_features_on_ratio_definitions().
    
    Args:
        bulk_data: DataFrame with bulk schema (adsh, tag, version, ddate, qtrs, uom, segments, coreg, value, footnote)
        cik: Company CIK (as string, will be zero-padded)
        form_type: Form type (e.g., '10-Q' or '10-K')
        df_tags_to_featurize: DataFrame with columns ['rank', 'tag'] specifying which tags to featurize
                             (default: uses read_tags_to_featurize() with DEFAULT_K_TOP_TAGS)
        N_qtrs_history_comp: Number of quarters for history comparison (default: DEFAULT_N_QUARTERS_HISTORY_COMP)
        ratio_definitions: Dictionary mapping ratio names to (numerator, denominator) tuples
                          (default: uses RATIO_DEFINITIONS constant)
    
    Returns:
        pd.DataFrame: DataFrame with ratio features computed (columns ending with '_current' and '_augment'),
                     or empty DataFrame if processing fails
    """
    if bulk_data.empty:
        return pd.DataFrame()
    
    # Set defaults for optional parameters
    if df_tags_to_featurize is None:
        df_tags_to_featurize = read_tags_to_featurize(K_top_tags=DEFAULT_K_TOP_TAGS)
    if N_qtrs_history_comp is None:
        N_qtrs_history_comp = DEFAULT_N_QUARTERS_HISTORY_COMP
    
    # Convert bulk_data to the format expected by featurize_df()
    # Required columns: ['cik', 'tag', 'ddate', 'qtrs', 'segments', 'uom', 'custom_tag', 'value', 'period', 'form']
    df_joined = bulk_data.copy()
    
    # Filter out rows with empty ddate (invalid dates)
    df_joined = df_joined[df_joined['ddate'].notna() & (df_joined['ddate'] != '')].copy()
    
    if df_joined.empty:
        return pd.DataFrame()
    
    # Add/convert required columns
    df_joined['cik'] = str(cik).zfill(10)
    # ddate should be string YYYYMMDD format (history_comparisons will convert to datetime)
    # Ensure ddate is string format (convert_to_bulk_schema already does this, but ensure it's clean)
    df_joined['ddate'] = df_joined['ddate'].astype(str).str[:8]
    # period should be integer YYYYMMDD format (used in grouping)
    df_joined['period'] = df_joined['ddate'].astype(int)
    df_joined['form'] = form_type
    df_joined['custom_tag'] = (df_joined['version'] == df_joined['adsh']).astype(float)
    
    # Ensure segments is present and convert empty strings to NaN (already in bulk_data)
    if 'segments' not in df_joined.columns:
        df_joined['segments'] = None
    else:
        # Convert empty strings to NaN for segments (segment_group_summary expects NaN for empty segments)
        df_joined['segments'] = df_joined['segments'].replace('', None)
    
    # Filter for USD units only (similar to segment_group_summary logic)
    if 'uom' in df_joined.columns:
        df_joined = df_joined[df_joined['uom'] == 'USD'].copy()
    
    # Use featurize_df() to process the data (applies segment_group_summary, history_comparisons, organize_feature_dataframe)
    df_features = featurize_df(df_joined, form_type, df_tags_to_featurize, 
                              N_qtrs_history_comp, debug_print=False)
    
    if df_features.empty:
        return pd.DataFrame()
    
    # Get the most recent period's data
    df_features['period_int'] = pd.to_datetime(df_features['period'], format='%Y%m%d')
    most_recent_period = df_features['period_int'].max()
    df_recent = df_features[df_features['period_int'] == most_recent_period].copy()
    
    if df_recent.empty:
        return pd.DataFrame()
    
    # Use compute_ratio_features_on_ratio_definitions() to calculate ratios
    if ratio_definitions is None:
        ratio_definitions = RATIO_DEFINITIONS
    df_with_ratios = compute_ratio_features_on_ratio_definitions(df_recent, ratio_definitions, debug_print=False)
    
    return df_with_ratios


def find_column_with_qtrs(df: pd.DataFrame, tag: str, preferred_qtrs: int, suffix: str = '_current') -> tuple:
    """
    Find a column for a tag and return both the column name and the qtrs value used.
    
    This ensures we use the exact same qtrs value across all comparisons.
    
    Args:
        df: DataFrame with feature columns
        tag: Tag name (e.g., 'GrossProfit', 'Revenues')
        preferred_qtrs: Preferred qtrs value to try first (e.g., 1 for 10-Q)
        suffix: Column suffix (default: '_current')
    
    Returns:
        tuple: (column_name, qtrs_value) or (None, None) if not found
    """
    # Try preferred qtrs first
    col = _find_feature_column(df, tag, preferred_qtrs, suffix)
    if col:
        return col, preferred_qtrs
    
    # Fall back to flexible search, but extract qtrs from the found column
    col = _find_feature_column_flexible(df, tag, suffix)
    if col:
        # Extract qtrs value from column name (e.g., "NetIncomeLoss_1qtrs_current" -> 1)
        import re
        match = re.search(r'_(\d+)qtrs' + re.escape(suffix), col)
        if match:
            qtrs_value = int(match.group(1))
            return col, qtrs_value
        else:
            # If pattern doesn't match, assume 0qtrs (point-in-time)
            return col, 0
    
    return None, None


def display_metrics_and_ratios(df_with_ratios: pd.DataFrame, 
                                key_metrics_present: list,
                                ratio_definitions_modified: dict,
                                form_type: str = '10-Q'):
    """
    Display metrics from key_metrics_present and ratios from ratio_definitions_modified.
    
    Consolidates get_metrics() and display_ratio_metrics() to display only the metrics
    and ratios that are actually available after substitutions.
    
    Args:
        df_with_ratios: DataFrame with feature columns (ending with suffixes like '_current', '_augment')
        key_metrics_present: List of metric tags that are present (original or substituted)
        ratio_definitions_modified: Dictionary of ratio definitions with substitutions applied
    """
    if df_with_ratios.empty:
        print("\n⚠️  No data available to display")
        return
    
    # Determine preferred qtrs based on form_type
    preferred_qtrs = 1 if form_type == '10-Q' else 4
    
    print("\n" + "=" * 80)
    print("KEY METRICS")
    print("=" * 80)
    
    # Display metrics from key_metrics_present
    for tag in key_metrics_present:
        # Find column and extract qtrs value to ensure consistency
        col, qtrs_used = find_column_with_qtrs(df_with_ratios, tag, preferred_qtrs, suffix='_current')
        if col:
            value = df_with_ratios[col].iloc[0]
            if pd.notna(value):
                print(f"  {tag}: {float(value):,.0f} (from {col})")
            else:
                print(f"  {tag}: [NaN]")
        else:
            print(f"  {tag}: [Column not found]")
    
    print("\n" + "=" * 80)
    print("RATIO METRICS")
    print("=" * 80)
    
    # Display ratios from ratio_definitions_modified
    for ratio_name, (num_component, denom_component) in ratio_definitions_modified.items():
        # Extract and display numerator component(s)
        if isinstance(num_component, tuple):
            # Tuple numerator (e.g., QuickRatio: AssetsCurrent - InventoryNet)
            num_values = []
            num_display = []
            for comp in num_component:
                comp_col, comp_qtrs = find_column_with_qtrs(df_with_ratios, comp, preferred_qtrs, suffix='_current')
                if comp_col:
                    value = df_with_ratios[comp_col].iloc[0]
                    if pd.notna(value):
                        num_values.append(float(value))
                        num_display.append(f"{comp}: {float(value):,.0f}")
                    else:
                        num_display.append(f"{comp}: [NaN]")
                else:
                    num_display.append(f"{comp}: [Column not found]")
            
            # Display numerator components
            for disp in num_display:
                print(f"  {disp}")
            
            # For tuple numerators, we might need to compute (e.g., AssetsCurrent - InventoryNet)
            # But the ratio itself should already be computed in the dataframe
        else:
            # Single numerator component
            num_col, num_qtrs = find_column_with_qtrs(df_with_ratios, num_component, preferred_qtrs, suffix='_current')
            if num_col:
                value = df_with_ratios[num_col].iloc[0]
                if pd.notna(value):
                    print(f"  {num_component}: {float(value):,.0f}")
                else:
                    print(f"  {num_component}: [NaN]")
            else:
                print(f"  {num_component}: [Column not found]")
        
        # Display denominator component
        denom_col, denom_qtrs = find_column_with_qtrs(df_with_ratios, denom_component, preferred_qtrs, suffix='_current')
        if denom_col:
            value = df_with_ratios[denom_col].iloc[0]
            if pd.notna(value):
                print(f"  {denom_component}: {float(value):,.0f}")
            else:
                print(f"  {denom_component}: [NaN]")
        else:
            print(f"  {denom_component}: [Column not found]")
        
        # Display ratio (should be in format 'ratio_name_augment')
        ratio_col = f'{ratio_name}_augment'
        if ratio_col in df_with_ratios.columns:
            ratio_value = df_with_ratios[ratio_col].iloc[0]
            if pd.notna(ratio_value):
                print(f"  {ratio_name}: {float(ratio_value):.4f}")
            else:
                print(f"  {ratio_name}: [Cannot calculate - NaN result]")
        else:
            print(f"  {ratio_name}: [Cannot calculate - ratio not computed]")
        print()


def check_missing_metrics(bulk_data: pd.DataFrame) -> tuple:
    """
    Check which KEY_METRICS tags are missing from bulk_data and interactively ask user
    for substitutions. Creates modified metrics and ratio definitions based on substitutions.
    
    Args:
        bulk_data: DataFrame with bulk schema containing 'tag' column
    
    Returns:
        tuple: (key_metrics_present, ratio_definitions_modified)
            - key_metrics_present: List of metrics that are present (original or substituted) for display
            - ratio_definitions_modified: Dictionary of ratio definitions with substitutions applied,
                                        ratios with missing components are excluded
    """
    if bulk_data.empty or 'tag' not in bulk_data.columns:
        # If no data, return empty lists
        return [], {}
    
    # Get unique tags from bulk_data
    available_tags = set(bulk_data['tag'].unique())
    
    # Find missing KEY_METRICS tags
    missing_key_metrics = sorted([tag for tag in KEY_METRICS if tag not in available_tags])
    
    # Load tag statistics for popularity information
    tag_stats_file = os.path.join(MODEL_DIR, 'tag_stats_sorted.csv')
    tag_stats_dict = {}
    if os.path.exists(tag_stats_file):
        try:
            df_tag_stats = pd.read_csv(tag_stats_file)
            tag_stats_dict = dict(zip(df_tag_stats['tag'], df_tag_stats['distinct_companies']))
        except Exception as e:
            print(f"⚠️  Warning: Could not load tag statistics: {e}")
    
    # Build substitution dictionary for KEY_METRICS
    key_metrics_substitution_dict = {}
    
    # For each missing KEY_METRIC, find closest matches and ask user
    if missing_key_metrics and available_tags:
        print(f"\n🔍 Finding closest matching tags for {len(missing_key_metrics)} missing KEY_METRICS...")
        for missing_tag in missing_key_metrics:
            matches = find_tag_matches(missing_tag, available_tags, threshold=0.7)
            if matches:
                print(f"\n  Missing: {missing_tag}")
                print(f"    Closest matches (top 10):")
                for i, (tag, score) in enumerate(matches[:10], 1):
                    num_companies = tag_stats_dict.get(tag, None)
                    if num_companies is not None:
                        print(f"      {i}. {tag} (similarity: {score:.3f}, used by {num_companies:,} companies)")
                    else:
                        print(f"      {i}. {tag} (similarity: {score:.3f}, popularity: unknown)")
                
                # Ask user to choose a substitution
                while True:
                    try:
                        choice = input(f"\n    Choose a substitution for '{missing_tag}' (enter number 1-10, or '0' for no match, or 'skip' to skip this tag): ").strip().lower()
                        if choice == '0' or choice == 'skip':
                            key_metrics_substitution_dict[missing_tag] = None
                            print(f"    → No substitution chosen for '{missing_tag}'")
                            break
                        elif choice.isdigit():
                            choice_num = int(choice)
                            if 1 <= choice_num <= len(matches):
                                chosen_tag = matches[choice_num - 1][0]
                                key_metrics_substitution_dict[missing_tag] = chosen_tag
                                print(f"    → Substituting '{missing_tag}' with '{chosen_tag}'")
                                break
                            else:
                                print(f"    ⚠️  Please enter a number between 1 and {len(matches)}, or '0' for no match")
                        else:
                            print(f"    ⚠️  Invalid input. Please enter a number between 1 and {len(matches)}, or '0' for no match")
                    except (ValueError, KeyboardInterrupt):
                        print(f"    ⚠️  Invalid input. Please enter a number between 1 and {len(matches)}, or '0' for no match")
            else:
                print(f"\n  Missing: {missing_tag}")
                print(f"    No similar tags found")
                # Ask user if they want to skip this tag
                while True:
                    try:
                        choice = input(f"    No matches found for '{missing_tag}'. Skip this tag? (y/n): ").strip().lower()
                        if choice in ['y', 'yes', 'n', 'no']:
                            key_metrics_substitution_dict[missing_tag] = None
                            print(f"    → No substitution for '{missing_tag}'")
                            break
                        else:
                            print("    ⚠️  Please enter 'y' or 'n'")
                    except KeyboardInterrupt:
                        key_metrics_substitution_dict[missing_tag] = None
                        break
    
    # Build key_metrics_present: metrics that are available (either original or substituted)
    key_metrics_present = []
    key_metrics_missing_after_match = []
    
    for tag in KEY_METRICS:
        if tag in available_tags:
            # Tag is present in bulk_data
            key_metrics_present.append(tag)
        elif tag in key_metrics_substitution_dict:
            substitution = key_metrics_substitution_dict[tag]
            if substitution is not None:
                # User chose a substitution
                key_metrics_present.append(substitution)
            else:
                # User declined substitution
                key_metrics_missing_after_match.append(tag)
        else:
            # Tag was not missing, so it's present
            key_metrics_present.append(tag)
    
    # Deduplicate key_metrics_present while preserving order
    seen = set()
    key_metrics_present_deduped = []
    for tag in key_metrics_present:
        if tag not in seen:
            seen.add(tag)
            key_metrics_present_deduped.append(tag)
    key_metrics_present = key_metrics_present_deduped
    
    # Build ratio_definitions_modified using the substitution dictionary
    ratio_definitions_modified = {}
    ratio_definitions_missing = {}
    
    for ratio_name, (num_component, denom_component) in RATIO_DEFINITIONS.items():
        skip_ratio = False
        new_num_component = num_component
        new_denom_component = denom_component
        
        # Process numerator component
        if isinstance(num_component, tuple):
            # Handle tuple numerator (e.g., ('AssetsCurrent', 'InventoryNet'))
            new_num_components = []
            for comp in num_component:
                if comp in key_metrics_substitution_dict:
                    substitution = key_metrics_substitution_dict[comp]
                    if substitution is None:
                        skip_ratio = True
                        break
                    new_num_components.append(substitution)
                else:
                    # Check if component is in available_tags
                    if comp in available_tags:
                        new_num_components.append(comp)
                    else:
                        skip_ratio = True
                        break
            new_num_component = tuple(new_num_components) if not skip_ratio else num_component
        else:
            # Handle single numerator component
            if num_component in key_metrics_substitution_dict:
                substitution = key_metrics_substitution_dict[num_component]
                if substitution is None:
                    skip_ratio = True
                else:
                    new_num_component = substitution
            else:
                # Check if component is in available_tags
                if num_component not in available_tags:
                    skip_ratio = True
        
        # Process denominator component (only if numerator didn't cause skip)
        if not skip_ratio:
            if denom_component in key_metrics_substitution_dict:
                substitution = key_metrics_substitution_dict[denom_component]
                if substitution is None:
                    skip_ratio = True
                else:
                    new_denom_component = substitution
            else:
                # Check if component is in available_tags
                if denom_component not in available_tags:
                    skip_ratio = True
        
        # Add ratio to appropriate dictionary
        if skip_ratio:
            ratio_definitions_missing[ratio_name] = (num_component, denom_component)
        else:
            ratio_definitions_modified[ratio_name] = (new_num_component, new_denom_component)
    
    # Print summary of missing tags and skipped ratios
    print("\n" + "=" * 80)
    print("SUMMARY: Missing Metrics and Skipped Ratios")
    print("=" * 80)
    
    if key_metrics_missing_after_match:
        print(f"\n❌ Missing KEY_METRICS (no data available after match attempts): {len(key_metrics_missing_after_match)}")
        for tag in sorted(key_metrics_missing_after_match):
            print(f"  - {tag}")
    else:
        print("\n✅ All KEY_METRICS are available (original or substituted)")
    
    if ratio_definitions_missing:
        print(f"\n⚠️  Skipped Ratios (missing required components): {len(ratio_definitions_missing)}")
        for ratio_name, (num_component, denom_component) in ratio_definitions_missing.items():
            # Extract missing components by checking if they're actually available
            missing_components = []
            
            def is_component_available(comp):
                """Check if a component is available (present or has substitution)."""
                if comp in available_tags:
                    return True
                if comp in key_metrics_substitution_dict:
                    return key_metrics_substitution_dict[comp] is not None
                return False
            
            if isinstance(num_component, tuple):
                for comp in num_component:
                    if not is_component_available(comp):
                        missing_components.append(comp)
            else:
                if not is_component_available(num_component):
                    missing_components.append(num_component)
            
            if not is_component_available(denom_component):
                missing_components.append(denom_component)
            
            missing_str = f" (missing: {', '.join(missing_components)})" if missing_components else ""
            print(f"  - {ratio_name}: {num_component} / {denom_component}{missing_str}")
    else:
        print("\n✅ All ratios are available (original or substituted)")
    
    print("\n" + "=" * 80)
    print(f"\n📊 Summary:")
    print(f"  - KEY_METRICS available (after substitutions): {len(key_metrics_present)}/{len(KEY_METRICS)}")
    print(f"  - Ratios available (after substitutions): {len(ratio_definitions_modified)}/{len(RATIO_DEFINITIONS)}")
    if len(ratio_definitions_modified) < len(RATIO_DEFINITIONS):
        skipped_count = len(RATIO_DEFINITIONS) - len(ratio_definitions_modified)
        print(f"  - Skipped ratios (missing required components): {skipped_count}")
    
    return key_metrics_present, ratio_definitions_modified


def plot_historical_trends(df_featurized_all_quarters: pd.DataFrame,
                           df_with_ratios: pd.DataFrame,
                           cik: str,
                           sic: str,
                           key_metrics_present: list,
                           ratio_definitions_modified: dict,
                           ticker: str,
                           form_type: str,
                           return_figures: bool = False) -> List[plt.Figure]:
    """
    Plot historical trends for key metrics and ratios.
    
    For each metric/ratio, plots:
    - Company's historical values over time
    - Current value from most recent filing (df_with_ratios)
    - Industry benchmark (average across same SIC code) over time
    
    Args:
        df_featurized_all_quarters: DataFrame with historical featurized data for all companies
        df_with_ratios: DataFrame with current company's most recent filing metrics/ratios
        cik: Company CIK (as string, zero-padded)
        sic: Company SIC code (as string)
        key_metrics_present: List of metric tags to plot
        ratio_definitions_modified: Dictionary of ratio definitions to plot
        ticker: Stock ticker symbol for plot titles
        form_type: Form type ('10-Q' for quarterly, '10-K' for annual)
        return_figures: If True, return list of figures instead of showing them
    
    Returns:
        List[plt.Figure]: List of figure objects (one per metric/ratio) if return_figures=True, else []
    """
    if df_featurized_all_quarters.empty or df_with_ratios.empty:
        print("\n⚠️  Cannot plot historical trends: missing data")
        return
    
    # Normalize CIK format
    cik_padded = str(cik).zfill(10)
    df_featurized_all_quarters['cik'] = df_featurized_all_quarters['cik'].astype(str).str.zfill(10)
    
    # Filter to current company's data
    df_company = df_featurized_all_quarters[df_featurized_all_quarters['cik'] == cik_padded].copy()
    
    if df_company.empty:
        print(f"\n⚠️  No historical data found for company CIK: {cik_padded}")
        return
    
    # Filter to companies in same industry (same SIC code)
    if 'sic' not in df_featurized_all_quarters.columns:
        print(f"\n⚠️  Cannot compute industry benchmarks: SIC codes not available")
        return
    
    df_industry = df_featurized_all_quarters[
        (df_featurized_all_quarters['sic'] == sic) & 
        df_featurized_all_quarters['sic'].notna()
    ].copy()
    
    if df_industry.empty:
        print(f"\n⚠️  No industry data found for SIC code: {sic}")
        return
    
    # Convert period to datetime for plotting
    df_company['period_dt'] = pd.to_datetime(df_company['period'].astype(str), format='%Y%m%d', errors='coerce')
    df_industry['period_dt'] = pd.to_datetime(df_industry['period'].astype(str), format='%Y%m%d', errors='coerce')
    
    # Remove rows with invalid dates
    df_company = df_company[df_company['period_dt'].notna()].copy()
    df_industry = df_industry[df_industry['period_dt'].notna()].copy()
    
    # Sort by period for proper time series plotting
    df_company = df_company.sort_values('period_dt')
    df_industry = df_industry.sort_values('period_dt')
    
    # Get current period from df_with_ratios
    if 'period' in df_with_ratios.columns:
        current_period_dt = pd.to_datetime(str(df_with_ratios['period'].iloc[0]), format='%Y%m%d', errors='coerce')
    else:
        current_period_dt = None
    
    # Collect all metrics and ratios to plot
    items_to_plot = []
    for metric in key_metrics_present:
        items_to_plot.append(('metric', metric))
    for ratio_name in ratio_definitions_modified.keys():
        items_to_plot.append(('ratio', ratio_name))
    if not items_to_plot:
        print("\n⚠️  No metrics or ratios to plot")
        return [] if return_figures else None
    
    # Count number of companies in industry for label
    num_companies_industry = df_industry['cik'].nunique()
    
    # Determine qtrs value based on form_type to ensure we compare like with like
    # 10-Q reports quarterly data (1 quarter), 10-K reports annual data (4 quarters)
    if form_type == '10-K':
        target_qtrs = 4  # Annual filing
    else:
        target_qtrs = 1  # Quarterly filing (10-Q)
    
    figures = []
    
    # Plot figures per page (2 columns x 3 rows = 6 items per page)
    n_cols = 1
    n_rows = 4
    items_per_page = n_cols * n_rows
    
    for page_idx in range(0, len(items_to_plot), items_per_page):
        page_items = items_to_plot[page_idx:page_idx + items_per_page]
        
        # Create figure with subplots for this page (portrait orientation for 2x3 layout)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 14))
        # Always flatten axes to a 1D array for consistent indexing
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            # If axes is a single axes object (shouldn't happen with n_rows > 1, n_cols > 1), wrap it
            axes = [axes]
        
        for subplot_idx, (item_type, item_name) in enumerate(page_items):
            ax = axes[subplot_idx]
            
            # Determine the qtrs value to use for this metric by checking what exists in current filing
            # This ensures we use the EXACT same qtrs value across all comparisons
            metric_qtrs = None
            col_name = None
            ratio_col_name = None
            
            if item_type == 'metric':
                # Find column in current filing and extract qtrs value
                # This ensures we use the EXACT same qtrs value across all comparisons
                current_col_found, metric_qtrs = find_column_with_qtrs(df_with_ratios, item_name, target_qtrs, suffix='_current')
                if current_col_found and metric_qtrs is not None:
                    # Now use this exact qtrs value for all comparisons
                    col_name = _find_feature_column(df_company, item_name, metric_qtrs, suffix='_current')
                    # If not found with exact qtrs, try flexible but extract qtrs from result
                    if not col_name:
                        col_name = _find_feature_column_flexible(df_company, item_name, suffix='_current')
                        if col_name:
                            # Extract qtrs from the found column to ensure consistency
                            import re
                            match = re.search(r'_(\d+)qtrs_current', col_name)
                            if match:
                                metric_qtrs = int(match.group(1))
                else:
                    # If not found in current filing, try flexible search in historical data
                    col_name = _find_feature_column_flexible(df_company, item_name, suffix='_current')
                    if col_name:
                        # Extract qtrs from the found column
                        import re
                        match = re.search(r'_(\d+)qtrs_current', col_name)
                        if match:
                            metric_qtrs = int(match.group(1))
                suffix = '_current'
            else:
                # Plot ratio (ratios don't have qtrs suffix, just _augment)
                ratio_col_name = f'{item_name}_augment'
                suffix = '_augment'
            
            # Get company historical data - use the exact same qtrs value
            company_historical_plotted = False
            if item_type == 'metric' and metric_qtrs is not None and col_name:
                # Use the exact qtrs value determined above
                historical_col = _find_feature_column(df_company, item_name, metric_qtrs, suffix='_current')
                if historical_col and historical_col in df_company.columns:
                    company_data = df_company[['period_dt', historical_col]].copy()
                    company_data = company_data[company_data[historical_col].notna()]
                    if not company_data.empty:
                        ax.plot(company_data['period_dt'], company_data[historical_col], 
                               'o-', label=f'{ticker} (historical)', linewidth=2, markersize=4)
                        company_historical_plotted = True
                        col_name = historical_col  # Update col_name for industry benchmark
                elif col_name and col_name in df_company.columns:
                    # Fallback: use the column found (should have same qtrs)
                    company_data = df_company[['period_dt', col_name]].copy()
                    company_data = company_data[company_data[col_name].notna()]
                    if not company_data.empty:
                        ax.plot(company_data['period_dt'], company_data[col_name], 
                               'o-', label=f'{ticker} (historical)', linewidth=2, markersize=4)
                        company_historical_plotted = True
            elif ratio_col_name and ratio_col_name in df_company.columns:
                # For ratios, check the ratio column in company data
                company_data = df_company[['period_dt', ratio_col_name]].copy()
                company_data = company_data[company_data[ratio_col_name].notna()]
                if not company_data.empty:
                    ax.plot(company_data['period_dt'], company_data[ratio_col_name], 
                           'o-', label=f'{ticker} (historical)', linewidth=2, markersize=4)
                    company_historical_plotted = True
            
            # Get current value from df_with_ratios - use the exact same qtrs value
            if item_type == 'metric' and metric_qtrs is not None:
                current_col = _find_feature_column(df_with_ratios, item_name, metric_qtrs, suffix='_current')
                # If exact match not found, fall back to what we found earlier
                if not current_col and current_col_found:
                    current_col = current_col_found
            else:
                current_col = ratio_col_name if ratio_col_name in df_with_ratios.columns else None
            
            if current_col and current_col in df_with_ratios.columns:
                current_value = df_with_ratios[current_col].iloc[0]
                if pd.notna(current_value) and current_period_dt is not None:
                    ax.scatter([current_period_dt], [current_value], 
                              s=150, color='red', marker='*', 
                              label=f'{ticker} (current)', zorder=5)
            
            # Compute industry benchmark over time using aggregation
            # Use the EXACT same qtrs value as determined for current and historical data
            industry_label = f'Industry avg of {num_companies_industry} companies (SIC: {sic})'
            if item_type == 'metric' and metric_qtrs is not None and col_name:
                # Use the exact same qtrs value for industry benchmark
                industry_col = _find_feature_column(df_industry, item_name, metric_qtrs, suffix='_current')
                if industry_col and industry_col in df_industry.columns:
                    industry_benchmark = df_industry.groupby('period_dt')[industry_col].mean().reset_index()
                    industry_benchmark = industry_benchmark[industry_benchmark[industry_col].notna()]
                    if not industry_benchmark.empty:
                        ax.plot(industry_benchmark['period_dt'], industry_benchmark[industry_col],
                               's--', label=industry_label, linewidth=1.5, markersize=3, alpha=0.7, color='green')
                elif col_name and col_name in df_industry.columns:
                    # Fallback: use the same column name (should have same qtrs)
                    industry_benchmark = df_industry.groupby('period_dt')[col_name].mean().reset_index()
                    industry_benchmark = industry_benchmark[industry_benchmark[col_name].notna()]
                    if not industry_benchmark.empty:
                        ax.plot(industry_benchmark['period_dt'], industry_benchmark[col_name],
                               's--', label=industry_label, linewidth=1.5, markersize=3, alpha=0.7, color='green')
            elif ratio_col_name and ratio_col_name in df_industry.columns:
                # Group by period_dt and compute mean
                industry_benchmark = df_industry.groupby('period_dt')[ratio_col_name].mean().reset_index()
                industry_benchmark = industry_benchmark[industry_benchmark[ratio_col_name].notna()]
                if not industry_benchmark.empty:
                    ax.plot(industry_benchmark['period_dt'], industry_benchmark[ratio_col_name],
                           's--', label=industry_label, linewidth=1.5, markersize=3, alpha=0.7, color='green')
            
            ax.set_title(f'{item_name}: {ticker} vs Industry', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Period', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            # Only add legend if there are labels (handles empty plot case)
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(page_items), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        figures.append(fig)
    
    if return_figures:
        return figures
    else:
        plt.show()
        return []
    

def get_metrics(df_with_ratios: pd.DataFrame, suffixes: list = ['_current', '_augment']) -> dict:
    """
    Extract metrics from a dataframe into a dictionary, handling suffix removal.
    
    Converts dataframe columns (with suffixes like '_current' and '_augment') into
    a dictionary with shortened names (suffixes removed).
    
    This is a helper function kept for backward compatibility, but the main display
    should use display_metrics_and_ratios() instead.
    
    Args:
        df_with_ratios: DataFrame with feature columns ending in specified suffixes
        suffixes: List of suffixes to process (default: ['_current', '_augment'])
                 Columns ending with these suffixes will be extracted and suffix removed
    
    Returns:
        dict: Dictionary mapping shortened metric names (without suffixes) to values
             - For '_current' columns: includes None for NaN values
             - For '_augment' columns: only includes non-NaN values
    """
    if df_with_ratios.empty:
        return {}
    
    metrics = {}
    
    # Extract values for each suffix type
    for suffix in suffixes:
        # Find all columns ending with this suffix
        columns_with_suffix = [col for col in df_with_ratios.columns if col.endswith(suffix)]
        
        for col in columns_with_suffix:
            value = df_with_ratios[col].iloc[0]
            short_name = col.replace(suffix, '')
            
            # For '_current' columns, include None values
            # For '_augment' columns, only include non-NaN values
            if suffix == '_current':
                metrics[short_name] = None if pd.isna(value) else float(value)
            elif suffix == '_augment':
                if pd.notna(value):
                    metrics[short_name] = float(value)
            else:
                # Generic handling for other suffixes
                if pd.notna(value):
                    metrics[short_name] = float(value)
    
    return metrics


def plot_stock_price_comparison(company_data: dict, ticker: str):
    """
    Create stock price comparison chart with company and S&P 500.
    
    Args:
        company_data: Dictionary containing company data with 'stock_price' key
        ticker: Stock ticker symbol
    
    Returns:
        matplotlib.figure.Figure: Figure object for the stock price chart, or None if no data
    """
    stock_price = company_data.get('stock_price', {})
    
    if not stock_price:
        return None
    
    # Get company stock data
    dates = pd.to_datetime(list(stock_price.keys()))
    prices = list(stock_price.values())
    df_company = pd.DataFrame({'date': dates, 'price': prices}).sort_values('date')
    
    # Get S&P 500 data for same period
    start_date = df_company['date'].min()
    end_date = df_company['date'].max()
    
    try:
        sp500 = yf.Ticker('^GSPC')
        hist_sp500 = sp500.history(start=start_date, end=end_date)
        
        if not hist_sp500.empty:
            df_sp500 = hist_sp500.reset_index()
            df_sp500 = df_sp500[['Date', 'Close']].rename(columns={'Date': 'date', 'Close': 'price'})
            
            # Normalize both to start at 100 for better comparison
            df_company['normalized'] = (df_company['price'] / df_company['price'].iloc[0]) * 100
            df_sp500['normalized'] = (df_sp500['price'] / df_sp500['price'].iloc[0]) * 100
        else:
            df_sp500 = None
    except Exception as e:
        print(f"⚠️  Warning: Could not fetch S&P 500 data: {e}")
        df_sp500 = None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot company stock
    ax.plot(df_company['date'], df_company['price'], 
           linewidth=2, label=ticker, color='blue')
    
    # Plot S&P 500 if available
    if df_sp500 is not None:
        ax.plot(df_sp500['date'], df_sp500['price'],
               linewidth=2, label='S&P 500', color='orange', alpha=0.7)
    
    ax.set_title(f'Stock Price Comparison: {ticker} vs S&P 500', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def create_text_figure(text_content: str, title: str = "Report Output") -> plt.Figure:
    """
    Create a matplotlib figure containing text content.
    
    Args:
        text_content: Text content to display
        title: Title for the figure
    
    Returns:
        matplotlib.figure.Figure: Figure object with text content
    """
    fig, ax = plt.subplots(figsize=(11, 8.5))  # Letter size
    ax.axis('off')
    
    # Add title
    ax.text(0.5, 0.98, title, transform=ax.transAxes,
           fontsize=16, fontweight='bold', ha='center', va='top')
    
    # Split text into lines and format for display
    lines = text_content.split('\n')
    # Limit number of lines to fit on page (approximately 80 lines at fontsize 9)
    max_lines = 75
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"\n... ({len(lines) - max_lines} more lines truncated) ..."]
    
    # Create text string with proper spacing
    display_text = '\n'.join(lines)
    
    # Add text content
    ax.text(0.05, 0.94, display_text, transform=ax.transAxes,
           fontsize=8, fontfamily='monospace', va='top', ha='left')
    
    plt.tight_layout()
    return fig


def generate_pdf_report(company_data: dict,
                        df_featurized_all_quarters: pd.DataFrame,
                        df_with_ratios: pd.DataFrame,
                        company_info: dict,
                        filing_info: dict,
                        key_metrics_present: list,
                        ratio_definitions_modified: dict,
                        ticker: str,
                        captured_output: str = ""):
    """
    Generate a PDF report with stock price comparison and metric/ratio trends.
    
    Args:
        company_data: Dictionary with company stock price data
        df_featurized_all_quarters: DataFrame with historical featurized data
        df_with_ratios: DataFrame with current company's metrics/ratios
        company_info: Dictionary with company information (cik, sic, etc.)
        filing_info: Dictionary with filing information (report_date, etc.)
        key_metrics_present: List of metric tags to include
        ratio_definitions_modified: Dictionary of ratio definitions to include
        ticker: Stock ticker symbol
    """
    # Get most recent period for filename
    if 'period' in df_with_ratios.columns and not df_with_ratios.empty:
        most_recent_period = str(df_with_ratios['period'].iloc[0])
    elif filing_info.get('report_date'):
        most_recent_period = filing_info['report_date'].strftime('%Y%m%d')
    else:
        most_recent_period = datetime.now().strftime('%Y%m%d')
    
    # Create PDF filename
    pdf_filename = os.path.join(GRAPH_SAVE_DIR, f'{ticker}_{most_recent_period}_report.pdf')
    
    print(f"\n📄 Generating PDF report: {pdf_filename}")
    
    figures = []
    
    # Add text output page if captured output exists
    if captured_output.strip():
        text_fig = create_text_figure(captured_output, f"Report Output: {ticker}")
        figures.append(text_fig)
    
    # Add stock price comparison chart
    stock_fig = plot_stock_price_comparison(company_data, ticker)
    if stock_fig:
        figures.append(stock_fig)
    
    # Add metric/ratio trend charts
    if (not df_featurized_all_quarters.empty and 
        company_info.get('sic') and 
        (key_metrics_present or ratio_definitions_modified) and
        not df_with_ratios.empty):
        trend_figures = plot_historical_trends(
            df_featurized_all_quarters,
            df_with_ratios,
            company_info['cik'],
            company_info['sic'],
            key_metrics_present,
            ratio_definitions_modified,
            ticker,
            filing_info['form_type'],
            return_figures=True
        )
        figures.extend(trend_figures)
    
    # Save all figures to PDF
    if figures:
        with PdfPages(pdf_filename) as pdf:
            for fig in figures:
                pdf.savefig(fig, bbox_inches='tight')
            # Add metadata
            d = pdf.infodict()
            d['Title'] = f'Financial Report: {ticker}'
            d['Author'] = 'SEC Data Analysis'
            d['Subject'] = f'Company financial metrics and ratios for {ticker}'
            d['Keywords'] = f'{ticker}, SEC, financial metrics, ratios'
        
        # Close figures to free memory
        for fig in figures:
            plt.close(fig)
        
        print(f"✅ PDF report saved: {pdf_filename}")
        print(f"   Total pages: {len(figures)}")
    else:
        print("⚠️  No figures to save to PDF")


def main():
    """Main function to collect and display company data."""
    # Prompt user for ticker
    ticker = input("Enter stock ticker symbol (e.g., AAPL): ").strip().upper()
    if not ticker:
        print("⚠️  No ticker provided, using default: AAPL")
        ticker = 'AAPL'
    
    # Collect data (output captured for PDF)
    captured_output = io.StringIO()
    
    with redirect_stdout(captured_output):
        company_data = collect_stock_price(ticker)
        
        # Collect SEC filing data
        client = SECCompanyFilings(ticker=ticker)
        company_info = client.get_company_info()
        print(company_info)
        filing_info = collect_sec_filing(client=client)
        company_data['filing'] = filing_info
        company_data['company_info'] = company_info
        
        # Display filing URL
        if filing_info['filing_url']:
            print(f"\nSEC Filing URL: {filing_info['filing_url']}")
            print(f"Form Type: {filing_info['form_type']}")
            if filing_info['report_date']:
                print(f"Report Date: {filing_info['report_date'].strftime('%Y-%m-%d')}")
            if company_info.get('sic'):
                print(f"SIC Code: {company_info['sic']}")
        
    # Print captured output so far to console
    captured_so_far = captured_output.getvalue()
    print(captured_so_far)
    
    # Check for missing metrics OUTSIDE redirect_stdout so input() prompts work
    key_metrics_present = []
    ratio_definitions_modified = {}
    df_with_ratios = pd.DataFrame()
    
    if not filing_info['bulk_data'].empty:
        # Check for missing metrics - this needs to run outside redirect_stdout
        # to allow input() prompts to work correctly
        key_metrics_present, ratio_definitions_modified = check_missing_metrics(filing_info['bulk_data'])
    
    # Continue capturing output
    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
        if not filing_info['bulk_data'].empty:
            df_with_ratios = prep_data_featurize_ratio(filing_info['bulk_data'], 
                                                    company_info['cik'], 
                                                    filing_info['form_type'], 
                                                    df_tags_to_featurize = read_tags_to_featurize(K_top_tags=DEFAULT_K_TOP_TAGS), 
                                                    N_qtrs_history_comp = DEFAULT_N_QUARTERS_HISTORY_COMP,
                                                    ratio_definitions = ratio_definitions_modified)
            
            # Display metrics and ratios
            display_metrics_and_ratios(df_with_ratios, key_metrics_present, ratio_definitions_modified, filing_info['form_type'])
        
        # read in featurized data for all companies
        df_featurized_all_quarters = pd.read_csv(FEATURIZED_ALL_QUARTERS_FILE)
        df_featurized_all_quarters = df_featurized_all_quarters[df_featurized_all_quarters['form'] == filing_info['form_type']]
        
        # Compute ratios if we have ratio definitions (from current filing or use defaults)
        if ratio_definitions_modified:
            df_featurized_all_quarters = compute_ratio_features_on_ratio_definitions(
                df_featurized_all_quarters, 
                ratio_definitions = ratio_definitions_modified, debug_print=False)
        else:
            # Use default ratio definitions if no modifications available
            df_featurized_all_quarters = compute_ratio_features_on_ratio_definitions(
                df_featurized_all_quarters, 
                ratio_definitions = RATIO_DEFINITIONS, debug_print=False)
        
        # Join with (cik, sic) dataframe from utility_data.py:main()
        most_recent_quarter = df_featurized_all_quarters['data_qtr'].max()
        sic_file = os.path.join(SAVE_DIR, f'sic_codes_{most_recent_quarter}.csv')
        if os.path.exists(sic_file):
            df_sic_codes = pd.read_csv(sic_file)
            # Normalize CIK format for joining (zero-padded string)
            df_sic_codes['cik'] = df_sic_codes['cik'].astype(str).str.zfill(10)
            df_featurized_all_quarters['cik'] = df_featurized_all_quarters['cik'].astype(str).str.zfill(10)
            # Join on cik
            df_featurized_all_quarters = df_featurized_all_quarters.merge(df_sic_codes, on='cik', how='left')
    
    # Get all captured output and combine
    captured_text = captured_so_far + captured_output.getvalue()
    
    # Print remaining captured output to console
    remaining_output = captured_output.getvalue()
    if remaining_output:
        print(remaining_output)
    
    print(f"\n📄 Generating PDF report...")
    
    # Generate PDF report (outside redirect_stdout so PDF generation messages print to console)
    generate_pdf_report(
        company_data,
        df_featurized_all_quarters,
        df_with_ratios,
        company_info,
        filing_info,
        key_metrics_present,
        ratio_definitions_modified,
        ticker,
        captured_text
    )



if __name__ == "__main__":
    main()

