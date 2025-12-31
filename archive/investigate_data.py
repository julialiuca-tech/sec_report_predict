#!/usr/bin/env python3
"""
Investigate data discrepancies between df_all and df_train/df_test

This script:
1. Loads df_all.pkl, df_train.pkl, and df_test.pkl from SAVE_DIR
2. Splits df_all into before/after 2024-06-01 cutoff
3. Compares key identifiers (cik, period, data_qtr) between datasets
4. Compares feature values for matching records
"""

import os
import pandas as pd
import numpy as np
from config import SAVE_DIR


def load_dataframes():
    """Load all three pickle files"""
    print("=" * 80)
    print("STEP 1: Loading dataframes")
    print("=" * 80)
    
    df_all_file = os.path.join(SAVE_DIR, 'df_all.pkl')
    df_train_file = os.path.join(SAVE_DIR, 'df_train.pkl')
    df_test_file = os.path.join(SAVE_DIR, 'df_test.pkl')
    
    if not os.path.exists(df_all_file):
        raise FileNotFoundError(f"df_all.pkl not found: {df_all_file}")
    if not os.path.exists(df_train_file):
        raise FileNotFoundError(f"df_train.pkl not found: {df_train_file}")
    if not os.path.exists(df_test_file):
        raise FileNotFoundError(f"df_test.pkl not found: {df_test_file}")
    
    df_all = pd.read_pickle(df_all_file)
    df_train = pd.read_pickle(df_train_file)
    df_test = pd.read_pickle(df_test_file)
    
    print(f"\n📊 df_all shape: {df_all.shape}")
    print(f"📊 df_train shape: {df_train.shape}")
    print(f"📊 df_test shape: {df_test.shape}")
    
    return df_all, df_train, df_test


def split_df_all_by_cutoff(df_all):
    """Split df_all into before and after 2024-06-01"""
    print("\n" + "=" * 80)
    print("STEP 2: Splitting df_all by cutoff date (2024-06-01)")
    print("=" * 80)
    
    # Convert period to datetime if needed
    if 'period' in df_all.columns:
        if df_all['period'].dtype != 'int64':
            # Try to convert period to datetime
            df_all = df_all.copy()
            df_all['period_dt'] = pd.to_datetime(df_all['period'], format='%Y%m%d', errors='coerce')
        else:
            df_all = df_all.copy()
            df_all['period_dt'] = pd.to_datetime(df_all['period'].astype(str), format='%Y%m%d', errors='coerce')
        
        cutoff_date = pd.Timestamp('2024-06-01')
        df_before_cutoff = df_all[df_all['period_dt'] < cutoff_date].copy()
        df_after_cutoff = df_all[df_all['period_dt'] >= cutoff_date].copy()
        
        print(f"\n📅 Cutoff date: 2024-06-01")
        print(f"📊 df_before_cutoff shape: {df_before_cutoff.shape}")
        print(f"📊 df_after_cutoff shape: {df_after_cutoff.shape}")
        
        # Show period range
        if len(df_before_cutoff) > 0:
            print(f"   df_before_cutoff period range: {df_before_cutoff['period_dt'].min()} to {df_before_cutoff['period_dt'].max()}")
        if len(df_after_cutoff) > 0:
            print(f"   df_after_cutoff period range: {df_after_cutoff['period_dt'].min()} to {df_after_cutoff['period_dt'].max()}")
    else:
        print("❌ 'period' column not found in df_all")
        df_before_cutoff = pd.DataFrame()
        df_after_cutoff = pd.DataFrame()
    
    return df_before_cutoff, df_after_cutoff


def identify_feature_columns(df):
    """Identify feature columns (ending with '_current', '_augment', or containing 'change')"""
    feature_cols = []
    for col in df.columns:
        if col.endswith('_current') or col.endswith('_augment') or 'change' in col.lower():
            feature_cols.append(col)
    return feature_cols


def compare_key_identifiers(df1, df2, name1, name2):
    """Compare key identifier columns [cik, period, data_qtr] between two dataframes"""
    print("\n" + "=" * 80)
    print(f"STEP 3: Comparing key identifiers: {name1} vs {name2}")
    print("=" * 80)
    
    key_cols = ['cik', 'period', 'data_qtr']
    
    # Check if key columns exist
    missing_cols_1 = [col for col in key_cols if col not in df1.columns]
    missing_cols_2 = [col for col in key_cols if col not in df2.columns]
    
    if missing_cols_1:
        print(f"❌ Missing columns in {name1}: {missing_cols_1}")
        return None, None
    if missing_cols_2:
        print(f"❌ Missing columns in {name2}: {missing_cols_2}")
        return None, None
    
    # Create identifier sets
    df1_keys = df1[key_cols].drop_duplicates()
    df2_keys = df2[key_cols].drop_duplicates()
    
    print(f"\n📊 {name1}: {len(df1_keys):,} unique (cik, period, data_qtr) combinations")
    print(f"📊 {name2}: {len(df2_keys):,} unique (cik, period, data_qtr) combinations")
    
    # Convert to sets of tuples for comparison
    df1_key_set = set(df1_keys.apply(tuple, axis=1))
    df2_key_set = set(df2_keys.apply(tuple, axis=1))
    
    # Find differences
    only_in_df1 = df1_key_set - df2_key_set
    only_in_df2 = df2_key_set - df1_key_set
    in_both = df1_key_set & df2_key_set
    
    print(f"\n📊 Comparison results:")
    print(f"   Keys in both datasets: {len(in_both):,}")
    print(f"   Keys only in {name1}: {len(only_in_df1):,} ({len(only_in_df1)/len(df1_key_set)*100:.1f}%)")
    print(f"   Keys only in {name2}: {len(only_in_df2):,} ({len(only_in_df2)/len(df2_key_set)*100:.1f}%)")
    
    if len(only_in_df1) > 0:
        print(f"\n   Sample keys only in {name1} (first 10):")
        only_in_df1_df = pd.DataFrame(list(only_in_df1)[:10], columns=key_cols)
        print(only_in_df1_df.to_string())
    
    if len(only_in_df2) > 0:
        print(f"\n   Sample keys only in {name2} (first 10):")
        only_in_df2_df = pd.DataFrame(list(only_in_df2)[:10], columns=key_cols)
        print(only_in_df2_df.to_string())
    
    return df1_keys, df2_keys


def compare_feature_values(df1, df2, name1, name2, key_cols=['cik', 'period', 'data_qtr']):
    """Compare feature values for records that exist in both dataframes"""
    print("\n" + "=" * 80)
    print(f"STEP 4: Comparing feature values: {name1} vs {name2}")
    print("=" * 80)
    
    # Identify feature columns in both dataframes
    features1 = identify_feature_columns(df1)
    features2 = identify_feature_columns(df2)
    common_features = list(set(features1) & set(features2))
    
    print(f"\n📊 Feature column counts:")
    print(f"   {name1} features: {len(features1):,}")
    print(f"   {name2} features: {len(features2):,}")
    print(f"   Common features: {len(common_features):,}")
    
    if len(common_features) == 0:
        print("⚠️  No common features found to compare")
        return
    
    # Merge on key columns to find matching records
    df1_keyed = df1[key_cols + common_features].copy()
    df2_keyed = df2[key_cols + common_features].copy()
    
    # Create multi-index for merging
    df1_keyed['_merge_key'] = df1_keyed[key_cols].apply(tuple, axis=1)
    df2_keyed['_merge_key'] = df2_keyed[key_cols].apply(tuple, axis=1)
    
    # Merge to find common records
    merged = df1_keyed.merge(
        df2_keyed,
        on='_merge_key',
        how='inner',
        suffixes=('_1', '_2')
    )
    
    print(f"\n📊 Matching records: {len(merged):,}")
    
    if len(merged) == 0:
        print("⚠️  No matching records found to compare feature values")
        return
    
    # Compare feature values
    feature_differences = []
    
    for feature in common_features:
        col1 = f'{feature}_1'
        col2 = f'{feature}_2'
        
        if col1 not in merged.columns or col2 not in merged.columns:
            continue
        
        # Calculate differences
        # Handle NaN values - consider them as different if one is NaN and other is not
        both_notna = merged[col1].notna() & merged[col2].notna()
        both_na = merged[col1].isna() & merged[col2].isna()
        
        # Absolute difference for non-NaN pairs
        if both_notna.sum() > 0:
            abs_diff = (merged.loc[both_notna, col1] - merged.loc[both_notna, col2]).abs()
            mean_abs_diff = abs_diff.mean()
            max_abs_diff = abs_diff.max()
            
            # Relative difference (percentage)
            # Avoid division by zero
            col1_values = merged.loc[both_notna, col1].abs()
            relative_diff = abs_diff / (col1_values + 1e-10) * 100  # Add small epsilon to avoid division by zero
            mean_rel_diff = relative_diff.mean()
            max_rel_diff = relative_diff.max()
        else:
            mean_abs_diff = np.nan
            max_abs_diff = np.nan
            mean_rel_diff = np.nan
            max_rel_diff = np.nan
        
        # Count mismatches (NaN differences + non-zero differences)
        nan_mismatches = (~both_na & (merged[col1].isna() | merged[col2].isna())).sum()
        if both_notna.sum() > 0:
            value_mismatches = (abs_diff > 1e-10).sum()  # Small epsilon for floating point comparison
        else:
            value_mismatches = 0
        
        total_mismatches = nan_mismatches + value_mismatches
        mismatch_pct = (total_mismatches / len(merged)) * 100
        
        feature_differences.append({
            'feature': feature,
            'mean_abs_diff': mean_abs_diff,
            'max_abs_diff': max_abs_diff,
            'mean_rel_diff_pct': mean_rel_diff,
            'max_rel_diff_pct': max_rel_diff,
            'mismatches': total_mismatches,
            'mismatch_pct': mismatch_pct,
            'matching_pairs': both_notna.sum()
        })
    
    # Create DataFrame with differences
    diff_df = pd.DataFrame(feature_differences)
    diff_df = diff_df.sort_values('mismatch_pct', ascending=False)
    
    print(f"\n📊 Feature comparison summary:")
    print(f"   Total features compared: {len(diff_df):,}")
    print(f"   Features with any mismatches: {(diff_df['mismatch_pct'] > 0).sum():,}")
    print(f"   Features with >1% mismatches: {(diff_df['mismatch_pct'] > 1).sum():,}")
    print(f"   Features with >10% mismatches: {(diff_df['mismatch_pct'] > 10).sum():,}")
    
    # Show top features with most mismatches
    print(f"\n📊 Top 20 features with highest mismatch percentage:")
    top_mismatches = diff_df.head(20)[['feature', 'mismatch_pct', 'mean_abs_diff', 'max_abs_diff', 'mean_rel_diff_pct']]
    print(top_mismatches.to_string())
    
    # Show top features with highest absolute differences
    print(f"\n📊 Top 20 features with highest mean absolute difference:")
    top_abs_diff = diff_df.nlargest(20, 'mean_abs_diff')[['feature', 'mean_abs_diff', 'max_abs_diff', 'mismatch_pct']]
    print(top_abs_diff.to_string())
    
    return diff_df


def main():
    """Main investigation function"""
    print("🔍 Investigating Data Discrepancies")
    print("=" * 80)
    
    # Step 1: Load dataframes
    df_all, df_train, df_test = load_dataframes()
    
    # Step 2: Split df_all by cutoff
    df_before_cutoff, df_after_cutoff = split_df_all_by_cutoff(df_all)
    
    # Step 3: Compare key identifiers
    train_keys_1, train_keys_2 = compare_key_identifiers(
        df_before_cutoff, df_train, 
        'df_before_cutoff', 'df_train'
    )
    
    test_keys_1, test_keys_2 = compare_key_identifiers(
        df_after_cutoff, df_test,
        'df_after_cutoff', 'df_test'
    )
    
    # Step 4: Compare feature values
    if train_keys_1 is not None and train_keys_2 is not None:
        print("\n" + "=" * 80)
        print("COMPARING TRAINING DATA")
        print("=" * 80)
        compare_feature_values(
            df_before_cutoff, df_train,
            'df_before_cutoff', 'df_train'
        )
    
    if test_keys_1 is not None and test_keys_2 is not None:
        print("\n" + "=" * 80)
        print("COMPARING TEST DATA")
        print("=" * 80)
        compare_feature_values(
            df_after_cutoff, df_test,
            'df_after_cutoff', 'df_test'
        )
    
    print("\n" + "=" * 80)
    print("✅ Investigation complete")
    print("=" * 80)


def investigate_specific_feature(df1, df2, name1, name2, feature_name, key_cols=['cik', 'period', 'data_qtr']):
    """
    Investigate a specific feature that differs between two datasets.
    
    Args:
        df1, df2: DataFrames to compare
        name1, name2: Names for the dataframes (for printing)
        feature_name: Name of the feature column to investigate
        key_cols: Key columns for matching records
    """
    print("\n" + "=" * 80)
    print(f"INVESTIGATING SPECIFIC FEATURE: {feature_name}")
    print(f"Comparing: {name1} vs {name2}")
    print("=" * 80)
    
    # Check if feature exists in both datasets
    if feature_name not in df1.columns:
        print(f"❌ Feature '{feature_name}' not found in {name1}")
        return
    if feature_name not in df2.columns:
        print(f"❌ Feature '{feature_name}' not found in {name2}")
        return
    
    # Create merge keys - merge directly on key columns to preserve them
    df1_keyed = df1[key_cols + [feature_name]].copy()
    df2_keyed = df2[key_cols + [feature_name]].copy()
    
    # Rename feature columns before merge to avoid conflicts
    df1_keyed = df1_keyed.rename(columns={feature_name: f'{feature_name}_1'})
    df2_keyed = df2_keyed.rename(columns={feature_name: f'{feature_name}_2'})
    
    # Merge directly on key columns - this preserves them without suffixes
    merged = df1_keyed.merge(
        df2_keyed,
        on=key_cols,
        how='inner'
    )
    
    print(f"\n📊 Matching records: {len(merged):,}")
    
    if len(merged) == 0:
        print("⚠️  No matching records found")
        return
    
    # Identify records where values differ
    col1 = f'{feature_name}_1'
    col2 = f'{feature_name}_2'
    
    # Check for differences (accounting for NaN)
    both_notna = merged[col1].notna() & merged[col2].notna()
    both_na = merged[col1].isna() & merged[col2].isna()
    one_na = (~both_notna) & (~both_na)
    
    # For non-NaN pairs, check if values differ (with small epsilon for floating point)
    value_diff = pd.Series(False, index=merged.index)
    if both_notna.sum() > 0:
        value_diff.loc[both_notna] = (merged.loc[both_notna, col1] - merged.loc[both_notna, col2]).abs() > 1e-10
    
    # Records that differ
    differs = value_diff | one_na
    
    print(f"\n📊 Records with matching keys: {len(merged):,}")
    print(f"   Records where both values are NaN: {both_na.sum():,}")
    print(f"   Records where one value is NaN: {one_na.sum():,}")
    print(f"   Records where both values exist but differ: {value_diff.sum():,}")
    print(f"   Total records where values differ: {differs.sum():,} ({differs.sum()/len(merged)*100:.1f}%)")
    
    if differs.sum() == 0:
        print("✅ All matching records have the same values for this feature")
        return
    
    # Get detailed comparison of differing records
    diff_records = merged[differs].copy()
    
    print(f"\n📊 Detailed comparison of {len(diff_records)} differing records:")
    print(f"   Showing first 50 records...")
    
    # Create comparison DataFrame - use the extracted key columns and feature columns
    comparison_cols = key_cols + [col1, col2]
    comparison_df = diff_records[comparison_cols].copy()
    comparison_df.rename(columns={
        col1: f'{feature_name}_{name1}',
        col2: f'{feature_name}_{name2}'
    }, inplace=True)
    
    # Add difference column
    both_notna_diff = diff_records[col1].notna() & diff_records[col2].notna()
    comparison_df['abs_diff'] = np.nan
    comparison_df['rel_diff_pct'] = np.nan
    if both_notna_diff.sum() > 0:
        comparison_df.loc[both_notna_diff, 'abs_diff'] = (
            diff_records.loc[both_notna_diff, col1] - 
            diff_records.loc[both_notna_diff, col2]
        ).abs()
        # Relative difference
        col1_vals = diff_records.loc[both_notna_diff, col1].abs()
        comparison_df.loc[both_notna_diff, 'rel_diff_pct'] = (
            comparison_df.loc[both_notna_diff, 'abs_diff'] / (col1_vals + 1e-10) * 100
        )
    
    # Sort by absolute difference (largest first) - NaNs will be at the end by default
    comparison_df = comparison_df.sort_values('abs_diff', ascending=False)
    
    print(comparison_df.head(50).to_string())
    
    # Statistics
    print(f"\n📊 Statistics for differing values:")
    if both_notna_diff.sum() > 0:
        abs_diffs = comparison_df['abs_diff'].dropna()
        rel_diffs = comparison_df['rel_diff_pct'].dropna()
        
        print(f"   Absolute difference:")
        print(f"      Mean: {abs_diffs.mean():.6f}")
        print(f"      Median: {abs_diffs.median():.6f}")
        print(f"      Max: {abs_diffs.max():.6f}")
        print(f"      Min: {abs_diffs.min():.6f}")
        
        print(f"\n   Relative difference (%):")
        print(f"      Mean: {rel_diffs.mean():.2f}%")
        print(f"      Median: {rel_diffs.median():.2f}%")
        print(f"      Max: {rel_diffs.max():.2f}%")
        print(f"      Min: {rel_diffs.min():.2f}%")
    
    return comparison_df


def main_with_specific_feature():
    """Main function with specific feature investigation"""
    print("🔍 Investigating Data Discrepancies with Specific Feature Focus")
    print("=" * 80)
    
    # Step 1: Load dataframes
    df_all, df_train, df_test = load_dataframes()
    
    # Step 2: Split df_all by cutoff
    df_before_cutoff, df_after_cutoff = split_df_all_by_cutoff(df_all)
    
    # Investigate specific feature
    feature_name = 'OtherAssets_0qtrs_change_q1'
    
    print("\n" + "=" * 80)
    print("INVESTIGATING SPECIFIC FEATURE IN TRAINING DATA")
    print("=" * 80)
    investigate_specific_feature(
        df_after_cutoff, df_test,
        'df_after_cutoff', 'df_test',
        feature_name
    )
    
    print("\n" + "=" * 80)
    print("✅ Specific feature investigation complete")
    print("=" * 80)


def investigate_raw_sec_data(quarter_dir, target_cik, target_period):
    """
    Investigate raw SEC data in a specific quarter directory to check if a record exists.
    
    Args:
        quarter_dir (str): Path to the quarter directory (e.g., 'data/SEC_raw_2015_to_2025/2024q2')
        target_cik (int): CIK to search for
        target_period (int): Period to search for (YYYYMMDD format)
    """
    import os
    from config import DATA_BASE_DIR
    
    print("\n" + "=" * 80)
    print(f"INVESTIGATING RAW SEC DATA")
    print(f"Quarter Directory: {quarter_dir}")
    print(f"Target CIK: {target_cik}")
    print(f"Target Period: {target_period}")
    print("=" * 80)
    
    # Construct full path if relative path provided
    if not os.path.isabs(quarter_dir):
        quarter_dir = os.path.join(DATA_BASE_DIR, os.path.basename(quarter_dir))
    
    print(f"\n📁 Checking directory: {quarter_dir}")
    if not os.path.exists(quarter_dir):
        print(f"❌ Directory does not exist: {quarter_dir}")
        return
    
    sub_file = os.path.join(quarter_dir, 'sub.txt')
    num_file = os.path.join(quarter_dir, 'num.txt')
    
    # Check submissions file (sub.txt)
    print(f"\n🔍 Checking submissions file: {sub_file}")
    if os.path.exists(sub_file):
        print(f"   Loading submissions...")
        submissions = pd.read_csv(sub_file, sep='\t', low_memory=False, keep_default_na=False)
        print(f"   Total submissions: {len(submissions):,}")
        
        # Filter for target CIK
        cik_records = submissions[submissions['cik'] == target_cik].copy()
        print(f"   Records with CIK={target_cik}: {len(cik_records)}")
        
        if len(cik_records) > 0:
            print(f"\n   All records for CIK={target_cik}:")
            print(cik_records[['adsh', 'cik', 'name', 'form', 'period', 'fy', 'fp']].to_string(index=False))
            
            # Check period column type and convert target_period if needed
            period_dtype = cik_records['period'].dtype
            print(f"\n   Period column dtype: {period_dtype}")
            print(f"   Target period type: {type(target_period)}, value: {target_period}")
            
            # Convert target_period to match the period column type
            if period_dtype == 'object' or cik_records['period'].dtype == 'object':
                # Period is stored as string, convert target to string
                target_period_str = str(target_period)
                print(f"   Converting target_period to string: {target_period_str}")
                period_records = cik_records[cik_records['period'] == target_period_str]
            else:
                # Period is numeric, keep as is or convert if needed
                target_period_num = int(target_period) if isinstance(target_period, str) else target_period
                print(f"   Using target_period as numeric: {target_period_num}")
                period_records = cik_records[cik_records['period'] == target_period_num]
            if len(period_records) > 0:
                print(f"\n   ✅ FOUND {len(period_records)} submission(s) with period={target_period}:")
                print(period_records[['adsh', 'cik', 'name', 'form', 'period', 'fy', 'fp', 'fye']].to_string(index=False))
            else:
                print(f"\n   ❌ No submissions found with period={target_period}")
                unique_periods = sorted(cik_records['period'].unique())
                print(f"   Available periods for this CIK: {unique_periods}")
                print(f"   Period types: {[type(p).__name__ for p in unique_periods[:3]]}")  # Show first 3 types
                # Try alternative comparison methods
                if period_dtype == 'object':
                    # Try converting to int and comparing
                    try:
                        cik_records['period_int'] = pd.to_numeric(cik_records['period'], errors='coerce')
                        alt_records = cik_records[cik_records['period_int'] == target_period]
                        if len(alt_records) > 0:
                            print(f"   ✅ FOUND when comparing as integers: {len(alt_records)} record(s)")
                    except:
                        pass
        else:
            print(f"   ❌ No records found for CIK={target_cik}")
            print(f"   Checking all unique CIKs in this quarter...")
            print(f"   Total unique CIKs: {submissions['cik'].nunique():,}")
            if target_cik in submissions['cik'].values:
                print(f"   ⚠️  CIK exists but period filtering removed it")
            else:
                print(f"   ❌ CIK {target_cik} does not exist in this quarter")
    else:
        print(f"   ❌ File does not exist: {sub_file}")
    
    # Check numeric facts file (num.txt) if submission exists
    print(f"\n🔍 Checking numeric facts file: {num_file}")
    if os.path.exists(num_file):
        print(f"   Loading numeric facts...")
        # First, get the adsh for the target record if it exists
        if os.path.exists(sub_file):
            submissions = pd.read_csv(sub_file, sep='\t', low_memory=False, keep_default_na=False)
            target_submissions = submissions[
                (submissions['cik'] == target_cik) & 
                (submissions['period'] == target_period)
            ]
            
            # Check period column type and convert target_period if needed
            if len(submissions) > 0:
                period_dtype = submissions['period'].dtype
                if period_dtype == 'object' or submissions['period'].dtype == 'object':
                    target_period_str = str(target_period)
                    target_submissions = submissions[
                        (submissions['cik'] == target_cik) & 
                        (submissions['period'] == target_period_str)
                    ]
                else:
                    target_period_num = int(target_period) if isinstance(target_period, str) else target_period
                    target_submissions = submissions[
                        (submissions['cik'] == target_cik) & 
                        (submissions['period'] == target_period_num)
                    ]
            
            if len(target_submissions) > 0:
                target_adsh_list = target_submissions['adsh'].tolist()
                print(f"   Found {len(target_adsh_list)} submission(s) (adsh) for this (cik, period)")
                
                # Read numeric facts (this can be large, so we'll filter while reading)
                print(f"   Loading numeric facts for these submissions...")
                numeric_facts = pd.read_csv(num_file, sep='\t', low_memory=False)
                print(f"   Total numeric facts: {len(numeric_facts):,}")
                
                # Filter for target adsh
                adsh_facts = numeric_facts[numeric_facts['adsh'].isin(target_adsh_list)]
                print(f"   Numeric facts for target submissions: {len(adsh_facts):,}")
                
                if len(adsh_facts) > 0:
                    print(f"\n   ✅ FOUND numeric facts for this (cik, period) combination:")
                    print(f"   Sample records (first 10):")
                    print(adsh_facts[['adsh', 'tag', 'ddate', 'qtrs', 'value']].head(10).to_string(index=False))
                    print(f"   Unique tags: {adsh_facts['tag'].nunique()}")
                else:
                    print(f"   ❌ No numeric facts found for these submissions")
            else:
                print(f"   ⚠️  Skipping numeric facts check (no matching submissions found)")
        else:
            print(f"   ⚠️  Cannot check numeric facts (submissions file not available)")
    else:
        print(f"   ❌ File does not exist: {num_file}")
    
    print("\n" + "=" * 80)
    print("Investigation complete")
    print("=" * 80)


if __name__ == "__main__":

    # Investigate raw SEC data for the oddity
    investigate_raw_sec_data(
        quarter_dir='2024q2',
        target_cik=1532619,
        target_period=20241231
    )
    
    # Uncomment to run full investigation
    # main()

    # Run specific feature investigation
    # main_with_specific_feature()
    

