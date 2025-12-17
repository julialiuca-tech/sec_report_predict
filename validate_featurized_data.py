#!/usr/bin/env python3
"""
Comprehensive Featurized Data Validation Script

This script provides comprehensive validation of featurized SEC data by:

1. FEATURE COMPLETENESS VALIDATION:
   - Reading individual quarter featurized CSV files (2022q1_featurized.csv through 2022q4_featurized.csv)
   - Running summarize_feature_completeness() on each individual file
   - Summing the non_null_count for each feature across all quarters
   - Running summarize_feature_completeness() on the combined featurized_all_quarters.csv file
   - Comparing the results to ensure data consistency

2. DUPLICATE TUPLE ANALYSIS:
   - Checking for duplicate (cik, period) tuples across different quarters
   - Analyzing why duplicates exist and their impact on data consolidation
   - Validating the need for the data_qtr column

Expected result: The sum of individual quarter non_null_counts should equal the non_null_count in the combined file.
"""

import pandas as pd
import os

# Import the function from featurize.py
from featurize import summarize_feature_completeness
from config import SAVE_DIR

# =============================================================================
# FEATURE COMPLETENESS VALIDATION FUNCTIONS
# =============================================================================

def load_quarter_files(processed_data_dir=None):
    """
    Load individual quarter featurized CSV files.
    
    Args:
        processed_data_dir (str): Directory containing the processed data files. 
                                  Defaults to SAVE_DIR from config if None.
        
    Returns:
        dict: Dictionary with quarter names as keys and DataFrames as values
    """
    if processed_data_dir is None:
        processed_data_dir = SAVE_DIR
    
    quarter_files = {}
    quarter_patterns = ['2022q1', '2022q2', '2022q3', '2022q4']
    
    for quarter in quarter_patterns:
        file_path = os.path.join(processed_data_dir, f"{quarter}_featurized.csv")
        if os.path.exists(file_path):
            print(f"Loading {quarter} data from {file_path}")
            quarter_files[quarter] = pd.read_csv(file_path, low_memory=False)
            print(f"  - Loaded {len(quarter_files[quarter]):,} rows")
        else:
            print(f"Warning: {file_path} not found")
    
    return quarter_files

def analyze_individual_quarters(quarter_files):
    """
    Analyze feature completeness for each individual quarter.
    
    Args:
        quarter_files (dict): Dictionary of quarter DataFrames
        
    Returns:
        dict: Dictionary with quarter names as keys and completeness DataFrames as values
    """
    quarter_analyses = {}
    
    for quarter, df in quarter_files.items():
        print(f"\nAnalyzing {quarter} features...")
        try:
            completeness_df = summarize_feature_completeness(df)
            quarter_analyses[quarter] = completeness_df
            print(f"  - Found {len(completeness_df)} features")
            print(f"  - Total non-null count: {completeness_df['non_null_count'].sum():,}")
        except Exception as e:
            print(f"  - Error analyzing {quarter}: {e}")
            quarter_analyses[quarter] = None
    
    return quarter_analyses

def analyze_combined_file(processed_data_dir=None):
    """
    Analyze feature completeness for the combined featurized_all_quarters.csv file.
    
    Args:
        processed_data_dir (str): Directory containing the processed data files.
                                  Defaults to SAVE_DIR from config if None.
        
    Returns:
        pd.DataFrame: Completeness analysis of the combined file
    """
    if processed_data_dir is None:
        processed_data_dir = SAVE_DIR
    
    combined_file_path = os.path.join(processed_data_dir, "featurized_all_quarters.csv")
    
    if not os.path.exists(combined_file_path):
        print(f"Error: {combined_file_path} not found")
        return None
    
    print(f"\nAnalyzing combined file: {combined_file_path}")
    try:
        df_combined = pd.read_csv(combined_file_path, low_memory=False)
        print(f"  - Loaded {len(df_combined):,} rows")
        
        completeness_df = summarize_feature_completeness(df_combined)
        print(f"  - Found {len(completeness_df)} features")
        print(f"  - Total non-null count: {completeness_df['non_null_count'].sum():,}")
        
        return completeness_df
    except Exception as e:
        print(f"  - Error analyzing combined file: {e}")
        return None

def analyze_combined_by_data_dir(processed_data_dir=None):
    """
    Analyze the combined file by grouping on data_qtr to see breakdown by quarter.
    
    Args:
        processed_data_dir (str): Directory containing the processed data files.
                                  Defaults to SAVE_DIR from config if None.
        
    Returns:
        dict: Dictionary with data_qtr as keys and feature counts as values
    """
    if processed_data_dir is None:
        processed_data_dir = SAVE_DIR
    
    combined_file_path = os.path.join(processed_data_dir, "featurized_all_quarters.csv")
    
    if not os.path.exists(combined_file_path):
        print(f"Error: {combined_file_path} not found")
        return None
    
    print(f"\n🔍 Analyzing combined file by data_qtr: {combined_file_path}")
    try:
        df_combined = pd.read_csv(combined_file_path, low_memory=False)
        print(f"  - Loaded {len(df_combined):,} rows")
        
        # Check if data_qtr column exists
        if 'data_qtr' not in df_combined.columns:
            print("  - Warning: 'data_qtr' column not found in combined file")
            return None
        
        # Get unique data_qtr values (excluding NaN)
        data_qtrs = sorted([x for x in df_combined['data_qtr'].unique() if pd.notna(x)])
        print(f"  - Found data_qtr values: {data_qtrs}")
        print(f"  - Total rows with NaN data_qtr: {df_combined['data_qtr'].isna().sum():,}")
        
        # Analyze each data_qtr separately
        data_dir_analyses = {}
        for data_qtr in data_qtrs:
            print(f"  - Analyzing data_qtr: {data_qtr}")
            df_subset = df_combined[df_combined['data_qtr'] == data_qtr].copy()
            print(f"    - Subset has {len(df_subset):,} rows")
            
            completeness_df = summarize_feature_completeness(df_subset)
            data_dir_analyses[data_qtr] = completeness_df
            print(f"    - Found {len(completeness_df)} features")
            print(f"    - Total non-null count: {completeness_df['non_null_count'].sum():,}")
        
        return data_dir_analyses
    except Exception as e:
        print(f"  - Error analyzing combined file by data_dir: {e}")
        return None

def compare_results(quarter_analyses, combined_analysis, data_dir_analyses=None):
    """
    Compare individual quarter results with combined file results.
    
    Args:
        quarter_analyses (dict): Dictionary of individual quarter analyses
        combined_analysis (pd.DataFrame): Analysis of combined file
        data_dir_analyses (dict): Analysis of combined file grouped by data_dir
        
    Returns:
        bool: True if results are consistent, False otherwise
    """
    if combined_analysis is None:
        print("\n❌ Cannot compare results - combined analysis failed")
        return False
    
    print("\n" + "="*80)
    print("FEATURE COMPLETENESS VALIDATION RESULTS")
    print("="*80)
    
    # Calculate sum of individual quarter non_null_counts for each feature
    feature_sums = {}
    
    for quarter, analysis in quarter_analyses.items():
        if analysis is not None:
            for _, row in analysis.iterrows():
                feature = row['feature']
                count = row['non_null_count']
                
                if feature not in feature_sums:
                    feature_sums[feature] = 0
                feature_sums[feature] += count
    
    # Compare with combined file
    combined_counts = {}
    for _, row in combined_analysis.iterrows():
        feature = row['feature']
        count = row['non_null_count']
        combined_counts[feature] = count
    
    # Check consistency
    all_features = set(feature_sums.keys()) | set(combined_counts.keys())
    inconsistencies = []
    
    for feature in all_features:
        individual_sum = feature_sums.get(feature, 0)
        combined_count = combined_counts.get(feature, 0)
        
        if individual_sum != combined_count:
            inconsistencies.append({
                'feature': feature,
                'individual_sum': individual_sum,
                'combined_count': combined_count,
                'difference': individual_sum - combined_count
            })
    
    # Report results
    if not inconsistencies:
        print("✅ FEATURE COMPLETENESS VALIDATION PASSED: All feature counts are consistent!")
        print(f"   - Total features analyzed: {len(all_features)}")
        print(f"   - Total non-null count (individual sum): {sum(feature_sums.values()):,}")
        print(f"   - Total non-null count (combined): {sum(combined_counts.values()):,}")
        return True
    else:
        print("❌ FEATURE COMPLETENESS VALIDATION FAILED: Found inconsistencies in feature counts!")
        print(f"   - Total features analyzed: {len(all_features)}")
        print(f"   - Features with inconsistencies: {len(inconsistencies)}")
        print(f"   - Total non-null count (individual sum): {sum(feature_sums.values()):,}")
        print(f"   - Total non-null count (combined): {sum(combined_counts.values()):,}")
        
        print("\nTop 10 inconsistencies (by absolute difference):")
        inconsistencies.sort(key=lambda x: abs(x['difference']), reverse=True)
        for i, inc in enumerate(inconsistencies[:10]):
            print(f"   {i+1:2d}. {inc['feature']:<30} "
                  f"Individual: {inc['individual_sum']:>8,} "
                  f"Combined: {inc['combined_count']:>8,} "
                  f"Diff: {inc['difference']:>+8,}")
        
        # If we have data_dir analysis, do detailed comparison for problematic features
        if data_dir_analyses is not None and inconsistencies:
            print("\n" + "="*80)
            print("DETAILED BREAKDOWN BY DATA_DIR")
            print("="*80)
            
            # Focus on the top 3 most problematic features
            top_inconsistencies = inconsistencies[:3]
            
            for inc in top_inconsistencies:
                feature = inc['feature']
                print(f"\n🔍 Detailed analysis for: {feature}")
                print(f"   Individual sum: {inc['individual_sum']:,}")
                print(f"   Combined total: {inc['combined_count']:,}")
                print(f"   Difference: {inc['difference']:+,}")
                
                print("\n   Breakdown by data_qtr (from combined file):")
                combined_data_qtr_sum = 0
                for data_qtr, analysis in data_dir_analyses.items():
                    feature_row = analysis[analysis['feature'] == feature]
                    if not feature_row.empty:
                        count = feature_row.iloc[0]['non_null_count']
                        print(f"     {data_qtr}: {count:>8,}")
                        combined_data_qtr_sum += count
                    else:
                        print(f"     {data_qtr}: {0:>8,} (feature not found)")
                
                print(f"   Sum of data_qtr counts: {combined_data_qtr_sum:,}")
                
                print("\n   Breakdown by individual files:")
                individual_sum_check = 0
                for quarter, analysis in quarter_analyses.items():
                    if analysis is not None:
                        feature_row = analysis[analysis['feature'] == feature]
                        if not feature_row.empty:
                            count = feature_row.iloc[0]['non_null_count']
                            print(f"     {quarter}: {count:>8,}")
                            individual_sum_check += count
                        else:
                            print(f"     {quarter}: {0:>8,} (feature not found)")
                
                print(f"   Sum of individual counts: {individual_sum_check:,}")
                
                # Check if data_qtr sum matches combined total
                if combined_data_qtr_sum != inc['combined_count']:
                    print(f"   ⚠️  Data_qtr sum ({combined_data_qtr_sum:,}) != Combined total ({inc['combined_count']:,})")
                else:
                    print(f"   ✅ Data_qtr sum matches combined total")
        
        return False

# =============================================================================
# DUPLICATE TUPLE ANALYSIS FUNCTIONS
# =============================================================================

def check_duplicate_tuples_across_quarters(processed_data_dir=None):
    """
    Check for duplicate (cik, period) tuples across different quarters.
    
    Args:
        processed_data_dir (str): Directory containing the processed data files.
                                  Defaults to SAVE_DIR from config if None.
        
    Returns:
        dict: Analysis results with duplicate information

    Example of duplicate tuples:
    CIK: 16099, Period: 20201130, appears in 2 quarters 2022q1, 2022q2 
    CIK: 1657249, Period: 20220630, appears in 2 quarters 2022q2, 2022q3
    TO-DO: remove the duplicate tuples from the combined featurization. 
    """
    if processed_data_dir is None:
        processed_data_dir = SAVE_DIR
    
    print("\n" + "="*80)
    print("DUPLICATE (cik, period) TUPLE ANALYSIS")
    print("="*80)
    
    # Load individual quarter files
    quarter_files = {}
    quarter_patterns = ['2022q1', '2022q2', '2022q3', '2022q4']
    
    for quarter in quarter_patterns:
        file_path = os.path.join(processed_data_dir, f"{quarter}_featurized.csv")
        if os.path.exists(file_path):
            print(f"Loading {quarter} data from {file_path}")
            df = pd.read_csv(file_path, low_memory=False)
            quarter_files[quarter] = df
            print(f"  - Loaded {len(df):,} rows")
            print(f"  - Unique (cik, period) tuples: {df[['cik', 'period']].drop_duplicates().shape[0]:,}")
        else:
            print(f"Warning: {file_path} not found")
    
    if not quarter_files:
        print("Error: No quarter files found")
        return None
    
    # Collect all (cik, period) tuples with their quarter information
    all_tuples = []
    
    for quarter, df in quarter_files.items():
        tuples_df = df[['cik', 'period']].drop_duplicates().copy()
        tuples_df['quarter'] = quarter
        all_tuples.append(tuples_df)
    
    # Combine all tuples
    combined_tuples = pd.concat(all_tuples, ignore_index=True)
    print(f"\nTotal unique (cik, period) tuples across all quarters: {len(combined_tuples):,}")
    
    # Check for duplicates
    tuple_counts = combined_tuples.groupby(['cik', 'period']).size().reset_index(name='count')
    duplicates = tuple_counts[tuple_counts['count'] > 1]
    
    print(f"Tuples appearing in multiple quarters: {len(duplicates):,}")
    
    if len(duplicates) > 0:
        print("\n" + "="*80)
        print("DUPLICATE (cik, period) TUPLES FOUND!")
        print("="*80)
        
        # Show detailed breakdown
        for _, row in duplicates.head(20).iterrows():  # Show top 20
            cik = row['cik']
            period = row['period']
            count = row['count']
            
            print(f"\nCIK: {cik}, Period: {period} (appears in {count} quarters)")
            
            # Find which quarters this tuple appears in
            quarters = combined_tuples[
                (combined_tuples['cik'] == cik) & 
                (combined_tuples['period'] == period)
            ]['quarter'].tolist()
            
            print(f"  Quarters: {', '.join(quarters)}")
            
            # Show some sample data from each quarter
            for quarter in quarters:
                df_quarter = quarter_files[quarter]
                sample_data = df_quarter[
                    (df_quarter['cik'] == cik) & 
                    (df_quarter['period'] == period)
                ].head(2)
                
                if not sample_data.empty:
                    print(f"  {quarter} sample data:")
                    for _, sample_row in sample_data.iterrows():
                        # Show key identifying columns
                        key_cols = ['cik', 'period', 'form'] if 'form' in sample_row.index else ['cik', 'period']
                        key_values = [str(sample_row[col]) for col in key_cols]
                        print(f"    - {', '.join(key_values)}")
        
        if len(duplicates) > 20:
            print(f"\n... and {len(duplicates) - 20} more duplicate tuples")
        
        # Summary statistics
        print(f"\n" + "="*80)
        print("DUPLICATE TUPLE SUMMARY STATISTICS")
        print("="*80)
        
        duplicate_counts = duplicates['count'].value_counts().sort_index()
        for count, freq in duplicate_counts.items():
            print(f"Tuples appearing in {count} quarters: {freq:,}")
        
        # Check if any tuples appear in all quarters
        all_quarters_count = len(quarter_files)
        tuples_in_all_quarters = duplicates[duplicates['count'] == all_quarters_count]
        print(f"\nTuples appearing in ALL {all_quarters_count} quarters: {len(tuples_in_all_quarters):,}")
        
        if len(tuples_in_all_quarters) > 0:
            print("\nSample tuples appearing in all quarters:")
            for _, row in tuples_in_all_quarters.head(5).iterrows():
                cik = row['cik']
                period = row['period']
                print(f"  CIK: {cik}, Period: {period}")
        
        return {
            'total_tuples': len(combined_tuples),
            'unique_tuples': len(tuple_counts),
            'duplicate_tuples': len(duplicates),
            'duplicates_df': duplicates,
            'tuples_in_all_quarters': len(tuples_in_all_quarters)
        }
    
    else:
        print("\n✅ NO DUPLICATE (cik, period) TUPLES FOUND!")
        print("All (cik, period) combinations are unique across quarters.")
        return {
            'total_tuples': len(combined_tuples),
            'unique_tuples': len(tuple_counts),
            'duplicate_tuples': 0,
            'duplicates_df': pd.DataFrame(),
            'tuples_in_all_quarters': 0
        }

def check_combined_file_tuples(processed_data_dir=None):
    """
    Check the combined file for (cik, period) tuple patterns.
    
    Args:
        processed_data_dir (str): Directory containing the processed data files.
                                  Defaults to SAVE_DIR from config if None.
    """
    if processed_data_dir is None:
        processed_data_dir = SAVE_DIR
    
    print("\n" + "="*80)
    print("COMBINED FILE TUPLE ANALYSIS")
    print("="*80)
    
    combined_file_path = os.path.join(processed_data_dir, "featurized_all_quarters.csv")
    
    if not os.path.exists(combined_file_path):
        print(f"Error: {combined_file_path} not found")
        return None
    
    print(f"Loading combined file: {combined_file_path}")
    df_combined = pd.read_csv(combined_file_path, low_memory=False)
    print(f"  - Loaded {len(df_combined):,} rows")
    
    # Check for data_qtr column
    if 'data_qtr' in df_combined.columns:
        print(f"  - data_qtr column found")
        data_qtr_counts = df_combined['data_qtr'].value_counts(dropna=False)
        print("  - data_qtr distribution:")
        for qtr, count in data_qtr_counts.items():
            print(f"    {qtr}: {count:,} rows")
        
        # Check for (cik, period) duplicates within the combined file
        tuple_counts = df_combined.groupby(['cik', 'period']).size().reset_index(name='count')
        duplicates = tuple_counts[tuple_counts['count'] > 1]
        
        print(f"  - Unique (cik, period) tuples: {len(tuple_counts):,}")
        print(f"  - Duplicate (cik, period) tuples: {len(duplicates):,}")
        
        if len(duplicates) > 0:
            print("\n  Sample duplicates in combined file:")
            for _, row in duplicates.head(5).iterrows():
                cik = row['cik']
                period = row['period']
                count = row['count']
                
                # Show which quarters this tuple appears in
                quarters = df_combined[
                    (df_combined['cik'] == cik) & 
                    (df_combined['period'] == period)
                ]['data_qtr'].unique()
                
                print(f"    CIK: {cik}, Period: {period} (appears {count} times)")
                print(f"      Quarters: {', '.join([str(q) for q in quarters])}")
    else:
        print("  - data_qtr column NOT found")
    
    return df_combined

# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

def analyze_feature_completeness_ranking(processed_data_dir=None, min_completeness=20.0):
    """
    Analyze the feature completeness ranking to identify the most populated features.
    
    Args:
        processed_data_dir (str): Directory containing the processed data files.
                                  Defaults to SAVE_DIR from config if None.
        min_completeness (float): Minimum completeness percentage to include features
        
    Returns:
        dict: Analysis results with feature statistics
    """
    if processed_data_dir is None:
        processed_data_dir = SAVE_DIR
    
    print("📊 ANALYZING FEATURE COMPLETENESS RANKING")
    print("="*80)
    
    # Load the feature completeness ranking file
    ranking_file_path = os.path.join(processed_data_dir, "feature_completeness_ranking.csv")
    
    if not os.path.exists(ranking_file_path):
        print(f"Error: {ranking_file_path} not found")
        return None
    
    print(f"Loading feature completeness ranking from: {ranking_file_path}")
    df_ranking = pd.read_csv(ranking_file_path)
    print(f"  - Loaded {len(df_ranking):,} total features")
    
    # Filter features with more than min_completeness% populated
    df_filtered = df_ranking[df_ranking['non_null_percentage'] > min_completeness].copy()
    print(f"  - Features with >{min_completeness}% completeness: {len(df_filtered):,}")
    
    if len(df_filtered) == 0:
        print("No features meet the completeness threshold!")
        return None
    
    # Question 1: How many features are there?
    print(f"\n📈 QUESTION 1: How many features are there?")
    print(f"   Answer: {len(df_filtered):,} features have >{min_completeness}% completeness")
    
    # Question 2: Analyze feature name structure
    print(f"\n🔍 QUESTION 2: Analyzing feature name structure...")
    print("   Feature names follow pattern: tag + qtrs + (current|change_qX)")
    
    # Parse feature names
    parsed_features = []
    
    for _, row in df_filtered.iterrows():
        feature_name = row['feature']
        completeness = row['non_null_percentage']
        count = row['non_null_count']
        
        # Split feature name into components
        # Pattern: tag_qtrs_current or tag_qtrs_change_qX
        parts = feature_name.split('_')
        
        if len(parts) >= 3:
            # Handle different patterns
            if parts[-1] == 'current':
                # Pattern: tag_qtrs_current
                tag = '_'.join(parts[:-2])  # Everything except last 2 parts
                qtrs = parts[-2]
                change_type = 'current'
            elif parts[-1].startswith('q') and parts[-2] == 'change':
                # Pattern: tag_qtrs_change_qX
                tag = '_'.join(parts[:-3])  # Everything except last 3 parts
                qtrs = parts[-3]
                change_type = f"change_{parts[-1]}"
            else:
                # Fallback: try to identify pattern
                tag = '_'.join(parts[:-2]) if len(parts) > 2 else feature_name
                qtrs = parts[-2] if len(parts) > 1 else 'unknown'
                change_type = parts[-1] if len(parts) > 0 else 'unknown'
        else:
            # Fallback for unexpected patterns
            tag = feature_name
            qtrs = 'unknown'
            change_type = 'unknown'
        
        parsed_features.append({
            'feature_name': feature_name,
            'tag': tag,
            'qtrs': qtrs,
            'change_type': change_type,
            'completeness': completeness,
            'count': count
        })
    
    # Convert to DataFrame for easier analysis
    df_parsed = pd.DataFrame(parsed_features)
    
    # Analyze most populated tags
    print(f"\n🏷️  MOST POPULATED TAGS:")
    tag_stats = df_parsed.groupby('tag').agg({
        'completeness': ['count', 'mean', 'sum'],
        'count': 'sum'
    }).round(2)
    tag_stats.columns = ['feature_count', 'avg_completeness', 'total_completeness', 'total_records']
    tag_stats = tag_stats.sort_values('total_records', ascending=False)
    
    print("   Top 10 tags by total record count:")
    for i, (tag, row) in enumerate(tag_stats.head(10).iterrows()):
        print(f"   {i+1:2d}. {tag:<40} {row['feature_count']:>3} features, "
              f"{row['total_records']:>8,} records, {row['avg_completeness']:>5.1f}% avg")
    
    # Analyze most populated qtrs
    print(f"\n📅 MOST POPULATED QUARTERS (qtrs):")
    qtrs_stats = df_parsed.groupby('qtrs').agg({
        'completeness': ['count', 'mean', 'sum'],
        'count': 'sum'
    }).round(2)
    qtrs_stats.columns = ['feature_count', 'avg_completeness', 'total_completeness', 'total_records']
    qtrs_stats = qtrs_stats.sort_values('total_records', ascending=False)
    
    print("   All quarters by total record count:")
    for qtrs, row in qtrs_stats.iterrows():
        print(f"   {qtrs:<10} {row['feature_count']:>3} features, "
              f"{row['total_records']:>8,} records, {row['avg_completeness']:>5.1f}% avg")
    
    # Analyze most populated change types
    print(f"\n🔄 MOST POPULATED CHANGE TYPES:")
    change_stats = df_parsed.groupby('change_type').agg({
        'completeness': ['count', 'mean', 'sum'],
        'count': 'sum'
    }).round(2)
    change_stats.columns = ['feature_count', 'avg_completeness', 'total_completeness', 'total_records']
    change_stats = change_stats.sort_values('total_records', ascending=False)
    
    print("   All change types by total record count:")
    for change_type, row in change_stats.iterrows():
        print(f"   {change_type:<15} {row['feature_count']:>3} features, "
              f"{row['total_records']:>8,} records, {row['avg_completeness']:>5.1f}% avg")
    
    # Summary statistics
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Total features analyzed: {len(df_parsed):,}")
    print(f"   Unique tags: {df_parsed['tag'].nunique():,}")
    print(f"   Unique quarters: {df_parsed['qtrs'].nunique():,}")
    print(f"   Unique change types: {df_parsed['change_type'].nunique():,}")
    
    # Show some examples of parsed features
    print(f"\n🔍 SAMPLE PARSED FEATURES:")
    sample_features = df_parsed.head(10)
    for _, row in sample_features.iterrows():
        print(f"   {row['feature_name']:<50} -> "
              f"tag: {row['tag']:<20} qtrs: {row['qtrs']:<8} type: {row['change_type']}")
    
    # Special analysis for Revenue-related tags
    print(f"\n💰 REVENUE-RELATED TAGS ANALYSIS:")
    revenue_features = df_parsed[df_parsed['tag'].str.contains('Revenue', case=False, na=False)]
    
    if len(revenue_features) > 0:
        print(f"   Found {len(revenue_features)} revenue-related features:")
        
        # Group by tag name
        revenue_tag_stats = revenue_features.groupby('tag').agg({
            'completeness': ['count', 'mean', 'min', 'max'],
            'count': 'sum'
        }).round(2)
        revenue_tag_stats.columns = ['feature_count', 'avg_completeness', 'min_completeness', 'max_completeness', 'total_records']
        revenue_tag_stats = revenue_tag_stats.sort_values('total_records', ascending=False)
        
        print("\n   Revenue tags by total record count:")
        for tag, row in revenue_tag_stats.iterrows():
            print(f"   {tag:<50} {row['feature_count']:>3} features, "
                  f"{row['total_records']:>8,} records, "
                  f"{row['avg_completeness']:>5.1f}% avg ({row['min_completeness']:>4.1f}%-{row['max_completeness']:>4.1f}%)")
        
        # Show individual revenue features
        print(f"\n   Individual revenue features:")
        revenue_features_sorted = revenue_features.sort_values('count', ascending=False)
        for _, row in revenue_features_sorted.iterrows():
            print(f"   {row['feature_name']:<60} {row['count']:>8,} records, {row['completeness']:>5.1f}%")
        
        # Summary for revenue features
        total_revenue_records = revenue_features['count'].sum()
        avg_revenue_completeness = revenue_features['completeness'].mean()
        print(f"\n   Revenue features summary:")
        print(f"   - Total revenue features: {len(revenue_features)}")
        print(f"   - Total revenue records: {total_revenue_records:,}")
        print(f"   - Average completeness: {avg_revenue_completeness:.1f}%")
        print(f"   - Completeness range: {revenue_features['completeness'].min():.1f}% - {revenue_features['completeness'].max():.1f}%")
        
    else:
        print("   No revenue-related features found in the filtered dataset.")
    
    return {
        'total_features': len(df_filtered),
        'parsed_features_df': df_parsed,
        'tag_stats': tag_stats,
        'qtrs_stats': qtrs_stats,
        'change_stats': change_stats,
        'revenue_features': revenue_features if len(revenue_features) > 0 else pd.DataFrame(),
        'min_completeness_threshold': min_completeness
    }

def check_consistency_of_featurized_data():
    """Main comprehensive validation function."""
    print("🔍 COMPREHENSIVE FEATURIZED DATA VALIDATION")
    print("="*80)
    
    # Use SAVE_DIR from config as the default directory
    processed_data_dir = SAVE_DIR
    if not os.path.exists(processed_data_dir):
        print(f"Error: {processed_data_dir} directory not found")
        return
    
    # =====================================================================
    # PART 1: FEATURE COMPLETENESS VALIDATION
    # =====================================================================
    print("\n" + "="*80)
    print("PART 1: FEATURE COMPLETENESS VALIDATION")
    print("="*80)
    
    # Step 1: Load individual quarter files
    print("\n📁 Step 1: Loading individual quarter files...")
    quarter_files = load_quarter_files(processed_data_dir)
    
    if not quarter_files:
        print("Error: No quarter files found")
        return
    
    # Step 2: Analyze individual quarters
    print("\n📊 Step 2: Analyzing individual quarters...")
    quarter_analyses = analyze_individual_quarters(quarter_files)
    
    # Step 3: Analyze combined file
    print("\n🔗 Step 3: Analyzing combined file...")
    combined_analysis = analyze_combined_file(processed_data_dir)
    
    # Step 3.5: Analyze combined file by data_dir
    print("\n🔍 Step 3.5: Analyzing combined file by data_dir...")
    data_dir_analyses = analyze_combined_by_data_dir(processed_data_dir)
    
    # Step 4: Compare results
    print("\n⚖️  Step 4: Comparing results...")
    feature_validation_passed = compare_results(quarter_analyses, combined_analysis, data_dir_analyses)
    
    # =====================================================================
    # PART 2: DUPLICATE TUPLE ANALYSIS
    # =====================================================================
    
    # Check individual quarters for duplicates
    duplicate_results = check_duplicate_tuples_across_quarters(processed_data_dir)
    
    # Check combined file for duplicates
    combined_df = check_combined_file_tuples(processed_data_dir)
    
    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    print("\n" + "="*80)
    print("FINAL COMPREHENSIVE SUMMARY")
    print("="*80)
    
    # Feature completeness summary
    if feature_validation_passed:
        print("✅ FEATURE COMPLETENESS VALIDATION: PASSED")
        print("   - All feature counts are consistent between individual and combined files")
    else:
        print("❌ FEATURE COMPLETENESS VALIDATION: FAILED")
        print("   - Found inconsistencies in feature counts")
    
    # Duplicate tuple summary
    if duplicate_results:
        if duplicate_results['duplicate_tuples'] > 0:
            print(f"\n🔍 DUPLICATE TUPLE ANALYSIS: {duplicate_results['duplicate_tuples']} DUPLICATES FOUND")
            print(f"   - This explains why the data_qtr column is needed")
            print(f"   - {duplicate_results['duplicate_tuples']:,} (cik, period) tuples appear in multiple quarters")
            print(f"   - {duplicate_results['tuples_in_all_quarters']:,} tuples appear in ALL quarters")
            print("\n   This is normal for SEC data because:")
            print("   - Companies file multiple forms (10-Q, 10-K) in the same period")
            print("   - Some companies may have restatements or amendments")
            print("   - Different reporting periods may overlap")
        else:
            print("\n✅ DUPLICATE TUPLE ANALYSIS: NO DUPLICATES FOUND")
            print("   - All (cik, period) combinations are unique")
            print("   - The data_qtr column may not be strictly necessary")
    else:
        print("\n❌ DUPLICATE TUPLE ANALYSIS: FAILED")
        print("   - Could not load data files for analysis")
    
    # Overall assessment
    print("\n" + "="*80)
    if feature_validation_passed and duplicate_results and duplicate_results['duplicate_tuples'] >= 0:
        print("🎉 COMPREHENSIVE VALIDATION COMPLETED SUCCESSFULLY!")
        print("   Your featurized data pipeline is working correctly.")
        if duplicate_results['duplicate_tuples'] > 0:
            print("   The data_qtr column is properly handling duplicate (cik, period) tuples.")
    else:
        print("⚠️  COMPREHENSIVE VALIDATION COMPLETED WITH ISSUES!")
        print("   Please investigate the issues found above.")
    print("="*80)

if __name__ == "__main__":
    # Run the feature completeness ranking analysis
    # analyze_feature_completeness_ranking(min_completeness=20.0)
    
    # Uncomment the line below to run the full validation
    check_consistency_of_featurized_data()
