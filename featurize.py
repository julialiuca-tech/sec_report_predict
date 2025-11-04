from utility_data import load_and_join_sec_xbrl_data, print_featurization, read_tags_to_featurize, get_cik_ticker_mapping
from config import (
    SAVE_DIR, DATA_BASE_DIR, FEATURIZED_ALL_QUARTERS_FILE, FEATURE_COMPLETENESS_RANKING_FILE,
    FEATURIZED_SIMPLIFIED_FILE, QUARTER_FEATURIZED_PATTERN, DEFAULT_K_TOP_TAGS,
    DEFAULT_MIN_COMPLETENESS, DEFAULT_DEBUG_FLAG, DEFAULT_N_QUARTERS_HISTORY_COMP
)
import pandas as pd
import numpy as np
import os

# =============================================================================
# QUARTER MAPPING CONSTANTS
# =============================================================================

# Quarter to days mapping
QUARTER_DAYS = {
    0: 0,      # No quarter
    1: 91,     # 1 quarter
    2: 182,    # 2 quarters  
    3: 273,    # 3 quarters
    4: 365,    # 4 quarters (1 year)
    8: 730     # 8 quarters (2 years)
}

# Reverse mapping: days to quarter
DAYS_TO_QUARTER = {v: k for k, v in QUARTER_DAYS.items()}

def segment_group_summary(df_joined, form_type, debug_print=DEFAULT_DEBUG_FLAG):
    """
    Utility function to identify segment groups in SEC filings and return 
    summary records.
    
    A segment group is defined as a group of records with identical 
    (cik, tag, qtrs, ddate, period, form) but with variable "segments" attributes.
    If such a group is identified and contains a record with segments=NaN, 
    that record is declared as the summary with summary_flag=1.
    
    Args:
        df_joined (DataFrame): Pre-loaded and joined SEC filing data
        form_type (str): The form type to analyze (e.g., '10-K', '10-Q', '8-K')
    
    Returns:
        DataFrame: a dataframe, condensed from df_joined, which only contains 
        the segment summary records. 

    Logic: 
    - If there is only one record in the segment group, that record is the summary; 
    - if the segment has multiple records but with one record w/ segments=Nan, then 
        the segments=Nan record is the summary
    - If the segment has multiple record but no single record with segments=Nan, 
    then use the record with max(value) as the summary. 
    - When the above logic completes, return the dataframe with summary records.   

    Note: we use table join for the three strategies above. This is computationally 
    far more efficient than iterating over the segment groups. 
    """
    
    # Check that df_joined has all required columns
    required_columns = ['cik', 'tag', 'ddate', 'qtrs', 'segments', 'uom',
                         'custom_tag', 'value', 'period', 'form']
    missing_columns = [col for col in required_columns if col not in df_joined.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in df_joined: {missing_columns}. "
                       f"Required columns: {required_columns}")
    
    df_filtered = df_joined[
        (df_joined['custom_tag'] == 0) & 
        (df_joined['form'] == form_type) &
        (df_joined['value'].notna()) & 
        (df_joined['uom'] == 'USD')
    ][['cik', 'tag', 'ddate', 'qtrs', 'segments', 'value', 'period', 'form']].copy()
    df_filtered.drop_duplicates(inplace=True)
    
    # Optimize by setting index for faster lookups
    df_filtered.set_index(['cik', 'tag', 'ddate', 'qtrs', 'period', 'form'], inplace=True)
    
    # Group by (cik, name, tag, ddate, qtrs) and aggregate segments column
    segment_group = df_filtered.groupby(['cik', 'tag', 'ddate', 'qtrs', 'period', 'form'], 
                                        dropna=False).agg({
        'segments': [
            'nunique',  # (i) number of distinct segments
            lambda x: x.isna().any(), 
            lambda x: len(x)==1, 
            lambda x: x.isna().any() or len(x)==1 # (ii) whether a summary record exists (segments=NaN)
        ]
    }).reset_index()
    
    # Reset index back to normal for easier processing
    df_filtered.reset_index(inplace=True)
    
    # Flatten column names
    segment_group.columns = [
        'cik', 'tag', 'ddate', 'qtrs', 'period', 'form',
        'distinct_segments_non_null', 'has_segment_null', 
        'only_1_segment', 'has_summary_record'
    ]
    
    # Now implement the optimized logic to identify summary records
    summary_records = []
    
    # Strategy 1: For rows with 'only_1_segment' being True, directly copy from df_filtered
    single_segment_groups = segment_group[segment_group['only_1_segment'] == True]
    if len(single_segment_groups) > 0:
        # Join with df_filtered to get the actual records
        single_segment_records = single_segment_groups.merge(
            df_filtered, 
            on=['cik', 'tag', 'ddate', 'qtrs', 'period', 'form'], 
            how='left'
        )
        # Add summary metadata
        single_segment_records['summary_flag'] = 1
        single_segment_records['summary_reason'] = 'single_record'
        summary_records.append(single_segment_records)
        
        # Assertion: single_segment_groups and single_segment_records should have the same length
        assert len(single_segment_groups) == len(single_segment_records), \
            f"Length mismatch in Strategy 1: " \
            f"single_segment_groups ({len(single_segment_groups)}) != " \
            f"single_segment_records ({len(single_segment_records)})"
    
    # Strategy 2: For rows with 'only_1_segment' being False but has_segment_null, 
    # join df_filtered (filtered for segments==NULL records) and segment_group
    multi_segment_with_null = segment_group[
        (segment_group['only_1_segment'] == False) & 
        (segment_group['has_segment_null'] == True)
    ]
    if len(multi_segment_with_null) > 0:
        # Filter df_filtered for records with segments==NULL
        null_segment_records = df_filtered[df_filtered['segments'].isna()].copy()
        # Join with segment_group to get the summary records
        null_summary_records = multi_segment_with_null.merge(
            null_segment_records,
            on=['cik', 'tag', 'ddate', 'qtrs', 'period', 'form'],
            how='left'
        )
        # Add summary metadata
        null_summary_records['summary_flag'] = 1
        null_summary_records['summary_reason'] = 'segments_null'
        summary_records.append(null_summary_records)
        
        # Assertion: multi_segment_with_null and null_summary_records should have the same length
        assert len(multi_segment_with_null) == len(null_summary_records), \
            f"Length mismatch in Strategy 2: " \
            f"multi_segment_with_null ({len(multi_segment_with_null)}) != " \
            f"null_summary_records ({len(null_summary_records)})"
    
    # Strategy 3: Only process records that need line-by-line copy 
    # These are groups with multiple segments and no null segments
    remaining_groups = segment_group[
        (segment_group['only_1_segment'] == False) & 
        (segment_group['has_segment_null'] == False)
    ]
    
    if len(remaining_groups) > 0:
        # Use merge to get all relevant records at once
        all_remaining_records = df_filtered.merge(
            remaining_groups[['cik', 'tag', 'ddate', 'qtrs', 'period', 'form']], 
            on=['cik', 'tag', 'ddate', 'qtrs', 'period', 'form'], 
            how='inner'
        )
        
        # Group by the keys and find max value for each group
        max_value_records = all_remaining_records.loc[
            all_remaining_records.groupby(['cik', 'tag', 'ddate', 'qtrs', 'period', 'form'])['value'].idxmax()
        ].copy()
        
        # Add summary metadata
        max_value_records['summary_flag'] = 1
        max_value_records['summary_reason'] = 'max_value'
        
        summary_records.append(max_value_records)
        
        # Assertion: remaining_groups and max_value_records should have the same length
        assert len(remaining_groups) == len(max_value_records), \
            f"Length mismatch in Strategy 3: " \
            f"remaining_groups ({len(remaining_groups)}) != " \
            f"max_value_records ({len(max_value_records)})"
    
    # Combine all summary records efficiently
    if summary_records:
        # Use concat with ignore_index=True for better performance
        df_summary = pd.concat(summary_records, ignore_index=True, copy=False)
    else:
        df_summary = pd.DataFrame()
    
    # Print summary statistics 
    if debug_print: 
        print(f"SEGMENT GROUP SUMMARY - {form_type} FORMS")
        print(f"Total segment groups processed: {len(segment_group):,}")
        print(f"Total summary records created: {len(df_summary):,}")
        print(f"Summary records by reason:")
        print(f"  - Single record: {len(df_summary[df_summary['summary_reason'] == 'single_record']):,}")
        print(f"  - Segments null: {len(df_summary[df_summary['summary_reason'] == 'segments_null']):,}")
        print(f"  - Max value: {len(df_summary[df_summary['summary_reason'] == 'max_value']):,}")

    return df_summary



def validate_and_fix_dates(df, date_column, debug_print=False):
    """
    Validate and fix potential date typos in YYYYMMDD format.
    
    Common issues detected and fixed:
    - Year typos: 29230630 -> 20230630 (29 -> 20), 28230630 -> 20230630 (28 -> 20), etc.
    - Month typos: 20231301 -> 20230301 (13 -> 03)
    - Day typos: 20230632 -> 20230630 (32 -> 30)
    - Future dates beyond reasonable range
    
    Args:
        df (DataFrame): Input dataframe
        date_column (str): Name of the date column to validate
        debug_print (bool): Whether to print debug information (ignored for corrections)
        
    Returns:
        DataFrame: DataFrame with corrected dates
    """
    df_work = df.copy()
    
    if date_column not in df_work.columns:
        print(f"⚠️  Column '{date_column}' not found in dataframe")
        return df_work
    
    # Convert to string for easier manipulation
    df_work[date_column] = df_work[date_column].astype(str)
    
    # Track corrections made
    corrections_made = []
    
    def fix_date_typo(date_str):
        """Fix common date typos in YYYYMMDD format"""
        if len(date_str) != 8 or not date_str.isdigit():
            return date_str  # Skip non-standard formats
        
        year = date_str[:4]
        month = date_str[4:6]
        day = date_str[6:8]
        
        # Fix year typos - more general approach for 2x, 3x, 4x, etc. -> 20xx
        if int(year) > 2500:  # Any future year beyond 2025
            # Extract last 2 digits and prepend '20'
            year = '20' + year[2:]
            corrected_date = year + month + day
            corrections_made.append(f"{date_str} -> {corrected_date}")
            return corrected_date
        
        return date_str
    
    # Apply date fixing
    df_work[date_column] = df_work[date_column].apply(fix_date_typo)
    
    # Always report corrections if any were made (bypass debug_print flag)
    if corrections_made:
        print(f"🔧 Date corrections detected in column '{date_column}':")
        unique_corrections = list(set(corrections_made))
        for correction in unique_corrections:
            print(f"   {correction}")
        print(f"   Total corrections: {len(corrections_made)}")
        
        # Ask user for confirmation
        response = input("\n❓ Do you want to apply these date corrections? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            print("✅ Applying date corrections...")
            return df_work
        else:
            print("❌ Date corrections cancelled. Returning original data.")
            return df
    
    return df_work


def history_comparisons(df, debug_print=DEFAULT_DEBUG_FLAG):
    """
    Performs historical comparisons for SEC filing data.
    
    Logic:
    Groups data by ['cik', 'period', 'form', 'tag', 'qtrs'] and for each group:
    1. Identifies the most recent date (crt_ddate) and its value (crt_value)
    2. Collects all historical dates and their corresponding values
    3. Calculates time intervals from historical dates to current date, 
      rounded to nearest quarters
    4. Computes percentage changes as (crt_value - historical_value) / crt_value
    
    Args:
        df (DataFrame): Input dataframe with SEC filing data
    
    Returns:
        DataFrame: Processed dataframe with historical comparisons with columns
        ['cik', 'tag', 'qtrs', 'crt_ddate', 'crt_value', 
         'quarter_intervals', 'percentage_diffs', 'num_historical_points']
        where quarter_intervals contains quarter numbers (0-8) instead of days
    
    Raises:
        ValueError: If required columns are missing or form validation fails
    """
    
    # Required columns assertion
    required_columns = ['cik', 'tag', 'ddate', 'qtrs', 'period', 'form', 'value']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in df: {missing_columns}. "
                        f"Required columns: {required_columns}")
    
    # Convert ddate to datetime for proper date calculations
    df_work = df.copy()
    
    # Validate and fix potential date typos before conversion
    df_work = validate_and_fix_dates(df_work, 'ddate', debug_print)
    
    df_work['ddate'] = pd.to_datetime(df_work['ddate'], format='%Y%m%d')
    
    # Group by ['cik', 'name', 'tag', 'segments', 'qtrs'] and aggregate
    def aggregate_historical_data(group):
        """
        Custom aggregation function for historical data processing.
        
        Returns:
            dict: Contains current date/value and historical comparisons
        """
        # Sort by ddate to get chronological order
        group_sorted = group.sort_values('ddate')
        
        # (a) Get the most recent ddate and corresponding value
        crt_ddate = group_sorted['ddate'].iloc[-1]
        crt_value = group_sorted['value'].iloc[-1]
        
        # (b) Get list of other ddates and their corresponding values
        if len(group_sorted) > 1:
            other_ddates = group_sorted['ddate'].iloc[:-1].tolist()
            other_values = group_sorted['value'].iloc[:-1].tolist()
        else:
            other_ddates = []
            other_values = []
        
        # (c) Round the interval from all ddates to crt_ddate to nearest quarter interval
        quarter_intervals = []
        for other_ddate in other_ddates:
            interval_days = (crt_ddate - other_ddate).days
            rounded_interval_days = round_to_nearest_quarter_days(interval_days, QUARTER_DAYS)
            # Convert days to quarter number using the reverse mapping for O(1) lookup
            quarter_number = DAYS_TO_QUARTER[rounded_interval_days]
            quarter_intervals.append(quarter_number)
        
        # (d) Compute percentage difference as (crt_value - value)/crt_value
        percentage_diffs = []
        for other_value in other_values:
            if crt_value != 0:
                pct_diff = (crt_value - other_value) / crt_value
            else:
                pct_diff = np.nan  # Handle division by zero
            percentage_diffs.append(pct_diff)
        
        return pd.Series({
            'crt_ddate': crt_ddate,
            'crt_value': crt_value,
            'other_ddates': other_ddates,
            'other_values': other_values,
            'quarter_intervals': quarter_intervals,
            'percentage_diffs': percentage_diffs,
            'num_historical_points': len(other_ddates)
        })
    
    
    grouped_history = df_work.groupby(['cik', 'period', 'form', 'tag', 'qtrs']).apply(
        aggregate_historical_data, include_groups=False
    ).reset_index()
    # Set index to [cik, tag, period, qtrs] for efficient lookups
    grouped_history.set_index(['cik', 'period', 'form', 'tag', 'qtrs'], inplace=True)

    if debug_print:
        print("HISTORY_COMPARISONS: ")
        print(f"Input dataframe has {len(df):,} records...") 
        print(f"Created {len(grouped_history):,} feature groups.")
    
    return grouped_history


def organize_feature_dataframe(grouped_history, 
                               df_tags_to_featurize, 
                               N_qtrs_history_comp, 
                               debug_print=DEFAULT_DEBUG_FLAG):
    """
    Reorganizes the grouped_history dataframe into a featurized format with columns for
    current values and historical percentage changes for each tag.
    
    Args:
    - grouped_history (DataFrame): Output from history_comparisons() function, 
     has columns ['cik', 'period', 'form', 'tag', 'qtrs', 'crt_ddate', 'crt_value', 
     'quarter_intervals', 'percentage_diffs']. 
    - df_tags_to_featurize (DataFrame): DataFrame with columns ['rank', 'tag'] 
        specifying which tags to featurize
    - N_qtrs_history_comp (int, default=4): # of quarters to featurize history comparisons
    
   The output df_featurized dataframe should have one row per distinct 
   ['cik', 'period', 'form'] tuple. 
   The df_featurized's columns are [cik, period, form] + [TagName_Qtrs_current, 
   TagName_Qtrs_change_q1, TagName_Qtrs_change_q2, ...] 
   where 
   - the "TagName" part should be substituted with tag name, 
   -  Qtrs should be substituted by the qtrs attribute,  
   - the "_current" part fetches the 'crt_value' field
   - the  "change_q1" to  "change_q..." part should be the changes
     (percentage) from the historic values in preceeding quarters. 
   In the case where we don't have data for the suitable quarter, fill in with Nan. 
 """
    
    # Validate df_tags_to_featurize input
    required_columns = ['rank', 'tag']
    missing_columns = [col for col in required_columns if col not in df_tags_to_featurize.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in df_tags_to_featurize: {missing_columns}. "
                        f"Required columns: {required_columns}")
    
    # Extract tags to featurize from the input dataframe
    top_tags = df_tags_to_featurize['tag'].tolist()
    
    # Reset index to make grouped_history a regular dataframe for easier processing
    df_history_work = grouped_history.reset_index()
    # Filter to only include tags we want to featurize
    df_history_filtered = df_history_work[(df_history_work['tag'].isin(top_tags)) & 
                                          (df_history_work['qtrs'] <=4)
                                          ].copy()
  
    # Create base dataframe with all unique (cik, period) combinations
    df_featurized = df_history_work[['cik', 'period', 'form']].drop_duplicates().reset_index(drop=True)

    # Create (tag, qtrs) combination column for pivot
    df_history_filtered['tag_qtrs'] = df_history_filtered['tag'] + \
                                      '_' + df_history_filtered['qtrs'].astype(str) + 'qtrs'
    
    current_values = df_history_filtered.pivot_table(
        index=['cik', 'period', 'form'], 
        columns='tag_qtrs', 
        values='crt_value', 
        aggfunc='first'  # Use first in case of duplicates
    )
    
    # Rename columns to add '_current' suffix
    current_values.columns = [f'{col}_current' for col in current_values.columns]
    
    # Join current values to base dataframe
    df_featurized = df_featurized.merge(
        current_values.reset_index(), 
        on=['cik', 'period', 'form'], 
        how='left'
    )
    
    # Explode the quarter_intervals and percentage_diffs lists to create long format
    df_exploded = df_history_filtered.copy()
    df_exploded = df_exploded.explode(['quarter_intervals', 'percentage_diffs'])
    
    # Remove rows where quarter_intervals is NaN (no historical data)
    df_exploded = df_exploded.dropna(subset=['quarter_intervals', 'percentage_diffs'])
    
    # Convert quarter_intervals to int and filter for desired quarters
    df_exploded['quarter_intervals'] = df_exploded['quarter_intervals'].astype(int)
    df_exploded = df_exploded[df_exploded['quarter_intervals'].between(1, N_qtrs_history_comp)]
    
    # Create pivot table for historical changes - collect all quarters first
    quarter_dataframes = []
    
    for q in range(1, N_qtrs_history_comp + 1): 
        
        # Filter for this specific quarter
        quarter_data = df_exploded[df_exploded['quarter_intervals'] == q]
        
        if len(quarter_data) == 0:
            continue
            
        # Pivot to create columns for each (tag, qtrs) combination
        quarter_pivot = quarter_data.pivot_table(
            index=['cik', 'period', 'form'],
            columns='tag_qtrs',
            values='percentage_diffs',
            aggfunc='first'  # Use first in case of duplicates
        )
        
        # Rename columns to add quarter suffix
        quarter_pivot.columns = [f'{col}_change_q{q}' for col in quarter_pivot.columns]
        
        # Add to list for later concatenation
        quarter_dataframes.append(quarter_pivot.reset_index())
    
    # Join all quarter dataframes at once to avoid fragmentation
    if quarter_dataframes:
        # Set index for efficient concatenation
        for i, quarter_df in enumerate(quarter_dataframes):
            quarter_dataframes[i] = quarter_df.set_index(['cik', 'period', 'form'])
        
        # Concatenate all quarter dataframes along columns axis
        all_quarters_df = pd.concat(quarter_dataframes, axis=1, sort=False)
        
        # Join to main dataframe
        df_featurized = df_featurized.merge(
            all_quarters_df.reset_index(),
            on=['cik', 'period', 'form'],
            how='left'
        )
    
    # Ensure all expected columns exist (fill missing with NaN)
    # Create expected columns based on actual (tag, qtrs) combinations in the data
    unique_tag_qtrs = df_history_filtered['tag_qtrs'].unique()
    
    expected_columns = ['cik', 'period', 'form']
    for tag_qtrs in unique_tag_qtrs:
        expected_columns.append(f'{tag_qtrs}_current')
        for q in range(1, N_qtrs_history_comp + 1):
            expected_columns.append(f'{tag_qtrs}_change_q{q}')
    
    # Add missing columns with NaN values efficiently using concat
    missing_columns = set(expected_columns) - set(df_featurized.columns)
    if missing_columns:
        # Create a DataFrame with missing columns filled with NaN
        missing_df = pd.DataFrame(
            index=df_featurized.index,
            columns=list(missing_columns),
            dtype=float  # Use float to allow NaN values
        )
        # Concatenate with existing dataframe
        df_featurized = pd.concat([df_featurized, missing_df], axis=1, sort=False)
    
    # Reorder columns to match expected format (keep cik, period first, then sort other columns)
    base_columns = ['cik', 'period', 'form']
    feature_columns = sorted([col for col in expected_columns if col not in base_columns])
    df_featurized = df_featurized[base_columns + feature_columns]
    
    if debug_print:
        print(f"ORGANIZE_FEATURE_DATAFRAME: ")
        print(f"Final dataframe shape: {df_featurized.shape}")
        print(f"Feature columns created: {len(unique_tag_qtrs)} (tag,qtrs) combinations × ({N_qtrs_history_comp} quarters + 1 current) = {len(unique_tag_qtrs) * (N_qtrs_history_comp + 1)} features")
    
    return df_featurized



def round_to_nearest_quarter_days(interval_days, quarter_days):
    """
    Utility function to round a date interval to the nearest quarter boundary 
    and return the day count.
    
    Args:
        interval_days (int): Number of days in the interval
        quarter_days (dict): Dictionary mapping quarter numbers to days
    
    Returns:
        int: Rounded interval in days (e.g., 91, 182, 365, 730)
        Always returns a valid quarter boundary value that exists in the quarter_days dictionary.
    
    Quarter boundaries:
    - 1 quarter: 91 days
    - 2 quarters: 182 days  
    - 3 quarters: 273 days
    - 4 quarters (1 year): 365 days
    - 8 quarters (2 years): 730 days
    """
    # Find the nearest quarter boundary
    if interval_days <= 0:
        return interval_days  # Return original value for invalid intervals
    
    # Find the closest quarter boundary from the dictionary values
    quarter_values = list(quarter_days.values())
    closest_boundary = min(quarter_values, key=lambda x: abs(x - interval_days))
    
    # Always return the closest quarter boundary, regardless of distance
    # This ensures we always get a valid quarter boundary that exists in DAYS_TO_QUARTER
    return closest_boundary


def summarize_feature_completeness(df_features):
    """
    Analyzes feature completeness in the featurized dataframe and ranks all features
    by their non-null percentage. Outputs a CSV file with feature completeness rankings.
    
    Args:
        df_features (DataFrame): Featurized dataframe from featurize_quarter_data() or featurize_multi_qtrs()
    
    Returns:
        dict: Dictionary with 'features_ranked' key containing list of (feature, count, percentage) tuples
              sorted by completeness in descending order, plus summary statistics
    """
    
    # Identify feature columns (exclude key columns)
    key_columns = ['cik', 'period', 'form', 'data_qtr']
    feature_columns = [col for col in df_features.columns if col not in key_columns]
    
    # Calculate non-null percentage for each feature column using vectorized operations
    total_rows = len(df_features)
    
    if total_rows > 0 and feature_columns:
        # Get non-null counts for all feature columns at once
        non_null_counts = df_features[feature_columns].notna().sum()
        # Calculate percentages for all columns at once
        non_null_percentages = (non_null_counts / total_rows) * 100
        
        # Create ranked list with both count and percentage
        all_features_ranked = list(zip(non_null_percentages.index, non_null_counts.values, non_null_percentages.values))
        # Sort by percentage (descending)
        all_features_ranked.sort(key=lambda x: x[2], reverse=True)
    else:
        all_features_ranked = []
    
    # Create DataFrame for CSV output
    feature_completeness_df = pd.DataFrame(all_features_ranked, columns=['feature', 'non_null_count', 'non_null_percentage'])
    feature_completeness_df['rank'] = range(1, len(feature_completeness_df) + 1)
    
    # Reorder columns: rank, feature, non_null_count, non_null_percentage
    feature_completeness_df = feature_completeness_df[['rank', 'feature', 'non_null_count', 'non_null_percentage']]
    return feature_completeness_df 


def featurize_quarter_data(data_directory, 
                           df_tags_to_featurize, 
                           N_qtrs_history_comp, 
                           debug_print=DEFAULT_DEBUG_FLAG):
    """
    Top-level function to featurize quarterly SEC data for both 10-Q and 10-K forms.
    Args:
        data_directory (str): Path to data directory (e.g., 'data/2022q1')
        df_tags_to_featurize (DataFrame): DataFrame with columns ['rank', 'tag'] specifying which tags to featurize
        N_qtrs_history_comp (int, default=4): Number of quarters to featurize
    Returns:
        DataFrame: Combined featurized dataframe with both 10-Q and 10-K features.
        Single row per (cik, period) with best available feature values
        Columns: ['cik', 'period'] + feature columns like 
        'tag_Nqtrs_current', 'tag_Nqtrs_change_qX'
    
    Consolidation Logic:
    - Form types are processed in order: ['10-Q', '10-K']
    - First form type (10-Q) establishes the base feature dataframe
    - Subsequent form types (10-K) only fill missing/null values in existing features 
    """
    
    # Load data using load_and_join_sec_xbrl_data() 
    df_joined = load_and_join_sec_xbrl_data([data_directory]) 
    
    # Process both 10-Q and 10-K forms
    form_types = ['10-Q', '10-K'] 
    df_featurize_qtr = pd.DataFrame()

    for i, form_type in enumerate(form_types): 
         
        df_summary = segment_group_summary(df_joined, form_type=form_type) 
        if len(df_summary) > 0: 
            grouped_history = history_comparisons(df_summary)
            df_features = organize_feature_dataframe(grouped_history, df_tags_to_featurize, N_qtrs_history_comp=N_qtrs_history_comp)
             
            if i == 0:
                df_featurize_qtr = df_features 
            else:
                df_featurize_qtr = \
                    df_featurize_qtr.set_index(['cik', 'period']).combine_first(
                        df_features.set_index(['cik', 'period'])
                    ).reset_index()

            if debug_print:
                print(f"FEATURIZE_QUARTER_DATA: for {form_type} ...")
                print(f"Getting features with shape: {df_features.shape}")
                print(f"Consolidated with shape: {df_featurize_qtr.shape}")  
    
    return df_featurize_qtr


def featurize_multi_qtrs(data_directories, 
                         df_tags_to_featurize, 
                         N_qtrs_history_comp, 
                         save_dir=SAVE_DIR, 
                         debug_print=DEFAULT_DEBUG_FLAG):
    """
    Top-level function to featurize multiple quarters of SEC data.
    """
    df_featurized_all_quarters = pd.DataFrame()
    for (i_dir, data_directory) in enumerate(data_directories):
        quarter_name = data_directory.split('/')[-1]
        save_file = os.path.join(save_dir, QUARTER_FEATURIZED_PATTERN.format(quarter_name))
        if os.path.exists(save_file):
            df_featurized_quarter = pd.read_csv(save_file)
        else:
            df_featurized_quarter = featurize_quarter_data(data_directory, 
                                                        df_tags_to_featurize, 
                                                        N_qtrs_history_comp)
            df_featurized_quarter.drop('form', axis=1, inplace=True)
            df_featurized_quarter.to_csv(save_file, index=False)

        print(f"Featurized {data_directory} with shape: {df_featurized_quarter.shape}")
        df_featurized_quarter['data_qtr'] = quarter_name
        
        if i_dir == 0:
            df_featurized_all_quarters = df_featurized_quarter.copy()
        else:
            df_featurized_all_quarters = \
                df_featurized_all_quarters.set_index(['cik', 'period', 'data_qtr']).combine_first(
                    df_featurized_quarter.set_index(['cik', 'period', 'data_qtr'])
                ).reset_index()

    completeness_stats = summarize_feature_completeness(df_featurized_all_quarters)
    completeness_stats.to_csv(os.path.join(save_dir, FEATURE_COMPLETENESS_RANKING_FILE), index=False)

    df_featurized_all_quarters.to_csv(os.path.join(save_dir, FEATURIZED_ALL_QUARTERS_FILE), index=False)


def simplify_featurized_data(processed_data_dir=SAVE_DIR, min_completeness=DEFAULT_MIN_COMPLETENESS):
    """
    Simplify the featurized data by keeping only well-populated features and removing duplicates.
    
    This function:
    1. Reads the featurized_all_quarters.csv file
    2. Filters out CIKs that don't map to ticker symbols (for stock price data availability)
    3. Reads the feature_completeness_ranking.csv file
    4. Keeps only columns that are more than min_completeness% populated
    5. For duplicate (cik, period) combinations, retains only the most recent record (max data_qtr)
    
    Args:
        processed_data_dir (str): Directory containing the processed data files
        min_completeness (float): Minimum completeness percentage to keep features (default: 15.0)
        
    Returns:
        pd.DataFrame: Simplified featurized data with well-populated features and no duplicates
    """
    # Step 1: Read the featurized_all_quarters.csv file
    featurized_file = os.path.join(processed_data_dir, FEATURIZED_ALL_QUARTERS_FILE)
    if not os.path.exists(featurized_file):
        raise FileNotFoundError(f"Featurized data file not found: {featurized_file}")
    
    df_featurized = pd.read_csv(featurized_file, low_memory=False)
    print(f"📊 Input data shape: {df_featurized.shape}")
    
    # Step 2: Filter out CIKs that don't map to tickers
    print("🔍 Filtering CIKs that don't map to ticker symbols...")
    cik_to_ticker, ticker_to_cik = get_cik_ticker_mapping()
    
    if cik_to_ticker:
        # Convert CIK to string with zero padding for comparison
        df_featurized['cik_str'] = df_featurized['cik'].astype(str).str.zfill(10)
        
        # Filter to keep only CIKs that have ticker mappings
        df_featurized = df_featurized[df_featurized['cik_str'].isin(cik_to_ticker.keys())]
        
        # Drop the temporary cik_str column
        df_featurized = df_featurized.drop('cik_str', axis=1)
        
        print(f"📊 After CIK filtering: {df_featurized.shape}")
        print(f"✅ Kept only CIKs with ticker symbols for stock price data availability")
    else:
        print("⚠️  Warning: Could not load CIK->ticker mapping. Keeping all CIKs.")
    
    # Step 3: Read the feature_completeness_ranking.csv file
    ranking_file = os.path.join(processed_data_dir, FEATURE_COMPLETENESS_RANKING_FILE)
    if not os.path.exists(ranking_file):
        raise FileNotFoundError(f"Feature completeness ranking file not found: {ranking_file}")
    
    df_ranking = pd.read_csv(ranking_file)
    
    # Step 4: Keep only columns that are more than min_completeness% populated
    well_populated_features = df_ranking[df_ranking['non_null_percentage'] > min_completeness]['feature'].tolist()
    
    # Identify columns to keep (cik, period, data_qtr + well-populated features)
    base_columns = ['cik', 'period', 'data_qtr']
    available_base_columns = [col for col in base_columns if col in df_featurized.columns]
    available_features = [col for col in well_populated_features if col in df_featurized.columns]
    columns_to_keep = available_base_columns + available_features
    
    # Filter the dataframe to keep only selected columns
    df_simplified = df_featurized[columns_to_keep].copy()
    
    # Step 5: Handle duplicate (cik, period) combinations
    # Check for duplicates before processing
    duplicate_check = df_simplified.groupby(['cik', 'period']).size()
    duplicates = duplicate_check[duplicate_check > 1]
    
    if len(duplicates) > 0:
        # Sort by data_qtr to ensure we get the most recent record
        # data_qtr strings like "2022q1", "2022q2" are already lexicographically sortable
        df_simplified = df_simplified.sort_values(['cik', 'period', 'data_qtr'], ascending=[True, True, False])
        df_simplified = df_simplified.drop_duplicates(subset=['cik', 'period'], keep='first')
    
    print(f"📊 Output data shape: {df_simplified.shape}")
    
    # Step 6: Write the simplified dataframe to CSV file
    simplified_file = os.path.join(processed_data_dir, FEATURIZED_SIMPLIFIED_FILE)
    df_simplified.to_csv(simplified_file, index=False)
    print(f"💾 Simplified data saved to: {simplified_file}")
    
    return df_simplified


def enhance_tags_w_gradient(df_work, quarters_for_gradient_comp):
    """
    Enhance featurized data with gradient features by computing relative changes
    from previous quarters.
    
    Args:
        df_work (pd.DataFrame): Input dataframe with features ending in '_current'
        quarters_for_gradient_comp (list): List of quarters to compute gradients from (e.g., [1, 2] for 1 and 2 quarters ago)
    
    This function:
    1. Takes the input dataframe df_work
    2. For each feature ending with "_current", looks up values from specified quarters ago
    3. Computes relative change percentages and creates new features with "_change_q{N}" suffixes
    4. Returns enhanced dataframe with gradient features
    
    Returns:
        pd.DataFrame: Enhanced dataframe with gradient features (df_work_w_gradient)
    """
    print("🔄 Enhancing features with gradient information...")
    print("=" * 60)
    
    print(f"📊 Input dataframe shape: {df_work.shape}")
    
    # Identify features ending with "_current"
    current_features = [col for col in df_work.columns if col.endswith('_current')]
    print(f"🔍 Found {len(current_features)} features ending with '_current'")
    
    if not current_features:
        print("⚠️  No features ending with '_current' found. Returning original dataframe.")
        return df_work
    
    # Convert period to datetime for easier manipulation
    df_work['period'] = pd.to_datetime(df_work['period'], format='%Y%m%d')
    
    # Create historical data for each specified quarter
    historical_data = {}
    for q in quarters_for_gradient_comp:
        df_historical = df_work.copy()
        # Subtract months to go back in time (q quarters = q * 3 months)
        df_historical['period'] = df_historical['period'] - pd.DateOffset(months=q * 3)
        # Convert back to original format for joining
        df_historical['period'] = df_historical['period'].dt.strftime('%Y%m%d').astype(int)
        historical_data[q] = df_historical
    
    # Convert back to original format for joining
    df_work['period'] = df_work['period'].dt.strftime('%Y%m%d').astype(int)
    
    # Create gradient features
    df_work_w_gradient = df_work.copy()
    
    print(f"📈 Computing gradient features for quarters: {quarters_for_gradient_comp}")
    
    # Prepare historical data with all current features for each quarter
    historical_subsets = {}
    total_duplicates = 0
    
    for q in quarters_for_gradient_comp:
        # Note: Main deduplication is handled upstream, but we need to dedupe historical data
        # since shifting periods might create new duplicates
        df_historical_subset = historical_data[q][['cik', 'period'] + current_features].drop_duplicates(subset=['cik', 'period'])
        
        # Rename columns to indicate they're from previous quarters
        q_columns = {feature: f'{feature}_q{q}' for feature in current_features}
        df_historical_subset = df_historical_subset.rename(columns=q_columns)
        
        # Debug: Check for duplicates in historical data
        original_count = len(historical_data[q][['cik', 'period'] + current_features])
        deduped_count = len(df_historical_subset)
        duplicates = original_count - deduped_count
        total_duplicates += duplicates
        
        historical_subsets[q] = df_historical_subset
    
    if total_duplicates > 0:
        print(f"🔍 Removed {total_duplicates} total duplicates from historical data")
    
    # Merge all historical data
    for q in quarters_for_gradient_comp:
        df_work_w_gradient = df_work_w_gradient.merge(historical_subsets[q], on=['cik', 'period'], how='left')
    
    # Verify row count hasn't changed
    if len(df_work_w_gradient) != len(df_work):
        print(f"⚠️  Warning: Row count changed from {len(df_work)} to {len(df_work_w_gradient)}")
    
    # Compute all gradient features at once using vectorized operations
    print(f"🔄 Computing gradients for {len(current_features)} features across {len(quarters_for_gradient_comp)} quarters...")
    
    # Create all gradient feature names and compute gradients for each quarter
    all_gradient_features = []
    all_gradient_dfs = []
    temp_columns_to_remove = []
    
    for q in quarters_for_gradient_comp:
        # Create gradient feature names for this quarter
        q_features = [f"{feature.replace('_current', '')}_change_q{q}" for feature in current_features]
        all_gradient_features.extend(q_features)
        
        # Get historical columns for this quarter
        q_current_cols = [f'{feature}_q{q}' for feature in current_features]
        temp_columns_to_remove.extend(q_current_cols)
        
        # Vectorized computation for gradients
        with np.errstate(divide='ignore', invalid='ignore'):
            q_gradients = (df_work_w_gradient[current_features].values - df_work_w_gradient[q_current_cols].values) / df_work_w_gradient[q_current_cols].values
        q_gradients = np.where(df_work_w_gradient[q_current_cols].values != 0, q_gradients, np.nan)
        
        # Create gradient DataFrame for this quarter
        q_gradient_df = pd.DataFrame(q_gradients, columns=q_features, index=df_work_w_gradient.index)
        all_gradient_dfs.append(q_gradient_df)
    
    # Concatenate all gradient features at once to avoid fragmentation
    if all_gradient_dfs:
        df_work_w_gradient = pd.concat([df_work_w_gradient] + all_gradient_dfs, axis=1)
    
    # Clean up temporary columns
    df_work_w_gradient = df_work_w_gradient.drop(columns=temp_columns_to_remove, errors='ignore')
    
    print(f"✅ Enhanced dataframe shape: {df_work_w_gradient.shape}")
    
    # Verify row count is preserved
    if len(df_work_w_gradient) == len(df_work):
        print(f"✅ Row count preserved: {len(df_work_w_gradient)} rows (same as original)")
    else:
        print(f"❌ Row count mismatch: {len(df_work_w_gradient)} vs {len(df_work)} (original)")
        raise ValueError(f"Row count changed from {len(df_work)} to {len(df_work_w_gradient)}")
    
    # Count new gradient features
    gradient_features = [col for col in df_work_w_gradient.columns if any(col.endswith(f'_change_q{q}') for q in quarters_for_gradient_comp)]
    print(f"📊 Added {len(gradient_features)} gradient features")
    
    # Show sample of new features
    if gradient_features:
        print("🔍 Sample gradient features:")
        for feature in gradient_features[:5]:  # Show first 5
            print(f"  - {feature}")
        if len(gradient_features) > 5:
            print(f"  ... and {len(gradient_features) - 5} more")
    
    return df_work_w_gradient


if __name__ == "__main__": 

    # Create tags to featurize (this can be reused across multiple quarters)
    df_tags_to_featurize = read_tags_to_featurize(K_top_tags=DEFAULT_K_TOP_TAGS)
    
    # Find all quarter directories in the data directory
    quarter_directories = []
    
    if os.path.exists(DATA_BASE_DIR):
        # Get all subdirectories in data/ that look like quarters (e.g., 2022q1, 2022q2, etc.)
        for item in os.listdir(DATA_BASE_DIR):
            item_path = os.path.join(DATA_BASE_DIR, item)
            if os.path.isdir(item_path) and ('q' in item.lower() or 'quarter' in item.lower()):
                quarter_directories.append(item_path)
    
    # Sort directories to ensure consistent processing order
    quarter_directories.sort()
    
    print(f"Found {len(quarter_directories)} quarter directories:")
    for qtr_dir in quarter_directories:
        print(f"  - {qtr_dir}")
    
    if quarter_directories:
        # Check if combined featurized file already exists
        combined_file = os.path.join(SAVE_DIR, FEATURIZED_ALL_QUARTERS_FILE)
        
        if os.path.exists(combined_file):
            print(f"✅ Combined featurized file already exists: {combined_file}")
            print("Skipping featurize_multi_qtrs() to avoid reprocessing...")
        else:
            # Process all quarters using featurize_multi_qtrs
            print(f"\nProcessing all quarters with featurize_multi_qtrs...")
            featurize_multi_qtrs(quarter_directories, df_tags_to_featurize, 
                                 N_qtrs_history_comp= DEFAULT_N_QUARTERS_HISTORY_COMP)
        
        # Simplify the featurized data and save to CSV
        print(f"\nSimplifying featurized data...")
        df_simplified = simplify_featurized_data(processed_data_dir=SAVE_DIR, min_completeness=DEFAULT_MIN_COMPLETENESS)
        print(f"✅ Featurization pipeline completed successfully!")