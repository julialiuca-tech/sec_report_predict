import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utility_data import load_and_join_sec_xbrl_data, top_tags


def tags_w_revenue_str(df_joined):
    """
    Comprehensive analysis of revenue tag usage and transitions:
    1. Identifies companies using both "Revenue" and 
       "RevenueFromContractWithCustomerExcluding..." tags
    2. Shows detailed usage patterns and quarterly transitions 

    Conclusion: there are companies who use both tags. 
    For instance, KwikClick uses 
        RevenueFromContractWithCustomerExcludingAssessedTaxes for sales revenue 
        Revenues for total revenue (= sales revenue + license revenue)
    and 
    """
    print("\n" + "="*80)
    print("REVENUE TAG USAGE & TRANSITION ANALYSIS")
    print("="*80)
    
    # Check for companies using different revenue tag types
    companies_with_revenues = df_joined[
        df_joined['tag'] == 'Revenues'
    ]['cik'].unique()
    companies_with_revenue_contract = df_joined[
        df_joined['tag'].str.contains(
            'RevenueFromContractWithCustomerExcluding', 
            case=False, 
            na=False
        )
    ]['cik'].unique()
    companies_with_both = set(companies_with_revenues) & set(companies_with_revenue_contract)
    
    print(f"\nCompanies using 'Revenues' tag: {len(companies_with_revenues):,}")
    print(f"Companies using 'RevenueFromContractWithCustomerExcluding...' tags: "
          f"{len(companies_with_revenue_contract):,}")
    print(f"Companies using BOTH tag types: {len(companies_with_both):,}")
    
    if len(companies_with_both) > 0:
        print(f"\nCompanies using both tag types (showing first 20):")
        for cik in sorted(list(companies_with_both))[:20]:
            company_name = df_joined[
                df_joined['cik'] == cik
            ]['name'].iloc[0]
            print(f"  - CIK: {cik}, Name: {company_name}")
        
        if len(companies_with_both) > 20:
            print(f"  ... and {len(companies_with_both) - 20} more companies")
        
        # Detailed analysis and quarterly transitions for companies using both tags
        print(f"\nDetailed analysis for companies using both tag types:")
        sample_companies = list(companies_with_both)[:5]  # Analyze first 5
        
        for cik in sample_companies:
            company_name = df_joined[
                df_joined['cik'] == cik
            ]['name'].iloc[0]
            company_data = df_joined[df_joined['cik'] == cik]
            
            revenues_usage = company_data[
                company_data['tag'] == 'Revenues'
            ]
            revenue_contract_usage = company_data[
                company_data['tag'].str.contains(
                    'RevenueFromContractWithCustomerExcluding', 
                    case=False, 
                    na=False
                )
            ]
            
            print(f"\n  Company: {company_name} (CIK: {cik})")
            print(f"    'Revenues' tag used {len(revenues_usage):,} times")
            print(f"    'RevenueFromContractWithCustomerExcluding...' tags used "
                  f"{len(revenue_contract_usage):,} times")
            
            quarterly_usage = revenues_usage.groupby(
                ['fy', 'fp', 'tag', 'version']
            ).size().reset_index(name='count')
            print(f"    Quarterly usage patterns:")
            for _, row in quarterly_usage.iterrows():
                print(f"      {row['fy']} {row['fp']}:  ({row['version']} ) "
                      f"{row['tag']} - {row['count']} occurrences")
    
            quarterly_usage = revenue_contract_usage.groupby(
                ['fy', 'fp', 'tag', 'version']
            ).size().reset_index(name='count')
            print(f"    Quarterly usage patterns:")
            for _, row in quarterly_usage.iterrows():
                print(f"      {row['fy']} {row['fp']}:  ({row['version']} ) "
                      f"{row['tag']} - {row['count']} occurrences")
    
    return companies_with_both, None


def tags_10k_10q_overlap(df_joined):
    """
    Utility function to analyze 10-K vs 10-Q filing patterns:
    1. Identifies companies with both 10-K and 10-Q forms
    2. Finds tags used in both forms with identical attributes (segments, ddate)
    3. Compares values to see if 10-Q data is repeated in 10-K or skipped
    
    Args:
        df_joined (DataFrame): Pre-loaded and joined SEC filing data
    
    This helps understand whether companies report quarterly data in annual reports
    or if they skip certain quarterly disclosures in 10-K filings.

    CONCLUSION: 
    10-K and 10-Q may contain overlapping identical data. However, 
    10-K may list  the comparison to the previous year, 
    while 10-Q lists the comparison to the previous quarter. 
    the field ddate may be different and thus the values may be different. 

    Cash Flow statement and operating statement are different between 10-K and 10-Q. 
    Balance sheet has overlapping data. 
    """

    print("\n" + "="*80)
    print("10-K vs 10-Q FILING PATTERN ANALYSIS")
    print("="*80)
    
    print(f"Total records loaded: {len(df_joined):,}")
    print(f"Total unique companies (CIK): {df_joined['cik'].nunique():,}")
    print(f"Total unique tags: {df_joined['tag'].nunique():,}")
    
    # Step 1: Identify companies with both 10-K and 10-Q forms
    print("\nStep 1: Identifying companies with both 10-K and 10-Q forms...")
    
    # Get companies by form type
    companies_10k = df_joined[
        df_joined['form'] == '10-K'
    ]['cik'].unique()
    companies_10q = df_joined[
        df_joined['form'] == '10-Q'
    ]['cik'].unique()
    companies_with_both = set(companies_10k) & set(companies_10q)
    
    print(f"Companies with 10-K forms: {len(companies_10k):,}")
    print(f"Companies with 10-Q forms: {len(companies_10q):,}")
    print(f"Companies with BOTH forms: {len(companies_with_both):,}")
    
    if len(companies_with_both) == 0:
        print("No companies found with both 10-K and 10-Q forms in 2022q1.")
        return
    
    # Show sample companies with both forms
    print(f"\nSample companies with both forms (first 10):")
    for cik in sorted(list(companies_with_both))[:10]:
        company_name = df_joined[
            df_joined['cik'] == cik
        ]['name'].iloc[0]
        print(f"  - CIK: {cik}, Name: {company_name}")
    
    # Step 2: Find tags used in both forms with identical attributes
    print(f"\nStep 2: Finding tags used in both forms with identical attributes...")
    
    # Focus on a few companies for detailed analysis
    sample_companies = list(companies_with_both)[:5]  # Analyze first 5 companies
    
    for cik in sample_companies:
        company_name = df_joined[
            df_joined['cik'] == cik
        ]['name'].iloc[0]
        company_data = df_joined[df_joined['cik'] == cik]
        
        print(f"\nAnalyzing {company_name} (CIK: {cik}):")
        
        # Separate data by form type
        data_10k = company_data[company_data['form'] == '10-K']
        data_10q = company_data[company_data['form'] == '10-Q']
        
        print(f"  10-K records: {len(data_10k):,}")
        print(f"  10-Q records: {len(data_10q):,}")
        
        # Find common tags between 10-K and 10-Q
        tags_10k = set(data_10k['tag'].unique())
        tags_10q = set(data_10q['tag'].unique())
        common_tags = tags_10k & tags_10q
        
        print(f"  Common tags between forms: {len(common_tags):,}")
        
        # For each common tag, find records with identical attributes
        for tag in list(common_tags)[:10]:  # Limit to first 10 tags for readability
            tag_data_10k = data_10k[data_10k['tag'] == tag]
            tag_data_10q = data_10q[data_10q['tag'] == tag]
            
            print(f"\n    Tag: {tag}")
            print(f"      Appears in 10-K: {len(tag_data_10k):,} times")
            print(f"      Appears in 10-Q: {len(tag_data_10q):,} times")
            
            # Find records with identical attributes (segments, ddate, version)
            # Create composite key for comparison
            tag_data_10k['composite_key'] = (
                tag_data_10k['segments'].fillna('') + '|' +
                tag_data_10k['ddate'].astype(str) + '|' +
                tag_data_10k['version'].fillna('')
            )
            
            tag_data_10q['composite_key'] = (
                tag_data_10q['segments'].fillna('') + '|' +
                tag_data_10q['ddate'].astype(str) + '|' +
                tag_data_10q['version'].fillna('')
            )
            
            # Find matching keys
            keys_10k = set(tag_data_10k['composite_key'])
            keys_10q = set(tag_data_10q['composite_key'])
            matching_keys = keys_10k & keys_10q
            
            print(f"      Records with identical attributes: {len(matching_keys):,}")
            
            # Compare values for matching records
            if len(matching_keys) > 0:
                print(f"      Value comparison for matching records:")
                for key in list(matching_keys)[:5]:  # Show first 5 matches
                    value_10k = tag_data_10k[tag_data_10k['composite_key'] == key ]['value'].iloc[0]
                    value_10q = tag_data_10q[tag_data_10q['composite_key'] == key ]['value'].iloc[0]
                    
                    # Parse the composite key back to components
                    segments, ddate, version = key.split('|')
                    segments = segments if segments else 'None'
                    version = version if version else 'None'
                    
                    print(f"        {ddate} ({version}) - Segments: {segments}")
                    print(f"          10-K value: {value_10k}")
                    print(f"          10-Q value: {value_10q}")
                    
                    # Check if values are identical
                    if abs(float(value_10k) - float(value_10q)) < 0.01:  # Allow small floating point differences
                        print(f"          Status: IDENTICAL ✓")
                    else:
                        print(f"          Status: DIFFERENT ✗ (Difference: "
                              f"{abs(float(value_10k) - float(value_10q)):.2f})")
                
                if len(matching_keys) > 5:
                    print(f"        ... and {len(matching_keys) - 5} more matching records")
    
    # Step 4: Summary analysis
    print(f"\n" + "="*80)
    print("SUMMARY ANALYSIS")
    print("="*80)
    
    print("This analysis reveals whether companies:")
    print("1. Repeat quarterly data in annual reports (10-K)")
    print("2. Skip certain quarterly disclosures in annual reports")
    print("3. Use different values for the same metrics across forms")
    print("4. Maintain consistency between quarterly and annual reporting")
    
    return df_joined, companies_with_both


def tags_w_percent_str(df_joined):
    """
    Utility function to:
    1. Print out the first 10 records with 'percent' in the tag name 
       where custom_tag is 0
    
    This helps identify standard (non-custom) percentage-based financial metrics
    that companies report in their SEC filings.

    Args:
        df_joined (DataFrame): Pre-loaded and joined SEC filing data

    CONCLUSION: 
    there are some tags with "percent" in the name but very few, 
    and there are not top tags by popularity. 
    """
    print("\n" + "="*80)
    print("PERCENT TAG ANALYSIS - 2022Q1 DATA")
    print("="*80)
    
    # Step 1: Find records with 'percent' in tag name and custom_tag = 0
    print("Step 1: Finding records with 'percent' in tag name and custom_tag = 0...")
    # Filter for percent tags with custom_tag = 0
    percent_records = df_joined[
        (df_joined['tag'].str.contains('percent', case=False, na=False)) & 
        (df_joined['custom_tag'] == 0)
    ]

    percent_tags = percent_records['tag'].value_counts().reset_index()
    percent_tags.rename(columns={'count': 'tag_count_in_q1'}) 

    # now join with the tag stats table to see how popular these percentage tags are 
    top_tags_df = pd.read_csv('tag_stats_sorted.csv')
    top_tags_df['rank'] = range(len(top_tags_df))
    percent_tags = pd.merge(top_tags_df, percent_tags, on='tag')

    print(percent_tags.head(10))


def tag_stats_for_form(df_joined, form_type):
    """
    Utility function to analyze form statistics for a specified form type:
    1. Identifies all forms of the specified type (e.g., '10-K' or '10-Q')
    2. Counts total tags and unique tags reported in each form
    3. Shows statistics and sample data for analysis
    
    Args:
        df_joined (DataFrame): Pre-loaded and joined SEC filing data
        form_type (str): The form type to analyze (e.g., '10-K', '10-Q', '8-K')
    
    This helps understand the scope and diversity of financial disclosures
    in different types of SEC filings across different companies.
    """
    print(f"\n" + "="*80)
    print(f"{form_type} FORM STATISTICS ANALYSIS")
    print("="*80)
    
    print(f"Total records loaded: {len(df_joined):,}")
    print(f"Total unique companies (CIK): {df_joined['cik'].nunique():,}")
    print(f"Total unique tags: {df_joined['tag'].nunique():,}")
    
    # Step 1: Filter for specified form type
    print(f"\nStep 1: Analyzing {form_type} forms...")
    df_form = df_joined[df_joined['form'] == form_type]
    
    if len(df_form) == 0:
        print(f"No {form_type} forms found in the data.")
        return None, None
    
    print(f"Found {len(df_form):,} total records from {form_type} forms")
    print(f"Unique companies with {form_type} forms: {df_form['cik'].nunique():,}")
    
    # Group by company and calculate statistics for all tags
    company_stats = df_form.groupby(['cik', 'name']).agg({
        'adsh': 'nunique',  # Number of unique accession numbers (forms)
        'tag': ['count', 'nunique'],  # Total tags and unique tags
        'custom_tag': 'sum'  # Count of custom tags
    }).reset_index()
    # Flatten column names
    company_stats.columns = [
        'cik', 'name', 'num_forms', 'total_tags', 'unique_tags', 'custom_tags'
    ]
    
    # Filter for standard tags only and calculate statistics
    df_form_standard = df_form[df_form['custom_tag'] == 0]
    company_stats_standard = df_form_standard.groupby(['cik', 'name']).agg({
        'tag': ['count', 'nunique']  # Total standard tags and unique standard tags
    }).reset_index()
    # Flatten column names for standard stats
    company_stats_standard.columns = [
        'cik', 'name', 'total_standard_tags', 'unique_standard_tags'
    ]
    
    # Join the two stats tables together
    company_stats_combined = company_stats.merge(
        company_stats_standard, 
        on=['cik', 'name'], 
        how='left'
    )
    
    # Fill NaN values with 0 for companies that might not have standard tags
    company_stats_combined['total_standard_tags'] = (
        company_stats_combined['total_standard_tags'].fillna(0)
    )
    company_stats_combined['unique_standard_tags'] = (
        company_stats_combined['unique_standard_tags'].fillna(0)
    )
    
    # Sort by total tags (most comprehensive reports first)
    company_stats_sorted = company_stats_combined.sort_values('total_tags', ascending=False)
    print("A typical company has:")
    print(f"  {company_stats_sorted['total_tags'].mean():.1f} total tags")
    print(f"  {company_stats_sorted['unique_tags'].mean():.1f} unique tags")
    print(f"  {company_stats_sorted['total_standard_tags'].mean():.1f} total standard tags")
    print(f"  {company_stats_sorted['unique_standard_tags'].mean():.1f} unique standard tags")  
    print("\n")


def tags_w_historic_comparisons(df_joined, form_type):
    """
    Utility function to identify tags with historic comparisons in SEC filings.
    
    A record has a historic comparison if there are other records with identical 
    (cik, name, tag, segments, qtrs) attributes but with an earlier ddate.
    
    Args:
        df_joined (DataFrame): Pre-loaded and joined SEC filing data
        form_type (str): The form type to analyze (e.g., '10-K', '10-Q', '8-K')
    
    Returns:
        DataFrame: tag_history_group table with historic comparison analysis

    CONCLUSION: 
    - Almost all tags have historic comparisons, 
    - most of them have only 1- or 2-yr horizon. 
    - Featurization: we should set a prediction horizon for each tag, and compute the 
      change percentage between the current and the historical value.
    - Featurization will see null values (when a historic comparison date has no record), 
      This should be fine for the ML algorithm.  
    """
    print(f"\n" + "="*80)
    print(f"HISTORIC COMPARISONS IDENTIFICATION - {form_type} FORMS")
    print("="*80)
    
    # Step 1: Filter data for standard tags and specified form type
    print(f"Step 1: Filtering data for {form_type} forms with standard tags...")
    
    # Filter for standard tags (custom_tag == 0) and specified form type
    # Also filter out records with NaN values in the 'value' field
    df_filtered = df_joined[
        (df_joined['custom_tag'] == 0) & 
        (df_joined['form'] == form_type) &
        (df_joined['value'].notna())
    ].copy()
    
    if len(df_filtered) == 0:
        print(f"No {form_type} forms with standard tags found in the data.")
        return None
    
    print(f"Found {len(df_filtered):,} records with standard tags in {form_type} forms")
    print(f"Unique companies: {df_filtered['cik'].nunique():,}")
    print(f"Unique tags: {df_filtered['tag'].nunique():,}")
    
    # Step 2: Convert ddate to datetime for proper analysis
    print(f"\nStep 2: Converting dates to datetime format...")
    df_filtered['ddate'] = pd.to_datetime(
        df_filtered['ddate'], 
        format='%Y%m%d', 
        errors='coerce'
    )
    
    # Remove records with invalid dates
    df_filtered = df_filtered.dropna(subset=['ddate'])
    print(f"Records with valid dates: {len(df_filtered):,}")
    
    # Step 3: Group records by key attributes and aggregate date information
    print(f"\nStep 3: Grouping records by key attributes and aggregating date information...")
    
    # Group by (cik, name, tag, segments, qtrs, form) and aggregate ddate
    tag_history_group = df_filtered\
        .groupby(['cik', 'name', 'tag', 'segments', 'qtrs'], dropna=False)\
        .agg({'ddate': [
            'nunique',  # (i) count of distinct ddate
            'max',      # (ii) max ddate (most recent)
            lambda x: sorted(x.unique())  # (iii) list of all historic ddates
             ] }
             ).reset_index()
    
    # Flatten column names
    tag_history_group.columns = [
        'cik', 'name', 'tag', 'segments', 'qtrs', 
        'distinct_ddate', 'max_ddate', 'historic_dates'
    ]
    
    print(f"Created {len(tag_history_group):,} grouped records")
    
    # Step 4: Compute time intervals between historic dates and max date
    print(f"\nStep 4: Computing time intervals for historic comparisons...")
    # Function to compute time intervals
    def compute_time_intervals(historic_dates, max_date):
        """Compute time intervals between historic dates and max date"""
        if not historic_dates or max_date is None:
            return []
        
        intervals = []
        for hist_date in historic_dates:
            if hist_date < max_date:
                interval_days = (max_date - hist_date).days
                intervals.append(interval_days)
        
        return sorted(intervals, reverse=True)  # Sort from longest to shortest interval
    
    # Apply the function to compute time intervals
    tag_history_group.loc[:, 'time_intervals'] = tag_history_group.apply(
        lambda row: compute_time_intervals(row['historic_dates'], row['max_ddate']), 
        axis=1
    )
    
    # Apply round_to_nearest_quarter to each interval in the time_intervals list
    tag_history_group.loc[:, 'time_intervals'] = \
     tag_history_group['time_intervals'].apply(
        lambda intervals: [round_to_nearest_quarter_days(interval) for interval in intervals]
    )

    tag_history_group.loc[:, 'hist_horizon'] = \
    tag_history_group['time_intervals'].apply(
        lambda intervals: max([0] + intervals)  
    )

    tag_stats = tag_history_group[tag_history_group['distinct_ddate']>1]\
        .groupby(['tag'], dropna=False)\
        .agg({
            'cik': 'nunique', 
            'distinct_ddate': ['mean', pd.Series.mode],
            'hist_horizon': ['max', lambda x: x.quantile(0.75), 'median']
        })\
        .reset_index()
    tag_stats.columns= [
        'tag', 'distinct_cik', 'mean_disctint_ddates', 'mode_distinct_ddates', 
        'hist_horizon_max', 'hist_horizon_75pct', 'hist_horizon_median'
    ]
    tag_stats.sort_values(by='distinct_cik', ascending=False)\
             .to_csv('tag_historic_comp_stats_sorted.csv', index=False)


def tags_w_segment_groups(df_joined, form_type):
    """
    Utility function to identify segment groups in SEC filings.
    
    A segment group is defined as a group of records with identical 
    (cik, name, tag, qtrs, ddate) but with variable "segments" attributes.
    If such a group is identified and contains a record with segments=NaN, 
    that record is declared as the summary with summary_flag=1.
    
    Args:
        df_joined (DataFrame): Pre-loaded and joined SEC filing data
        form_type (str): The form type to analyze (e.g., '10-K', '10-Q', '8-K')
    
    Returns:
        DataFrame: segment_group table with segment group analysis


    Observations: 
    - good news is the 97% of the segment groups have a summary record 
    - Some tags have segment groups with no clear summary record, and I am not sure 
      how we should generate the summary. 
    - problematic tags mostly have "stock" or "shares" in the tag name. 
    There seem to be duplicate values for these tags, so we cannot simply add them up. 
    Getting the max is probably safer. 
    - I am not sure we need to worry about generating the summary since it only 
      account for 2-3% of segment groups. 
    """
    print(f"\n" + "="*80)
    print(f"SEGMENT GROUP IDENTIFICATION - {form_type} FORMS")
    print("="*80)
    
    # Step 1: Filter data for standard tags and specified form type
    print(f"Step 1: Filtering data for {form_type} forms with standard tags...")
    
    df_filtered = df_joined[
        (df_joined['custom_tag'] == 0) & 
        (df_joined['form'] == form_type) &
        (df_joined['value'].notna())
    ].copy()

    print(f"Found {len(df_filtered):,} records with standard tags in {form_type} forms")
    print(f"Unique companies: {df_filtered['cik'].nunique():,}")
    print(f"Unique tags: {df_filtered['tag'].nunique():,}")
    
    # Step 2: Group records by key attributes and aggregate segments information
    print(f"\nStep 2: Grouping records by (cik, name, tag, ddate, qtrs) and analyzing segments...")
    
    # Group by (cik, name, tag, ddate, qtrs) and aggregate segments column
    segment_group = df_filtered.groupby(['cik', 'name', 'tag', 'ddate', 'qtrs'], dropna=False).agg({
        'segments': [
            'nunique',  # (i) number of distinct segments
            lambda x: x.isna().any(), 
            lambda x: len(x)==1, 
            lambda x: x.isna().any() or len(x)==1 # (ii) whether a summary record exists (segments=NaN)
        ]
    }).reset_index()
    
    # Flatten column names
    segment_group.columns = [
        'cik', 'name', 'tag', 'ddate', 'qtrs',
        'distinct_segments', 'has_segment_null', 'only_1_segment', 'has_summary_record'
    ]
    
    print(f"Created {len(segment_group):,} grouped records")

    print(len(segment_group), 'total segment groups')
    print(sum(segment_group.has_summary_record), 'has summary record (either has only 1 segment or has segment=Nan)')
    print("%.2f"%(segment_group.has_summary_record.mean()*100), '% has summary record')
    print("%.2f"%(segment_group.only_1_segment.mean()*100), '% has only 1 segment')
    print("%.2f"%(segment_group.has_segment_null.mean()*100), '% has segment==null record')

    # need to spot check the problematic segment groups
    segment_group_problematic = segment_group[
        segment_group['distinct_segments']>1
    ][segment_group['has_summary_record']==False]
    print(len(segment_group_problematic), 
          'segment groups are problematic: multiple segments but no summary')
    print(' from ', segment_group_problematic.cik.nunique(), 'distinct companies')
    print(' from ', segment_group_problematic.tag.nunique(), 'distinct tags')

    tag_has_segment_summary_stats = (
        segment_group.groupby('tag').agg({'has_summary_record':'mean'})
    ).reset_index().sort_values(by='has_summary_record', ascending=True)

    # now join with the tag stats table to see how popular these percentage tags are 
    top_tags_df = pd.read_csv('tag_stats_sorted.csv')
    top_tags_df['rank'] = range(len(top_tags_df))
    tag_has_segment_summary_stats = pd.merge(
        tag_has_segment_summary_stats, 
        top_tags_df[['tag', 'rank']], 
        on='tag'
    )
   
    # print the tags that have less than 90% summary records and are in the top 500 tags
    problematic_tags = tag_has_segment_summary_stats[
        (tag_has_segment_summary_stats['has_summary_record']<0.9) & 
        (tag_has_segment_summary_stats['rank']<500)
        ].sort_values(by='rank')
    
    if len(problematic_tags) > 0:
        print(f"\nTags with <90% summary records (Top 500 tags):")
        print(f"{'Rank':<8} {'Tag':<50} {'Summary %':<12}")
        print("-" * 70)
        for _, row in problematic_tags.iterrows():
            summary_pct = f"{row['has_summary_record']*100:.1f}%"
            # Truncate long tag names for display
            display_tag = row['tag'][:47] + "..." if len(row['tag']) > 50 else row['tag']
            print(f"{row['rank']:<8} {display_tag:<50} {summary_pct:<12}")
        print(f"\nFound {len(problematic_tags)} tags with <90% summary records")
    else:
        print("\nAll top 500 tags have >=90% summary records")


def round_to_nearest_quarter_days(interval_days):
    """
    Utility function to round a date interval to the nearest quarter boundary 
    and return the day count.
    
    Args:
        interval_days (int): Number of days in the interval
    
    Returns:
        int: Rounded interval in days (e.g., 91, 182, 365, 730)
    
    Quarter boundaries:
    - 1 quarter: 91 days
    - 2 quarters: 182 days  
    - 3 quarters: 273 days
    - 4 quarters (1 year): 365 days
    - 8 quarters (2 years): 730 days
    """
    # Define quarter boundaries in days
    quarter_boundaries = [91, 182, 273, 365, 456, 547, 638, 730]
    
    # Find the nearest quarter boundary
    if interval_days <= 0:
        return interval_days  # Return original value for invalid intervals
    
    # Find the closest quarter boundary
    closest_boundary = min(quarter_boundaries, key=lambda x: abs(x - interval_days))
    distance = abs(closest_boundary - interval_days)
    
    # If the distance is more than 15 days from any boundary, return original value
    if distance > 15:
        return interval_days
    
    return closest_boundary


if __name__ == "__main__": 
    data_dir = ['data/2022q1', 
                'data/2022q2', 
                'data/2022q3', 
                'data/2022q4']
    df_joined = load_and_join_sec_xbrl_data(data_dir) 
    top_tags(df_joined)  
    
    # now focus on Q1 data only
    data_dir = ['data/2022q1']
    df_joined = load_and_join_sec_xbrl_data(data_dir) 

    ## Run percent tag analysis for 2022q1
    # tags_w_percent_str(df_joined)
    
    # Run form statistics analysis for 10-K
    tag_stats_for_form(df_joined, '10-K')
    
    # Run form statistics analysis for 10-Q
    tag_stats_for_form(df_joined, '10-Q')
    
    # # Run historic comparisons identification for 10-K
    # tag_history_group_10k = tags_w_historic_comparisons(df_joined, '10-K')
    
    # # Run historic comparisons identification for 10-Q
    # tag_history_group_10q = tags_w_historic_comparisons(df_joined, '10-Q')
    
    # Run segment group identification for 10-K
    segment_group_10k = tags_w_segment_groups(df_joined, '10-K')
    
    # # Run segment group identification for 10-Q
    segment_group_10q = tags_w_segment_groups(df_joined, '10-Q')


## sample prompts
# SEC filings sometimes have tags with multiple segments values. For instance, 
# revenues may be reported based on services/products and sometimes with 
# further details on finer granularity on product groups, and then there is 
# a summary record where segments=NaN reporting on total revenue. We'd like 
# to write a utility function to identify "segment groups". A segment group 
# is defined as a group of records with identical (cik, name, tag, qtrs, 
# ddate) but with variable "segments" attributes. If such a group is 
# identified, also if there is a record with segments=NaN, then declare 
# this record as the summary, with summary_flag=1. The other records have 
# summary_flag=0. 

# We will build this utility function step by step. The function takes two 
# parameters: a dataframe df_joined, and a form type (10-K or 10-Q). First, 
# from the dataframe, filter on records with custom_tag == 0 (standard tags) 
# and form_type. Then group records by (cik, name, tag, ddate, qtrs) and 
# aggregate the segments column to get: (i) the number of distinct segments, 
# (ii) whether a summary record exists. We call the aggregate table 
# "segment_group".  
