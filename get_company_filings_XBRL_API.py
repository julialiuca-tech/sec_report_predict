#!/usr/bin/env python3
"""
SEC Company Filings XBRL Data Retrieval

Retrieves XBRL tags and values for a company's 10-Q or 10-K filing from SEC JSON API.
Saves the data to a CSV file named {ticker}_{form_type}_{report_date}.csv

Features:
  - Provides programmatic access to company filing history via `SECCompanyFilings` class
  - Supports retrieval by CIK or ticker symbol
  - Can filter by form type (10-Q, 10-K, 8-K, etc.)
  - Returns data as pandas DataFrame or CSV 
  - Converts XBRL JSON API schema to bulk data schema (NUM.txt format)

Usage Examples:
  - Command line: `python get_company_filings_XBRL_API.py --ticker MSFT --form 10-Q --target-date 2025-04-30`
  - Programmatic: `from get_company_filings_XBRL_API import SECCompanyFilings`

Note: SEC API rate limit is 10 requests/second, User-Agent header required

===============================================================================
SEC XBRL Schema Mapping: JSON API vs Bulk Data
===============================================================================

This module converts data between two different SEC data schemas:
1. XBRL JSON API (Company Facts API) - Structured JSON format from data.sec.gov
2. Bulk Data - Tab-delimited text files (num.txt, sub.txt) from quarterly downloads

Field Mappings:

Date Fields:
  - filing_date / filed (JSON) ↔ filed (Bulk SUB) - Filing submission date
    Format: YYYY-MM-DD (JSON) / YYYYMMDD (bulk)
  - period_end / end (JSON) ↔ ddate (Bulk NUM) - Period end date
    Format: YYYY-MM-DD (JSON) / YYYYMMDD (bulk)
    Key mapping: period_end ≈ ddate
  - period_start / start (JSON) - Period start date (JSON API only)
  - report_date (JSON) ↔ period (Bulk SUB) - Balance sheet date
    Key mapping: report_date ≈ period

Duration Fields:
  - frame (JSON) ↔ qtrs (Bulk) - Duration indicator
  
  Frame Format (JSON API):
    - Format: CY{YYYY}Q{Q}I or CY{YYYY}Q{Q}D
    - CY = Calendar Year (vs FY = Fiscal Year)
    - {YYYY} = Year (e.g., 2024)
    - Q{Q} = Quarter (1, 2, 3, 4)
    - I = Instant (point in time, like balance sheet)
    - D = Duration (period of time, like income statement)
    - Example: CY2024Q1I = Calendar Year 2024 Q1 Instant (March 31, 2024 balance sheet)
  
  Qtrs Field (Bulk Data):
    - 0 = Instant (point in time, e.g., balance sheet)
    - 1 = 1 quarter duration
    - 2 = 2 quarter duration (half year)
    - 3 = 3 quarter duration
    - 4 = 4 quarter duration (full year)
  
  Mapping Logic:
    - qtrs = 0 corresponds to frame ending with 'I' (Instant)
    - qtrs > 0 corresponds to frame with duration periods

Submission/Metadata Fields:
  - accn (JSON) ↔ adsh (Bulk) - Accession number (20-character SEC identifier)
  - form (JSON) ↔ form (Bulk) - Form type (10-Q, 10-K, etc.)
  - fy, fp (Bulk only) - Fiscal Year and Fiscal Period

Data Structure Differences:

JSON API Structure:
  {
    "facts": {
      "us-gaap": {
        "Assets": {
          "units": {
            "USD": [{
              "val": 1000000,
              "filed": "2024-11-05",
              "start": null,
              "end": "2024-09-30",
              "frame": "CY2024Q3I",
              "form": "10-Q",
              "accn": "0000320193-24-000077"
            }]
          }
        }
      }
    }
  }

Bulk Data Structure:
  - NUM.txt (numeric facts): adsh, tag, version, ddate, qtrs, uom, value, coreg, footnote, segments
  - SUB.txt (submissions): adsh, cik, name, sic, fye, fy, fp, form, period, filed, etc.

Key Relationships:
  1. Period End Date: period_end (JSON) → ddate (Bulk NUM)
     period_end.strftime('%Y%m%d') == ddate
  2. Report Date: report_date (JSON) → period (Bulk SUB)
     Note: period may differ slightly from ddate due to rounding to month-end
  3. Filing Date: filing_date (JSON) → filed (Bulk SUB)
     filing_date.strftime('%Y%m%d') == filed
  4. Duration: frame (JSON) → qtrs (Bulk)
     See frame_to_qtrs() function for conversion logic

Direct Mappings:
  - filing_date ↔ filed - Filing submission date
  - period_end / end ↔ ddate (NUM) / period (SUB) - Period end date
  - accn ↔ adsh - Accession number
  - form ↔ form - Form type

Derived/Computed Fields:
  - report_date - Generally equals period_end in JSON API, maps to period in SUB
  - qtrs - Derived from frame format (0 for instant, numeric for duration)

JSON API Only:
  - period_start / start - Period start date
  - frame - Formatted duration string (bulk has numeric qtrs instead)

Bulk Data Only:
  - fy - Fiscal year
  - fp - Fiscal period (FY, Q1, Q2, Q3, Q4)

References:
  - SEC Bulk Data Documentation: data/SEC_raw_*/readme.htm
  - SEC JSON API Documentation: https://www.sec.gov/edgar/sec-api-documentation
  - XBRL Guide: https://www.sec.gov/files/edgar/filer-information/specifications/xbrl-guide-2024-07-08.pdf
"""

import requests
import argparse
import pandas as pd
import re
from typing import Dict, Optional


class SECCompanyFilings:
    """Class to retrieve company XBRL data from SEC JSON API."""
    
    BASE_URL = "https://data.sec.gov"
    USER_AGENT = "SEC Data Analysis (your-email@domain.com)"
    
    def __init__(self, cik: str = None, ticker: str = None, user_agent: str = None):
        """Initialize with either CIK or ticker symbol."""
        if user_agent:
            self.user_agent = user_agent
        else:
            self.user_agent = self.USER_AGENT
            
        self.cik = cik
        if ticker and not cik:
            self.cik = self._ticker_to_cik(ticker)
            self.ticker = ticker.upper()
        elif cik:
            self.ticker = None  # Will be fetched from company info
        
        if not self.cik:
            raise ValueError("Must provide either CIK or ticker symbol")
        
        # Ensure CIK is 10 digits with leading zeros
        self.cik = str(self.cik).zfill(10)
        
    def _ticker_to_cik(self, ticker: str) -> str:
        """Convert ticker symbol to CIK."""
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {'User-Agent': self.user_agent}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        for entry in data.values():
            if entry['ticker'] == ticker.upper():
                return str(entry['cik_str']).zfill(10)
        
        raise ValueError(f"Ticker {ticker} not found in SEC database")
    
    def get_company_info(self) -> Dict:
        """Get basic company information including ticker."""
        url = f"{self.BASE_URL}/submissions/CIK{self.cik}.json"
        headers = {'User-Agent': self.user_agent}
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        tickers = data.get('tickers', [])
        ticker = tickers[0] if tickers else ''
        
        if not hasattr(self, 'ticker') or not self.ticker:
            self.ticker = ticker
        
        return {
            'cik': self.cik,
            'name': data.get('name', ''),
            'ticker': ticker,
        }
    
    def get_company_facts(self) -> Dict:
        """Get all XBRL company facts (tags and values)."""
        url = f"{self.BASE_URL}/api/xbrl/companyfacts/CIK{self.cik}.json"
        headers = {'User-Agent': self.user_agent}
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        return response.json()
    
    def check_tag_exists(self, tag_name: str):
        """
        Check if a tag exists in the raw XBRL JSON API data and show details.
        
        Args:
            tag_name: Tag name to search for (case-sensitive)
        
        Prints diagnostic information about the tag.
        """
        facts = self.get_company_facts()
        
        if 'facts' not in facts:
            print(f"❌ No facts found in XBRL data")
            return
        
        all_taxonomies = list(facts.get('facts', {}).keys())
        print(f"🔍 Checking for tag: {tag_name}")
        print(f"   Available taxonomies: {all_taxonomies}")
        print("=" * 80)
        
        found_in_taxonomies = []
        
        for taxonomy in all_taxonomies:
            taxonomy_facts = facts.get('facts', {}).get(taxonomy, {})
            
            if tag_name in taxonomy_facts:
                found_in_taxonomies.append(taxonomy)
                concept_data = taxonomy_facts[tag_name]
                units = concept_data.get('units', {})
                
                total_facts = 0
                forms = set()
                accns = set()
                periods = set()
                
                for facts_list in units.values():
                    total_facts += len(facts_list)
                    for fact in facts_list:
                        forms.add(fact.get('form', 'NO_FORM'))
                        accns.add(fact.get('accn', 'NO_ACCN'))
                        if fact.get('end'):
                            periods.add(fact.get('end'))
                
                print(f"\n✅ Found in taxonomy: '{taxonomy}'")
                print(f"   Total facts: {total_facts}")
                print(f"   Forms: {sorted(forms)}")
                print(f"   Unique accns: {len(accns)}")
                if len(accns) <= 5:
                    print(f"   Accns: {sorted(list(accns))}")
                else:
                    print(f"   Sample accns: {sorted(list(accns))[:5]} ... ({len(accns)} total)")
                print(f"   Unique periods: {len(periods)}")
                if len(periods) <= 5:
                    print(f"   Periods: {sorted(list(periods))}")
                else:
                    print(f"   Sample periods: {sorted(list(periods))[:5]} ... ({len(periods)} total)")
        
        if not found_in_taxonomies:
            print(f"\n❌ Tag '{tag_name}' NOT FOUND in any taxonomy")
            # Check for similar tag names
            print("\n   Searching for similar tags...")
            similar_found = False
            tag_lower = tag_name.lower()
            for taxonomy in all_taxonomies:
                taxonomy_facts = facts.get('facts', {}).get(taxonomy, {})
                similar = [t for t in taxonomy_facts.keys() 
                          if tag_lower in t.lower() or 
                          ('proceed' in tag_lower and 'proceed' in t.lower()) or
                          ('investment' in tag_lower and 'investment' in t.lower())]
                if similar:
                    similar_found = True
                    print(f"   Similar tags in '{taxonomy}': {similar[:10]}")
            if not similar_found:
                print("   No similar tags found")
        
        print("=" * 80)
    
    def get_xbrl_data_dataframe(self, form_type: str, taxonomy: str = 'us-gaap', 
                                target_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get XBRL tags and values as a pandas DataFrame for a specific form type.
        
        Args:
            form_type: Form type ('10-Q' or '10-K')
            taxonomy: XBRL taxonomy to use (default: 'us-gaap')
            target_date: Optional target date (YYYY-MM-DD format). If provided, returns data
                        for the report whose report_date is closest to target_date.
                        If not provided, returns the most recent report's data.
            
        Returns:
            DataFrame with columns: tag, value, period_start, period_end, form, filing_date, etc.
            Returns data for a single filing (either closest to target_date or most recent).
        """
        facts = self.get_company_facts()
        
        if 'facts' not in facts:
            return pd.DataFrame()
        
        taxonomy_facts = facts.get('facts', {}).get(taxonomy, {})
        
        if not taxonomy_facts:
            return pd.DataFrame()
        
        records = []
        
        # Iterate through all tags/concepts
        for tag, concept_data in taxonomy_facts.items():
            units = concept_data.get('units', {})
            
            # Process each unit (e.g., USD, shares, etc.)
            for unit, facts_list in units.items():
                for fact in facts_list:
                    # Filter by form type
                    fact_form = fact.get('form', '')
                    if fact_form != form_type:
                        continue
                    
                    record = {
                        'tag': tag,
                        'taxonomy': taxonomy,
                        'unit': unit,
                        'value': fact.get('val'),
                        'form': fact_form,
                        'filing_date': fact.get('filed'),
                        'period_start': fact.get('start'),
                        'period_end': fact.get('end'),
                        'report_date': fact.get('end'),  # Alias for period_end
                        'frame': fact.get('frame'),
                        'accn': fact.get('accn'),  # Accession number
                    }
                    
                    records.append(record)
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        # Convert dates
        date_cols = ['filing_date', 'period_start', 'period_end', 'report_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Filter to a single filing based on target_date
        # A filing can contain multiple periods (comparative data), so we need to:
        # 1. Find the target report_date (closest to target_date or most recent)
        # 2. Identify the filing (accn) that contains that report_date
        # 3. Return ALL records from that filing (including all comparative periods)
        
        if target_date:
            # Find the report_date closest to target_date
            target_dt = pd.to_datetime(target_date, errors='coerce')
            if pd.isna(target_dt):
                print(f"⚠️  Warning: Invalid target_date format '{target_date}'. Using most recent report.")
                target_date = None
        
        # Get unique report dates
        unique_report_dates = df['report_date'].dropna().unique()
        if len(unique_report_dates) == 0:
            print("⚠️  Warning: No valid report dates found. Returning empty DataFrame.")
            return pd.DataFrame()
        
        if target_date:
            # Find report_date closest to target_date
            target_dt = pd.to_datetime(target_date)
            date_diffs = abs(unique_report_dates - target_dt)
            closest_report_date = unique_report_dates[date_diffs.argmin()]
            
            # Find the filing (accn) that contains this report_date
            filing_matches = df[df['report_date'] == closest_report_date]
            if filing_matches.empty:
                print("⚠️  Warning: No filing found with the closest report_date. Returning empty DataFrame.")
                return pd.DataFrame()
            
            # Get the accession number(s) for filings containing this report_date
            # accn uniquely identifies a filing
            accn_values = filing_matches['accn'].dropna().unique()
            
            if len(accn_values) == 0:
                print("⚠️  Warning: No accession numbers found. Using filing_date as fallback.")
                filing_dates = filing_matches['filing_date'].unique()
                df = df[df['filing_date'].isin(filing_dates)].copy()
            else:
                # Filter to all records from filings with these accn values
                # This includes all periods (comparative data) in the filing
                df = df[df['accn'].isin(accn_values)].copy()
            
            # Get the primary report_date (the one closest to target)
            primary_report_date = closest_report_date
            
            print(f"📅 Selected filing(s) with primary report_date: {primary_report_date.strftime('%Y-%m-%d')} "
                  f"(closest to target: {target_date})")
            print(f"   Included {df['report_date'].nunique()} unique periods from {df['accn'].nunique()} filing(s)")
        else:
            # Get most recent report (by report_date)
            most_recent_report_date = unique_report_dates.max()
            
            # Find the filing that contains this most recent report_date
            filing_matches = df[df['report_date'] == most_recent_report_date]
            
            # Get the accession number(s) for the most recent filing
            accn_values = filing_matches['accn'].dropna().unique()
            
            if len(accn_values) == 0:
                print("⚠️  Warning: No accession numbers found. Using filing_date as fallback.")
                filing_dates = filing_matches['filing_date'].unique()
                df = df[df['filing_date'].isin(filing_dates)].copy()
            else:
                # Filter to all records from the most recent filing(s)
                # This includes all periods (comparative data) in the filing
                df = df[df['accn'].isin(accn_values)].copy()
            
            if not df.empty:
                filing_date = df['filing_date'].iloc[0]
                accn = df['accn'].iloc[0] if pd.notna(df['accn'].iloc[0]) else 'N/A'
                print(f"📅 Selected most recent filing with primary report_date: {most_recent_report_date.strftime('%Y-%m-%d')}")
                print(f"   Filing date: {filing_date.strftime('%Y-%m-%d') if pd.notna(filing_date) else 'N/A'}, "
                      f"Accession: {accn}")
                print(f"   Included {df['report_date'].nunique()} unique periods from {df['accn'].nunique()} filing(s)")
        
        # Sort by tag, then by period_end for consistency
        df = df.sort_values(['tag', 'period_end'], ascending=[True, False]).reset_index(drop=True)
        
        return df


def frame_to_qtrs(frame: Optional[str], period_start: Optional[str] = None, 
                  period_end: Optional[str] = None) -> int:
    """
    Convert XBRL frame format to bulk data qtrs value.
    
    Converts the JSON API frame format (e.g., "CY2024Q1I") to the bulk data numeric
    qtrs value representing duration in quarters.
    
    The qtrs field represents duration in quarters:
    - 0 = Instant (point in time, like balance sheet)
      - 1 = 1 quarter duration
      - 2 = 2 quarter duration (half year)
      - 3 = 3 quarter duration
      - 4 = 4 quarter duration (full year)
    
    Frame Format (JSON API):
      - Format: CY{YYYY}Q{Q}I or CY{YYYY}Q{Q}D
      - CY = Calendar Year (vs FY = Fiscal Year)
      - {YYYY} = Year (e.g., 2024)
      - Q{Q} = Quarter (1, 2, 3, 4)
      - I = Instant (point in time, like balance sheet)
      - D = Duration (period of time, like income statement)
      - Example: CY2024Q1I = Calendar Year 2024 Q1 Instant
    
    Mapping Logic:
      - qtrs = 0 corresponds to frame ending with 'I' (Instant)
      - qtrs > 0 corresponds to frame with duration periods
    
    The function prioritizes calculation from period dates (most accurate), then
    falls back to parsing the frame format.
    
    Args:
        frame: Frame string like 'CY2024Q1I' or 'CY2024Q3D'
        period_start: Period start date (YYYY-MM-DD) - used to calculate duration
                     (most accurate method)
        period_end: Period end date (YYYY-MM-DD) - used to calculate duration
                   (most accurate method)
    
    Returns:
        int: Number of quarters (0 = instant, 1-4 = quarters duration)
    
    Examples:
        >>> frame_to_qtrs('CY2024Q1I')  # Returns 0 (instant)
        >>> frame_to_qtrs('CY2024Q3D', '2024-07-01', '2024-09-30')  # Returns 1 (1 quarter)
    """
    # First, check if frame indicates instant (I suffix)
    if frame and frame.endswith('I'):
        return 0  # Instant (point in time)
    
    # Calculate from period dates if available (most accurate)
    if period_start and period_end:
        try:
            start = pd.Timestamp(period_start)
            end = pd.Timestamp(period_end)
            
            # If same date or very close, it's an instant value
            days = (end - start).days
            if days <= 1:
                return 0
            
            # Calculate quarters: each quarter is approximately 90-92 days
            # Round to nearest quarter
            qtrs = round(days / 91.25)  # Average days per quarter
            
            # Cap at reasonable range (0-4 quarters is most common)
            qtrs = max(0, min(qtrs, 4))
            
            return qtrs
        except Exception:
            pass
    
    # Fallback: parse frame if available
    if frame:
        # Frame format: CY2024Q1I, CY2024Q3D, etc.
        # D = Duration
        if frame.endswith('D'):
            # Extract quarter number and try to infer duration
            # This is approximate - actual duration is better calculated from dates
            match = re.search(r'Q(\d+)', frame)
            if match:
                # For duration frames, return the quarter number as approximation
                # Q1=1 quarter, Q2=2 quarters (half year), Q3=3 quarters, Q4=4 quarters (full year)
                return int(match.group(1))
    
    # Default to 0 if unable to determine
    return 0


def convert_to_bulk_schema(df_xbrl: pd.DataFrame, taxonomy: str = 'us-gaap') -> pd.DataFrame:
    """
    Convert XBRL JSON API schema DataFrame to bulk data schema (NUM.txt format).
    
    This function transforms data from the SEC JSON API format to match the bulk data
    schema used in NUM.txt files from quarterly SEC downloads. This enables interoperability
    between the two data sources.
    
    Field Mappings:
      - accn (JSON) → adsh (Bulk) - Accession number
      - tag (JSON) → tag (Bulk) - Tag name
      - taxonomy → version (Bulk) - Taxonomy identifier (or adsh for custom tags)
      - period_end (JSON) → ddate (Bulk) - Period end date (converted to YYYYMMDD format)
      - frame (JSON) → qtrs (Bulk) - Duration in quarters (via frame_to_qtrs())
      - unit (JSON) → uom (Bulk) - Unit of measure
      - value (JSON) → value (Bulk) - Numeric value
    
    Fields Not Available in JSON API (set to empty):
      - segments - XBRL segments (empty in this conversion)
      - coreg - Coregistrant (empty in this conversion)
      - footnote - Footnote reference (empty in this conversion)
    
    Bulk data schema columns (in order):
      1. adsh - Accession number (from accn)
      2. tag - Tag name
      3. version - Taxonomy identifier (or adsh for custom tags)
      4. ddate - Period end date (YYYYMMDD format)
      5. qtrs - Duration in quarters (0=instant)
      6. uom - Unit of measure
      7. segments - XBRL segments (empty)
      8. coreg - Coregistrant (empty)
      9. value - Numeric value
      10. footnote - Footnote reference (empty)
    
    Args:
        df_xbrl: DataFrame with XBRL JSON API schema columns:
                 tag, taxonomy, unit, value, form, filing_date, period_start,
                 period_end, report_date, frame, accn
        taxonomy: Taxonomy name (default: 'us-gaap') - used for version field
    
    Returns:
        DataFrame with bulk data schema matching NUM.txt format:
        Columns: adsh, tag, version, ddate, qtrs, uom, segments, coreg, value, footnote
    
    Example:
        >>> df_xbrl = client.get_xbrl_data_dataframe(form_type='10-Q')
        >>> df_bulk = convert_to_bulk_schema(df_xbrl, taxonomy='us-gaap')
        >>> # df_bulk now matches the schema of NUM.txt from bulk downloads
    """
    if df_xbrl.empty:
        return pd.DataFrame()
    
    df_bulk = pd.DataFrame()
    
    # Map fields
    df_bulk['adsh'] = df_xbrl['accn'].fillna('')
    
    df_bulk['tag'] = df_xbrl['tag'].fillna('')
    
    # Version: for standard tags, use taxonomy identifier; for custom tags, use adsh
    # Since we're filtering by taxonomy, assume all are standard tags
    df_bulk['version'] = taxonomy
    
    # Convert period_end to ddate (YYYYMMDD format)
    # Handle both string and datetime formats
    period_end_dates = pd.to_datetime(df_xbrl['period_end'], errors='coerce')
    df_bulk['ddate'] = period_end_dates.dt.strftime('%Y%m%d').fillna('')
    
    # Convert frame to qtrs
    def calculate_qtrs(row):
        # Convert dates to string format if they're datetime objects
        period_start = row.get('period_start')
        period_end = row.get('period_end')
        
        # Convert datetime to string if needed
        if pd.notna(period_start) and isinstance(period_start, pd.Timestamp):
            period_start = period_start.strftime('%Y-%m-%d')
        elif pd.notna(period_start):
            period_start = str(period_start)
        else:
            period_start = None
            
        if pd.notna(period_end) and isinstance(period_end, pd.Timestamp):
            period_end = period_end.strftime('%Y-%m-%d')
        elif pd.notna(period_end):
            period_end = str(period_end)
        else:
            period_end = None
        
        return frame_to_qtrs(
            row.get('frame'),
            period_start,
            period_end
        )
    
    df_bulk['qtrs'] = df_xbrl.apply(calculate_qtrs, axis=1)
    
    # Unit of measure
    df_bulk['uom'] = df_xbrl['unit'].fillna('')
    
    # Segments: empty (not available in JSON API)
    df_bulk['segments'] = ''
    
    # Coregistrant: empty (not available in JSON API)
    df_bulk['coreg'] = ''
    
    # Value
    df_bulk['value'] = df_xbrl['value']
    
    # Footnote: empty (not directly available in JSON API)
    df_bulk['footnote'] = ''
    
    # Reorder columns to match bulk data schema
    column_order = ['adsh', 'tag', 'version', 'ddate', 'qtrs', 'uom', 'segments', 'coreg', 'value', 'footnote']
    df_bulk = df_bulk[column_order]
    
    return df_bulk


def main():
    """Command-line interface for retrieving company XBRL data."""
    parser = argparse.ArgumentParser(
        description="Retrieve XBRL tags and values for a company's 10-Q or 10-K filing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get 10-Q XBRL data for Apple (most recent)
  python get_company_filings_XBRL_API.py --ticker AAPL --form 10-Q

  # Get 10-K XBRL data using CIK
  python get_company_filings_XBRL_API.py --cik 0000320193 --form 10-K

  # Get 10-Q data for a specific date (closest report)
  python get_company_filings_XBRL_API.py --ticker AAPL --form 10-Q --target-date 2024-06-30
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--cik', type=str, help='Company CIK (10-digit)')
    group.add_argument('--ticker', type=str, help='Company ticker symbol')
    
    parser.add_argument('--form', type=str, choices=['10-Q', '10-K'], required=True,
                       help='Form type (10-Q or 10-K)')
    
    parser.add_argument('--target-date', type=str, dest='target_date',
                       help='Target date (YYYY-MM-DD) to find closest report. If not specified, uses most recent report.')
    
    parser.add_argument('--check-tag', type=str, dest='check_tag',
                       help='Debug mode: check if a specific tag exists in XBRL data and show details.')
    
    args = parser.parse_args()
    
    try:
        # Initialize client
        client = SECCompanyFilings(
            cik=args.cik,
            ticker=args.ticker
        )
        
        # Get company info (to ensure we have ticker)
        company_info = client.get_company_info()
        ticker = client.ticker or company_info['ticker']
        
        if not ticker:
            print("⚠️  Warning: Could not determine ticker symbol. Using CIK in filename.")
            ticker = client.cik.lstrip('0') or client.cik
        
        print(f"Fetching {args.form} XBRL data for {company_info['name']} ({ticker})...")
        
        # Get XBRL data (filtered by target_date if provided, otherwise most recent)
        target_date = args.target_date
        df_xbrl = client.get_xbrl_data_dataframe(
            form_type=args.form,
            target_date=target_date
        )
        
        if df_xbrl.empty:
            print(f"❌ No {args.form} XBRL data found for this company.")
            return 1
        
        # Get primary report date (most recent period_end in the filing)
        report_date = df_xbrl['report_date'].max()
        
        if pd.isna(report_date):
            print("❌ Error: Could not determine report date from XBRL data.")
            return 1
        
        # Format report date as YYYYMMDD
        report_date_str = pd.Timestamp(report_date).strftime('%Y%m%d')
        
        # Create base filename: {ticker}_{form_type}_{report_date}
        base_filename = f"{ticker}_{args.form}_{report_date_str}"
        
        # Save XBRL schema data for debugging
        xbrl_filename = f"{base_filename}_xbrl_schema.csv"
        df_xbrl.to_csv(xbrl_filename, index=False)
        print(f"\n💾 Saved {len(df_xbrl):,} XBRL records to {xbrl_filename}")
        print(f"   Unique periods: {df_xbrl['report_date'].nunique()} (dates: {sorted(df_xbrl['report_date'].dropna().unique().astype(str))})")
        
        # Convert to bulk schema
        print("\nConverting XBRL schema to bulk data schema...")
        df_bulk = convert_to_bulk_schema(df_xbrl, taxonomy='us-gaap')
        
        if df_bulk.empty:
            print("❌ Error: Conversion to bulk schema resulted in empty DataFrame.")
            return 1
        
        # Save bulk schema data
        bulk_filename = f"{base_filename}_bulk_schema.csv"
        df_bulk.to_csv(bulk_filename, index=False)
        
        print(f"\n✅ Saved {len(df_bulk):,} records to {bulk_filename} (bulk schema format)")
        print(f"   Primary report date: {pd.Timestamp(report_date).strftime('%Y-%m-%d')}")
        filing_date = df_xbrl['filing_date'].iloc[0]
        print(f"   Filing date: {pd.Timestamp(filing_date).strftime('%Y-%m-%d') if pd.notna(filing_date) else 'N/A'}")
        print(f"   Unique periods in bulk data: {df_bulk['ddate'].nunique()}")
        print(f"   Period dates: {sorted(df_bulk['ddate'].unique())}")
        print(f"   Unique tags: {df_bulk['tag'].nunique()}")
        print(f"\n   Bulk schema columns: {', '.join(df_bulk.columns)}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


# spot check code, for reference only 
from spotcheck_tag_synonyms import find_10q_reports, get_company_ciks, get_cik_ticker_mapping
from utility_data import load_and_join_sec_xbrl_data
from config import DATA_BASE_DIR
import os
def spot_check_xbrl_and_bulk_data_match():
    # pre-requisite: python get_company_filings_XBRL_API.py --ticker MSFT --form 10-Q --target-date 2025-04-30
    # this saves a file called MSFT_10-Q_20250430_bulk_schema.csv
    df_xbrl = pd.read_csv('../MSFT_10-Q_20250331_bulk_schema.csv')

    # load data from SEC directory 
    quarter_path = os.path.join(DATA_BASE_DIR, '2025q2') 
    df_2025q2 = load_and_join_sec_xbrl_data([quarter_path]) 
    cik_to_ticker, ticker_to_cik = get_cik_ticker_mapping()
    df_crt = find_10q_reports(df_2025q2, ticker_to_cik["MSFT"])

    # joining bulk data with xbrl data for comparison 
    df_join = pd.merge(df_crt[df_crt['segments'].isna()], 
                    df_xbrl, 
                    on=['tag', 'ddate', 'qtrs'], suffixes= ['_crt', '_xbrl'])
    print(len(df_join), 'records from bulk data and xbrl data with same tag')
    print(sum(df_join['value_crt']!= df_join["value_xbrl"]), 'values do not match')

    # tags that are in bulk but not in xbrl
    tags_in_crt_not_xbrl = set(df_crt.tag).difference(set(df_xbrl.tag))
    print('tags in crt but not in xbrl:', '\n\t'.join(tags_in_crt_not_xbrl))

    df_no_match = df_crt[df_crt.tag.apply(lambda x: x in tags_in_crt_not_xbrl)] 
    print(len(df_no_match), 'records in crt but not in xbrl')
    print('out of these,', sum(df_no_match['custom_tag']), 'records have customer tags')
    df_no_match = df_no_match[df_no_match['custom_tag']==0]
    print('out of the remaining,', sum(~df_no_match['segments'].isna()), 'records have null segments')
    df_no_match = df_no_match[df_no_match['segments'].isna()]
    print('out of the remaining,', sum(df_no_match['value'].isna()), 'records has null value')
    df_no_match = df_no_match[~df_no_match['value'].isna()]
    print('out of the remaining,', len(df_no_match), 'see no match')

    # tags that are in xbrl but not in bulk are expected, 
    # because SEC only compiles data in the main section and ignores the notes section
