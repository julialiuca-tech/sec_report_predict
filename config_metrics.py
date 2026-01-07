#!/usr/bin/env python3
"""
Configuration file for financial metrics and ratio definitions.

This module centralizes key financial metrics and ratio definitions used across
the SEC data exploration project.
"""

# Key financial metrics for tag matching and analysis
# These are the fundamental financial metrics we track across SEC filings
KEY_METRICS = [
    'AccountsReceivableNetCurrent',
    'Assets',
    'AssetsCurrent',
    'CashAndCashEquivalentsAtCarryingValue',
    'CommonStockSharesOutstanding',  # For per-share metrics
    'CostOfGoodsAndServicesSold',
    # 'DepreciationDepletionAndAmortization',
    'GrossProfit',
    'InterestExpense',
    'InventoryNet',
    'LiabilitiesCurrent',
    'LongTermDebtNoncurrent',
    # 'NetCashProvidedByUsedInOperatingActivities',
    'NetIncomeLoss',
    'OperatingIncomeLoss',
    # 'PaymentsToAcquirePropertyPlantAndEquipment',
    'Revenues',
    'RevenueFromContractWithCustomerExcludingAssessedTax',   
    'StockholdersEquity'
]


# Ratio definitions mapping ratio names to (numerator, denominator) tuples
# Used for calculating and displaying financial ratios from SEC filing data
# Component tags are base tag names without suffixes (suffixes like _1qtrs or _0qtrs
# are added during featurization based on the time period for the metric)
RATIO_DEFINITIONS = {
    'GrossMargin': ('GrossProfit', 'Revenues'),
    'OperatingMargin': ('OperatingIncomeLoss', 'Revenues'),
    'NetMargin': ('NetIncomeLoss', 'Revenues'),
    'ROE': ('NetIncomeLoss', 'StockholdersEquity'),
    'ROA': ('NetIncomeLoss', 'Assets'),
    'CurrentRatio': ('AssetsCurrent', 'LiabilitiesCurrent'),
    'QuickRatio': (('AssetsCurrent', 'InventoryNet'), 'LiabilitiesCurrent'),
    'CashRatio': ('CashAndCashEquivalentsAtCarryingValue', 'LiabilitiesCurrent'),
    'DebtToEquity': (('LongTermDebtNoncurrent', 'LiabilitiesCurrent'), 'StockholdersEquity'),
    'DebtToAssets': (('LongTermDebtNoncurrent', 'LiabilitiesCurrent'), 'Assets'),
    'InterestCoverage': ('OperatingIncomeLoss', 'InterestExpense'),
    'AssetTurnover': ('Revenues', 'Assets'),
    'InventoryTurnover': ('CostOfGoodsAndServicesSold', 'InventoryNet'),
    'ReceivablesTurnover': ('Revenues', 'AccountsReceivableNetCurrent')
}


def main():
    """
    Examine the tags involved in RATIO_DEFINITIONS and report which ones 
    are not in KEY_METRICS already.
    """
    # Extract all tags used in RATIO_DEFINITIONS
    tags_in_ratios = set()
    
    for ratio_name, (num_component, denom_component) in RATIO_DEFINITIONS.items():
        # Handle numerator component (can be a single tag or tuple of tags)
        if isinstance(num_component, tuple):
            for tag in num_component:
                tags_in_ratios.add(tag)
        else:
            tags_in_ratios.add(num_component)
        
        # Handle denominator component (always a single tag)
        tags_in_ratios.add(denom_component)
    
    # Convert KEY_METRICS to a set for efficient lookup
    key_metrics_set = set(KEY_METRICS)
    
    # Find tags in ratios that are not in KEY_METRICS
    missing_tags = sorted(tags_in_ratios - key_metrics_set)
    
    # Print results
    print("=" * 80)
    print("RATIO_DEFINITIONS Tag Analysis")
    print("=" * 80)
    print(f"\nTotal unique tags in RATIO_DEFINITIONS: {len(tags_in_ratios)}")
    print(f"Tags in KEY_METRICS: {len(key_metrics_set)}")
    
    if missing_tags:
        print(f"\n⚠️  Tags in RATIO_DEFINITIONS that are NOT in KEY_METRICS ({len(missing_tags)} tags):")
        for tag in missing_tags:
            # Find which ratios use this tag
            ratios_using_tag = []
            for ratio_name, (num_component, denom_component) in RATIO_DEFINITIONS.items():
                if isinstance(num_component, tuple):
                    if tag in num_component:
                        ratios_using_tag.append(ratio_name)
                elif num_component == tag:
                    ratios_using_tag.append(ratio_name)
                if denom_component == tag:
                    ratios_using_tag.append(ratio_name)
            ratios_str = ', '.join(ratios_using_tag)
            print(f"  - {tag} (used in: {ratios_str})")
    else:
        print("\n✅ All tags in RATIO_DEFINITIONS are present in KEY_METRICS")
    
    # Also show tags that are in KEY_METRICS but not used in ratios (for reference)
    unused_key_metrics = sorted(key_metrics_set - tags_in_ratios)
    if unused_key_metrics:
        print(f"\nℹ️  Tags in KEY_METRICS but not used in RATIO_DEFINITIONS ({len(unused_key_metrics)} tags):")
        for tag in unused_key_metrics:
            print(f"  - {tag}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

