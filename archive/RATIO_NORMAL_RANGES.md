# Normal Ranges for Financial Ratios

This document provides typical/normal ranges for the financial ratios computed in `feature_augment.py`. These ranges are general guidelines and can vary significantly by industry, company size, and economic conditions.

## Profitability Ratios

### GrossMargin_1qtrs
- **Formula:** GrossProfit / Revenues
- **Typical Range:** 20-50% for most industries
- **Industry Examples:**
  - Retail: 30-40%
  - Technology: 50-70%
  - Manufacturing: 20-35%
- **Interpretation:** Higher is generally better. Indicates pricing power and production efficiency.

### OperatingMargin_1qtrs
- **Formula:** OperatingIncomeLoss / Revenues
- **Typical Range:** 5-15% for healthy companies
- **Good Range:** 10-15% is considered good for many industries
- **Note:** Can be negative (operating losses)
- **Interpretation:** Measures operational efficiency after operating expenses.

### NetMargin_1qtrs
- **Formula:** NetIncomeLoss / Revenues
- **Typical Range:** 5-15% for healthy companies
- **Strong Range:** 10% or higher is considered strong
- **Note:** Can be negative (net losses)
- **Interpretation:** Shows overall profitability after all expenses and taxes.

### ROE_1qtrs (Return on Equity)
- **Formula:** NetIncomeLoss / StockholdersEquity
- **Typical Range:** 10-15% is acceptable
- **Strong Range:** 15-20% is considered strong
- **Note:** Can be negative (losses or negative equity)
- **Interpretation:** Measures return generated on shareholders' equity investment.

### ROA_1qtrs (Return on Assets)
- **Formula:** NetIncomeLoss / Assets
- **Typical Range:** 5% or higher is generally considered good
- **Note:** Can be negative (losses)
- **Interpretation:** Measures how effectively a company uses its assets to generate profit.

## Liquidity Ratios

### CurrentRatio_0qtrs
- **Formula:** Current Assets / Current Liabilities
- **Typical Range:** 1.5 to 3.0
- **Red Flags:**
  - Below 1.0: Indicates potential liquidity issues
  - Above 3.0: May suggest inefficient use of assets
- **Interpretation:** Measures ability to cover short-term obligations with short-term assets.

### QuickRatio_0qtrs (Acid-Test Ratio)
- **Formula:** (Current Assets - Inventory) / Current Liabilities
- **Typical Range:** 1.0 to 2.0
- **Red Flags:**
  - Below 1.0: Might indicate potential liquidity problems
  - Below 0.5: May indicate risk of running out of working capital
- **Interpretation:** Assesses ability to meet short-term obligations without relying on inventory.

### CashRatio_0qtrs
- **Formula:** Cash / Current Liabilities
- **Typical Range:** 0.2 to 0.5 for healthy companies
- **Interpretation:** Most conservative liquidity measure, showing cash available to cover short-term liabilities.
- **Note:** Lower than quick ratio since it excludes receivables and inventory.

## Leverage Ratios

### DebtToEquity_0qtrs
- **Formula:** (LongTermDebt + CurrentLiabilities) / StockholdersEquity
- **Typical Range:** Below 2.0 is generally preferred
- **Industry Variation:** Capital-intensive industries may have higher acceptable ratios
- **Note:** Can be negative if equity is negative
- **Interpretation:** Evaluates proportion of debt financing relative to equity financing.

### DebtToAssets_0qtrs
- **Formula:** (LongTermDebt + CurrentLiabilities) / Assets
- **Typical Range:** 0.3 to 0.6 for healthy companies
- **Boundary:** Should be between 0 and 1 (debt cannot exceed assets)
- **Interpretation:** Shows percentage of assets financed by debt. Higher ratios indicate greater financial risk.

### InterestCoverage_1qtrs
- **Formula:** OperatingIncomeLoss / InterestExpense
- **Typical Range:** 2.0 or higher indicates good coverage
- **Red Flags:**
  - Below 1.0: Indicates inability to cover interest payments
- **Note:** Can be negative (operating losses)
- **Interpretation:** Assesses company's ability to meet interest payment obligations.

## Efficiency Ratios

### AssetTurnover_1qtrs
- **Formula:** Revenues / Assets
- **Typical Range:**
  - **Strong:** Greater than 0.45 (45%)
  - **Caution:** 0.30-0.45 (30-45%)
- **Industry Variation:** Varies significantly by industry
- **Interpretation:** Indicates how efficiently a company uses its assets to generate sales.

### InventoryTurnover_1qtrs
- **Formula:** CostOfGoodsSold / Inventory
- **Typical Range:** Varies significantly by industry
- **Industry Examples:**
  - Retail: 4-6 times per year
  - Manufacturing: 6-12 times per year
- **Interpretation:** Higher ratios generally indicate efficient inventory management and faster inventory turnover.

### ReceivablesTurnover_1qtrs
- **Formula:** Revenues / AccountsReceivable
- **Typical Range:** 10-12 times per year (approximately 30-36 days collection period)
- **Interpretation:** Higher ratios indicate faster collection of receivables and better cash flow management.
- **Industry Variation:** Varies by industry and payment terms

## Important Notes

1. **Industry-Specific Benchmarks:** These ranges are general guidelines. Industry-specific benchmarks should be consulted for more accurate assessments.

2. **Context Matters:** Financial ratios should be analyzed in context:
   - Compare to industry peers
   - Consider company size and lifecycle stage
   - Evaluate trends over time
   - Consider economic conditions

3. **Combined Analysis:** No single ratio provides a complete picture. Use multiple ratios together for comprehensive financial analysis.

4. **Boundary Violations:** The `flag_outlier_by_ratio_hard_limits()` function in `feature_augment.py` validates logical bounds (e.g., ratios that should be ≤ 1.0 or ≥ 0.0). Values outside typical ranges may still be valid but warrant investigation.

## References

These ranges are based on general financial analysis literature and industry standards. For specific industry benchmarks, consult:
- Industry-specific financial databases
- SEC industry guides
- Financial analysis handbooks (e.g., CFA Institute materials)
- Industry trade associations

