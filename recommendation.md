# Financial Report Analysis Recommendations

## How Analysts and Investors Read Financial Reports

### Primary Financial Statements

1. **Income Statement (P&L)**
   - Revenue/Sales (growth, consistency, quality)
   - Gross margin, operating margin, net margin
   - Operating expenses (SG&A, R&D)
   - Earnings per share (EPS)
   - Non-recurring items

2. **Balance Sheet**
   - Cash and equivalents (especially important)
   - Debt levels (total debt, debt-to-equity)
   - Working capital (current assets vs current liabilities)
   - Inventory turnover (if applicable)
   - Shareholders' equity

3. **Cash Flow Statement**
   - Operating cash flow (quality of earnings indicator)
   - Free cash flow (operating cash flow minus capital expenditures)
   - Capital expenditures
   - Cash flow vs net income (discrepancies are red flags)

### Key Ratios and Metrics Analysts Focus On

**Profitability:**
- ROE (Return on Equity), ROA (Return on Assets), ROIC (Return on Invested Capital)
- Profit margins (gross, operating, net)
- EBITDA margin

**Liquidity:**
- Current ratio, quick ratio
- Cash position relative to debt/market cap

**Efficiency:**
- Asset turnover
- Inventory days, receivables days

**Valuation:**
- P/E, P/B, EV/EBITDA
- PEG ratio (growth-adjusted P/E)

**Growth:**
- Revenue growth, earnings growth, cash flow growth
- Forward guidance and management commentary

### Key Areas of Focus

1. **Trends over time** (quarterly and annual comparisons)
2. **Segment performance** (if multi-business)
3. **Guidance vs actuals** and management commentary
4. **Management discussion (MD&A)** — provides context and outlook
5. **Footnotes** — accounting policies, risks, contingencies
6. **10-K vs 10-Q** — more detail in annual reports

---

## Current Feature Analysis

### What We Currently Have ✅

You're already capturing many important XBRL tags. Your top features include:

- **Assets** (current, total)
- **Liabilities** (current, total)
- **Equity** (StockholdersEquity)
- **Cash and equivalents** (CashAndCashEquivalentsAtCarryingValue)
- **Net Income** (multi-period: 1qtr, 2qtr, 3qtr, 4qtr)
- **Operating Income/Loss** (OperatingIncomeLoss)
- **Earnings Per Share** (basic/diluted)
- **Revenue/Revenues** (Revenues, RevenueFromContractWithCustomerExcludingAssessedTax)
- **Cash Flow** (operating, investing, financing activities)
- **Long-term debt** (LongTermDebtNoncurrent)
- **Various expense items** (SG&A, R&D, Interest, etc.)

### Missing or Underutilized Important Metrics ⚠️

#### Critical Missing Ratios (Analysts Use These Heavily)

**1. Profitability Ratios (not in raw form):**
- Gross Margin = Gross Profit / Revenue
- Operating Margin = Operating Income / Revenue
- Net Margin = Net Income / Revenue
- Return on Assets (ROA) = Net Income / Assets
- Return on Equity (ROE) = Net Income / Stockholders Equity

**2. Liquidity Ratios:**
- Current Ratio = Current Assets / Current Liabilities
- Quick Ratio = (Current Assets - Inventory) / Current Liabilities
- Cash Ratio = Cash / Current Liabilities

**3. Leverage Ratios:**
- Debt-to-Equity = Total Debt / Stockholders Equity
- Debt-to-Assets = Total Debt / Total Assets
- Interest Coverage = Operating Income / Interest Expense

**4. Efficiency Ratios:**
- Asset Turnover = Revenue / Assets
- Inventory Turnover = Cost of Goods Sold / Inventory
- Receivables Turnover = Revenue / Accounts Receivable

**5. Valuation Metrics (if you have market cap data):**
- P/E Ratio = Price / EPS
- P/B Ratio = Price / Book Value per Share

#### Important Tags That May Be Missing or Low-Ranked

From `tag_stats_sorted.csv`, these tags should be prioritized:

- **`RevenueFromContractWithCustomerExcludingAssessedTax`** (rank 61) — more standardized revenue tag
- **`CostOfGoodsAndServicesSold`** (rank 73) — needed for gross margin calculation
- **`DepreciationDepletionAndAmortization`** (rank 89) — needed for EBITDA calculation
- **`ResearchAndDevelopmentExpense`** (rank 56) — important for tech/biotech companies
- **`SellingGeneralAndAdministrativeExpense`** (rank 48) — operating efficiency indicator
- **Free Cash Flow** — may need to calculate: Operating Cash Flow - CapEx
- **`PaymentsToAcquirePropertyPlantAndEquipment`** (already captured) — needed for CapEx

---

## Recommendations for Feature Enhancement

### 1. Create Ratio Features in `featurize.py`

After creating `_current` features, add computed ratio features:

```python
# Profitability Ratios
GrossMargin_1qtrs = GrossProfit_1qtrs_current / Revenues_1qtrs_current
OperatingMargin_1qtrs = OperatingIncomeLoss_1qtrs_current / Revenues_1qtrs_current
NetMargin_1qtrs = NetIncomeLoss_1qtrs_current / Revenues_1qtrs_current
ROE_1qtrs = NetIncomeLoss_1qtrs_current / StockholdersEquity_0qtrs_current
ROA_1qtrs = NetIncomeLoss_1qtrs_current / Assets_0qtrs_current

# Liquidity Ratios
CurrentRatio_0qtrs = AssetsCurrent_0qtrs_current / LiabilitiesCurrent_0qtrs_current
QuickRatio_0qtrs = (AssetsCurrent_0qtrs_current - InventoryNet_0qtrs_current) / LiabilitiesCurrent_0qtrs_current
CashRatio_0qtrs = CashAndCashEquivalentsAtCarryingValue_0qtrs_current / LiabilitiesCurrent_0qtrs_current

# Leverage Ratios
DebtToEquity_0qtrs = (LongTermDebtNoncurrent_0qtrs_current + LiabilitiesCurrent_0qtrs_current) / StockholdersEquity_0qtrs_current
DebtToAssets_0qtrs = (LongTermDebtNoncurrent_0qtrs_current + LiabilitiesCurrent_0qtrs_current) / Assets_0qtrs_current
InterestCoverage_1qtrs = OperatingIncomeLoss_1qtrs_current / InterestExpense_1qtrs_current

# Efficiency Ratios
AssetTurnover_1qtrs = Revenues_1qtrs_current / Assets_0qtrs_current
InventoryTurnover_1qtrs = CostOfGoodsAndServicesSold_1qtrs_current / InventoryNet_0qtrs_current
ReceivablesTurnover_1qtrs = Revenues_1qtrs_current / AccountsReceivableNetCurrent_0qtrs_current
```

#### Ratio Feature Constraints and Validation

When implementing ratio features, it's important to validate that they fall within expected logical bounds. Some ratios represent proportions that cannot exceed 1.0, while others should be non-negative. The following constraints should be applied:

**Ratios that should be capped at ≤ 1.0:**

These ratios represent proportions where the numerator cannot exceed the denominator:

1. **GrossMargin_1qtrs** = GrossProfit / Revenues
   - Gross profit cannot exceed revenue
   - Should be: `0 ≤ GrossMargin ≤ 1`

2. **OperatingMargin_1qtrs** = OperatingIncomeLoss / Revenues  
   - If positive, operating income cannot exceed revenue
   - Can be negative (losses), but if positive, should be ≤ 1
   - Should be: `OperatingMargin ≤ 1` (when positive)

3. **NetMargin_1qtrs** = NetIncomeLoss / Revenues
   - If positive, net income cannot exceed revenue
   - Can be negative (losses), but if positive, should be ≤ 1
   - Should be: `NetMargin ≤ 1` (when positive)

4. **DebtToAssets_0qtrs** = Total Debt / Assets
   - Total debt cannot exceed total assets
   - Should be: `0 ≤ DebtToAssets ≤ 1`

**Ratios that should be non-negative (≥ 0):**

These ratios represent rates or proportions that logically cannot be negative:

1. **GrossMargin_1qtrs** - Already covered above (0 ≤ GrossMargin ≤ 1)

2. **CurrentRatio_0qtrs** = Current Assets / Current Liabilities
   - All components are non-negative
   - Should be: `CurrentRatio ≥ 0`

3. **QuickRatio_0qtrs** = (Current Assets - Inventory) / Current Liabilities
   - All components are non-negative
   - Should be: `QuickRatio ≥ 0`

4. **CashRatio_0qtrs** = Cash / Current Liabilities
   - All components are non-negative
   - Should be: `CashRatio ≥ 0`

5. **DebtToAssets_0qtrs** - Already covered above (0 ≤ DebtToAssets ≤ 1)

6. **AssetTurnover_1qtrs** = Revenues / Assets
   - Both are non-negative
   - Should be: `AssetTurnover ≥ 0`

7. **InventoryTurnover_1qtrs** = COGS / Inventory
   - Both are non-negative
   - Should be: `InventoryTurnover ≥ 0`

8. **ReceivablesTurnover_1qtrs** = Revenues / Accounts Receivable
   - Both are non-negative
   - Should be: `ReceivablesTurnover ≥ 0`

**Ratios that can be negative:**

These ratios can legitimately be negative and should NOT be constrained:

1. **OperatingMargin_1qtrs** - Can be negative (operating losses)

2. **NetMargin_1qtrs** - Can be negative (net losses)

3. **ROE_1qtrs** = Net Income / Equity - Can be negative (losses or negative equity)

4. **ROA_1qtrs** = Net Income / Assets - Can be negative (losses)

5. **DebtToEquity_0qtrs** - Can be negative if equity is negative

6. **InterestCoverage_1qtrs** = Operating Income / Interest Expense - Can be negative (operating losses)

**Summary Table:**

| Ratio | Lower Bound | Upper Bound | Notes |
|-------|-------------|-------------|-------|
| GrossMargin | 0 | 1 | Proportion, cannot exceed 1 |
| OperatingMargin | None | 1 | Can be negative, but if positive ≤ 1 |
| NetMargin | None | 1 | Can be negative, but if positive ≤ 1 |
| ROE | None | None | Can be negative (losses) |
| ROA | None | None | Can be negative (losses) |
| CurrentRatio | 0 | None | Non-negative |
| QuickRatio | 0 | None | Non-negative |
| CashRatio | 0 | None | Non-negative |
| DebtToEquity | None | None | Can be negative (negative equity) |
| DebtToAssets | 0 | 1 | Proportion, cannot exceed 1 |
| InterestCoverage | None | None | Can be negative (losses) |
| AssetTurnover | 0 | None | Non-negative |
| InventoryTurnover | 0 | None | Non-negative |
| ReceivablesTurnover | 0 | None | Non-negative |

**Implementation Note:** The function `flag_outlier_by_ratio_hard_limits()` in `feature_augment.py` implements these validations by adding a boolean column to document bound violations. This allows downstream analysis to identify data quality issues while preserving the original ratio values.

### 2. Prioritize Essential Tags in Top K Selection

When selecting top K tags for featurization, ensure these are included:

- `CostOfGoodsAndServicesSold` — needed for gross margin
- `GrossProfit` — for margin calculations
- `DepreciationDepletionAndAmortization` — for EBITDA and cash flow quality
- `RevenueFromContractWithCustomerExcludingAssessedTax` — more standardized revenue metric

### 3. Add Cash Flow Quality Metrics

- **Operating Cash Flow / Net Income** — cash flow quality indicator (should be close to 1)
- **Free Cash Flow** = Operating CF - CapEx
  - Use: `NetCashProvidedByUsedInOperatingActivities_1qtrs_current - PaymentsToAcquirePropertyPlantAndEquipment_1qtrs_current`
- **Cash Flow Margin** = Operating Cash Flow / Revenue

### 4. Consider Per-Share Metrics

You already have EPS (basic/diluted), but also consider:
- Book Value per Share = Stockholders Equity / Shares Outstanding
- Cash per Share = Cash / Shares Outstanding
- Revenue per Share = Revenue / Shares Outstanding

### 5. Industry-Specific Considerations

Different industries focus on different metrics:

- **Technology:** R&D as % of revenue, free cash flow
- **Retail:** Inventory turnover, same-store sales
- **Banks:** Different metrics entirely (tier 1 capital, loan loss reserves)
- **Capital-intensive (manufacturing, utilities):** Asset turnover, ROIC

---

## Implementation Priorities

### High Priority (Should Implement Soon)

1. **Add ratio features** to `featurize.py`:
   - Profitability ratios (margins, ROE, ROA)
   - Liquidity ratios (current ratio, quick ratio)
   - Leverage ratios (debt-to-equity, debt-to-assets)

2. **Ensure base tags are included** in top K selection:
   - CostOfGoodsAndServicesSold
   - DepreciationDepletionAndAmortization
   - RevenueFromContractWithCustomerExcludingAssessedTax

3. **Add Free Cash Flow calculation** as a feature

### Medium Priority (Nice to Have)

4. Efficiency ratios (asset turnover, inventory turnover)
5. Cash flow quality metrics (OCF/Net Income ratio)
6. Per-share metrics (book value per share, cash per share)

### Low Priority (Future Enhancements)

7. Industry-specific feature sets
8. Forward-looking metrics (if guidance data available)
9. Comparative metrics (vs industry averages)

---

## Notes

- Always handle division by zero when creating ratios (use np.where or fillna)
- Some ratios may have negative denominators (e.g., negative equity) — consider absolute values or indicator flags
- Ratio features should also include `_change` variants (quarter-over-quarter changes) like your existing features
- Consider logarithmic transformations for highly skewed ratio distributions before modeling


# Unique list of tags used in ratio calculations (alphabetically sorted)
TAGS_FOR_RATIO_FEATURES = [
    'AccountsReceivableNetCurrent',
    'Assets',
    'AssetsCurrent',
    'CashAndCashEquivalentsAtCarryingValue',
    'CommonStockSharesOutstanding',  # For per-share metrics
    'CostOfGoodsAndServicesSold',
    'DepreciationDepletionAndAmortization',
    'GrossProfit',
    'InterestExpense',
    'InventoryNet',
    'LiabilitiesCurrent',
    'LongTermDebtNoncurrent',
    'NetCashProvidedByUsedInOperatingActivities',
    'NetIncomeLoss',
    'OperatingIncomeLoss',
    'PaymentsToAcquirePropertyPlantAndEquipment',
    'Revenues',
    'RevenueFromContractWithCustomerExcludingAssessedTax',  # Alternative revenue tag
    'StockholdersEquity',
]
  