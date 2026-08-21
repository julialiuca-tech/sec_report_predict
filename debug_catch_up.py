#!/usr/bin/env python3
"""
Debug: compare catch-up API featurization vs quarterly-batch featurization.

Hypothesis
----------
The same SEC report may appear in both:
  - ``data/featurized_since_2011/featurized_all_quarters.csv`` (batch / featurize.py)
  - ``data/SEC_catchup_query/featurized_all_catchup.csv`` (API / catch_up_sec_reports.py)

For overlapping reports, ``*_current`` features should be nearly identical.
``period`` may differ by a few days (companyfacts vs bulk month-end), within
about 90 days.

What this script does
---------------------
1. Builds a ticker/date event list with ``as_of`` late enough that recent
   filings exist in the batch featurized file.
2. Calls ``catch_up_from_events(..., skip_bulk=False)`` so batch filings are
   re-downloaded via the live API (limited to a few recent filings per CIK).
3. Re-runs ``featurize_df_catchup()``.
4. Matches catch-up rows to batch rows on (cik, form) with |period| <= 90 days
   and compares shared ``*_current`` feature values.

Examples
--------
# All default tickers vs a single quarter batch file:
python debug_catch_up.py \\
  --batch-file data/featurized_since_2011/2026q1_featurized.csv

# Subset of tickers, custom as-of (needed for late filers like WMT):
python debug_catch_up.py \\
  --batch-file data/featurized_since_2011/2026q1_featurized.csv \\
  --tickers WMT AAPL TSLA SHOP AMZN \\
  --as-of 2026-03-20

# Skip SEC re-download; only re-featurize catchup + compare:
python debug_catch_up.py \\
  --batch-file data/featurized_since_2011/2026q1_featurized.csv \\
  --skip-download
"""

from __future__ import annotations

import argparse
from datetime import date, datetime

import numpy as np
import pandas as pd

from catch_up_sec_reports import (
    catch_up_from_events,
    featurize_df_catchup,
    pad_cik,
)
from config import FEATURIZED_ALL_QUARTERS_FILE, FEATURIZED_CATCHUP_FILE, QUARTER_FEATURIZED_PATTERN
from utility_data import get_cik_ticker_mapping

# Default tickers exercised in batch-vs-catchup checks.
DEBUG_TICKERS = [
    "AAPL", "MSFT", "CAT", "META",
    "IBM", "INTU", "INTC", "AMD",
    "WMT", "TSLA", "SHOP", "AMZN", "NBIS",
]

# Late enough to include WMT FY2026 10-K (filed 2026-03-13) and similar.
DEBUG_AS_OF = date(2026, 3, 20)

# Re-download at most this many most-recent filings per CIK on/before as_of.
MAX_FILINGS_PER_CIK = 2

# Max |period_bulk - period_catchup| in days to treat as the same report.
PERIOD_MATCH_TOLERANCE_DAYS = 90

# Relative tolerance for numeric feature equality.
VALUE_RTOL = 1e-5
VALUE_ATOL = 1.0  # absolute slack for large USD magnitudes / float noise


def _parse_as_of(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def build_debug_events(
    tickers=DEBUG_TICKERS,
    as_of: date = DEBUG_AS_OF,
) -> pd.DataFrame:
    _, ticker_to_cik = get_cik_ticker_mapping()
    rows = []
    missing = []
    for ticker in tickers:
        ticker_u = ticker.upper()
        cik = ticker_to_cik.get(ticker_u)
        if cik is None:
            missing.append(ticker_u)
            continue
        rows.append({"ticker": ticker_u, "cik": pad_cik(cik), "date": as_of})
    if missing:
        print(f"⚠️  Skipping tickers with no CIK mapping: {missing}")
    return pd.DataFrame(rows)


def _period_to_timestamp(period_series: pd.Series) -> pd.Series:
    return pd.to_datetime(period_series.astype(str), format="%Y%m%d", errors="coerce")


def match_overlapping_reports(
    df_batch: pd.DataFrame,
    df_catchup: pd.DataFrame,
    tickers_ciks: set[int],
    tol_days: int = PERIOD_MATCH_TOLERANCE_DAYS,
) -> pd.DataFrame:
    """
    Pair catch-up rows to batch rows on (cik, form) with closest period
    within ``tol_days``.
    """
    batch = df_batch[df_batch["cik"].isin(tickers_ciks)].copy()
    catch = df_catchup[df_catchup["cik"].isin(tickers_ciks)].copy()
    if batch.empty or catch.empty:
        return pd.DataFrame()

    batch["_period_ts"] = _period_to_timestamp(batch["period"])
    catch["_period_ts"] = _period_to_timestamp(catch["period"])
    batch = batch.dropna(subset=["_period_ts"])
    catch = catch.dropna(subset=["_period_ts"])

    pairs = []
    for _, crow in catch.iterrows():
        candidates = batch[
            (batch["cik"] == crow["cik"]) & (batch["form"] == crow["form"])
        ].copy()
        if candidates.empty:
            continue
        candidates["_delta_days"] = (
            candidates["_period_ts"] - crow["_period_ts"]
        ).abs().dt.days
        candidates = candidates[candidates["_delta_days"] <= tol_days]
        if candidates.empty:
            continue
        best = candidates.sort_values("_delta_days").iloc[0]
        pairs.append(
            {
                "cik": int(crow["cik"]),
                "form": crow["form"],
                "period_catchup": int(crow["period"]),
                "period_batch": int(best["period"]),
                "period_delta_days": int(best["_delta_days"]),
                "catchup_idx": crow.name,
                "batch_idx": best.name,
            }
        )
    return pd.DataFrame(pairs)


def compare_feature_values(
    df_batch: pd.DataFrame,
    df_catchup: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Compare shared ``*_current`` columns for each matched report pair."""
    batch_feats = [c for c in df_batch.columns if c.endswith("_current")]
    catch_feats = [c for c in df_catchup.columns if c.endswith("_current")]
    shared = sorted(set(batch_feats) & set(catch_feats))
    only_batch = sorted(set(batch_feats) - set(catch_feats))
    only_catch = sorted(set(catch_feats) - set(batch_feats))

    print(f"\nShared *_current features: {len(shared)}")
    print(f"Only in batch:   {len(only_batch)}")
    print(f"Only in catchup: {len(only_catch)}")

    rows = []
    for _, pair in pairs.iterrows():
        b = df_batch.loc[pair["batch_idx"], shared]
        c = df_catchup.loc[pair["catchup_idx"], shared]
        b_num = pd.to_numeric(b, errors="coerce")
        c_num = pd.to_numeric(c, errors="coerce")

        both_nan = b_num.isna() & c_num.isna()
        one_nan = b_num.isna() ^ c_num.isna()
        both_present = ~(b_num.isna() | c_num.isna())
        close = np.isclose(
            b_num[both_present].astype(float),
            c_num[both_present].astype(float),
            rtol=VALUE_RTOL,
            atol=VALUE_ATOL,
            equal_nan=False,
        )
        n_both = int(both_present.sum())
        n_close = int(close.sum()) if n_both else 0
        n_disagree = n_both - n_close
        n_both_nan = int(both_nan.sum())
        n_one_nan = int(one_nan.sum())

        disagree_cols = []
        if n_both and n_disagree:
            disagree_cols = list(b_num[both_present].index[~close][:10])

        rows.append(
            {
                "cik": pair["cik"],
                "form": pair["form"],
                "period_catchup": pair["period_catchup"],
                "period_batch": pair["period_batch"],
                "period_delta_days": pair["period_delta_days"],
                "n_shared_features": len(shared),
                "n_both_present": n_both,
                "n_value_close": n_close,
                "n_disagree": n_disagree,
                "match_rate_present": (n_close / n_both) if n_both else np.nan,
                "n_both_nan": n_both_nan,
                "n_one_nan": n_one_nan,
                "sample_disagreements": ",".join(disagree_cols),
            }
        )
    return pd.DataFrame(rows)


def main(
    batch_file: str = FEATURIZED_ALL_QUARTERS_FILE,
    tickers: list[str] | None = None,
    as_of: date = DEBUG_AS_OF,
    max_filings_per_cik: int = MAX_FILINGS_PER_CIK,
    skip_download: bool = False,
    skip_featurize: bool = False,
) -> None:
    tickers = [t.upper() for t in (tickers or list(DEBUG_TICKERS))]
    print("=" * 70)
    print("DEBUG: catch-up vs batch featurization")
    print("=" * 70)
    print(f"as_of={as_of}, tickers={tickers}")
    print(f"period match tolerance={PERIOD_MATCH_TOLERANCE_DAYS} days")
    print(f"max_filings_per_cik={max_filings_per_cik}")
    print(f"batch file={batch_file}")

    events = build_debug_events(tickers=tickers, as_of=as_of)
    if events.empty:
        print("No debug events to process.")
        return
    print("\nDebug events:")
    print(events.to_string(index=False))

    cik_to_ticker = {
        int(pad_cik(r.cik)): r.ticker for r in events.itertuples()
    }

    if not skip_download:
        print("\n--- Step 1: catch_up_from_events(skip_bulk=False) ---")
        catch_up_from_events(
            events,
            skip_bulk=False,
            skip_catchup=False,  # refresh even if previously caught up
            max_filings_per_cik=max_filings_per_cik,
        )
    else:
        print("\n--- Step 1: skipped (--skip-download) ---")

    if not skip_featurize:
        print("\n--- Step 2: featurize_df_catchup() ---")
        df_catchup = featurize_df_catchup()
    else:
        print("\n--- Step 2: load existing catchup featurized file ---")
        df_catchup = pd.read_csv(FEATURIZED_CATCHUP_FILE, low_memory=False)
        print(f"Loaded {FEATURIZED_CATCHUP_FILE}: {df_catchup.shape}")

    if df_catchup.empty:
        print("Catch-up featurization produced no rows.")
        return

    print("\n--- Step 3: load batch featurized file ---")
    df_batch = pd.read_csv(batch_file, low_memory=False)
    # Normalize cik dtype for matching.
    df_batch["cik"] = df_batch["cik"].map(lambda x: int(pad_cik(x)))
    df_catchup["cik"] = df_catchup["cik"].map(lambda x: int(pad_cik(x)))
    print(f"Batch featurized: {df_batch.shape}")
    print(f"Catchup featurized: {df_catchup.shape}")

    ciks = set(cik_to_ticker.keys())
    df_catchup_dbg = df_catchup[df_catchup["cik"].isin(ciks)].copy()
    print(f"Catchup rows for debug CIKs: {len(df_catchup_dbg)}")

    tickers_with_catchup = {
        cik_to_ticker[c] for c in df_catchup_dbg["cik"].unique() if c in cik_to_ticker
    }
    tickers_no_catchup = sorted(set(tickers) - tickers_with_catchup)
    if tickers_no_catchup:
        print(f"⚠️  No catchup rows for: {tickers_no_catchup}")

    print("\n--- Step 4: match overlapping reports ---")
    pairs = match_overlapping_reports(df_batch, df_catchup_dbg, ciks)
    if pairs.empty:
        print("No overlapping (cik, form, period±90d) pairs found.")
        batch_present = sorted(
            {
                cik_to_ticker[c]
                for c in df_batch.loc[df_batch["cik"].isin(ciks), "cik"].unique()
                if c in cik_to_ticker
            }
        )
        print(f"Tickers present in batch file: {batch_present or 'none'}")
        return

    pairs.insert(0, "ticker", pairs["cik"].map(cik_to_ticker))
    # Keep one best pair per (cik, form): closest period, then most recent catchup.
    pairs = (
        pairs.sort_values(
            ["cik", "form", "period_delta_days", "period_catchup"],
            ascending=[True, True, True, False],
        )
        .drop_duplicates(subset=["cik", "form"], keep="first")
        .reset_index(drop=True)
    )

    print(f"Matched pairs: {len(pairs)}")
    print(
        pairs[
            ["ticker", "cik", "form", "period_catchup", "period_batch", "period_delta_days"]
        ].to_string(index=False)
    )

    matched_tickers = set(pairs["ticker"])
    unmatched = sorted(set(tickers) - matched_tickers)
    if unmatched:
        print(f"⚠️  No batch overlap for: {unmatched}")

    print("\n--- Step 5: compare *_current features ---")
    comparison = compare_feature_values(df_batch, df_catchup_dbg, pairs)
    comparison.insert(0, "ticker", comparison["cik"].map(cik_to_ticker))
    display_cols = [
        "ticker", "form", "period_catchup", "period_batch", "period_delta_days",
        "n_both_present", "n_value_close", "n_disagree", "match_rate_present",
        "n_one_nan", "sample_disagreements",
    ]
    print(comparison[display_cols].to_string(index=False))

    if len(comparison):
        present = int(comparison["n_both_present"].sum())
        close = int(comparison["n_value_close"].sum())
        disagree = int(comparison["n_disagree"].sum())
        overall = close / present if present else np.nan
        print("\n" + "=" * 70)
        print(
            f"OVERALL: {close:,} / {present:,} present feature values match "
            f"({overall:.2%}); disagreements={disagree:,} "
            f"(rtol={VALUE_RTOL}, atol={VALUE_ATOL})"
        )
        print(
            f"Period deltas (days): "
            f"min={comparison['period_delta_days'].min()}, "
            f"median={comparison['period_delta_days'].median()}, "
            f"max={comparison['period_delta_days'].max()}"
        )
        if unmatched:
            print(f"Unmatched tickers: {unmatched}")
        print(f"Catchup file: {FEATURIZED_CATCHUP_FILE}")
        print(f"Batch file:   {batch_file}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--batch-file",
        default=FEATURIZED_ALL_QUARTERS_FILE,
        help=(
            "Batch featurized CSV to compare against "
            f"(default: {FEATURIZED_ALL_QUARTERS_FILE}; "
            f"quarter files look like {QUARTER_FEATURIZED_PATTERN.format('2026q1')})"
        ),
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help=f"Tickers to compare (default: {DEBUG_TICKERS})",
    )
    parser.add_argument(
        "--as-of",
        type=_parse_as_of,
        default=DEBUG_AS_OF,
        help=f"Only consider filings accepted on/before this date YYYY-MM-DD (default: {DEBUG_AS_OF})",
    )
    parser.add_argument(
        "--max-filings-per-cik",
        type=int,
        default=MAX_FILINGS_PER_CIK,
        help=f"Most-recent filings to (re)download per CIK (default: {MAX_FILINGS_PER_CIK})",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip SEC API re-download; use filings already under SEC_catchup_query/",
    )
    parser.add_argument(
        "--skip-featurize",
        action="store_true",
        help="Skip catchup featurize; load existing featurized_all_catchup.csv",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        batch_file=args.batch_file,
        tickers=args.tickers,
        as_of=args.as_of,
        max_filings_per_cik=args.max_filings_per_cik,
        skip_download=args.skip_download,
        skip_featurize=args.skip_featurize,
    )
