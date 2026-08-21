#!/usr/bin/env python3
"""
Catch up on SEC 10-K / 10-Q filings newer than the quarterly bulk dumps.

Reads an events CSV with columns ``ticker`` and ``date`` (other columns ignored),
maps each ticker to a CIK, and downloads filings with ``filed <= date`` that are
not already present under ``data/SEC_raw_since_2011/`` or
``data/SEC_catchup_query/``.

Output layout: ``data/SEC_catchup_query/{adsh}/sub.txt`` + ``num.txt``
(bulk-compatible fact schema).

SEC APIs (rate-limited to 10 req/sec; descriptive User-Agent required):
- ``https://data.sec.gov/submissions/CIK##########.json``
- ``https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json``
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime
from typing import Optional, Sequence

import pandas as pd
import requests

from config import (
    CATCHUP_EVENTS_FILE,
    DATA_BASE_DIR,
    DEFAULT_K_TOP_TAGS,
    FEATURIZED_CATCHUP_FILE,
    SEC_CATCHUP_DIR,
    SEC_CATCHUP_INDEX_FILE,
    SEC_REQUEST_INTERVAL_SEC,
    SEC_USER_AGENT,
)
from featurize import organize_feature_dataframe
from get_company_filings_XBRL_API import convert_to_bulk_schema
from utility_data import get_cik_ticker_mapping, read_tags_to_featurize


SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
DEFAULT_FORMS = ("10-K", "10-Q")


class RateLimitedSession:
    """``requests`` wrapper that paces calls under the SEC fair-access limit."""

    def __init__(
        self,
        user_agent: str = SEC_USER_AGENT,
        min_interval_sec: float = SEC_REQUEST_INTERVAL_SEC,
    ):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )
        self.min_interval_sec = min_interval_sec
        self._last_request_at = 0.0

    def get(self, url: str, timeout: int = 60) -> requests.Response:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval_sec:
            time.sleep(self.min_interval_sec - elapsed)
        response = self.session.get(url, timeout=timeout)
        self._last_request_at = time.monotonic()
        response.raise_for_status()
        return response


def pad_cik(cik) -> str:
    return str(int(str(cik).strip())).zfill(10)


def load_ticker_date_events(events_file: str) -> pd.DataFrame:
    """
    Load an events CSV and attach CIKs.

    Expects columns ``ticker`` and ``date`` (any other columns are ignored).
    Returns columns: ticker, date (datetime.date), cik (10-digit str).
    """
    df = pd.read_csv(events_file, usecols=["ticker", "date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["ticker", "date"]).drop_duplicates()

    _, ticker_to_cik = get_cik_ticker_mapping()
    df["cik"] = df["ticker"].map(ticker_to_cik)

    missing = df[df["cik"].isna()]["ticker"].unique().tolist()
    if missing:
        print(f"⚠️  No CIK mapping for {len(missing)} ticker(s): {missing}")
        df = df.dropna(subset=["cik"]).copy()

    df["cik"] = df["cik"].map(pad_cik)

    # One as-of date per CIK: keep the latest event date if duplicates appear.
    df = (
        df.sort_values("date")
        .groupby(["cik", "ticker"], as_index=False)["date"]
        .max()
    )
    return df.reset_index(drop=True)


def collect_known_adsh(
    bulk_dir: str = DATA_BASE_DIR,
    catchup_dir: str = SEC_CATCHUP_DIR,
) -> tuple[set[str], set[str]]:
    """
    Accession numbers already present locally.

    Returns
    -------
    bulk_adsh : set[str]
        Accessions from quarterly batch dumps under ``bulk_dir``.
    catchup_adsh : set[str]
        Accessions from prior catch-up downloads under ``catchup_dir``.
    """
    bulk_adsh: set[str] = set()
    catchup_adsh: set[str] = set()

    if os.path.isdir(bulk_dir):
        for name in os.listdir(bulk_dir):
            sub_path = os.path.join(bulk_dir, name, "sub.txt")
            if not os.path.isfile(sub_path):
                continue
            try:
                sub = pd.read_csv(
                    sub_path,
                    sep="\t",
                    usecols=["adsh"],
                    dtype=str,
                    low_memory=False,
                )
                bulk_adsh.update(sub["adsh"].dropna().astype(str))
            except Exception as exc:
                print(f"⚠️  Skipping bulk sub.txt {sub_path}: {exc}")

    if os.path.isdir(catchup_dir):
        for name in os.listdir(catchup_dir):
            filing_dir = os.path.join(catchup_dir, name)
            if not os.path.isdir(filing_dir):
                continue
            if os.path.isfile(os.path.join(filing_dir, "num.txt")):
                catchup_adsh.add(name)
            sub_path = os.path.join(filing_dir, "sub.txt")
            if os.path.isfile(sub_path):
                try:
                    sub = pd.read_csv(sub_path, sep="\t", usecols=["adsh"], dtype=str)
                    catchup_adsh.update(sub["adsh"].dropna().astype(str))
                except Exception:
                    pass

    return bulk_adsh, catchup_adsh


def parse_submissions_filings(
    submissions_json: dict,
    forms: Sequence[str] = DEFAULT_FORMS,
    as_of_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Flatten ``filings.recent`` for the requested form types.

    If ``as_of_date`` is set, keep only filings with ``filingDate <= as_of_date``.
    """
    recent = submissions_json.get("filings", {}).get("recent", {})
    empty_cols = [
        "adsh",
        "form",
        "filed",
        "period",
        "accepted",
        "primary_document",
        "is_xbrl",
    ]
    if not recent or "accessionNumber" not in recent:
        return pd.DataFrame(columns=empty_cols)

    n = len(recent["accessionNumber"])
    rows = []
    for i in range(n):
        form = recent.get("form", [""] * n)[i]
        if form not in forms:
            continue
        filed_str = recent.get("filingDate", [None] * n)[i]
        if as_of_date and filed_str:
            try:
                filed_d = datetime.strptime(filed_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            if filed_d > as_of_date:
                continue
        rows.append(
            {
                "adsh": recent["accessionNumber"][i],
                "form": form,
                "filed": filed_str,
                "period": recent.get("reportDate", [None] * n)[i],
                "accepted": recent.get("acceptanceDateTime", [None] * n)[i],
                "primary_document": recent.get("primaryDocument", [None] * n)[i],
                "is_xbrl": recent.get("isXBRL", [None] * n)[i],
            }
        )
    return pd.DataFrame(rows)


def facts_for_accession(
    company_facts: dict,
    adsh: str,
    taxonomies: Sequence[str] = ("us-gaap", "ifrs-full", "dei"),
) -> pd.DataFrame:
    """Extract XBRL facts belonging to one accession number (JSON API schema)."""
    records = []
    facts_root = company_facts.get("facts", {})
    for taxonomy in taxonomies:
        taxonomy_facts = facts_root.get(taxonomy, {})
        for tag, concept_data in taxonomy_facts.items():
            for unit, facts_list in concept_data.get("units", {}).items():
                for fact in facts_list:
                    if fact.get("accn") != adsh:
                        continue
                    records.append(
                        {
                            "tag": tag,
                            "taxonomy": taxonomy,
                            "unit": unit,
                            "value": fact.get("val"),
                            "form": fact.get("form"),
                            "filing_date": fact.get("filed"),
                            "period_start": fact.get("start"),
                            "period_end": fact.get("end"),
                            "report_date": fact.get("end"),
                            "frame": fact.get("frame"),
                            "accn": fact.get("accn"),
                        }
                    )
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    for col in ("filing_date", "period_start", "period_end", "report_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def build_sub_row(
    cik: str,
    company_name: str,
    sic,
    filing: dict,
) -> pd.DataFrame:
    """One-row SUB-like table for a catch-up filing."""
    period = filing.get("period") or ""
    filed = filing.get("filed") or ""
    return pd.DataFrame(
        [
            {
                "adsh": filing["adsh"],
                "cik": int(cik),
                "name": company_name,
                "sic": sic if sic is not None else "",
                "fye": "",
                "fy": "",
                "fp": "",
                "form": filing["form"],
                "period": period.replace("-", "") if period else "",
                "filed": filed.replace("-", "") if filed else "",
                "accepted": filing.get("accepted") or "",
            }
        ]
    )


def write_filing_bundle(
    catchup_dir: str,
    sub_df: pd.DataFrame,
    num_df: pd.DataFrame,
) -> str:
    adsh = str(sub_df["adsh"].iloc[0])
    out_dir = os.path.join(catchup_dir, adsh)
    os.makedirs(out_dir, exist_ok=True)
    sub_df.to_csv(os.path.join(out_dir, "sub.txt"), sep="\t", index=False)
    num_df.to_csv(os.path.join(out_dir, "num.txt"), sep="\t", index=False)
    return out_dir


def append_catchup_index(rows: list[dict], index_path: str = SEC_CATCHUP_INDEX_FILE) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    new_df = pd.DataFrame(rows)
    if os.path.isfile(index_path):
        old_df = pd.read_csv(index_path, dtype=str)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["adsh"], keep="last")
    else:
        combined = new_df
    combined.to_csv(index_path, index=False)


def catch_up_from_events(
    events: pd.DataFrame,
    forms: Sequence[str] = DEFAULT_FORMS,
    http: Optional[RateLimitedSession] = None,
    skip_bulk: bool = True,
    skip_catchup: bool = True,
    max_filings_per_cik: Optional[int] = None,
) -> pd.DataFrame:
    """
    Download missing SEC filings for each (ticker/cik, as-of date) event.

    ``events`` must contain columns ``cik``, ``ticker``, ``date``.

    Parameters
    ----------
    skip_bulk : bool
        If True (default), skip accession numbers already in the quarterly batch.
        Set False for debug re-downloads of batch filings via the live API.
    skip_catchup : bool
        If True (default), skip accession numbers already under ``SEC_CATCHUP_DIR``.
    max_filings_per_cik : int, optional
        If set, only download the N most recent candidate filings per CIK
        (by ``filed`` date). Useful for debug runs.
    """
    required = {"cik", "ticker", "date"}
    missing_cols = required - set(events.columns)
    if missing_cols:
        raise ValueError(f"events missing columns: {sorted(missing_cols)}")

    http = http or RateLimitedSession()
    os.makedirs(SEC_CATCHUP_DIR, exist_ok=True)

    print("Indexing accession numbers already in bulk / catch-up data...")
    bulk_adsh, catchup_adsh = collect_known_adsh()
    print(
        f"  Known accession numbers: batch={len(bulk_adsh):,}, catchup={len(catchup_adsh):,}"
    )
    print(
        f"Looking for {', '.join(forms)} with filed <= each event date "
        f"(skip_bulk={skip_bulk}, skip_catchup={skip_catchup}"
        f"{f', max_filings_per_cik={max_filings_per_cik}' if max_filings_per_cik else ''})"
    )

    summary_rows = []
    index_rows = []
    downloaded = skipped_bulk = skipped_catchup = skipped_no_xbrl = failed = 0

    events = events.copy()
    events["cik"] = events["cik"].map(pad_cik)

    for i, row in enumerate(events.itertuples(index=False), 1):
        cik = row.cik
        ticker = row.ticker
        as_of = row.date
        print(f"\n[{i}/{len(events)}] {ticker} (CIK {cik}) as_of {as_of}")

        try:
            submissions = http.get(SUBMISSIONS_URL.format(cik=cik)).json()
        except requests.RequestException as exc:
            print(f"  ✗ submissions fetch failed: {exc}")
            failed += 1
            continue

        company_name = submissions.get("name", "")
        sic = submissions.get("sic")
        filings = parse_submissions_filings(
            submissions, forms=forms, as_of_date=as_of
        )
        if filings.empty:
            print("  No matching filings on/before as-of date.")
            continue

        in_bulk = filings["adsh"].isin(bulk_adsh)
        in_catchup = filings["adsh"].isin(catchup_adsh)

        skip_mask = pd.Series(False, index=filings.index)
        if skip_bulk:
            skip_mask |= in_bulk
        if skip_catchup:
            skip_mask |= in_catchup

        new_filings = filings[~skip_mask].copy()
        if max_filings_per_cik is not None and not new_filings.empty:
            new_filings = (
                new_filings.assign(
                    _filed=pd.to_datetime(new_filings["filed"], errors="coerce")
                )
                .sort_values("_filed", ascending=False)
                .head(max_filings_per_cik)
                .drop(columns="_filed")
            )

        n_bulk = int(in_bulk.sum())
        n_catchup = int(in_catchup.sum())
        skipped_bulk += int((skip_mask & in_bulk).sum())
        skipped_catchup += int((skip_mask & in_catchup & ~in_bulk).sum())

        print(
            f"  Found {len(filings)} filing(s) on/before {as_of}; "
            f"{n_bulk} in batch, {n_catchup} in catchup; "
            f"{len(new_filings)} to download."
        )
        if new_filings.empty:
            for _, filing in filings.iterrows():
                if filing["adsh"] in bulk_adsh:
                    action = "already_in_batch"
                elif filing["adsh"] in catchup_adsh:
                    action = "already_in_catchup"
                else:
                    action = "skipped"
                summary_rows.append(
                    {
                        **filing.to_dict(),
                        "cik": cik,
                        "ticker": ticker,
                        "as_of_date": as_of,
                        "action": action,
                    }
                )
            continue

        try:
            company_facts = http.get(
                COMPANYFACTS_URL.format(cik=cik), timeout=120
            ).json()
        except requests.RequestException as exc:
            print(f"  ✗ companyfacts fetch failed: {exc}")
            failed += len(new_filings)
            for _, filing in new_filings.iterrows():
                summary_rows.append(
                    {
                        **filing.to_dict(),
                        "cik": cik,
                        "ticker": ticker,
                        "as_of_date": as_of,
                        "action": "facts_failed",
                    }
                )
            continue

        for _, filing in new_filings.iterrows():
            adsh = filing["adsh"]
            df_xbrl = facts_for_accession(company_facts, adsh)
            if df_xbrl.empty:
                skipped_no_xbrl += 1
                print(f"  ⚠️  No XBRL facts for {adsh}; skipping.")
                summary_rows.append(
                    {
                        **filing.to_dict(),
                        "cik": cik,
                        "ticker": ticker,
                        "as_of_date": as_of,
                        "action": "no_xbrl_facts",
                    }
                )
                continue

            num_parts = [
                convert_to_bulk_schema(part, taxonomy=taxonomy)
                for taxonomy, part in df_xbrl.groupby("taxonomy")
            ]
            num_df = (
                pd.concat(num_parts, ignore_index=True) if num_parts else pd.DataFrame()
            )
            if num_df.empty:
                skipped_no_xbrl += 1
                summary_rows.append(
                    {
                        **filing.to_dict(),
                        "cik": cik,
                        "ticker": ticker,
                        "as_of_date": as_of,
                        "action": "empty_num",
                    }
                )
                continue

            sub_df = build_sub_row(cik, company_name, sic, filing.to_dict())
            out_dir = write_filing_bundle(SEC_CATCHUP_DIR, sub_df, num_df)
            catchup_adsh.add(adsh)
            downloaded += 1
            print(f"  ✓ Saved {adsh} → {out_dir} ({len(num_df):,} num rows)")
            summary_rows.append(
                {
                    **filing.to_dict(),
                    "cik": cik,
                    "ticker": ticker,
                    "as_of_date": as_of,
                    "action": "downloaded",
                    "path": out_dir,
                }
            )
            index_rows.append(
                {
                    "adsh": adsh,
                    "cik": cik,
                    "ticker": ticker,
                    "name": company_name,
                    "form": filing["form"],
                    "period": filing.get("period"),
                    "filed": filing.get("filed"),
                    "as_of_date": as_of,
                    "num_rows": len(num_df),
                    "path": out_dir,
                    "caught_up_at": datetime.now().isoformat(timespec="seconds"),
                }
            )

    append_catchup_index(index_rows)

    summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 60)
    print("CATCH-UP SUMMARY")
    print("=" * 60)
    print(f"Downloaded:             {downloaded}")
    print(f"Already in batch:       {skipped_bulk}")
    print(f"Already in catchup:     {skipped_catchup}")
    print(f"No XBRL facts:          {skipped_no_xbrl}")
    print(f"Failed CIKs/filings:    {failed}")
    if index_rows:
        print(f"Index updated:         {SEC_CATCHUP_INDEX_FILE}")
    return summary


def load_catchup_joined(catchup_dir: str = SEC_CATCHUP_DIR) -> pd.DataFrame:
    """
    Load all catch-up ``sub.txt`` / ``num.txt`` pairs and join on ``adsh``.

    Skips non-filing directories (e.g. files sitting directly under ``catchup_dir``).
    """
    frames = []
    for name in sorted(os.listdir(catchup_dir)):
        filing_dir = os.path.join(catchup_dir, name)
        num_path = os.path.join(filing_dir, "num.txt")
        sub_path = os.path.join(filing_dir, "sub.txt")
        if not (os.path.isdir(filing_dir) and os.path.isfile(num_path) and os.path.isfile(sub_path)):
            continue

        sub = pd.read_csv(sub_path, sep="\t", dtype=str, low_memory=False)
        num = pd.read_csv(num_path, sep="\t", low_memory=False)
        if sub.empty or num.empty:
            continue

        joined = num.merge(
            sub[["adsh", "cik", "name", "sic", "form", "period", "filed"]],
            on="adsh",
            how="inner",
        )
        frames.append(joined)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["cik"] = pd.to_numeric(df["cik"], errors="coerce").astype("Int64")
    df["period"] = pd.to_numeric(df["period"], errors="coerce")
    df["ddate"] = pd.to_numeric(df["ddate"], errors="coerce")
    df["qtrs"] = pd.to_numeric(df["qtrs"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def build_catchup_current_groups(df_joined: pd.DataFrame) -> pd.DataFrame:
    """
    Build a ``history_comparisons``-compatible frame with current values only.

    Catch-up companyfacts data has empty segments, so we skip segment collapsing.
    For each (cik, period, form, tag, qtrs) group, keep the row with the latest
    ``ddate`` as ``crt_value`` and leave historical change lists empty.
    """
    required = {"cik", "period", "form", "tag", "qtrs", "ddate", "value", "uom"}
    missing = required - set(df_joined.columns)
    if missing:
        raise ValueError(f"Catch-up join missing columns: {sorted(missing)}")

    df = df_joined[
        df_joined["form"].isin(DEFAULT_FORMS)
        & (df_joined["uom"] == "USD")
        & df_joined["value"].notna()
        & df_joined["cik"].notna()
        & df_joined["period"].notna()
        & df_joined["ddate"].notna()
        & df_joined["qtrs"].notna()
        & (df_joined["qtrs"] <= 4)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    # Drop custom-tag style rows if version equals accession (bulk convention).
    if "version" in df.columns and "adsh" in df.columns:
        df = df[df["version"].astype(str) != df["adsh"].astype(str)].copy()

    df = df.sort_values("ddate")
    current = (
        df.groupby(["cik", "period", "form", "tag", "qtrs"], as_index=False)
        .agg(crt_ddate=("ddate", "last"), crt_value=("value", "last"))
    )
    current["quarter_intervals"] = [[] for _ in range(len(current))]
    current["percentage_diffs"] = [[] for _ in range(len(current))]
    current["cik"] = current["cik"].astype(int)
    current["period"] = current["period"].astype(int)
    current["qtrs"] = current["qtrs"].astype(int)

    return current.set_index(["cik", "period", "form", "tag", "qtrs"])


def featurize_df_catchup(
    catchup_dir: str = SEC_CATCHUP_DIR,
    output_file: str = FEATURIZED_CATCHUP_FILE,
    k_top_tags: int = DEFAULT_K_TOP_TAGS,
) -> pd.DataFrame:
    """
    Featurize all catch-up filings into a bulk-compatible feature table.

    Simplified vs ``featurize.py``:
    - no ``segment_group_summary`` (catch-up facts have no segments)
    - no ``history_comparisons`` / QoQ change features
    - reuses ``organize_feature_dataframe`` with ``N_qtrs_history_comp=0``

    Writes ``output_file`` (default ``data/SEC_catchup_query/featurized_all_catchup.csv``)
    with columns ``cik``, ``period``, ``form``, ``data_qtr``, plus ``*_current`` features.
    """
    print(f"Loading catch-up filings from: {catchup_dir}")
    df_joined = load_catchup_joined(catchup_dir)
    if df_joined.empty:
        print("No catch-up filings found.")
        return pd.DataFrame()

    print(
        f"Loaded {len(df_joined):,} fact rows from "
        f"{df_joined['adsh'].nunique():,} filing(s), "
        f"{df_joined['cik'].nunique():,} CIK(s)"
    )

    df_tags = read_tags_to_featurize(K_top_tags=k_top_tags)
    grouped = build_catchup_current_groups(df_joined)
    if grouped.empty:
        print("No USD 10-K/10-Q facts available to featurize.")
        return pd.DataFrame()

    df_features = organize_feature_dataframe(
        grouped,
        df_tags,
        N_qtrs_history_comp=0,
        debug_print=True,
    )
    df_features["data_qtr"] = "catchup"

    # Match featurized_all_quarters column order: ids first, then features.
    id_cols = ["cik", "period", "form", "data_qtr"]
    feature_cols = [c for c in df_features.columns if c not in id_cols]
    df_features = df_features[id_cols + sorted(feature_cols)]

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    df_features.to_csv(output_file, index=False)
    print(f"Saved featurized catch-up data ({df_features.shape}) to: {output_file}")
    return df_features


def main(events_file: str = CATCHUP_EVENTS_FILE) -> pd.DataFrame:
    print(f"Loading events from: {events_file}")
    events = load_ticker_date_events(events_file)
    if events.empty:
        print("No tickers with CIK mappings to process.")
        return pd.DataFrame()

    print(f"Processing {len(events)} ticker(s):")
    print(events.to_string(index=False))
    return catch_up_from_events(events)


if __name__ == "__main__":
    main()
