"""Fetch and parse NOAA SWPC RTSW real-time solar wind JSON feeds.

NOAA's Real-Time Solar Wind (RTSW) service publishes plasma (`rtsw_wind_1m`)
and magnetic field (`rtsw_mag_1m`) as separate ~1-minute JSON files, each a
**list of records** (one JSON object per row). This replaces the legacy
`products/solar-wind/plasma-7-day.json` / `mag-7-day.json` feeds (header-first
list-of-lists), which were retired together with the DSCOVR mission.

Every timestamp may appear multiple times — once per available spacecraft
(`source` ∈ {SOLAR1, ACE, IMAP, ...}). SOLAR1 is the primary that replaced
DSCOVR; ACE/IMAP are backups. `_select_primary_source` keeps SOLAR1 per
timestamp and only falls back to NOAA's `active`-flagged row where SOLAR1 is
missing, so a SOLAR1 gap degrades to the backup instead of dropping data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .._vendor.download import download_json

logger = logging.getLogger(__name__)


# Primary spacecraft: SOLAR1 replaced DSCOVR as the operational L1 monitor.
_PRIMARY_SOURCE = "SOLAR1"

# RTSW plasma field names → internal short names.
_PLASMA_RENAME = {
    "proton_density": "np",
    "proton_speed": "v",
    "proton_temperature": "t",
}

# RTSW mag field names → internal short names (GSM components, matching training).
_MAG_RENAME = {
    "bx_gsm": "bx",
    "by_gsm": "by",
    "bz_gsm": "bz",
    "bt": "bt",
}


def _records_to_dataframe(records: list) -> pd.DataFrame:
    """Convert an RTSW list-of-objects payload into a typed DataFrame.

    Adds a tz-naive UTC `datetime` column derived from `time_tag`. All other
    fields (including `source` and `active`) are preserved for source selection.
    """
    if not isinstance(records, list) or not records:
        raise ValueError("NOAA RTSW response did not contain any records")

    df = pd.DataFrame.from_records(records)

    if "time_tag" not in df.columns:
        raise ValueError(
            f"NOAA RTSW response missing 'time_tag'; got {list(df.columns)}"
        )

    df["datetime"] = pd.to_datetime(df["time_tag"], utc=True).dt.tz_convert(None)
    df = df.drop(columns=["time_tag"])
    return df


def _select_primary_source(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-spacecraft rows to one row per timestamp.

    Preference per timestamp: SOLAR1 → NOAA `active`-flagged row → any. This
    yields a SOLAR1-only series in normal operation while letting a backup
    (e.g. ACE) fill individual timestamps SOLAR1 is missing, matching the old
    single-source behaviour of the retired 7-day products.
    """
    if "datetime" not in df.columns:
        raise ValueError("Input DataFrame must include a 'datetime' column")

    work = df.copy()

    # Normalize to 1-minute marks. Backup spacecraft (e.g. IMAP) report on
    # sub-minute-offset timestamps; without flooring they would not dedup
    # against SOLAR1's on-the-minute rows and would leak into the series even
    # where SOLAR1 already covers that minute.
    work["datetime"] = work["datetime"].dt.floor("min")

    # Lower rank = higher priority.
    work["_rank"] = 2
    if "active" in work.columns:
        work.loc[work["active"] == True, "_rank"] = 1  # noqa: E712
    if "source" in work.columns:
        work.loc[work["source"] == _PRIMARY_SOURCE, "_rank"] = 0
    else:
        logger.warning("RTSW payload has no 'source' column; deduping by datetime only")

    work = work.sort_values(["datetime", "_rank"], kind="stable")
    work = work.drop_duplicates("datetime", keep="first")
    work = work.drop(columns="_rank")

    if "source" in df.columns:
        n_primary = int((df["source"] == _PRIMARY_SOURCE).sum())
        if n_primary == 0:
            logger.warning(
                "No %s rows in RTSW payload; fell back to active/backup source",
                _PRIMARY_SOURCE,
            )
    return work


def _numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce selected columns to float, treating sentinels/None as NaN."""
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _cache_raw_json(payload: list, cache_dir: Optional[Path], filename: str) -> None:
    """Persist the raw JSON response for reproducibility if a cache dir is set."""
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / filename
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp)
    logger.debug("Cached raw NOAA response → %s", out_path)


def fetch_plasma(url: str, timeout: int = 30, max_retries: int = 3,
                 cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """Download the NOAA RTSW plasma (wind) JSON and return a typed DataFrame.

    Args:
        url: Full URL of rtsw_wind_1m.json.
        timeout: Request timeout in seconds.
        max_retries: Max retry attempts.
        cache_dir: If given, raw JSON is cached under `cache_dir/plasma.json`.

    Returns:
        DataFrame with columns [datetime, np, v, t] (one SOLAR1-preferred row
        per timestamp) sorted by datetime.
    """
    payload = download_json(url, timeout=timeout, max_retries=max_retries)
    if payload is None:
        raise RuntimeError(f"NOAA RTSW plasma download failed: {url}")

    _cache_raw_json(payload, cache_dir, "plasma.json")

    df = _records_to_dataframe(payload)
    df = _select_primary_source(df)
    df = df.rename(columns=_PLASMA_RENAME)
    df = _numeric(df, ["np", "v", "t"])
    df = df[["datetime", "np", "v", "t"]].sort_values("datetime").reset_index(drop=True)
    return df


def fetch_mag(url: str, timeout: int = 30, max_retries: int = 3,
              cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """Download the NOAA RTSW magnetic field JSON and return a typed DataFrame.

    Args:
        url: Full URL of rtsw_mag_1m.json.
        timeout: Request timeout in seconds.
        max_retries: Max retry attempts.
        cache_dir: If given, raw JSON is cached under `cache_dir/mag.json`.

    Returns:
        DataFrame with columns [datetime, bx, by, bz, bt] (one SOLAR1-preferred
        row per timestamp) sorted by datetime.
    """
    payload = download_json(url, timeout=timeout, max_retries=max_retries)
    if payload is None:
        raise RuntimeError(f"NOAA RTSW mag download failed: {url}")

    _cache_raw_json(payload, cache_dir, "mag.json")

    df = _records_to_dataframe(payload)
    df = _select_primary_source(df)
    df = df.rename(columns=_MAG_RENAME)
    df = _numeric(df, ["bx", "by", "bz", "bt"])
    df = df[["datetime", "bx", "by", "bz", "bt"]].sort_values("datetime").reset_index(drop=True)
    return df


def fetch_swpc(plasma_url: str, mag_url: str, timeout: int = 30,
               max_retries: int = 3,
               cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """Fetch both RTSW plasma and mag feeds and join on datetime.

    Args:
        plasma_url: NOAA RTSW wind (plasma) JSON URL.
        mag_url: NOAA RTSW mag JSON URL.
        timeout: Request timeout per endpoint.
        max_retries: Max retries per endpoint.
        cache_dir: Optional directory to cache raw responses.

    Returns:
        Outer-joined DataFrame with columns [datetime, v, np, t, bx, by, bz, bt]
        sorted by datetime. Missing measurements are NaN.
    """
    plasma = fetch_plasma(plasma_url, timeout=timeout, max_retries=max_retries,
                          cache_dir=cache_dir)
    mag = fetch_mag(mag_url, timeout=timeout, max_retries=max_retries,
                    cache_dir=cache_dir)

    merged = plasma.merge(mag, on="datetime", how="outer")
    merged = merged.sort_values("datetime").reset_index(drop=True)

    ordered = ["datetime", "v", "np", "t", "bx", "by", "bz", "bt"]
    for col in ordered:
        if col not in merged.columns:
            merged[col] = np.nan
    merged = merged[ordered]

    logger.info("NOAA RTSW fetched: %d rows, %s → %s",
                len(merged), merged["datetime"].min(), merged["datetime"].max())
    return merged
