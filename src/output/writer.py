"""Write forecast results as JSON + CSV pairs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ForecastArtifacts:
    """Paths to the JSON and CSV outputs of a single run."""
    json_path: Path
    csv_path: Path


def _iso_utc(ts) -> str:
    """Format a timestamp as an ISO-8601 UTC string with a trailing Z."""
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def write_forecast(
    forecast: np.ndarray,
    t_end: pd.Timestamp,
    event_csv: Path,
    results_dir: Path,
    model_meta: Dict,
    source_urls: Dict[str, str],
    missing_data_filled_fraction: float,
    interval_minutes: int = 30,
    analysis: Optional[Dict] = None,
    target_variable: str = "ap30",
) -> ForecastArtifacts:
    """Persist the forecast to `results_dir/{YYYYMMDD}/{anchor}.{json,csv}`.

    Args:
        forecast: Array of shape (forecast_steps,) of predictions for
            `target_variable` (ap30 or hp30).
        t_end: Anchor timestamp (end of input window).
        event_csv: Path to the event CSV used as input.
        results_dir: Root results directory (`results/predictions`).
        model_meta: Dict with keys `profile`, `checkpoint_path`,
            `checkpoint_sha256`, and any additional provenance fields.
        source_urls: Dict with keys `noaa_plasma_url`, `noaa_mag_url`,
            `gfz_hpo_url`.
        missing_data_filled_fraction: Telemetry from the align step.
        interval_minutes: Forecast step size in minutes.
        analysis: Optional dict embedded under the `"analysis"` key with
            MCD summary stats, attention metadata, and plot paths.
        target_variable: Forecast index name; used as the JSON value key and
            the `{target}_pred` CSV column (default "ap30").

    Returns:
        ForecastArtifacts with both written paths.
    """
    run_ts = datetime.now(tz=timezone.utc)
    anchor_iso = _iso_utc(t_end)

    day_dir = Path(results_dir) / t_end.strftime("%Y%m%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    stem = t_end.strftime("%Y%m%d%H%M%S")
    json_path = day_dir / f"{stem}.json"
    csv_path = day_dir / f"{stem}.csv"

    forecast_entries = []
    rows = []
    for step_idx, value in enumerate(forecast, start=1):
        minutes = step_idx * interval_minutes
        target_ts = (t_end + pd.Timedelta(minutes=minutes)).to_pydatetime()
        entry = {
            "horizon_steps": step_idx,
            "horizon_minutes": minutes,
            "target_timestamp_utc": _iso_utc(target_ts),
            target_variable: float(value),
        }
        forecast_entries.append(entry)
        rows.append({
            "horizon_steps": step_idx,
            "horizon_minutes": minutes,
            "target_timestamp_utc": entry["target_timestamp_utc"],
            f"{target_variable}_pred": float(value),
        })

    payload = {
        "run_timestamp_utc": _iso_utc(run_ts),
        "anchor_timestamp_utc": anchor_iso,
        "model": model_meta,
        "input": {
            "event_csv": str(event_csv),
            "sources": source_urls,
            "missing_data_filled_fraction": round(float(missing_data_filled_fraction), 6),
        },
        "forecast": forecast_entries,
    }
    if analysis:
        payload["analysis"] = analysis

    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    logger.info("Forecast written: %s / %s", json_path, csv_path)
    return ForecastArtifacts(json_path=json_path, csv_path=csv_path)
