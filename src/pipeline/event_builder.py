"""Write the aligned lookback window as a training-schema event CSV."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def build_event_csv(
    aligned: pd.DataFrame,
    t_end: pd.Timestamp,
    out_dir: Path,
    input_variables: Sequence[str],
) -> Path:
    """Persist the aligned window to `{out_dir}/{t_end:%Y%m%d%H%M%S}.csv`.

    The CSV schema matches the training event format: `datetime` + 21 solar
    wind parameters + `ap30` + `hp30`. Both geomagnetic indices are always
    written regardless of the model target, so the same CSV serves an ap30 or
    hp30 model (each selects its own columns by name downstream).

    Args:
        aligned: Aligned window from `pipeline.align.align`.
        t_end: Anchor timestamp used to name the file.
        out_dir: Destination directory (created if missing).
        input_variables: Ordered input list for the active target (21 SW + the
            target index, e.g. ap30 or hp30). The solar-wind columns are taken
            from it; both ap30 and hp30 are always appended.

    Returns:
        Path to the written CSV.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Solar-wind columns come from input_variables (minus whichever geomag index
    # it carries); both ap30 and hp30 are always written to match the training
    # schema and to stay target-agnostic.
    geomag = ("ap30", "hp30")
    sw_cols = [c for c in input_variables if c not in geomag]
    expected_cols = ["datetime", *sw_cols, "ap30", "hp30"]
    missing = [col for col in expected_cols if col not in aligned.columns]
    if missing:
        raise ValueError(f"Aligned frame is missing columns required for CSV: {missing}")

    ordered = aligned[expected_cols].copy()
    ordered["datetime"] = pd.to_datetime(ordered["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    filename = f"{t_end.strftime('%Y%m%d%H%M%S')}.csv"
    out_path = out_dir / filename
    ordered.to_csv(out_path, index=False)
    logger.info("Wrote event CSV %s (%d rows, %d cols)",
                out_path, len(ordered), len(ordered.columns))
    return out_path
