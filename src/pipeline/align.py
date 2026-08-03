"""Align 30-minute solar wind + GFZ HPo frames onto the lookback grid.

This is where the missing-data policy is enforced: fill small gaps inside each
column's observed range, rollback the anchor when the GFZ nowcast tail is NaN,
and raise InsufficientDataError when the most recent steps are unusable.

The filling must not reach past the last real observation. Until 2026-08 it did:
ffill(limit=2) followed by an unbounded interpolate(limit_direction="both") padded
trailing NaNs with the last valid value, which removed the very NaNs the recency
guard, the residual check, the rollback loop and InsufficientDataError were all
written to detect. See docs/geoindex-realtime/clock-audit-2026-08.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class InsufficientDataError(RuntimeError):
    """Raised when the aligned window cannot be made dense enough to predict."""


@dataclass
class AlignResult:
    """Output of the align step.

    Attributes:
        frame: Merged 30-min DataFrame indexed by datetime, covering
            `t_end - (lookback_steps - 1) * 30min` through `t_end`.
        t_end: Anchor timestamp (most recent 30-min boundary used).
        filled_fraction: Fraction of entries that required forward-fill /
            interpolation (0.0–1.0), useful for telemetry.
    """

    frame: pd.DataFrame
    t_end: pd.Timestamp
    filled_fraction: float


def _floor_to_boundary(now: datetime, offset_minutes: int) -> pd.Timestamp:
    """Return the latest 30-min boundary that is `offset_minutes` in the past."""
    reference = pd.Timestamp(now) - pd.Timedelta(minutes=offset_minutes)
    floored_minute = 30 if reference.minute >= 30 else 0
    return reference.replace(minute=floored_minute, second=0, microsecond=0)


def _count_tail_nans(series: pd.Series, n: int) -> int:
    """Return the count of NaNs among the last `n` values of a series."""
    return int(series.iloc[-n:].isna().sum()) if len(series) >= n else int(series.isna().sum())


def align(
    sw_30min: pd.DataFrame,
    hpo: pd.DataFrame,
    now: datetime,
    lookback_steps: int = 96,
    boundary_offset_minutes: int = 2,
    max_gap_fraction: float = 0.10,
    ffill_limit_steps: int = 2,
    require_recent_steps_present: int = 3,
    anchor_rollback_max_attempts: int = 2,
) -> AlignResult:
    """Build the final 96-row window used as model input.

    Args:
        sw_30min: 30-min aggregated solar wind frame indexed by datetime, with
            21 columns (v_avg, ..., bt_max).
        hpo: DataFrame with columns [datetime, hp30, ap30].
        now: Reference wall-clock time (usually datetime.utcnow()).
        lookback_steps: Number of 30-min steps in the output window (default 96).
        boundary_offset_minutes: Safety margin past :00/:30 before trusting the
            newest bin. Bins are left-labelled, so a value below 30 can select an
            interval that has barely begun and hand the model a partial aggregate
            whose min/max span is far narrower than any training row's.
        max_gap_fraction: Reject the run if a single column is NaN in more than
            this fraction of steps after forward-filling.
        ffill_limit_steps: Maximum consecutive fill length, forward or backward,
            before falling back to linear interpolation. Applies to gaps inside a
            column's observed range only; trailing NaNs are preserved so the
            recency guard and the anchor rollback can see them.
        require_recent_steps_present: Number of most-recent steps that must be
            non-NaN after filling; otherwise rollback or raise.
        anchor_rollback_max_attempts: How many times to roll `t_end` back by
            30 min when the GFZ tail is still NaN.

    Returns:
        AlignResult carrying the aligned DataFrame, the resolved anchor, and
        the filled-fraction telemetry number.

    Raises:
        InsufficientDataError: If gap constraints cannot be satisfied.
    """
    hpo_indexed = hpo.copy()
    hpo_indexed["datetime"] = pd.to_datetime(hpo_indexed["datetime"])
    hpo_indexed = hpo_indexed.set_index("datetime").sort_index()

    sw_indexed = sw_30min.sort_index()

    for attempt in range(anchor_rollback_max_attempts + 1):
        t_end = _floor_to_boundary(now - timedelta(minutes=30 * attempt),
                                   boundary_offset_minutes)
        t_start = t_end - pd.Timedelta(minutes=30 * (lookback_steps - 1))
        grid = pd.date_range(t_start, t_end, freq="30min")

        sw_window = sw_indexed.reindex(grid)
        hpo_window = hpo_indexed.reindex(grid)

        merged = pd.concat([sw_window, hpo_window[["ap30", "hp30"]]], axis=1)
        merged.index.name = "datetime"

        # Telemetry: fraction NaN before any filling.
        before_nan = merged.isna().sum().sum()
        total_cells = merged.size

        # Fill gaps INSIDE each column's observed range only. A trailing NaN means the
        # publisher has not released that interval yet -- the GFZ Hp30 nowcast for
        # [T, T+30) cannot exist before T+30, so the newest one to three ap30 rows are
        # always missing -- and the guards below are written to detect exactly that.
        # Filling it forward hides the condition and hands the model a fabricated anchor
        # asserting the index has stopped moving, which is worst during a rising storm.
        # `before_last` is True at or before each column's last real observation.
        before_last = merged.notna()[::-1].cummax()[::-1]
        filled = merged.ffill(limit=ffill_limit_steps)
        filled = filled.interpolate(method="linear", limit_area="inside")
        filled = filled.bfill(limit=ffill_limit_steps)
        filled = filled.where(before_last)

        after_nan = filled.isna().sum().sum()
        filled_fraction = float((before_nan - after_nan) / max(total_cells, 1))

        # Gap fraction per column.
        nan_frac = filled.isna().mean()
        violating = nan_frac[nan_frac > max_gap_fraction]
        if not violating.empty:
            logger.warning("Columns over gap threshold %.2f (attempt %d): %s",
                           max_gap_fraction, attempt, violating.to_dict())
            continue

        # Recent rows must be dense.
        tail_nans = {
            col: _count_tail_nans(filled[col], require_recent_steps_present)
            for col in filled.columns
        }
        if any(count > 0 for count in tail_nans.values()):
            logger.warning("Tail NaN in recent %d steps (attempt %d): %s",
                           require_recent_steps_present, attempt,
                           {k: v for k, v in tail_nans.items() if v > 0})
            continue

        # Still NaN anywhere? Then interpolation ran out of neighbours.
        if filled.isna().any().any():
            logger.warning("Residual NaNs after fill (attempt %d)", attempt)
            continue

        frame = filled.reset_index()
        logger.info("Aligned window anchor=%s, filled_fraction=%.4f",
                    t_end, filled_fraction)
        return AlignResult(frame=frame, t_end=pd.Timestamp(t_end),
                           filled_fraction=filled_fraction)

    raise InsufficientDataError(
        f"Could not build a dense {lookback_steps}-step window after "
        f"{anchor_rollback_max_attempts + 1} anchor attempts. "
        "Check NOAA SWPC and GFZ publisher status."
    )
