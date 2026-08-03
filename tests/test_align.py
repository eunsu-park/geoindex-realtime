"""Tests for the window alignment and its missing-data policy.

There was no test file here until 2026-08, and the gap hid a real defect: `ffill(limit=2)`
followed by an unbounded `interpolate(limit_direction="both")` padded trailing NaNs with the
last valid value, which removed the very NaNs that the recency guard, the residual check, the
anchor rollback and `InsufficientDataError` are all written to detect. The service therefore
handed the model a fabricated anchor whenever the GFZ nowcast tail was missing -- which is
always, since the nowcast for [T, T+30) cannot exist before T+30 -- and reported
`filled_fraction` around 0.007, indistinguishable from healthy.

The tests below pin the property that matters: **nothing past a column's last real observation
is ever filled**, so the anchor is always a timestamp the data actually reaches.

See docs/geoindex-realtime/clock-audit-2026-08.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline.align import align, InsufficientDataError

SW_COLS = ["v_avg", "v_min", "v_max", "np_avg", "np_min", "np_max",
           "t_avg", "t_min", "t_max", "bx_avg", "bx_min", "bx_max",
           "by_avg", "by_min", "by_max", "bz_avg", "bz_min", "bz_max",
           "bt_avg", "bt_min", "bt_max"]
NOW = pd.Timestamp("2026-07-05 12:03:00")
LAST_BOUNDARY = pd.Timestamp("2026-07-05 12:00:00")
LOOKBACK = 24


def _grid(steps: int = 96) -> pd.DatetimeIndex:
    return pd.date_range(end=LAST_BOUNDARY, periods=steps, freq="30min")


def _sw(grid: pd.DatetimeIndex, missing_tail: int = 0) -> pd.DataFrame:
    """Solar wind frame indexed by datetime, optionally missing its newest rows."""
    df = pd.DataFrame({c: np.linspace(300.0, 500.0, len(grid)) for c in SW_COLS}, index=grid)
    df.index.name = "datetime"
    if missing_tail:
        df.iloc[-missing_tail:] = np.nan
    return df


def _hpo(grid: pd.DatetimeIndex, values=None, missing_tail: int = 0) -> pd.DataFrame:
    """GFZ frame with a `datetime` column, optionally missing its newest rows."""
    ap = np.full(len(grid), 20.0) if values is None else np.asarray(values, dtype=float)
    df = pd.DataFrame({"datetime": grid, "ap30": ap, "hp30": ap / 14.0})
    if missing_tail:
        df.loc[df.index[-missing_tail:], ["ap30", "hp30"]] = np.nan
    return df


def _last_real(hpo: pd.DataFrame) -> pd.Timestamp:
    return hpo.dropna(subset=["ap30"])["datetime"].max()


class TestTailIsNeverFabricated:
    """The property the missing-data policy exists to guarantee."""

    @pytest.mark.parametrize("missing_tail", [1, 2, 3, 4])
    def test_anchor_never_passes_the_last_real_observation(self, missing_tail):
        """However stale the nowcast is, the anchor lands on data that exists."""
        grid = _grid()
        hpo = _hpo(grid, missing_tail=missing_tail)
        result = align(_sw(grid), hpo, now=NOW, lookback_steps=LOOKBACK,
                       anchor_rollback_max_attempts=4)
        assert result.t_end <= _last_real(hpo)

    @pytest.mark.parametrize("missing_tail", [1, 2, 3])
    def test_no_value_in_the_window_is_past_its_column_last_observation(self, missing_tail):
        grid = _grid()
        hpo = _hpo(grid, missing_tail=missing_tail)
        result = align(_sw(grid), hpo, now=NOW, lookback_steps=LOOKBACK,
                       anchor_rollback_max_attempts=4)
        served = result.frame.set_index("datetime")["ap30"]
        assert not served.index.max() > _last_real(hpo)
        assert served.notna().all()

    def test_a_rising_storm_is_not_flat_lined(self):
        """The failure that motivated this file.

        With the tail filled forward, the two newest inputs asserted the index had stopped
        rising at the moment it was rising fastest. The served tail must be the real rise.
        """
        grid = _grid()
        ap = np.full(len(grid), 10.0)
        ap[-9:-2] = [15, 22, 32, 48, 70, 94, 132]
        hpo = _hpo(grid, values=ap, missing_tail=2)
        result = align(_sw(grid), hpo, now=NOW, lookback_steps=LOOKBACK,
                       anchor_rollback_max_attempts=4)
        tail = result.frame.set_index("datetime")["ap30"].tail(3)
        assert list(tail) == [70.0, 94.0, 132.0]
        assert tail.iloc[-1] > tail.iloc[-2] > tail.iloc[-3]   # still rising, as observed

    def test_filled_fraction_reports_zero_when_nothing_was_filled(self):
        """Telemetry used to read ~0.007 while two rows were fabricated."""
        grid = _grid()
        result = align(_sw(grid), _hpo(grid, missing_tail=2), now=NOW,
                       lookback_steps=LOOKBACK, anchor_rollback_max_attempts=4)
        assert result.filled_fraction == pytest.approx(0.0)


class TestRollbackAndRefusal:
    """The guards that the unbounded fill had made unreachable."""

    def test_anchor_rolls_back_one_step_per_missing_row(self):
        grid = _grid()
        hpo = _hpo(grid, missing_tail=2)
        result = align(_sw(grid), hpo, now=NOW, lookback_steps=LOOKBACK,
                       anchor_rollback_max_attempts=4)
        steps_back = (LAST_BOUNDARY - result.t_end) / pd.Timedelta(minutes=30)
        assert steps_back == 2

    def test_refuses_rather_than_fabricating_when_the_feed_is_too_stale(self):
        """Beyond the rollback budget the correct answer is no forecast, not a made-up one."""
        grid = _grid()
        with pytest.raises(InsufficientDataError):
            align(_sw(grid), _hpo(grid, missing_tail=6), now=NOW,
                  lookback_steps=LOOKBACK, anchor_rollback_max_attempts=2)

    def test_a_larger_budget_recovers_a_staler_feed(self):
        grid = _grid()
        hpo = _hpo(grid, missing_tail=5)
        result = align(_sw(grid), hpo, now=NOW, lookback_steps=LOOKBACK,
                       anchor_rollback_max_attempts=6)
        assert result.t_end == _last_real(hpo)

    def test_solar_wind_staleness_also_rolls_the_anchor_back(self):
        """The policy is per column, not specific to the index."""
        grid = _grid()
        result = align(_sw(grid, missing_tail=2), _hpo(grid), now=NOW,
                       lookback_steps=LOOKBACK, anchor_rollback_max_attempts=4)
        assert result.t_end == LAST_BOUNDARY - pd.Timedelta(minutes=60)


class TestInteriorGapsAreStillFilled:
    """Bounding the fill must not make the service brittle to ordinary dropouts."""

    def test_a_short_interior_gap_is_filled(self):
        grid = _grid()
        hpo = _hpo(grid)
        hpo.loc[hpo.index[-8:-6], ["ap30", "hp30"]] = np.nan
        result = align(_sw(grid), hpo, now=NOW, lookback_steps=LOOKBACK,
                       anchor_rollback_max_attempts=4)
        assert result.frame["ap30"].notna().all()
        assert result.filled_fraction > 0

    def test_a_leading_gap_is_filled(self):
        """A hole at the far end of the history is the least influential place to fill."""
        grid = _grid()
        sw = _sw(grid)
        oldest = grid[-LOOKBACK]
        sw.loc[oldest, "v_avg"] = np.nan
        result = align(sw, _hpo(grid), now=NOW, lookback_steps=LOOKBACK,
                       anchor_rollback_max_attempts=4)
        assert result.frame["v_avg"].notna().all()

    def test_a_column_that_is_entirely_missing_is_refused(self):
        grid = _grid()
        sw = _sw(grid)
        sw["bz_avg"] = np.nan
        with pytest.raises(InsufficientDataError):
            align(sw, _hpo(grid), now=NOW, lookback_steps=LOOKBACK,
                  anchor_rollback_max_attempts=4)


class TestWindowShape:
    """What the caller asserts downstream (run_realtime.py:210)."""

    def test_row_count_matches_lookback_steps(self):
        grid = _grid()
        result = align(_sw(grid), _hpo(grid, missing_tail=2), now=NOW,
                       lookback_steps=LOOKBACK, anchor_rollback_max_attempts=4)
        assert len(result.frame) == LOOKBACK

    def test_window_ends_on_the_anchor_and_is_contiguous(self):
        grid = _grid()
        result = align(_sw(grid), _hpo(grid, missing_tail=2), now=NOW,
                       lookback_steps=LOOKBACK, anchor_rollback_max_attempts=4)
        stamps = pd.to_datetime(result.frame["datetime"])
        assert stamps.iloc[-1] == result.t_end
        assert (stamps.diff().dropna() == pd.Timedelta(minutes=30)).all()
