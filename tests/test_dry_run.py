"""Integration test over the pinned fixtures: fetch parsing -> aggregate -> align.

`tests/test_align.py` pins the missing-data policy against synthetic frames. This runs the
same policy over the real parsers and the real NOAA/GFZ payload shapes, from the same fixtures
`--dry-run` uses, so a regression in the parsing or the aggregation boundary is caught too.

The model is deliberately not loaded — the checkpoint lives on the deployment host — so this
covers everything up to the point where the window is handed to the predictor.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

from src._vendor.parse_hpo import HP30, parse_hpo
from src.fetch.noaa_swpc import (_MAG_RENAME, _PLASMA_RENAME, _numeric,
                                 _records_to_dataframe)
from src.pipeline.aggregate import aggregate_30min
from src.pipeline.align import align

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
NOW = pd.Timestamp("2026-07-05 12:03:00")
LOOKBACK = 24
ROLLBACK = 4
# The fixture withholds the two newest GFZ bins, so this is the newest anchor that exists.
NEWEST_PUBLISHED = pd.Timestamp("2026-07-05 11:00:00")


@pytest.fixture(scope="module")
def sources():
    """Parse the fixtures exactly as `run_realtime.py --dry-run` does."""
    plasma = _records_to_dataframe(json.loads((FIXTURES / "plasma.json").read_text()))
    plasma = plasma.rename(columns=_PLASMA_RENAME)
    plasma = _numeric(plasma, ["np", "v", "t"])[["datetime", "np", "v", "t"]]

    mag = _records_to_dataframe(json.loads((FIXTURES / "mag.json").read_text()))
    mag = mag.rename(columns=_MAG_RENAME)
    mag = _numeric(mag, ["bx", "by", "bz", "bt"])[["datetime", "bx", "by", "bz", "bt"]]

    swpc = plasma.merge(mag, on="datetime", how="outer").sort_values("datetime")
    swpc = swpc.reset_index(drop=True)

    hpo = parse_hpo((FIXTURES / "hpo.txt").read_text(), HP30)
    hpo = hpo.rename(columns={"Hp30": "hp30"})
    hpo = hpo.dropna(subset=["datetime"])[["datetime", "hp30", "ap30"]].reset_index(drop=True)
    return swpc, hpo


def _aligned(sources):
    swpc, hpo = sources
    now_ts = NOW.floor("30min")
    agg_start = now_ts - pd.Timedelta(minutes=30 * (LOOKBACK + ROLLBACK + 2))
    sw_30min = aggregate_30min(swpc, start=agg_start.to_pydatetime(),
                               end=(now_ts + pd.Timedelta(minutes=30)).to_pydatetime())
    return align(sw_30min, hpo, now=NOW, lookback_steps=LOOKBACK,
                 anchor_rollback_max_attempts=ROLLBACK)


def test_the_fixture_really_withholds_its_newest_index_bins(sources):
    """Guard the guard: if the fixture stopped exercising the tail case, say so here."""
    _, hpo = sources
    assert hpo["ap30"].isna().sum() >= 2
    assert hpo.dropna(subset=["ap30"])["datetime"].max() == NEWEST_PUBLISHED


def test_anchor_rolls_back_to_the_newest_published_bin(sources):
    result = _aligned(sources)
    assert result.t_end == NEWEST_PUBLISHED


def test_nothing_in_the_served_window_is_fabricated(sources):
    result = _aligned(sources)
    assert result.filled_fraction == pytest.approx(0.0)
    assert result.frame.notna().all().all()


def test_the_rising_index_reaches_the_model_still_rising(sources):
    """The regression this whole file exists for.

    The fixture's ap30 climbs into the withheld tail. Filling that tail forward would hand the
    model a flat anchor at the moment the index is climbing fastest.
    """
    result = _aligned(sources)
    tail = result.frame["ap30"].tail(4).tolist()
    assert tail == sorted(tail), f"served index tail is not monotonically rising: {tail}"
    assert len(set(tail)) == len(tail), f"served index tail is flat-lined: {tail}"


def test_window_shape_matches_what_the_caller_asserts(sources):
    result = _aligned(sources)
    assert len(result.frame) == LOOKBACK
    stamps = pd.to_datetime(result.frame["datetime"])
    assert stamps.iloc[-1] == result.t_end
    assert (stamps.diff().dropna() == pd.Timedelta(minutes=30)).all()
