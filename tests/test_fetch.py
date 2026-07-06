"""Tests for the NOAA RTSW fetch/parse + source-selection layer."""

import pandas as pd

from src.fetch import noaa_swpc


def _wind_record(time_tag, source, active, speed, density, temp):
    return {
        "time_tag": time_tag,
        "active": active,
        "source": source,
        "proton_speed": speed,
        "proton_temperature": temp,
        "proton_density": density,
    }


def _mag_record(time_tag, source, active, bt, bx, by, bz):
    return {
        "time_tag": time_tag,
        "active": active,
        "source": source,
        "bt": bt,
        "bx_gsm": bx,
        "by_gsm": by,
        "bz_gsm": bz,
    }


def test_prefers_solar1_over_backups_same_minute(monkeypatch):
    """When SOLAR1 and a backup report the same minute, SOLAR1 wins."""
    payload = [
        _wind_record("2026-07-05T00:00:00", "ACE", False, 400.0, 1.0, 1e5),
        _wind_record("2026-07-05T00:00:00", "SOLAR1", True, 500.0, 2.0, 2e5),
    ]
    monkeypatch.setattr(noaa_swpc, "download_json", lambda *a, **k: payload)

    df = noaa_swpc.fetch_plasma("http://x")
    assert list(df.columns) == ["datetime", "np", "v", "t"]
    assert len(df) == 1
    assert df.iloc[0]["v"] == 500.0  # SOLAR1, not ACE
    assert df.iloc[0]["np"] == 2.0


def test_imap_suboffset_floors_and_loses_to_solar1(monkeypatch):
    """A backup on a sub-minute-offset timestamp must floor to the minute and
    lose to SOLAR1 rather than leak in as an extra row."""
    payload = [
        _wind_record("2026-07-05T00:00:00", "SOLAR1", True, 500.0, 2.0, 2e5),
        _wind_record("2026-07-05T00:00:47", "IMAP", False, 999.0, 9.0, 9e5),
    ]
    monkeypatch.setattr(noaa_swpc, "download_json", lambda *a, **k: payload)

    df = noaa_swpc.fetch_plasma("http://x")
    assert len(df) == 1  # not 2 — IMAP collapsed onto the same minute
    assert df.iloc[0]["v"] == 500.0


def test_backup_fills_minute_where_solar1_absent(monkeypatch):
    """Where SOLAR1 has no row for a minute, the active/backup row fills it."""
    payload = [
        _wind_record("2026-07-05T00:00:00", "SOLAR1", True, 500.0, 2.0, 2e5),
        _wind_record("2026-07-05T00:01:30", "ACE", False, 410.0, 1.5, 1e5),
    ]
    monkeypatch.setattr(noaa_swpc, "download_json", lambda *a, **k: payload)

    df = noaa_swpc.fetch_plasma("http://x").sort_values("datetime").reset_index(drop=True)
    assert len(df) == 2
    assert df.iloc[0]["v"] == 500.0  # SOLAR1
    assert df.iloc[1]["v"] == 410.0  # ACE fills the SOLAR1-less minute


def test_mag_column_rename(monkeypatch):
    payload = [
        _mag_record("2026-07-05T00:00:00", "SOLAR1", True, 5.6, 0.7, 0.1, 5.5),
    ]
    monkeypatch.setattr(noaa_swpc, "download_json", lambda *a, **k: payload)

    df = noaa_swpc.fetch_mag("http://x")
    assert list(df.columns) == ["datetime", "bx", "by", "bz", "bt"]
    assert df.iloc[0]["bz"] == 5.5
    assert df.iloc[0]["bt"] == 5.6


def test_fetch_swpc_merges_to_ordered_frame(monkeypatch):
    wind = [_wind_record("2026-07-05T00:00:00", "SOLAR1", True, 500.0, 2.0, 2e5)]
    mag = [_mag_record("2026-07-05T00:00:00", "SOLAR1", True, 5.6, 0.7, 0.1, 5.5)]

    def fake_download(url, *a, **k):
        return wind if "wind" in url else mag

    monkeypatch.setattr(noaa_swpc, "download_json", fake_download)

    merged = noaa_swpc.fetch_swpc("http://x/rtsw_wind_1m.json", "http://x/rtsw_mag_1m.json")
    assert list(merged.columns) == ["datetime", "v", "np", "t", "bx", "by", "bz", "bt"]
    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["v"] == 500.0 and row["bz"] == 5.5


def test_null_measurements_become_nan(monkeypatch):
    payload = [
        _wind_record("2026-07-05T00:00:00", "SOLAR1", True, None, None, None),
    ]
    monkeypatch.setattr(noaa_swpc, "download_json", lambda *a, **k: payload)

    df = noaa_swpc.fetch_plasma("http://x")
    assert df.iloc[0][["v", "np", "t"]].isna().all()
