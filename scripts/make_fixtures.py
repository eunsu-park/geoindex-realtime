"""Regenerate the `--dry-run` fixtures under `tests/fixtures/`.

The fixtures are deterministic (seeded) and pinned to `--now 2026-07-05T12:03:00`. They
reproduce the ordinary live condition rather than an idealised one:

  * the GFZ nowcast tail is two bins short of the newest solar-wind minute, because the nowcast
    for [T, T+30) cannot exist before T+30 -- this is the case the anchor rollback exists to
    handle, and it was silently fabricated until 2026-08;
  * a shock arrives in the last few hours, so the index is rising into that missing tail and a
    regression that fills it forward is visible instead of hidden in a quiet period.

Usage:
    python scripts/make_fixtures.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

NOW = pd.Timestamp("2026-07-05 12:03:00")
HISTORY_HOURS = 20
SEED = 7
FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def _records(minutes: pd.DatetimeIndex, columns: dict) -> list:
    """RTSW-shaped list of objects: one record per minute, values as strings."""
    return [
        {
            "time_tag": stamp.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "source": "SOLAR1",
            "active": True,
            **{name: f"{values[i]:.2f}" for name, values in columns.items()},
        }
        for i, stamp in enumerate(minutes)
    ]


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    minutes = pd.date_range(NOW.floor("30min") - pd.Timedelta(hours=HISTORY_HOURS),
                            NOW.floor("min"), freq="1min")
    rng = np.random.default_rng(SEED)
    n = len(minutes)

    # A shock ramping in over the last ~4 hours: speed and |B| up, Bz strongly southward.
    ramp = np.clip((np.arange(n) - (n - 260)) / 260.0, 0.0, 1.0)
    plasma = {
        "proton_density": 4 + 14 * ramp + rng.normal(0, 0.8, n),
        "proton_speed": 380 + 340 * ramp + rng.normal(0, 12, n),
        "proton_temperature": 6e4 + 3.2e5 * ramp + rng.normal(0, 6e3, n),
    }
    mag = {
        "bx_gsm": rng.normal(0, 3.0, n),
        "by_gsm": rng.normal(0, 3.5, n),
        "bz_gsm": -1 - 17 * ramp + rng.normal(0, 1.6, n),
        "bt": 5 + 16 * ramp + rng.normal(0, 0.9, n),
    }
    (FIXTURES / "plasma.json").write_text(json.dumps(_records(minutes, plasma), indent=0))
    (FIXTURES / "mag.json").write_text(json.dumps(_records(minutes, mag), indent=0))

    # GFZ Hp30 text: 30-min bins, left-labelled by hh_start, -1 for unpublished intervals.
    bins = pd.date_range(minutes[0].floor("30min"), NOW.floor("30min"), freq="30min")
    ap_ramp = np.clip((np.arange(len(bins)) - (len(bins) - 9)) / 8.0, 0.0, 1.0)
    ap30 = np.round(6 + 200 * ap_ramp ** 2).astype(int)
    hp30 = np.round(np.cbrt(ap30 / 2.5), 3)
    ap30[-2:] = -1                      # the two newest bins are not published yet
    hp30[-2:] = -1.000
    lines = ["# Hp30 / ap30 nowcast fixture -- see tests/fixtures/README.md"]
    for stamp, hp, ap in zip(bins, hp30, ap30):
        hh = stamp.hour + stamp.minute / 60.0
        doy = stamp.dayofyear + hh / 24.0
        lines.append(f"{stamp.year} {stamp.month:2d} {stamp.day:2d} {hh:5.1f} {hh + 0.25:5.1f} "
                     f"{doy:10.4f} {doy + 0.0104:10.4f} {hp:7.3f} {ap:5d} 0")
    (FIXTURES / "hpo.txt").write_text("\n".join(lines) + "\n")

    print(f"wrote {FIXTURES}")
    print(f"  solar wind : {n} minutes, {minutes[0]} .. {minutes[-1]}")
    print(f"  hpo        : {len(bins)} bins, newest published {bins[-3]}, "
          f"ap30 tail {list(ap30[-6:-2])} then unpublished")
    print(f"  run with   : python scripts/run_realtime.py --dry-run --now "
          f"{NOW.strftime('%Y-%m-%dT%H:%M:%S')} --output-root /tmp/rt")


if __name__ == "__main__":
    main()
