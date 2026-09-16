"""Build a 30-min solar-wind + ap30 table (training schema) from archived NOAA RTSW snapshots.

Input: a folder of raw NOAA JSON snapshots as saved by the collector —
`rtsw_wind_1m_*.json` / `rtsw_mag_1m_*.json` (list-of-records, multi-spacecraft; ~1 day
each) and, before 2026-07-02, the retired `plasma-7-day_*.json` / `mag-7-day_*.json`
(header-first list-of-lists). Snapshots overlap and leave gaps; every minute is taken from
the newest snapshot that has it, and the SOLAR1 → active → any source preference of the
live pipeline (`src.fetch.noaa_swpc._select_primary_source`) is applied per minute.

Output: one parquet with `datetime` + the 21 aggregated wind columns (`v_avg … bt_max`, via
`src.pipeline.aggregate.aggregate_30min`, exactly the live path) + observed `ap30`/`hp30`
from GFZ's `Hp30_ap30_complete_series.txt`, plus provenance columns (`era`, `n_plasma`,
`n_mag`, `src_solar1_frac`). Also an anchor index for a given io window listing every 30-min
time whose input window is complete and whose target window has observed ap30 — the file
`analysis/infer_checkpoint.py` (geoindex-model) takes as `--anchor-index`.

    python scripts/build_nrt_table.py --src ~/PostBox/GeoIndex/NRT \\
        --hpo ~/Downloads/Hp30_ap30_complete_series.txt \\
        --out ~/Projects/GeoIndex/datasets/realtime_nrt/nrt_30min.parquet --io in6h_out6h
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fetch.noaa_swpc import _records_to_dataframe, _select_primary_source  # noqa: E402
from src.pipeline.aggregate import aggregate_30min  # noqa: E402

INPUT_VARIABLES = [f"{v}_{s}" for v in ("v", "np", "t", "bx", "by", "bz", "bt") for s in ("avg", "min", "max")] + ["ap30"]
_SNAP = re.compile(r"_(\d{6}-\d{4})\.json$")


def snapshot_time(path: Path) -> pd.Timestamp:
    return pd.to_datetime(_SNAP.search(path.name).group(1), format="%y%m%d-%H%M")


def read_rtsw(files: list[Path], rename: dict[str, str]) -> pd.DataFrame:
    """Newest-snapshot-wins merge of RTSW list-of-records files, one preferred source per minute."""
    parts = []
    for f in sorted(files, key=snapshot_time):
        df = _records_to_dataframe(json.loads(f.read_text()))
        df = _select_primary_source(df)
        df["_snap"] = snapshot_time(f)
        parts.append(df)
    allrows = pd.concat(parts, ignore_index=True).sort_values(["datetime", "_snap"])
    allrows = allrows.drop_duplicates("datetime", keep="last")
    out = allrows.rename(columns=rename)
    keep = ["datetime", *rename.values(), "source"]
    out = out[keep].copy()
    for c in rename.values():
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.sort_values("datetime").reset_index(drop=True)


def read_legacy(files: list[Path], rename: dict[str, str]) -> pd.DataFrame:
    """Retired 7-day products (header row + rows of strings)."""
    parts = []
    for f in sorted(files, key=snapshot_time):
        rows = json.loads(f.read_text())
        df = pd.DataFrame(rows[1:], columns=rows[0])
        df["datetime"] = pd.to_datetime(df["time_tag"])
        df["_snap"] = snapshot_time(f)
        parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["datetime", *rename.values(), "source"])
    allrows = pd.concat(parts, ignore_index=True).sort_values(["datetime", "_snap"]).drop_duplicates("datetime", keep="last")
    out = allrows.rename(columns=rename)
    out["source"] = "DSCOVR-7day"
    out = out[["datetime", *rename.values(), "source"]].copy()
    for c in rename.values():
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.sort_values("datetime").reset_index(drop=True)


def parse_hpo_series(path: Path) -> pd.DataFrame:
    """GFZ Hp30/ap30 text (YYYY MM DD hh.h hh._m days days_m Hp30 ap30 D) → datetime, hp30, ap30."""
    raw = pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                      names=["y", "m", "d", "hh", "hm", "days", "days_m", "hp30", "ap30", "D"])
    dt = pd.to_datetime(dict(year=raw.y, month=raw.m, day=raw.d)) + pd.to_timedelta(raw.hh, unit="h")
    out = pd.DataFrame({"datetime": pd.DatetimeIndex(dt).as_unit("ns"), "hp30": raw.hp30.to_numpy(dtype=float),
                        "ap30": raw.ap30.to_numpy(dtype=float)})
    out.loc[out.hp30 < 0, "hp30"] = np.nan
    out.loc[out.ap30 < 0, "ap30"] = np.nan
    return out


def despike(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Physical-bounds guard on the 1-min plasma feed (unflagged garbage segments exist in the
    live stream — v ≈ 140 km/s for an hour, single-sample density spikes)."""
    bad_v = df["v"] < 250
    bad_np = (df["np"] <= 0) | (df["np"] > 80)
    bad_t = df["t"] <= 0
    counts = {"v_lt_250": int(bad_v.sum()), "np_out_of_range": int(bad_np.sum()), "t_le_0": int(bad_t.sum())}
    df = df.copy()
    df.loc[bad_v, ["v"]] = np.nan
    df.loc[bad_np, ["np"]] = np.nan
    df.loc[bad_t, ["t"]] = np.nan
    return df, counts


def io_steps(io: str) -> tuple[int, int]:
    def steps(tok):
        return int(round(float(tok[:-1]) * (48 if tok[-1] == "d" else 2)))
    a, b = io.split("_")
    return steps(a[2:]), steps(b[3:])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="folder with the raw NOAA JSON snapshots")
    p.add_argument("--hpo", required=True, help="GFZ Hp30_ap30_complete_series.txt (or nowcast) file")
    p.add_argument("--out", required=True, help="output parquet")
    p.add_argument("--io", default="in6h_out6h", help="io window for the anchor index")
    p.add_argument("--no-despike", action="store_true")
    args = p.parse_args()

    src = Path(args.src).expanduser()
    files = sorted(src.glob("*.json"))
    plasma = pd.concat([read_legacy([f for f in files if f.name.startswith("plasma-7-day")],
                                    {"density": "np", "speed": "v", "temperature": "t"}),
                        read_rtsw([f for f in files if f.name.startswith("rtsw_wind_1m")],
                                  {"proton_density": "np", "proton_speed": "v", "proton_temperature": "t"})],
                       ignore_index=True)
    mag = pd.concat([read_legacy([f for f in files if f.name.startswith("mag-7-day")],
                                 {"bx_gsm": "bx", "by_gsm": "by", "bz_gsm": "bz", "bt": "bt"}),
                     read_rtsw([f for f in files if f.name.startswith("rtsw_mag_1m")],
                               {"bx_gsm": "bx", "by_gsm": "by", "bz_gsm": "bz", "bt": "bt"})],
                    ignore_index=True)
    # where a legacy and an rtsw file overlap on a minute, prefer rtsw (it is the live feed of record)
    plasma = plasma.sort_values(["datetime"]).drop_duplicates("datetime", keep="last")
    mag = mag.sort_values(["datetime"]).drop_duplicates("datetime", keep="last")
    guard = {}
    if not args.no_despike:
        plasma, guard = despike(plasma)
    merged = plasma.merge(mag, on="datetime", how="outer", suffixes=("_p", "_m")).sort_values("datetime")
    merged["source"] = merged["source_p"].fillna(merged["source_m"])
    one_min = merged[["datetime", "v", "np", "t", "bx", "by", "bz", "bt", "source"]].reset_index(drop=True)

    start = one_min.datetime.min().floor("30min")
    end = one_min.datetime.max().floor("30min")
    table = aggregate_30min(one_min[["datetime", "v", "np", "t", "bx", "by", "bz", "bt"]], start, end)
    # provenance per 30-min bin
    bins = one_min.set_index("datetime").resample("30min")
    table["n_plasma"] = bins["v"].count().reindex(table.index).fillna(0).astype(int)
    table["n_mag"] = bins["bt"].count().reindex(table.index).fillna(0).astype(int)
    table["src_solar1_frac"] = bins["source"].apply(lambda s_: float((s_ == "SOLAR1").mean()) if len(s_) else np.nan).reindex(table.index)
    table["era"] = np.where(table.index < pd.Timestamp("2026-07-02"), "legacy7day", "rtsw")
    hpo = parse_hpo_series(Path(args.hpo).expanduser()).set_index("datetime")
    table = table.join(hpo[["ap30", "hp30"]], how="left")
    table = table.reset_index().rename(columns={"index": "datetime"})
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)

    # anchor index for the io window
    k_in, k_out = io_steps(args.io)
    X = table[INPUT_VARIABLES].to_numpy(dtype=float)
    ap = table["ap30"].to_numpy(dtype=float)
    ok_in = np.isfinite(X).all(axis=1)
    ok_y = np.isfinite(ap)
    n = len(table)
    anchors = []
    for i in range(k_in, n - k_out + 1):
        if ok_in[i - k_in:i].all() and ok_y[i:i + k_out].all():
            anchors.append(table.datetime.iloc[i])
    idx = pd.DataFrame({"datetime": pd.DatetimeIndex(anchors).strftime("%Y-%m-%d %H:%M:%S"), "label": 0})
    idx_path = out.with_name(f"anchors_{args.io}.csv")
    idx.to_csv(idx_path, index=False)
    # a one-row "train" index so the dataset class can construct (statistics come from the pickle)
    pd.DataFrame({"datetime": idx.datetime.iloc[:1], "label": 0}).to_csv(out.with_name("train_index_dummy.csv"), index=False)

    era_anchor = table.set_index("datetime").loc[pd.DatetimeIndex(anchors), "era"].value_counts().to_dict() if anchors else {}
    complete_input = pd.Series(ok_in).groupby(table["era"].to_numpy()).mean().round(3).to_dict()
    summary = {
        "snapshots": {"rtsw_wind": len([f for f in files if f.name.startswith("rtsw_wind")]),
                      "rtsw_mag": len([f for f in files if f.name.startswith("rtsw_mag")]),
                      "legacy_plasma": len([f for f in files if f.name.startswith("plasma-7")]),
                      "legacy_mag": len([f for f in files if f.name.startswith("mag-7")])},
        "one_min_rows": int(len(one_min)), "one_min_span": [str(one_min.datetime.min()), str(one_min.datetime.max())],
        "source_share_1min": one_min.source.value_counts(normalize=True).round(3).to_dict(),
        "despike_masked": guard,
        "bins_30min": int(n), "bins_with_complete_inputs_by_era": complete_input,
        "bins_with_ap30": int(ok_y.sum()), "ap30_last": str(hpo.dropna().index.max()),
        "anchors": {"io": args.io, "n": len(anchors), "by_era": era_anchor,
                    "first": str(anchors[0]) if anchors else None, "last": str(anchors[-1]) if anchors else None},
        "gaps_gt_1h": [],
    }
    t = table.loc[ok_in, "datetime"]
    d = t.diff().dt.total_seconds().div(3600)
    for a, b, h in zip(t.shift(1)[d > 1], t[d > 1], d[d > 1]):
        summary["gaps_gt_1h"].append([str(a), str(b), round(float(h), 1)])
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f"table → {out}\nanchors → {idx_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
