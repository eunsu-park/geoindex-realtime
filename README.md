# geoindex-realtime

Real-time geomagnetic index prediction from live solar-wind and geomagnetic
nowcast feeds. The engine is **index-agnostic** — the predicted index is
determined by the vendored checkpoint; the current default checkpoint targets
**ap30** (30-minute ap index).

---

## Overview

This project is the on-demand inference companion of [geoindex-model](../geoindex-model)
and [geoindex-data](../geoindex-data). It downloads live data from NOAA SWPC (solar
wind) and GFZ Potsdam (Hp30/ap30 nowcast), preprocesses it into the same
30-minute event format used for training, and runs the best-performing trained
model to produce a 12-hour ap30 forecast.

- **Default model**: `in12h_out12h_gnn_patchtst` (Val Loss 0.245454, Val MAE 0.3781)
- **Input**: 12-hour lookback (24 steps at 30-min cadence, 22 variables → tensor (1, 24, 22))
- **Output**: 24-step ap30 forecast (30 min → 12 hours ahead)
- **Execution**: On-demand CLI (single-run). Run manually when a forecast is needed.

---

## Architecture

```
NOAA RTSW wind.json    ─┐
NOAA RTSW mag.json     ─┼─► fetch ─► aggregate(30-min) ─┐
GFZ Hp30/ap30 nowcast  ─┘                               ├─► align ─► event CSV ─► predict ─► results JSON/CSV
                                                         ┘
```

All vendored dependencies (downloader, normalizer, model code) live under
`src/_vendor/`. The project has no runtime dependency on the sibling folders.

---

## Quickstart

```bash
conda activate ap
pip install -r requirements.txt

# Windows (default) — assumes workspace at D:/realtime/ with the standard layout
python scripts/run_realtime.py

# macOS / Linux — uses ~/realtime/... paths (see configs/realtime.mac.yaml)
python scripts/run_realtime.py --config configs/realtime.mac.yaml
```

Both configs share the same schema; only `paths.*` differ. The default is
Windows (`configs/realtime.yaml`, `D:/realtime/...`); on macOS switch with
`--config configs/realtime.mac.yaml`. Because the code is entirely based on
`pathlib.Path` and `encoding="utf-8"`, it works on both platforms without
modification, and forward-slash (`D:/...`) paths remain valid on Windows.

All profiles (24 input windows × 9 model architectures) are defined in
[`configs/profile/io/`](configs/profile/io/) and
[`configs/profile/model/`](configs/profile/model/); point `profile.io` and
`profile.model` in your runtime yaml to pick any combination.

Optional flags:

| Flag | Purpose |
|---|---|
| `--config PATH` | Override the config path (default `configs/realtime.yaml`) |
| `--now ISO8601`  | Pin the anchor time for reproducibility (e.g. `2026-04-19T12:00:00`) |
| `--dry-run`      | Use cached fixtures under `tests/fixtures/` instead of live network |
| `--device`       | `cpu`, `cuda`, or `mps` (overrides YAML) |
| `--verbose`      | Enable DEBUG-level logging |

---

## Configuration

Edit `configs/realtime.yaml` to change URLs, paths, window size, or the
missing-data policy. The checkpoint and training statistics paths default to the
OneDrive share used during training; point them at local copies if preferred.

Key keys:

```yaml
paths:
  checkpoint: "/path/to/model_best.pth"
  stats_file: "/path/to/table_stats.pkl"
sources:
  noaa_plasma_url: "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
  noaa_mag_url:    "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"
  gfz_hpo_url:     "https://www-app3.gfz-potsdam.de/kp_index/Hp30_ap30_nowcast.txt"
window:
  lookback_steps: 24
  forecast_steps: 24
```

---

## Data Sources

| Source | URL | Cadence | Purpose |
|---|---|---|---|
| NOAA RTSW wind | https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json | ~1 min | density, speed, temperature |
| NOAA RTSW mag  | https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json  | ~1 min | bx/by/bz/bt (GSM) |
| GFZ HPo nowcast | https://www-app3.gfz-potsdam.de/kp_index/Hp30_ap30_nowcast.txt | 30 min | Hp30, ap30 |

NOAA's RTSW feeds publish one record per timestamp **per spacecraft**
(`source` ∈ {SOLAR1, ACE, IMAP, ...}). SOLAR1 replaced DSCOVR as the primary L1
monitor; the fetch layer keeps SOLAR1 per timestamp and only falls back to
NOAA's `active`-flagged backup where SOLAR1 is missing. The retired
`products/solar-wind/*-7-day.json` feeds (DSCOVR-era, header-first format) are
no longer served.

Each RTSW file covers a rolling ~24 hours — enough for the 12-hour lookback but
not for multi-day backtests. For historical backtests, use the OMNI archive via
`geoindex-data` instead.

---

## Output Schema

Each run produces a JSON and a CSV file in
`results/predictions/{YYYYMMDD}/{anchor_timestamp}.{json,csv}`.

```json
{
  "run_timestamp_utc": "2026-04-19T12:17:03Z",
  "anchor_timestamp_utc": "2026-04-19T12:00:00Z",
  "model": {
    "profile": "in12h_out12h_gnn_patchtst",
    "checkpoint_path": "...",
    "checkpoint_sha256": "abcd1234...",
    "val_loss_at_train": 0.245454,
    "val_mae_at_train": 0.3781
  },
  "input": {
    "event_csv": "dataset/events/20260419120000.csv",
    "sources": { "noaa_plasma_url": "...", "noaa_mag_url": "...", "gfz_hpo_url": "..." },
    "missing_data_filled_fraction": 0.017
  },
  "forecast": [
    {"horizon_steps": 1, "horizon_minutes": 30, "target_timestamp_utc": "2026-04-19T12:30:00Z", "ap30": 7.2}
  ]
}
```

CSV columns: `horizon_steps, horizon_minutes, target_timestamp_utc, ap30_pred`.

---

## Model

The default profile `in12h_out12h_gnn_patchtst` is an 8-node GNN encoder with a
PatchTST temporal backbone, selected as the single best model across the
24 × 9 = 216 profile grid (24 input/output windows × 9 model architectures;
lowest validation loss of 0.245454). The checkpoint and the per-variable
statistics file (`table_stats.pkl`) used at training time are required for
inference.

Checkpoint size: ~4.5 MB. CPU inference latency: ~100 ms per request.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `InsufficientDataError` | NOAA/GFZ outage or many recent NaNs | Retry later; check source URLs |
| `FileNotFoundError: table_stats.pkl` | Stats file path wrong | Update `paths.stats_file` in config |
| Unrealistic forecast values | Stats/model mismatch | Ensure stats file matches checkpoint training |
| SSL warnings | `urllib3` InsecureRequestWarning | Emitted when fetching from NOAA SWPC / GFZ Potsdam over legacy TLS — safe to ignore |

---

## License

MIT License. See [LICENSE](LICENSE).
