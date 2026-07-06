# geoindex-realtime

Real-time geomagnetic index prediction from live solar-wind and geomagnetic
nowcast feeds. The engine is **index-agnostic** — the predicted index is
determined by the vendored checkpoint; the current default checkpoint targets
**ap30** (30-minute ap index).

태양풍·지자기 실황 데이터를 사용한 실시간 지자기 지수 예측기. 엔진 자체는 특정
지수에 종속되지 않으며(**index-agnostic**), 예측 대상 지수는 반입된 체크포인트가
결정합니다. 현재 기본 체크포인트는 **ap30** (30분 ap 지수) 을 예측합니다.

---

## Overview / 개요

This project is the on-demand inference companion of [geoindex-model](../geoindex-model)
and [geoindex-data](../geoindex-data). It downloads live data from NOAA SWPC (solar
wind) and GFZ Potsdam (Hp30/ap30 nowcast), preprocesses it into the same
30-minute event format used for training, and runs the best-performing trained
model to produce a 12-hour ap30 forecast.

본 프로젝트는 [geoindex-model](../geoindex-model)·[geoindex-data](../geoindex-data)
의 실시간 추론 파트너입니다. NOAA SWPC(태양풍)와 GFZ Potsdam(Hp30/ap30
nowcast)에서 실황 데이터를 내려받아, 학습 때와 동일한 30분 이벤트 포맷으로
전처리한 뒤, 가장 성능이 좋은 학습 모델로 향후 12시간의 ap30 을 예측합니다.

- **Default model / 기본 모델**: `in12h_out12h_gnn_patchtst` (Val Loss 0.245454, Val MAE 0.3781)
- **Input / 입력**: 12-hour lookback (24 steps at 30-min cadence, 22 variables → tensor (1, 24, 22))
- **Output / 출력**: 24-step ap30 forecast (30 min → 12 hours ahead)
- **Execution / 실행 방식**: On-demand CLI (single-run). Run manually when a forecast is needed.

---

## Architecture / 구조

```
NOAA SWPC plasma.json  ─┐
NOAA SWPC mag.json     ─┼─► fetch ─► aggregate(30-min) ─┐
GFZ Hp30/ap30 nowcast  ─┘                               ├─► align ─► event CSV ─► predict ─► results JSON/CSV
                                                         ┘
```

All vendored dependencies (downloader, normalizer, model code) live under
`src/_vendor/`. The project has no runtime dependency on the sibling folders.

재사용된 코드는 `src/_vendor/` 에 모두 복사되어 있으며, 실행 시 자매 폴더를 참조하지
않습니다.

---

## Quickstart / 빠른 시작

```bash
conda activate ap
pip install -r requirements.txt

# Windows (default) — assumes workspace at D:/realtime/ with the standard layout
python scripts/run_realtime.py

# macOS / Linux — uses ~/realtime/... paths (see configs/realtime.mac.yaml)
python scripts/run_realtime.py --config configs/realtime.mac.yaml
```

Both configs share the same schema; only `paths.*` differ.
기본값은 Windows (`configs/realtime.yaml`, `D:/realtime/...`) 이고, macOS 에서는
`--config configs/realtime.mac.yaml` 로 전환합니다. 코드는 전부 `pathlib.Path` 와
`encoding="utf-8"` 기반이라 양쪽에서 수정 없이 동작하며, 포워드 슬래시(`D:/...`)
경로는 Windows 에서도 그대로 유효합니다.

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

## Configuration / 설정

Edit `configs/realtime.yaml` to change URLs, paths, window size, or the
missing-data policy. The checkpoint and training statistics paths default to the
OneDrive share used during training; point them at local copies if preferred.

URL, 경로, 윈도우 크기, 결측 정책은 모두 `configs/realtime.yaml` 에서 수정합니다.
체크포인트와 학습 통계 경로는 학습 시 사용했던 OneDrive 공유 폴더를 기본값으로
하며, 로컬 복사본으로 바꿔도 무방합니다.

Key keys:

```yaml
paths:
  checkpoint: "/path/to/model_best.pth"
  stats_file: "/path/to/table_stats.pkl"
sources:
  noaa_plasma_url: "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
  noaa_mag_url:    "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json"
  gfz_hpo_url:     "https://www-app3.gfz-potsdam.de/kp_index/Hp30_ap30_nowcast.txt"
window:
  lookback_steps: 24
  forecast_steps: 24
```

---

## Data Sources / 데이터 소스

| Source | URL | Cadence | Purpose |
|---|---|---|---|
| NOAA SWPC plasma | https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json | ~1 min | density, speed, temperature |
| NOAA SWPC mag    | https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json    | ~1 min | bx/by/bz/bt (GSM) |
| GFZ HPo nowcast  | https://www-app3.gfz-potsdam.de/kp_index/Hp30_ap30_nowcast.txt       | 30 min | Hp30, ap30 |

NOAA provides only the past 7 days of data; the pipeline cannot backtest further
into the past from this source. For historical backtests, use OMNI archive via
`geoindex-data` instead.

NOAA 는 최근 7일만 제공하므로, 그 이전 시점의 재현 테스트는 본 파이프라인으로
불가능합니다. 장기간 backtest 는 `geoindex-data` 로 OMNI archive 를 사용하세요.

---

## Output Schema / 출력 스키마

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

## Model / 모델

The default profile `in12h_out12h_gnn_patchtst` is an 8-node GNN encoder with a
PatchTST temporal backbone, selected as the single best model across the
24 × 9 = 216 profile grid (24 input/output windows × 9 model architectures;
lowest validation loss of 0.245454). The checkpoint and the per-variable
statistics file (`table_stats.pkl`) used at training time are required for
inference.

기본 프로파일 `in12h_out12h_gnn_patchtst` 는 8-노드 GNN 인코더 + PatchTST 시계열
인코더 조합이며, 24 × 9 = 216 개 프로파일 조합(입출력 윈도우 24 × 모델 구조 9)
중 검증 손실(Val Loss 0.245454)이 가장 낮아 선정되었습니다. 추론에는 학습 당시의
체크포인트와 변수별 통계 파일 (`table_stats.pkl`) 이 필요합니다.

Checkpoint size: ~4.5 MB. CPU inference latency: ~100 ms per request.

---

## Troubleshooting / 문제 해결

| Symptom | Likely cause | Fix |
|---|---|---|
| `InsufficientDataError` | NOAA/GFZ outage or many recent NaNs | Retry later; check source URLs |
| `FileNotFoundError: table_stats.pkl` | Stats file path wrong | Update `paths.stats_file` in config |
| Unrealistic forecast values | Stats/model mismatch | Ensure stats file matches checkpoint training |
| SSL warnings | `urllib3` InsecureRequestWarning | Emitted when fetching from NOAA SWPC / GFZ Potsdam over legacy TLS — safe to ignore |

---

## License / 라이선스

MIT License. See [LICENSE](LICENSE).

MIT 라이선스로 배포됩니다. [LICENSE](LICENSE) 파일을 참고하세요.
