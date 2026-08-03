# `--dry-run` fixtures

Pinned inputs for `python scripts/run_realtime.py --dry-run --now 2026-07-05T12:03:00`.
Regenerate with `python scripts/make_fixtures.py`.

| file | contents |
|---|---|
| `plasma.json` | NOAA RTSW `rtsw_wind_1m` records, 1204 minutes ending 12:03 |
| `mag.json` | NOAA RTSW `rtsw_mag_1m` records, same minutes |
| `hpo.txt` | GFZ Hp30/ap30 nowcast text, 41 half-hour bins ending 12:00 |

Two properties are deliberate, because a fixture that only covers the happy path tests nothing:

- **The GFZ tail is two bins short of the newest solar-wind minute** (`ap30 = -1` for 11:30 and
  12:00). That is the ordinary live condition — the nowcast for `[T, T+30)` cannot exist before
  `T+30` — and it is the case the anchor rollback exists to handle. A correct run logs two
  rollback attempts and settles on anchor `11:00`.
- **A shock arrives in the last ~4 hours**, so `ap30` rises 34 → 56 → 84 → 118 into the missing
  tail. A regression that fills the tail forward would flat-line that rise, and the difference is
  visible in the served window instead of hidden in a quiet period.

Expected log from a correct run:

```
Tail NaN in recent 3 steps (attempt 0): {'ap30': 2, 'hp30': 2}
Tail NaN in recent 3 steps (attempt 1): {'ap30': 1, 'hp30': 1}
Aligned window anchor=2026-07-05 11:00:00, filled_fraction=0.0000
```

See `docs/geoindex-realtime/clock-audit-2026-08.md` in the `geoindex` repo.
