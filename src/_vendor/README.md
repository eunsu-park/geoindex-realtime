# Vendored Code

Files under this directory are **verbatim or near-verbatim copies** from the
two sibling projects. They carry upstream identifiers so changes can be
tracked and resynchronized.

## Provenance

| Vendored path | Upstream path | Commit |
|---|---|---|
| `download.py` | `geoindex-data/core/download.py` (subset) | `de72933` |
| `parse_hpo.py` | `geoindex-data/core/parse.py` (HP30 subset) | `de72933` |
| `normalizer.py` | `geoindex-model/src/pipeline/normalizer.py` (Normalizer only) | `2d89767` |
| `checkpoint.py` | `geoindex-model/src/utils.py` (load_model, setup_device) | `2d89767` |
| `networks/_registry.py` | `geoindex-model/src/networks/_registry.py` | `2d89767` |
| `networks/_base.py` | `geoindex-model/src/networks/_base.py` | `2d89767` |
| `networks/gnn.py` | `geoindex-model/src/networks/gnn.py` | `2d89767` |
| `networks/transformer.py` | `geoindex-model/src/networks/transformer.py` | `2d89767` |
| `networks/tcn.py` | `geoindex-model/src/networks/tcn.py` (for gnn.py import) | `2d89767` |
| `networks/patchtst.py` | `geoindex-model/src/networks/patchtst.py` (for gnn.py import) | `2d89767` |

## Resync Procedure

1. Diff the upstream file against the vendored copy.
2. Port non-trivial upstream changes over manually.
3. Update the commit hash in the table above and in the per-file header.
4. Re-run `pytest tests/` to confirm nothing broke.

## DO NOT Edit Directly

If you must adjust vendored code (e.g. to change an import path), keep the
change minimal and note it in the per-file header as `# Local patch:`.
Anything beyond a path tweak should instead live in the adapter modules that
wrap the vendored symbols (`src/fetch/`, `src/inference/`, etc.).
