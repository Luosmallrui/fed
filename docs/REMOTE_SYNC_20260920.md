# Remote workspace sync — 2026-09-20

Source: `/data/home/202422011138/bishe` on `10.10.1.1:6988`

Destination: `/Users/xuboyuan.6/Documents/bishe`

## Included

- Project code and metadata: `README.md`, `.gitignore`, `pyproject.toml`, `requirements.txt`
- `configs/`, `src/`, `scripts/`, `tests/`
- Complete `docs/` tree, including all experiment results, reports, logs, CSV files, and checkpoints
- Complete `datasets/` tree

The merge was non-destructive: remote files replaced matching local paths, while local-only files were retained. Replaced local versions were copied to:

`tmp/remote_sync_20260920/pre_sync_backup`

A stale local-only run lock and generated `__pycache__` directories were moved into that backup rather than deleted. The only intentional local-only file left inside the synchronized trees is this sync report.

## Verification

| Group | Files | Bytes | SHA-256 tree digest |
|---|---:|---:|---|
| code | 123 | 1,638,114 | `f8826e2ab310350ecee5783eccd0aabccd5478a74549db318d4e46b8a57acbcb` |
| docs | 17,828 | 5,782,963,142 | `ec4749963d94891a8282ddd8b9d581320fbc85d2b466e612206bf8e825d32a1a` |
| datasets | 202 | 1,293,141,949 | `848cf8a9aaff4ae176cd9deddb3453f9805ed939385bbc2ad8af2d94b25725b2` |

Remote and staging digests matched for every substantive file. A post-merge checksum pass reported zero files requiring transfer.

The Full-LoRA Top-k no-EF campaign was also checked after synchronization: `COMPLETE`, 18 completed, 0 failed, and 18/18 `VALID` report rows.

## Intentionally excluded

- `artifacts/` and `cache/`: reproducible caches and frozen execution snapshots
- `.venv/`, `.deps/`, `.compat/`: server-specific runtime environments
- `models/`: approximately 20 GB of reproducible model cache
- top-level legacy `logs/` and `bishe_fedpeft_code.tgz`
- macOS AppleDouble files named `._*` and `.DS_Store`
