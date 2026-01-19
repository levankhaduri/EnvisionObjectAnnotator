Updates.md

Config
- Frontend exposes: auto‑tune, cache caps, chunk size/seconds/overlap, compress masks, stride, ROI.

Annotation
- Added frame slider + thumbnails for fast frame navigation.
- Point removal + undo + per‑object point list.
- Test‑mask caching.

Processing
- Auto‑tune defaults for cache/stride/chunk + GPU memory fraction.
- Chunked processing + cleanup between chunks.
- Mask compression (packbits) option.
- ROI + frame stride + interpolation.
- Partial outputs saved on failure.
- Video save prefers mp4v and reports save progress.
- CSV export in background thread.
- Persist FPS from ffprobe and reuse on save.

Results
- results response includes outputs_meta for CSV status/progress.
- CSV download disabled while export runs.
