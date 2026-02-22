Updates.md

Config
- Frontend exposes: auto‑tune, cache caps, chunk size/seconds/overlap, compress masks, stride, ROI.

Annotation
- Added frame slider + thumbnails for fast frame navigation.
- Point removal + undo + per‑object point list.
- Test‑mask caching.
- Delete annotated objects (backend endpoint + frontend X button).
- Moved "Suggest Optimal Frames" button to right column, above Shortcuts.
- Right panel wrapped in scrollable sticky container.

Processing
- Auto‑tune defaults for cache/stride/chunk + GPU memory fraction.
- Chunked processing + cleanup between chunks.
- Mask compression (packbits) option.
- ROI + frame stride + interpolation.
- Partial outputs saved on failure.
- Video save prefers mp4v and reports save progress.
- CSV export in background thread.
- Persist FPS from ffprobe and reuse on save.
- Bidirectional propagation enabled by default.
- Live progress bar during SAM2 propagation (frame count updates).
- Error status shown on ProcessingPage with link to partial results.

Results
- results response includes outputs_meta for CSV status/progress.
- CSV download disabled while export runs.
- Download buttons fixed (disabled state works, opens in new tab).
- Backend checks actual file existence on disk.
