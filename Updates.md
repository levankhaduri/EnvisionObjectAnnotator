Updates.md

Backend
- Store real video FPS at extract time (ffprobe) and reuse it for saving, so output length matches input.
- CSV export runs in the background now, with progress stored in outputs_meta.
- Results API returns outputs_meta so the UI can show CSV progress.
- Faster video codec order (mp4v first) + save progress callback.
- ROI + frame stride settings wired in processing, with interpolation fill.
- Partial outputs save on failure stays in place.

Frontend
- Config page now exposes all new processing knobs (auto-tune, cache, chunk, stride, ROI, compression).
- Results page disables CSV download while it’s still exporting and shows %.
