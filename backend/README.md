# EnvisionObjectAnnotator API

This is a minimal FastAPI scaffold that will eventually wrap the existing SAM2
processing pipeline from `Script/runEnvisionObjectAnnotator.py`.

## Run (when deps are installed)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Notes
- Endpoints are stubs for now (sessions, uploads, config, annotation points).
- Processing is not yet wired to SAM2; this will be the next integration step.
