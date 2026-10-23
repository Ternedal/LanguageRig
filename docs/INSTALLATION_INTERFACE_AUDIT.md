# LanguageRig installation and interface audit

Source-inspection date: 2026-10-23. Reviewed `pyproject.toml`, `scripts/prepare-pilot.ps1` and `docs/MODELRIG.md`. This is **not** a runtime execution or GPU qualification report.

## Verified package requirements

- Python **>=3.10**, setuptools build backend; project version **0.1.0**.
- Base dependencies: `pypdf>=5,<7` and `packaging>=24`.
- Optional `train` extra includes PyTorch, Transformers, PEFT, Accelerate, bitsandbytes and datasets with version ranges defined in `pyproject.toml`.
- CLI entry point: `languagerig = languagerig.cli:main`.

## Verified Windows pilot wrapper behavior

`scripts/prepare-pilot.ps1` requires **PowerShell >=7.3**, mandatory `-Books` and `-Name`, and explicit `-TrainingAllowed`. It validates the pilot name, resolves the books directory, selects Python 3.12/3.11/3.10 where available, creates `.venv` if needed, installs the package editable, and runs `prepare-pilot` followed by `doctor`. Default workspace: `data/pilots/<Name>`; default `MaxSteps`: 100.

The wrapper does **not** itself prove successful training, model registration or improved Danish quality. Only supply `-TrainingAllowed` when rights to use the source material are confirmed.

## Documented ModelRig handoff (not independently exercised)

`docs/MODELRIG.md` specifies:

- Ollama endpoint via `MODELRIG_OLLAMA_URL`, default `http://127.0.0.1:11434`; alternate port 11435 is an example.
- Model discovery via Ollama `/api/tags` and model chat via `/api/chat`.
- RAG publishing via `POST /api/v1/rag/ingest`, using `MODELRIG_DEVICE_TOKEN`; ModelRig base URL defaults to `http://127.0.0.1:8080`.
- Training, registration and comparison are distinct states; no automatic ModelRig default-model switch.

## Still requires evidence

- [ ] Execute Windows wrapper with a licensed small corpus and capture exact SHA/toolchain versions.
- [ ] Test Windows-to-WSL endpoint reachability and failure paths.
- [ ] Verify the real ModelRig RAG endpoint schema, auth and ingest receipt at an exact server revision.
- [ ] Test GPU train/resume/merge and evaluate hold-out Danish quality.
- [ ] Verify candidate activation, rollback and client behavior.

See [pilot instructions](PILOT.md), [ModelRig integration](MODELRIG.md) and [acceptance checklist](INTEGRATION_ACCEPTANCE_2026-10-23.md).
