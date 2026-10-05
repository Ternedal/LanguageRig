"""Prepare an auditable pilot snapshot without GPU work or model downloads."""
from __future__ import annotations

import re
from pathlib import Path

from .core import LanguageRigError, init_workspace, now, write_json
from .corpus import build_dataset, verify_dataset
from .ingest import import_books
from .train import DEFAULTS, training_plan


def prepare_pilot(workspace: Path, source: Path, name: str, *, training_allowed=False,
                  language=None, genre="unknown", topics=None, model_id=None,
                  model_revision="main", max_steps=100) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,60}", name):
        raise LanguageRigError("Pilot name must be a simple filename.")
    if not source.exists() or not (source.is_dir() or source.suffix.lower() in (".epub", ".pdf")):
        raise LanguageRigError("Choose an existing ebook folder, EPUB or PDF.")
    if not training_allowed:
        raise LanguageRigError("prepare-pilot requires --training-allowed for the chosen material.")
    if type(max_steps) is not int or max_steps < 1:
        raise LanguageRigError("max_steps must be a positive integer.")
    if model_id is not None and (not isinstance(model_id, str) or not model_id.strip()):
        raise LanguageRigError("Model ID is empty.")
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise LanguageRigError("Model revision is empty.")
    workspace = workspace.resolve()
    config_path = workspace / "configs" / f"{name}.json"
    report_path = workspace / "pilots" / f"{name}.json"
    dataset_path = workspace / "datasets" / name
    for target in (config_path, report_path, dataset_path, workspace / "runs" / name):
        if target.exists():
            raise LanguageRigError("Pilot name already used; choose a new name or use its existing config.")
    init_workspace(workspace)
    receipt = {"format": "languagerig-pilot/v1", "status": "preparing",
               "name": name, "started_at": now(), "source": str(source.resolve()),
               "workspace": str(workspace), "config_path": str(config_path),
               "dataset_path": str(dataset_path), "report_path": str(report_path),
               "model_downloaded": False, "training_executed": False}
    write_json(report_path, receipt)
    try:
        imported = import_books(workspace, source, genre=genre, language=language,
                                topics=topics, training_allowed=True)
        receipt["import"] = {"imported": len(imported["imported"]),
                             "already_imported": len(imported["skipped"]),
                             "errors": imported["errors"]}
        write_json(report_path, receipt)
        manifest = build_dataset(workspace, name)
        verify_dataset(dataset_path)
        config = {**DEFAULTS, "model_id": model_id or DEFAULTS["model_id"],
                  "model_revision": model_revision, "max_steps": max_steps,
                  "dataset": f"../datasets/{name}", "output_dir": f"../runs/{name}"}
        write_json(config_path, config)
        plan = training_plan(config_path)
        receipt.update(status="prepared_with_import_errors" if imported["errors"] else "prepared",
                       completed_at=now(), dataset_sha256=manifest["dataset_sha256"],
                       counts=manifest["counts"], excluded=manifest["excluded"],
                       mode=manifest["mode"], training_plan=plan, errors=imported["errors"])
        write_json(report_path, receipt)
        return receipt
    except Exception as exc:
        receipt.update(status="failed", ended_at=now(), error_type=type(exc).__name__,
                       error=str(exc)[:600])
        write_json(report_path, receipt)
        raise
