"""The first LanguageRig operator surface: import, snapshots, pilot and handoff."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .core import LanguageRigError, books, init_workspace, label_book
from .corpus import build_dataset, verify_dataset
from .doctor import doctor
from .evaluate import evaluate
from .fit import fit_probe
from .ingest import import_books
from .integrate import export_rag, merge_adapter, package_model, publish_rag, register_model
from .pilot import prepare_pilot
from .train import run_training


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LanguageRig — dansk modeltræning og bogimport")
    p.add_argument("--workspace", type=Path, default=Path("data"))
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    check = sub.add_parser("doctor")
    check.add_argument("--config", type=Path)
    check.add_argument("--gpu", type=int)
    check.add_argument("--require-training", action="store_true")
    check.add_argument("--report", type=Path)
    fit = sub.add_parser("fit-probe")
    fit.add_argument("config", type=Path)
    fit.add_argument("--execute", action="store_true",
                     help="Load the configured model and run one bounded training microstep")
    fit.add_argument("--report", type=Path)
    pilot = sub.add_parser("prepare-pilot")
    pilot.add_argument("source", type=Path)
    pilot.add_argument("--name", required=True)
    pilot.add_argument("--training-allowed", action="store_true")
    pilot.add_argument("--language")
    pilot.add_argument("--genre", choices=("unknown", "fiction", "nonfiction"), default="unknown")
    pilot.add_argument("--topic", action="append", default=[])
    pilot.add_argument("--model")
    pilot.add_argument("--revision", default="main")
    pilot.add_argument("--max-steps", type=int, default=100)
    imp = sub.add_parser("import")
    imp.add_argument("source", type=Path)
    imp.add_argument("--genre", choices=("unknown", "fiction", "nonfiction"), default="unknown")
    imp.add_argument("--language", help="Verified corpus language, e.g. da")
    imp.add_argument("--topic", action="append", default=[])
    imp.add_argument("--training-allowed", action="store_true", help="Select material you can use for training")
    sub.add_parser("inventory")
    label = sub.add_parser("label")
    label.add_argument("source_id")
    label.add_argument("--genre", choices=("unknown", "fiction", "nonfiction"))
    label.add_argument("--language")
    label.add_argument("--topic", action="append", dest="topics")
    label.add_argument("--work-id", help="Shared ID for editions/translations of one work")
    label.add_argument("--training-allowed", action=argparse.BooleanOptionalAction, default=None)
    label.add_argument("--accept-quality", action="store_true", default=None)
    ds = sub.add_parser("dataset")
    ds.add_argument("name")
    ds.add_argument("--seed", type=int, default=42)
    ds.add_argument("--validation-fraction", type=float, default=0.15)
    ds.add_argument("--test-fraction", type=float, default=0.15)
    ds.add_argument("--chunk-chars", type=int, default=6000)
    ds.add_argument("--genre", choices=("unknown", "fiction", "nonfiction"))
    ds.add_argument("--topic")
    ds.add_argument("--instructions", type=Path, help="Reviewed, source-bound conversational JSONL")
    verify = sub.add_parser("verify")
    verify.add_argument("dataset", type=Path)
    tr = sub.add_parser("train")
    tr.add_argument("config", type=Path)
    tr.add_argument("--execute", action="store_true", help="Download weights and start GPU training")
    tr.add_argument("--resume", type=Path)
    tr.add_argument("--fit-report", type=Path,
                    help="Successful fit-probe report matching this config and dataset")
    merge = sub.add_parser("merge")
    merge.add_argument("run", type=Path)
    merge.add_argument("output", type=Path)
    merge.add_argument("--execute", action="store_true")
    pack = sub.add_parser("package-model")
    pack.add_argument("run", type=Path)
    pack.add_argument("gguf", type=Path)
    pack.add_argument("--name", required=True)
    default_ollama = os.getenv("MODELRIG_OLLAMA_URL", "http://127.0.0.1:11434")
    reg = sub.add_parser("register-model")
    reg.add_argument("package", type=Path)
    reg.add_argument("--url", default=default_ollama)
    ev = sub.add_parser("evaluate")
    ev.add_argument("cases", type=Path)
    ev.add_argument("output", type=Path)
    ev.add_argument("--baseline", required=True)
    ev.add_argument("--candidate", required=True)
    ev.add_argument("--url", default=default_ollama)
    ev.add_argument("--dataset", type=Path)
    rag = sub.add_parser("export-rag")
    rag.add_argument("output", type=Path)
    pub = sub.add_parser("publish-rag")
    pub.add_argument("export", type=Path)
    pub.add_argument("--url", default="http://127.0.0.1:8080")
    pub.add_argument("--token-env", default="MODELRIG_DEVICE_TOKEN")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        match args.command:
            case "doctor":
                result = doctor(args.workspace, config=args.config, gpu=args.gpu,
                                require_training=args.require_training, report=args.report)
            case "fit-probe":
                result = fit_probe(args.config, execute=args.execute, report=args.report)
            case "prepare-pilot":
                result = prepare_pilot(args.workspace, args.source, args.name,
                                       training_allowed=args.training_allowed, language=args.language,
                                       genre=args.genre, topics=args.topic, model_id=args.model,
                                       model_revision=args.revision, max_steps=args.max_steps)
            case "init":
                init_workspace(args.workspace)
                result = {"initialized": str(args.workspace.resolve())}
            case "import":
                result = import_books(args.workspace, args.source, genre=args.genre, language=args.language,
                                      topics=args.topic, training_allowed=args.training_allowed)
            case "inventory":
                result = [{k: v for k, v in row.items() if k != "sections"} for row in books(args.workspace)]
            case "label":
                result = label_book(args.workspace, args.source_id, genre=args.genre, language=args.language,
                                    topics=args.topics, work_id=args.work_id, training_allowed=args.training_allowed,
                                    quality_accepted=args.accept_quality)
            case "dataset":
                result = build_dataset(args.workspace, args.name, seed=args.seed,
                                       validation_fraction=args.validation_fraction, test_fraction=args.test_fraction,
                                       chunk_chars=args.chunk_chars, genre=args.genre, topic=args.topic,
                                       instructions=args.instructions)
            case "verify":
                result = verify_dataset(args.dataset)
            case "train":
                if args.resume and not args.execute:
                    raise LanguageRigError("--resume requires --execute.")
                result = run_training(args.config, execute=args.execute, resume=args.resume,
                                      fit_report=args.fit_report)
            case "merge":
                result = merge_adapter(args.run, args.output, execute=args.execute)
            case "package-model":
                result = package_model(args.workspace, args.run, args.gguf, args.name)
            case "register-model":
                result = register_model(args.package, args.url)
            case "evaluate":
                result = evaluate(args.cases, args.output, url=args.url, baseline=args.baseline,
                                  candidate=args.candidate, dataset=args.dataset)
            case "export-rag":
                result = export_rag(args.workspace, args.output)
            case "publish-rag":
                result = publish_rag(args.export, url=args.url, token=os.getenv(args.token_env, ""))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if isinstance(result, dict):
            if result.get("errors"):
                return 1
            if result.get("training_gate") in ("review", "blocked"):
                return 1
        return 0
    except (LanguageRigError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"LanguageRig: {exc}", file=sys.stderr)
        return 2
