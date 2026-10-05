"""Bounded local environment checks; never load or download model weights."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement

from .core import LanguageRigError, now, write_json
from .train import training_plan

PROBE_MARKER = "LANGUAGERIG_GPU_PROBE="
PROBE_TIMEOUT = 45
GPU_PROBE = r'''
import json
result = {"status": "failed", "devices": [], "qlora_kernel_probe": "not_run"}
try:
    import torch
    result["torch_version"] = torch.__version__
    result["cuda_build"] = torch.version.cuda
    result["cuda_available"] = torch.cuda.is_available()
    result["visible_gpu_count"] = torch.cuda.device_count()
    if not result["cuda_available"]:
        raise RuntimeError("CUDA is not available in this Python environment.")
    if result["visible_gpu_count"] != 1:
        raise RuntimeError("Select exactly one GPU with CUDA_VISIBLE_DEVICES or --gpu.")
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.memory.mem_get_info(0)
    result["devices"] = [{"logical_index": 0, "name": props.name,
                          "compute_capability": [props.major, props.minor],
                          "total_memory_bytes": total, "free_memory_bytes": free}]
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
    from peft import LoraConfig, prepare_model_for_kbit_training
    import accelerate
    import datasets
    import bitsandbytes as bnb
    layer = bnb.nn.Linear4bit(32, 32, bias=False, compute_dtype=torch.float16,
                             quant_type="nf4", compress_statistics=True).to("cuda:0")
    with torch.no_grad():
        output = layer(torch.ones((1, 32), dtype=torch.float16, device="cuda:0"))
    torch.cuda.synchronize()
    if tuple(output.shape) != (1, 32) or not torch.isfinite(output).all().item():
        raise RuntimeError("The 4-bit CUDA probe did not produce finite output.")
    result.update(status="passed", qlora_kernel_probe="passed")
except Exception as exc:
    result.update(error_type=type(exc).__name__, error=str(exc)[:600])
print("LANGUAGERIG_GPU_PROBE=" + json.dumps(result, ensure_ascii=True, allow_nan=False))
'''


def dependency_status() -> list[dict]:
    """Use installed distribution requirements, including the training extra."""
    result = []
    for specification in importlib.metadata.requires("languagerig") or []:
        requirement = Requirement(specification)
        if requirement.marker and not requirement.marker.evaluate({"extra": "train"}):
            continue
        training_only = bool(requirement.marker and not requirement.marker.evaluate({"extra": ""}))
        try:
            installed = importlib.metadata.version(requirement.name)
            compatible = installed in requirement.specifier
        except importlib.metadata.PackageNotFoundError:
            installed, compatible = None, False
        result.append({"name": requirement.name, "required": str(requirement.specifier),
                       "installed": installed, "compatible": compatible,
                       "scope": "training" if training_only else "base"})
    if not any(row["name"] == "torch" for row in result):
        raise LanguageRigError("Training requirements are missing; reinstall LanguageRig.")
    return result


def gpu_probe(gpu: int | None = None) -> dict:
    environment = os.environ.copy()
    if gpu is not None:
        if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
            raise LanguageRigError("GPU index must be a nonnegative integer.")
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1",
                       PYTHONIOENCODING="utf-8")
    try:
        completed = subprocess.run(
            [sys.executable, "-c", GPU_PROBE], env=environment,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=PROBE_TIMEOUT, check=False)
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error_type": "TimeoutExpired",
                "error": "CUDA/dependency probe exceeded 45 seconds."}
    except OSError as exc:
        return {"status": "failed", "error_type": type(exc).__name__,
                "error": "Cannot start the CUDA/dependency probe."}
    if completed.returncode:
        return {"status": "failed", "error_type": "ProbeProcessFailed",
                "returncode": completed.returncode}
    try:
        payloads = [line[len(PROBE_MARKER):] for line in completed.stdout.splitlines()
                    if line.startswith(PROBE_MARKER)]
        if len(payloads) != 1:
            raise ValueError("missing or multiple probe results")
        result = json.loads(payloads[0])
        if not isinstance(result, dict) or result.get("status") not in ("passed", "failed"):
            raise ValueError("invalid probe status")
        if result["status"] == "passed":
            devices = result.get("devices")
            if (result.get("cuda_available") is not True or result.get("visible_gpu_count") != 1
                    or result.get("qlora_kernel_probe") != "passed"
                    or not isinstance(devices, list) or len(devices) != 1):
                raise ValueError("incomplete successful probe")
            for key in ("total_memory_bytes", "free_memory_bytes"):
                if type(devices[0].get(key)) is not int or devices[0][key] < 0:
                    raise ValueError("invalid memory measurement")
            if not 0 < devices[0]["total_memory_bytes"] >= devices[0]["free_memory_bytes"]:
                raise ValueError("invalid free/total memory measurement")
        return result
    except (ValueError, TypeError, KeyError, AttributeError):
        return {"status": "failed", "error_type": "InvalidProbeResult"}


def doctor(workspace: Path, *, config: Path | None = None, gpu: int | None = None,
           require_training=False, report: Path | None = None) -> dict:
    if gpu is not None and (type(gpu) is not int or gpu < 0):
        raise LanguageRigError("GPU index must be a nonnegative integer.")
    dependencies = dependency_status()
    base_errors, blockers = [], []
    if sys.version_info < (3, 10):
        base_errors.append("python_below_3_10")
    for row in dependencies:
        if not row["compatible"]:
            (blockers if row["scope"] == "training" else base_errors).append(
                "dependency_missing_or_incompatible:" + row["name"])
    if os.getenv("WORLD_SIZE", "1") != "1":
        blockers.append("distributed_training_not_supported")
    plan = None
    if config:
        try:
            plan = training_plan(config)
        except (OSError, ValueError) as exc:
            base_errors.append("invalid_training_config_or_dataset")
            plan = {"status": "failed", "error": str(exc)[:600]}
    probe = {"status": "not_run", "reason": "training_dependencies_not_ready"}
    if not base_errors and not blockers:
        probe = gpu_probe(gpu)
        if probe["status"] != "passed":
            blockers.append("cuda_or_4bit_probe_failed")
    target = workspace.resolve()
    existing = target
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    if not existing.is_dir():
        base_errors.append("workspace_parent_not_directory")
    storage = {"existing_parent": str(existing), "writable": os.access(existing, os.W_OK)}
    try:
        usage = shutil.disk_usage(existing)
        storage.update(total_bytes=usage.total, free_bytes=usage.free)
    except OSError:
        base_errors.append("storage_measurement_failed")
    if not storage["writable"]:
        base_errors.append("workspace_parent_not_writable")
    ready = not base_errors and not blockers and probe["status"] == "passed"
    result = {"format": "languagerig-doctor/v1", "checked_at": now(),
              "status": "blocked" if base_errors or (require_training and not ready) else "checked",
              "python": {"executable": sys.executable, "version": platform.python_version()},
              "platform": {"system": platform.system(), "release": platform.release(),
                           "wsl": bool(os.getenv("WSL_DISTRO_NAME") or
                                       re.search("microsoft", platform.release(), re.I))},
              "workspace": str(target), "storage": storage, "dependencies": dependencies,
              "gpu_selection": str(gpu) if gpu is not None else os.getenv("CUDA_VISIBLE_DEVICES"),
              "gpu_probe": probe, "training_environment_ready": ready,
              "training_blockers": blockers, "errors": base_errors + (blockers if require_training else []),
              "model_downloaded": False, "training_executed": False,
              "model_fit": "not_measured", "model_quality": "not_measured"}
    if plan:
        result["training_plan"] = plan
    if report:
        write_json(report, result)
    return result
