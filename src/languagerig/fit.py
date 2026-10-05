"""Bounded one-microstep model-fit probe for the configured QLoRA candidate."""
from __future__ import annotations

import importlib.metadata
import math
import os
from pathlib import Path

from .core import LanguageRigError, now, write_json
from .train import training_plan


def fit_probe(config_path: Path, *, execute: bool = False, report: Path | None = None) -> dict:
    """Measure whether one real QLoRA microstep fits; never save an adapter."""
    plan = training_plan(config_path)
    result = {
        "format": "languagerig-fit-probe/v1",
        "status": "planned" if not execute else "running",
        "checked_at": now(),
        "config_sha256": plan["config_sha256"],
        "dataset_sha256": plan["dataset_sha256"],
        "model_id": plan["config"]["model_id"],
        "model_revision": plan["config"]["model_revision"],
        "max_seq_length": plan["config"]["max_seq_length"],
        "batch_size": 1,
        "gradient_accumulation_steps": plan["config"]["gradient_accumulation_steps"],
        "training_executed": False,
        "adapter_saved": False,
        "model_fit": "not_measured",
        "model_quality": "not_measured",
        "model_downloaded": False if not execute else None,
    }
    if report:
        write_json(report, result)
    if not execute:
        return result

    missing = [name for name, available in plan["dependencies"].items() if not available]
    if missing:
        raise LanguageRigError("Install languagerig[train]; missing: " + ", ".join(missing))
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise LanguageRigError("Fit probe supports one GPU/process only.")

    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise LanguageRigError("Select exactly one CUDA GPU before the fit probe.")

        config = plan["config"]
        resolved = AutoConfig.from_pretrained(
            config["model_id"], revision=config["model_revision"], trust_remote_code=False)
        revision = getattr(resolved, "_commit_hash", None)
        if not revision:
            raise LanguageRigError("Cannot pin model revision; use a Hugging Face model repository.")

        tokenizer = AutoTokenizer.from_pretrained(
            config["model_id"], revision=revision, trust_remote_code=False)
        if tokenizer.eos_token_id is None:
            raise LanguageRigError("Tokenizer needs an EOS token.")

        result.update(resolved_revision=revision, model_load_status="loading",
                      gpu=torch.cuda.get_device_name(0),
                      runtime_versions={name: importlib.metadata.version(name) for name in
                                        ("torch", "transformers", "peft", "accelerate",
                                         "bitsandbytes", "datasets")})
        if report:
            write_json(report, result)

        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"], revision=revision, trust_remote_code=False,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16),
            dtype=torch.float16, device_map={"": 0})
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(
            r=config["lora_rank"], lora_alpha=config["lora_rank"] * 2,
            lora_dropout=0.05, target_modules="all-linear",
            task_type="CAUSAL_LM", bias="none"))
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})

        seed_ids = tokenizer(
            "Dansk modeltræning med LanguageRig. ",
            add_special_tokens=False)["input_ids"]
        if not seed_ids:
            raise LanguageRigError("Tokenizer produced no tokens for the fit probe.")
        length = config["max_seq_length"]
        repeats = math.ceil(length / len(seed_ids))
        ids = (list(seed_ids) * repeats)[:length]
        if ids:
            ids[-1] = tokenizer.eos_token_id

        input_ids = torch.tensor([ids], dtype=torch.long, device="cuda:0")
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise LanguageRigError("LoRA preparation produced no trainable parameters.")
        optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"])

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        before_free, total = torch.cuda.mem_get_info(0)
        model.train()
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        if not torch.isfinite(loss).item():
            raise LanguageRigError("Fit probe produced a non-finite loss.")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(0)
        after_free, _ = torch.cuda.mem_get_info(0)

        result.update(
            status="passed", completed_at=now(), model_load_status="loaded",
            model_fit="single_full_sequence_microstep_passed",
            measured_sequence_tokens=len(ids),
            loss=float(loss.detach().cpu()),
            vram={
                "total_bytes": int(total),
                "free_before_step_bytes": int(before_free),
                "free_after_step_bytes": int(after_free),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            },
            note=("This proves one batch=1 forward/backward/AdamW step at the configured "
                  "sequence length. It does not prove long-run stability or model quality."),
        )
    except BaseException as exc:
        result.update(status="failed", ended_at=now(), model_fit="failed",
                      error_type=type(exc).__name__, error=str(exc)[:600],
                      errors=["fit_probe_failed"])
    finally:
        if report:
            write_json(report, result)
    return result
