from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from languagerig.cli import main
from languagerig.core import LanguageRigError, books, read_json, write_json
from languagerig.corpus import verify_dataset
from languagerig.doctor import PROBE_MARKER, dependency_status, doctor, gpu_probe
from languagerig.ingest import import_books
from languagerig.fit import fit_probe
from languagerig.pilot import prepare_pilot
from languagerig.train import training_plan, verify_fit_gate
from test_languagerig import WorkspaceCase, epub, pdf


def compatible_dependencies():
    return [{"name": "pypdf", "scope": "base", "compatible": True},
            {"name": "torch", "scope": "training", "compatible": True}]


def passed_probe():
    return {"status": "passed", "cuda_available": True, "visible_gpu_count": 1,
            "qlora_kernel_probe": "passed",
            "devices": [{"name": "fixture GPU", "total_memory_bytes": 12000,
                         "free_memory_bytes": 9000}]}


class DoctorTests(WorkspaceCase):
    def test_declared_versions_accept_cuda_build_and_reject_old_peft(self):
        requirements = ["pypdf>=5,<7", 'torch>=2.4; extra == "train"',
                        'peft>=0.18,<1; extra == "train"']
        versions = {"pypdf": "6.19.0", "torch": "2.14.0+cu130", "peft": "0.17.0"}
        with patch("languagerig.doctor.importlib.metadata.requires", return_value=requirements), \
                patch("languagerig.doctor.importlib.metadata.version", side_effect=versions.__getitem__):
            result = dependency_status()
        self.assertTrue(result[1]["compatible"])
        self.assertFalse(result[2]["compatible"])
        self.assertEqual(result[2]["scope"], "training")

    def test_missing_gpu_dependencies_are_recorded_without_probe(self):
        missing = compatible_dependencies()
        missing[1]["compatible"] = False
        report = self.root / "check.json"
        with patch("languagerig.doctor.dependency_status", return_value=missing), \
                patch("languagerig.doctor.gpu_probe") as probe:
            result = doctor(self.workspace, require_training=True, report=report)
        probe.assert_not_called()
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["training_environment_ready"])
        self.assertFalse(result["model_downloaded"])
        self.assertEqual(read_json(report), result)

    def test_successful_environment_never_claims_model_fit_or_quality(self):
        with patch("languagerig.doctor.dependency_status", return_value=compatible_dependencies()), \
                patch("languagerig.doctor.gpu_probe", return_value=passed_probe()):
            result = doctor(self.workspace, gpu=1, require_training=True)
        self.assertTrue(result["training_environment_ready"])
        self.assertEqual(result["model_fit"], "not_measured")
        self.assertEqual(result["model_quality"], "not_measured")
        self.assertFalse(result["training_executed"])

    def test_invalid_config_and_multi_process_never_get_green(self):
        config = self.config()
        (self.workspace / "datasets/pilot/train.jsonl").write_text("{}", encoding="utf-8")
        with patch("languagerig.doctor.dependency_status", return_value=compatible_dependencies()), \
                patch("languagerig.doctor.gpu_probe") as probe, \
                patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            result = doctor(self.workspace, config=config, require_training=True)
        probe.assert_not_called()
        self.assertIn("invalid_training_config_or_dataset", result["errors"])
        self.assertIn("distributed_training_not_supported", result["errors"])

    def test_child_process_uses_offline_environment_and_one_gpu(self):
        completed = subprocess.CompletedProcess([], 0, "warning\n" + PROBE_MARKER + json.dumps(passed_probe()), "")
        with patch("languagerig.doctor.subprocess.run", return_value=completed) as run:
            result = gpu_probe(1)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(run.call_args.kwargs["env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(run.call_args.kwargs["timeout"], 45)
        self.assertEqual(run.call_args.args[0][0], sys.executable)

    def test_timeout_exit_and_incomplete_result_are_failures(self):
        bad = passed_probe()
        bad["visible_gpu_count"] = 2
        for completed in (subprocess.CompletedProcess([], 3, "", ""),
                          subprocess.CompletedProcess([], 0, "not json", ""),
                          subprocess.CompletedProcess([], 0, PROBE_MARKER + json.dumps(bad), ""),
                          subprocess.CompletedProcess([], 0, PROBE_MARKER + '{"status":"passed"}', "")):
            with patch("languagerig.doctor.subprocess.run", return_value=completed):
                self.assertEqual(gpu_probe()["status"], "failed")
        with patch("languagerig.doctor.subprocess.run", side_effect=subprocess.TimeoutExpired("probe", 45)):
            self.assertEqual(gpu_probe()["error_type"], "TimeoutExpired")

    def test_invalid_gpu_rejected_even_without_dependencies(self):
        for value in (-1, True, "0,1"):
            with self.assertRaises(LanguageRigError):
                doctor(self.workspace, gpu=value)

    def test_cli_training_requirement_exits_nonzero_and_keeps_report(self):
        missing = compatible_dependencies()
        missing[1]["compatible"] = False
        report = self.root / "cli-check.json"
        with patch("languagerig.doctor.dependency_status", return_value=missing), \
                contextlib.redirect_stdout(io.StringIO()):
            code = main(["--workspace", str(self.workspace), "doctor", "--require-training",
                         "--report", str(report)])
        self.assertEqual(code, 1)
        self.assertEqual(read_json(report)["status"], "blocked")


class FitProbeTests(WorkspaceCase):
    def test_plan_mode_never_loads_model_or_trains(self):
        config = self.config()
        result = fit_probe(config)
        self.assertEqual(result["status"], "planned")
        self.assertEqual(result["model_fit"], "not_measured")
        self.assertFalse(result["training_executed"])
        self.assertFalse(result["adapter_saved"])
        self.assertFalse(result["model_downloaded"])

    def test_cli_plan_mode_is_safe_without_training_dependencies(self):
        config = self.config()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(["fit-probe", str(config)])
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "planned")
        self.assertFalse(result["training_executed"])


class TrainingGateTests(WorkspaceCase):
    def fit_report(self, config, *, status="passed", gate="pass", config_hash=None,
                   dataset_hash=None):
        plan = training_plan(config)
        path = self.root / "fit-probe.json"
        write_json(path, {
            "format": "languagerig-fit-probe/v1",
            "status": status,
            "training_gate": gate,
            "config_sha256": config_hash or plan["config_sha256"],
            "dataset_sha256": dataset_hash or plan["dataset_sha256"],
            "vram": {"minimum_observed_free_bytes": 1024 * 1024 * 1024},
        })
        return path

    def test_training_gate_accepts_only_matching_passed_probe(self):
        config = self.config()
        report = self.fit_report(config)
        result = verify_fit_gate(config, report)
        self.assertEqual(result["training_gate"], "pass")

        for kwargs in (
            {"status": "failed", "gate": "blocked"},
            {"status": "passed", "gate": "review"},
            {"config_hash": "stale-config"},
            {"dataset_hash": "stale-dataset"},
        ):
            report = self.fit_report(config, **kwargs)
            with self.assertRaises(LanguageRigError):
                verify_fit_gate(config, report)

    def test_training_gate_requires_report_file(self):
        config = self.config()
        with self.assertRaises(LanguageRigError):
            verify_fit_gate(config, self.root / "missing.json")


class PilotTests(WorkspaceCase):
    def fixture_books(self, count=3):
        for number in range(count):
            epub(self.input / f"{number}.epub", f"Værk {number}")

    def test_mixed_pilot_builds_verified_relative_config_without_gpu(self):
        self.fixture_books()
        pdf(self.input / "fagbog.pdf", ["A PDF chapter with text. " * 15] * 3)
        with patch("languagerig.train.importlib.util.find_spec", return_value=None):
            result = prepare_pilot(self.workspace, self.input, "mixed", training_allowed=True, language="da")
        config = read_json(Path(result["config_path"]))
        self.assertEqual(config["dataset"], "../datasets/mixed")
        self.assertEqual(result["import"]["imported"], 4)
        self.assertEqual(result["status"], "prepared")
        self.assertEqual(verify_dataset(Path(result["dataset_path"]))["dataset_sha256"], result["dataset_sha256"])
        self.assertFalse(result["model_downloaded"])
        self.assertFalse((self.workspace / "runs/mixed").exists())

    def test_selection_names_and_steps_fail_before_import(self):
        self.fixture_books()
        for name, options in (("../escape", {"training_allowed": True}),
                              ("pilot", {}), ("pilot", {"training_allowed": True, "max_steps": 0})):
            with self.assertRaises(LanguageRigError):
                prepare_pilot(self.workspace, self.input, name, **options)
        self.assertEqual(books(self.workspace), [])
        self.assertFalse((self.workspace / "pilots").exists())

    def test_reused_name_preserves_config_report_and_snapshot(self):
        self.fixture_books()
        result = prepare_pilot(self.workspace, self.input, "one", training_allowed=True)
        original = Path(result["report_path"]).read_bytes()
        with self.assertRaises(LanguageRigError):
            prepare_pilot(self.workspace, self.input, "one", training_allowed=True)
        self.assertEqual(Path(result["report_path"]).read_bytes(), original)

    def test_insufficient_books_have_failed_receipt_without_config(self):
        self.fixture_books(2)
        with self.assertRaises(LanguageRigError):
            prepare_pilot(self.workspace, self.input, "small", training_allowed=True)
        self.assertEqual(read_json(self.workspace / "pilots/small.json")["status"], "failed")
        self.assertFalse((self.workspace / "configs/small.json").exists())

    def test_partial_import_prepares_valid_books_but_cli_is_not_green(self):
        self.fixture_books()
        (self.input / "bad.pdf").write_bytes(b"invalid PDF")
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["--workspace", str(self.workspace), "prepare-pilot", str(self.input),
                         "--name", "partial", "--training-allowed"])
        self.assertEqual(code, 1)
        result = read_json(self.workspace / "pilots/partial.json")
        self.assertEqual(result["status"], "prepared_with_import_errors")
        self.assertEqual(len(result["errors"]), 1)
        self.assertTrue((self.workspace / "configs/partial.json").is_file())

    def test_existing_inventory_is_reused_without_duplicate_books(self):
        self.fixture_books()
        import_books(self.workspace, self.input, training_allowed=True)
        result = prepare_pilot(self.workspace, self.input, "new", training_allowed=True)
        self.assertEqual(result["import"]["already_imported"], 3)
        self.assertEqual(len(books(self.workspace)), 3)


@unittest.skipIf(sys.platform == "win32" or not shutil.which("bash"), "Linux/bash transport check")
class ShellLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "folder with spaces ' $ ;"
        self.root.mkdir()
        self.config = self.root / "configs" / "pilot.json"
        self.config.parent.mkdir()
        self.config.write_text("{}", encoding="utf-8")
        self.log = self.root / "calls.jsonl"
        (self.root / "checks").mkdir()
        (self.root / "checks/fit-probe.json").write_text("{}", encoding="utf-8")
        self.launcher = self.root / "fake python"
        self.launcher.write_text("#!" + sys.executable + "\n" + r'''
import json,os,pathlib,sys
if sys.argv[1] == "-c":
    print(pathlib.Path(sys.argv[3]).resolve().parent.parent)
else:
    with open(os.environ["STUB_LOG"], "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"args":sys.argv[1:],"gpu":os.environ.get("CUDA_VISIBLE_DEVICES")})+"\n")
    if "doctor" in sys.argv and os.environ.get("STUB_BLOCK") == "1":
        sys.exit(7)
''', encoding="utf-8")
        self.launcher.chmod(0o755)
        self.script = Path(__file__).resolve().parents[1] / "scripts/train-pilot-wsl.sh"

    def tearDown(self):
        self.temp.cleanup()

    def run_launcher(self, mode="check", *, blocked=False, resume="-", gpu="1"):
        environment = {**os.environ, "STUB_LOG": str(self.log), "STUB_BLOCK": "1" if blocked else "0"}
        completed = subprocess.run(["bash", str(self.script), str(self.config), gpu, mode,
                                    str(self.launcher), resume], env=environment,
                                   capture_output=True, text=True, timeout=10)
        rows = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return completed, rows

    def test_check_mode_never_starts_training(self):
        completed, rows = self.run_launcher()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(rows), 1)
        self.assertIn("doctor", rows[0]["args"])
        self.assertEqual(rows[0]["gpu"], "1")
        self.assertIn(str(self.config), rows[0]["args"])

    def test_failed_probe_stops_execute(self):
        completed, rows = self.run_launcher("execute", blocked=True)
        self.assertEqual(completed.returncode, 7)
        self.assertEqual(len(rows), 1)

    def test_fit_mode_runs_doctor_then_bounded_probe(self):
        completed, rows = self.run_launcher("fit")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(rows), 2)
        self.assertIn("doctor", rows[0]["args"])
        self.assertIn("fit-probe", rows[1]["args"])
        self.assertIn("--execute", rows[1]["args"])
        self.assertEqual(rows[1]["gpu"], "1")

    def test_execute_and_resume_preserve_argument_boundaries(self):
        resume = str(self.root / "checkpoint ' $ ; 20")
        completed, rows = self.run_launcher("execute", resume=resume)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(rows), 2)
        self.assertIn("train", rows[1]["args"])
        self.assertEqual(rows[1]["args"][-2:], ["--resume", resume])

    def test_multiple_gpus_and_resume_check_fail_before_python(self):
        for options in ({"gpu": "0,1"}, {"resume": "checkpoint"}, {"mode": "fit", "resume": "checkpoint"}):
            completed, rows = self.run_launcher(**options)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(rows, [])

    def test_real_blocked_cli_writes_report_to_configured_workspace(self):
        source = self.root / "books"
        source.mkdir()
        for number in range(3):
            epub(source / f"{number}.epub", f"Bog {number}")
        workspace = self.root / "actual workspace"
        pilot = prepare_pilot(workspace, source, "real", training_allowed=True)
        external = self.root / "elsewhere" / "external.json"
        configuration = read_json(Path(pilot["config_path"]))
        configuration.update(dataset="../actual workspace/datasets/real",
                             output_dir="../actual workspace/runs/real")
        write_json(external, configuration)
        completed = subprocess.run(
            ["bash", str(self.script), str(external), "0", "execute", sys.executable, "-"],
            env={**os.environ, "WORLD_SIZE": "2"}, capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 1, completed.stderr)
        report = read_json(workspace / "checks/train-doctor.json")
        self.assertIn("distributed_training_not_supported", report["errors"])
        self.assertFalse(report["training_executed"])
        self.assertFalse((workspace / "runs/real").exists())
        self.assertFalse((self.root / "checks/train-doctor.json").exists())
