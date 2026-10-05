#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: train-pilot-wsl.sh CONFIG GPU check|fit|execute PYTHON RESUME|-" >&2
  exit 2
fi
config_path=$1
gpu_index=$2
mode=$3
python_exe=$4
resume_path=$5
if [[ ! $gpu_index =~ ^[0-9]+$ || ( $mode != check && $mode != fit && $mode != execute ) ]]; then
  echo "Invalid GPU index or mode." >&2
  exit 2
fi
if [[ $mode != execute && $resume_path != - ]]; then
  echo "Resume requires execute mode." >&2
  exit 2
fi
if [[ ! -f $config_path || ! -x $python_exe ]]; then
  echo "Config/Python missing. Install the WSL training environment first." >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$gpu_index"
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export PYTHONIOENCODING=utf-8
workspace_path=$("$python_exe" -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); c=json.loads(p.read_text(encoding="utf-8")); print((p.parent/c["output_dir"]).resolve().parent.parent)' "$config_path")
report_path="$workspace_path/checks/train-doctor.json"
"$python_exe" -m languagerig --workspace "$workspace_path" doctor \
  --config "$config_path" --require-training --report "$report_path"
if [[ $mode == fit ]]; then
  "$python_exe" -m languagerig fit-probe "$config_path" --execute \
    --report "$workspace_path/checks/fit-probe.json"
elif [[ $mode == execute ]]; then
  fit_report="$workspace_path/checks/fit-probe.json"
  if [[ ! -f $fit_report ]]; then
    echo "Training blocked: run -FitProbe first; fit-probe report is missing." >&2
    exit 3
  fi
  if [[ $resume_path == - ]]; then
    "$python_exe" -m languagerig train "$config_path" --execute --fit-report "$fit_report"
  else
    "$python_exe" -m languagerig train "$config_path" --execute --fit-report "$fit_report" --resume "$resume_path"
  fi
fi
