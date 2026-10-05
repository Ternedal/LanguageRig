#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: train-pilot-wsl.sh CONFIG GPU check|execute PYTHON RESUME|-" >&2
  exit 2
fi
config_path=$1
gpu_index=$2
mode=$3
python_exe=$4
resume_path=$5
if [[ ! $gpu_index =~ ^[0-9]+$ || ( $mode != check && $mode != execute ) ]]; then
  echo "Invalid GPU index or mode." >&2
  exit 2
fi
if [[ $mode == check && $resume_path != - ]]; then
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
workspace_path=$("$python_exe" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve().parent.parent)' "$config_path")
report_path="$workspace_path/checks/train-doctor.json"
"$python_exe" -m languagerig --workspace "$workspace_path" doctor \
  --config "$config_path" --require-training --report "$report_path"
if [[ $mode == execute ]]; then
  if [[ $resume_path == - ]]; then
    "$python_exe" -m languagerig train "$config_path" --execute
  else
    "$python_exe" -m languagerig train "$config_path" --execute --resume "$resume_path"
  fi
fi
