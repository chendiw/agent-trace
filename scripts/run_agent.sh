#!/usr/bin/env bash
# run_agent.sh — run mini-swe-agent against the local SGLang endpoint.
#
# Usage:
#   bash run_agent.sh "Fix the off-by-one error in utils.py"
#   bash run_agent.sh --7b "Fix the off-by-one error in utils.py"
#
# Prerequisites on the pod:
#   - SGLang running:  see sglang_endpoint.yaml launch_cmd_smoke / launch_cmd_real
#   - mini-swe-agent:  git clone + pip install -e . at /root/mini-swe-agent
#   - /root/.config/mini-swe-agent/.env contains MSWEA_CONFIGURED=1

set -euo pipefail

MINI_SWE_AGENT_DIR="/root/mini-swe-agent"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$(dirname "$SCRIPT_DIR")/mini_swe_agent"

MODEL_CONFIG="$CONFIG_DIR/local.yaml"
if [[ "${1:-}" == "--7b" ]]; then
    MODEL_CONFIG="$CONFIG_DIR/local_7b.yaml"
    shift
fi

TASK="${1:-List the files in /root and exit.}"

mini-swe-agent \
    -c "$MINI_SWE_AGENT_DIR/src/minisweagent/config/mini.yaml" \
    -c "$MODEL_CONFIG" \
    --task "$TASK" \
    --yolo
