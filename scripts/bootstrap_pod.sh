#!/usr/bin/env bash
# bootstrap_pod.sh — bring a fresh RunPod pytorch pod to full Phase 0 working state.
#
# Run once after SSH-ing into a new pod:
#   bash bootstrap_pod.sh
#
# Prerequisites (set at pod creation time in the console):
#   - Image: runpod/pytorch:2.8.0-py3.12-cuda12.8.1-devel-ubuntu24.04 (or similar)
#   - Network volume samw2fy7d8 selected; it mounts at /workspace on Community Cloud
#   - GPU: RTX 4090
#
# After running, symlink /workspace → /root/.cache/huggingface/hub is created so
# the 7B model weights survive pod termination on the network volume.

set -euo pipefail

echo "=== 0. Symlink network volume as HF cache ==="
ln -sfn /workspace /root/.cache/huggingface/hub

echo "=== 1. Install SGLang ==="
# --break-system-packages required on Ubuntu 24.04 / Python 3.12
pip install --break-system-packages --quiet "sglang[all]"

echo "=== 2. Clone agent-trace repo ==="
git clone --depth 1 https://github.com/chendiw/agent-trace.git /root/agent-trace

echo "=== 3. Clone and install mini-swe-agent ==="
git clone --depth 1 https://github.com/SWE-agent/mini-swe-agent.git /root/mini-swe-agent
pip install --break-system-packages --quiet -e /root/mini-swe-agent

echo "=== 4. Write global mini-swe-agent config ==="
mkdir -p /root/.config/mini-swe-agent
cat > /root/.config/mini-swe-agent/.env << 'EOF'
OPENAI_API_BASE=http://localhost:30000/v1
OPENAI_API_KEY=none
MSWEA_CONFIGURED=1
EOF

echo "=== 5. Start SGLang 7B server (downloads to network volume if not cached) ==="
nohup python3 -m sglang.launch_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 30000 --host 0.0.0.0 \
    --enable-metrics \
    --max-total-tokens 20000 \
    --attention-backend triton \
    --sampling-backend pytorch \
    --tool-call-parser qwen \
    > /root/sglang_7b.log 2>&1 &
echo "SGLang pid: $!"

echo ""
echo "=== Bootstrap complete ==="
echo "SGLang is downloading/loading the 7B model in the background."
echo "Watch progress: tail -f /root/sglang_7b.log"
echo "Check ready:    curl http://localhost:30000/health"
echo ""
echo "Run agent:  bash /root/agent-trace/scripts/run_agent.sh --7b 'your task'"
