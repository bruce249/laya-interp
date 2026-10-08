#!/usr/bin/env bash
# Prepare an Ubuntu VM for Laya via Ollaya (CPU) and the Streamlit UI.
set -euo pipefail

cd "$(dirname "$0")"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Run setup.sh on the Ubuntu VM." >&2
  exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo is required." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

echo "Updating packages..."
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv python3-virtualenv curl ca-certificates

echo "Installing Ollaya..."
# https://ollaya.dev/ is the site. The installer is /install.sh (quickstart).
curl -fsSL https://ollaya.dev/install.sh | sh

echo "Pinning the service to CPU..."
sudo mkdir -p /etc/systemd/system/ollaya.service.d
sudo tee /etc/systemd/system/ollaya.service.d/cpu.conf >/dev/null <<'EOF'
[Service]
Environment="OLLAYA_DEVICE=cpu"
EOF

echo "Enabling and starting ollaya..."
sudo systemctl daemon-reload
sudo systemctl enable ollaya
sudo systemctl restart ollaya

echo "Waiting for http://127.0.0.1:11435/ ..."
ready=0
for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:11435/ 2>/dev/null | grep -q "Ollaya is running"; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  echo "Ollaya did not answer. Check: journalctl -u ollaya" >&2
  exit 1
fi

echo "Pulling laya (router plus laya:en and laya:multilingual)..."
ollaya pull laya

echo "Creating .venv and installing streamlit and requests..."
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install streamlit requests

echo "Ready. Next: source .venv/bin/activate && streamlit run app.py --server.address 127.0.0.1 --server.port 8501"
