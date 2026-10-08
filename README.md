# Laya on an Azure CPU VM

Ollaya serves Laya on `127.0.0.1:11435`. Streamlit runs on `127.0.0.1:8501`. Both stay on the VM. Your laptop reaches them through SSH.

## 1. Set up the VM

From this folder on the VM (Remote-SSH or a normal SSH session):

```bash
chmod +x setup.sh
./setup.sh
```

The script updates apt, installs Python 3, pip, and virtualenv, installs Ollaya from `https://ollaya.dev/install.sh`, enables the `ollaya` systemd service on CPU, pulls `laya`, and creates `.venv` with Streamlit and requests.

The site root (`https://ollaya.dev`) is HTML. The installer that the quickstart documents is `/install.sh`.

## 2. Tunnel the ports

On your laptop, leave this session open:

```bash
ssh -L 11435:localhost:11435 -L 8501:localhost:8501 username@<IP>
```

`localhost:11435` on the laptop is the Ollaya API. `localhost:8501` is the UI.

## 3. Open the UI

On the VM, in this folder:

```bash
source .venv/bin/activate
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

In the laptop browser, open [http://localhost:8501](http://localhost:8501).

The first Analyze can take a while while Laya loads into RAM. Later calls stay warm for the service keep-alive (default 5 minutes).

## Request shape

`POST http://localhost:11435/api/decide` with:

```json
{
  "model": "laya",
  "state": "<input text>",
  "questions": { }
}
```

Ollaya does not accept `POST /api/decision`, and it ignores unknown fields such as `preset` and `text`. A decision needs `state` plus `questions`, unless the model was created with questions baked in.

The dropdown uses the built-in CLI presets (`triage`, `email`, `guard`, `moderation`, `router`). The app expands each one into the `questions` object those presets ask. `custom` lets you send your own questions JSON.

Docs: [API reference](https://ollaya.dev/docs/api) and [quickstart](https://ollaya.dev/docs/quickstart).
