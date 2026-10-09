"""Laya interpretability lab on Modal (see INTERP_PLAN.md).

Phase 0:
  modal run modal_interp.py::inspect_pkg
  modal run modal_interp.py::show_source --paths "laya/router.py,laya/model.py"
"""

from __future__ import annotations

import modal

base_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "laya",
    "einops",
    "nnsight",
    "datasets",
    "scikit-learn",
    "matplotlib",
    "pandas",
    "openai",
    "fastapi[standard]",
    "sentencepiece",
)
image = base_image.add_local_python_source("interp")

weights = modal.Volume.from_name("laya-weights", create_if_missing=True)
store = modal.Volume.from_name("laya-interp", create_if_missing=True)
VOLUMES = {"/weights": weights, "/store": store}
HF_ENV = {"HF_HOME": "/weights/hf", "HUGGINGFACE_HUB_CACHE": "/weights/hf", "USE_TF": "0"}

app = modal.App("laya-interp", image=image)


@app.function(cpu=4, memory=16384, volumes=VOLUMES, timeout=1800, env=HF_ENV)
def inspect_pkg() -> str:
    import importlib.metadata as md
    import pathlib

    import laya

    root = pathlib.Path(laya.__file__).parent
    lines = [f"laya {md.version('laya')} at {root}"]
    for name in ("torch", "transformers", "huggingface_hub", "nnsight"):
        try:
            lines.append(f"{name} {md.version(name)}")
        except md.PackageNotFoundError:
            lines.append(f"{name} missing")
    lines.append("files:")
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            lines.append(f"  {path.relative_to(root.parent)}  {path.stat().st_size}")
    lines.append("public names: " + ", ".join(n for n in dir(laya) if not n.startswith("_")))
    return "\n".join(lines)


@app.function(cpu=2, memory=4096, timeout=600, env=HF_ENV)
def show_source(paths: str) -> str:
    import pathlib

    import laya

    root = pathlib.Path(laya.__file__).parent.parent
    out = []
    for rel in paths.split(","):
        file = root / rel.strip()
        out.append(f"===== {rel.strip()} =====")
        out.append(file.read_text(errors="replace") if file.exists() else "(missing)")
    return "\n".join(out)


@app.function(cpu=2, memory=4096, timeout=600, env=HF_ENV)
def grep_source(pattern: str, paths: str = "laya") -> str:
    import pathlib
    import re

    import laya

    root = pathlib.Path(laya.__file__).parent.parent
    regex = re.compile(pattern)
    out = []
    for rel in paths.split(","):
        target = root / rel.strip()
        files = [target] if target.is_file() else sorted(target.rglob("*.py"))
        for file in files:
            for number, line in enumerate(file.read_text(errors="replace").splitlines(), 1):
                if regex.search(line):
                    out.append(f"{file.relative_to(root)}:{number}: {line.rstrip()}")
    return "\n".join(out)


@app.function(cpu=2, memory=4096, timeout=600, env=HF_ENV)
def show_lines(path: str, start: int, end: int) -> str:
    import pathlib

    import laya

    file = pathlib.Path(laya.__file__).parent.parent / path
    lines = file.read_text(errors="replace").splitlines()
    return "\n".join(f"{n}: {lines[n - 1]}" for n in range(start, min(end, len(lines)) + 1))


@app.function(cpu=2, memory=8192, volumes=VOLUMES, timeout=1200, env=HF_ENV,
              secrets=[modal.Secret.from_dict({})])
def hub_facts(repos: str) -> str:
    import json

    from huggingface_hub import HfApi, hf_hub_download

    import laya
    from laya import common

    out = [f"QTYPES={laya.QTYPES} DEFAULT_MODELS={laya.DEFAULT_MODELS}"]
    out.append(f"PINNED_REVISIONS={laya.PINNED_REVISIONS}")
    out.append(f"noul labels default={getattr(common, '_DEFAULT_NOUL_LABELS', None)}")
    api = HfApi()
    for repo in repos.split(","):
        repo = repo.strip()
        try:
            files = [f.rfilename for f in api.model_info(repo, files_metadata=False).siblings]
            out.append(f"== {repo}: {len(files)} files: {files[:40]}")
            for name in [f for f in files if f.endswith("rl_agent_config.json")][:3]:
                cfg = json.load(open(hf_hub_download(repo, name)))
                out.append(f"  {name}: {json.dumps(cfg)[:1500]}")
            for name in [f for f in files if f.endswith("encoder/config.json")][:3]:
                cfg = json.load(open(hf_hub_download(repo, name)))
                keep = {k: cfg.get(k) for k in ("num_hidden_layers", "hidden_size", "num_attention_heads",
                        "intermediate_size", "global_attn_every_n_layers", "local_attention",
                        "sliding_window", "layer_types", "max_position_embeddings")}
                out.append(f"  {name}: {json.dumps(keep)[:1500]}")
        except Exception as exc:  # report and continue
            out.append(f"== {repo}: ERROR {type(exc).__name__}: {exc}")
    weights.commit()
    return "\n".join(out)


@app.function(cpu=4, memory=16384, volumes=VOLUMES, timeout=1800, env=HF_ENV)
def module_tree() -> str:
    import inspect

    from transformers.models.modernbert import modeling_modernbert as mb

    from interp.model import load_agent

    agent = load_agent("laya", device="cpu")
    weights.commit()
    out = [repr(agent.model)[:6000]]
    out.append(f"attn impl: {agent.model.encoder.config._attn_implementation}")
    for name in ("ModernBertModel", "ModernBertEncoderLayer", "ModernBertAttention", "ModernBertMLP",
                 "ModernBertEmbeddings"):
        cls = getattr(mb, name, None)
        if cls is not None:
            out.append(f"===== {name}.forward =====\n{inspect.getsource(cls.forward)}")
    for fn in ("eager_attention_forward", "apply_rotary_pos_emb"):
        if hasattr(mb, fn):
            out.append(f"===== {fn} =====\n{inspect.getsource(getattr(mb, fn))}")
    return "\n".join(out)


def _save_json(rel: str, obj) -> str:
    import json
    import pathlib

    path = pathlib.Path("/store/results") / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    store.commit()
    return str(path)


@app.function(gpu="H100", volumes=VOLUMES, timeout=3600, env=HF_ENV, scaledown_window=60)
def verify(model: str = "laya", n: int = 200) -> dict:
    import json
    import time

    from interp import verify as v
    from interp.model import load_agent

    start = time.time()
    agent = load_agent(model, device="cuda")
    weights.commit()
    report = v.run(agent, n=n)
    report["model"] = model
    report["wall_s"] = round(time.time() - start, 1)
    _save_json(f"phase0/verify_{model}.json", report)
    print(json.dumps(report, indent=2, default=str))
    return report


@app.function(gpu="H100", volumes=VOLUMES, timeout=3 * 3600, env=HF_ENV, scaledown_window=60)
def track1(model: str = "laya", casting: str = "") -> dict:
    from interp import runs
    from interp.hooks import Runner
    from interp.model import load_agent

    agent = load_agent(model, device="cuda")
    weights.commit()
    out = runs.track1(agent, Runner(agent), model, casting=casting or None,
                      log=lambda s: print(s, flush=True))
    store.commit()
    return runs.public({k: out[k] for k in ("model", "casting", "wall_s")})


@app.function(gpu="H100", volumes=VOLUMES, timeout=4 * 3600, env=HF_ENV, scaledown_window=60)
def track1b(model: str = "laya", casting: str = "generic") -> dict:
    from interp import runs
    from interp.hooks import Runner
    from interp.model import load_agent

    agent = load_agent(model, device="cuda")
    out = runs.track1b(agent, Runner(agent), model, casting=casting, log=lambda s: print(s, flush=True))
    store.commit()
    return runs.public({k: out[k] for k in ("model", "wall_s", "top_competition_heads", "top_position_heads")})


@app.function(gpu="H100", volumes=VOLUMES, timeout=4 * 3600, env=HF_ENV, scaledown_window=60)
def track2(model: str = "laya", casting: str = "generic") -> dict:
    from interp import runs
    from interp.hooks import Runner
    from interp.model import load_agent

    agent = load_agent(model, device="cuda")
    out = runs.track2(agent, Runner(agent), model, casting=casting, log=lambda s: print(s, flush=True))
    store.commit()
    return {"model": model, "wall_s": out["wall_s"]}


@app.function(cpu=2, memory=8192, volumes=VOLUMES, timeout=1800, env=HF_ENV)
def probe_datasets(names: str) -> str:
    from datasets import load_dataset

    out = []
    for spec in names.split(";"):
        parts = spec.split("|")
        name, config, split = parts[0], (parts[1] or None) if len(parts) > 1 else None, \
            parts[2] if len(parts) > 2 else "train"
        try:
            ds = load_dataset(name, config, split=split)
            out.append(f"OK {name}|{config}|{split}: {len(ds)} rows, cols={ds.column_names}, "
                       f"row0={str(ds[0])[:300]}")
        except Exception as exc:
            out.append(f"FAIL {name}|{config}|{split}: {type(exc).__name__}: {str(exc)[:200]}")
    return "\n".join(out)


@app.function(gpu="H100", volumes=VOLUMES, timeout=4 * 3600, env=HF_ENV, scaledown_window=60, memory=65536)
def track4(model: str = "laya") -> dict:
    from interp import runs_track4
    from interp.hooks import Runner
    from interp.model import load_agent

    agent = load_agent(model, device="cuda")
    out = runs_track4.track4(agent, Runner(agent), model, log=lambda s: print(s, flush=True))
    store.commit()
    return {"model": model, "wall_s": out["wall_s"]}


@app.function(gpu="H100", volumes=VOLUMES, timeout=4 * 3600, env=HF_ENV, scaledown_window=60, memory=32768)
def track3(model: str = "laya") -> dict:
    from interp import runs
    from interp.hooks import Runner
    from interp.model import load_agent

    agent = load_agent(model, device="cuda")
    out = runs.track3(agent, Runner(agent), model, log=lambda s: print(s, flush=True))
    store.commit()
    return {"model": model, "wall_s": out["wall_s"]}


@app.function(gpu="H100", volumes=VOLUMES, timeout=2 * 3600, env=HF_ENV, scaledown_window=60, memory=32768)
def track43() -> dict:
    from interp import runs
    from interp.hooks import Runner
    from interp.model import load_agent

    agent = load_agent("multilingual", device="cuda")
    weights.commit()
    out = runs.track43(agent, Runner(agent), log=lambda s: print(s, flush=True))
    store.commit()
    return {"wall_s": out["wall_s"]}


@app.function(cpu=2, memory=8192, volumes=VOLUMES, timeout=1800)
def report(costs: str = "") -> str:
    from interp import report as rp

    path = rp.build({"modal_billing_summary": costs} if costs else None)
    store.commit()
    return path


@app.function(cpu=1, memory=2048, volumes=VOLUMES, secrets=[modal.Secret.from_name("report-token")])
@modal.fastapi_endpoint(method="GET")
def view(token: str = ""):
    import hmac
    import os

    from fastapi.responses import HTMLResponse

    if not hmac.compare_digest(token, os.environ["REPORT_TOKEN"]):
        return HTMLResponse("Forbidden", status_code=403)
    store.reload()
    return HTMLResponse(open("/store/report/index.html").read())


@app.function(gpu="H100", volumes=VOLUMES, timeout=2 * 3600, env=HF_ENV, scaledown_window=60)
def track2_norm(model: str = "laya") -> str:
    from interp import runs
    from interp.hooks import Runner
    from interp.model import load_agent

    agent = load_agent(model, device="cuda")
    runs.track2_norm(agent, Runner(agent), model, log=lambda s: print(s, flush=True))
    store.commit()
    return "ok"


OPENAI = modal.Secret.from_name("openai-secret")


@app.function(cpu=1, memory=2048, secrets=[OPENAI], timeout=300)
def llm_check() -> str:
    import os

    from openai import OpenAI

    keys = sorted(k for k in os.environ if "OPENAI" in k or "API_KEY" in k)
    out = [f"env keys: {keys}"]
    try:
        names = sorted(m.id for m in OpenAI().models.list().data)
        out.append("models: " + ", ".join(n for n in names if n.startswith(("gpt", "o"))))
    except Exception as exc:
        out.append(f"ERROR {type(exc).__name__}: {str(exc)[:300]}")
    return chr(10).join(out)


@app.function(gpu="H100", volumes=VOLUMES, timeout=6 * 3600, env=HF_ENV, scaledown_window=60, memory=65536)
def followups() -> dict:
    from interp import followups as fu
    from interp.hooks import Runner
    from interp.model import load_agent

    agent = load_agent("laya", device="cuda")
    out = fu.run_all(agent, Runner(agent), log=lambda s: print(s, flush=True))
    store.commit()
    return out


@app.function(gpu="H100", volumes=VOLUMES, timeout=4 * 3600, scaledown_window=60, memory=32768, secrets=[OPENAI],
              env=HF_ENV | {"LLM_BULK": "gpt-5.6-luna", "LLM_CAREFUL": "gpt-5.6-terra"})
def llm_steps() -> dict:
    from interp import followups as fu
    from interp.hooks import Runner
    from interp.model import load_agent

    agent = load_agent("laya", device="cuda")
    out = fu.llm_steps(agent, Runner(agent), log=lambda s: print(s, flush=True))
    store.commit()
    return {"usage": out["llm_usage"]}


@app.function(gpu="H100", volumes=VOLUMES, timeout=4 * 3600, env=HF_ENV, scaledown_window=60, memory=65536)
def fp32_only() -> str:
    from interp import followups as fu
    from interp.hooks import Runner
    from interp.model import load_agent

    agent = load_agent("laya", device="cuda")
    fu.fp32_only(agent, Runner(agent), log=lambda s: print(s, flush=True))
    store.commit()
    return "ok"


@app.function(gpu="H100", volumes=VOLUMES, timeout=5 * 3600, env=HF_ENV, scaledown_window=60, memory=65536)
def retrain_arm(loss: str = "soft-ce", seed: int = 0) -> dict:
    from interp import retrain

    out = retrain.train_arm(loss, seed=seed, log=lambda s: print(s, flush=True))
    weights.commit()
    store.commit()
    store.reload()
    suffix = "" if seed == 0 else f"_s{seed}"
    ev = retrain.evaluate_arm(f"ctrl_{loss.replace('-', '_')}{suffix}", log=lambda s: print(s, flush=True))
    store.commit()
    return {"train": out, "eval_keys": list(ev)}


@app.function(gpu="H100", volumes=VOLUMES, timeout=5 * 3600, scaledown_window=60, memory=131072, secrets=[OPENAI],
              env=HF_ENV | {"LLM_BULK": "gpt-5.6-luna"})
def sae_run() -> dict:
    from interp import sae
    from interp.hooks import Runner
    from interp.model import load_agent

    agent = load_agent("laya", device="cuda")
    out = sae.run(agent, Runner(agent), log=lambda s: print(s, flush=True))
    store.commit()
    return {"wall_s": out["wall_s"]}


@app.function(gpu="H100", volumes=VOLUMES, timeout=4 * 3600, env=HF_ENV, scaledown_window=60, memory=32768)
def label_guard() -> str:
    from interp import followups as fu

    fu.label_guard_eval(log=lambda s: print(s, flush=True))
    store.commit()
    return "ok"


@app.function(cpu=2, memory=8192, volumes=VOLUMES, timeout=1800, env=HF_ENV)
def freeze_confirm() -> dict:
    from collections import Counter

    from interp import factorial

    items = factorial.mc_confirm()
    store.commit()
    return {"n": len(items), "sources": dict(Counter(it["source"] for it in items))}


@app.function(gpu="H100", volumes=VOLUMES, timeout=3 * 3600, env=HF_ENV, scaledown_window=60, memory=32768)
def phase1(model: str) -> dict:
    from interp import factorial
    from interp.hooks import Runner
    from interp.model import load_agent
    from interp.runs import save

    store.reload()
    agent = load_agent(model, device="cuda")
    out = factorial.run(agent, Runner(agent), model, log=lambda s: print(s, flush=True))
    save(f"phase1/{model}", out)
    store.commit()
    return {"model": model, "ok": True}


@app.function(gpu="H100", volumes=VOLUMES, timeout=1800, env=HF_ENV)
def debug_b() -> str:
    import torch

    from interp import competition as cp
    from interp.hooks import Runner
    from interp.inputs import collate, to_device
    from interp.model import load_agent
    from interp.runs import mc_items

    agent = load_agent("laya", device="cuda")
    r = Runner(agent)
    items = [it for it in mc_items() if it["split"] == "main"][:3]
    out = []
    for it in items:
        e = cp.encode_mc(agent, it, "generic")
        for name, kw, pp in (("base", {}, False), ("B", {}, True)):
            b = to_device(collate([e], agent.tok.pad_token_id, parallel_positions=pp), r.device)
            lg, _, cache = r.forward(b, cache=["enc_final", "L0.resid_pre", "L1.resid_pre"], **kw)
            out.append(f"{name} k={e.k} logits={lg[0, :e.k].tolist()}")
            out.append(f"   pos[:60]={b['position_ids'][0, :60].tolist()}")
            out.append(f"   markers={e.markers} head_len={e.head_len} len={len(e.ids)} nan_enc={torch.isnan(cache['enc_final']).any().item()} "
                       f"nan_L1={torch.isnan(cache['L1.resid_pre']).any().item()}")
    return chr(10).join(out)


@app.local_entrypoint()
def main(task: str = "inspect", paths: str = "", pattern: str = "", start: int = 1, end: int = 200):
    if task == "inspect":
        print(inspect_pkg.remote())
    elif task == "source":
        print(show_source.remote(paths))
    elif task == "grep":
        print(grep_source.remote(pattern, paths or "laya"))
    elif task == "hub":
        print(hub_facts.remote(paths))
    elif task == "tree":
        print(module_tree.remote())
    elif task == "verify":
        for name in (paths or "laya").split(","):
            verify.remote(name)
    elif task == "track1":
        names = (paths or "laya").split(",")
        first = track1.remote(names[0], pattern)
        print(first)
        for name in names[1:]:
            print(track1.remote(name, first["casting"]))
    elif task == "track1b":
        for name in (paths or "laya").split(","):
            print(track1b.remote(name, pattern or "generic"))
    elif task == "track2":
        for name in (paths or "laya").split(","):
            print(track2.remote(name, pattern or "generic"))
    elif task == "datasets":
        print(probe_datasets.remote(paths))
    elif task == "track4":
        for name in (paths or "laya").split(","):
            print(track4.remote(name))
    elif task == "track3":
        for name in (paths or "laya").split(","):
            print(track3.remote(name))
    elif task == "track43":
        print(track43.remote())
    elif task == "report":
        import subprocess

        summary = subprocess.run(["modal", "billing", "summary"], capture_output=True, text=True).stdout
        print(report.remote(summary))
    elif task == "track2_norm":
        print(track2_norm.remote(paths or "laya"))
    elif task == "llm_check":
        print(llm_check.remote())
    elif task == "followups":
        print(followups.remote())
    elif task == "llm_steps":
        print(llm_steps.remote())
    elif task == "fp32":
        print(fp32_only.remote())
    elif task == "stretch":
        calls = [retrain_arm.spawn("soft-ce"), retrain_arm.spawn("rlcd"), sae_run.spawn()]
        for c in calls:
            print(c.get())
    elif task == "seeds":
        calls = [retrain_arm.spawn(loss, seed) for loss in ("soft-ce", "rlcd") for seed in (1, 2)]
        for c in calls:
            print(c.get())
    elif task == "label_guard":
        print(label_guard.remote())
    elif task == "phase1":
        print(freeze_confirm.remote())
        models = (paths or "laya,smolaya,multilingual,ctrl_soft_ce,ctrl_rlcd,ctrl_soft_ce_s1,ctrl_soft_ce_s2,"
                           "ctrl_rlcd_s1,ctrl_rlcd_s2").split(",")
        calls = [phase1.spawn(m) for m in models]
        for c in calls:
            print(c.get())
    elif task == "debug_b":
        print(debug_b.remote())
    elif task == "lines":
        print(show_lines.remote(paths, start, end))
