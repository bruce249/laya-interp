"""TopK SAE on laya's [MASK] residuals: do choice and confidence live in distinct features?"""

from __future__ import annotations

import json
import random
import time

import numpy as np
import torch

from . import metrics as mt


class TopKSAE(torch.nn.Module):
    def __init__(self, d: int, m: int, k: int):
        super().__init__()
        self.k = k
        self.W_enc = torch.nn.Parameter(torch.randn(d, m) / d ** 0.5)
        self.b_enc = torch.nn.Parameter(torch.zeros(m))
        self.W_dec = torch.nn.Parameter(self.W_enc.data.T.clone())
        self.b_dec = torch.nn.Parameter(torch.zeros(d))
        self.normalize()

    @torch.no_grad()
    def normalize(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)

    def encode(self, x):
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        vals, idx = pre.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, idx, torch.relu(vals))
        return acts

    def forward(self, x):
        a = self.encode(x)
        return a @ self.W_dec + self.b_dec, a


# ----- data: [MASK] residuals from laya on the train splits (option orders rotated) -----


@torch.no_grad()
def collect(agent, runner, layers, n_mc: int = 20000, n_bool: int = 6000, seed: int = 0, log=print):
    from .hooks import slot_logits  # noqa: F401  (kept for parity with other modules)
    from .inputs import collate, encode, to_device
    from .labels import LABELS
    from .retrain import train_rows

    rng = random.Random(seed)
    rows = train_rows(seed)
    mc = [r for r in rows if r["q"]["type"] == "choice"][:n_mc]
    bq = [r for r in rows if r["q"]["type"] == "noul"][:n_bool]
    specs = []
    for r in mc:
        k = len(r["q"]["criteria"])
        shifts = rng.sample(range(k), min(2, k))
        for s in shifts:
            order = list(range(k))[s:] + list(range(k))[:s]
            specs.append((r, dict(r["q"], option_order=order), order))
    for r in bq:
        for labels in (LABELS["semantic_tf"][0], LABELS["symbolic_ab"][0]):
            specs.append((r, dict(r["q"], labels=labels), None))
    log(f"[sae] sequences={len(specs)}")
    sites = [f"L{l}.resid_pre" for l in layers]
    acts = {l: [] for l in layers}
    meta = []
    bs = 128
    encs_all = [encode(agent, r["state"], q) for r, q, _ in specs]
    order_idx = sorted(range(len(specs)), key=lambda i: len(encs_all[i].ids))
    for start in range(0, len(order_idx), bs):
        idx = order_idx[start:start + bs]
        b = to_device(collate([encs_all[i] for i in idx], agent.tok.pad_token_id), runner.device)
        logits, _, cache = runner.forward(b, cache=sites, amp=True)
        mp = b["marker_pos"]
        for li, l in enumerate(layers):
            g = torch.gather(cache[sites[li]], 1, mp[:, :, None].expand(-1, -1, runner.d))
            for row, i in enumerate(idx):
                acts[l].append(g[row, : encs_all[i].k].half())
        lg = logits.float().cpu().numpy()
        for row, i in enumerate(idx):
            r, q, order = specs[i]
            k = encs_all[i].k
            s = lg[row, :k]
            z = mt.centered(s)
            for slot in range(k):
                opt = order[slot] if order else slot
                meta.append({"seq": i, "slot": slot, "option": opt, "k": k,
                             "is_argmax": int(np.argmax(s) == slot), "is_gold": int(opt == r["gold"]),
                             "log_norm": float(np.log(np.linalg.norm(z) + 1e-9)), "margin": float(mt.margin(s)),
                             "qtype": q["type"], "source": r["source"],
                             "question": q["instructions"][:160],
                             "option_text": (list(q["criteria"])[opt] if q["type"] == "choice"
                                             else ["false", "true"][opt])[:80]})
    X = {l: torch.cat(acts[l]) for l in layers}
    return X, meta


def train_sae(X: torch.Tensor, m: int = 16384, k: int = 32, steps: int = 12000, batch: int = 4096,
              lr: float = 2e-4, seed: int = 0, log=print):
    torch.manual_seed(seed)
    n, d = X.shape
    perm = torch.randperm(n, device=X.device)
    n_val = max(2000, n // 20)
    val, tr = X[perm[:n_val]].float(), X[perm[n_val:]]
    scale = float(tr[: min(len(tr), 50000)].float().norm(dim=-1).mean() / d ** 0.5)
    sae = TopKSAE(d, m, k).cuda()
    with torch.no_grad():
        sae.b_dec.data = (tr[: min(len(tr), 50000)].float() / scale).median(0).values
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    fired = torch.zeros(m, device="cuda")
    for step in range(steps):
        xb = tr[torch.randint(0, len(tr), (batch,), device=tr.device)].float() / scale
        rec, a = sae(xb)
        loss = ((rec - xb) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        # keep decoder rows unit-norm: remove the gradient component parallel to each row
        with torch.no_grad():
            W = sae.W_dec
            W.grad -= (W.grad * W.data).sum(1, keepdim=True) * W.data
        opt.step()
        sae.normalize()
        fired += (a > 0).float().sum(0)
        if step % 2000 == 0 or step == steps - 1:
            with torch.no_grad():
                vr, _ = sae(val / scale)
                fve = 1 - ((vr - val / scale) ** 2).sum() / ((val / scale - (val / scale).mean(0)) ** 2).sum()
            log(f"[sae] step {step} loss={loss.item():.3f} val_FVE={fve.item():.3f} "
                f"dead={(fired == 0).float().mean().item():.3f}")
            fired.zero_()
    with torch.no_grad():
        vr, va = sae(val / scale)
        fve = float(1 - ((vr - val / scale) ** 2).sum() / ((val / scale - (val / scale).mean(0)) ** 2).sum())
        alive = float((sae.encode(tr[:200000].float() / scale) > 0).any(0).float().mean())
    return sae, scale, {"val_fve": fve, "alive_frac": alive, "n_train": int(len(tr)), "m": m, "k": k}


@torch.no_grad()
def feature_scores(sae, scale, X, meta):
    """Choice score: within-item AUROC of the latent for 'this slot wins'. Confidence score: corr of the
    item-mean latent with the item's log ||z||. Computed on choice items only."""
    idx = [i for i, r in enumerate(meta) if r["qtype"] == "choice"]
    A = torch.cat([sae.encode(X[idx[s:s + 8192]].float() / scale) for s in range(0, len(idx), 8192)]).cpu().numpy()
    seqs = np.array([meta[i]["seq"] for i in idx])
    is_arg = np.array([meta[i]["is_argmax"] for i in idx])
    lognorm = np.array([meta[i]["log_norm"] for i in idx])
    # within-item centred activation for the choice score
    uniq, inv = np.unique(seqs, return_inverse=True)
    counts = np.bincount(inv)
    sums = np.zeros((len(uniq), A.shape[1]))
    np.add.at(sums, inv, A)
    item_mean = sums / counts[:, None]
    Ac = A - item_mean[inv]
    item_lognorm = np.zeros(len(uniq))
    item_lognorm[inv] = lognorm
    freq = (A > 0).mean(0)
    choice, conf = np.full(A.shape[1], np.nan), np.full(A.shape[1], np.nan)
    active = np.where(freq > 0.002)[0]
    from sklearn.metrics import roc_auc_score
    for j in active:
        if Ac[:, j].std() > 0:
            choice[j] = abs(roc_auc_score(is_arg, Ac[:, j]) - 0.5) * 2
        if item_mean[:, j].std() > 0:
            conf[j] = abs(np.corrcoef(item_mean[:, j], item_lognorm)[0, 1])
    return {"choice": choice, "confidence": conf, "freq": freq, "A": A, "idx": idx}


def ablate_features_edit(sae, scale, layer: int, feats: list[int]):
    """Remove the selected latents' decoder contributions at the [MASK]s (error term kept)."""
    from .inputs import SEG_MASK

    W = sae.W_dec[feats]                                   # [f, d]

    def f(x, batch):
        m = batch["segment"] == SEG_MASK
        xs = x[m].float() / scale
        a = sae.encode(xs)[:, feats]                       # [n, f]
        x = x.clone()
        x[m] = (xs - a @ W).mul(scale).to(x.dtype)
        return x
    return {f"L{layer}.resid_pre": f}


def recon_edit(sae, scale, layer: int):
    from .inputs import SEG_MASK

    def f(x, batch):
        m = batch["segment"] == SEG_MASK
        x = x.clone()
        x[m] = (sae(x[m].float() / scale)[0] * scale).to(x.dtype)
        return x
    return {f"L{layer}.resid_pre": f}


def run(agent, runner, log=print, label_with_llm: bool = True) -> dict:
    from . import competition as cp
    from . import confidence as cf
    from .hooks import slot_logits
    from .runs import RESULTS, mc_items, save

    t0 = time.time()
    n = runner.n_enc
    layers = [21, n]                                        # margin forms at 21; L28 = after the B MLPs
    X, meta = collect(agent, runner, layers, log=log)
    log(f"[sae] collected {X[layers[0]].shape[0]} [MASK] vectors ({time.time() - t0:.0f}s)")
    items = [it for it in mc_items() if it["split"] == "heldout"]
    encs = [cp.encode_mc(agent, it, "generic") for it in items]
    base = slot_logits(runner, encs)
    greedy = json.loads((RESULTS / "followups/greedy_sets.json").read_text())
    out = {"layers": layers, "n_vectors": int(X[layers[0]].shape[0])}
    for layer in layers:
        sae, scale, stats = train_sae(X[layer].cuda(), log=log)
        rows = slot_logits(runner, encs, edits=recon_edit(sae, scale, layer))
        e = cf.per_item_effects(base, rows)
        stats["downstream_argmax_kept"] = float(1 - e["argmax_change"].mean())
        stats["downstream_abs_log_scale"] = float(np.abs(e["log_scale"]).mean())
        sc = feature_scores(sae, scale, X[layer].cuda(), meta)
        ch, co = np.nan_to_num(sc["choice"], nan=-1), np.nan_to_num(sc["confidence"], nan=-1)
        top_c = [int(j) for j in np.argsort(-ch)[:25]]
        top_f = [int(j) for j in np.argsort(-co)[:25]]
        res = {"stats": stats, "top_choice": top_c, "top_confidence": top_f,
               "overlap_top25": len(set(top_c) & set(top_f)),
               "corr_choice_vs_conf_scores": float(np.corrcoef(ch[(ch >= 0) & (co >= 0)],
                                                               co[(ch >= 0) & (co >= 0)])[0, 1]),
               "top_choice_scores": [float(ch[j]) for j in top_c], "top_conf_scores": [float(co[j]) for j in top_f]}
        # feature-level separability: remove top confidence vs top choice latents
        for name, feats in (("ablate_conf10", top_f[:10]), ("ablate_choice10", top_c[:10]),
                            ("ablate_random10", [int(j) for j in np.random.default_rng(0).choice(
                                np.where(sc["freq"] > 0.002)[0], 10, replace=False)])):
            rows = slot_logits(runner, encs, edits=ablate_features_edit(sae, scale, layer, feats))
            e = cf.per_item_effects(base, rows)
            res[name] = {"abs_log_scale": float(np.abs(e["log_scale"]).mean()),
                         "log_scale": float(e["log_scale"].mean()),
                         "rotation_deg": float(e["rotation_deg"].mean()),
                         "argmax_kept": float(1 - e["argmax_change"].mean())}
        # which latents does the greedy-B MLP ablation move? (only meaningful downstream of the MLPs)
        if layer > max(c["layer"] for c in greedy["B_set"]):
            from . import ablation as ab
            from .inputs import SEG_MASK, collate, to_device
            means = ab.segment_means(runner, encs[:600])
            edits = cf.set_edits(greedy["B_set"], means)
            deltas = []
            with torch.no_grad():
                for s in range(0, len(encs), 64):
                    b = to_device(collate(encs[s:s + 64], agent.tok.pad_token_id), runner.device)
                    _, _, c0 = runner.forward(b, cache=[f"L{layer}.resid_pre"])
                    _, _, c1 = runner.forward(b, cache=[f"L{layer}.resid_pre"], edits=edits)
                    m = b["segment"] == SEG_MASK
                    a0 = sae.encode(c0[f"L{layer}.resid_pre"][m].float() / scale)
                    a1 = sae.encode(c1[f"L{layer}.resid_pre"][m].float() / scale)
                    deltas.append((a1 - a0).abs().mean(0).cpu())
            d = torch.stack(deltas).mean(0).numpy()
            top_moved = [int(j) for j in np.argsort(-d)[:25]]
            res["greedyB_top_moved"] = top_moved
            res["greedyB_moved_overlap_conf25"] = len(set(top_moved) & set(top_f))
            res["greedyB_moved_overlap_choice25"] = len(set(top_moved) & set(top_c))
        # examples for labelling
        A, idx = sc["A"], sc["idx"]
        examples = {}
        for j in sorted(set(top_c[:15]) | set(top_f[:15])):
            best = np.argsort(-A[:, j])[:8]
            examples[j] = [{"question": meta[idx[b]]["question"], "option": meta[idx[b]]["option_text"],
                            "is_argmax": meta[idx[b]]["is_argmax"], "log_norm": round(meta[idx[b]]["log_norm"], 2)}
                           for b in best]
        res["examples"] = examples
        if label_with_llm:
            try:
                from . import llm
                prompts = [f"Latent {j}. Top-activating [MASK] contexts (question / option / whether this option "
                           f"won / item log-confidence):\n" + "\n".join(json.dumps(x) for x in ex)
                           for j, ex in examples.items()]
                sysmsg = ("You label sparse-autoencoder features of a multiple-choice classifier. Give a short label "
                          "for what the contexts share (topic, option property, winning/losing, confidence). "
                          'Reply as JSON: {"label": "...", "confidence": "low|medium|high"}')
                import os
                labs = llm.ask_many(os.environ.get("LLM_BULK", "gpt-5.6-luna"), sysmsg, prompts)
                res["labels"] = {j: json.loads(t).get("label") if t else None for j, t in zip(examples, labs)}
            except Exception as exc:
                res["labels_error"] = f"{type(exc).__name__}: {exc}"
        out[f"L{layer}"] = res
        log(f"[sae L{layer}] FVE={stats['val_fve']:.3f} alive={stats['alive_frac']:.3f} "
            f"recon kept={stats['downstream_argmax_kept']:.3f} |ls|={stats['downstream_abs_log_scale']:.3f} "
            f"overlap25={res['overlap_top25']} conf10={res['ablate_conf10']} choice10={res['ablate_choice10']} "
            f"rand10={res['ablate_random10']} B-moved∩conf={res.get('greedyB_moved_overlap_conf25')} "
            f"∩choice={res.get('greedyB_moved_overlap_choice25')}")
        (RESULTS / "sae").mkdir(parents=True, exist_ok=True)
        torch.save({"state": sae.state_dict(), "scale": scale}, RESULTS / "sae" / f"sae_L{layer}.pt")
    out["wall_s"] = round(time.time() - t0, 1)
    save("sae/summary", out)
    return out
