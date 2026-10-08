"""Track 2: choice (direction of the centered logit vector) vs confidence (its norm)."""

from __future__ import annotations

import numpy as np
import torch

from . import ablation as ab
from . import competition as cp
from . import metrics as mt
from .hooks import slot_logits
from .inputs import SEG_MASK, collate, to_device

# ----- 2.1 lenses -----


@torch.no_grad()
def marker_residuals(runner, encs, batch_size: int = 64):
    """Per item: X [L+1, k, d] residual at the [MASK]s entering each layer, plus marker_h last."""
    L = runner.n_layers
    sites = [f"L{l}.resid_pre" for l in range(L)] + ["marker_h"]
    out = [None] * len(encs)
    order = sorted(range(len(encs)), key=lambda i: len(encs[i].ids))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = to_device(collate([encs[i] for i in idx], runner.agent.tok.pad_token_id), runner.device)
        _, _, cache = runner.forward(batch, cache=sites)
        mp = batch["marker_pos"]
        gather = lambda x: torch.gather(x, 1, mp[:, :, None].expand(-1, -1, x.shape[-1]))  # noqa: E731
        stacked = torch.stack([gather(cache[s]) for s in sites[:-1]] + [cache["marker_h"]], 1)  # [B, L+1, K, d]
        stacked = stacked.float().cpu()
        for row, i in enumerate(idx):
            out[i] = stacked[row, :, : encs[i].k].numpy()
    return out


@torch.no_grad()
def lens_logits(runner, layer: int, x: torch.Tensor, qtype: torch.Tensor, tuned=None) -> torch.Tensor:
    """x [N, d] residual entering `layer` (or marker_h when layer == n_layers) -> scorer logits [N]."""
    m = runner.model
    if tuned is not None:
        W, b = tuned
        x = x @ W + b
        return m.scorer(x).squeeze(-1).float()
    if layer < runner.n_enc:
        x = runner.enc.final_norm(x) + m.type_emb(qtype)
    return m.scorer(x).squeeze(-1).float()


def fit_tuned(X: np.ndarray, Y: np.ndarray, lam: float = 1e-2):
    """Ridge affine map X -> Y (closed form, on GPU)."""
    Xt = torch.tensor(X, device="cuda", dtype=torch.float64)
    Yt = torch.tensor(Y, device="cuda", dtype=torch.float64)
    mx, my = Xt.mean(0), Yt.mean(0)
    Xc, Yc = Xt - mx, Yt - my
    d = Xc.shape[1]
    scale = (Xc ** 2).mean()
    W = torch.linalg.solve(Xc.T @ Xc + lam * scale * len(Xc) * torch.eye(d, device="cuda", dtype=torch.float64),
                           Xc.T @ Yc)
    b = my - mx @ W
    return W.float(), b.float()


def lenses(agent, runner, train_items, test_items, casting) -> dict:
    tr = [cp.encode_mc(agent, it, casting) for it in train_items]
    te = [cp.encode_mc(agent, it, casting) for it in test_items]
    Xtr, Xte = marker_residuals(runner, tr), marker_residuals(runner, te)
    L = runner.n_layers
    dev = runner.device
    qt = lambda n: torch.full((n,), tr[0].qtype, device=dev, dtype=torch.long)  # noqa: E731  (all choice)
    final_te = [mt.centered(lens_logits(runner, L, torch.tensor(x[L], device=dev), qt(x.shape[1])).cpu().numpy())
                for x in Xte]
    res = {"direct": {}, "tuned": {}}
    for l in range(L + 1):
        tuned = None
        if l < L:
            A = np.concatenate([x[l] for x in Xtr])
            Y = np.concatenate([x[L] for x in Xtr])
            tuned = fit_tuned(A, Y)
        for kind in ("direct", "tuned"):
            cosv, agree, ratio, nl, nf = [], [], [], [], []
            for x, zf in zip(Xte, final_te):
                xt = torch.tensor(x[l], device=dev)
                z = lens_logits(runner, l, xt, qt(len(xt)), tuned if (kind == "tuned" and l < L) else None)
                z = mt.centered(z.cpu().numpy())
                du, df = mt.direction(z), mt.direction(zf)
                cosv.append(float((du * df).sum()))
                agree.append(float(z.argmax() == zf.argmax()))
                nl.append(float(np.linalg.norm(z)))
                nf.append(float(np.linalg.norm(zf)))
                ratio.append(nl[-1] / max(nf[-1], 1e-9))
            res[kind][l] = {"dir_cos": float(np.mean(cosv)), "argmax_agree": float(np.mean(agree)),
                            "norm_ratio_median": float(np.median(ratio)),
                            "norm_corr_with_final": float(np.corrcoef(nl, nf)[0, 1]) if np.std(nl) > 0 else 0.0}
    for kind in ("direct", "tuned"):
        rows = res[kind]
        res[f"{kind}_dir_stable_layer"] = next((l for l in range(L + 1)
                                                if rows[l]["argmax_agree"] >= 0.9 and rows[l]["dir_cos"] >= 0.9), None)
        res[f"{kind}_norm_90_layer"] = next((l for l in range(L + 1)
                                             if 0.9 <= rows[l]["norm_ratio_median"] <= 1.1 and
                                             rows[l]["norm_corr_with_final"] >= 0.9), None)
    return res


# ----- 2.2 MLP screen (heads come from track1b's mean screen) -----


def mlp_screen(agent, runner, items, casting, means) -> dict:
    encs = [cp.encode_mc(agent, it, casting) for it in items]
    base = slot_logits(runner, encs, amp=True)
    out = {k: np.zeros(runner.n_layers) for k in ("rotation_deg", "log_scale", "abs_log_scale",
                                                  "argmax_change", "l1_centered")}
    for l in range(runner.n_layers):
        rows = slot_logits(runner, encs, amp=True, edits=ab.mean_mlps(means, [l]))
        e = effects_by_k(base, rows)
        for k, v in e.items():
            out[k][l] = v
    return out


def effects_by_k(base, rows) -> dict:
    by_k: dict[int, list[int]] = {}
    for i, r in enumerate(base):
        by_k.setdefault(len(r), []).append(i)
    acc: dict[str, list] = {}
    for idx in by_k.values():
        b, r = np.stack([base[i] for i in idx]), np.stack([rows[i] for i in idx])
        vals = {"rotation_deg": mt.rotation(b, r), "log_scale": mt.log_scale(b, r),
                "abs_log_scale": np.abs(mt.log_scale(b, r)),
                "argmax_change": (b.argmax(1) != r.argmax(1)).astype(float),
                "l1_centered": np.abs(mt.centered(r) - mt.centered(b)).sum(1)}
        for k, v in vals.items():
            acc.setdefault(k, []).append(v)
    return {k: float(np.concatenate(v).mean()) for k, v in acc.items()}


def per_item_effects(base, rows) -> dict:
    rot = np.array([mt.rotation(b, r) for b, r in zip(base, rows)])
    ls = np.array([mt.log_scale(b, r) for b, r in zip(base, rows)])
    arg = np.array([float(np.argmax(b) != np.argmax(r)) for b, r in zip(base, rows)])
    return {"rotation_deg": rot, "log_scale": ls, "argmax_change": arg}


# ----- 2.3 separability -----


def components(head_screen: dict, mlp: dict, runner) -> list[dict]:
    comps = []
    for l in range(head_screen["rotation_deg"].shape[0]):
        for h in range(runner.n_heads(l)):
            comps.append({"kind": "head", "layer": l, "head": h,
                          **{k: float(head_screen[k][l, h]) for k in ("rotation_deg", "log_scale",
                                                                       "abs_log_scale", "argmax_change")}})
    for l in range(runner.n_layers):
        comps.append({"kind": "mlp", "layer": l, **{k: float(mlp[k][l]) for k in ("rotation_deg", "log_scale",
                                                                                     "abs_log_scale",
                                                                                     "argmax_change")}})
    return comps


def select_sets(comps: list[dict], K: int) -> tuple[list[dict], list[dict]]:
    rot_med = np.median([c["rotation_deg"] for c in comps])
    sc_med = np.median([c["abs_log_scale"] for c in comps])
    a = [c for c in comps if c["abs_log_scale"] <= sc_med]
    a.sort(key=lambda c: c["rotation_deg"], reverse=True)
    b = [c for c in comps if c["argmax_change"] <= 0.02 and c["rotation_deg"] <= rot_med]
    b.sort(key=lambda c: c["abs_log_scale"], reverse=True)
    return a[:K], b[:K]


def set_edits(comp_set: list[dict], means) -> dict:
    heads: dict[int, list[int]] = {}
    mlps = []
    for c in comp_set:
        if c["kind"] == "head":
            heads.setdefault(c["layer"], []).append(c["head"])
        else:
            mlps.append(c["layer"])
    return ab.merge(ab.mean_heads(means, heads), ab.mean_mlps(means, mlps))


def summarize_effect(base, rows, gold=None, temps=None) -> dict:
    e = per_item_effects(base, rows)
    out = {"rotation_deg": mt.ci_dict(e["rotation_deg"], n=500),
           "log_scale": mt.ci_dict(e["log_scale"], n=500),
           "abs_log_scale": mt.ci_dict(np.abs(e["log_scale"]), n=500),
           "argmax_unchanged": float(1 - e["argmax_change"].mean())}
    p0 = [mt.softmax(b) for b in base]
    p1 = [mt.softmax(r) for r in rows]
    out["entropy_change"] = float(np.mean([mt.entropy(b) - mt.entropy(a) for a, b in zip(p0, p1)]))
    if gold is not None:
        out |= calibration_block(rows, gold, temps)
    return out


def calibration_block(rows, gold, temps=None) -> dict:
    correct = np.array([np.argmax(r) == g for r, g in zip(rows, gold)])
    marg = np.array([mt.margin(r) for r in rows])
    conf = np.array([mt.softmax(r).max() for r in rows])
    out = {"accuracy": float(correct.mean()), "auroc_margin": mt.auroc(marg, correct),
           "ece_raw": mt.ece(conf, correct)}
    if temps is not None:
        conf_t = np.array([mt.softmax(r, t).max() for r, t in zip(rows, temps)])
        out["ece_tempered"] = mt.ece(conf_t, correct)
    return out


@torch.no_grad()
def norm_direction(agent, runner, train_items, test_items, casting, layer: int, r_choice: int = 32,
                   alphas=(-0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5), seed: int = 0) -> dict:
    """Direction at `layer` (shared across an item's [MASK]s) that predicts log||z||, orthogonal to the
    choice subspace (top PCs of [MASK]-minus-item-mean residuals). Steer with it, report dose response.
    Doses are fractions of the mean residual norm at the [MASK]s, added to every [MASK] of the item."""
    tr = [cp.encode_mc(agent, it, casting) for it in train_items]
    te = [cp.encode_mc(agent, it, casting) for it in test_items]
    Xtr = marker_residuals(runner, tr)
    logits_tr = slot_logits(runner, tr)
    mean_vec = np.stack([x[layer].mean(0) for x in Xtr])                     # [N, d]
    y = np.log(np.array([np.linalg.norm(mt.centered(s)) for s in logits_tr]) + 1e-9)
    diffs = np.concatenate([x[layer] - x[layer].mean(0, keepdims=True) for x in Xtr])
    U = np.linalg.svd(diffs - diffs.mean(0), full_matrices=False)[2][:r_choice]   # [r, d]
    Xc = mean_vec - mean_vec.mean(0)
    w = np.linalg.solve(Xc.T @ Xc + 1e-2 * np.trace(Xc.T @ Xc) / Xc.shape[1] * np.eye(Xc.shape[1]),
                        Xc.T @ (y - y.mean()))
    w_perp = w - U.T @ (U @ w)
    pred_r = float(np.corrcoef(Xc @ w_perp, y)[0, 1])
    direction = w_perp / np.linalg.norm(w_perp)
    sigma = float(np.std(Xc @ direction))
    rnorm = float(np.mean([np.linalg.norm(x[layer], axis=-1).mean() for x in Xtr]))
    rng = np.random.default_rng(seed)
    rand = rng.standard_normal(direction.shape)
    rand -= U.T @ (U @ rand)
    rand /= np.linalg.norm(rand)
    base = slot_logits(runner, te)
    res = {"layer": layer, "train_corr_log_norm": pred_r, "sigma": sigma, "resid_norm": rnorm,
           "dose_unit": "fraction of mean [MASK] residual norm", "dose": {}, "random_dose": {}}
    for name, vec in (("dose", direction), ("random_dose", rand)):
        v = torch.tensor(vec, device=runner.device, dtype=torch.float32)
        for a in alphas:
            def edit(x, batch, a=a, v=v):
                m = (batch["segment"] == SEG_MASK)[..., None]
                return x + m * (a * rnorm) * v.to(x.dtype)
            rows = slot_logits(runner, te, edits={f"L{layer}.resid_pre": edit})
            e = per_item_effects(base, rows)
            res[name][a] = {"log_scale": float(e["log_scale"].mean()),
                            "rotation_deg": float(e["rotation_deg"].mean()),
                            "argmax_unchanged": float(1 - e["argmax_change"].mean())}
    return res


# ----- 2.5 overconfidence -----


def fit_temperature(rows, gold) -> float:
    from scipy.optimize import minimize_scalar

    def nll(logt):
        t = np.exp(logt)
        return -np.mean([np.log(mt.softmax(r, t)[g] + 1e-12) for r, g in zip(rows, gold)])

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def overconfidence(rows, gold, items, marker_norms) -> dict:
    gold = np.asarray(gold)
    T = fit_temperature(rows, gold)
    norms = np.array([np.linalg.norm(mt.centered(r)) for r in rows])
    out = {"global_T": T, "by_norm_quintile": [], "by_source": {}}
    qs = np.quantile(norms, [0, .2, .4, .6, .8, 1.0])
    for lo, hi in zip(qs[:-1], qs[1:]):
        sel = (norms >= lo) & (norms <= hi)
        rs = [r for r, s in zip(rows, sel) if s]
        out["by_norm_quintile"].append({"norm_range": [float(lo), float(hi)], "n": int(sel.sum()),
                                        "T": fit_temperature(rs, gold[sel]),
                                        "accuracy": float(np.mean([np.argmax(r) == g for r, g in
                                                                   zip(rs, gold[sel])]))})
    for src in sorted({it["source"] for it in items}):
        sel = np.array([it["source"] == src for it in items])
        rs = [r for r, s in zip(rows, sel) if s]
        out["by_source"][src] = fit_temperature(rs, gold[sel])
    conf_T = np.array([mt.softmax(r, T).max() for r in rows])
    correct = np.array([np.argmax(r) == g for r, g in zip(rows, gold)])
    out["ece_after_global_T"] = mt.ece(conf_T, correct)
    out["corr_logitnorm_vs_markerh_norm"] = float(np.corrcoef(norms, marker_norms)[0, 1])
    out["corr_logitnorm_vs_correct"] = float(np.corrcoef(norms, correct.astype(float))[0, 1])
    return out
