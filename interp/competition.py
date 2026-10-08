"""Track 1: option order, interaction matrices and the parallel-layout control (C1)."""

from __future__ import annotations

import copy
import random

import numpy as np

from . import metrics as mt
from .data import mc_query
from .hooks import slot_logits, to_option_order
from .inputs import encode

# ----- encoding helpers -----


def encode_mc(agent, item, casting, option_order=None, options=None):
    state, qdef = mc_query(item, casting, option_order=option_order, options=options)
    return encode(agent, state, qdef, meta={"id": item["id"]})


def with_filler(enc, slot: int, filler_id: int):
    """Same sequence with the option tokens of `slot` (not its [MASK]) replaced by `filler_id`."""
    out = copy.copy(enc)
    a, b = enc.option_spans[slot]
    out.ids = list(enc.ids)
    for t in range(a + 1, b):
        out.ids[t] = filler_id
    return out


def filler_token(tok, text: str = " ...") -> int:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return ids[0]


def option_logits(runner, encs, orders, **kw):
    rows = slot_logits(runner, encs, **kw)
    return [to_option_order(r, o) for r, o in zip(rows, orders)]


# ----- baselines -----


def baseline(agent, runner, items, casting, temperature=None, **kw) -> dict:
    encs = [encode_mc(agent, it, casting) for it in items]
    rev_orders = [list(range(len(it["options"])))[::-1] for it in items]
    encs_rev = [encode_mc(agent, it, casting, option_order=o) for it, o in zip(items, rev_orders)]
    s = option_logits(runner, encs, [None] * len(items), **kw)
    s_rev = option_logits(runner, encs_rev, rev_orders, **kw)
    gold = np.array([it["gold"] for it in items])
    pred = np.array([np.argmax(r) for r in s])
    pred_rev = np.array([np.argmax(r) for r in s_rev])
    correct = pred == gold
    t = temperature if temperature is not None else encs[0].temperature
    conf_raw = np.array([mt.softmax(r).max() for r in s])
    conf_t = np.array([mt.softmax(r, e.temperature).max() for r, e in zip(s, encs)])
    return {
        "casting": casting, "n": len(items),
        "accuracy": mt.ci_dict(correct.astype(float)),
        "accuracy_reversed": mt.ci_dict((pred_rev == gold).astype(float)),
        "reversal_flip_rate": mt.ci_dict((pred != pred_rev).astype(float)),
        "ece_raw": mt.ece(conf_raw, correct), "ece_tempered": mt.ece(conf_t, correct),
        "mean_conf_raw": float(conf_raw.mean()), "mean_conf_tempered": float(conf_t.mean()),
        "temperature_example": float(t),
        "by_source": {src: float(correct[[it["source"] == src for it in items]].mean())
                      for src in sorted({it["source"] for it in items})},
        "_logits": s, "_correct": correct,
    }


# ----- 1.1 order map -----


def orders_for(k: int, n_random: int, rng: random.Random):
    base = list(range(k))
    orders = [base[i:] + base[:i] for i in range(k)]          # cyclic shifts, identity first
    orders.append(base[::-1])
    seen = {tuple(o) for o in orders}
    tries = 0
    while len(orders) < k + 1 + n_random and tries < 100:
        o = base[:]
        rng.shuffle(o)
        if tuple(o) not in seen:
            seen.add(tuple(o))
            orders.append(o)
        tries += 1
    return orders


def order_map(agent, runner, items, casting, n_random: int = 6, seed: int = 0, **kw) -> dict:
    rng = random.Random(seed)
    encs, orders, owner = [], [], []
    plan = []
    for n, it in enumerate(items):
        os_ = orders_for(len(it["options"]), n_random, rng)
        plan.append(os_)
        for o in os_:
            encs.append(encode_mc(agent, it, casting, option_order=o))
            orders.append(o)
            owner.append(n)
    rows = option_logits(runner, encs, orders, **kw)

    flips_rev, flips_any, agree_modal = [], [], []
    slot_resid: dict[tuple[int, int], list] = {}
    resid_all, slot_pred_all, ctx_feat, ctx_resid = [], [], [], []
    per_item = []
    cursor = 0
    for n, it in enumerate(items):
        os_ = plan[n]
        Z = np.stack([mt.centered(rows[cursor + r]) for r in range(len(os_))])   # [orders, k] option order
        cursor += len(os_)
        k = Z.shape[1]
        preds = Z.argmax(1)
        flips_rev.append(float(preds[0] != preds[k]))
        flips_any.append(float(len(set(preds.tolist())) > 1))
        agree_modal.append(float(np.mean(preds == np.bincount(preds).argmax())))
        R = Z - Z.mean(0, keepdims=True)                     # order-induced residual per option
        for r, o in enumerate(os_):
            for slot, opt in enumerate(o):
                slot_resid.setdefault((k, slot), []).append(R[r, opt])
        per_item.append((os_, Z, R))

    slot_mean = {key: float(np.mean(v)) for key, v in slot_resid.items()}
    for os_, Z, R in per_item:
        k = Z.shape[1]
        base = Z.mean(0)
        for r, o in enumerate(os_):
            for slot, opt in enumerate(o):
                pred = slot_mean[(k, slot)]
                resid_all.append(R[r, opt])
                slot_pred_all.append(pred)
                neighbours = [o[s] for s in (slot - 1, slot + 1) if 0 <= s < k]
                ctx_feat.append(np.mean([base[m] for m in neighbours]))
                ctx_resid.append(R[r, opt] - pred)
    resid_all, slot_pred_all = np.array(resid_all), np.array(slot_pred_all)
    ss_total = float((resid_all ** 2).sum())
    ss_after_slot = float(((resid_all - slot_pred_all) ** 2).sum())
    ctx_feat, ctx_resid = np.array(ctx_feat), np.array(ctx_resid)
    neighbour_corr = float(np.corrcoef(ctx_feat, ctx_resid)[0, 1])

    slot_curves = {}
    for (k, slot), v in sorted(slot_resid.items()):
        slot_curves.setdefault(str(k), []).append(mt.ci_dict(np.array(v), n=500))
    return {
        "casting": casting, "n_items": len(items), "n_sequences": len(encs),
        "reversal_flip_rate": mt.ci_dict(np.array(flips_rev)),
        "any_order_flip_rate": mt.ci_dict(np.array(flips_any)),
        "modal_agreement": mt.ci_dict(np.array(agree_modal)),
        "order_variance_explained_by_slot": 1 - ss_after_slot / ss_total if ss_total else float("nan"),
        "order_variance_total": ss_total / len(resid_all),
        "neighbour_strength_vs_residual_corr": neighbour_corr,
        "slot_curves": slot_curves,
    }


# ----- 1.2 interaction matrix -----


def interaction(agent, runner, items, casting, filler_text: str = " ...", with_removal: bool = True,
                order_kind: str = "canonical", **kw) -> dict:
    """M[i, j] = change in option i's logit when option j's tokens become filler (positions fixed)."""
    fid = filler_token(agent.tok, filler_text)
    base_encs, fill_encs, fill_owner, rem_encs, rem_owner = [], [], [], [], []
    for n, it in enumerate(items):
        k = len(it["options"])
        order = None if order_kind == "canonical" else list(range(k))[::-1]
        e = encode_mc(agent, it, casting, option_order=order)
        base_encs.append(e)
        slot_of = {opt: s for s, opt in enumerate(order or range(k))}
        for j in range(k):
            fill_encs.append(with_filler(e, slot_of[j], fid))
            fill_owner.append((n, j))
            if with_removal:
                keep = [m for m in range(k) if m != j]
                rem_encs.append(encode_mc(agent, it, casting, options=[it["options"][m] for m in keep]))
                rem_owner.append((n, j))
    orders = [e.option_order for e in base_encs]
    base = option_logits(runner, base_encs, orders, **kw)
    fill = option_logits(runner, fill_encs, [orders[n] for n, _ in fill_owner], **kw)
    rem = option_logits(runner, rem_encs, [None] * len(rem_encs), **kw) if with_removal else []

    Ms, Ms_c, Rs = [], [], []
    for n, it in enumerate(items):
        k = len(it["options"])
        Ms.append(np.full((k, k), np.nan))
        Ms_c.append(np.full((k, k), np.nan))
        Rs.append(np.full((k, k), np.nan))
    for (n, j), row in zip(fill_owner, fill):
        s0 = base[n]
        others = [m for m in range(len(s0)) if m != j]
        for i in range(len(s0)):
            Ms[n][i, j] = row[i] - s0[i]
        c0 = s0[others] - s0[others].mean()
        c1 = row[others] - row[others].mean()
        for idx, i in enumerate(others):
            Ms_c[n][i, j] = c1[idx] - c0[idx]
    for (n, j), row in zip(rem_owner, rem):
        s0 = base[n]
        others = [m for m in range(len(s0)) if m != j]
        c0 = s0[others] - s0[others].mean()
        c1 = row - row.mean()
        for idx, i in enumerate(others):
            Rs[n][i, j] = c1[idx] - c0[idx]
    return summarize_interaction(items, base, Ms, Ms_c, Rs, agent, runner) | {
        "casting": casting, "filler": filler_text, "order": order_kind,
        "_M": Ms, "_Mc": Ms_c, "_R": Rs, "_base": base}


def _offdiag(mats):
    vals = []
    for M in mats:
        k = M.shape[0]
        vals += [M[i, j] for i in range(k) for j in range(k) if i != j and not np.isnan(M[i, j])]
    return np.array(vals)


def summarize_interaction(items, base, Ms, Ms_c, Rs, agent=None, runner=None) -> dict:
    off_raw, off_c = _offdiag(Ms), _offdiag(Ms_c)
    out = {
        "n_items": len(items),
        "mean_abs_M_raw": mt.ci_dict(np.abs(off_raw)),
        "mean_abs_M_centered": mt.ci_dict(np.abs(off_c)),
        "mean_M_raw": mt.ci_dict(off_raw),
        "self_effect_raw": mt.ci_dict(np.array([M[j, j] for M in Ms for j in range(M.shape[0])])),
    }
    if Rs and not np.all(np.isnan(_offdiag(Rs))):
        out["mean_abs_removal_centered"] = mt.ci_dict(np.abs(_offdiag(Rs)))
    # symmetry
    a, b = [], []
    for M in Ms_c:
        k = M.shape[0]
        for i in range(k):
            for j in range(i + 1, k):
                if not (np.isnan(M[i, j]) or np.isnan(M[j, i])):
                    a.append(M[i, j])
                    b.append(M[j, i])
    out["symmetry_corr_centered"] = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")
    # distractor suppression: neutralise the strongest distractor and watch the gold margin
    dm, gold_delta = [], []
    for it, s0, M in zip(items, base, Ms):
        g = it["gold"]
        wrong = [m for m in range(len(s0)) if m != g]
        j = max(wrong, key=lambda m: s0[m])
        rest = [m for m in wrong if m != j]
        before = s0[g] - max(s0[m] for m in rest) if rest else np.nan
        after = (s0[g] + M[g, j]) - max(s0[m] + M[m, j] for m in rest) if rest else np.nan
        dm.append(after - before)
        gold_delta.append(M[g, j])
    out["gold_margin_change_when_top_distractor_filled"] = mt.ci_dict(np.array(dm))
    out["gold_logit_change_when_top_distractor_filled"] = mt.ci_dict(np.array(gold_delta))
    # effect of a strong vs weak competitor: does a high-logit option j push others down more?
    strength, eff = [], []
    for s0, M in zip(base, Ms_c):
        for j in range(len(s0)):
            col = [M[i, j] for i in range(len(s0)) if i != j and not np.isnan(M[i, j])]
            if col:
                strength.append(s0[j] - s0.mean())
                eff.append(np.mean(col))
    out["competitor_strength_vs_mean_effect_corr"] = float(np.corrcoef(strength, eff)[0, 1])
    if agent is not None and runner is not None:
        out["similarity_vs_abs_M_corr"] = _similarity_corr(items, Ms_c, agent, runner)
    return out


def _similarity_corr(items, Ms_c, agent, runner) -> float:
    import torch

    emb = runner.enc.embeddings.tok_embeddings.weight
    sims, effs = [], []
    with torch.no_grad():
        for it, M in zip(items, Ms_c):
            vecs = []
            for text in it["options"]:
                ids = agent.tok(" " + text, add_special_tokens=False)["input_ids"][:48]
                v = emb[torch.tensor(ids, device=emb.device)].float().mean(0)
                vecs.append(v / v.norm())
            V = torch.stack(vecs)
            S = (V @ V.T).cpu().numpy()
            k = len(vecs)
            for i in range(k):
                for j in range(k):
                    if i != j and not np.isnan(M[i, j]):
                        sims.append(S[i, j])
                        effs.append(abs(M[i, j]))
    return float(np.corrcoef(sims, effs)[0, 1])


# ----- 1.5 parallel-layout control -----

BANDS_ENC28 = {
    "early_0_9": list(range(0, 10)),
    "mid_10_18": list(range(10, 19)),
    "late_19_27": list(range(19, 28)),
    "head_28_29": [28, 29],
    "enc_global": [l for l in range(28) if l % 3 == 0],
    "enc_local": [l for l in range(28) if l % 3 != 0],
    "enc_all": list(range(28)),
}


def bands_for(runner) -> dict:
    n = runner.n_enc
    third = [list(range(0, n // 3)), list(range(n // 3, 2 * n // 3)), list(range(2 * n // 3, n))]
    head = list(range(n, runner.n_layers))
    glob = [l for l in range(n) if runner.layer_types[l] == "full_attention"]
    return {
        f"early_0_{third[0][-1]}": third[0],
        f"mid_{third[1][0]}_{third[1][-1]}": third[1],
        f"late_{third[2][0]}_{third[2][-1]}": third[2],
        f"head_{head[0]}_{head[-1]}": head,
        "enc_global": glob,
        "enc_local": [l for l in range(n) if l not in glob],
        "enc_all": list(range(n)),
    }


def parallel_conditions(runner) -> dict:
    conds = {
        "sequential": dict(mode="sequential"),
        "masked_parallel_all": dict(mode="masked_parallel", head_masks=True),
        "shipped_parallel_positions": dict(mode="shipped_parallel", head_masks=False, parallel_positions=True),
        "shipped_parallel_positions_headmask": dict(mode="shipped_parallel", head_masks=True,
                                                    parallel_positions=True),
        "options_blocked_seqpos": dict(mode="shipped_parallel", head_masks=True),
    }
    for name, layers in bands_for(runner).items():
        conds[f"band_masked_parallel_{name}"] = dict(mode="masked_parallel", head_masks=True, mask_layers=layers)
    return conds
