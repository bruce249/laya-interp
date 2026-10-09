"""Phase 1 (plan v4): 2x2 factorial for C1 on the same weights.

Factors: option-text <-> option-text attention {allowed, blocked} x option positions {original, shared}, with
[MASK] binding held fixed in all four cells (each [MASK] reads only its own option). Sharing positions with
unrestricted attention makes every [MASK] identical (same token, same position, same keys), so that condition is
kept only as a documented degenerate reference.
Plus a matched random-position control (attention enabled): every option block is shifted by a random
offset with the same mean absolute size as the shift that "shared" applies, so the model is disturbed by
a similar amount but the order of options is still encoded.

Primary metrics (frozen): reversal-flip rate and accuracy, reported jointly; mean abs(M) (centered).
Gate (frozen): if "shared positions" alone costs more than 10 accuracy points, flag it.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np

from . import competition as cp
from . import metrics as mt
from .hooks import slot_logits, to_option_order

GATE_ACC_DROP = 0.10


# ----- frozen confirmatory MC set -----

def build_mc_confirm(seed: int = 7, n_obqa: int = 500, n_mmlu: int = 500) -> list[dict]:
    from datasets import load_dataset

    from .data import _clean

    rng = random.Random(seed)
    items = []
    rows = list(load_dataset("allenai/openbookqa", "main", split="test"))
    rng.shuffle(rows)
    for r in rows[:n_obqa]:
        labels, texts = r["choices"]["label"], [_clean(t) for t in r["choices"]["text"]]
        if len(set(texts)) == len(texts) and r["answerKey"] in labels:
            items.append({"id": f"OBQA:{r['id']}", "source": "OpenBookQA", "question": _clean(r["question_stem"]),
                          "options": texts, "gold": labels.index(r["answerKey"])})
    subjects = ["high_school_biology", "high_school_geography", "high_school_psychology", "nutrition",
                "conceptual_physics", "high_school_world_history"]
    pool = []
    for sub in subjects:
        for i, r in enumerate(load_dataset("cais/mmlu", sub, split="test")):
            texts = [_clean(t) for t in r["choices"]]
            if len(set(texts)) == 4 and all(len(t) <= 200 for t in texts):
                pool.append({"id": f"MMLU:{sub}:{i}", "source": f"MMLU-{sub}", "question": _clean(r["question"]),
                             "options": texts, "gold": int(r["answer"])})
    rng.shuffle(pool)
    items += pool[:n_mmlu]
    for it in items:
        it["split"] = "confirm"
    return items


def mc_confirm() -> list[dict]:
    from . import data

    path = data.DATA / "mc_confirm.jsonl"
    if not path.exists():
        data.save_jsonl("mc_confirm.jsonl", build_mc_confirm())
    return data.load_jsonl("mc_confirm.jsonl")


# ----- position functions -----

def jitter_positions(enc) -> list[int]:
    """Shift each option block by a random integer offset whose mean absolute size matches the shift that
    shared positions would apply. Seeded by item id and slot, so the jitter is fixed per slot."""
    m0 = enc.markers[0]
    spans = enc.option_spans
    disp = [a - m0 for a, _ in spans]
    D = max(1, int(round(np.mean(np.abs(disp)))))
    seed = int(hashlib.sha1(str(enc.meta.get("id", "")).encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    pos = list(range(m0))
    ends = []
    for s, (a, b) in enumerate(spans):
        start = max(1, a + rng.randint(-2 * D, 2 * D))
        pos += [start + k for k in range(b - a)]
        ends.append(start + (b - a))
    after = max([enc.head_len - 1] + ends)
    pos += [after + k for k in range(len(enc.ids) - len(pos))]
    return pos


CONDITIONS = {
    # reference
    "base": dict(),
    "jitter": dict(position_fn=jitter_positions),
    "shared_only_degenerate": dict(parallel_positions=True),   # [MASK]s become indistinguishable
    # 2x2 with [MASK] binding held fixed (each [MASK] reads only its own option); encoder masks only
    "f00_bind_origpos": dict(mode="bind_masks", head_masks=False),
    "f10_block_origpos": dict(mode="shipped_parallel", head_masks=False),
    "f01_bind_shared": dict(mode="bind_masks", head_masks=False, parallel_positions=True),
    "f11_block_shared": dict(mode="shipped_parallel", head_masks=False, parallel_positions=True),
}


# ----- runs -----

def flips_and_acc(agent, runner, items, casting, kw):
    encs = [cp.encode_mc(agent, it, casting) for it in items]
    rev_orders = [list(range(len(it["options"])))[::-1] for it in items]
    encs_rev = [cp.encode_mc(agent, it, casting, option_order=o) for it, o in zip(items, rev_orders)]
    s = [to_option_order(r, None) for r in slot_logits(runner, encs, **kw)]
    sr = [to_option_order(r, o) for r, o in zip(slot_logits(runner, encs_rev, **kw), rev_orders)]
    gold = np.array([it["gold"] for it in items])
    pred = np.array([np.argmax(r) for r in s])
    pred_r = np.array([np.argmax(r) for r in sr])
    return (pred != pred_r).astype(float), (pred == gold).astype(float)


def paired_effects(F: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """2x2 effects with paired bootstrap CIs over items. F maps condition -> per-item array."""
    rng = np.random.default_rng(seed)
    n = len(F["base"])
    idx = rng.integers(0, n, size=(n_boot, n))

    def stat(fn):
        est = fn(np.arange(n))
        boots = np.array([fn(i) for i in idx])
        return {"est": float(est), "lo": float(np.quantile(boots, .025)), "hi": float(np.quantile(boots, .975))}

    m = lambda c, i: F[c][i].mean()  # noqa: E731
    return {
        "position_effect_text_interaction_on": stat(lambda i: m("f00_bind_origpos", i) - m("f01_bind_shared", i)),
        "position_effect_text_interaction_blocked": stat(lambda i: m("f10_block_origpos", i) - m("f11_block_shared", i)),
        "interaction_effect_positions_original": stat(lambda i: m("f00_bind_origpos", i) - m("f10_block_origpos", i)),
        "interaction_effect_positions_shared": stat(lambda i: m("f01_bind_shared", i) - m("f11_block_shared", i)),
        "position_x_interaction": stat(lambda i: (m("f00_bind_origpos", i) - m("f01_bind_shared", i))
                                       - (m("f10_block_origpos", i) - m("f11_block_shared", i))),
        "binding_mask_cost": stat(lambda i: m("base", i) - m("f00_bind_origpos", i)),
        "jitter_effect": stat(lambda i: m("base", i) - m("jitter", i)),
    }


def run(agent, runner, model: str, log=print, n_inter: int = 600) -> dict:
    from .runs import mc_items

    runner.force_fp32 = True
    items = mc_items()
    sets = {"explore_main": [it for it in items if it["split"] == "main"],
            "explore_heldout": [it for it in items if it["split"] == "heldout"],
            "confirm": mc_confirm()}
    out: dict = {"model": model, "precision": "fp32", "conditions": list(CONDITIONS)}
    for sname, subset in sets.items():
        flips, accs, res = {}, {}, {}
        for cname, kw in CONDITIONS.items():
            f, a = flips_and_acc(agent, runner, subset, "generic", kw)
            flips[cname], accs[cname] = f, a
            inter = cp.interaction(agent, runner, subset[:n_inter], "generic", with_removal=False, **kw)
            res[cname] = {"reversal_flip_rate": mt.ci_dict(f, n=1000), "accuracy": mt.ci_dict(a, n=1000),
                          "mean_abs_M_centered": inter["mean_abs_M_centered"]}
            log(f"[{model} {sname}] {cname}: flips={f.mean():.3f} acc={a.mean():.3f} "
                f"|M|c={inter['mean_abs_M_centered']['est']:.3f}")
        acc_drop_B = float(accs["f00_bind_origpos"].mean() - accs["f01_bind_shared"].mean())
        out[sname] = {"n_items": len(subset), "per_condition": res,
                      "flip_effects": paired_effects(flips), "accuracy_effects": paired_effects(accs),
                      "gate_B_accuracy_drop": acc_drop_B, "gate_B_failed": acc_drop_B > GATE_ACC_DROP}
        fe = out[sname]["flip_effects"]
        log(f"[{model} {sname}] flips: position effect (text on)={fe['position_effect_text_interaction_on']['est']:+.3f} "
            f"(text blocked)={fe['position_effect_text_interaction_blocked']['est']:+.3f} "
            f"text-interaction effect={fe['interaction_effect_positions_original']['est']:+.3f} "
            f"pos x int={fe['position_x_interaction']['est']:+.3f} binding cost={fe['binding_mask_cost']['est']:+.3f} "
            f"jitter={fe['jitter_effect']['est']:+.3f} | shared-pos acc drop={acc_drop_B:+.3f} "
            f"gate_failed={acc_drop_B > GATE_ACC_DROP}")
    runner.force_fp32 = False
    return out
