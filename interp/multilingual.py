"""Track 4.3: does an English jailbreak direction, fit on laya-multilingual, steer Hindi inputs?"""

from __future__ import annotations

import numpy as np
import torch

from . import metrics as mt


@torch.no_grad()
def translate(texts: list[str], tgt: str = "hin_Deva", model_id: str = "facebook/nllb-200-distilled-600M",
              batch_size: int = 16) -> list[str]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float16).cuda().eval()
    out = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        gen = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tgt), max_new_tokens=300,
                             num_beams=2)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    del model
    torch.cuda.empty_cache()
    return out


def run(agent, runner, pairs: list[dict], natural: list[dict], log=print) -> dict:
    from . import safety as sf
    from .verify import run as verify_run

    out: dict = {}
    v = verify_run(agent, n=50, bench=False)
    out["verify"] = {k: v[k] for k in ("prob_maxdiff_vs_predict", "prob_pass_1e-4_after_rounding", "window_chosen")}
    log(f"[multilingual] verify {out['verify']}")

    pairs = pairs[:200]
    hi_jb = translate([p["jailbreak"] for p in pairs])
    hi_ct = translate([p["control"] for p in pairs])
    out["examples"] = [{"en": p["jailbreak"], "hi": h} for p, h in zip(pairs[:3], hi_jb[:3])]

    tr = [r for r in natural if r["split"] == "main"][:1500]
    S_tr, M_tr = sf.layer_features(agent, runner, [r["text"] for r in tr])
    ytr = np.array([r["label"] for r in tr])
    en_jb = sf.score_texts(agent, runner, [p["jailbreak"] for p in pairs])
    en_ct = sf.score_texts(agent, runner, [p["control"] for p in pairs])
    hj = sf.score_texts(agent, runner, hi_jb)
    hc = sf.score_texts(agent, runner, hi_ct)
    out["baseline"] = {"en_jb_frac_true": float((en_jb > 0).mean()), "en_ctl_frac_true": float((en_ct > 0).mean()),
                       "hi_jb_frac_true": float((hj > 0).mean()), "hi_ctl_frac_true": float((hc > 0).mean())}
    log(f"[multilingual] baseline {out['baseline']}")

    # pick the layer by an English probe, fit the diff-of-means direction on English only
    from sklearn.linear_model import LogisticRegression

    accs = []
    half = len(tr) // 2
    for l in range(runner.n_layers):
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(S_tr[:half, l], ytr[:half])
        accs.append(clf.score(S_tr[half:, l], ytr[half:]))
    best = int(np.argmax(accs))
    d, sigma = sf.diff_means_direction(S_tr, M_tr, ytr, best, "state")
    out["layer"], out["english_probe_acc"] = best, accs
    S_hi, _ = sf.layer_features(agent, runner, hi_jb + hi_ct)
    proj = S_hi[:, best] @ d
    lab = np.array([1] * len(hi_jb) + [0] * len(hi_ct))
    out["hindi_projection_auroc"] = mt.auroc(proj, lab)
    steer = {}
    for a in (0, 1, 2, 4, 8):
        e = sf.steer_edit(d, sigma, a, best, "state")
        lo = sf.score_texts(agent, runner, hi_ct, edits=e)
        steer[a] = {"hi_control_frac_true": float((lo > 0).mean()), "mean_logodds": float(lo.mean())}
    out["steer_hindi_controls"] = steer
    lo_abl = sf.score_texts(agent, runner, hi_jb, edits=sf.ablate_direction_edit(d, best))
    out["ablate_hindi_jailbreaks"] = {"before": float((hj > 0).mean()), "after": float((lo_abl > 0).mean())}
    rnd = np.random.default_rng(0).standard_normal(d.shape)
    rnd /= np.linalg.norm(rnd)
    lo_rnd = sf.score_texts(agent, runner, hi_ct, edits=sf.steer_edit(rnd, sigma, 8, best, "state"))
    out["random_direction_alpha8_frac_true"] = float((lo_rnd > 0).mean())
    log(f"[multilingual] L{best} hindi proj auroc={out['hindi_projection_auroc']:.3f} steer={steer} "
        f"ablate={out['ablate_hindi_jailbreaks']} random@8={out['random_direction_alpha8_frac_true']:.3f}")
    return out
