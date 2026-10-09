"""Label guard: stop noul answers from following the option label word.

Install on a loaded laya Agent; every later predict / predict_batch call uses it:

    import laya
    from interp.label_guard import install
    agent = laya.load("convaiinnovations/laya")
    install(agent)

For noul rows, each option's [MASK] is blocked from attending to the label tokens (the word before
the ':' in "true: yes, the statement holds") in ModernBERT's local (sliding-window) layers only.
Global layers, the head layers, choice and score rows are untouched. Depends only on laya + torch.
"""

from __future__ import annotations

import torch

NOUL = 2


def _colon_ids(tok) -> torch.Tensor:
    ids = [i for i in range(len(tok)) if ":" in tok.decode([i])]
    return torch.tensor(sorted(ids))


def _window(cfg) -> int:
    sw = getattr(cfg, "sliding_window", None)
    return int(sw) if sw else int(getattr(cfg, "local_attention", 128)) // 2


def label_mask(input_ids, marker_pos, marker_mask, qtype, colon_ids) -> torch.Tensor:
    """[B, T] True on label-word tokens of noul rows: after each [MASK], up to (not incl.) the ':' token."""
    B, T = input_ids.shape
    is_colon = torch.isin(input_ids, colon_ids.to(input_ids.device))
    out = torch.zeros((B, T), dtype=torch.bool, device=input_ids.device)
    for b in torch.nonzero(qtype == NOUL).flatten().tolist():
        ms = marker_pos[b][marker_mask[b]].tolist()
        for s, m in enumerate(ms):
            end = ms[s + 1] if s + 1 < len(ms) else T
            t = m + 1
            while t < end and not bool(is_colon[b, t]):
                out[b, t] = True
                t += 1
    return out


def install(agent) -> None:
    model, tok = agent.model, agent.tok
    enc = model.encoder
    if getattr(model, "_label_guard", False):
        return
    colon = _colon_ids(tok)
    window = _window(enc.config)
    mask_id = tok.mask_token_id
    state = {"masks": None}
    enc_forward, model_forward = enc.forward, model.forward

    def guarded_encoder(*args, attention_mask=None, **kwargs):
        if state["masks"] is not None:
            attention_mask = state["masks"]
        return enc_forward(*args, attention_mask=attention_mask, **kwargs)

    def guarded_model(input_ids, attention_mask, marker_pos, marker_mask, qtype, *args, **kwargs):
        if kwargs.get("option_ids") is not None or not bool((qtype == NOUL).any()):
            return model_forward(input_ids, attention_mask, marker_pos, marker_mask, qtype, *args, **kwargs)
        labels = label_mask(input_ids, marker_pos, marker_mask, qtype, colon)
        T = input_ids.shape[1]
        keys = attention_mask.bool()[:, None, :].expand(-1, T, -1)
        pos = torch.arange(T, device=input_ids.device)
        near = (pos[:, None] - pos[None, :]).abs() <= window
        is_mask = input_ids == mask_id
        block = is_mask[:, :, None] & labels[:, None, :]
        full, sliding = keys, keys & near[None] & ~block
        if enc.config._attn_implementation == "eager":
            neg = torch.finfo(torch.float32).min
            to_add = lambda a: torch.where(a, 0.0, neg).to(torch.float32)[:, None]  # noqa: E731
            masks = {"full_attention": to_add(full), "sliding_attention": to_add(sliding)}
        else:
            masks = {"full_attention": full[:, None], "sliding_attention": sliding[:, None]}
        state["masks"] = masks
        try:
            return model_forward(input_ids, attention_mask, marker_pos, marker_mask, qtype, *args, **kwargs)
        finally:
            state["masks"] = None

    enc.forward = guarded_encoder
    model.forward = guarded_model
    model._label_guard = True
