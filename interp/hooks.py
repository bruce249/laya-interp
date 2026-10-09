"""Explicit forward pass of laya's DecisionModel with named sites for caching, patching and steering.

Layer index l runs over the encoder layers (0..L_enc-1) and then the head layers (L_enc, L_enc+1).
Sites, per layer l:
    L{l}.resid_pre   [B, T, d]       residual stream entering the layer
    L{l}.z           [B, T, H, Dh]   per-head attention output before the output projection
    L{l}.pattern     [B, H, T, T]    attention probabilities (cache only)
    L{l}.attn_out    [B, T, d]
    L{l}.mlp_out     [B, T, d]
Global sites:
    enc_final        [B, T, d]       encoder output after final_norm
    head_in          [B, T, d]       enc_final + type embedding
    final            [B, T, d]       after the head layers
    marker_h         [B, K, d]       `final` gathered at the [MASK]s, the scorer input

`edits` maps a site to fn(tensor, batch) -> tensor. `cache` is a set of site names (or "all").
Mask modes:
    sequential        the trained layout
    masked_parallel   option spans cannot see other option spans, shared tokens cannot see options
    shipped_parallel  laya.common.parallel_option_masks semantics: other options blocked only;
                      use with collate(parallel_positions=True) for the package's position ids
`mask_layers` restricts the non-sequential mask to those layer indices (default: all layers).
"""

from __future__ import annotations

import contextlib
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

Edit = Callable[[torch.Tensor, dict], torch.Tensor]


class Runner:
    def __init__(self, agent):
        from transformers.models.modernbert import modeling_modernbert as mb

        self.agent = agent
        self.model = agent.model
        self.enc = agent.model.encoder
        self.cfg = self.enc.config
        self.rope = mb.apply_rotary_pos_emb
        self.n_enc = len(self.enc.layers)
        self.head_layers = list(self.model.head.layers) if self.model.head is not None else []
        self.n_layers = self.n_enc + len(self.head_layers)
        self.H = self.cfg.num_attention_heads
        self.d = self.cfg.hidden_size
        self.Dh = self.d // self.H
        self.Hh = self.head_layers[0].self_attn.num_heads if self.head_layers else 0
        self.layer_types = list(self.cfg.layer_types)
        self.window = window_from_config(self.cfg)
        self.device = next(self.model.parameters()).device
        self.eager = False   # True forces the explicit-softmax path everywhere
        self.force_fp32 = False   # True ignores amp=True requests (fp32 confirmation runs)

    # ----- masks -----
    def build_masks(self, batch: dict, mode: str = "sequential"):
        """Boolean allowed[B, T, T] for full and sliding layers, sequential and alternative."""
        keys = batch["attention_mask"].bool()[:, None, :]
        pos = batch["position_ids"]
        near = (pos[:, :, None] - pos[:, None, :]).abs() <= self.window
        base_full = keys.expand(-1, keys.shape[-1], -1)
        seq = {"full": base_full, "sliding": base_full & near}
        if mode == "sequential":
            return seq, seq
        opt = batch["option_ids"]
        oq, ok = opt[:, :, None], opt[:, None, :]
        blocked = (oq > 0) & (ok > 0) & (oq != ok)
        if mode == "masked_parallel":
            blocked = blocked | ((oq == 0) & (ok > 0))
        elif mode == "bind_masks":
            # binding only: a [MASK] may not read other options; option text may still read other options
            from .inputs import SEG_MASK
            blocked = blocked & (batch["segment"] == SEG_MASK)[:, :, None]
        elif mode != "shipped_parallel":
            raise ValueError(f"unknown mask mode {mode!r}")
        alt = {"full": base_full & ~blocked, "sliding": base_full & ~blocked & near}
        return seq, alt

    @staticmethod
    def additive(allowed: torch.Tensor, dtype) -> torch.Tensor:
        zero = torch.zeros((), dtype=dtype, device=allowed.device)
        neg = torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=allowed.device)
        out = torch.where(allowed, zero, neg)
        return out[:, None] if out.dim() == 3 else out

    # ----- forward -----
    def forward(self, batch: dict, *, mode: str = "sequential", mask_layers: Iterable[int] | None = None,
                head_masks: bool | None = None, edits: dict[str, Edit] | None = None,
                cache: Iterable[str] | str | None = None, amp: bool = False,
                cross_option_heads: dict[int, list[int]] | None = None,
                rope_alt_heads: dict[int, list[int]] | None = None,
                edge_blocks: list | None = None):
        """Returns (logits [B, K] raw, act_logits [B, n_act], cache dict).

        cross_option_heads: {layer: [heads]} whose attention between different option spans is blocked.
        rope_alt_heads: {layer: [heads]} that rotate q/k with batch["alt_position_ids"] (the
            order-blind parallel layout) instead of the sequential positions. Masks are unchanged.
        """
        self._cross = cross_option_heads or {}
        self._rope_alt = rope_alt_heads or {}
        # edge_blocks: [(layers, fn(batch) -> bool [B, T, T] of blocked query->key edges)], all heads
        self._edge_blocks = edge_blocks or []
        edits = edits or {}
        store: dict[str, torch.Tensor] = {}
        want_all = cache == "all"
        want = set() if (cache is None or want_all) else set(cache)
        keep = lambda name: want_all or name in want  # noqa: E731

        def site(name, x):
            if name in edits:
                x = edits[name](x, batch)
            if keep(name):
                store[name] = x
            return x

        mask_layers = set(range(self.n_layers)) if mask_layers is None else set(mask_layers)
        if head_masks is None:
            head_masks = mode == "masked_parallel"
        amp = amp and not self.force_fp32
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()
        with ctx:
            return self._forward(batch, mode, mask_layers, head_masks, site, keep, store)

    def _forward(self, batch, mode, mask_layers, head_masks, site, keep, store):
        ids, att, pos = batch["input_ids"], batch["attention_mask"], batch["position_ids"]
        seq, alt = self.build_masks(batch, mode)
        x = self.enc.embeddings(input_ids=ids)
        dtype = x.dtype if not torch.is_autocast_enabled() else torch.bfloat16
        rope = {t: self.enc.rotary_emb(x, pos, t) for t in set(self.layer_types)}
        rope_alt = ({t: self.enc.rotary_emb(x, batch["alt_position_ids"], t) for t in set(self.layer_types)}
                    if self._rope_alt else None)
        opt = batch["option_ids"]
        cross = ((opt[:, :, None] > 0) & (opt[:, None, :] > 0) & (opt[:, :, None] != opt[:, None, :])
                 if self._cross else None)
        self._edge_cache = [(set(layers), fn(batch)) for layers, fn in self._edge_blocks]

        for l, layer in enumerate(self.enc.layers):
            x = site(f"L{l}.resid_pre", x)
            kind = "full" if self.layer_types[l] == "full_attention" else "sliding"
            allowed = (alt if l in mask_layers else seq)[kind]
            a_in = layer.attn_norm(x)
            B, T, _ = a_in.shape
            qkv = layer.attn.Wqkv(a_in).view(B, T, 3, self.H, self.Dh)
            q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
            cos, sin = rope[self.layer_types[l]]
            if l in self._rope_alt:
                heads = self._rope_alt[l]
                q_alt, k_alt = self.rope(q, k, *rope_alt[self.layer_types[l]], unsqueeze_dim=1)
                q, k = self.rope(q, k, cos, sin, unsqueeze_dim=1)
                q, k = q.clone(), k.clone()
                q[:, heads], k[:, heads] = q_alt[:, heads], k_alt[:, heads]
            else:
                q, k = self.rope(q, k, cos, sin, unsqueeze_dim=1)
            allowed = self._per_head(allowed, cross, l, self.H)
            z = self._attend(q, k, v, allowed, f"L{l}.pattern", keep, store)   # [B, T, H, Dh]
            z = site(f"L{l}.z", z)
            attn_out = site(f"L{l}.attn_out", layer.attn.Wo(z.reshape(B, T, self.d)))
            x = x + attn_out
            mlp_out = site(f"L{l}.mlp_out", layer.mlp(layer.mlp_norm(x)))
            x = x + mlp_out

        x = site("enc_final", self.enc.final_norm(x))
        x = site("head_in", x + self.model.type_emb(batch["qtype"])[:, None, :])

        pad_allowed = att.bool()[:, None, :].expand(-1, att.shape[1], -1)
        for j, layer in enumerate(self.head_layers):
            l = self.n_enc + j
            x = site(f"L{l}.resid_pre", x)
            allowed = alt["full"] if (head_masks and l in mask_layers and mode != "sequential") else pad_allowed
            sa = layer.self_attn
            allowed = self._per_head(allowed, cross, l, sa.num_heads)
            a_in = layer.norm1(x)
            B, T, _ = a_in.shape
            Dh = self.d // sa.num_heads
            q, k, v = (t.view(B, T, sa.num_heads, Dh).transpose(1, 2)
                       for t in F.linear(a_in, sa.in_proj_weight, sa.in_proj_bias).chunk(3, dim=-1))
            z = site(f"L{l}.z", self._attend(q, k, v, allowed, f"L{l}.pattern", keep, store))
            attn_out = site(f"L{l}.attn_out", sa.out_proj(z.reshape(B, T, self.d)))
            x = x + attn_out
            mlp_out = site(f"L{l}.mlp_out", layer.linear2(layer.activation(layer.linear1(layer.norm2(x)))))
            x = x + mlp_out

        x = site("final", x)
        idx = batch["marker_pos"].clamp(min=0)[:, :, None].expand(-1, -1, x.size(-1))
        marker_h = site("marker_h", torch.gather(x, 1, idx))
        logits = self.model.scorer(marker_h).squeeze(-1).float()
        logits = logits.masked_fill(~batch["marker_mask"], -1e4)
        act = self._act(x, logits, batch["marker_mask"])
        return logits, act, store

    def _per_head(self, allowed, cross, layer, n_heads):
        for layers, blocked in self._edge_cache:
            if layer in layers:
                allowed = allowed & ~blocked
                # never leave a query row with nothing to attend to
                allowed = allowed | (~allowed.any(-1, keepdim=True) & torch.eye(
                    allowed.shape[-1], dtype=torch.bool, device=allowed.device))
        if layer not in self._cross:
            return allowed
        per = allowed[:, None].expand(-1, n_heads, -1, -1).clone()
        heads = self._cross[layer]
        per[:, heads] = per[:, heads] & ~cross[:, None]
        return per

    def _attend(self, q, k, v, allowed, name, keep, store):
        """Attention over [B, H, T, Dh]; explicit softmax when the pattern is cached, SDPA otherwise."""
        if keep(name) or self.eager:
            scores = torch.matmul(q, k.transpose(2, 3)).float() * q.shape[-1] ** -0.5
            scores = scores + self.additive(allowed, torch.float32)
            p = torch.softmax(scores, dim=-1).to(v.dtype)
            if keep(name):
                store[name] = p
            return torch.matmul(p, v).transpose(1, 2)
        mask = allowed[:, None] if allowed.dim() == 3 else allowed
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2)

    def _act(self, x, logits, marker_mask):
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        if p.size(-1) >= 2:
            top2 = p.topk(2, -1).values
        else:
            top1 = p.topk(1, -1).values
            top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        act_in = torch.cat([x[:, 0].float(), feats], -1).to(self.model.act_head[0].weight.dtype)
        return self.model.act_head(act_in)

    # ----- helpers -----
    def head_contribution(self, z: torch.Tensor, layer: int, head: int) -> torch.Tensor:
        """Write of one head into the residual stream: z[..., head, :] @ W_O[:, head block].T."""
        if layer < self.n_enc:
            W = self.enc.layers[layer].attn.Wo.weight
            Dh = self.Dh
        else:
            W = self.head_layers[layer - self.n_enc].self_attn.out_proj.weight
            Dh = self.d // self.head_layers[layer - self.n_enc].self_attn.num_heads
        block = W[:, head * Dh:(head + 1) * Dh]
        return z[..., head, :] @ block.T

    def n_heads(self, layer: int) -> int:
        return self.H if layer < self.n_enc else self.head_layers[layer - self.n_enc].self_attn.num_heads


@torch.no_grad()
def slot_logits(runner: Runner, encs, *, batch_size: int = 128, parallel_positions: bool = False,
                position_fn=None, **forward_kwargs) -> list:
    """Raw logits per item, in slot order (numpy, length k). Sorted by length to limit padding."""
    from .inputs import collate, to_device

    order = sorted(range(len(encs)), key=lambda i: len(encs[i].ids))
    out = [None] * len(encs)
    pad = runner.agent.tok.pad_token_id
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = to_device(collate([encs[i] for i in idx], pad, parallel_positions=parallel_positions,
                                  position_fn=position_fn), runner.device)
        logits, _, _ = runner.forward(batch, **forward_kwargs)
        logits = logits.float().cpu().numpy()
        for row, i in enumerate(idx):
            out[i] = logits[row, : encs[i].k]
    return out


def to_option_order(slot_row, option_order):
    """Slot-ordered logits back to option order (option_order[s] is the option in slot s)."""
    import numpy as np

    if option_order is None:
        return np.asarray(slot_row)
    row = np.empty(len(slot_row))
    row[np.asarray(option_order)] = slot_row
    return row


def window_from_config(cfg) -> int:
    """Half-width of ModernBERT's local attention: |i - j| <= window."""
    sw = getattr(cfg, "sliding_window", None)
    if sw:
        return int(sw)
    return int(getattr(cfg, "local_attention", 128)) // 2


def probs(logits: torch.Tensor, temperature: torch.Tensor, marker_mask: torch.Tensor) -> torch.Tensor:
    z = logits / temperature[:, None]
    z = z.masked_fill(~marker_mask, -1e9)
    return torch.softmax(z, -1)
