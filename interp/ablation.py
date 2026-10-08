"""Mean and resample ablation of heads and MLPs, using per-segment means from a reference set."""

from __future__ import annotations

import torch

from .inputs import N_SEG, collate, to_device


@torch.no_grad()
def segment_means(runner, encs, batch_size: int = 64, amp: bool = True) -> dict:
    """{"z": [per layer: (N_SEG, H, Dh)], "mlp": [per layer: (N_SEG, d)]} averaged over real tokens."""
    L = runner.n_layers
    sums_z = [None] * L
    sums_m = [None] * L
    counts = torch.zeros(N_SEG, device=runner.device)

    def acc(kind, l):
        def f(x, batch):
            seg = batch["segment"][batch["attention_mask"].bool()]
            flat = x[batch["attention_mask"].bool()].float()
            target = sums_z if kind == "z" else sums_m
            if target[l] is None:
                target[l] = torch.zeros((N_SEG,) + flat.shape[1:], device=x.device)
            target[l].index_add_(0, seg, flat)
            return x
        return f

    edits = {}
    for l in range(L):
        edits[f"L{l}.z"] = acc("z", l)
        edits[f"L{l}.mlp_out"] = acc("mlp", l)
    order = sorted(range(len(encs)), key=lambda i: len(encs[i].ids))
    for start in range(0, len(order), batch_size):
        batch = to_device(collate([encs[i] for i in order[start:start + batch_size]],
                                  runner.agent.tok.pad_token_id), runner.device)
        seg = batch["segment"][batch["attention_mask"].bool()]
        counts.index_add_(0, seg, torch.ones_like(seg, dtype=torch.float))
        runner.forward(batch, edits=edits, amp=amp)
    denom = counts.clamp(min=1)
    view_z = lambda t: t / denom.view(-1, *([1] * (t.dim() - 1)))  # noqa: E731
    return {"z": [view_z(t) for t in sums_z], "mlp": [view_z(t) for t in sums_m], "counts": counts.cpu()}


def mean_heads(means: dict, layer_heads: dict[int, list[int]]) -> dict:
    edits = {}
    for l, heads in layer_heads.items():
        table = means["z"][l]

        def f(z, batch, table=table, heads=list(heads)):
            z = z.clone()
            rep = table[batch["segment"]].to(z.dtype)            # [B, T, H, Dh]
            z[:, :, heads] = rep[:, :, heads]
            return z
        edits[f"L{l}.z"] = f
    return edits


def mean_mlps(means: dict, layers) -> dict:
    edits = {}
    for l in layers:
        table = means["mlp"][l]
        edits[f"L{l}.mlp_out"] = lambda x, batch, table=table: table[batch["segment"]].to(x.dtype)
    return edits


def resample_heads(layer_heads: dict[int, list[int]], shift: int = 1) -> dict:
    """Replace head outputs with those of another row in the batch (rows are length-sorted)."""
    edits = {}
    for l, heads in layer_heads.items():
        def f(z, batch, heads=list(heads)):
            z = z.clone()
            perm = torch.roll(torch.arange(z.shape[0], device=z.device), shift)
            z[:, :, heads] = z[perm][:, :, heads]
            return z
        edits[f"L{l}.z"] = f
    return edits


def merge(*edit_dicts) -> dict:
    """Compose edits that touch the same site, applied left to right."""
    out: dict = {}
    for d in edit_dicts:
        for site, fn in d.items():
            if site in out:
                prev = out[site]
                out[site] = lambda x, b, prev=prev, fn=fn: fn(prev(x, b), b)
            else:
                out[site] = fn
    return out
