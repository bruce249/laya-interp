"""Load laya checkpoints and expose the raw DecisionModel for hooking."""

from __future__ import annotations

import torch

CHECKPOINTS = {
    "laya": dict(repo="convaiinnovations/laya", subfolder=None),
    "smolaya": dict(repo="anon767tom/smolaya", subfolder=None),
    "multilingual": dict(repo="convaiinnovations/laya-multilingual", subfolder=None),
    "ctrl_soft_ce": dict(repo="/store/ckpt/ctrl_soft_ce", subfolder=None),
    "ctrl_rlcd": dict(repo="/store/ckpt/ctrl_rlcd", subfolder=None),
}


def load_agent(name: str = "laya", device: str = "cuda", fp32: bool = True, eager_attn: bool = True):
    """laya.Agent on the eager backend, AMP off when `fp32`, eager encoder attention when `eager_attn`."""
    import laya

    spec = CHECKPOINTS[name]
    kwargs = {"device": device}
    if spec["subfolder"]:
        kwargs["subfolder"] = spec["subfolder"]
    agent = laya.load(spec["repo"], **kwargs)
    try:
        agent.set_backend("eager")
    except Exception:
        pass
    if fp32:
        agent.amp_enabled = False
        agent.dtype = torch.float32
    agent.model.eval()
    if eager_attn:
        set_attn_impl(agent.model.encoder, "eager")
    return agent


def set_attn_impl(encoder, impl: str) -> None:
    if hasattr(encoder, "set_attn_implementation"):
        encoder.set_attn_implementation(impl)
    else:
        encoder.config._attn_implementation = impl
