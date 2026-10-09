"""Build laya sequences exactly as the package does, plus the token spans the experiments need.

Layout (laya.common.build_sequence):
    [CLS] <type> question: <ins> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

SEG_SPECIAL, SEG_QUESTION, SEG_OPTION, SEG_STATE, SEG_MASK = 0, 1, 2, 3, 4
N_SEG = 5


@dataclass
class Encoded:
    ids: list[int]
    markers: list[int]              # [MASK] index of each slot
    qtype: int
    head_len: int                   # tokens up to and including the head's closing [SEP]
    option_spans: list[tuple[int, int]]   # [start, end) of each slot, starting at its [MASK]
    question_span: tuple[int, int]
    state_span: tuple[int, int]
    temperature: float
    option_order: list[int] | None = None
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.markers)


def temperature_for(agent, qtype: int, k: int) -> float:
    from laya.common import temp_bucket

    return float(agent.temperature_by_options.get(temp_bucket(qtype, k), agent.temperature[qtype]))


def encode(agent, state, qdef: dict, max_len: int | None = None, head_max_len: int | None = None,
           meta: dict | None = None) -> Encoded:
    from laya.common import build_head

    internal = agent._to_internal(qdef)
    item = agent._encode_state(state, ["q"], {"q": internal}, max_len, head_max_len)[0]
    head_max_len = agent.cfg.get("head_max_len", 192) if head_max_len is None else head_max_len
    head_ids, _, _ = build_head(agent.tok, internal, head_max_len, option_order=internal.get("option_order"))
    head_len = len(head_ids)
    ids, markers = item["ids"], item["markers"]
    ends = markers[1:] + [head_len - 1]
    return Encoded(
        ids=ids,
        markers=markers,
        qtype=item["qtype"],
        head_len=head_len,
        option_spans=list(zip(markers, ends)),
        question_span=(1, markers[0] - 1),
        state_span=(head_len, len(ids) - 1),
        temperature=temperature_for(agent, item["qtype"], len(markers)),
        option_order=internal.get("option_order"),
        meta=dict(meta or {}),
    )


def collate(encs: list[Encoded], pad_id: int, parallel_positions: bool = False, position_fn=None) -> dict:
    """Tensors for `Runner.forward`. `option_ids` is s+1 on slot s tokens, 0 elsewhere."""
    from laya.common import parallel_layout

    n, length = len(encs), max(len(e.ids) for e in encs)
    kmax = max(e.k for e in encs)
    batch = {
        "input_ids": torch.full((n, length), pad_id, dtype=torch.long),
        "attention_mask": torch.zeros((n, length), dtype=torch.long),
        "marker_pos": torch.zeros((n, kmax), dtype=torch.long),
        "marker_mask": torch.zeros((n, kmax), dtype=torch.bool),
        "qtype": torch.tensor([e.qtype for e in encs]),
        "position_ids": torch.arange(length).repeat(n, 1),
        "alt_position_ids": torch.arange(length).repeat(n, 1),
        "option_ids": torch.zeros((n, length), dtype=torch.long),
        "segment": torch.zeros((n, length), dtype=torch.long),
        "temperature": torch.tensor([e.temperature for e in encs]),
    }
    for i, e in enumerate(encs):
        size = len(e.ids)
        batch["input_ids"][i, :size] = torch.tensor(e.ids)
        batch["attention_mask"][i, :size] = 1
        batch["marker_pos"][i, : e.k] = torch.tensor(e.markers)
        batch["marker_mask"][i, : e.k] = True
        for s, (a, b) in enumerate(e.option_spans):
            batch["option_ids"][i, a:b] = s + 1
            batch["segment"][i, a:b] = SEG_OPTION
        for m in e.markers:
            batch["segment"][i, m] = SEG_MASK
        batch["segment"][i, e.question_span[0]:e.question_span[1]] = SEG_QUESTION
        batch["segment"][i, e.state_span[0]:e.state_span[1]] = SEG_STATE
        for name, positions in e.meta.get("masks", {}).items():
            if name not in batch:
                batch[name] = torch.zeros((n, length), dtype=torch.bool)
            batch[name][i, list(positions)] = True
        layout = torch.tensor(parallel_layout(e.markers, e.head_len, size)["position_ids"])
        batch["alt_position_ids"][i, :size] = layout
        if parallel_positions:
            batch["position_ids"][i, :size] = layout
        elif position_fn is not None:
            batch["position_ids"][i, :size] = torch.tensor(position_fn(e))
    names = {k for e in encs for k in e.meta.get("masks", {})}
    for name in names:
        batch.setdefault(name, torch.zeros((n, length), dtype=torch.bool))
    return batch


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
