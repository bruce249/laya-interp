"""Datasets for the study. Built on Modal and stored on the laya-interp Volume under /store/data."""

from __future__ import annotations

import json
import pathlib
import random

DATA = pathlib.Path("/store/data")

# ----- multiple choice (C1, C2, C5) -----

CASTINGS = {
    # instructions, state
    "generic": lambda q: ("Which option correctly answers the question in the text?", q),
    "question_both": lambda q: (q, q),
    "question_only": lambda q: (q, ""),
}


def _clean(text: str) -> str:
    return " ".join(str(text).split())


def load_mc(seed: int = 0, n_arc_c: int = 500, n_arc_e: int = 300, n_csqa: int = 700, n_qasc: int = 500,
            qasc_options: int = 4) -> list[dict]:
    from datasets import load_dataset

    rng = random.Random(seed)
    items: list[dict] = []

    def add(source, uid, question, options, gold):
        options = [_clean(o) for o in options]
        if len(set(options)) != len(options) or not (3 <= len(options) <= 5):
            return
        items.append({"id": f"{source}:{uid}", "source": source, "question": _clean(question),
                      "options": options, "gold": gold})

    for config, n in (("ARC-Challenge", n_arc_c), ("ARC-Easy", n_arc_e)):
        ds = load_dataset("allenai/ai2_arc", config, split="test")
        rows = list(ds)
        rng.shuffle(rows)
        for row in rows[:n]:
            labels = row["choices"]["label"]
            if row["answerKey"] not in labels:
                continue
            add(config, row["id"], row["question"], row["choices"]["text"], labels.index(row["answerKey"]))

    ds = load_dataset("tau/commonsense_qa", split="validation")
    rows = list(ds)
    rng.shuffle(rows)
    for row in rows[:n_csqa]:
        labels = row["choices"]["label"]
        add("CSQA", row["id"], row["question"], row["choices"]["text"], labels.index(row["answerKey"]))

    ds = load_dataset("allenai/qasc", split="validation")
    rows = list(ds)
    rng.shuffle(rows)
    for row in rows[:n_qasc]:
        labels, texts = row["choices"]["label"], row["choices"]["text"]
        gold = labels.index(row["answerKey"])
        distractors = [i for i in range(len(texts)) if i != gold]
        keep = sorted(rng.sample(distractors, qasc_options - 1) + [gold])
        add("QASC", row["id"], row["question"], [texts[i] for i in keep], keep.index(gold))

    rng.shuffle(items)
    for i, item in enumerate(items):
        item["split"] = "heldout" if i % 10 < 3 else "main"
    return items


def mc_query(item: dict, casting: str = "generic", option_order: list[int] | None = None,
             options: list[str] | None = None):
    """(state, qdef) for one MC item. Option keys are the answer texts, rendered without descriptions."""
    ins, state = CASTINGS[casting](item["question"])
    opts = options if options is not None else item["options"]
    qdef = {"type": "choice", "instructions": ins, "criteria": {o: None for o in opts}}
    if option_order is not None:
        qdef["option_order"] = list(option_order)
    return state, qdef


# ----- persistence -----

def save_jsonl(name: str, rows: list[dict]) -> pathlib.Path:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / name
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_jsonl(name: str) -> list[dict]:
    with (DATA / name).open() as f:
        return [json.loads(line) for line in f]
