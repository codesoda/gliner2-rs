#!/usr/bin/env python3
"""Generate complete-distribution classification goldens from the pinned upstream.

Every case goes through the public ``extract`` path of the pinned GLiNER2
package. A forward hook on ``model.classifier`` records each task's raw logits
without changing anything. Probabilities are then computed here with torch
from ``raw_logits / classification_temperature`` so the Rust side is compared
against an independent activation, not against its own arithmetic.

Output is one JSON file per checkpoint, small enough to commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
EXPORT_DIR = SCRIPT_DIR.parent / "export"
sys.path.insert(0, str(EXPORT_DIR))

from bundle_profiles import PROFILES, verify_source  # noqa: E402
from gliner2.processing.word_splitter import word_splitter_from  # noqa: E402
from common import (  # noqa: E402
    GLINER2_COMMIT,
    configure_determinism,
    load_reference_model,
    package_versions,
)

SEED = 1729
FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--model-dir", required=True, help="verified source snapshot")
    parser.add_argument("--corpus", default=str(SCRIPT_DIR / "classification_corpus.json"))
    parser.add_argument("--out-dir", default="fixtures/classification-scores")
    return parser.parse_args()


def expand_case(raw: dict[str, Any]) -> dict[str, Any]:
    case = dict(raw)
    if "text" not in case:
        case["text"] = " ".join([case.pop("repeat_word")] * int(case.pop("repeat_count"))) + case.pop("tail")
    return case


def load_corpus(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    document = json.loads(raw)
    if document["format_version"] != FORMAT_VERSION:
        raise AssertionError("unexpected corpus format version")
    cases = [expand_case(case) for case in document["cases"]]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise AssertionError("corpus case IDs must be unique")
    return cases, hashlib.sha256(raw).hexdigest()


def build_schema(model, tasks: list[dict[str, Any]]):
    schema = model.create_schema()
    for task in tasks:
        config = dict(task)
        name = config.pop("task")
        labels = config.pop("labels")
        schema.classification(name, labels, **config)
    return schema


def request_view(task: dict[str, Any]) -> dict[str, Any]:
    labels = task["labels"]
    if isinstance(labels, dict):
        names = list(labels)
        descriptions = [[label, labels[label]] for label in names]
    else:
        names = list(labels)
        descriptions = []
    multi_label = bool(task.get("multi_label", False))
    class_act = task.get("class_act", "auto")
    if class_act == "auto":
        activation = "sigmoid" if multi_label else "softmax"
    else:
        activation = class_act
    return {
        "task": task["task"],
        "labels": names,
        "instruction": task.get("prompt"),
        "label_descriptions": descriptions,
        "multi_label": multi_label,
        "cls_threshold": float(task.get("cls_threshold", 0.5)),
        "class_act": class_act,
        "activation": activation,
    }


def activate(logits: torch.Tensor, temperature: float, activation: str) -> list[float]:
    scaled = logits / temperature
    if activation == "softmax":
        return torch.softmax(scaled, dim=-1).tolist()
    if activation == "sigmoid":
        return torch.sigmoid(scaled).tolist()
    raise ValueError(activation)


def main() -> None:
    args = parse_args()
    configure_determinism(SEED)
    profile = PROFILES[args.profile]
    source_hashes = verify_source(args.profile, args.model_dir)
    model, config_data = load_reference_model(args.model_dir)
    head = getattr(model.config, "boundary_head", None)
    if isinstance(head, dict):
        temperature = float(head.get("classification_temperature", 1.0))
    elif head is not None:
        temperature = float(getattr(head, "classification_temperature", 1.0))
    else:
        temperature = float(config_data.get("boundary_head", {}).get("classification_temperature", 1.0))
    max_len = int(config_data.get("max_len", 4096))

    logits_seen: list[torch.Tensor] = []
    encoder_lengths: list[int] = []

    def classifier_hook(_module, _args, output):
        logits_seen.append(output.detach().to(torch.float32).squeeze(-1).cpu().clone())

    def encoder_hook(_module, args, kwargs, _output):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        encoder_lengths.append(int(input_ids.shape[-1]))

    splitter = word_splitter_from(model)
    if splitter is None:
        raise RuntimeError("model has no word splitter attached")

    handles = [
        model.classifier.register_forward_hook(classifier_hook),
        model.encoder.register_forward_hook(encoder_hook, with_kwargs=True),
    ]

    cases, corpus_sha256 = load_corpus(Path(args.corpus))
    results = []
    try:
        for case in cases:
            logits_seen.clear()
            encoder_lengths.clear()
            schema = build_schema(model, case["tasks"])
            with torch.inference_mode():
                public = model.extract(
                    case["text"],
                    schema,
                    threshold=0.5,
                    format_results=True,
                    include_confidence=True,
                    max_len=max_len,
                )
            if len(logits_seen) != len(case["tasks"]):
                raise AssertionError(
                    f"{case['id']}: classifier ran {len(logits_seen)} times for {len(case['tasks'])} tasks"
                )
            if len(encoder_lengths) != 1:
                raise AssertionError(f"{case['id']}: encoder ran {len(encoder_lengths)} times")
            words = len(case["text"].split())
            # The cap applies to upstream splitter tokens, not whitespace words.
            split_tokens = sum(1 for _ in splitter(case["text"]))
            tasks_out = []
            for task, logits in zip(case["tasks"], logits_seen):
                view = request_view(task)
                if logits.ndim != 1 or logits.shape[0] != len(view["labels"]):
                    raise AssertionError(f"{case['id']}: logits shape {tuple(logits.shape)}")
                view["raw_logits"] = logits.tolist()
                view["probabilities"] = activate(logits, temperature, view["activation"])
                view["public_result"] = public[view["task"]]
                tasks_out.append(view)
            print(
                f"[{case['id']}] tokens={encoder_lengths[0]} words={words} tasks={len(tasks_out)}",
                flush=True,
            )
            results.append(
                {
                    "id": case["id"],
                    "text": case["text"],
                    "word_count": words,
                    "input_tokens": encoder_lengths[0],
                    "split_tokens": split_tokens,
                    "truncated_words": max(0, split_tokens - max_len),
                    "tasks": tasks_out,
                }
            )
    finally:
        for handle in handles:
            handle.remove()

    document = {
        "format_version": FORMAT_VERSION,
        "profile": args.profile,
        "bundle_name": profile.bundle_name,
        "hf_model": profile.model_id,
        "hf_revision": profile.revision,
        "gliner2_commit": GLINER2_COMMIT,
        "source_sha256": source_hashes,
        "corpus_sha256": corpus_sha256,
        "classification_temperature": temperature,
        "max_len": max_len,
        "seed": SEED,
        "dependencies": package_versions(),
        "cases": results,
    }
    out = Path(args.out_dir) / f"{profile.bundle_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, ensure_ascii=False, indent=1, allow_nan=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
