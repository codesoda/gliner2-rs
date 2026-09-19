#!/usr/bin/env python3
"""Export the shared fp32 GLiNER2 classifier MLP with a dynamic row axis."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import (
    assert_finite,
    configure_determinism,
    copy_runtime_metadata,
    load_reference_model,
    normalize_onnx_mask_constants,
    update_partial_manifest,
)


class ClassifierWrapper(torch.nn.Module):
    def __init__(self, classifier: torch.nn.Module):
        super().__init__()
        self.classifier = classifier

    def forward(self, cls_embeds: torch.Tensor):
        return self.classifier(cls_embeds).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/gliner2-base-v1")
    parser.add_argument("--out-dir", default="onnx/gliner2-base-v1")
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--num-labels", type=int, default=5)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.opset != 17:
        raise ValueError("GLiNER2 bundles require opset 17")
    configure_determinism(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, model_config = load_reference_model(args.model_dir, device=args.device)
    state = model.state_dict()
    required = (
        "classifier.0.weight",
        "classifier.0.bias",
        "classifier.3.weight",
        "classifier.3.bias",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise KeyError(f"checkpoint is missing classifier keys: {missing}")
    hidden = int(model.hidden_size)
    if tuple(state["classifier.0.weight"].shape) != (2 * hidden, hidden):
        raise ValueError("classifier.0.weight has an unexpected shape")
    if tuple(state["classifier.3.weight"].shape) != (1, 2 * hidden):
        raise ValueError("classifier.3.weight has an unexpected shape")

    wrapper = ClassifierWrapper(model.classifier).float().eval()
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    dummy = torch.randn(
        (args.num_labels, hidden), generator=generator, dtype=torch.float32
    )
    with torch.inference_mode():
        assert_finite("classifier export probe", wrapper(dummy))

    out_path = out_dir / "classifier.onnx"
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=["cls_embeds"],
        output_names=["logits"],
        dynamic_axes={
            "cls_embeds": {0: "rows"},
            "logits": {0: "rows"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    mask_audit = normalize_onnx_mask_constants(out_path)
    copy_runtime_metadata(args.model_dir, out_dir)
    temperature = float(
        model_config.get("boundary_head", {}).get(
            "classification_temperature", 1.0
        )
    )
    manifest = update_partial_manifest(
        out_dir,
        args.model_dir,
        opset=args.opset,
        graph_name="classifier.onnx",
        graph_signature={
            "inputs": {"cls_embeds": ["rows", hidden]},
            "outputs": {"logits": ["rows"]},
            "classifier_keys": list(required),
            "classification_temperature": temperature,
            "mask_audit": mask_audit,
        },
    )
    print(f"saved {out_path}")
    print(f"updated incomplete M1 manifest {manifest}")


if __name__ == "__main__":
    main()
