import argparse
import os

import torch
from gliner2 import GLiNER2


class ClassifierWrapper(torch.nn.Module):
    def __init__(self, classifier: torch.nn.Module):
        super().__init__()
        self.classifier = classifier

    def forward(self, cls_embeds: torch.Tensor):
        # cls_embeds: [N, H]
        logits = self.classifier(cls_embeds).squeeze(-1)  # [N]
        return logits


def parse_args():
    parser = argparse.ArgumentParser(description="Export GLiNER2 classifier head to ONNX")
    parser.add_argument("--model-dir", default="models/gliner2-base-v1", help="Path to local model directory")
    parser.add_argument("--out-dir", default="onnx/gliner2-base-v1", help="Output directory for ONNX files")
    parser.add_argument("--device", default="cpu", help="Torch device (cpu/cuda)")
    parser.add_argument("--num-labels", type=int, default=5, help="Dummy number of labels for export graph")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device
    model = GLiNER2.from_pretrained(args.model_dir).to(device)
    model.eval()

    wrapper = ClassifierWrapper(model.classifier).to(device).eval()

    hidden = model.hidden_size
    num_labels = args.num_labels
    dummy = torch.randn((num_labels, hidden), dtype=torch.float32, device=device)

    out_path = os.path.join(args.out_dir, "classifier.onnx")

    torch.onnx.export(
        wrapper,
        (dummy,),
        out_path,
        input_names=["cls_embeds"],
        output_names=["logits"],
        dynamic_axes={"cls_embeds": {0: "num_labels"}, "logits": {0: "num_labels"}},
        opset_version=args.opset,
    )

    print(f"Saved classifier ONNX to {out_path}")


if __name__ == "__main__":
    main()
