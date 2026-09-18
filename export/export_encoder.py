import argparse
import os

import torch
from gliner2 import GLiNER2


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


def parse_args():
    parser = argparse.ArgumentParser(description="Export GLiNER2 encoder to ONNX")
    parser.add_argument("--model-dir", default="models/gliner2-base-v1", help="Path to local model directory")
    parser.add_argument("--out-dir", default="onnx/gliner2-base-v1", help="Output directory for ONNX files")
    parser.add_argument("--device", default="cpu", help="Torch device (cpu/cuda)")
    parser.add_argument("--batch", type=int, default=2, help="Dummy batch size for export graph")
    parser.add_argument("--seq-len", type=int, default=16, help="Dummy sequence length for export graph")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device
    model = GLiNER2.from_pretrained(args.model_dir).to(device)
    model.eval()

    wrapper = EncoderWrapper(model.encoder).to(device).eval()

    # small dummy inputs for export graph
    batch, seq_len = args.batch, args.seq_len
    dummy_ids = torch.ones((batch, seq_len), dtype=torch.long, device=device)
    dummy_mask = torch.ones((batch, seq_len), dtype=torch.long, device=device)

    out_path = os.path.join(args.out_dir, "encoder.onnx")

    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=args.opset,
    )

    print(f"Saved encoder ONNX to {out_path}")


if __name__ == "__main__":
    main()
