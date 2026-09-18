import argparse
import os

import torch
from gliner2 import GLiNER2


def parse_args():
    parser = argparse.ArgumentParser(description="Export GLiNER2 extractor head to ONNX (padded schema)")
    parser.add_argument("--model-dir", default="models/gliner2-base-v1", help="Path to local model directory")
    parser.add_argument("--out-dir", default="onnx/gliner2-base-v1", help="Output directory for ONNX files")
    parser.add_argument("--device", default="cpu", help="Torch device (cpu/cuda)")
    # Max number of fields/labels supported by this ONNX (excluding the [P] embedding).
    # This is a pragmatic cap; we provide a mask so padding does not affect real fields.
    parser.add_argument("--max-fields", type=int, default=64, help="Max number of schema fields (excludes [P])")
    parser.add_argument("--text-len", type=int, default=8, help="Dummy text length for export graph")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return parser.parse_args()


class ExtractorWrapper(torch.nn.Module):
    def __init__(self, span_rep: torch.nn.Module, count_pred: torch.nn.Module, count_embed: torch.nn.Module, max_fields: int):
        super().__init__()
        self.span_rep = span_rep
        self.count_pred = count_pred
        self.count_embed = count_embed
        self.max_fields = max_fields

    def forward(self, text_emb: torch.Tensor, schema_emb_padded: torch.Tensor, schema_mask: torch.Tensor, spans_idx: torch.Tensor):
        # text_emb: [L, H]
        # schema_emb_padded: [1 + max_fields, H] = [P] + padded field marker embeddings
        # schema_mask: [max_fields] (1 for real fields, 0 for padding)
        # spans_idx: [L * max_width, 2] arranged in the same order as compute_span_rep

        text_emb = text_emb.unsqueeze(0)
        schema_emb_padded = schema_emb_padded.unsqueeze(0)
        spans_idx = spans_idx.unsqueeze(0)

        span_mask = (spans_idx < 0).any(dim=-1)
        safe_spans = torch.where(span_mask.unsqueeze(-1), torch.zeros_like(spans_idx), spans_idx)

        # span representations: [1, L, max_width, H] -> [L, max_width, H]
        span_rep = self.span_rep(text_emb, safe_spans).squeeze(0)

        # count logits from [P] embeddings (first special token per schema)
        count_logits = self.count_pred(schema_emb_padded[:, 0, :])

        # Fixed-size pc_emb (padded).
        pc_emb = schema_emb_padded[0, 1:, :]  # [max_fields, H]
        # Always incorporate the mask so the exported graph keeps `schema_mask` as an input.
        # For models without a cross-field transformer (e.g. CountLSTM), this also ensures padded
        # fields have zero embeddings and do not affect outputs.
        pc_emb = pc_emb * schema_mask.to(pc_emb.dtype).unsqueeze(-1)
        max_count = getattr(self.count_embed, "max_count", 20)

        if hasattr(self.count_embed, "transformer"):
            # ---- CountLSTMv2 forward (copied, but with transformer padding mask) ----
            # Build positional sequence: (max_count, max_fields, H)
            full_idx = torch.arange(max_count, device=pc_emb.device)
            pos_seq = self.count_embed.pos_embedding(full_idx)  # (max_count, H)
            pos_seq = pos_seq.unsqueeze(1).expand(-1, self.max_fields, -1)  # (max_count, max_fields, H)

            # GRU over count steps (independent per field)
            h0 = pc_emb.unsqueeze(0)  # (1, max_fields, H)
            output, _ = self.count_embed.gru(pos_seq, h0)  # (max_count, max_fields, H)
            pc_broadcast = pc_emb.unsqueeze(0).expand_as(output)
            x = output + pc_broadcast  # (max_count, max_fields, H)

            # Transformer across fields, with padding mask to ignore padded positions.
            # Transformer is batch_first=True, so (batch=max_count, seq=max_fields, hidden)
            padding = ~schema_mask.to(torch.bool)  # True = padding
            padding = padding.unsqueeze(0).expand(max_count, -1)  # (max_count, max_fields)

            tr = self.count_embed.transformer  # DownscaledTransformer
            x_proj = tr.in_projector(x)
            x_proj = tr.transformer(x_proj, src_key_padding_mask=padding)
            x_cat = torch.cat([x_proj, x], dim=-1)
            struct_proj = tr.out_projector(x_cat)  # (max_count, max_fields, H)
        else:
            # Older models (e.g. count_lstm) expose the projection as a callable module.
            struct_proj = self.count_embed(pc_emb, max_count)  # (max_count, max_fields, H)

        # span_scores: [max_count, max_fields, L, max_width]
        span_scores = torch.einsum("lkd,pmd->pmlk", span_rep, struct_proj)

        return count_logits, span_scores


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device
    model = GLiNER2.from_pretrained(args.model_dir).to(device)
    model.eval()

    wrapper = ExtractorWrapper(model.span_rep, model.count_pred, model.count_embed, args.max_fields).to(device).eval()

    hidden = model.hidden_size
    max_width = model.max_width
    max_count = getattr(model.count_embed, "max_count", 20)

    text_len = args.text_len
    num_spans = text_len * max_width

    dummy_text_emb = torch.randn((text_len, hidden), dtype=torch.float32, device=device)
    dummy_schema_emb = torch.randn((1 + args.max_fields, hidden), dtype=torch.float32, device=device)
    dummy_schema_mask = torch.ones((args.max_fields,), dtype=torch.bool, device=device)
    dummy_spans_idx = torch.zeros((num_spans, 2), dtype=torch.long, device=device)

    out_path = os.path.join(args.out_dir, "extractor_padded.onnx")

    torch.onnx.export(
        wrapper,
        (dummy_text_emb, dummy_schema_emb, dummy_schema_mask, dummy_spans_idx),
        out_path,
        input_names=["text_emb", "schema_emb_padded", "schema_mask", "spans_idx"],
        output_names=["count_logits", "span_scores"],
        dynamic_axes={
            "text_emb": {0: "text_len"},
            "spans_idx": {0: "num_spans"},
            "span_scores": {2: "text_len"},
        },
        opset_version=args.opset,
    )

    print(f"Saved padded extractor ONNX to {out_path}")
    print(f"span_scores shape: [{max_count}, {args.max_fields}, text_len, {max_width}]")


if __name__ == "__main__":
    main()
