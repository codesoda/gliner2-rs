import os
import torch
from gliner2 import GLiNER2


MODEL_DIR = "models/gliner2-base-v1"
OUT_DIR = "onnx/gliner2-base-v1"


class ExtractorWrapper(torch.nn.Module):
    def __init__(self, span_rep: torch.nn.Module, count_pred: torch.nn.Module, count_embed: torch.nn.Module):
        super().__init__()
        self.span_rep = span_rep
        self.count_pred = count_pred
        self.count_embed = count_embed

    def forward(self, text_emb: torch.Tensor, schema_emb: torch.Tensor, spans_idx: torch.Tensor):
        # text_emb: [L, H]
        # schema_emb: [S, H] where S = 1 + num_fields (includes [P] then one marker per field)
        # spans_idx: [L * max_width, 2] arranged in the same order as compute_span_rep
        text_emb = text_emb.unsqueeze(0)
        schema_emb = schema_emb.unsqueeze(0)
        spans_idx = spans_idx.unsqueeze(0)

        span_mask = (spans_idx < 0).any(dim=-1)
        safe_spans = torch.where(span_mask.unsqueeze(-1), torch.zeros_like(spans_idx), spans_idx)

        # span representations: [1, L, max_width, H] -> [L, max_width, H]
        span_rep = self.span_rep(text_emb, safe_spans).squeeze(0)

        # count logits from [P] embeddings (first special token per schema)
        count_logits = self.count_pred(schema_emb[:, 0, :])

        # structure projections for full max_count to keep ONNX output shape stable
        pc_emb = schema_emb[0, 1:, :]
        max_count = getattr(self.count_embed, "max_count", 20)
        struct_proj = self.count_embed(pc_emb, max_count)  # [max_count, num_fields, H]

        # span_scores: [max_count, num_fields, L, max_width]
        span_scores = torch.einsum("lkd,pmd->pmlk", span_rep, struct_proj)

        return count_logits, span_scores


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    device = "cpu"
    model = GLiNER2.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    wrapper = ExtractorWrapper(model.span_rep, model.count_pred, model.count_embed).to(device).eval()

    hidden = model.hidden_size
    max_width = model.max_width

    # Example sizes for export; dynamic shapes will generalize these dimensions.
    text_len = 8
    schema_tokens = 6  # 1 + num_fields
    num_spans = text_len * max_width

    dummy_text_emb = torch.randn((text_len, hidden), dtype=torch.float32, device=device)
    dummy_schema_emb = torch.randn((schema_tokens, hidden), dtype=torch.float32, device=device)
    dummy_spans_idx = torch.zeros((num_spans, 2), dtype=torch.long, device=device)

    out_path = os.path.join(OUT_DIR, "extractor_dynamic.onnx")

    torch.onnx.export(
        wrapper,
        (dummy_text_emb, dummy_schema_emb, dummy_spans_idx),
        out_path,
        input_names=["text_emb", "schema_emb", "spans_idx"],
        output_names=["count_logits", "span_scores"],
        opset_version=17,
        dynamo=True,
        # Use the legacy-style symbolic axes for better compatibility with the exporter.
        dynamic_axes={
            "text_emb": {0: "text_len"},
            "schema_emb": {0: "schema_tokens"},
            "spans_idx": {0: "num_spans"},
            "span_scores": {1: "num_fields", 2: "text_len"},
        },
    )

    print(f"Saved dynamic extractor ONNX to {out_path}")


if __name__ == "__main__":
    main()
