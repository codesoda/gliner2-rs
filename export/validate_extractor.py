import json
import numpy as np
import onnxruntime as ort


MODEL_DIR = "models/gliner2-base-v1"
ONNX_PATH = "onnx/gliner2-base-v1/extractor.onnx"


def load_dims():
    encoder_cfg = json.load(open(f"{MODEL_DIR}/encoder_config/config.json"))
    model_cfg = json.load(open(f"{MODEL_DIR}/config.json"))
    return encoder_cfg["hidden_size"], model_cfg["max_width"]


def build_spans(text_len: int, max_width: int):
    spans = []
    for i in range(text_len):
        for j in range(max_width):
            if i + j < text_len:
                spans.append([i, i + j])
            else:
                spans.append([-1, -1])
    return np.array(spans, dtype=np.int64)


def main():
    hidden_size, max_width = load_dims()
    text_len = 6
    schema_tokens = 6  # [P] + a few specials for dummy call

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    feeds = {
        "text_emb": np.random.randn(text_len, hidden_size).astype(np.float32),
        "schema_emb": np.random.randn(schema_tokens, hidden_size).astype(np.float32),
        "spans_idx": build_spans(text_len, max_width),
    }
    outs = sess.run(None, feeds)
    print("count_logits shape:", outs[0].shape)
    print("span_scores shape:", outs[1].shape)


if __name__ == "__main__":
    main()
