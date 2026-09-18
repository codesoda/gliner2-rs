import json
import numpy as np
import onnxruntime as ort


MODEL_DIR = "models/gliner2-base-v1"
ONNX_PATH = "onnx/gliner2-base-v1/extractor_dynamic.onnx"


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


def run_case(sess, text_len: int, schema_tokens: int, hidden_size: int, max_width: int):
    feeds = {
        "text_emb": np.random.randn(text_len, hidden_size).astype(np.float32),
        "schema_emb": np.random.randn(schema_tokens, hidden_size).astype(np.float32),
        "spans_idx": build_spans(text_len, max_width),
    }
    outs = sess.run(None, feeds)
    print(
        f"text_len={text_len}, schema_tokens={schema_tokens} -> "
        f"count_logits={outs[0].shape}, span_scores={outs[1].shape}"
    )


def main():
    hidden_size, max_width = load_dims()
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    # Vary schema_tokens to confirm the model is truly dynamic over field count.
    run_case(sess, text_len=6, schema_tokens=3, hidden_size=hidden_size, max_width=max_width)
    run_case(sess, text_len=6, schema_tokens=6, hidden_size=hidden_size, max_width=max_width)
    run_case(sess, text_len=10, schema_tokens=12, hidden_size=hidden_size, max_width=max_width)


if __name__ == "__main__":
    main()

