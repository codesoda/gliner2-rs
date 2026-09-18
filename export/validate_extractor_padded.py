import json
import numpy as np
import onnxruntime as ort


MODEL_DIR = "models/gliner2-base-v1"
ONNX_PATH = "onnx/gliner2-base-v1/extractor_padded.onnx"


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
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    schema_emb_shape = sess.get_inputs()[1].shape  # [1 + MAX_FIELDS, H]
    max_fields = int(schema_emb_shape[0]) - 1

    text_len = 7
    feeds_base = {
        "text_emb": np.random.randn(text_len, hidden_size).astype(np.float32),
        "schema_emb_padded": np.random.randn(1 + max_fields, hidden_size).astype(np.float32),
        "spans_idx": build_spans(text_len, max_width),
    }

    for n_fields in [1, 2, 5, min(17, max_fields), max_fields]:
        schema_mask = np.zeros((max_fields,), dtype=np.bool_)
        schema_mask[:n_fields] = True

        feeds = dict(feeds_base)
        feeds["schema_mask"] = schema_mask

        count_logits, span_scores = sess.run(None, feeds)
        print("n_fields:", n_fields)
        print("  count_logits:", count_logits.shape)
        print("  span_scores :", span_scores.shape)


if __name__ == "__main__":
    main()

