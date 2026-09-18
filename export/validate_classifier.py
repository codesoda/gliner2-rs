import json
import numpy as np
import onnxruntime as ort


MODEL_DIR = "models/gliner2-base-v1"
ONNX_PATH = "onnx/gliner2-base-v1/classifier.onnx"


def load_hidden_size():
    encoder_cfg = json.load(open(f"{MODEL_DIR}/encoder_config/config.json"))
    return encoder_cfg["hidden_size"]


def main():
    hidden = load_hidden_size()
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    n = 7
    feeds = {"cls_embeds": np.random.randn(n, hidden).astype(np.float32)}
    (logits,) = sess.run(None, feeds)
    print("logits shape:", logits.shape)


if __name__ == "__main__":
    main()

