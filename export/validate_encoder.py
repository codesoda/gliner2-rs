import numpy as np
import onnxruntime as ort


def main():
    sess = ort.InferenceSession("onnx/gliner2-base-v1/encoder.onnx", providers=["CPUExecutionProvider"])
    batch, seq_len = 2, 12
    feeds = {
        "input_ids": np.ones((batch, seq_len), dtype=np.int64),
        "attention_mask": np.ones((batch, seq_len), dtype=np.int64),
    }
    outs = sess.run(None, feeds)
    print("last_hidden_state shape:", outs[0].shape)


if __name__ == "__main__":
    main()
