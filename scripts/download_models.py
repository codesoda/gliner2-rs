#!/usr/bin/env python3
"""Download the ONNX model exports from Hugging Face into ./onnx.

Usage:
    python3 scripts/download_models.py [--model base|large|all]

Requires: pip install huggingface_hub
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "codesoda/gliner2-onnx"
MODELS = {
    "base": "gliner2-base-v1",
    "large": "gliner2-large-v1",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["base", "large", "all"],
        default="all",
        help="Which model bundle to download (default: all)",
    )
    parser.add_argument(
        "--dest",
        default=str(Path(__file__).resolve().parent.parent / "onnx"),
        help="Destination directory (default: ./onnx)",
    )
    args = parser.parse_args()

    names = list(MODELS.values()) if args.model == "all" else [MODELS[args.model]]
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    for name in names:
        print(f"Downloading {name} from {REPO_ID} ...")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="model",
            allow_patterns=[f"{name}/*"],
            local_dir=str(dest),
        )
    print(f"Done. Models available under {dest}/")


if __name__ == "__main__":
    main()
