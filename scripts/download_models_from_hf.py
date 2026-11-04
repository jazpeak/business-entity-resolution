#!/usr/bin/env python3
"""Download the trained models from the Hugging Face Hub into `models/`.

Fetches the fine-tuned transformer weights (and the LightGBM/XGBoost models)
published by scripts/upload_models_to_hf.py, placing them so the pipeline can
load them directly from `models/`.

Usage:
    python scripts/download_models_from_hf.py --repo <your-hf-user>/business-entity-resolution-models

Run from the repository root. Requires `huggingface_hub` (in requirements.txt).
A public repo needs no token; a private one needs --token or HF_TOKEN.
"""
import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="Source HF repo id, e.g. your-user/business-entity-resolution-models")
    ap.add_argument("--dest", default="models", help="Local destination directory (default: models)")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token for private repos (or set HF_TOKEN)")
    args = ap.parse_args()

    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. `pip install -r requirements.txt`.", file=sys.stderr)
        return 2

    print(f"Downloading https://huggingface.co/{args.repo} -> {dest}")
    snapshot_download(
        repo_id=args.repo,
        repo_type="model",
        local_dir=str(dest),
        token=args.token,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
