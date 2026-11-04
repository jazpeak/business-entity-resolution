#!/usr/bin/env python3
"""Publish the trained models to the Hugging Face Hub.

The five fine-tuned transformer weights (~5.5 GB total) are too large for a
normal git repo, so they are hosted on the HF Hub. This uploads the whole
`models/` directory (weights, configs, tokenizers, the LightGBM prefilter and
the XGBoost fusion) to a single model repo.

Usage:
    export HF_TOKEN=hf_...                       # a write token from https://huggingface.co/settings/tokens
    python scripts/upload_models_to_hf.py --repo <your-hf-user>/business-entity-resolution-models

Run from the repository root. Requires `huggingface_hub` (in requirements.txt).
"""
import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="Target HF repo id, e.g. your-user/business-entity-resolution-models")
    ap.add_argument("--models-dir", default="models", help="Local models directory to upload (default: models)")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF write token (or set HF_TOKEN)")
    ap.add_argument("--private", action="store_true", help="Create the repo as private")
    ap.add_argument("--commit-message", default="Upload business entity resolution models")
    args = ap.parse_args()

    if not args.token:
        print("ERROR: no token. Pass --token or set HF_TOKEN.", file=sys.stderr)
        return 2

    models_dir = Path(args.models_dir).resolve()
    if not models_dir.is_dir():
        print(f"ERROR: models dir not found: {models_dir}", file=sys.stderr)
        return 2

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub not installed. `pip install -r requirements.txt`.", file=sys.stderr)
        return 2

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)
    print(f"Uploading {models_dir} -> https://huggingface.co/{args.repo}")
    api.upload_folder(
        folder_path=str(models_dir),
        repo_id=args.repo,
        repo_type="model",
        commit_message=args.commit_message,
        # don't re-upload git/OS cruft
        ignore_patterns=["**/__pycache__/**", "**/.DS_Store", "**/*.log"],
    )
    print(f"Done. View at https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
