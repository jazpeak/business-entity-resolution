"""Train or score an mmBERT-base (jhu-clsp/mmBERT-base, MIT) cross-encoder; same input format and interface as ce_mdeberta.py."""
import os as _os
ER_ROOT = _os.environ.get("ER_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # package root (code/business_entity_resolution)

import argparse
import math
import os
import random

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

DEFAULT_BASE = os.environ.get("MMBERT_BASE", "jhu-clsp/mmBERT-base")
MAXLEN = 128


class Pairs(Dataset):
    def __init__(self, df):
        self.d = df.to_dict(as_series=False)

    def __len__(self):
        return len(self.d["s1"])

    def __getitem__(self, i):
        left = f"name: {self.d['name1'][i]} | address: {self.d['addr1'][i]}"
        right = f"name: {self.d['name2'][i]} | address: {self.d['addr2'][i]}"
        return left, right, self.d["label"][i], self.d["s1"][i], self.d["a"][i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "score"])
    ap.add_argument("--parts", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "mmBERT final training/scoring requires CUDA"
    assert args.mode == "train" or args.out, "--out is required in score mode"
    torch.manual_seed(17); random.seed(17); np.random.seed(17)

    wanted = args.parts.split(",")
    cols = ["s1", "a", "label", "name1", "addr1", "name2", "addr2"]
    df = pl.scan_parquet(args.data).filter(pl.col("ce_part").is_in(wanted)).select(cols).collect()
    assert df.height > 0, f"no rows for parts {wanted} in {args.data}"
    assert df.select("s1", "a").unique().height == df.height, "duplicate (s1, a) input rows"

    source = args.base if args.mode == "train" else args.model
    local_only = args.offline or args.mode == "score"
    tok = AutoTokenizer.from_pretrained(source, local_files_only=local_only, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=1, local_files_only=local_only, dtype=torch.float32
    )
    model.cuda(); model.train(args.mode == "train")

    def collate(rows):
        left, right, labels, s1, a = zip(*rows)
        enc = tok(list(left), list(right), padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
        return enc, torch.tensor(labels, dtype=torch.float32), s1, a

    loader = DataLoader(
        Pairs(df), batch_size=args.batch, shuffle=args.mode == "train", num_workers=args.workers,
        pin_memory=True, collate_fn=collate, persistent_workers=args.workers > 0,
    )
    if args.mode == "train":
        steps = math.ceil(len(loader) * args.epochs)
        assert steps > 0
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, max(1, steps // 20), steps)
        opt.zero_grad(set_to_none=True)
        for step, (enc, labels, _, _) in enumerate(loader):
            if step >= steps:
                break
            enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
            labels = labels.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc).logits.squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            assert torch.isfinite(loss), f"non-finite loss at step {step}: {loss.item()}"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if step % 250 == 0:
                print(step, steps, float(loss), flush=True)
        os.makedirs(args.model, exist_ok=True)
        model.save_pretrained(args.model); tok.save_pretrained(args.model)
    else:
        rows = []
        with torch.inference_mode():
            for step, (enc, _, s1, a) in enumerate(loader):
                enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**enc).logits.squeeze(-1).float().cpu().numpy()
                assert np.isfinite(logits).all(), f"non-finite logits at batch {step}"
                rows.append(pl.DataFrame({"s1": s1, "a": a, "ce_logit": logits}))
                if step % 500 == 0:
                    print(step, len(loader), flush=True)
        scored = pl.concat(rows)
        assert scored.height == df.height
        assert scored.select("s1", "a").unique().height == scored.height
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        scored.write_parquet(args.out)


if __name__ == "__main__":
    main()
