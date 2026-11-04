"""Train or score an mDeBERTa-v3-base cross-encoder on 'name: ... | address: ...' pairs (fp32 master weights, bf16 autocast).
Train: python ce_mdeberta.py train --parts A0 --model <out> --data <parquet>
Score: python ce_mdeberta.py score --parts <parts> --model <dir> --data <parquet> --out <parquet>"""
import os as _os
ER_ROOT = _os.environ.get("ER_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # package root (code/business_entity_resolution)

import argparse, math, os, random
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

BASE = f"{ER_ROOT}/models/base_mdeberta_v3"
DATA = "artifacts/ce_candidates.parquet"
MAXLEN = 128

class Pairs(Dataset):
    def __init__(self, df): self.d = df.to_dict(as_series=False)
    def __len__(self): return len(self.d["s1"])
    def __getitem__(self, i):
        a = f"name: {self.d['name1'][i]} | address: {self.d['addr1'][i]}"
        b = f"name: {self.d['name2'][i]} | address: {self.d['addr2'][i]}"
        return a, b, self.d["label"][i], self.d["s1"][i], self.d["a"][i]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["train", "score"])
    ap.add_argument("--parts", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--out"); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--workers", type=int, default=8); args = ap.parse_args()
    torch.manual_seed(17); random.seed(17); np.random.seed(17)
    wanted = args.parts.split(",")
    cols = ["s1", "a", "label", "name1", "addr1", "name2", "addr2"]
    df = pl.scan_parquet(args.data).filter(pl.col("ce_part").is_in(wanted)).select(cols).collect()
    tok = AutoTokenizer.from_pretrained(BASE if args.mode == "train" else args.model, local_files_only=True, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(BASE if args.mode == "train" else args.model, num_labels=1, local_files_only=True,
        dtype=torch.float32)  # transformers>=5 loads this checkpoint as fp16 by default -> AdamW NaN; keep fp32 master weights
    model.cuda(); model.train(args.mode == "train")
    def collate(rows):
        x, y, lab, s1, a = zip(*rows)
        enc = tok(list(x), list(y), padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
        return enc, torch.tensor(lab, dtype=torch.float32), s1, a
    loader = DataLoader(Pairs(df), batch_size=args.batch, shuffle=args.mode == "train", num_workers=args.workers,
                        pin_memory=True, collate_fn=collate, persistent_workers=args.workers > 0)
    if args.mode == "train":
        steps = math.ceil(len(loader) * args.epochs)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, max(1, steps // 20), steps)
        scaler = torch.amp.GradScaler("cuda", enabled=False); opt.zero_grad(set_to_none=True)
        for step, (enc, labels, _, _) in enumerate(loader):
            if step >= steps: break
            enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}; labels = labels.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc).logits.squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if step % 250 == 0: print(step, steps, float(loss), flush=True)
        os.makedirs(args.model, exist_ok=True); model.save_pretrained(args.model); tok.save_pretrained(args.model)
    else:
        assert args.out
        rows = []
        with torch.inference_mode():
            for step, (enc, _, s1, a) in enumerate(loader):
                enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16): logits = model(**enc).logits.squeeze(-1).float().cpu().numpy()
                rows.append(pl.DataFrame({"s1": s1, "a": a, "ce_logit": logits}))
                if step % 500 == 0: print(step, len(loader), flush=True)
        pl.concat(rows).write_parquet(args.out)

if __name__ == "__main__": main()
