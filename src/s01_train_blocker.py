"""Fine-tune intfloat/multilingual-e5-base (MIT) as a bi-encoder for S1 -> S2/S3 retrieval on dev-split true pairs (in-batch negatives)."""
import os as _os
ER_ROOT = _os.environ.get("ER_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # package root (code/business_entity_resolution)

import argparse, os, sys, polars as pl
sys.path.insert(0, f"{ER_ROOT}/src/common")
from erdata import load
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from sentence_transformers.training_args import BatchSamplers

ap = argparse.ArgumentParser(); ap.add_argument("--pairs", type=int, default=800_000); ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--out", default="models/e5b_dev_v1"); args = ap.parse_args()

def text(df):
    return ("query: " + df["business_name"] + " | " + df["business_address"]).to_list()

s1, aux, edges, split = load()
dev = split.filter(pl.col("split") == "dev").select(pl.col("source1_entity_id").alias("s1"))
pairs = (edges.join(dev, on="s1").sample(args.pairs, seed=0)
              .join(s1.select(pl.col("entity_id").alias("s1"), n1="business_name", a1="business_address"), on="s1")
              .join(aux.select(pl.col("entity_id").alias("a"), n2="business_name", a2="business_address"), on="a"))
ds = Dataset.from_dict({
    "anchor": text(pairs.select(business_name="n1", business_address="a1")),
    "positive": text(pairs.select(business_name="n2", business_address="a2"))})
print(f"training pairs: {len(ds):,}", flush=True)

model = SentenceTransformer("intfloat/multilingual-e5-base")
model.max_seq_length = 64
trainer = SentenceTransformerTrainer(
    model=model,
    args=SentenceTransformerTrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, per_device_train_batch_size=args.bs, learning_rate=3e-5,
        warmup_ratio=0.05, bf16=True, batch_sampler=BatchSamplers.NO_DUPLICATES, logging_steps=100, save_strategy="no",
        report_to="none", dataloader_num_workers=4, seed=0),
    train_dataset=ds, loss=losses.MultipleNegativesRankingLoss(model))
trainer.train()
model.save(args.out)
print("saved", args.out)
