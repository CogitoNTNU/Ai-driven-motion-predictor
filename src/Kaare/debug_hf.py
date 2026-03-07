"""Quick debug: print extra_fields for the first few rows of a subset."""
from datasets import load_dataset
from Kaare import config

SUBSET = "fnspid_news"

ds = load_dataset(
    "Brianferrell787/financial-news-multisource",
    data_files=[f"data/{SUBSET}/*.parquet"],
    split="train",
    streaming=True,
    token=config.HF_TOKEN or None,
)

for i, row in enumerate(ds):
    print(f"--- row {i} ---")
    print("keys:", list(row.keys()))
    print("extra_fields type:", type(row.get("extra_fields")))
    print("extra_fields value:", row.get("extra_fields"))
    print()
    if i >= 4:
        break
