
from datasets import load_dataset

nli_datasets = [
    "boun-tabilab/SNLI-TR",
    "boun-tabilab/MultiNLI-TR",
    "boun-tabilab/Med-NLI-TR",
]

for ds_id in nli_datasets:
    try:
        print(f"--- {ds_id} ---")
        ds = load_dataset(ds_id, streaming=True)
        # Check first available split
        split = next(iter(ds.keys()))
        sample = next(iter(ds[split]))
        print(f"  {split}: {list(sample.keys())}")
    except Exception as e:
        print(f"  Error: {e}")
