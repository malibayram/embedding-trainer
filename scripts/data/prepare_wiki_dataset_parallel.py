#!/usr/bin/env python3
"""
Prepare a multi-language Wikipedia dataset for embedding training.
PARALLEL VERSION - Downloads multiple languages concurrently.

Downloads Wikipedia articles from 40 languages using streaming,
creates train/validation/test splits with language-specific quotas,
and pushes to HuggingFace Hub.

Usage:
    python prepare_wiki_dataset_parallel.py
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
HUB_REPO = "alibayram/wikipedia-40-langs"

# Number of parallel workers (be careful not to overwhelm the API)
MAX_WORKERS = 8

# Language configurations: (code, train_samples, val_samples, test_samples)
# Turkish and English get 100K/30K/10K, others get 10K/3K/1K
LANGUAGES = [
    # High-resource languages (100K train, 30K val, 10K test)
    ("en", 100_000, 30_000, 10_000),     # English
    ("tr", 100_000, 30_000, 10_000),     # Turkish
    
    # Standard languages (10K train, 3K val, 1K test)
    ("ceb", 10_000, 3_000, 1_000),       # Cebuano
    ("nl", 10_000, 3_000, 1_000),        # Dutch
    ("it", 10_000, 3_000, 1_000),        # Italian
    ("pl", 10_000, 3_000, 1_000),        # Polish
    ("arz", 10_000, 3_000, 1_000),       # Egyptian Arabic
    ("de", 10_000, 3_000, 1_000),        # German
    ("ja", 10_000, 3_000, 1_000),        # Japanese
    ("zh", 10_000, 3_000, 1_000),        # Chinese
    ("ar", 10_000, 3_000, 1_000),        # Arabic
    ("vi", 10_000, 3_000, 1_000),        # Vietnamese
    ("war", 10_000, 3_000, 1_000),       # Waray
    ("es", 10_000, 3_000, 1_000),        # Spanish
    ("pt", 10_000, 3_000, 1_000),        # Portuguese
    ("fa", 10_000, 3_000, 1_000),        # Persian
    ("fr", 10_000, 3_000, 1_000),        # French
    ("ca", 10_000, 3_000, 1_000),        # Catalan
    ("id", 10_000, 3_000, 1_000),        # Indonesian
    ("ko", 10_000, 3_000, 1_000),        # Korean
    ("ce", 10_000, 3_000, 1_000),        # Chechen
    ("no", 10_000, 3_000, 1_000),        # Norwegian
    ("fi", 10_000, 3_000, 1_000),        # Finnish
    ("cs", 10_000, 3_000, 1_000),        # Czech
    ("hu", 10_000, 3_000, 1_000),        # Hungarian
    ("tt", 10_000, 3_000, 1_000),        # Tatar
    ("ro", 10_000, 3_000, 1_000),        # Romanian
    ("eu", 10_000, 3_000, 1_000),        # Basque
    ("sh", 10_000, 3_000, 1_000),        # Serbian
    ("ms", 10_000, 3_000, 1_000),        # Malay
    ("eo", 10_000, 3_000, 1_000),        # Esperanto
    ("he", 10_000, 3_000, 1_000),        # Hebrew
    ("ru", 10_000, 3_000, 1_000),        # Russian
    ("hy", 10_000, 3_000, 1_000),        # Armenian
    ("uz", 10_000, 3_000, 1_000),        # Uzbek
    ("da", 10_000, 3_000, 1_000),        # Danish
    ("bg", 10_000, 3_000, 1_000),        # Bulgarian
    ("cy", 10_000, 3_000, 1_000),        # Welsh
    ("simple", 10_000, 3_000, 1_000),    # Simple English
    ("el", 10_000, 3_000, 1_000),        # Greek
]


def collect_samples_streaming(lang_code: str, train_n: int, val_n: int, test_n: int) -> dict:
    """
    Stream Wikipedia dataset for a language and collect samples.
    Returns dict with train/val/test lists.
    """
    total_needed = train_n + val_n + test_n
    config = f"20231101.{lang_code}"
    samples = []
    
    try:
        ds = load_dataset(
            "wikimedia/wikipedia",
            config,
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        for item in ds:
            samples.append({
                "lang": lang_code,
                "title": item.get("title", ""),
                "text": item.get("text", ""),
                "url": item.get("url", ""),
            })
            
            if len(samples) >= total_needed:
                break
                
    except Exception as e:
        print(f"⚠️  Error loading {lang_code}: {e}")
        return {"lang": lang_code, "train": [], "val": [], "test": [], "error": str(e)}
    
    # Split samples
    total_available = len(samples)
    if total_available < total_needed:
        ratio = total_available / total_needed
        train_n = int(train_n * ratio)
        val_n = int(val_n * ratio)
        test_n = total_available - train_n - val_n
    
    train = samples[:train_n]
    val = samples[train_n:train_n + val_n]
    test = samples[train_n + val_n:train_n + val_n + test_n]
    
    return {"lang": lang_code, "train": train, "val": val, "test": test}


def main():
    print("=" * 70)
    print("WIKIPEDIA 40-LANGUAGE DATASET PREPARATION (PARALLEL)")
    print("=" * 70)
    print(f"Target repository: {HUB_REPO}")
    print(f"Languages: {len(LANGUAGES)}")
    print(f"Parallel workers: {MAX_WORKERS}")
    print()
    
    # Calculate totals
    total_train = sum(L[1] for L in LANGUAGES)
    total_val = sum(L[2] for L in LANGUAGES)
    total_test = sum(L[3] for L in LANGUAGES)
    print(f"Target samples - Train: {total_train:,} | Val: {total_val:,} | Test: {total_test:,}")
    print()
    
    all_train = []
    all_val = []
    all_test = []
    
    # Process languages in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(collect_samples_streaming, lang, train_n, val_n, test_n): lang
            for lang, train_n, val_n, test_n in LANGUAGES
        }
        
        with tqdm(total=len(LANGUAGES), desc="Languages") as pbar:
            for future in as_completed(futures):
                lang_code = futures[future]
                try:
                    result = future.result()
                    if "error" not in result:
                        all_train.extend(result["train"])
                        all_val.extend(result["val"])
                        all_test.extend(result["test"])
                        pbar.write(f"✓ [{result['lang']}] {len(result['train']):,} train, {len(result['val']):,} val, {len(result['test']):,} test")
                    else:
                        pbar.write(f"⚠️ [{result['lang']}] Error: {result['error']}")
                except Exception as e:
                    pbar.write(f"⚠️ [{lang_code}] Exception: {e}")
                pbar.update(1)
    
    print()
    print("=" * 70)
    print("Creating datasets...")
    print(f"  Train: {len(all_train):,} samples")
    print(f"  Validation: {len(all_val):,} samples")
    print(f"  Test: {len(all_test):,} samples")
    
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(all_train),
        "validation": Dataset.from_list(all_val),
        "test": Dataset.from_list(all_test),
    })
    
    # Save locally first for faster Xet Storage upload
    local_path = "./wikipedia-40-langs-local"
    print()
    print(f"Saving locally to {local_path}...")
    dataset_dict.save_to_disk(local_path)
    
    print()
    print(f"Pushing to HuggingFace Hub: {HUB_REPO}")
    dataset_dict.push_to_hub(HUB_REPO, token=HF_TOKEN)
    
    print()
    print("=" * 70)
    print("✅ DONE!")
    print(f"Dataset available at: https://huggingface.co/datasets/{HUB_REPO}")
    print("=" * 70)


if __name__ == "__main__":
    main()
