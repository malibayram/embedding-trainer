#!/usr/bin/env python3
"""
Generate embeddings for alibayram/wikiset dataset.

1. Load dataset and split texts to max 2K tokens
2. Embed using ollama embeddinggemma model
3. Save with lang, text, embeddings columns
4. Push to HuggingFace
"""

import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import ollama
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Configuration
MAX_TOKENS = 2000
BATCH_SIZE = 256  # Smaller batch size for parallel processing
NUM_WORKERS = 8   # Number of parallel workers (match OLLAMA_NUM_PARALLEL)
PUSH_EVERY = 10000  # Push to hub every N samples
HUB_REPO = "alibayram/wikiset-embeddings"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Thread-safe lock for processed_data
data_lock = threading.Lock()

def split_text_by_tokens(text: str, tokenizer, max_tokens: int = MAX_TOKENS) -> list[str]:
    """Split text into chunks of max_tokens."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text)
    
    return chunks


def embed_batch(texts: list[str], model: str = "embeddinggemma") -> list[list[float]]:
    """Embed a batch of texts using ollama."""
    response = ollama.embed(model=model, input=texts)
    return response.embeddings


def push_checkpoint(processed_data: dict, push_count: int):
    """Push current data to hub, filtering out empty embeddings."""
    if not processed_data["lang"]:
        return
    
    # Filter out entries with empty embeddings
    valid_indices = [i for i, emb in enumerate(processed_data["embedding"]) if emb]
    if not valid_indices:
        print(f"\nNo valid embeddings to push in checkpoint {push_count}")
        return
    
    filtered_data = {
        "lang": [processed_data["lang"][i] for i in valid_indices],
        "text": [processed_data["text"][i] for i in valid_indices],
        "embedding": [processed_data["embedding"][i] for i in valid_indices],
    }
    
    new_ds = Dataset.from_dict(filtered_data)
    print(f"\nPushing checkpoint {push_count} ({len(new_ds):,} valid samples, filtered {len(processed_data['lang']) - len(valid_indices)} empty) to {HUB_REPO}...")
    new_ds.push_to_hub(HUB_REPO, token=HF_TOKEN)
    print(f"Checkpoint {push_count} pushed!")


CHUNKS_CACHE_FILE = "wikiset_chunks.pkl"


def main():
    import pickle
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it", token=HF_TOKEN)
    
    # Check for cached chunks
    if os.path.exists(CHUNKS_CACHE_FILE):
        print(f"Loading cached chunks from {CHUNKS_CACHE_FILE}...")
        with open(CHUNKS_CACHE_FILE, "rb") as f:
            all_chunks = pickle.load(f)
        print(f"Loaded {len(all_chunks):,} cached chunks")
    else:
        print("Loading dataset alibayram/wikiset...")
        ds = load_dataset("alibayram/wikiset", split="train")
        print(f"Total rows: {len(ds):,}")
        
        # Process and split texts
        print("Splitting texts to max 2K tokens...")
        all_chunks = []  # (lang, text) tuples
        
        for row in tqdm(ds, desc="Splitting texts"):
            text = row["text"]
            lang = row["lang"]
            
            chunks = split_text_by_tokens(text, tokenizer, MAX_TOKENS)
            for chunk in chunks:
                all_chunks.append((lang, chunk))
        
        # Save chunks to cache
        print(f"Saving {len(all_chunks):,} chunks to {CHUNKS_CACHE_FILE}...")
        with open(CHUNKS_CACHE_FILE, "wb") as f:
            pickle.dump(all_chunks, f)
        print("Chunks cached!")
    
    print(f"Total chunks: {len(all_chunks):,}")
    
    processed_data = {"lang": [], "text": [], "embedding": []}
    
    # Embed in batches with parallel processing
    print(f"Embedding {len(all_chunks):,} chunks with {NUM_WORKERS} parallel workers (batch size {BATCH_SIZE})...")
    
    push_count = 0
    last_push_count = 0
    
    # Create batches
    batches = []
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        batches.append((i, batch))
    
    def process_batch(batch_info):
        """Process a single batch and return results."""
        idx, batch = batch_info
        langs = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        
        try:
            embeddings = embed_batch(texts)
            return [(lang, text, emb) for lang, text, emb in zip(langs, texts, embeddings)]
        except Exception as e:
            print(f"Error embedding batch {idx}: {e}")
            return [(lang, text, []) for lang, text in zip(langs, texts)]
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_batch, batch): batch for batch in batches}
        
        for future in tqdm(as_completed(futures), total=len(batches), desc="Embedding"):
            results = future.result()
            
            with data_lock:
                for lang, text, emb in results:
                    processed_data["lang"].append(lang)
                    processed_data["text"].append(text)
                    processed_data["embedding"].append(emb)
                
                # Push checkpoint every PUSH_EVERY samples
                current_count = len(processed_data["lang"])
                if current_count >= last_push_count + PUSH_EVERY:
                    push_count += 1
                    push_checkpoint(processed_data, push_count)
                    last_push_count = current_count
    
    # Final push
    push_count += 1
    push_checkpoint(processed_data, push_count)
    print("Done!")


if __name__ == "__main__":
    main()
