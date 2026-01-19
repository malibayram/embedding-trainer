#!/usr/bin/env python3
"""
Generate embeddings for alibayram/wikiset dataset using sentence-transformers (FAST).

Uses direct GPU inference instead of ollama HTTP for 10-50x speedup.
"""

import os
import pickle
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Configuration
MAX_TOKENS = 2000
BATCH_SIZE = 128  # Larger batches for GPU efficiency
PUSH_EVERY = 20000  # Push to hub every N samples
HUB_REPO = "alibayram/wikiset-embeddings"
HF_TOKEN = os.environ.get("HF_TOKEN")
CHUNKS_CACHE_FILE = "wikiset_chunks.pkl"

# Use embeddinggemma or any sentence-transformer model
EMBED_MODEL = "google/embeddinggemma-300m"  # or "alibayram/magibu-200m-tr"


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


def push_checkpoint(processed_data: dict, push_count: int):
    """Push current data to hub, filtering out empty embeddings."""
    if not processed_data["lang"]:
        return
    
    # Filter out entries with empty embeddings
    valid_indices = [i for i, emb in enumerate(processed_data["embedding"]) if len(emb) > 0]
    if not valid_indices:
        print(f"\nNo valid embeddings to push in checkpoint {push_count}")
        return
    
    filtered_data = {
        "lang": [processed_data["lang"][i] for i in valid_indices],
        "text": [processed_data["text"][i] for i in valid_indices],
        "embedding": [processed_data["embedding"][i] for i in valid_indices],
    }
    
    new_ds = Dataset.from_dict(filtered_data)
    print(f"\nPushing checkpoint {push_count} ({len(new_ds):,} valid samples) to {HUB_REPO}...")
    new_ds.push_to_hub(HUB_REPO, token=HF_TOKEN)
    print(f"Checkpoint {push_count} pushed!")


def main():
    print("=" * 60)
    print("FAST EMBEDDING GENERATOR - Direct GPU Inference")
    print("=" * 60)
    
    # Load tokenizer for splitting
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
        
        print("Splitting texts to max 2K tokens...")
        all_chunks = []
        
        for row in tqdm(ds, desc="Splitting texts"):
            text = row["text"]
            lang = row["lang"]
            
            chunks = split_text_by_tokens(text, tokenizer, MAX_TOKENS)
            for chunk in chunks:
                all_chunks.append((lang, chunk))
        
        print(f"Saving {len(all_chunks):,} chunks to {CHUNKS_CACHE_FILE}...")
        with open(CHUNKS_CACHE_FILE, "wb") as f:
            pickle.dump(all_chunks, f)
        print("Chunks cached!")
    
    print(f"Total chunks: {len(all_chunks):,}")
    
    # Load embedding model directly on GPU
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = SentenceTransformer(EMBED_MODEL, device=device, trust_remote_code=True)
    model.eval()
    print(f"Model loaded! Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # Embed in batches - MUCH faster than ollama HTTP
    print(f"\nEmbedding {len(all_chunks):,} chunks in batches of {BATCH_SIZE}...")
    
    processed_data = {"lang": [], "text": [], "embedding": []}
    push_count = 0
    last_push_count = 0
    
    # Group chunks into batches
    for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="Embedding"):
        batch = all_chunks[i:i + BATCH_SIZE]
        langs = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        
        # Direct GPU inference - no HTTP overhead!
        with torch.no_grad():
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        for lang, text, emb in zip(langs, texts, embeddings):
            processed_data["lang"].append(lang)
            processed_data["text"].append(text)
            processed_data["embedding"].append(emb.tolist())
        
        # Push checkpoint
        current_count = len(processed_data["lang"])
        if current_count >= last_push_count + PUSH_EVERY:
            push_count += 1
            push_checkpoint(processed_data, push_count)
            last_push_count = current_count
    
    # Final push
    push_count += 1
    push_checkpoint(processed_data, push_count)
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
