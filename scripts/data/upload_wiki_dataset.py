#!/usr/bin/env python3
"""
Upload saved dataset to HuggingFace using direct file upload (Xet Storage compatible).
Run this after prepare_wiki_dataset_parallel.py has saved the dataset locally.
"""

import os
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
HUB_REPO = "alibayram/wikipedia-40-langs"
LOCAL_PATH = "./wikipedia-40-langs-local"


def main():
    print("=" * 70)
    print("UPLOADING DATASET WITH XET STORAGE")
    print("=" * 70)
    
    api = HfApi(token=HF_TOKEN)
    
    # Create repo if doesn't exist
    print(f"Ensuring repo exists: {HUB_REPO}")
    try:
        create_repo(HUB_REPO, repo_type="dataset", token=HF_TOKEN, exist_ok=True)
    except Exception as e:
        print(f"Repo creation note: {e}")
    
    # Upload entire folder directly - this uses Xet Storage for large files
    print(f"\nUploading {LOCAL_PATH} to {HUB_REPO}...")
    print("This uses Xet Storage for faster uploads!")
    
    api.upload_folder(
        folder_path=LOCAL_PATH,
        repo_id=HUB_REPO,
        repo_type="dataset",
        commit_message="Upload wikipedia-40-langs dataset",
    )
    
    print()
    print("=" * 70)
    print("✅ DONE!")
    print(f"Dataset available at: https://huggingface.co/datasets/{HUB_REPO}")
    print("=" * 70)


if __name__ == "__main__":
    main()
