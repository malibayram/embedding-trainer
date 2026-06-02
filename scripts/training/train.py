#!/usr/bin/env python3
"""Training script for distillation using precomputed embeddings."""

import os

# API keys from environment variables (more secure)
# Set these before running: export WANDB_API_KEY=xxx HF_TOKEN=xxx
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

import json

from huggingface_hub import HfApi, hf_hub_download

model_id = "alibayram/magibu-200m"

# First, fix the tokenizer_config.json issue
print("Fixing tokenizer_config.json...")
try:
    path = hf_hub_download(model_id, "tokenizer_config.json")
    with open(path) as f:
        config = json.load(f)

    if isinstance(config.get("extra_special_tokens"), list):
        config["extra_special_tokens"] = {}
        with open("/tmp/tokenizer_config.json", "w") as f:
            json.dump(config, f, indent=2)

        api = HfApi()
        api.upload_file(
            path_or_fileobj="/tmp/tokenizer_config.json",
            path_in_repo="tokenizer_config.json",
            repo_id=model_id,
            token=HF_TOKEN,
        )
        print("Tokenizer config fixed and uploaded!")
    else:
        print("Tokenizer config already correct.")
except Exception as e:
    print(f"Warning: Could not fix tokenizer: {e}")

# Now run training
from distil_trainer import EmbeddingDistillationTrainer, EmbeddingTrainerConfig

config = EmbeddingTrainerConfig(
    student_model=model_id,
    target_type="final",
    # Training hyperparameters
    num_epochs=10,  # ← Train for 2 epochs for better convergence
    batch_size=1024,
    learning_rate=5e-5,  # ← Increased from 2e-5
    warmup_ratio=0.01,  # ← 1% warmup steps
    weight_decay=0.01,  # ← L2 regularization
    max_grad_norm=1.0,  # ← Gradient clipping
    # Loss function
    loss_type="cosine",  # ← Better for embedding similarity
    # Optimization
    use_bf16=True,
    gradient_checkpointing=True,
    compile_model=True,  # ← Enable torch.compile for faster training
    # Dataset columns
    text_column="text",
    final_embedding_column="teacher_embedding_final",
    # Output
    output_dir="./trained_model",
    save_steps=100,
    logging_steps=20,
    # WandB
    use_wandb=True,
    wandb_project="distillation-b300",
    wandb_run_name="optimized-distillation-wikipedia",
    # Push to Hub
    push_to_hub=True,
    hub_model_id=model_id,
    hub_token=HF_TOKEN,
)

trainer = EmbeddingDistillationTrainer(config)
metrics = trainer.train("alibayram/wikipedia-40-langs-with-embeddings")

print(f"Final loss: {metrics['train_loss']:.4f}")
