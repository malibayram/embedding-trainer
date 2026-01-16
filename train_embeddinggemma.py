import os
import logging
import sys

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("training.log")],
)
logger = logging.getLogger(__name__)

# Environment variables
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    logger.warning("HF_TOKEN not set! Model push to hub will fail.")

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import get_scheduler
from sentence_transformers import SentenceTransformer


def freeze_all_except_embeddings(model: SentenceTransformer) -> tuple[int, int]:
    """Freeze all model parameters except the token embedding layer."""
    total_params = 0
    trainable_params = 0

    # Get transformer from SentenceTransformer
    transformer = None
    for module in model:
        if hasattr(module, "auto_model"):
            transformer = module.auto_model
            break

    if transformer is None:
        raise ValueError("Could not find transformer in SentenceTransformer")

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
        total_params += param.numel()

    # Find and unfreeze embedding layer
    embed_layer = None

    if hasattr(transformer, "model") and hasattr(transformer.model, "embed_tokens"):
        embed_layer = transformer.model.embed_tokens
    elif hasattr(transformer, "embed_tokens"):
        embed_layer = transformer.embed_tokens
    else:
        # Search for it
        for name, module in transformer.named_modules():
            if "embed_tokens" in name or "word_embeddings" in name:
                if hasattr(module, "weight"):
                    embed_layer = module
                    logger.info(f"Found embedding layer at: {name}")
                    break

    if embed_layer is None:
        raise ValueError("Could not find embedding layer")

    # Unfreeze embedding layer
    for param in embed_layer.parameters():
        param.requires_grad = True
        trainable_params += param.numel()

    logger.info(
        f"Embedding: {type(embed_layer).__name__}, shape: {embed_layer.weight.shape}"
    )
    return trainable_params, total_params


def forward_with_gradients(
    model: SentenceTransformer,
    texts: list[str],
    device: torch.device,
    original_model=None,
) -> torch.Tensor:
    """Forward pass with gradient tracking."""
    # Use original model for tokenization and iteration (compiled model may not be iterable)
    base_model = original_model if original_model is not None else model

    features = base_model.tokenize(texts)
    features = {k: v.to(device) for k, v in features.items()}

    for module in base_model:
        features = module(features)

    embedding = features.get("sentence_embedding")
    if embedding is None:
        embedding = features.get("token_embeddings")
        if embedding is not None and len(embedding.shape) == 3:
            embedding = embedding.mean(dim=1)

    return embedding


def main():
    # ========== CONFIGURATION ==========
    config = {
        # Model
        "model_id": "alibayram/magibu-200m-base",
        # Dataset
        "dataset": "alibayram/cosmos-corpus-0-05-with-embeddings",
        "text_column": "text",
        "target_column": "teacher_embedding_final",
        # Training - optimized for H100 80GB
        "num_epochs": 3,
        "batch_size": 384,  # Optimized for H100 80GB (with gradient checkpointing)
        "learning_rate": 5e-4,  # Higher LR for embedding-only training
        "warmup_ratio": 0.02,
        "weight_decay": 0.01,
        "max_grad_norm": 0.5,
        "gradient_accumulation_steps": 1,  # No accumulation needed
        # Loss
        "loss_type": "cosine",
        # Output
        "output_dir": "./trained_model_embedding_only",
        "save_steps": 200,
        "logging_steps": 20,
        # WandB
        "use_wandb": True,
        "wandb_project": "magibu-200m-base",
        "wandb_run_name": "magibu-200m-base-h100",
        # Hub
        "push_to_hub": True,
        "hub_model_id": "alibayram/magibu-200m-tr",
        "hub_token": HF_TOKEN,
        # B200 Optimization
        "use_bf16": True,
        "use_flash_attention": True,
        "compile_model": False,  # Disabled - uses too much memory for cache
        "dataloader_num_workers": 8,
    }

    # Print config
    logger.info("=" * 60)
    logger.info("EMBEDDING-ONLY TRAINING - H100 OPTIMIZED")
    logger.info("=" * 60)
    for k, v in config.items():
        if "token" not in k.lower():
            logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    # Device
    if not torch.cuda.is_available():
        logger.error("CUDA not available! This script is optimized for A100 GPU.")
        return

    device = torch.device("cuda")
    logger.info(f"Using device: {device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(
        f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )

    # Load model with Flash Attention
    logger.info(f"Loading model: {config['model_id']}")
    model_kwargs = {}
    if config["use_flash_attention"]:
        try:
            import flash_attn

            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention 2 enabled")
        except ImportError:
            logger.warning("flash-attn not installed, using standard attention")

    model = SentenceTransformer(config["model_id"], model_kwargs=model_kwargs)
    model.to(device)

    # Apply bf16
    if config["use_bf16"]:
        model = model.to(dtype=torch.bfloat16)
        logger.info("Using bfloat16 precision")

    # Enable gradient checkpointing to save memory (CRITICAL for large batches)
    for module in model:
        if hasattr(module, "auto_model"):
            if hasattr(module.auto_model, "gradient_checkpointing_enable"):
                module.auto_model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")

    # Freeze all except embeddings
    trainable_params, total_params = freeze_all_except_embeddings(model)
    logger.info(
        f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)"
    )

    # Store original model reference (needed for iteration after compile)
    original_model = model

    # torch.compile for faster training on CUDA
    if config["compile_model"]:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Model compiled with torch.compile (reduce-overhead mode)")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
            model = original_model

    # Load dataset
    logger.info(f"Loading dataset: {config['dataset']}")
    train_ds = load_dataset(config["dataset"], split="train")
    logger.info(f"Dataset size: {len(train_ds):,}")

    # Initialize WandB
    wandb = None
    if config["use_wandb"] and WANDB_API_KEY:
        try:
            import wandb as wb

            wb.init(
                project=config["wandb_project"],
                name=config["wandb_run_name"],
                config={
                    "training_mode": "embedding_only",
                    "trainable_params": trainable_params,
                    "total_params": total_params,
                    "gpu": torch.cuda.get_device_name(0),
                    **{k: v for k, v in config.items() if "token" not in k.lower()},
                },
            )
            wandb = wb
            logger.info("WandB initialized")
        except Exception as e:
            logger.warning(f"WandB init failed: {e}")

    # Optimizer - only trainable params
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Scheduler
    total_steps = (len(train_ds) // config["batch_size"]) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])

    scheduler = get_scheduler(
        "cosine",  # Cosine annealing for smoother training
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(f"Total steps: {total_steps:,}, Warmup: {warmup_steps:,}")

    # Loss function
    if config["loss_type"] == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        loss_fn = torch.nn.CosineEmbeddingLoss()

    # Set training mode manually (avoid recursion issue)
    for module in model.modules():
        if hasattr(module, "training"):
            module.training = True

    global_step = 0
    best_loss = float("inf")

    logger.info(f"Starting training for {config['num_epochs']} epochs...")

    for epoch in range(config["num_epochs"]):
        epoch_loss = 0.0
        num_batches = 0

        indices = list(range(len(train_ds)))

        progress = tqdm(
            range(0, len(train_ds), config["batch_size"]),
            desc=f"Epoch {epoch + 1}/{config['num_epochs']}",
        )

        for start_idx in progress:
            end_idx = min(start_idx + config["batch_size"], len(train_ds))
            batch_indices = indices[start_idx:end_idx]
            batch = train_ds.select(batch_indices)

            texts = batch[config["text_column"]]
            targets = torch.tensor(batch[config["target_column"]]).to(device)

            # Forward
            student_output = forward_with_gradients(
                model, texts, device, original_model
            )
            targets = targets.to(dtype=student_output.dtype)

            # Loss
            if config["loss_type"] == "cosine":
                labels = torch.ones(len(texts), dtype=student_output.dtype).to(device)
                loss = loss_fn(student_output, targets, labels)
            else:
                loss = loss_fn(student_output, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            if config["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    config["max_grad_norm"],
                )

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            progress.set_postfix({"loss": f"{loss.item():.4f}"})

            # Logging
            if global_step % config["logging_steps"] == 0:
                avg_loss = epoch_loss / num_batches
                lr = scheduler.get_last_lr()[0]
                logger.info(f"Step {global_step}: loss={avg_loss:.4f}, lr={lr:.2e}")

                if wandb:
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/learning_rate": lr,
                            "train/epoch": epoch + (num_batches / len(progress)),
                        },
                        step=global_step,
                    )

                # Clear CUDA cache to prevent fragmentation OOM
                torch.cuda.empty_cache()

            # Checkpoint
            if global_step % config["save_steps"] == 0:
                # save_path = f"{config['output_dir']}/checkpoint-{global_step}"
                # model.save(save_path)
                # logger.info(f"Saved: {save_path}")

                # Push checkpoint
                if config["push_to_hub"] and config["hub_model_id"]:
                    try:
                        model.push_to_hub(
                            repo_id=config["hub_model_id"],
                            token=config["hub_token"],
                            exist_ok=True,
                        )
                        logger.info(f"Pushed checkpoint {global_step}")
                    except Exception as e:
                        logger.warning(f"Push failed: {e}")

        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch + 1} completed: avg_loss={avg_epoch_loss:.4f}")

        # Track best
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            model.save(f"{config['output_dir']}/best")
            logger.info(f"New best model saved!")

    # Convert to bf16 and save final
    logger.info("Converting to bfloat16 for final save...")
    for module in model:
        if hasattr(module, "auto_model"):
            module.auto_model = module.auto_model.to(dtype=torch.bfloat16)
        elif hasattr(module, "linear"):
            module.linear = module.linear.to(dtype=torch.bfloat16)

    model.save(config["output_dir"])
    logger.info(f"Final model saved to: {config['output_dir']}")

    # Final push
    if config["push_to_hub"] and config["hub_model_id"]:
        try:
            url = model.push_to_hub(
                repo_id=config["hub_model_id"],
                token=config["hub_token"],
                exist_ok=True,
            )
            logger.info(f"Final model pushed to: {url}")
        except Exception as e:
            logger.error(f"Final push failed: {e}")

    if wandb:
        wandb.finish()

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED!")
    logger.info(f"Final loss: {avg_epoch_loss:.4f}")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
