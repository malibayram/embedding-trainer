#!/usr/bin/env python3
"""
Multi-Task SFT for Turkish Embedding Model (Magibu-128k-200m)

Loss Mapping: MNRL (retrieval/bitext), CoSENT (STS), Softmax (classification/NLI)

Verified datasets: 16 datasets, ~800k samples (fast training)
"""

import os, logging, argparse
from dataclasses import dataclass, field
from dotenv import load_dotenv
load_dotenv()

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from transformers import TrainerCallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

WANDB_KEY = os.environ.get("WANDB_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================================
# DATASET REGISTRY - Fast, verified datasets only
# ============================================================================

@dataclass
class DS:
    hf_id: str; task: str; cols: list = field(default_factory=lambda: ["anchor", "positive"])
    label: str = None; max_n: int = None; split: str = "train"

REGISTRY = {
    # === RETRIEVAL (anchor, positive) ===
    "msmarco_tr_ft": DS("selmanbaysan/msmarco-tr_fine_tuning_dataset", "ret", max_n=100_000),
    "nfcorpus_tr_ft": DS("selmanbaysan/nfcorpus-tr_fine_tuning_dataset", "ret"),
    "scifact_tr_ft": DS("selmanbaysan/scifact-tr_fine_tuning_dataset", "ret"),
    "fiqa_tr_ft": DS("selmanbaysan/fiqa-tr_fine_tuning_dataset", "ret"),
    "quora_tr_ft": DS("selmanbaysan/quora-tr_fine_tuning_dataset", "ret", split="dev"),
    
    # === BITEXT (sentence1, sentence2) ===
    "wmt16_en_tr_ft": DS("selmanbaysan/wmt16_en_tr_fine_tuning_dataset", "bitext"),
    "wmt16_en_tr": DS("selmanbaysan/wmt16_en_tr", "bitext", ["sentence1", "sentence2"], max_n=100_000),
    
    # === STS (sentence1, sentence2, score) ===
    "stsb_tr": DS("selmanbaysan/stsb-tr", "sts", ["sentence1", "sentence2"], "score"),
    
    # === CLASSIFICATION (text, label) ===
    "offenseval": DS("selmanbaysan/offenseval", "cls", ["text"], "label"),
    "news_cat": DS("selmanbaysan/news-cat", "cls", ["text"], "label"),
    "thy_sa": DS("selmanbaysan/thy_sa", "cls", ["text"], "label", split="test"),
    "irony_tr": DS("selmanbaysan/irony-tr", "cls", ["text"], "label", split="test"),
    "ts_timeline_news": DS("selmanbaysan/ts_timeline_news_category", "cls", ["text"], "label"),
    
    # === NLI (premise, hypothesis, label) ===
    "snli_tr_ft": DS("selmanbaysan/snli_tr_fine_tuning_dataset", "nli", ["premise", "hypothesis"], "label", max_n=100_000),
    "multinli_tr_ft": DS("selmanbaysan/multinli_tr_fine_tuning_dataset", "nli", ["premise", "hypothesis"], "label", max_n=100_000),
    "xnli_tr_ft": DS("selmanbaysan/xnli_tr_fine_tuning_dataset", "nli", ["premise", "hypothesis"], "label", split="validation"),
}

# ============================================================================
# MTEB CALLBACK
# ============================================================================

class MTEBCallback(TrainerCallback):
    def __init__(self, eval_steps=2000, tasks=None, out_dir="./mteb"):
        self.eval_steps, self.tasks = eval_steps, tasks or ["STSbTR"]
        self.out_dir, self.last_step = out_dir, -1

    def _eval(self, model, step):
        try:
            import mteb
            os.makedirs(f"{self.out_dir}/step_{step}", exist_ok=True)
            for t in self.tasks:
                res = mteb.MTEB(tasks=mteb.get_tasks(tasks=[t])).run(
                    model, output_folder=f"{self.out_dir}/step_{step}", eval_splits=["test"], verbosity=0)
                for r in res:
                    if hasattr(r, 'scores') and r.scores.get('test'):
                        logger.info(f"MTEB {t}: {r.scores['test'][0].get('main_score', 0):.4f}")
        except Exception as e:
            logger.warning(f"MTEB failed: {e}")

    def on_step_end(self, args, state, control, model=None, **kw):
        if self.eval_steps > 0 and state.global_step % self.eval_steps == 0 and state.global_step != self.last_step:
            self.last_step = state.global_step
            self._eval(model, state.global_step)

# ============================================================================
# CORE
# ============================================================================

def load_ds(cfg: DS, max_n=None):
    try:
        ds = load_dataset(cfg.hf_id, split=cfg.split)
    except Exception as e:
        logger.warning(f"  Cannot load {cfg.hf_id}: {e}")
        return None
    
    missing = [c for c in cfg.cols if c not in ds.column_names]
    if missing:
        logger.warning(f"  {cfg.hf_id}: missing {missing}")
        return None
    
    ds = ds.filter(lambda x: all(x.get(c) for c in cfg.cols))
    if len(ds) == 0:
        return None
    
    max_n = max_n or cfg.max_n
    if max_n and len(ds) > max_n:
        ds = ds.shuffle(seed=42).select(range(max_n))
    return ds


def get_emb_dim(model):
    for m in model:
        if hasattr(m, "get_word_embedding_dimension"):
            return m.get_word_embedding_dimension()
    return 768


def create_losses(model, datasets, dim):
    mnrl = losses.MultipleNegativesRankingLoss(model=model)
    cosent = losses.CoSENTLoss(model=model)
    out = {}
    
    for name, (ds, cfg) in datasets.items():
        if cfg.task in ("ret", "bitext"):
            out[name] = mnrl
        elif cfg.task == "sts":
            out[name] = cosent
        else:  # cls, nli
            n_labels = len(set(ds[cfg.label])) if cfg.label in ds.column_names else 3
            out[name] = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=dim, num_labels=n_labels)
    return out


def train(model_id="alibayram/magibu-128k-200m", output="./magibu-sft", hub_id="alibayram/magibu-128k-200m-sft",
          epochs=1, batch=64, lr=2e-5, save_steps=500, eval_steps=2000, dry_run=False):
    
    logger.info(f"Model: {model_id} | Epochs: {epochs} | Batch: {batch}")
    
    model = SentenceTransformer(model_id)
    if torch.cuda.is_available():
        model = model.to("cuda")
        if torch.cuda.is_bf16_supported():
            model = model.to(dtype=torch.bfloat16)
    
    dim = get_emb_dim(model)
    
    # Load datasets
    datasets = {}
    for name, cfg in REGISTRY.items():
        ds = load_ds(cfg)
        if ds and len(ds) > 0:
            datasets[name] = (ds, cfg)
            logger.info(f"  ✓ {name}: {len(ds):,}")
    
    if not datasets:
        logger.error("No datasets loaded!")
        return
    
    total = sum(len(d[0]) for d in datasets.values())
    logger.info(f"Total: {total:,} samples from {len(datasets)} datasets")
    
    if dry_run:
        logger.info("DRY RUN - exiting")
        return
    
    loss_dict = create_losses(model, datasets, dim)
    train_ds = {k: v[0] for k, v in datasets.items()}
    
    if WANDB_KEY:
        try:
            import wandb
            wandb.init(project="magibu-sft", config={"model": model_id, "samples": total, "datasets": len(datasets)})
        except:
            pass
    
    args = SentenceTransformerTrainingArguments(
        output_dir=output,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        logging_steps=100,
        push_to_hub=bool(HF_TOKEN),
        hub_model_id=hub_id,
        hub_strategy="every_save",
        hub_token=HF_TOKEN,
        report_to="wandb" if WANDB_KEY else "none",
    )
    
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        loss=loss_dict,
        callbacks=[MTEBCallback(eval_steps)] if eval_steps > 0 else [],
    )
    
    trainer.train()
    model.save_pretrained(output)
    if HF_TOKEN:
        model.push_to_hub(hub_id, token=HF_TOKEN)
    logger.info(f"Done! Saved to {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="alibayram/magibu-128k-200m-sft")
    p.add_argument("--output", default="./magibu-sft")
    p.add_argument("--hub-id", default="alibayram/magibu-128k-200m-sft")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--save-steps", type=int, default=5000)
    p.add_argument("--eval-steps", type=int, default=20000)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    
    train(a.model_id, a.output, a.hub_id, a.epochs, a.batch, a.lr, a.save_steps, a.eval_steps, a.dry_run)
