#!/usr/bin/env python3
"""
Multi-Task Supervised Fine-Tuning for Turkish Embedding Model

This script fine-tunes the alibayram/magibu-200m embedding model using a multi-objective
training approach with task-specific datasets and loss functions:

Task Groups:
    - Retrieval: msmarco-tr, squad-tr, nfcorpus-tr, fiqa-tr
    - Bitext Mining: wmt16_en_tr
    - STS (Semantic Textual Similarity): stsb-tr
    - Classification: thy_sa, offenseval, news-cat, 75haber
    - Pair Classification (NLI): snli_tr, multinli_tr, xnli_tr

Usage:
    python train_sft_multi_task.py --help
    python train_sft_multi_task.py --dry-run --max-samples-per-task 100
    python train_sft_multi_task.py --epochs 3 --batch-size 64
"""

import os
import sys
import logging
import argparse
from dataclasses import dataclass, field

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("sft_training.log")],
)
logger = logging.getLogger(__name__)

# Environment variables
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    logger.warning("HF_TOKEN not set! Model push to hub will fail.")

# Add mteb-tr to path to allow loading custom tasks
mteb_tr_path = os.path.abspath("mteb-tr")
if os.path.exists(mteb_tr_path) and mteb_tr_path not in sys.path:
    sys.path.insert(0, mteb_tr_path)
    logger.info(f"Added {mteb_tr_path} to sys.path")

import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    losses,
)
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from transformers import TrainerCallback


# ============================================================================
# MTEB EVALUATION CALLBACK
# ============================================================================

class MTEBEvaluationCallback(TrainerCallback):
    """
    Callback to run MTEB evaluation periodically during training.
    Results are logged to WandB for tracking.
    """
    
    # Quick evaluation tasks (fast but representative)
    QUICK_EVAL_TASKS = [
        "STSbTR",  # Turkish STS
        "TurkishMovieSentimentClassification",  # Classification
    ]
    
    # Comprehensive list of all Turkish tasks
    ALL_TR_TASKS = [
        # Retrieval
        "ArguAnaTR",
        "FiQA2018TR",
        "MSMarcoTRRetrieval",
        "NFCorpusTR",
        "QuoraRetrievalTR",
        "SCIDOCSTR",
        "SciFactTR",
        "SquadTRRetrieval",
        "TQuadRetrieval",
        "TurHistQuadRetrieval",
        "CQADupstackGamingRetrievalTR",
        
        # STS (Semantic Textual Similarity)
        "STSbTR",
        
        # Classification
        "TurkishMovieSentimentClassification",
        "TurkishProductSentimentClassification",
        "TurkishNewsCategoryClassification",
        "TurkishOffensiveLanguageClassification",
        "TurkishIronyClassification",
        "Turkish75NewsClassification",
        "THYSentimentClassification",
        "TSTimelineNewsCategoryClassification",
        
        # Clustering
        "TurkishAbstractCorpusClustering",
        "TurkishColumnWritingClustering",
        
        # Pair Classification (NLI)
        "SnliTr",
        "MnliTr",
    ]
    
    def __init__(
        self,
        model=None,
        eval_steps: int = 1000,
        eval_tasks: list = None,
        output_folder: str = "./mteb_results",
    ):
        """
        Args:
            model: SentenceTransformer model to evaluate
            eval_steps: Run evaluation every N steps
            eval_tasks: List of MTEB task names to run (None = QUICK_EVAL_TASKS)
            output_folder: Folder to save MTEB results
        """
        self.model = model
        self.eval_steps = eval_steps
        self.eval_tasks = eval_tasks or self.QUICK_EVAL_TASKS
        self.output_folder = output_folder
        self.last_eval_step = -1
        self._mteb_available = None
        self._baseline_done = False
        
    def _check_mteb_available(self):
        """Check if MTEB is available."""
        if self._mteb_available is None:
            try:
                import mteb
                self._mteb_available = True
            except ImportError:
                logger.warning("MTEB not installed. Run: pip install mteb")
                self._mteb_available = False
        return self._mteb_available
    
    def _run_mteb_evaluation(self, model, global_step: int):
        """Run MTEB evaluation and return results."""
        if not self._check_mteb_available():
            return None
            
        import mteb
        import traceback
        
        results = {}
        try:
            # Get tasks
            tasks = mteb.get_tasks(tasks=self.eval_tasks)
            if not tasks:
                logger.warning(f"No MTEB tasks found for: {self.eval_tasks}")
                return None
            
            logger.info(f"Evaluating {len(tasks)} MTEB tasks...")
            
            # Use step_0 folder for baseline, otherwise step_{step}
            step_eval_folder = f"{self.output_folder}/step_{global_step}"
            os.makedirs(step_eval_folder, exist_ok=True)
            
            # Initial print header
            if global_step == 0:
                print("\n" + "="*80)
                print(f"STARTING BASELINE MTEB EVALUATION ({len(tasks)} TASKS)")
                print("="*80 + "\n")
            
            # Run evaluation one by one to print results immediately
            task_scores = []
            
            for task in tasks:
                logger.info(f"Evaluating task: {task.metadata.name}")
                
                # Create specific evaluation object for this task
                evaluation = mteb.MTEB(tasks=[task])
                
                eval_results = evaluation.run(
                    model,
                    output_folder=step_eval_folder,
                    eval_splits=["test"],
                    verbosity=0,
                )
                
                # Parse results for this task
                for task_result in eval_results:
                    task_name = task_result.task_name
                    # Get main score based on task type
                    if hasattr(task_result, 'scores') and task_result.scores:
                        test_scores = task_result.scores.get('test', [])
                        if test_scores and len(test_scores) > 0:
                            score_dict = test_scores[0]
                            # Extract main metric
                            main_score = score_dict.get('main_score', 
                                         score_dict.get('cos_sim', {}).get('spearman',
                                         score_dict.get('accuracy', 0)))
                            
                            results[f"mteb/{task_name}"] = main_score
                            task_scores.append(main_score)
                            
                            # Print immediately
                            print(f"Task: {task_name:<40} Score: {main_score:.4f}")
            
            # Calculate mean
            if results:
                results["mteb/mean_score"] = sum(task_scores) / len(task_scores)
                
        except Exception as e:
            logger.error(f"MTEB evaluation failed: {e}")
            traceback.print_exc()
            return None
            
        return results
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Run MTEB evaluation before training starts to capture baseline scores."""
        # Skip if already done
        if self._baseline_done:
            return
        
        # Get model from stored reference, passed parameter, or trainer
        eval_model = model or self.model
        if eval_model is None and 'trainer' in kwargs:
            eval_model = kwargs['trainer'].model
        
        if eval_model is None:
            logger.warning("No model available for baseline MTEB evaluation")
            return
        
        self._baseline_done = True
        logger.info("Running baseline MTEB evaluation before training...")
        results = self._run_mteb_evaluation(eval_model, 0)
        
        if results:
            try:
                import wandb
                if wandb.run is not None:
                    # Log with "baseline/" prefix
                    baseline_results = {f"baseline/{k}": v for k, v in results.items()}
                    wandb.log(baseline_results, step=0)
                    logger.info(f"Baseline MTEB results logged to WandB")
            except Exception as e:
                logger.warning(f"Failed to log baseline to WandB: {e}")
            
            # Log to console
            logger.info("Baseline MTEB scores (before training):")
            for task, score in results.items():
                logger.info(f"  {task}: {score:.4f}")
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each training step."""
        if self.eval_steps <= 0:
            return
            
        # Check if we should evaluate
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            if state.global_step == self.last_eval_step:
                return  # Already evaluated at this step
                
            self.last_eval_step = state.global_step
            
            # Get the actual model from trainer
            if model is None:
                return
                
            # Run MTEB evaluation
            results = self._run_mteb_evaluation(model, state.global_step)
            
            if results:
                # Log to WandB
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log(results, step=state.global_step)
                        logger.info(f"MTEB results logged to WandB: {results}")
                except Exception as e:
                    logger.warning(f"Failed to log to WandB: {e}")
                    
                # Also log to console
                logger.info(f"MTEB Evaluation at step {state.global_step}:")
                for task, score in results.items():
                    logger.info(f"  {task}: {score:.4f}")
    
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Run final MTEB evaluation at the end of training."""
        if model is None:
            return
            
        logger.info("Running final MTEB evaluation...")
        results = self._run_mteb_evaluation(model, state.global_step)
        
        if results:
            try:
                import wandb
                if wandb.run is not None:
                    # Log with "final/" prefix
                    final_results = {f"final/{k}": v for k, v in results.items()}
                    wandb.log(final_results)
                    logger.info(f"Final MTEB results: {results}")
            except Exception as e:
                logger.warning(f"Failed to log final results to WandB: {e}")


# ============================================================================
# DATA CLASSES FOR CONFIGURATION
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    hf_id: str
    task_type: str  # retrieval, bitext, sts, classification, pair_classification
    text_columns: list = field(default_factory=list)  # Columns containing text
    label_column: str = None
    max_samples: int = None  # Max samples to use (for balancing)
    weight: float = 1.0  # Weight for this dataset in training


@dataclass
class TrainingConfig:
    """Main training configuration."""
    model_id: str = "alibayram/magibu-128k-200m"
    output_dir: str = "./magibu-128k-200m-sft"
    hub_model_id: str = "alibayram/magibu-128k-200m-sft"
    
    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Optimization
    use_bf16: bool = True
    use_fp16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    
    # Logging and saving
    logging_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    push_to_hub: bool = True
    
    # MTEB Evaluation
    eval_steps: int = 1000  # Run MTEB eval every N steps (0 to disable)
    eval_tasks: list = None # List of tasks to evaluate (None for default)
    eval_all_tasks: bool = False # Run all available Turkish tasksick subset)
    
    # WandB
    use_wandb: bool = True
    wandb_project: str = "magibu-sft"
    wandb_run_name: str = None
    
    # Task balancing
    max_samples_per_task: int = None  # Limit samples per task for balancing
    
    # Misc
    dry_run: bool = False
    seed: int = 42


# ============================================================================
# DATASET DEFINITIONS - ALL 38 DATASETS FROM SELMANBAYSAN
# ============================================================================

# All available datasets organized by task type
DATASET_REGISTRY = {
    # =========================================================================
    # RETRIEVAL DATASETS (query-passage pairs)
    # Format: anchor, positive (with in-batch negatives)
    # Loss: MultipleNegativesRankingLoss
    # =========================================================================
    
    # MS MARCO Turkish
    "msmarco_tr": DatasetConfig(
        name="msmarco_tr",
        hf_id="selmanbaysan/msmarco-tr",
        task_type="retrieval",
        text_columns=["anchor", "positive"],
        max_samples=200_000,  # Subsample from 1.75M
        weight=1.0,
    ),
    "msmarco_tr_ft": DatasetConfig(
        name="msmarco_tr_ft",
        hf_id="selmanbaysan/msmarco-tr_fine_tuning_dataset",
        task_type="retrieval",
        text_columns=["anchor", "positive"],
        max_samples=100_000,  # Subsample from 317k
        weight=1.0,
    ),
    
    # Squad Turkish (QA Retrieval)
    "squad_tr": DatasetConfig(
        name="squad_tr",
        hf_id="selmanbaysan/squad-tr",
        task_type="retrieval",
        text_columns=["anchor", "positive"],
        weight=1.0,
    ),
    "tquad": DatasetConfig(
        name="tquad",
        hf_id="selmanbaysan/tquad",
        task_type="retrieval",
        text_columns=["anchor", "positive"],
        weight=1.0,
    ),
    
    # Medical/Scientific Retrieval
    "nfcorpus_tr": DatasetConfig(
        name="nfcorpus_tr",
        hf_id="selmanbaysan/nfcorpus-tr",
        task_type="retrieval",
        text_columns=["anchor", "positive"],
        weight=1.5,  # Medical domain - higher weight
    ),
    "nfcorpus_tr_ft": DatasetConfig(
        name="nfcorpus_tr_ft",
        hf_id="selmanbaysan/nfcorpus-tr_fine_tuning_dataset",
        task_type="retrieval",
        text_columns=["anchor", "positive"],
        weight=1.5,
    ),
    "scifact_tr": DatasetConfig(
        name="scifact_tr",
        hf_id="selmanbaysan/scifact-tr",
        task_type="retrieval",
        text_columns=["anchor", "positive"],
        weight=1.5,
    ),
    "scifact_tr_ft": DatasetConfig(
        name="scifact_tr_ft",
        hf_id="selmanbaysan/scifact-tr_fine_tuning_dataset",
        task_type="retrieval",
        text_columns=["anchor", "positive"],
        weight=1.5,
    ),
    "scidocs_tr": DatasetConfig(
        name="scidocs_tr",
        hf_id="selmanbaysan/scidocs-tr",
        task_type="retrieval",
        text_columns=["anchor", "positive"],
        weight=1.5,
    ),
    
    # Financial Retrieval
    "fiqa_tr": DatasetConfig(
        name="fiqa_tr",
        hf_id="selmanbaysan/fiqa-tr",
        task_type="retrieval",
        text_columns=["anchor", "positive"],
        weight=1.2,
    ),
    "fiqa_tr_ft": DatasetConfig(
        name="fiqa_tr_ft",
        hf_id="selmanbaysan/fiqa-tr_fine_tuning_dataset",
        task_type="retrieval",
        text_columns=["anchor", "positive"],
        weight=1.2,
    ),
    
    # Duplicate Question Detection / Semantic Search
    "quora_tr": DatasetConfig(
        name="quora_tr",
        hf_id="selmanbaysan/quora-tr",
        task_type="retrieval",
        text_columns=["anchor", "positive"],
        max_samples=100_000,  # Subsample from 561k
        weight=1.0,
    ),
    "quora_tr_ft": DatasetConfig(
        name="quora_tr_ft",
        hf_id="selmanbaysan/quora-tr_fine_tuning_dataset",
        task_type="retrieval",
        text_columns=["anchor", "positive"],
        weight=1.0,
    ),
    
    # Argument Mining
    "arguana_tr": DatasetConfig(
        name="arguana_tr",
        hf_id="selmanbaysan/arguana-tr",
        task_type="retrieval",
        text_columns=["anchor", "positive"],
        weight=1.2,
    ),
    
    # Community QA
    "cqadupstack_gaming_tr": DatasetConfig(
        name="cqadupstack_gaming_tr",
        hf_id="selmanbaysan/cqadupstack-gaming-tr",
        task_type="retrieval",
        text_columns=["anchor", "positive"],
        weight=1.0,
    ),
    
    # =========================================================================
    # BITEXT MINING (parallel sentence pairs - translation)
    # Format: anchor, positive (translation pairs)
    # Loss: MultipleNegativesRankingLoss
    # =========================================================================
    "wmt16_en_tr": DatasetConfig(
        name="wmt16_en_tr",
        hf_id="selmanbaysan/wmt16_en_tr",
        task_type="bitext",
        text_columns=["anchor", "positive"],
        max_samples=100_000,  # Subsample from 210k
        weight=1.0,
    ),
    "wmt16_en_tr_ft": DatasetConfig(
        name="wmt16_en_tr_ft",
        hf_id="selmanbaysan/wmt16_en_tr_fine_tuning_dataset",
        task_type="bitext",
        text_columns=["anchor", "positive"],
        max_samples=100_000,  # Subsample from 207k
        weight=1.0,
    ),
    
    # =========================================================================
    # STS (Semantic Textual Similarity)
    # Format: sentence1, sentence2, score (0-5 or 0-1)
    # Loss: CoSENTLoss
    # =========================================================================
    "stsb_tr": DatasetConfig(
        name="stsb_tr",
        hf_id="selmanbaysan/stsb-tr",
        task_type="sts",
        text_columns=["sentence1", "sentence2"],
        label_column="score",
        weight=3.0,  # Higher weight - critical for benchmarks
    ),
    
    # =========================================================================
    # CLASSIFICATION (single text with label)
    # Format: text, label
    # Loss: SoftmaxLoss
    # =========================================================================
    "thy_sa": DatasetConfig(
        name="thy_sa",
        hf_id="selmanbaysan/thy_sa",
        task_type="classification",
        text_columns=["text"],
        label_column="label",
        weight=1.0,
    ),
    "offenseval": DatasetConfig(
        name="offenseval",
        hf_id="selmanbaysan/offenseval",
        task_type="classification",
        text_columns=["text"],
        label_column="label",
        weight=1.0,
    ),
    "news_cat": DatasetConfig(
        name="news_cat",
        hf_id="selmanbaysan/news-cat",
        task_type="classification",
        text_columns=["text"],
        label_column="label",
        weight=1.0,
    ),
    "haber_75": DatasetConfig(
        name="haber_75",
        hf_id="selmanbaysan/75haber",
        task_type="classification",
        text_columns=["text"],
        label_column="label",
        weight=1.0,
    ),
    "irony_tr": DatasetConfig(
        name="irony_tr",
        hf_id="selmanbaysan/irony-tr",
        task_type="classification",
        text_columns=["text"],
        label_column="label",
        weight=1.0,
    ),
    "ts_timeline_news_category": DatasetConfig(
        name="ts_timeline_news_category",
        hf_id="selmanbaysan/ts_timeline_news_category",
        task_type="classification",
        text_columns=["text"],
        label_column="label",
        weight=1.0,
    ),
    
    # =========================================================================
    # PAIR CLASSIFICATION (NLI: entailment, contradiction, neutral)
    # Format: premise, hypothesis, label OR anchor, positive, label
    # Loss: SoftmaxLoss with num_labels=3
    # =========================================================================
    "snli_tr": DatasetConfig(
        name="snli_tr",
        hf_id="selmanbaysan/snli_tr",
        task_type="pair_classification",
        text_columns=["anchor", "positive"],
        label_column="label",
        weight=1.0,
    ),
    "snli_tr_ft": DatasetConfig(
        name="snli_tr_ft",
        hf_id="selmanbaysan/snli_tr_fine_tuning_dataset",
        task_type="pair_classification",
        text_columns=["anchor", "positive"],
        label_column="label",
        max_samples=100_000,  # Subsample from 570k
        weight=1.0,
    ),
    "xnli_tr": DatasetConfig(
        name="xnli_tr",
        hf_id="selmanbaysan/xnli_tr",
        task_type="pair_classification",
        text_columns=["anchor", "positive"],
        label_column="label",
        weight=1.0,
    ),
    "xnli_tr_ft": DatasetConfig(
        name="xnli_tr_ft",
        hf_id="selmanbaysan/xnli_tr_fine_tuning_dataset",
        task_type="pair_classification",
        text_columns=["anchor", "positive"],
        label_column="label",
        weight=1.0,
    ),
    "multinli_tr": DatasetConfig(
        name="multinli_tr",
        hf_id="selmanbaysan/multinli_tr",
        task_type="pair_classification",
        text_columns=["anchor", "positive"],
        label_column="label",
        weight=1.0,
    ),
    "multinli_tr_ft": DatasetConfig(
        name="multinli_tr_ft",
        hf_id="selmanbaysan/multinli_tr_fine_tuning_dataset",
        task_type="pair_classification",
        text_columns=["anchor", "positive"],
        label_column="label",
        max_samples=100_000,  # Subsample from 413k
        weight=1.0,
    ),
    
    # =========================================================================
    # CONTRASTIVE LEARNING (general sentence pairs)
    # Large-scale datasets for contrastive pre-training
    # Format: anchor, positive
    # Loss: MultipleNegativesRankingLoss
    # =========================================================================
    "turkish_embedding_training": DatasetConfig(
        name="turkish_embedding_training",
        hf_id="selmanbaysan/turkish_embedding_model_training_data",
        task_type="contrastive",
        text_columns=["anchor", "positive"],
        max_samples=500_000,  # Subsample from 136M (very large!)
        weight=0.5,  # Lower weight since it's general
    ),
    "turkish_embedding_training_colab": DatasetConfig(
        name="turkish_embedding_training_colab",
        hf_id="selmanbaysan/cleaned_turkish_embedding_model_training_data_colab",
        task_type="contrastive",
        text_columns=["anchor", "positive"],
        max_samples=500_000,  # Subsample from 68.4M
        weight=0.5,
    ),
    "turkish_weakly_supervised": DatasetConfig(
        name="turkish_weakly_supervised",
        hf_id="selmanbaysan/turkish_weakly_supervised_contrastive_learning_dataset",
        task_type="contrastive",
        text_columns=["anchor", "positive"],
        max_samples=500_000,  # Subsample from 61M
        weight=0.5,
    ),
    "turkish_weakly_supervised_filtered": DatasetConfig(
        name="turkish_weakly_supervised_filtered",
        hf_id="selmanbaysan/turkish_weakly_supervised_contrastive_learning_dataset_filtered",
        task_type="contrastive",
        text_columns=["anchor", "positive"],
        weight=1.0,  # Small dataset, use as is
    ),
    
    # =========================================================================
    # PARAPHRASE / SEMANTIC SIMILARITY (P2P - paragraph to paragraph)
    # Format: anchor, positive (paraphrase pairs)
    # Loss: MultipleNegativesRankingLoss
    # =========================================================================
    "ts_abstract_corpus_p2p": DatasetConfig(
        name="ts_abstract_corpus_p2p",
        hf_id="selmanbaysan/ts_abstract_corpus_p2p",
        task_type="paraphrase",
        text_columns=["anchor", "positive"],
        weight=1.5,  # Academic abstracts - valuable
    ),
    "koseyazisi_p2p": DatasetConfig(
        name="koseyazisi_p2p",
        hf_id="selmanbaysan/630koseyazisi_p2p",
        task_type="paraphrase",
        text_columns=["anchor", "positive"],
        weight=1.5,  # Opinion columns - valuable
    ),
    "ts_abstract_corpus": DatasetConfig(
        name="ts_abstract_corpus",
        hf_id="selmanbaysan/ts_abstract_corpus",
        task_type="paraphrase",
        text_columns=["anchor", "positive"],
        weight=1.0,
    ),
    "koseyazisi": DatasetConfig(
        name="koseyazisi",
        hf_id="selmanbaysan/630koseyazisi",
        task_type="paraphrase",
        text_columns=["anchor", "positive"],
        weight=1.0,
    ),
}

# ============================================================================
# DEFAULT DATASET SELECTIONS
# ============================================================================

# Curated selection for fine-tuning (balanced, diverse)
DEFAULT_DATASETS = [
    # Retrieval (diverse domains)
    "msmarco_tr_ft",
    "squad_tr",
    "tquad",
    "nfcorpus_tr_ft",
    "fiqa_tr_ft",
    "quora_tr_ft",
    "scifact_tr_ft",
    "arguana_tr",
    # Bitext
    "wmt16_en_tr_ft",
    # STS
    "stsb_tr",
    # NLI
    "snli_tr_ft",
    "xnli_tr_ft",
    "multinli_tr_ft",
    # Classification
    "thy_sa",
    "offenseval",
    "news_cat",
    "irony_tr",
    # Paraphrase
    "ts_abstract_corpus_p2p",
    "koseyazisi_p2p",
]

# All datasets (for comprehensive training)
ALL_DATASETS = list(DATASET_REGISTRY.keys())

# Quick training subset (fastest)
QUICK_DATASETS = [
    "stsb_tr",
    "squad_tr",
    "snli_tr_ft",
    "thy_sa",
    "ts_abstract_corpus_p2p",
]

# Retrieval-focused subset
RETRIEVAL_DATASETS = [
    "msmarco_tr_ft",
    "squad_tr",
    "tquad",
    "nfcorpus_tr_ft",
    "fiqa_tr_ft",
    "quora_tr_ft",
    "scifact_tr_ft",
    "scidocs_tr",
    "arguana_tr",
    "cqadupstack_gaming_tr",
]


# ============================================================================
# DATASET LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_dataset(config: DatasetConfig, max_samples: int = None) -> Dataset:
    """Load a dataset and preprocess it for training."""
    logger.info(f"Loading dataset: {config.name} from {config.hf_id}")
    
    try:
        # Try train split first, fall back to test if train doesn't exist
        try:
            dataset = load_dataset(config.hf_id, split="train")
        except ValueError as e:
            if "train" in str(e):
                logger.info(f"  No train split, using test split for {config.name}")
                dataset = load_dataset(config.hf_id, split="test")
            else:
                raise e
    except Exception as e:
        logger.error(f"Failed to load dataset {config.hf_id}: {e}")
        return None
    
    original_size = len(dataset)
    
    # Filter out rows with None values in text columns
    text_cols = config.text_columns
    if text_cols:
        def filter_none_values(example):
            for col in text_cols:
                if col in example and example[col] is None:
                    return False
            return True
        
        dataset = dataset.filter(filter_none_values)
        if len(dataset) < original_size:
            logger.info(f"  Filtered None values: {original_size:,} -> {len(dataset):,}")
            original_size = len(dataset)
    
    # Apply max_samples limit
    effective_max = max_samples or config.max_samples
    if effective_max and len(dataset) > effective_max:
        dataset = dataset.shuffle(seed=42).select(range(effective_max))
        logger.info(f"  Subsampled {config.name}: {original_size:,} -> {len(dataset):,}")
    
    logger.info(f"  Loaded {config.name}: {len(dataset):,} samples")
    return dataset


def prepare_all_datasets(
    dataset_names: list,
    max_samples_per_task: int = None,
) -> dict:
    """Load and prepare all specified datasets."""
    datasets = {}
    
    for name in dataset_names:
        if name not in DATASET_REGISTRY:
            logger.warning(f"Unknown dataset: {name}, skipping")
            continue
        
        config = DATASET_REGISTRY[name]
        dataset = load_and_preprocess_dataset(config, max_samples_per_task)
        
        if dataset is not None:
            datasets[name] = {
                "dataset": dataset,
                "config": config,
            }
    
    return datasets


# ============================================================================
# LOSS FUNCTION CREATION
# ============================================================================

def create_loss_functions(
    model: SentenceTransformer,
    datasets: dict,
    embedding_dim: int = 768,
) -> dict:
    """
    Create appropriate loss functions for each dataset based on task type.
    
    Loss Mapping:
        - retrieval, bitext, contrastive, paraphrase: MultipleNegativesRankingLoss
        - sts: CoSENTLoss
        - classification: SoftmaxLoss (single text)
        - pair_classification: SoftmaxLoss (text pairs - NLI)
    """
    losses_dict = {}
    
    # Create shared loss instances per task type
    mnrl_loss = losses.MultipleNegativesRankingLoss(model=model)
    cosent_loss = losses.CoSENTLoss(model=model)
    
    # Track unique label counts for classification tasks
    classification_num_labels = {}
    
    for name, data in datasets.items():
        config = data["config"]
        dataset = data["dataset"]
        task_type = config.task_type
        
        if task_type in ("retrieval", "bitext", "contrastive", "paraphrase"):
            losses_dict[name] = mnrl_loss
            logger.info(f"  {name}: MultipleNegativesRankingLoss ({task_type})")
            
        elif task_type == "sts":
            losses_dict[name] = cosent_loss
            logger.info(f"  {name}: CoSENTLoss")
            
        elif task_type == "classification":
            # Get number of unique labels
            if config.label_column and config.label_column in dataset.column_names:
                num_labels = len(set(dataset[config.label_column]))
            else:
                num_labels = 3  # Default fallback
            
            softmax_loss = losses.SoftmaxLoss(
                model=model,
                sentence_embedding_dimension=embedding_dim,
                num_labels=num_labels,
            )
            losses_dict[name] = softmax_loss
            logger.info(f"  {name}: SoftmaxLoss (num_labels={num_labels})")
            
        elif task_type == "pair_classification":
            # NLI typically has 3 labels: entailment, neutral, contradiction
            num_labels = 3
            if config.label_column and config.label_column in dataset.column_names:
                num_labels = len(set(dataset[config.label_column]))
            
            softmax_loss = losses.SoftmaxLoss(
                model=model,
                sentence_embedding_dimension=embedding_dim,
                num_labels=num_labels,
            )
            losses_dict[name] = softmax_loss
            logger.info(f"  {name}: SoftmaxLoss (NLI, num_labels={num_labels})")
        
        else:
            logger.warning(f"  Unknown task type: {task_type} for {name}, using MNRL")
            losses_dict[name] = mnrl_loss
    
    return losses_dict


# ============================================================================
# EVALUATOR CREATION
# ============================================================================

def create_evaluator(model: SentenceTransformer, datasets: dict) -> EmbeddingSimilarityEvaluator:
    """Create an evaluator using the STS dataset if available."""
    for name, data in datasets.items():
        config = data["config"]
        if config.task_type == "sts":
            dataset = data["dataset"]
            
            # Check column names
            col_names = dataset.column_names
            
            # Try different column name patterns
            sent1_col = None
            sent2_col = None
            score_col = None
            
            for col in col_names:
                if col in ["sentence1", "sent1", "text1", "anchor"]:
                    sent1_col = col
                elif col in ["sentence2", "sent2", "text2", "positive"]:
                    sent2_col = col
                elif col in ["score", "similarity", "label"]:
                    score_col = col
            
            if sent1_col and sent2_col and score_col:
                # Use a subset for evaluation
                eval_samples = min(1000, len(dataset))
                eval_ds = dataset.shuffle(seed=42).select(range(eval_samples))
                
                evaluator = EmbeddingSimilarityEvaluator(
                    sentences1=eval_ds[sent1_col],
                    sentences2=eval_ds[sent2_col],
                    scores=eval_ds[score_col],
                    name=f"sts_eval_{name}",
                )
                logger.info(f"Created STS evaluator from {name} with {eval_samples} samples")
                return evaluator
    
    logger.info("No STS dataset found for evaluation")
    return None


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def get_embedding_dimension(model: SentenceTransformer) -> int:
    """Get the embedding dimension from the model."""
    # Try to get from pooling layer
    for module in model:
        if hasattr(module, "pooling_output_dimension"):
            return module.pooling_output_dimension
    
    # Try to get from dense layer
    for module in model:
        if hasattr(module, "linear"):
            return module.linear.out_features
    
    # Try to get from auto_model hidden size
    for module in model:
        if hasattr(module, "auto_model"):
            if hasattr(module.auto_model, "config"):
                return module.auto_model.config.hidden_size
    
    # Default fallback
    return 768


def setup_model(model_id: str, use_bf16: bool = True, gradient_checkpointing: bool = True) -> SentenceTransformer:
    """Load and configure the base model."""
    logger.info(f"Loading model: {model_id}")
    
    # Model kwargs for transformers
    model_kwargs = {}
    
    # Try to use Flash Attention 2
    try:
        import flash_attn
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Flash Attention 2 enabled")
    except ImportError:
        logger.info("flash-attn not installed, using standard attention")
    
    model = SentenceTransformer(model_id, model_kwargs=model_kwargs)
    
    # Apply precision settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    if use_bf16 and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            model = model.to(dtype=torch.bfloat16)
            logger.info("Using bfloat16 precision")
        else:
            logger.info("bfloat16 not supported on this GPU")
    
    # Enable gradient checkpointing
    if gradient_checkpointing:
        for module in model:
            if hasattr(module, "auto_model"):
                if hasattr(module.auto_model, "gradient_checkpointing_enable"):
                    module.auto_model.gradient_checkpointing_enable()
                    logger.info("Gradient checkpointing enabled")
    
    return model


# ============================================================================
# TRAINING ARGUMENT CREATION
# ============================================================================

def create_training_args(config: TrainingConfig) -> SentenceTransformerTrainingArguments:
    """Create training arguments from config."""
    
    # Determine precision - bf16/fp16 work on CUDA and MPS (Apple Silicon)
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    fp16 = config.use_fp16 and has_cuda  # fp16 only on CUDA
    bf16 = config.use_bf16 and (has_cuda or has_mps) and not fp16
    
    # MPS doesn't support multiprocessing dataloaders
    num_workers = 0 if has_mps else config.dataloader_num_workers
    pin_memory = not has_mps  # pin_memory not supported on MPS
    
    args = SentenceTransformerTrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        fp16=fp16,
        bf16=bf16,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_steps=config.logging_steps,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=pin_memory,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        hub_token=HF_TOKEN,
        seed=config.seed,
        # Report to WandB if enabled
        report_to="wandb" if (config.use_wandb and WANDB_API_KEY) else "none",
        run_name=config.wandb_run_name,
    )
    
    return args


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train(config: TrainingConfig, dataset_names: list = None):
    """Main training function."""
    
    # Use default datasets if none specified
    if dataset_names is None:
        dataset_names = DEFAULT_DATASETS
    
    logger.info("=" * 70)
    logger.info("MULTI-TASK SUPERVISED FINE-TUNING")
    logger.info("=" * 70)
    logger.info(f"Model: {config.model_id}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Datasets: {dataset_names}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info("=" * 70)
    
    # Check for GPU
    if not torch.cuda.is_available():
        logger.warning("CUDA not available! Training will be slow on CPU.")
    else:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model
    model = setup_model(
        config.model_id,
        use_bf16=config.use_bf16,
        gradient_checkpointing=config.gradient_checkpointing,
    )
    
    # Get embedding dimension
    embedding_dim = get_embedding_dimension(model)
    logger.info(f"Embedding dimension: {embedding_dim}")
    
    # Load datasets
    logger.info("\n--- Loading Datasets ---")
    datasets = prepare_all_datasets(dataset_names, config.max_samples_per_task)
    
    if not datasets:
        logger.error("No datasets loaded! Exiting.")
        return
    
    # Create loss functions
    logger.info("\n--- Creating Loss Functions ---")
    losses_dict = create_loss_functions(model, datasets, embedding_dim)
    
    # Prepare dataset dict for trainer
    train_datasets = {name: data["dataset"] for name, data in datasets.items()}
    
    # Print dataset statistics
    logger.info("\n--- Dataset Statistics ---")
    total_samples = 0
    for name, dataset in train_datasets.items():
        logger.info(f"  {name}: {len(dataset):,} samples")
        total_samples += len(dataset)
    logger.info(f"  TOTAL: {total_samples:,} samples")
    
    # Create evaluator
    evaluator = create_evaluator(model, datasets)
    
    # Dry run check
    if config.dry_run:
        logger.info("\n=== DRY RUN MODE ===")
        logger.info("Would train with the following configuration:")
        logger.info(f"  Datasets: {list(train_datasets.keys())}")
        logger.info(f"  Total samples: {total_samples:,}")
        logger.info(f"  Loss functions: {list(losses_dict.keys())}")
        logger.info("Exiting without training.")
        return
    
    # Initialize WandB if enabled
    if config.use_wandb and WANDB_API_KEY:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config={
                    "model_id": config.model_id,
                    "datasets": dataset_names,
                    "total_samples": total_samples,
                    "embedding_dim": embedding_dim,
                    "num_epochs": config.num_epochs,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "warmup_ratio": config.warmup_ratio,
                    "use_bf16": config.use_bf16,
                },
            )
            logger.info("WandB initialized")
        except Exception as e:
            logger.warning(f"WandB initialization failed: {e}")
    
    # Create training arguments
    args = create_training_args(config)
    
    # Create MTEB evaluation callback
    callbacks = []
    if config.eval_steps > 0:
        mteb_callback = MTEBEvaluationCallback(
            model=model,
            eval_steps=config.eval_steps,
            eval_tasks=config.eval_tasks,
            output_folder=f"{config.output_dir}/mteb_results",
        )
        callbacks.append(mteb_callback)
        logger.info(f"MTEB evaluation enabled every {config.eval_steps} steps")
    
    # Create trainer
    logger.info("\n--- Initializing Trainer ---")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_datasets,
        loss=losses_dict,
        evaluator=evaluator,
        callbacks=callbacks if callbacks else None,
    )
    
    # Start training
    logger.info("\n--- Starting Training ---")
    trainer.train()

    
    # Save final model
    logger.info("\n--- Saving Final Model ---")
    model.save(config.output_dir)
    logger.info(f"Model saved to: {config.output_dir}")
    
    # Push to Hub
    if config.push_to_hub and HF_TOKEN:
        try:
            url = model.push_to_hub(
                repo_id=config.hub_model_id,
                token=HF_TOKEN,
                exist_ok=True,
            )
            logger.info(f"Model pushed to Hub: {url}")
        except Exception as e:
            logger.error(f"Failed to push to Hub: {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 70)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Task Supervised Fine-Tuning for Turkish Embedding Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model arguments
    parser.add_argument(
        "--model-id",
        type=str,
        default="alibayram/magibu-200m",
        help="Base model ID from Hugging Face Hub",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./magibu-200m-sft",
        help="Output directory for checkpoints and final model",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default="alibayram/magibu-200m-sft",
        help="Model ID for pushing to Hugging Face Hub",
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio of total training steps",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="List of dataset names to use (overrides --preset)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["default", "all", "quick", "retrieval"],
        default="default",
        help="Dataset preset: 'default' (balanced 20 datasets), 'all' (38 datasets), "
             "'quick' (5 datasets for testing), 'retrieval' (10 retrieval-focused)",
    )
    parser.add_argument(
        "--max-samples-per-task",
        type=int,
        default=None,
        help="Maximum samples per task for balancing",
    )
    
    # Optimization arguments
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable bfloat16 precision",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 precision instead of bfloat16",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing",
    )
    
    # Logging arguments
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=100,
        help="Log every N steps",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=1000,
        help="Run MTEB evaluation every N steps (0 to disable)",
    )
    parser.add_argument(
        "--eval-all-tasks",
        action="store_true",
        help="Evaluate on ALL Turkish MTEB tasks (instead of quick subset)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="magibu-sft",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="WandB run name",
    )
    
    # Hub arguments
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Disable pushing to Hugging Face Hub",
    )
    
    # Misc arguments
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load datasets and print statistics without training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets and exit",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # List datasets mode
    if args.list_datasets:
        print("\n" + "=" * 70)
        print("AVAILABLE DATASETS (38 total)")
        print("=" * 70)
        
        # Group by task type
        task_groups = {}
        for name, config in DATASET_REGISTRY.items():
            if config.task_type not in task_groups:
                task_groups[config.task_type] = []
            task_groups[config.task_type].append((name, config))
        
        for task_type, datasets in task_groups.items():
            print(f"\n[{task_type.upper()}] ({len(datasets)} datasets)")
            print("-" * 50)
            for name, config in datasets:
                max_info = f" (max: {config.max_samples:,})" if config.max_samples else ""
                weight_info = f" w={config.weight}" if config.weight != 1.0 else ""
                print(f"  {name}{max_info}{weight_info}")
        
        print("\n" + "=" * 70)
        print("DATASET PRESETS")
        print("=" * 70)
        print(f"\n--preset default  ({len(DEFAULT_DATASETS)} datasets): Curated, balanced selection")
        print(f"--preset all      ({len(ALL_DATASETS)} datasets): All 38 datasets")
        print(f"--preset quick    ({len(QUICK_DATASETS)} datasets): Fast testing/debugging")
        print(f"--preset retrieval ({len(RETRIEVAL_DATASETS)} datasets): Retrieval-focused")
        print()
        return
    
    # Resolve dataset list from preset or explicit --datasets
    if args.datasets:
        dataset_names = args.datasets
    else:
        preset_map = {
            "default": DEFAULT_DATASETS,
            "all": ALL_DATASETS,
            "quick": QUICK_DATASETS,
            "retrieval": RETRIEVAL_DATASETS,
        }
        dataset_names = preset_map.get(args.preset, DEFAULT_DATASETS)
    
    # Build config from args
    config = TrainingConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        hub_model_id=args.hub_model_id,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        use_bf16=not args.no_bf16 and not args.fp16,
        use_fp16=args.fp16,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_all_tasks=args.eval_all_tasks,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        push_to_hub=not args.no_push,
        max_samples_per_task=args.max_samples_per_task,
        dry_run=args.dry_run,
        seed=args.seed,
    )
    
    # Set evaluation tasks based on flag
    if config.eval_all_tasks:
        config.eval_tasks = MTEBEvaluationCallback.ALL_TR_TASKS
        print(f"Enabled ALL {len(config.eval_tasks)} Turkish MTEB tasks for evaluation")
    else:
        config.eval_tasks = MTEBEvaluationCallback.QUICK_EVAL_TASKS
        print(f"Enabled QUICK subset ({len(config.eval_tasks)} tasks) for MTEB evaluation")

    # Run training
    train(config, dataset_names=dataset_names)


if __name__ == "__main__":
    main()

