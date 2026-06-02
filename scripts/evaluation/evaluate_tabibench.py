
import argparse
import logging
import sys
import os
import datasets
from typing import Type

# Convert generic task types to MTEB abstract base classes
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Ensure mteb-tr is in path
mteb_tr_path = os.path.abspath("mteb-tr")
if os.path.exists(mteb_tr_path) and mteb_tr_path not in sys.path:
    sys.path.insert(0, mteb_tr_path)

import mteb
from mteb import MTEB
from mteb.abstasks import AbsTaskClassification, AbsTaskRetrieval, AbsTaskSTS, AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---

CLASSIFICATION_DATASETS = {
    # All verified to use text/label
    "boun-tabilab/Turkish-Product-Reviews": {"text": "text", "label": "label"},
    "boun-tabilab/News-Cat": {"text": "text", "label": "label"},
    "boun-tabilab/BilTweetNews-Sentiment-Analysis": {"text": "text", "label": "label"},
    "boun-tabilab/Gender-Hate-Speech-TR": {"text": "text", "label": "label"},
    "boun-tabilab/Sci-Cite-TR": {"text": "text", "label": "label"},
    "boun-tabilab/Pubmed-RCT-10K-TR": {"text": "text", "label": "label"},
    "boun-tabilab/Thesis-Abstract-Classification-11K": {"text": "text", "label": "label"},
}

RETRIEVAL_DATASETS = {
    "boun-tabilab/MsMarco-TR": {"query": "anchor", "doc": "positive"},
    "boun-tabilab/Scifact-TR": {"query": "anchor", "doc": "positive"},
    "boun-tabilab/NFCorpus-TR": {"query": "anchor", "doc": "positive"},
    "boun-tabilab/Quora-TR": {"query": "anchor", "doc": "positive"},
    "boun-tabilab/Fiqa-TR": {"query": "anchor", "doc": "positive"},
    "boun-tabilab/Apps-Retrieval-TR": {"query": "query", "doc": "doc"},
    "boun-tabilab/Code-Search-Net-21K-TR": {"query": "doc", "doc": "query"},
    "boun-tabilab/Stackoverflow-QA-TR": {"query": "query", "doc": "doc"},
    "boun-tabilab/Cos-QA-TR": {"query": "query", "doc": "doc"},
}

STS_DATASETS = {
    "boun-tabilab/STSb-TR": {"sent1": "sentence1", "sent2": "sentence2", "score": "score"},
}

NLI_DATASETS = {
    "boun-tabilab/SNLI-TR": {"sent1": "premise", "sent2": "hypothesis", "label": "label"},
    "boun-tabilab/MultiNLI-TR": {"sent1": "premise", "sent2": "hypothesis", "label": "label"},
    "boun-tabilab/Med-NLI-TR": {"sent1": "sentence1", "sent2": "sentence2", "label": "label"},
}

# --- Dynamic Class Factory ---

def create_task_class(dataset_id: str, config: dict, task_type: str, base_cls: Type):
    short_name = dataset_id.split("/")[-1]
    
    # Define metadata
    meta = TaskMetadata(
        name=short_name,
        description=f"Generic {task_type} task for {dataset_id}",
        dataset={"path": dataset_id, "revision": "main"}, # Use main as default revision
        type=task_type,
        category="s2s" if task_type != "Retrieval" else "s2p", # Retrieval usually s2p
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="accuracy" if task_type == "Classification" else ("ndcg_at_10" if task_type == "Retrieval" else ("spearman" if task_type == "STS" else "accuracy")),
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="",
        prompt="Classify" if task_type == "Classification" else None
    )

    # Dynamic class
    class DynamicTask(base_cls):
        metadata = meta
        _dataset_id = dataset_id
        _config = config
        
        def load_data(self, **kwargs):
            if self.data_loaded: return
            
            ds = load_dataset(self._dataset_id)
            self.dataset = {}
            
            nonlocal task_type 
            # Note: nonlocal refers to create_task_class scope which is fine
            
            # --- CLASSIFICATION ---
            # --- CLASSIFICATION ---
            if task_type == "Classification":
                # Handle NLI vs Standard Loading
                is_nli = "sent1" in self._config
                
                if is_nli:
                    sent1_col = self._config["sent1"]
                    sent2_col = self._config["sent2"]
                    label_col = self._config["label"]
                else:
                    text_col = self._config["text"]
                    label_col = self._config["label"]

                for split in ds.keys():
                    def map_fn(ex):
                        if is_nli:
                            s1 = ex[sent1_col]
                            s2 = ex[sent2_col]
                            return {
                                "text": f"Premise: {s1} Hypothesis: {s2}",
                                "label": ex[label_col]
                            }
                        return {"text": ex[text_col], "label": ex[label_col]}
                    
                    self.dataset[split] = ds[split].map(map_fn, remove_columns=ds[split].column_names)

            # --- RETRIEVAL ---
            elif task_type == "Retrieval":
                query_col = self._config["query"]
                doc_col = self._config["doc"]
                
                # Setup retrieval structure
                split = "test" if "test" in ds else ("validation" if "validation" in ds else "train")
                if "test" not in ds: logger.warning(f"Using {split} for {self._dataset_id}")
                
                data = ds[split]
                corpus = {}
                queries = {}
                qrels = {}
                
                for idx, row in enumerate(data):
                    q_text = row[query_col]
                    d_text = row[doc_col]
                    if not q_text or not d_text: continue
                    
                    q_id = f"q{idx}"
                    d_id = f"d{idx}"
                    
                    queries[q_id] = q_text
                    corpus[d_id] = {"text": d_text, "title": ""}
                    if q_id not in qrels: qrels[q_id] = {}
                    qrels[q_id][d_id] = 1

                self.corpus = {split: corpus}
                self.queries = {split: queries}
                self.relevant_docs = {split: qrels}
                
                # MTEB expects self.dataset to have query/corpus/relevant_docs? 
                # Actually AbsTaskRetrieval uses these attributes directly usually, 
                # but let's check if we need to set self.dataset differently.
                # Standard implementation sets self.corpus/queries/relevant_docs. 
                # self.dataset is usually just the raw HF dataset.
                # But we override load_data so we are fine.
                return

            # --- STS ---
            elif task_type == "STS":
                s1 = self._config["sent1"]
                s2 = self._config["sent2"]
                sc = self._config["score"]
                for split in ds.keys():
                    self.dataset[split] = ds[split].rename_columns({s1: "sentence1", s2: "sentence2", sc: "score"})
            
            # Helper: ensure test split
            if "test" not in self.dataset and "validation" in self.dataset:
                self.dataset["test"] = self.dataset["validation"]
            
            self.data_loaded = True

    # Rename class for nicer logs
    DynamicTask.__name__ = short_name.replace("-", "_")
    return DynamicTask

# --- Main ---

def evaluate(model_name: str, output_folder: str, tasks: str, device: str, batch_size: int = 16, revision: str = None):
    logger.info(f"Loading model: {model_name} (revision: {revision})")
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device, revision=revision)
    
    # Collect requested tasks
    requested_ids = tasks.split(",") if tasks else None
    
    tasks_to_run = []
    
    def add_if_requested(datasets_map, cls_type, mteb_base):
        for name, cfg in datasets_map.items():
            short = name.split("/")[-1]
            if requested_ids and name not in requested_ids and short not in requested_ids:
                continue
            
            # Create dynamic class
            TaskClass = create_task_class(name, cfg, cls_type, mteb_base)
            # Instantiate
            tasks_to_run.append(TaskClass())

    add_if_requested(CLASSIFICATION_DATASETS, "Classification", AbsTaskClassification)
    add_if_requested(RETRIEVAL_DATASETS, "Retrieval", AbsTaskRetrieval)
    add_if_requested(STS_DATASETS, "STS", AbsTaskSTS)
    
    # NLI as Classification
    add_if_requested(NLI_DATASETS, "Classification", AbsTaskClassification)

    if not tasks_to_run:
        logger.error("No valid tasks found to run.")
        return

    logger.info(f"Running {len(tasks_to_run)} tasks...")
    
    # Pass instances to MTEB
    evaluation = MTEB(tasks=tasks_to_run)
    
    # Use raise_error=False to continue if a task crashes
    # Use overwrite_results=False (default) to skip already done tasks
    evaluation.run(
        model, 
        output_folder=output_folder, 
        verbosity=1, 
        raise_error=False,
        overwrite_results=False,
        encode_kwargs={"batch_size": batch_size}
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("--output-dir", default="tabibench_results")
    parser.add_argument("--tasks", help="Comma separated list of tasks")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for encoding (reduce if MPS errors)")
    parser.add_argument("--revision", type=str, default=None, help="Model revision (commit hash) to use")
    args = parser.parse_args()
    
    evaluate(args.model_name, args.output_dir, args.tasks, args.device, args.batch_size, args.revision)
