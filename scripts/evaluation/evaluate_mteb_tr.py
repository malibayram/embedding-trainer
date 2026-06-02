#!/usr/bin/env python3
"""
MTEB-TR Evaluation Script

Evaluate any SentenceTransformer model on Turkish MTEB tasks.
"""
import os
import sys
import argparse
import logging
from sentence_transformers import SentenceTransformer

# Add mteb-tr to path
mteb_tr_path = os.path.abspath("mteb-tr")
if os.path.exists(mteb_tr_path) and mteb_tr_path not in sys.path:
    sys.path.insert(0, mteb_tr_path)
    
import mteb

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# List of all Turkish tasks
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
    
    # STS
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
    
    # Pair Classification
    "SnliTr",
    "MnliTr",
]

QUICK_TASKS = ["STSbTR", "TurkishMovieSentimentClassification"]

def load_model(model_name: str, device: str = None):
    """Load model with trust_remote_code=True"""
    logger.info(f"Loading model: {model_name}")
    return SentenceTransformer(model_name, trust_remote_code=True, device=device)

def evaluate(model_name: str, output_folder: str, tasks: list, device: str = None):
    """Run evaluation"""
    model = load_model(model_name, device)
    
    logger.info(f"Loading {len(tasks)} tasks...")
    mteb_tasks = mteb.get_tasks(tasks=tasks)
    
    if not mteb_tasks:
        logger.error(f"No tasks found! Requested: {tasks}")
        return
        
    evaluation = mteb.MTEB(tasks=mteb_tasks)
    
    logger.info(f"Starting evaluation on {len(mteb_tasks)} tasks...")
    logger.info(f"Results will be saved to: {output_folder}")
    
    # Run evaluation
    results = evaluation.run(
        model,
        output_folder=output_folder,
        eval_splits=["test"],
        verbosity=1,
    )
    
    # Print Summary
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS ({model_name})")
    print("="*80)
    print(f"{'Task':<50} | {'Score':<10}")
    print("-" * 65)
    
    scores = []
    
    for task_result in results:
        task_name = task_result.task_name
        
        # Extract main score
        main_score = 0.0
        if hasattr(task_result, 'scores') and task_result.scores:
            test_scores = task_result.scores.get('test', [])
            if test_scores and len(test_scores) > 0:
                score_dict = test_scores[0]
                main_score = score_dict.get('main_score', 
                             score_dict.get('cos_sim', {}).get('spearman',
                             score_dict.get('accuracy', 0)))
        
        scores.append(main_score)
        print(f"{task_name:<50} | {main_score:.4f}")
        
    if scores:
        print("-" * 65)
        print(f"{'MEAN SCORE':<50} | {sum(scores)/len(scores):.4f}")
    print("="*80 + "\n")

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on TR-MTEB")
    parser.add_argument("model_name", type=str, help="Model ID or path (e.g. alibayram/magibu-200m)")
    parser.add_argument("--output-dir", type=str, default="mteb_results", help="Output folder")
    parser.add_argument("--all-tasks", action="store_true", help="Run all Turkish tasks")
    parser.add_argument("--task", type=str, action="append", help="Specific task(s) to run")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, cpu, mps)")
    
    args = parser.parse_args()
    
    # Determine tasks
    if args.all_tasks:
        eval_tasks = ALL_TR_TASKS
    elif args.task:
        eval_tasks = args.task
    else:
        print(f"No tasks specified. Using quick subset: {QUICK_TASKS}")
        eval_tasks = QUICK_TASKS
        
    evaluate(args.model_name, args.output_dir, eval_tasks, args.device)

if __name__ == "__main__":
    main()
