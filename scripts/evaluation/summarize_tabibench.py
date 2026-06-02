
import json
import os
import glob
import pandas as pd

RESULTS_DIR = "tabibench_results/magibu__embeddingmagibu-200m/e1351a62ee65e670f56df29ed597e0a2fa7e9848"

# Map file names to task types for grouping
TASK_TYPES = {
    # Classification
    "Turkish-Product-Reviews": "Classification",
    "News-Cat": "Classification",
    "BilTweetNews-Sentiment-Analysis": "Classification",
    "Gender-Hate-Speech-TR": "Classification",
    "Sci-Cite-TR": "Classification",
    "Pubmed-RCT-10K-TR": "Classification",
    "Thesis-Abstract-Classification-11K": "Classification",
    "SNLI-TR": "Classification (NLI)",
    "MultiNLI-TR": "Classification (NLI)",
    "Med-NLI-TR": "Classification (NLI)",
    
    # Retrieval
    "MsMarco-TR": "Retrieval",
    "Scifact-TR": "Retrieval",
    "NFCorpus-TR": "Retrieval",
    "Quora-TR": "Retrieval",
    "Fiqa-TR": "Retrieval",
    "Apps-Retrieval-TR": "Retrieval",
    "Code-Search-Net-21K-TR": "Retrieval",
    "Stackoverflow-QA-TR": "Retrieval",
    "Cos-QA-TR": "Retrieval",
    
    # STS
    "STSb-TR": "STS",
}

results = []

for filepath in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
    filename = os.path.basename(filepath)
    if filename == "model_meta.json": continue
    
    task_name = filename.replace(".json", "")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract main score
    # MTEB structure: scores -> test -> [0] -> main_score OR direct main_score key
    score = 0.0
    metric = "N/A"
    
    try:
        test_res = data.get("scores", {}).get("test", [])
        if isinstance(test_res, list) and len(test_res) > 0:
            # It's a list of dicts (one per subset usually, here usually 'default')
            res_dict = test_res[0] 
            # Or is it a dict of lists? MTEB Json structure varies slightly by version
            # Let's check 'test' type
        elif isinstance(test_res, dict):
            # Dict of subsets? e.g. {'default': {...}}
            if 'default' in test_res:
                res_dict = test_res['default']
            else:
                # Take first val
                res_dict = list(test_res.values())[0]
        else:
            res_dict = {}

        if "main_score" in res_dict:
            score = res_dict["main_score"] * 100
            # Infer metric
            if "accuracy" in res_dict: metric = "Accuracy"
            elif "ndcg_at_10" in res_dict: metric = "NDCG@10"
            elif "spearman" in res_dict: metric = "Spearman"
            elif "ap" in res_dict: metric = "AP"
        elif "accuracy" in res_dict:
            score = res_dict["accuracy"] * 100
            metric = "Accuracy"
        elif "ndcg_at_10" in res_dict:
            score = res_dict["ndcg_at_10"] * 100
            metric = "NDCG@10"
        elif "cos_sim" in res_dict and "spearman" in res_dict["cos_sim"]:
            score = res_dict["cos_sim"]["spearman"] * 100
            metric = "Spearman"
            
    except Exception as e:
        print(f"Error parsing {task_name}: {e}")
        
    task_type = TASK_TYPES.get(task_name, "Other")
    results.append({
        "Task": task_name,
        "Type": task_type,
        "Metric": metric,
        "Score": score
    })

# Create DataFrame
df = pd.DataFrame(results)
df = df.sort_values(by=["Type", "Task"])

# Print Markdown Table
print("\n### TabiBench Evaluation Results\n")
print(df.to_markdown(index=False, floatfmt=".2f"))

# Calculate Averages
print("\n### Averages by Type\n")
print(df.groupby("Type")["Score"].mean().to_markdown(floatfmt=".2f"))
