
import json
import os
import glob
import pandas as pd

RESULTS_ROOT = "tabibench_results"

# Map file names to task types
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

def parse_score(data):
    try:
        test_res = data.get("scores", {}).get("test", [])
        if isinstance(test_res, list) and len(test_res) > 0:
            res_dict = test_res[0]
        elif isinstance(test_res, dict):
            if 'default' in test_res:
                res_dict = test_res['default']
            else:
                res_dict = list(test_res.values())[0] if test_res else {}
        else:
            res_dict = {}

        score = 0.0
        if "main_score" in res_dict:
            score = res_dict["main_score"] * 100
        elif "accuracy" in res_dict:
            score = res_dict["accuracy"] * 100
        elif "ndcg_at_10" in res_dict:
            score = res_dict["ndcg_at_10"] * 100
        elif "cos_sim" in res_dict and "spearman" in res_dict["cos_sim"]:
            score = res_dict["cos_sim"]["spearman"] * 100
        elif "ap" in res_dict:
            score = res_dict["ap"] * 100
            
        return score
    except:
        return 0.0

all_records = []

# Directory output structure: tabibench_results/MODEL_NAME/REVISION/TASK.json
for model_dir in glob.glob(os.path.join(RESULTS_ROOT, "*")):
    if not os.path.isdir(model_dir): continue
    model_name_clean = os.path.basename(model_dir).replace("__", "/")
    
    # There might be multiple revisions, find the latest one or all?
    # Let's simple walk all revisions
    for rev_dir in glob.glob(os.path.join(model_dir, "*")):
        if not os.path.isdir(rev_dir): continue
        revision = os.path.basename(rev_dir)
        
        # Check if this revision has meaningful results
        json_files = glob.glob(os.path.join(rev_dir, "*.json"))
        if len(json_files) <= 1: continue # Only model_meta.json or empty
        
        short_rev = revision[:6]
        display_name = f"{model_name_clean} ({short_rev})"
        
        # We manually prefer specific revisions if known, otherwise just include all
        # User is comparing magibu (e1351) vs TabiBERT (newly fetched)
        
        for filepath in json_files:
            filename = os.path.basename(filepath)
            if filename == "model_meta.json": continue
            
            task_name = filename.replace(".json", "")
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            score = parse_score(data)
            all_records.append({
                "Task": task_name,
                "Type": TASK_TYPES.get(task_name, "Other"),
                "Model": display_name,
                "Score": score
            })

df = pd.DataFrame(all_records)

if df.empty:
    print("No results found.")
    exit()

# Pivot for comparison
pivot_df = df.pivot_table(index=["Type", "Task"], columns="Model", values="Score")
pivot_df = pivot_df.reset_index()

markdown_output = []
markdown_output.append("# TabiBench Comparison Report\n")
markdown_output.append("\n## Detailed Results\n")
markdown_output.append(pivot_df.to_markdown(index=False, floatfmt=".2f"))

# Averages
markdown_output.append("\n## Average Scores by Type\n")
avg_df = df.groupby(["Type", "Model"])["Score"].mean().unstack()
markdown_output.append(avg_df.to_markdown(floatfmt=".2f"))

final_md = "\n".join(markdown_output)

print(final_md)

with open("tabibench_comparison.md", "w") as f:
    f.write(final_md)
print(f"\nReport saved to {os.path.abspath('tabibench_comparison.md')}")
