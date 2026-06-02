import os
import json
import glob

results_dir = "mteb_full_results/84fa4ec89fd2fcb5e498c91376ff42ec5fe0366e"
json_files = glob.glob(os.path.join(results_dir, "*.json"))

results_list = []

for file_path in json_files:
    if "model_meta.json" in file_path:
        continue
        
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        task_name = data.get('task_name', os.path.basename(file_path).replace('.json', ''))
        
        # Try to find main score
        score = 0
        if 'scores' in data and 'test' in data['scores'] and len(data['scores']['test']) > 0:
            test_res = data['scores']['test'][0]
            # Try specific keys if main_score not present
            if 'main_score' in test_res:
                score = test_res['main_score']
            elif 'map' in test_res:
                score = test_res['map']
            elif 'accuracy' in test_res:
                score = test_res['accuracy']
            elif 'spearman' in test_res:
                score = test_res['spearman']
            elif 'cos_sim' in test_res and 'spearman' in test_res['cos_sim']:
                 score = test_res['cos_sim']['spearman']
        
        results_list.append((task_name, score))
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")

# Sort by task name
results_list.sort(key=lambda x: x[0])

print(f"{'Task Name':<40} | {'Score':<10}")
print("-" * 55)
for task, score in results_list:
    print(f"{task:<40} | {score:>8.4f}")
print("-" * 55)

if results_list:
    avg = sum(s for _, s in results_list) / len(results_list)
    print(f"{'AVERAGE':<40} | {avg:>8.4f}")
