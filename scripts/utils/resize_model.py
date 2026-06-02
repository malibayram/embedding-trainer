import torch
from transformers import AutoModel, AutoConfig

model_path = "/Users/alibayram/.cache/huggingface/hub/models--alibayram--embeddingmagibu-200m/snapshots/a2998e7a11b2e099f6fce8247a6227d1019031bd"
save_path = "resized_embeddingmagibu_200m"

print(f"Loading model from {model_path}...")
# Load config first to check current status
config = AutoConfig.from_pretrained(model_path)
print(f"Original vocab size in config: {config.vocab_size}")

# Load model
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True)

current_embeddings = model.get_input_embeddings()
print(f"Current embedding weight shape: {current_embeddings.weight.shape}")

target_vocab_size = 131073

if current_embeddings.weight.shape[0] != target_vocab_size:
    print(f"Resizing embeddings to {target_vocab_size}...")
    model.resize_token_embeddings(target_vocab_size)
    print(f"New embedding weight shape: {model.get_input_embeddings().weight.shape}")
else:
    print("Embeddings already have the correct size.")

print(f"Saving resized model to {save_path}...")
model.save_pretrained(save_path)
# Also save the tokenizer to the new directory so we have a complete model folder
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(save_path)
    print("Tokenizer saved.")
except Exception as e:
    print(f"Warning: Could not save tokenizer: {e}")

print("Done.")
