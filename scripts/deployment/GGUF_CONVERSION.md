# Guide: Converting HF Models to GGUF & Pushing to Ollama

This guide outlines the process of converting a Hugging Face model (specifically Gemma 3 based embedding models) to GGUF format and pushing it to the Ollama registry.

## Prerequisites

1.  **llama.cpp**: Clone and build the repository.
    ```bash
    git clone https://github.com/ggml-org/llama.cpp
    cd llama.cpp
    pip install -r requirements.txt
    ```
2.  **Ollama**: Installed and running.
3.  **Python Environment**: With `transformers`, `torch`, and `huggingface_hub` installed.

## Step 1: Prepare the Model

Locate your model. If it's a SentenceTransformer or Gemma 3 model, ensure the `vocab_size` in `config.json` matches the actual tensor dimensions.

### Troubleshooting: Vocab Size Mismatch

If you get a dimensionality error (e.g., `tensor 'token_embd.weight' has wrong shape`), you may need to resize the model's embedding layer to match the tokenizer's vocabulary size (often off-by-one, e.g., 131072 vs 131073).

**Script to resize embeddings:**

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_id = "your-model-id"
save_path = "resized_model"
target_vocab_size = 131073  # Check tokenizer.json for max id

# Load with ignore_mismatched_sizes in case it's already weird
model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True)
model.resize_token_embeddings(target_vocab_size)

model.save_pretrained(save_path)
print(f"Saved resized model to {save_path}")
```

**Important**: Ensure `modules.json` and other config files are present in the `resized_model` directory. If missing, copy them from the original snapshot.

## Step 2: Convert to GGUF

Use the `convert_hf_to_gguf.py` script from `llama.cpp`.

```bash
python3 /path/to/llama.cpp/convert_hf_to_gguf.py \
    /path/to/resized_model \
    --outfile model-bf16.gguf \
    --outtype bf16
```

### Troubleshooting: Unrecognized Tokenizer

If you get `BPE pre-tokenizer was not recognized`, you might need to patch `convert_hf_to_gguf.py` to add your model's tokenizer hash.

1.  Run the script and copy the `chkhsh` value from the error message.
2.  Edit `convert_hf_to_gguf.py` (around line ~1200, inside `get_vocab_base_pre`).
3.  Add your hash mapping:
    ```python
    if chkhsh == "YOUR_HASH_HERE":
        res = "default"
    ```

## Step 3: Create Ollama Model

1.  **Create a Modelfile**:

    ```dockerfile
    FROM ./model-bf16.gguf
    # Optional: Set parameters
    # PARAMETER temperature 0.7
    ```

2.  **Create Model**:
    ```bash
    ollama create username/modelname -f Modelfile
    ```

## Step 4: Verify Locally

Test the model before pushing.

```python
import ollama

response = ollama.embed(model="username/modelname", input=["Hello world"])
print(len(response['embeddings'][0])) # Should match hidden size (e.g., 768)
```

## Step 5: Push to Ollama

Once verified, push to the registry.

```bash
ollama push username/modelname
```
