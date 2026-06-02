<p align="center">
  <img src="assets/logo.png" alt="embeddingmagibu logo" width="180">
</p>

<h1 align="center">embeddingmagibu-200m</h1>

<p align="center">
  <strong>Turkish-Focused Sentence Embedding via Cross-Lingual Tokenizer Surgery & Offline Distillation</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.29992"><img src="https://img.shields.io/badge/arXiv-2605.29992-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/magibu/embeddingmagibu-200m"><img src="https://img.shields.io/badge/🤗_Model-embeddingmagibu--200m-yellow.svg" alt="HuggingFace Model"></a>
  <a href="https://ollama.com/alibayram/embeddingmagibu-200m"><img src="https://img.shields.io/badge/🦙_Ollama-embeddingmagibu--200m-white.svg" alt="Ollama"></a>
  <a href="https://huggingface.co/spaces/magibu/mteb-turkish"><img src="https://img.shields.io/badge/🏆_TR--MTEB-Leaderboard-blue.svg" alt="TR-MTEB Explorer"></a>
  <a href="https://pypi.org/project/transformer-cloner/"><img src="https://img.shields.io/badge/📦_PyPI-transformer--cloner-green.svg" alt="transformer-cloner"></a>
  <a href="https://pypi.org/project/distil-trainer/"><img src="https://img.shields.io/badge/📦_PyPI-distil--trainer-green.svg" alt="distil-trainer"></a>
  <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg" alt="License"></a>
</p>

---

This repository contains the **training, evaluation, and deployment pipeline** for **embeddingmagibu-200m** — a Turkish-optimized sentence embedding model that produces **768-dimensional ℓ₂-normalized vectors** with an **8,192-token context window**. The model achieves competitive results on Turkish benchmarks while using **33% fewer parameters** than its teacher and training in **~4 hours on a single GPU** at a cost of **$5–$20**.

> **📄 Paper:** [Adapting Multilingual Embedding Models to Turkish via Cross-Lingual Tokenizer Surgery and Offline Distillation](https://arxiv.org/abs/2605.29992)
>
> **Authors:** M. Ali Bayram, Banu Diri, Savaş Yıldırım — Yıldız Technical University & Istanbul Bilgi University

---

## ✨ Key Highlights

| Feature                 | Value                                           |
| ----------------------- | ----------------------------------------------- |
| **Embedding Dimension** | 768                                             |
| **Max Sequence Length** | 8,192 tokens                                    |
| **Parameters**          | ~200M (vs. 300M teacher)                        |
| **Vocabulary**          | 2¹⁷ = 131,072 tokens (Turkish-optimized hybrid) |
| **Training Cost**       | $5–$20, ~4 GPU hours                            |
| **STSbTR Pearson**      | **77.55%** (teacher: 73.84%)                    |
| **TR-MTEB Rank**        | **7th / 26** models                             |
| **Format**              | SentenceTransformers / GGUF (Ollama)            |

---

## 🏗️ Pipeline Overview

The model is built through an efficient **three-stage adaptation pipeline** that avoids full pretraining:

```
┌─────────────────────┐     ┌─────────────────────────┐     ┌───────────────────────────┐
│  1. TOKENIZER       │     │  2. MODEL CLONING       │     │  3. OFFLINE DISTILLATION  │
│     CONSTRUCTION    │────▶│     (Weight-Preserving)  │────▶│     (Cosine Similarity)   │
│                     │     │                         │     │                           │
│ Cosmos Corpus       │     │ EmbeddingGemma-300M     │     │ 580K Wikipedia examples   │
│ + Wikipedia-40-langs│     │ → Backbone preserved    │     │ 40 languages, balanced    │
│ → 128K hybrid vocab │     │ → Embedding remapped    │     │ Teacher inference offline  │
└─────────────────────┘     └─────────────────────────┘     └───────────────────────────┘
```

### Stage 1: Turkish-Optimized Tokenizer

A hybrid 128K-token vocabulary is constructed by:

1. Extracting the top **64K most frequent Turkish tokens** from a tokenizer trained on the [Cosmos Turkish Corpus](https://huggingface.co/datasets/ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0)
2. **Pruning redundant tokens** from the teacher's 256K vocabulary
3. **Adding multilingual tokens** via frequency analysis on the [Wikipedia-40-langs](https://huggingface.co/datasets/alibayram/wikipedia-40-langs) dataset

This reduces the embedding table from ~196M to ~100M parameters while dramatically improving Turkish morphological alignment.

### Stage 2: Weight-Preserving Model Cloning

The teacher model ([EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m)) is cloned via **mean-composition token mapping**: for each new token, the surface form is tokenized by the teacher, and the new embedding is initialized as the **mean of the corresponding teacher embeddings**. All transformer backbone weights (attention, feedforward, layer norm) are preserved exactly.

### Stage 3: Offline Embedding Distillation

Teacher embeddings are **precomputed once** and stored, enabling the student to train without running the teacher at each step. The student minimizes a **cosine similarity loss** against the precomputed teacher vectors over a balanced 40-language Wikipedia corpus (~580K examples).

---

## 📊 Results

### STSbTR — Semantic Textual Similarity (Turkish)

|  Rank | Model                                   |   Pearson |  Spearman |
| ----: | --------------------------------------- | --------: | --------: |
|     1 | emrecan/bert-base-tr-nli-stsb-tr        |     83.45 |     83.13 |
|     2 | intfloat/multilingual-e5-large-instruct |     80.50 |     81.23 |
|     3 | ytu-ce-cosmos/turkish-e5-large          |     79.73 |     79.99 |
| **6** | **embeddingmagibu-200m (Ours)**         | **77.55** | **77.45** |
|    11 | embeddinggemma-300m (Teacher)           |     73.84 |     72.92 |
|    12 | embeddingmagibu-152m                    |     72.92 |     71.84 |

> **+3.71% absolute improvement** over the 300M-parameter teacher on test Pearson.

### TR-MTEB — Multi-Task Embedding Benchmark (26 tasks)

| Model                           |     Rank |      Avg | Retrieval | Classification | Clustering |      STS |      NLI | Bitext |
| ------------------------------- | -------: | -------: | --------: | -------------: | ---------: | -------: | -------: | -----: |
| text-embedding-3-small          |     1/26 |     66.5 |      78.1 |           69.5 |       62.1 |     70.8 |     57.2 |   91.6 |
| ytu-ce-cosmos/turkish-e5-large  |     2/26 |     66.0 |      77.0 |           72.6 |       60.7 |     80.0 |     62.7 |   99.2 |
| embeddinggemma-300m (Teacher)   |     4/26 |     65.2 |      75.9 |           71.8 |       62.4 |     72.9 |     60.6 |   96.8 |
| **embeddingmagibu-200m (Ours)** | **7/26** | **63.9** |      72.2 |           68.5 |       61.4 | **77.5** | **67.9** |   97.0 |
| embeddingmagibu-152m            |    12/26 |     60.2 |      71.3 |           65.5 |       61.6 |     71.8 |     55.6 |   90.1 |

> Achieves **98% of teacher performance** with **33% fewer parameters**. **Outperforms the teacher** on STS (+4.6%), NLI (+7.3%), and Bitext Mining (+0.2%).

### Vocabulary Size Ablation (64K → 128K)

| Category        | 200m (128K) | 152m (64K) |         Δ |
| --------------- | ----------: | ---------: | --------: |
| Overall Average |        63.9 |       60.2 |  **+3.7** |
| NLI             |        67.9 |       55.6 | **+12.3** |
| Bitext Mining   |        97.0 |       90.1 |  **+6.9** |
| STS             |        77.5 |       71.8 |  **+5.7** |

---

## 🚀 Quick Start

### Using with SentenceTransformers

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("magibu/embeddingmagibu-200m")

sentences = [
    "İstanbul dünyanın en güzel şehirlerinden biridir.",
    "Boğaziçi Köprüsü İstanbul'un simgesidir.",
    "Bugün hava çok güneşli ve sıcak.",
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, 768)
```

### Using with Ollama (Local Deployment)

```bash
ollama pull alibayram/embeddingmagibu-200m
```

```python
import ollama

response = ollama.embed(
    model="alibayram/embeddingmagibu-200m",
    input=["Türkiye'nin başkenti Ankara'dır."]
)
print(len(response.embeddings[0]))  # 768
```

### Using with Hugging Face Transformers

```python
from transformers import AutoModel, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("magibu/embeddingmagibu-200m")
model = AutoModel.from_pretrained("magibu/embeddingmagibu-200m")

inputs = tokenizer("Merhaba dünya!", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
```

---

## 📁 Repository Structure

```
embedding-trainer/
│
├── assets/                              # Static assets
│   └── logo.png                         # Project logo
│
├── scripts/
│   ├── training/                        # 🎯 Training scripts
│   │   ├── train.py                     #    Main distillation (uses distil-trainer)
│   │   ├── train_embeddinggemma.py      #    Embedding-only training loop
│   │   ├── train_magibu_cosmos.py       #    Cosmos corpus training
│   │   ├── train_magibu_multi.py        #    Multilingual training (embeddings + dense)
│   │   ├── train_sft_clean.py           #    Multi-task SFT (retrieval, STS, NLI, cls)
│   │   └── train_sft_multi_task.py      #    Extended multi-task SFT
│   │
│   ├── evaluation/                      # 📊 Evaluation & benchmarking
│   │   ├── evaluate_sts_tr.py           #    STSbTR benchmark
│   │   ├── evaluate_sts_en.py           #    English STS benchmark
│   │   ├── evaluate_mteb_tr.py          #    TR-MTEB (26 tasks)
│   │   ├── evaluate_tabibench.py        #    TabiBench evaluation
│   │   ├── compare_results.py           #    Cross-model result comparison
│   │   ├── summarize_tabibench.py       #    TabiBench summary generator
│   │   ├── parse_mteb_results.py        #    Parse MTEB JSON results
│   │   └── fetch_results.py             #    Fetch results from remote servers
│   │
│   ├── data/                            # 📚 Dataset preparation
│   │   ├── prepare_wiki_dataset.py      #    Build balanced 40-lang Wikipedia corpus
│   │   ├── prepare_wiki_dataset_parallel.py  # Parallel version
│   │   ├── generate_wiki_embeddings.py  #    Generate teacher embeddings (Ollama)
│   │   ├── generate_wiki_embeddings_fast.py  # Fast embedding generation
│   │   └── upload_wiki_dataset.py       #    Upload datasets to HF Hub
│   │
│   ├── deployment/                      # 🚢 Deployment & serving
│   │   ├── deploy.py                    #    Remote server deployment
│   │   ├── setup_server.sh              #    A100 server setup
│   │   ├── verify_ollama.py             #    Verify Ollama deployment
│   │   ├── Modelfile                    #    Ollama model definition
│   │   └── GGUF_CONVERSION.md           #    HF → GGUF → Ollama guide
│   │
│   └── utils/                           # 🔧 Utility scripts
│       ├── clone_embeddinggemma.py      #    Weight-preserving model cloning
│       ├── resize_model.py              #    Resize embeddings for GGUF
│       ├── patch_transformers_local.py  #    Transformers compatibility patch
│       ├── check_nli.py                 #    NLI dataset verification
│       └── check_remote.py              #    Remote model verification
│
├── results/                             # 📈 Benchmark results & logs
│   ├── sts_benchmark_results.json       #    Collected STS benchmark results
│   ├── tabibench_comparison.md          #    TabiBench comparison report
│   └── sft_training.log                 #    SFT training log
│
├── notebooks/                           # 🔬 Experimentation
│   └── playground.ipynb                 #    Interactive notebook
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔧 Reproducing the Pipeline

### Prerequisites

```bash
# System setup (for GPU training)
bash scripts/deployment/setup_server.sh

# Or install manually:
pip install -U sentence-transformers datasets sentencepiece transformers
pip install -U transformer-cloner distil-trainer
pip install wandb tqdm python-dotenv
```

**Hardware:** Single NVIDIA A100 80GB (or equivalent with ≥40GB VRAM)

### Step 1: Prepare the Multilingual Wikipedia Dataset

```bash
python scripts/data/prepare_wiki_dataset.py
```

Creates a balanced 40-language corpus:

- **Turkish & English:** 100K training examples each
- **38 other languages:** 10K training examples each
- **Total:** ~580K training rows

### Step 2: Clone the Teacher Model

```bash
python scripts/utils/clone_embeddinggemma.py
```

This clones EmbeddingGemma-300M with the Turkish-optimized 128K tokenizer, preserving backbone weights and remapping embeddings via mean-composition.

### Step 3: Generate Teacher Embeddings

```bash
python scripts/data/generate_wiki_embeddings.py
```

Precomputes teacher embeddings using EmbeddingGemma-300M, storing both final and pre-dense representations.

### Step 4: Train via Offline Distillation

```bash
python scripts/training/train.py
```

Or using the `distil-trainer` package directly:

```python
from distil_trainer import EmbeddingDistillationTrainer, EmbeddingTrainerConfig

config = EmbeddingTrainerConfig(
    student_model="alibayram/magibu-200m",
    target_type="final",
    num_epochs=1,
    batch_size=256,
    learning_rate=5e-5,
    warmup_ratio=0.01,
    weight_decay=0.01,
    max_grad_norm=1.0,
    loss_type="cosine",
    use_bf16=True,
    gradient_checkpointing=True,
    compile_model=True,
)

trainer = EmbeddingDistillationTrainer(config)
metrics = trainer.train("alibayram/wikipedia-40-langs-with-embeddings")
```

### Step 5: Evaluate

```bash
# STSbTR
python scripts/evaluation/evaluate_sts_tr.py --model "magibu/embeddingmagibu-200m"

# TR-MTEB (all 26 tasks)
python scripts/evaluation/evaluate_mteb_tr.py "magibu/embeddingmagibu-200m" --all-tasks

# Compare multiple models
python scripts/evaluation/evaluate_sts_tr.py --model "magibu/embeddingmagibu-200m" \
                                                     "google/embeddinggemma-300m" \
                                                     "intfloat/multilingual-e5-base"
```

### Step 6: Deploy to Ollama (Optional)

See [GGUF_CONVERSION.md](scripts/deployment/GGUF_CONVERSION.md) for full instructions.

```bash
# Convert to GGUF
python3 llama.cpp/convert_hf_to_gguf.py ./model --outfile model-bf16.gguf --outtype bf16

# Create and push to Ollama
ollama create alibayram/embeddingmagibu-200m -f scripts/deployment/Modelfile
ollama push alibayram/embeddingmagibu-200m
```

---

## ⚙️ Training Hyperparameters

| Hyperparameter         | Value             |
| ---------------------- | ----------------- |
| Epochs                 | 1                 |
| Batch Size             | 256               |
| Learning Rate          | 5 × 10⁻⁵          |
| Warmup Ratio           | 0.01              |
| Weight Decay           | 0.01              |
| Max Gradient Norm      | 1.0               |
| Precision              | bf16              |
| Gradient Checkpointing | ✅                |
| torch.compile          | ✅                |
| Loss Function          | Cosine Similarity |
| Target Type            | Final embeddings  |

---

## 📦 Released Artifacts

| Artifact                 | Link                                                                                                                  |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| **Model Weights**        | [🤗 magibu/embeddingmagibu-200m](https://huggingface.co/magibu/embeddingmagibu-200m)                                  |
| **Ollama Model**         | [🦙 alibayram/embeddingmagibu-200m](https://ollama.com/alibayram/embeddingmagibu-200m)                                |
| **TR-MTEB Explorer**     | [🏆 Interactive Leaderboard](https://huggingface.co/spaces/magibu/mteb-turkish)                                       |
| **Distillation Dataset** | [📊 wikipedia-40-langs-with-embeddings](https://huggingface.co/datasets/alibayram/wikipedia-40-langs-with-embeddings) |
| **Wikipedia Corpus**     | [📚 wikipedia-40-langs](https://huggingface.co/datasets/alibayram/wikipedia-40-langs)                                 |
| **Cloning Tool**         | [📦 transformer-cloner (PyPI)](https://pypi.org/project/transformer-cloner/)                                          |
| **Distillation Tool**    | [📦 distil-trainer (PyPI)](https://pypi.org/project/distil-trainer/)                                                  |
| **Training Logs**        | [📈 Weights & Biases](https://api.wandb.ai/links/alibayram-ytu/srxzzhof)                                              |
| **Predecessor Model**    | [🤗 magibu/embeddingmagibu-152m](https://huggingface.co/magibu/embeddingmagibu-152m)                                  |

---

## 🔬 Parameter Footprint

| Model                           |    Vocab | Embedding Params |   Backbone |     Total |
| ------------------------------- | -------: | ---------------: | ---------: | --------: |
| multilingual-e5-large-instruct  |     250K |           256.0M |     304.0M |    560.0M |
| embeddinggemma-300m (Teacher)   |     256K |           196.6M |     104.0M |    300.6M |
| **embeddingmagibu-200m (Ours)** | **128K** |       **100.6M** | **104.0M** | **~205M** |
| embeddingmagibu-152m            |      64K |            49.5M |     104.0M |     ~154M |

> Trimming the vocabulary from 256K → 128K reduces embedding parameters by **48.8%**.

---

## 🌍 Supported Languages

The model is **optimized for Turkish** but retains multilingual capability across **40 languages** through the balanced distillation corpus:

<details>
<summary>Click to expand full language list</summary>

Turkish (tr), English (en), German (de), French (fr), Spanish (es), Italian (it), Portuguese (pt), Dutch (nl), Polish (pl), Czech (cs), Romanian (ro), Hungarian (hu), Finnish (fi), Danish (da), Norwegian (no), Bulgarian (bg), Greek (el), Russian (ru), Arabic (ar), Persian (fa), Hebrew (he), Japanese (ja), Chinese (zh), Korean (ko), Vietnamese (vi), Indonesian (id), Malay (ms), Catalan (ca), Basque (eu), Welsh (cy), Armenian (hy), Uzbek (uz), Tatar (tt), Chechen (ce), Cebuano (ceb), Waray (war), Egyptian Arabic (arz), Serbian (sh), Esperanto (eo), Simple English (simple)

</details>

---

## 📝 Citation

```bibtex
@article{bayram2026adapting,
  title={Adapting Multilingual Embedding Models to Turkish via Cross-Lingual Tokenizer Surgery and Offline Distillation},
  author={Bayram, M. Ali and Diri, Banu and Y{\i}ld{\i}r{\i}m, Sava\c{s}},
  journal={arXiv preprint arXiv:2605.29992},
  year={2026}
}
```

---

## 📄 License

This work is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

---

<p align="center">
  <sub>Built with ❤️ at Yıldız Technical University & Istanbul Bilgi University</sub>
</p>
