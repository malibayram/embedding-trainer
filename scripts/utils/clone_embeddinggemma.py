import torch
from sentence_transformers import SentenceTransformer
from transformers import Gemma3TextModel, AutoTokenizer

embeddinggemma = SentenceTransformer("google/embeddinggemma-300m")
embeddinggemma.save_pretrained("magibu-200m-base")

model = Gemma3TextModel.from_pretrained("google/embeddinggemma-300m")

embeddings = model.embed_tokens.weight
print(f"Embeddings shape: {embeddings.shape}")

org_tokenizer = AutoTokenizer.from_pretrained("google/embeddinggemma-300m")
target_tokenizer = AutoTokenizer.from_pretrained("alibayram/magibu-128k-processor")
print(f"Tokenizer vocab size: {org_tokenizer.vocab_size}, {target_tokenizer.vocab_size}")

source_vocab = org_tokenizer.get_vocab()
target_vocab = target_tokenizer.get_vocab()

# Get UNK token embedding as fallback
unk_token_id = org_tokenizer.unk_token_id if org_tokenizer.unk_token_id is not None else 0
unk_embedding = embeddings[unk_token_id].clone()
print(f"UNK token ID: {unk_token_id}")

token_id_map = {}
direct_matches = 0
tokenized_matches = 0

for token_str, target_id in target_vocab.items():
    if token_str in source_vocab:
        # Direct match - use source token ID directly
        token_id_map[target_id] = [source_vocab[token_str]]
        direct_matches += 1
    else:
        # Token not in source vocab - need to tokenize
        encoded = org_tokenizer.encode(token_str, add_special_tokens=False)
        if encoded:
            token_id_map[target_id] = encoded
            tokenized_matches += 1

print(f"Direct matches: {direct_matches}, Tokenized matches: {tokenized_matches}")

# Initialize all embeddings with UNK embedding (safe fallback)
new_embeddings = unk_embedding.unsqueeze(0).repeat(target_tokenizer.vocab_size, 1).clone()
print(f"New embeddings shape: {new_embeddings.shape}")

errors = []
for i in range(target_tokenizer.vocab_size):
    if i not in token_id_map or not token_id_map[i]:
        errors.append(i)
        # Keep UNK embedding for unmapped tokens
        continue

    source_ids = token_id_map[i]
    # MEAN token strategy: average the source token embeddings
    # remove <bos> if present
    if source_ids and source_ids[0] == org_tokenizer.bos_token_id:
        source_ids = source_ids[1:]
    
    # If source_ids is empty after removing bos, use UNK
    if not source_ids:
        errors.append(i)
        continue
        
    embeddings_to_average = embeddings[source_ids]
    new_embeddings[i] = embeddings_to_average.mean(dim=0)

    if (i + 1) % 10000 == 0:
        print(f"Mapped {i + 1}/{target_tokenizer.vocab_size} embeddings", flush=True)

model.resize_token_embeddings(new_embeddings.shape[0])
print(model)
model.embed_tokens.weight = torch.nn.Parameter(new_embeddings)
print(f"Completed embedding transfer with {len(errors)} errors.")

model = model.to(torch.bfloat16)

parameter_count = sum(p.numel() for p in model.parameters())
print(f"Model parameter count: {parameter_count / 1_000_000:.2f}M")

model.save_pretrained("magibu-200m-base")
target_tokenizer.save_pretrained("magibu-200m-base")

embeddingmagibu = SentenceTransformer("magibu-200m-base")

embeddingmagibu.push_to_hub("alibayram/magibu-200m-base", exist_ok=True)
