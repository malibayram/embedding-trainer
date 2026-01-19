import ollama
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

s_sentences = [
    "Bu mutlu bir adamdır",
    "Bu mutlu bir kadın",
    "Akşam sinemaya gideceğim",
    "Bugün hava güneşli"
]

print("Requesting embeddings from Ollama...")
try:
    batch = ollama.embed(model="alibayram/embeddingmagibu-200m", input=s_sentences)
    
    embeddings_matrix = np.array(batch.embeddings)
    print(f"Embeddings shape: {embeddings_matrix.shape}")
    
    similarity_matrix = cosine_similarity(embeddings_matrix)
    print("Similarity Matrix:")
    print(similarity_matrix)
    
except Exception as e:
    print(f"Error: {e}")
