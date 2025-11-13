import json
import numpy as np
import faiss
import pickle
from openai import OpenAI

client = OpenAI()

EMBED_MODEL = "text-embedding-3-large"
INPUT_FILE = "few_shot_examples.jsonl"

embeddings = []
metadata = []

# Step 1: Load all examples
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        metadata.append(obj)

        emb = client.embeddings.create(
            model=EMBED_MODEL,
            input=obj["query"]
        ).data[0].embedding
        
        embeddings.append(emb)

embeddings = np.array(embeddings).astype("float32")

# Step 2: Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Step 3: Save index and metadata
faiss.write_index(index, "db/embeddings.faiss")
with open("db/example_meta.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("Built FAISS index and metadata!")