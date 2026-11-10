"""Embed every lemma once with a frozen sentence encoder and cache it.

Run with:  uv run --with sentence-transformers python embed_lemmas.py
The speaker gets word meaning from these vectors instead of learning a
lookup table, so a word seen only in one role still carries its meaning
into new roles, and new words can be written later.
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
lemmas = json.loads((HERE / "data" / "lemmas.json").read_text())
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
texts = ["something" if l == "<arg>" else l for l in lemmas]
emb = model.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
np.save(HERE / "data" / "lemma_emb.npy", emb.astype(np.float32))
print(emb.shape)
