#!/usr/bin/env python3
"""Download GloVe Wiki-Gigaword 300-d embeddings and save as plain-text
`data/embeddings/glove.6B.300d.txt` so the R lexicon builder can use it.
Run:  python 00_download_embeddings.py
"""
import os, sys

try:
    import gensim.downloader as api
except ImportError:
    sys.stderr.write("gensim not installed.  Run `pip install gensim==4.3.2` and retry.\n")
    sys.exit(1)

OUT_DIR = os.path.join("data", "embeddings")
OUT_PATH = os.path.join(OUT_DIR, "glove.6B.300d.txt")

os.makedirs(OUT_DIR, exist_ok=True)

print("⏳ downloading 'glove-wiki-gigaword-300' (~374 MB)…")
model = api.load("glove-wiki-gigaword-300")

print("💾 writing plain-text vectors to", OUT_PATH)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for word in model.index_to_key:
        vec = model.get_vector(word)
        f.write(word + " " + " ".join(map(str, vec)) + "\n")

print("✅ Done.  File ready for R scripts.") 