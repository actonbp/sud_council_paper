#!/usr/bin/env python3
"""Download FastText Wiki-News (subword) 300-d embeddings via gensim and
write them as plain-text so the R lexicon builder can consume them.

Usage:
    python 01_download_fasttext.py

Output:
    data/embeddings/fasttext_wiki_news_300d.txt  (~5 GB)
"""
import os, sys, gzip

try:
    import gensim.downloader as api
except ImportError:
    sys.stderr.write("gensim not installed. Run `pip install gensim==4.3.2` and retry.\n")
    sys.exit(1)

OUT_DIR = os.path.join("data", "embeddings")
OUT_PATH = os.path.join(OUT_DIR, "fasttext_wiki_news_300d.txt")

os.makedirs(OUT_DIR, exist_ok=True)

print("⏳ downloading 'fasttext-wiki-news-subwords-300' (~5 GB)…")
model = api.load("fasttext-wiki-news-subwords-300")

print("💾 writing plain-text vectors to", OUT_PATH)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for word in model.index_to_key:
        vec = model.get_vector(word)
        f.write(word + " " + " ".join(map(str, vec)) + "\n")

print("✅ Done.  File ready for R scripts.") 