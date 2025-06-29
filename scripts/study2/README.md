# Study 2 Scripts

All code for the qualitative/text‐analysis portion of the project lives here.

```
scripts/
└── study2/
    ├── r/          # Primary R pipeline (tidytext + tidymodels)
    └── python/     # Optional helpers (embedding expansion, fast similarity lookups)
```

Planned numbered scripts (R):
1. `r/00_build_lexicon.R` – creates seed and expanded certainty/uncertainty lexicons.
2. `r/01_score_utterances.R` – tokenises transcripts, applies lexicon, outputs certainty scores.
3. `r/02_exploratory_lda.R` *(optional)* – topic modelling for triangulation.

Python helpers (under `python/`) will mirror the numbering when needed (e.g., `00_embedding_expand.py`).

Each script writes its outputs to `results/` or subfolders thereof so the manuscript chunks can source clean artefacts without rerunning heavy computation during compilation.

### Embedding download (one-time)

`00_build_lexicon.R` will auto-expand the seed dictionary **only if** the file
`data/embeddings/glove.6B.300d.txt` exists.  Use the helper:

```bash
# install dependency once
pip install gensim==4.3.2

# download and convert the vectors (~5–10 min)
python scripts/study2/python/00_download_embeddings.py
```

Then rerun `00_build_lexicon.R` and `01_score_utterances.R`. 