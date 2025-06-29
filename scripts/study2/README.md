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