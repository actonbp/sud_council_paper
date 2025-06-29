#!/usr/bin/env python
"""
reproduce_LDA_k5.py  –  Exhaustive reproducibility script

Put this file in the same folder as your six CSVs (or edit DATA_DIR / CSV_GLOB).
Run:  python reproduce_LDA_k5.py
"""

import glob, os, hashlib, re, textwrap, sys, platform
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# -------------------------------------------------------------------
# 0.  Environment check
# -------------------------------------------------------------------
print("Python:", sys.version.split()[0], "| Platform:", platform.platform())
import sklearn, numpy
print("scikit-learn:", sklearn.__version__, "| NumPy:", numpy.__version__)
print("-" * 60)

# -------------------------------------------------------------------
# 1.  Locate CSV files
# -------------------------------------------------------------------
DATA_DIR = "."                                    # edit if needed
CSV_GLOB = "*_Focus_Group*_full*.csv"             # strict pattern
paths = sorted(glob.glob(os.path.join(DATA_DIR, CSV_GLOB)))
if not paths:
    sys.exit(f"No files found via pattern {CSV_GLOB} in {DATA_DIR}")

print("Files detected (name | MD5 | raw rows):")
file_rows = {}
for p in paths:
    raw_rows = len(pd.read_csv(p))
    md5 = hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
    print(f"- {os.path.basename(p):40}  {md5}   {raw_rows}")
    file_rows[os.path.basename(p)] = raw_rows
print("-" * 60)

# -------------------------------------------------------------------
# 2.  Load, drop moderator lines
# -------------------------------------------------------------------

df_list = []
for p in paths:
    df = pd.read_csv(p)
    df = df[df["Text"].notna()]
    is_mod = df["Speaker"].astype(str).str.match(r"^[A-Z]{2,3}$")
    df = df[~is_mod].copy()
    df["session"] = os.path.basename(p)
    df_list.append(df)

corpus = pd.concat(df_list, ignore_index=True)
print(f"Rows after moderator removal: {len(corpus)}")

# -------------------------------------------------------------------
# 3.  Clean text (strip SUD terms, lower-case)
# -------------------------------------------------------------------
DOMAIN = re.compile(
    r"\b(substance|abuse|addict\w*|drug\w*|alcohol\w*|counsel\w*|mental\s+health)\b",
    re.I,
)
corpus["clean"] = corpus["Text"].str.lower().apply(lambda t: DOMAIN.sub("", t))

# Custom stop-word list (basic English + fillers)
STOP = set("""
a about above after again against all am an and any are as at be because been before being
between both but by could did do does doing down during each few for from further had has
have having he her here hers herself him himself his how i if in into is it its itself
just like me more most my myself nor not of off on once only or other our ours ourselves
out over own same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were what when where
which while who whom why will with you your yours yourself yourselves um uh yeah okay kinda
sorta right would know think really kind going lot can say definitely want guess something
able way actually maybe feel feels felt
""".split())

# -------------------------------------------------------------------
# 4.  Vectorise (unigram+bigram, min_df = 3)
# -------------------------------------------------------------------
vec = CountVectorizer(stop_words=list(STOP),
                      ngram_range=(1, 2),
                      min_df=3)

X = vec.fit_transform(corpus["clean"])
print(f"Docs kept after vectorisation: {X.shape[0]}")
print("-" * 60)

# -------------------------------------------------------------------
# 5.  Fit LDA (k = 5, deterministic)
# -------------------------------------------------------------------

a = 5
lda = LatentDirichletAllocation(
    n_components=a,
    random_state=42,
    learning_method="batch",
    max_iter=500,
)
doc_topic = lda.fit_transform(X)
vocab = vec.get_feature_names_out()

# -------------------------------------------------------------------
# 6.  Top terms per topic & counts
# -------------------------------------------------------------------
TOP_N = 12
print("\nTop-12 terms per topic:")
for t in range(a):
    idx = lda.components_[t].argsort()[-TOP_N:][::-1]
    print(f"Topic {t+1}: {', '.join(vocab[i] for i in idx)}")

dominant = doc_topic.argmax(axis=1)
counts = pd.Series(dominant).value_counts().sort_index()
print("\nUtterance count per dominant topic:")
for t, n in counts.items():
    print(f"Topic {t+1}: {n}")
print("-" * 60)

# -------------------------------------------------------------------
# 7.  Representative quote per topic (optional)
# -------------------------------------------------------------------
SHOW_QUOTES = True
if SHOW_QUOTES:
    print("\nRepresentative utterance for each topic:")
    for t in range(a):
        probs = doc_topic[:, t]
        best = probs.argmax()
        utter = textwrap.fill(corpus.iloc[best]["Text"], 110)
        print(f"\nTopic {t+1}  (P={probs[best]:.3f})  —  Session {corpus.iloc[best]['session']}")
        print("→", utter) 