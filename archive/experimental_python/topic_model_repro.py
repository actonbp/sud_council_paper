# topic_model_repro.py
# Reproduces the k = 5 LDA model & utterance counts

import glob, os, re, textwrap
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# -------------------------------------------------------------
# 1.  Load all focus-group CSVs
# -------------------------------------------------------------
DATA_DIR = "data"                                   # <-- adjust
paths = glob.glob(os.path.join(DATA_DIR, "*_Focus_[Gg]roup*full*.csv"))
assert paths, "No CSVs found: check DATA_DIR"

dfs = []
for p in paths:
    df = pd.read_csv(p)
    # keep only rows with non-empty Text
    df = df[df["Text"].notna()]
    # remove moderator lines (2–3 uppercase letters)
    df = df[~df["Speaker"].astype(str).str.match(r"^[A-Z]{2,3}$")]
    df["session"] = os.path.basename(p)
    dfs.append(df)

corpus_df = pd.concat(dfs, ignore_index=True)
print("Rows after moderator removal:", len(corpus_df))

# -------------------------------------------------------------
# 2.  Cleaning
# -------------------------------------------------------------
DOMAIN_RE = re.compile(
    r"\b(substance|abuse|addict\w*|drug\w*|alcohol\w*|counsel\w*|mental\s+health)\b",
    re.I)

def clean(txt: str) -> str:
    txt = DOMAIN_RE.sub("", txt.lower())
    return txt

corpus_df["clean"] = corpus_df["Text"].astype(str).apply(clean)

# -------------------------------------------------------------
# 3.  Build DTM (unigram + bigram, min_df = 3)
#     Rows that become empty are dropped automatically.
# -------------------------------------------------------------
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

vec = CountVectorizer(stop_words=list(STOP),
                      ngram_range=(1, 2),
                      min_df=3)

X = vec.fit_transform(corpus_df["clean"])
print("Docs kept after tokenisation:", X.shape[0])

# -------------------------------------------------------------
# 4.  Fit LDA (k = 5)
# -------------------------------------------------------------
lda = LatentDirichletAllocation(
    n_components=5,
    random_state=42,          # fixes randomness
    learning_method="batch",
    max_iter=500
)
doc_topic = lda.fit_transform(X)
vocab = vec.get_feature_names_out()

# -------------------------------------------------------------
# 5.  Top 12 terms per topic
# -------------------------------------------------------------
TOP_N = 12
for t in range(5):
    idx = lda.components_[t].argsort()[-TOP_N:][::-1]
    print(f"\nTopic {t+1}: {', '.join(vocab[i] for i in idx)}")

# -------------------------------------------------------------
# 6.  Utterance counts per dominant topic
# -------------------------------------------------------------
dominant = doc_topic.argmax(axis=1)
counts = pd.Series(dominant).value_counts().sort_index()
print("\nUtterance count per topic:")
for t, n in counts.items():
    print(f"Topic {t+1}: {n}")