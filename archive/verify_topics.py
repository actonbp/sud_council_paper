# verify_topics.py  (k = 5, utterance level)
import glob, os, re, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

DATA_DIR = "data"                               # folder with CSVs
CSV_GLOB  = "*_Focus_[Gg]roup*full*.csv"          # pattern; edit if needed
SEED      = 42

# ---------- 1. Load, drop moderators ----------
paths = sorted(glob.glob(os.path.join(DATA_DIR, CSV_GLOB)))
assert paths, "No matching CSVs"

df_list = []
for p in paths:
    df = pd.read_csv(p)
    df = df[df["Text"].notna()]
    df = df[~df["Speaker"].astype(str).str.match(r"^[A-Z]{2,3}$")]
    df["session"] = os.path.basename(p)
    df_list.append(df)

corpus = pd.concat(df_list, ignore_index=True)
print("Rows after moderator removal:", len(corpus))

# ---------- 2. Clean ----------
DOMAIN = re.compile(r"\b(substance|abuse|addict\w*|drug\w*|alcohol\w*|counsel\w*|mental\s+health)\b", re.I)
corpus["clean"] = corpus["Text"].str.lower().apply(lambda t: DOMAIN.sub("", t))

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

vec = CountVectorizer(stop_words=list(STOP), ngram_range=(1, 2), min_df=3)
X = vec.fit_transform(corpus["clean"])
print("Docs kept after vectorisation:", X.shape[0])

# ---------- 3. Fit LDA ----------
lda = LatentDirichletAllocation(n_components=5,
                                random_state=SEED,
                                learning_method="batch",
                                max_iter=500)
doc_topic = lda.fit_transform(X)
vocab = vec.get_feature_names_out()

# ---------- 4. Top terms & counts ----------
for t in range(5):
    idx = lda.components_[t].argsort()[-12:][::-1]
    print(f"\nTopic {t+1}: {', '.join(vocab[i] for i in idx)}")

counts = pd.Series(doc_topic.argmax(axis=1)).value_counts().sort_index()
print("\nUtterance counts:")
for t, n in counts.items():
    print(f"Topic {t+1}: {n}")

print("\nRows per file after moderator removal:")
for p, df in zip(paths, df_list):
    print(os.path.basename(p), len(df))