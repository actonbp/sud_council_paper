# ------------------------------------------------------------------
# 0.  Imports  (install sklearn & pandas if needed)
# ------------------------------------------------------------------
import glob, os, re, textwrap
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ------------------------------------------------------------------
# 1.  Load all focus-group CSVs
# ------------------------------------------------------------------
DATA_DIR = "../../data"                       # ← put your CSVs here
paths = glob.glob(os.path.join(DATA_DIR, "*_Focus_Group_full*.csv"))

rows = []
for p in paths:
    df = pd.read_csv(p)
    df = df[~df["Speaker"].astype(str).str.match(r"^[A-Z]{2,3}$")]   # drop moderators
    df["session"] = os.path.basename(p)
    rows.append(df)

corpus_df = pd.concat(rows, ignore_index=True)
original_texts = corpus_df["Text"].astype(str).tolist()

# ------------------------------------------------------------------
# 2.  Pre-process text
# ------------------------------------------------------------------
DOMAIN_RE = re.compile(
    r"\b(substance|abuse|addict\w*|drug\w*|alcohol\w*|counsel\w*|mental\s+health)\b",
    re.I,
)
STOP_WORDS = set("""
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

def clean_text(txt: str) -> str:
    txt = DOMAIN_RE.sub("", txt.lower())
    return txt

clean_texts = [clean_text(t) for t in original_texts]

# ------------------------------------------------------------------
# 3.  Vectorise (unigrams + bigrams, min_df = 3)
# ------------------------------------------------------------------
vectorizer = CountVectorizer(stop_words=list(STOP_WORDS),
                             ngram_range=(1, 2),
                             min_df=3)

X = vectorizer.fit_transform(clean_texts)
vocab = vectorizer.get_feature_names_out()

# ------------------------------------------------------------------
# 4.  Fit LDA with k = 5
# ------------------------------------------------------------------
k = 5
lda = LatentDirichletAllocation(n_components=k,
                                max_iter=500,
                                learning_method="batch",
                                random_state=42)      # ensures identical results
doc_topic = lda.fit_transform(X)

# ------------------------------------------------------------------
# 5.  Top-terms per topic
# ------------------------------------------------------------------
TOP_N = 12
topic_terms = []
for t in range(k):
    top_idx = lda.components_[t].argsort()[-TOP_N:][::-1]
    terms = ", ".join(vocab[i] for i in top_idx)
    topic_terms.append((f"Topic {t+1}", terms))

top_terms_df = pd.DataFrame(topic_terms, columns=["Topic", "Top terms"])
print("\n=== Top terms per topic ===")
print(top_terms_df.to_string(index=False))

# ------------------------------------------------------------------
# 6.  Representative utterance for each topic
# ------------------------------------------------------------------
rep_rows = []
for t in range(k):
    probs = doc_topic[:, t]
    best = probs.argmax()
    rep_rows.append({
        "Topic": f"Topic {t+1}",
        "Prob": round(float(probs[best]), 3),
        "Speaker": corpus_df.iloc[best]["Speaker"],
        "Session": corpus_df.iloc[best]["session"],
        "Utterance": textwrap.fill(original_texts[best], 120)
    })

rep_df = pd.DataFrame(rep_rows)
print("\n=== Representative utterances ===")
for _, r in rep_df.iterrows():
    print(f"\n{r['Topic']}  (P={r['Prob']}) — {r['Session']} — Speaker {r['Speaker']}\n"
          f"→ {r['Utterance']}\n")

# ------------------------------------------------------------------
# 7.  (optional) utterance counts per topic
# ------------------------------------------------------------------
topic_counts = (
    pd.Series(doc_topic.argmax(axis=1))
      .value_counts()
      .sort_index()
      .rename(lambda i: f"Topic {i+1}")
      .rename_axis("Topic")
      .reset_index(name="Utterance count")
)
print("\n=== Utterance count per topic ===")
print(topic_counts.to_string(index=False))

# Save results
print("\n=== Saving results ===")
results_dir = "../../results/"
top_terms_df.to_csv(os.path.join(results_dir, "simple_topic_terms.csv"), index=False)
rep_df.to_csv(os.path.join(results_dir, "simple_representative_utterances.csv"), index=False)
topic_counts.to_csv(os.path.join(results_dir, "simple_topic_counts.csv"), index=False)
print("Results saved to results/ directory")