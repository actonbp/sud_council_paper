#!/usr/bin/env python3
"""
LDA Topic Modeling for SUD Counselors Focus Group Analysis
===========================================================
This script performs Latent Dirichlet Allocation (LDA) topic modeling
on focus group transcripts to identify emergent themes.

Author: AI Agent for Bryan Acton
Date: December 2024
"""

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
DATA_DIR = "data"                       # Focus group CSVs location
paths = glob.glob(os.path.join(DATA_DIR, "*Focus_Group_full*.csv"))

print(f"\nFound {len(paths)} focus group files:")
for p in paths:
    print(f"  - {os.path.basename(p)}")

rows = []
for p in paths:
    df = pd.read_csv(p)
    # Remove moderator rows (2-3 letter speaker codes)
    df = df[~df["Speaker"].astype(str).str.match(r"^[A-Z]{2,3}$")]   
    df["session"] = os.path.basename(p)
    rows.append(df)

corpus_df = pd.concat(rows, ignore_index=True)
original_texts = corpus_df["Text"].astype(str).tolist()

print(f"\nTotal utterances after removing moderators: {len(original_texts)}")

# ------------------------------------------------------------------
# 2.  Pre-process text
# ------------------------------------------------------------------
# Domain-specific terms to remove (to focus on general themes)
DOMAIN_RE = re.compile(
    r"\b(substance|abuse|addict\w*|drug\w*|alcohol\w*|counsel\w*|mental\s+health)\b",
    re.I,
)

# Extended stop words list
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
    """Remove domain-specific terms and convert to lowercase"""
    txt = DOMAIN_RE.sub("", txt.lower())
    return txt

clean_texts = [clean_text(t) for t in original_texts]

# ------------------------------------------------------------------
# 3.  Vectorise (unigrams + bigrams, min_df = 3)
# ------------------------------------------------------------------
print("\nVectorizing text...")
vectorizer = CountVectorizer(stop_words=list(STOP_WORDS),
                             ngram_range=(1, 2),
                             min_df=3)

X = vectorizer.fit_transform(clean_texts)
vocab = vectorizer.get_feature_names_out()
print(f"Vocabulary size: {len(vocab)} terms")
print(f"Document-term matrix shape: {X.shape}")

# ------------------------------------------------------------------
# 4.  Fit LDA with k = 5
# ------------------------------------------------------------------
k = 5
print(f"\nFitting LDA with {k} topics...")
lda = LatentDirichletAllocation(n_components=k,
                                max_iter=500,
                                learning_method="batch",
                                random_state=42)      # ensures reproducibility
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
print("\n" + "="*60)
print("TOP TERMS PER TOPIC")
print("="*60)
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
print("\n" + "="*60)
print("REPRESENTATIVE UTTERANCES")
print("="*60)
for _, r in rep_df.iterrows():
    print(f"\n{r['Topic']}  (P={r['Prob']}) — {r['Session']} — Speaker {r['Speaker']}")
    print(f"→ {r['Utterance']}\n")

# ------------------------------------------------------------------
# 7.  Utterance counts per topic
# ------------------------------------------------------------------
topic_counts = (
    pd.Series(doc_topic.argmax(axis=1))
      .value_counts()
      .sort_index()
      .rename(lambda i: f"Topic {i+1}")
      .rename_axis("Topic")
      .reset_index(name="Utterance count")
)
print("\n" + "="*60)
print("UTTERANCE COUNT PER TOPIC")
print("="*60)
print(topic_counts.to_string(index=False))

# ------------------------------------------------------------------
# 8.  Save results to CSV files
# ------------------------------------------------------------------
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Save top terms
top_terms_df.to_csv(os.path.join(output_dir, "study2_lda_top_terms.csv"), index=False)
print(f"\nTop terms saved to: {os.path.join(output_dir, 'study2_lda_top_terms.csv')}")

# Save representative utterances
rep_df.to_csv(os.path.join(output_dir, "study2_lda_representative_quotes.csv"), index=False)
print(f"Representative quotes saved to: {os.path.join(output_dir, 'study2_lda_representative_quotes.csv')}")

# Save topic counts
topic_counts.to_csv(os.path.join(output_dir, "study2_lda_topic_counts.csv"), index=False)
print(f"Topic counts saved to: {os.path.join(output_dir, 'study2_lda_topic_counts.csv')}")

print("\n" + "="*60)
print("TOPIC MODELING COMPLETE")
print("="*60)