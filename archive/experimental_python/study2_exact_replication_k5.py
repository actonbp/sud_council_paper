#!/usr/bin/env python3
"""
Exact Replication of K=5 LDA Analysis
=====================================
Attempting to reproduce exact results from the other agent
"""

import glob, os, re, textwrap
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load all focus-group CSVs
DATA_DIR = "data"
paths = glob.glob(os.path.join(DATA_DIR, "*_Focus_Group_full*.csv"))

rows = []
for p in paths:
    df = pd.read_csv(p)
    # Remove moderator rows
    df = df[~df["Speaker"].astype(str).str.match(r"^[A-Z]{2,3}$")]   
    df["session"] = os.path.basename(p)
    rows.append(df)

corpus_df = pd.concat(rows, ignore_index=True)
original_texts = corpus_df["Text"].astype(str).tolist()

print(f"Total utterances: {len(original_texts)}")

# Pre-process text - matching exact pattern
DOMAIN_RE = re.compile(
    r"\b(substance|abuse|addict\w*|drug\w*|alcohol\w*|counsel\w*|mental\s+health)\b",
    re.I,
)

# Standard stop words list
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

# Vectorize with exact parameters
vectorizer = CountVectorizer(
    stop_words=list(STOP_WORDS),
    ngram_range=(1, 2),
    min_df=3
)

X = vectorizer.fit_transform(clean_texts)
vocab = vectorizer.get_feature_names_out()

print(f"Vocabulary size: {len(vocab)}")
print(f"Document-term matrix: {X.shape}")

# Fit LDA with k = 5
k = 5
lda = LatentDirichletAllocation(
    n_components=k,
    max_iter=500,
    learning_method="batch",
    random_state=42
)
doc_topic = lda.fit_transform(X)

# Extract top 12 terms per topic
TOP_N = 12
print("\n" + "="*60)
print("TOP 12 TERMS PER TOPIC (K=5)")
print("="*60)

for t in range(k):
    top_idx = lda.components_[t].argsort()[-TOP_N:][::-1]
    terms = ", ".join(vocab[i] for i in top_idx)
    print(f"\nTopic {t+1}: {terms}")

# Count utterances per topic
print("\n" + "="*60)
print("UTTERANCE COUNTS PER TOPIC")
print("="*60)

topic_assignments = doc_topic.argmax(axis=1)
for t in range(k):
    count = (topic_assignments == t).sum()
    print(f"Topic {t+1}: {count} utterances")

# Total check
print(f"\nTotal: {len(topic_assignments)} utterances")

# Save detailed results for comparison
results_df = pd.DataFrame({
    'utterance_id': range(len(original_texts)),
    'assigned_topic': topic_assignments + 1,
    'max_probability': doc_topic.max(axis=1)
})
results_df.to_csv('results/study2_k5_replication_assignments.csv', index=False)

# Save topic-term matrix
topic_term_df = pd.DataFrame(
    lda.components_.T,
    index=vocab,
    columns=[f'Topic_{i+1}' for i in range(k)]
)
topic_term_df.to_csv('results/study2_k5_topic_term_matrix.csv')

print("\nDetailed results saved for verification.")