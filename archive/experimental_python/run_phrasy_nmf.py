#!/usr/bin/env python3
# run_phrasy_nmf.py
"""
Phrase-aware TF-IDF + NMF topic model for the six SUD focus-group CSVs.
Outputs: MD5 integrity check, topic term lists, doc counts, basic stability & coherence.
"""

import os, re, hashlib, itertools, random, warnings
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import numpy as np

# ---------- 1.  INPUT FILES ----------
DATA_DIR = "./"              # change if needed
CSV_FILES = [
    "11_4_2024_11am_Focus_Group_full (1).csv",
    "11_6_2024_130pm_Focus_Group_full.csv",
    "11_8_2024_12pm_Focus_Group_full.csv",
    "11_11_2024_4pm_Focus_Group_full (1).csv",
    "11_12_2024_11am_Focus_Group_full.csv",
    "11_14_2024_4_pm_Focus_group__full.csv",
]

DOMAIN_RE = re.compile(
    r"\b(substance|abuse|addict\w*|drug\w*|alcohol\w*|counsel\w*|mental\s+health)\b",
    flags=re.I,
)

# ---------- 2.  LOAD & CLEAN ----------
docs = []
print("\n>>> File integrity check")
for fn in CSV_FILES:
    fp = os.path.join(DATA_DIR, fn)
    md5 = hashlib.md5(open(fp, "rb").read()).hexdigest()[:8]
    df  = pd.read_csv(fp)
    raw = len(df)
    df  = df[df["Text"].notna()]
    df  = df[~df["Speaker"].astype(str).str.fullmatch(r"[A-Z]{2,3}")]
    docs.extend(df["Text"].astype(str).tolist())
    print(f"{fn:45} {md5} | rows {raw:3} → kept {len(df):3}")
print(f"Total student utterances = {len(docs)}")

docs = [DOMAIN_RE.sub("", d.lower()) for d in docs]

# ---------- 3.  TOKENISE + PHRASE DETECT ----------
sentences = [simple_preprocess(d, deacc=True) for d in docs]
bigram  = Phrases(sentences, min_count=3, threshold=10, delimiter="_")
trigram = Phrases(bigram[sentences], threshold=9, delimiter="_")
phr     = Phraser(trigram)
docs_phr = [" ".join(phr[s]) for s in sentences]
tokens   = [phr[s] for s in sentences]          # for coherence later

# ---------- 4.  TF–IDF MATRIX ----------
extra_stop = set("""
feel think thing stuff kind gonna wanna like know work job family friends time
really actually maybe pretty basically bit lot kind_of sort_of yeah um uh okay
""".split())
stop_full = ENGLISH_STOP_WORDS.union(extra_stop)

tfidf = TfidfVectorizer(
    min_df=3, max_df=0.80,
    stop_words=stop_full,
    token_pattern=r"(?u)\b\w[\w'_]+\b",
)
X = tfidf.fit_transform(docs_phr)
vocab = tfidf.get_feature_names_out()
print(f"TF-IDF matrix = {X.shape[0]} docs × {X.shape[1]} terms")

# ---------- 5.  NMF TOPIC MODEL ----------
K = 5
nmf = NMF(n_components=K, init="nndsvd", max_iter=500, random_state=42)
W = nmf.fit_transform(X)
H = nmf.components_

def top_terms(mat, n=12):
    return [[vocab[i] for i in row.argsort()[-n:][::-1]] for row in mat]

topic_terms = top_terms(H, 15)
doc_assign  = W.argmax(axis=1)

print("\n>>> Topics (top-15 tokens)")
for k, terms in enumerate(topic_terms, 1):
    print(f"Topic {k}: {', '.join(terms)}")

print("\n>>> Utterances per topic")
for k in range(K):
    print(f"  Topic {k+1}: {(doc_assign==k).sum()}")

# ---------- 6.  QUICK ROBUSTNESS ----------
# 6a. Stability over 10 seeds
def jaccard(a,b): return len(a&b)/len(a|b)
seeds = range(10)
jac_mat = []
for s in seeds:
    m = NMF(n_components=K, init="nndsvd", max_iter=500, random_state=s)
    Hs = m.fit_transform(X)  # returns W, we want components_
    Hs = m.components_
    terms_s = top_terms(Hs,15)
    # Jaccard matching
    sim = np.zeros((K,K))
    for i in range(K):
        set_i = set(topic_terms[i])
        for j in range(K):
            sim[i,j] = jaccard(set_i, set(terms_s[j]))
    r,c = linear_sum_assignment(-sim)
    jac_mat.append([sim[i,j] for i,j in zip(r,c)])
jac_avg = np.mean(jac_mat, axis=0)
print(f"\nAvg Jaccard overlap per topic (10 seeds): {np.round(jac_avg,3)}   "
      f"overall = {jac_avg.mean():.3f}")

# 6b. Coherence
dictionary = Dictionary(tokens)
corpus     = [dictionary.doc2bow(t) for t in tokens]
co_umass = CoherenceModel(topics=topic_terms, texts=tokens,
                          dictionary=dictionary, coherence="u_mass").get_coherence()
co_cv    = CoherenceModel(topics=topic_terms, texts=tokens,
                          dictionary=dictionary, coherence="c_v").get_coherence()
print(f"Coherence:  UMass = {co_umass:.3f}   c_v = {co_cv:.3f}")