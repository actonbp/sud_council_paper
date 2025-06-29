#!/usr/bin/env python
"""
optimal_topic_analysis.py - Data-driven topic modeling with optimal k selection

This script:
1. Tests k = 2 through 10 topics
2. Uses TF-IDF weighting to reduce cross-topic terms
3. Calculates perplexity and silhouette scores for model selection
4. Provides coherence analysis
5. Shows topic distinctiveness metrics
"""

import glob, os, hashlib, re, textwrap, sys, platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

print("Python:", sys.version.split()[0], "| Platform:", platform.platform())
import sklearn
print("scikit-learn:", sklearn.__version__)
print("-" * 60)

# -------------------------------------------------------------------
# 1. Data Loading (same as before)
# -------------------------------------------------------------------
DATA_DIR = "data"
CSV_GLOB = "*_Focus_*_full*.csv"
paths = sorted(glob.glob(os.path.join(DATA_DIR, CSV_GLOB)))
paths = [p for p in paths if 'focus' in os.path.basename(p).lower()]

if not paths:
    sys.exit(f"No files found via pattern {CSV_GLOB} in {DATA_DIR}")

print("Files detected:")
for p in paths:
    md5 = hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
    print(f"- {os.path.basename(p):40}  {md5}")
print("-" * 60)

# Load and clean data
df_list = []
for p in paths:
    df = pd.read_csv(p)
    df = df[df["Text"].notna()]
    is_mod = df["Speaker"].astype(str).str.match(r"^[A-Z]{2,3}$")
    df = df[~is_mod].copy()
    df["session"] = os.path.basename(p)
    df_list.append(df)

corpus = pd.concat(df_list, ignore_index=True)
print(f"Total utterances after cleaning: {len(corpus)}")

# Text preprocessing
DOMAIN = re.compile(r"\b(substance|abuse|addict\w*|drug\w*|alcohol\w*|counsel\w*|mental\s+health)\b", re.I)
corpus["clean"] = corpus["Text"].str.lower().apply(lambda t: DOMAIN.sub("", t))

# Enhanced stop words
STOP = set("""
a about above after again against all am an and any are as at be because been before being
between both but by could did do does doing down during each few for from further had has
have having he her here hers herself him himself his how i if in into is it its itself
just like me more most my myself nor not of off on once only or other our ours ourselves
out over own same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were what when where
which while who whom why will with you your yours yourself yourselves um uh yeah okay kinda
sorta right would know think really kind going lot can say definitely want guess something
able way actually maybe feel feels felt don re ve ll got get goes didn wouldn couldn
shouldn won isn aren wasn hasn haven hadn even still always never already back come came
comes coming came well better best good bad worse worst big bigger biggest small smaller
smallest much many make makes made making take takes took taking give gives gave giving
put puts putting look looks looked looking see sees saw seeing tell tells told telling
""".split())

# -------------------------------------------------------------------
# 2. TF-IDF Vectorization (reduces cross-topic terms)
# -------------------------------------------------------------------
print("\nUsing TF-IDF to reduce cross-topic contamination...")
vectorizer = TfidfVectorizer(
    stop_words=list(STOP),
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.8,  # Remove terms in >80% of docs
    sublinear_tf=True  # Use log scaling
)

X = vectorizer.fit_transform(corpus["clean"])
vocab = vectorizer.get_feature_names_out()
print(f"Feature matrix: {X.shape[0]} docs × {X.shape[1]} features")

# -------------------------------------------------------------------
# 3. Test Multiple k Values (Data-Driven Approach)
# -------------------------------------------------------------------
k_range = range(2, 11)
results = []

print("\nTesting optimal number of topics...")
print("k\tPerplexity\tSilhouette\tLog-Likelihood")
print("-" * 50)

for k in k_range:
    # Fit LDA
    lda = LatentDirichletAllocation(
        n_components=k,
        random_state=42,
        learning_method="batch",
        max_iter=500,
        doc_topic_prior=0.1,  # Lower = more sparse topics (was alpha)
        topic_word_prior=0.01   # Lower = more focused topics (was beta)
    )
    
    doc_topic = lda.fit_transform(X)
    
    # Calculate metrics
    perplexity = lda.perplexity(X)
    log_likelihood = lda.score(X)
    
    # Silhouette score on topic assignments
    dominant_topics = doc_topic.argmax(axis=1)
    if len(np.unique(dominant_topics)) > 1:  # Need at least 2 clusters
        silhouette = silhouette_score(doc_topic, dominant_topics)
    else:
        silhouette = -1
    
    results.append({
        'k': k,
        'perplexity': perplexity,
        'silhouette': silhouette,
        'log_likelihood': log_likelihood,
        'lda_model': lda,
        'doc_topic': doc_topic
    })
    
    print(f"{k}\t{perplexity:.1f}\t\t{silhouette:.3f}\t\t{log_likelihood:.1f}")

# -------------------------------------------------------------------
# 4. Select Optimal k
# -------------------------------------------------------------------
# Convert to DataFrame for analysis
df_results = pd.DataFrame([{k: v for k, v in r.items() if k not in ['lda_model', 'doc_topic']} 
                          for r in results])

# Normalize metrics for comparison (0-1 scale)
df_results['perp_norm'] = 1 - (df_results['perplexity'] - df_results['perplexity'].min()) / \
                         (df_results['perplexity'].max() - df_results['perplexity'].min())
df_results['sil_norm'] = (df_results['silhouette'] - df_results['silhouette'].min()) / \
                        (df_results['silhouette'].max() - df_results['silhouette'].min())
df_results['ll_norm'] = (df_results['log_likelihood'] - df_results['log_likelihood'].min()) / \
                       (df_results['log_likelihood'].max() - df_results['log_likelihood'].min())

# Composite score (equal weighting)
df_results['composite_score'] = (df_results['perp_norm'] + df_results['sil_norm'] + df_results['ll_norm']) / 3

optimal_k = df_results.loc[df_results['composite_score'].idxmax(), 'k']
print(f"\nOptimal k based on composite score: {optimal_k}")

# -------------------------------------------------------------------
# 5. Analyze Optimal Model
# -------------------------------------------------------------------
optimal_result = next(r for r in results if r['k'] == optimal_k)
lda_optimal = optimal_result['lda_model']
doc_topic_optimal = optimal_result['doc_topic']

print(f"\n{'='*60}")
print(f"OPTIMAL MODEL ANALYSIS (k = {optimal_k})")
print(f"{'='*60}")

# Topic terms with distinctiveness scores
print(f"\n{'='*80}")
print("COMPLETE TERM ANALYSIS")
print(f"{'='*80}")

# Calculate distinctiveness for all terms
avg_probs = lda_optimal.components_.mean(axis=0)

print("\nDistinctiveness Score Explanation:")
print("• Score = (Topic Probability) / (Average Probability Across All Topics)")
print("• 1.0 = Term appears equally in all topics")
print("• >1.0 = Term is MORE distinctive to this topic")
print("• 2.0 = Term appears 2x more in this topic than average")
print("• <1.0 = Term appears LESS in this topic than average")

for t in range(optimal_k):
    print(f"\n{'-'*60}")
    print(f"TOPIC {t+1} - ALL TERMS (ranked by distinctiveness)")
    print(f"{'-'*60}")
    
    # Get topic-term probabilities
    topic_terms = lda_optimal.components_[t]
    distinctiveness = topic_terms / (avg_probs + 1e-10)
    
    # Sort by distinctiveness (most distinctive first)
    distinct_indices = distinctiveness.argsort()[::-1]
    
    print("Most Distinctive Terms (>1.2):")
    distinctive_terms = []
    for idx in distinct_indices:
        term = vocab[idx]
        prob = topic_terms[idx]
        distinct = distinctiveness[idx]
        if distinct > 1.2 and prob > 0.001:  # Only show meaningful terms
            distinctive_terms.append(f"{term}({distinct:.1f})")
        if len(distinctive_terms) >= 20:  # Limit output
            break
    print(f"  {', '.join(distinctive_terms)}")
    
    print(f"\nAll Terms with Probability >0.005 (sorted by distinctiveness):")
    all_terms = []
    for idx in distinct_indices:
        term = vocab[idx]
        prob = topic_terms[idx]
        distinct = distinctiveness[idx]
        if prob > 0.005:  # Show terms with meaningful probability
            all_terms.append(f"{term}({distinct:.1f}, p={prob:.3f})")
        if len(all_terms) >= 30:  # Reasonable limit
            break
    
    # Split into chunks for readability
    chunk_size = 5
    for i in range(0, len(all_terms), chunk_size):
        chunk = all_terms[i:i+chunk_size]
        print(f"  {', '.join(chunk)}")
        
    # Show topic probability distribution
    topic_probs = doc_topic_optimal[:, t]
    high_prob_docs = (topic_probs > 0.5).sum()
    print(f"\nTopic Statistics:")
    print(f"  • Documents strongly assigned (>50%): {high_prob_docs}")
    print(f"  • Average document probability: {topic_probs.mean():.3f}")
    print(f"  • Max document probability: {topic_probs.max():.3f}")

# Topic sizes
dominant = doc_topic_optimal.argmax(axis=1)
counts = pd.Series(dominant).value_counts().sort_index()
print(f"\nTopic distribution:")
for t, n in counts.items():
    pct = 100 * n / len(corpus)
    print(f"Topic {t+1}: {n} utterances ({pct:.1f}%)")

# -------------------------------------------------------------------
# 6. Topic Quality Metrics
# -------------------------------------------------------------------
print(f"\n{'='*60}")
print("TOPIC QUALITY ANALYSIS")
print(f"{'='*60}")

# Calculate topic coherence (simplified)
def topic_coherence(topic_terms, X, vocab, top_n=10):
    """Calculate topic coherence based on co-occurrence"""
    top_indices = topic_terms.argsort()[-top_n:][::-1]
    coherence_sum = 0
    count = 0
    
    # Convert sparse matrix to dense for boolean operations
    X_dense = X.toarray()
    
    for i, idx1 in enumerate(top_indices):
        for idx2 in top_indices[i+1:]:
            # Calculate PMI-like score
            term1_docs = (X_dense[:, idx1] > 0).sum()
            term2_docs = (X_dense[:, idx2] > 0).sum()
            both_docs = ((X_dense[:, idx1] > 0) & (X_dense[:, idx2] > 0)).sum()
            
            if both_docs > 0:
                coherence_sum += np.log((both_docs * X.shape[0]) / (term1_docs * term2_docs + 1e-10))
                count += 1
    
    return coherence_sum / count if count > 0 else 0

print("Topic coherence scores:")
for t in range(optimal_k):
    coherence = topic_coherence(lda_optimal.components_[t], X, vocab)
    print(f"Topic {t+1}: {coherence:.3f}")

# Topic separation
print(f"\nTopic separation (Jensen-Shannon divergence):")
from scipy.spatial.distance import jensenshannon

topic_similarities = []
for i in range(optimal_k):
    for j in range(i+1, optimal_k):
        js_div = jensenshannon(lda_optimal.components_[i], lda_optimal.components_[j])
        topic_similarities.append(js_div)
        print(f"Topics {i+1}-{j+1}: {js_div:.3f}")

avg_separation = np.mean(topic_similarities)
print(f"Average topic separation: {avg_separation:.3f} (higher = more distinct)")

print(f"\n{'='*60}")
print("SUMMARY RECOMMENDATIONS")
print(f"{'='*60}")
print(f"• Optimal k: {optimal_k} topics")
print(f"• Model uses TF-IDF weighting to reduce cross-topic contamination")
print(f"• Alpha/Beta parameters tuned for topic sparsity")
print(f"• Average topic separation: {avg_separation:.3f}")
print(f"• Perplexity: {optimal_result['perplexity']:.1f}")
print(f"• Silhouette score: {optimal_result['silhouette']:.3f}")