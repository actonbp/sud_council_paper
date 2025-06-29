#!/usr/bin/env python3
"""
Enhanced LDA Topic Analysis with Theme Labels
============================================
This script performs LDA topic modeling with proper theme labeling
based on both top terms and representative quotes analysis.

Author: AI Agent for Bryan Acton
Date: December 2024
"""

import glob, os, re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns

# Theme labels based on analysis
THEME_LABELS = {
    0: "Emotional Labor & Boundaries",
    1: "Personal Experience & Academic",  
    2: "Career Considerations",
    3: "Helping & Responsibility",
    4: "Family & Support Systems"
}

# Load and preprocess data
DATA_DIR = "data"
paths = glob.glob(os.path.join(DATA_DIR, "*Focus_Group_full*.csv"))

rows = []
for p in paths:
    df = pd.read_csv(p)
    df = df[~df["Speaker"].astype(str).str.match(r"^[A-Z]{2,3}$")]   
    df["session"] = os.path.basename(p)
    rows.append(df)

corpus_df = pd.concat(rows, ignore_index=True)
original_texts = corpus_df["Text"].astype(str).tolist()

# Preprocessing
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

# Vectorize
vectorizer = CountVectorizer(stop_words=list(STOP_WORDS),
                             ngram_range=(1, 2),
                             min_df=3)
X = vectorizer.fit_transform(clean_texts)
vocab = vectorizer.get_feature_names_out()

# Fit LDA
k = 5
lda = LatentDirichletAllocation(n_components=k,
                                max_iter=500,
                                learning_method="batch",
                                random_state=42)
doc_topic = lda.fit_transform(X)

# Create theme distribution visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Document distribution across themes
theme_counts = pd.Series(doc_topic.argmax(axis=1)).value_counts().sort_index()
theme_percentages = (theme_counts / len(doc_topic) * 100).round(1)

bars = ax1.bar(range(k), theme_counts.values, color=plt.cm.Set3(range(k)))
ax1.set_xlabel('Theme', fontsize=12)
ax1.set_ylabel('Number of Utterances', fontsize=12)
ax1.set_title('Distribution of Utterances Across Themes', fontsize=14, fontweight='bold')
ax1.set_xticks(range(k))
ax1.set_xticklabels([THEME_LABELS[i] for i in range(k)], rotation=45, ha='right')

# Add percentage labels on bars
for i, (count, pct) in enumerate(zip(theme_counts.values, theme_percentages.values)):
    ax1.text(i, count + 1, f'{pct}%', ha='center', va='bottom', fontsize=10)

# Plot 2: Top terms per theme (horizontal bar chart)
ax2.set_title('Top 8 Terms per Theme', fontsize=14, fontweight='bold')
y_positions = []
colors = []

for topic_idx in range(k):
    top_indices = lda.components_[topic_idx].argsort()[-8:][::-1]
    top_words = [vocab[i] for i in top_indices]
    top_scores = [lda.components_[topic_idx][i] for i in top_indices]
    
    # Add theme label
    y_start = topic_idx * 9
    y_positions.extend(range(y_start, y_start + 8))
    colors.extend([plt.cm.Set3(topic_idx)] * 8)
    
    # Plot bars
    ax2.barh(y_positions[-8:], top_scores, color=colors[-8:], alpha=0.8)
    
    # Add theme label
    ax2.text(-5, y_start + 3.5, THEME_LABELS[topic_idx], 
             fontsize=11, fontweight='bold', ha='right', va='center')
    
    # Add word labels
    for j, (word, score) in enumerate(zip(top_words, top_scores)):
        ax2.text(score + 0.5, y_start + j, word, va='center', fontsize=9)

ax2.set_ylim(-1, k * 9)
ax2.set_xlabel('Term Weight', fontsize=12)
ax2.set_yticks([])
ax2.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('results/study2_lda_theme_visualization.png', dpi=300, bbox_inches='tight')
plt.savefig('results/study2_lda_theme_visualization.pdf', bbox_inches='tight')
print("\nTheme visualization saved to results/study2_lda_theme_visualization.png")

# Create detailed theme summary
theme_summary = []
for t in range(k):
    # Get top 15 terms
    top_idx = lda.components_[t].argsort()[-15:][::-1]
    terms = ", ".join(vocab[i] for i in top_idx)
    
    # Get theme statistics
    utterance_count = (doc_topic.argmax(axis=1) == t).sum()
    percentage = round(utterance_count / len(doc_topic) * 100, 1)
    
    # Get most representative quote
    probs = doc_topic[:, t]
    best_idx = probs.argmax()
    best_prob = round(float(probs[best_idx]), 3)
    
    theme_summary.append({
        'Theme': THEME_LABELS[t],
        'Utterance Count': utterance_count,
        'Percentage': f"{percentage}%",
        'Top 15 Terms': terms,
        'Representative Quote Probability': best_prob,
        'Representative Speaker': corpus_df.iloc[best_idx]["Speaker"],
        'Representative Session': corpus_df.iloc[best_idx]["session"]
    })

# Save theme summary
theme_df = pd.DataFrame(theme_summary)
theme_df.to_csv('results/study2_lda_labeled_themes.csv', index=False)
print("\nLabeled theme summary saved to results/study2_lda_labeled_themes.csv")

# Print summary
print("\n" + "="*60)
print("THEMATIC ANALYSIS SUMMARY")
print("="*60)
for _, row in theme_df.iterrows():
    print(f"\n{row['Theme']}")
    print(f"  Utterances: {row['Utterance Count']} ({row['Percentage']})")
    print(f"  Key terms: {row['Top 15 Terms'][:80]}...")
    print(f"  Best example: P={row['Representative Quote Probability']} from {row['Representative Speaker']}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)