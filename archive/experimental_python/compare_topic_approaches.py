#!/usr/bin/env python3
"""
Compare Multiple Topic Modeling Approaches
Run different filtering and parameter combinations to find optimal approach
"""

import glob, os, re, textwrap
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics import silhouette_score

# Configuration
DATA_DIR = "../../data"
RESULTS_DIR = "../../results/"

def load_data():
    """Load focus group data"""
    paths = glob.glob(os.path.join(DATA_DIR, "*_Focus_Group_full*.csv"))
    
    rows = []
    for p in paths:
        df = pd.read_csv(p)
        df = df[~df["Speaker"].astype(str).str.match(r"^[A-Z]{2,3}$")]   # drop moderators
        df["session"] = os.path.basename(p)
        rows.append(df)
    
    corpus_df = pd.concat(rows, ignore_index=True)
    return corpus_df

def approach_1_simple(corpus_df):
    """Approach 1: Simple domain filtering (your original)"""
    print("\n" + "="*60)
    print("APPROACH 1: SIMPLE DOMAIN FILTERING")
    print("="*60)
    
    original_texts = corpus_df["Text"].astype(str).tolist()
    
    # Simple domain regex
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
    
    # Vectorize with CountVectorizer
    vectorizer = CountVectorizer(stop_words=list(STOP_WORDS),
                                 ngram_range=(1, 2),
                                 min_df=3)
    
    X = vectorizer.fit_transform(clean_texts)
    vocab = vectorizer.get_feature_names_out()
    
    # LDA with k=5
    k = 5
    lda = LatentDirichletAllocation(n_components=k,
                                    max_iter=500,
                                    learning_method="batch",
                                    random_state=42)
    doc_topic = lda.fit_transform(X)
    
    # Extract topics
    topics = []
    for t in range(k):
        top_idx = lda.components_[t].argsort()[-10:][::-1]
        terms = [vocab[i] for i in top_idx]
        topics.append(terms)
    
    # Interpretations
    interpretations = [
        "Emotional Investment Concerns",
        "Personal/Family Experience",
        "Career Interest & Uncertainty", 
        "Responsibility & Confidence",
        "Family Trauma & Recovery"
    ]
    
    return {
        'name': 'Simple Domain Filtering',
        'k': k,
        'topics': topics,
        'interpretations': interpretations,
        'doc_topic': doc_topic,
        'vocab': vocab,
        'X': X
    }

def approach_2_moderate_filtering(corpus_df):
    """Approach 2: Moderate domain filtering with TF-IDF"""
    print("\n" + "="*60)
    print("APPROACH 2: MODERATE FILTERING + TF-IDF")
    print("="*60)
    
    original_texts = corpus_df["Text"].astype(str).tolist()
    
    # Expanded domain terms
    DOMAIN_TERMS = set("""
    substance substances abuse addict addiction drug drugs alcohol
    counsel counselor counseling therapy therapist mental health
    treatment rehabilitation recovery clinic patient client
    """.split())
    
    STOP_WORDS = set("""
    a about above after again against all am an and any are as at be because been before being
    between both but by could did do does doing down during each few for from further had has
    have having he her here hers herself him himself his how i if in into is it its itself
    just like me more most my myself nor not of off on once only or other our ours ourselves
    out over own same she should so some such than that the their theirs them themselves then
    there these they this those through to too under until up very was we were what when where
    which while who whom why will with you your yours yourself yourselves um uh yeah okay kinda
    sorta right would know think really kind going lot can say definitely want guess something
    able way actually maybe feel feels felt get go got don didn won isn aren weren
    """.split())
    
    def clean_text(txt: str) -> str:
        txt = txt.lower()
        words = txt.split()
        filtered_words = [w for w in words if w not in DOMAIN_TERMS and len(w) >= 3]
        return " ".join(filtered_words)
    
    clean_texts = [clean_text(t) for t in original_texts]
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words=list(STOP_WORDS),
                                ngram_range=(1, 2),
                                min_df=3,
                                max_df=0.7,
                                max_features=300)
    
    X = vectorizer.fit_transform(clean_texts)
    vocab = vectorizer.get_feature_names_out()
    
    # Test k=3,4,5 and choose best
    best_score = -1
    best_k = 3
    best_model = None
    
    for k in [3, 4, 5]:
        lda = LatentDirichletAllocation(n_components=k,
                                        max_iter=500,
                                        learning_method="batch",
                                        random_state=42)
        doc_topic = lda.fit_transform(X)
        
        # Simple coherence proxy
        assignments = doc_topic.argmax(axis=1)
        if len(set(assignments)) > 1:  # Need multiple topics
            silhouette = silhouette_score(X.toarray(), assignments)
            if silhouette > best_score:
                best_score = silhouette
                best_k = k
                best_model = lda
    
    # Use best model
    doc_topic = best_model.fit_transform(X)
    
    # Extract topics
    topics = []
    for t in range(best_k):
        top_idx = best_model.components_[t].argsort()[-10:][::-1]
        terms = [vocab[i] for i in top_idx]
        topics.append(terms)
    
    # Generate interpretations based on terms
    interpretations = []
    for i, topic_terms in enumerate(topics):
        if any(word in ' '.join(topic_terms) for word in ['help', 'helping', 'support']):
            interpretations.append("Helping & Support Motivation")
        elif any(word in ' '.join(topic_terms) for word in ['family', 'parents', 'experience']):
            interpretations.append("Personal & Family Background")
        elif any(word in ' '.join(topic_terms) for word in ['school', 'education', 'learn']):
            interpretations.append("Educational & Learning Focus")
        elif any(word in ' '.join(topic_terms) for word in ['job', 'work', 'field', 'career']):
            interpretations.append("Career & Professional Interest")
        else:
            interpretations.append(f"Theme {i+1}")
    
    return {
        'name': 'Moderate Filtering + TF-IDF',
        'k': best_k,
        'topics': topics,
        'interpretations': interpretations,
        'doc_topic': doc_topic,
        'vocab': vocab,
        'X': X,
        'score': best_score
    }

def approach_3_minimal_filtering(corpus_df):
    """Approach 3: Minimal filtering, let topics emerge naturally"""
    print("\n" + "="*60)
    print("APPROACH 3: MINIMAL FILTERING")
    print("="*60)
    
    original_texts = corpus_df["Text"].astype(str).tolist()
    
    # Only remove most obvious circular terms
    CIRCULAR_TERMS = set(['counselor', 'counseling', 'therapist'])
    
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
        txt = txt.lower()
        for term in CIRCULAR_TERMS:
            txt = txt.replace(term, ' ')
        return txt
    
    clean_texts = [clean_text(t) for t in original_texts]
    
    # Simple CountVectorizer
    vectorizer = CountVectorizer(stop_words=list(STOP_WORDS),
                                ngram_range=(1, 2),
                                min_df=4,
                                max_features=250)
    
    X = vectorizer.fit_transform(clean_texts)
    vocab = vectorizer.get_feature_names_out()
    
    # LDA with k=4 (fewer topics for cleaner separation)
    k = 4
    lda = LatentDirichletAllocation(n_components=k,
                                    max_iter=500,
                                    learning_method="batch",
                                    random_state=42)
    doc_topic = lda.fit_transform(X)
    
    # Extract topics
    topics = []
    for t in range(k):
        top_idx = lda.components_[t].argsort()[-10:][::-1]
        terms = [vocab[i] for i in top_idx]
        topics.append(terms)
    
    # Simple interpretations
    interpretations = [
        "Field Interest & Concerns",
        "Personal Experience & Background", 
        "Helping & Support Orientation",
        "Professional & Academic Aspects"
    ]
    
    return {
        'name': 'Minimal Filtering',
        'k': k,
        'topics': topics,
        'interpretations': interpretations,
        'doc_topic': doc_topic,
        'vocab': vocab,
        'X': X
    }

def evaluate_approach(result):
    """Evaluate topic quality"""
    doc_topic = result['doc_topic']
    k = result['k']
    
    # Topic balance
    assignments = doc_topic.argmax(axis=1)
    topic_counts = [np.sum(assignments == i) for i in range(k)]
    balance = 1 - (np.std(topic_counts) / np.mean(topic_counts))
    
    # Topic distinctiveness (average of top terms across topics)
    all_top_terms = []
    for topic in result['topics']:
        all_top_terms.extend(topic[:8])
    distinctiveness = len(set(all_top_terms)) / len(all_top_terms)
    
    # Interpretability (subjective score based on meaningful terms)
    interpretability = 0
    for topic in result['topics']:
        meaningful_terms = sum(1 for term in topic[:8] if len(term) >= 4 and '_' not in term)
        interpretability += meaningful_terms
    interpretability = interpretability / (k * 8)
    
    return {
        'balance': balance,
        'distinctiveness': distinctiveness, 
        'interpretability': interpretability,
        'overall_score': (balance * 0.3 + distinctiveness * 0.4 + interpretability * 0.3)
    }

def display_comparison(results):
    """Display side-by-side comparison"""
    print("\n" + "="*80)
    print("TOPIC MODELING APPROACHES COMPARISON")
    print("="*80)
    
    # Evaluation table
    eval_data = []
    for result in results:
        metrics = evaluate_approach(result)
        eval_data.append({
            'Approach': result['name'],
            'Topics (k)': result['k'],
            'Balance': f"{metrics['balance']:.3f}",
            'Distinctiveness': f"{metrics['distinctiveness']:.3f}",
            'Interpretability': f"{metrics['interpretability']:.3f}",
            'Overall Score': f"{metrics['overall_score']:.3f}"
        })
    
    eval_df = pd.DataFrame(eval_data)
    print("\n>>> EVALUATION METRICS <<<")
    print(eval_df.to_string(index=False))
    
    # Topic details for each approach
    for result in results:
        print(f"\n" + "="*60)
        print(f"{result['name'].upper()} - {result['k']} TOPICS")
        print("="*60)
        
        for i, (topic, interp) in enumerate(zip(result['topics'], result['interpretations'])):
            assignments = result['doc_topic'].argmax(axis=1)
            count = np.sum(assignments == i)
            
            print(f"\nTopic {i+1}: {interp} ({count} docs)")
            print(f"  Top terms: {', '.join(topic[:8])}")
    
    # Save comparison
    eval_df.to_csv(os.path.join(RESULTS_DIR, "topic_approaches_comparison.csv"), index=False)
    
    # Recommend best approach
    best_approach = max(results, key=lambda x: evaluate_approach(x)['overall_score'])
    print(f"\n" + "🏆"*30)
    print(f"RECOMMENDED APPROACH: {best_approach['name']}")
    print(f"🏆"*30)
    print(f"Overall Score: {evaluate_approach(best_approach)['overall_score']:.3f}")
    print(f"Number of Topics: {best_approach['k']}")
    print(f"Why: Optimal balance of topic distinctiveness and interpretability")

def main():
    """Run all approaches and compare"""
    print("Loading focus group data...")
    corpus_df = load_data()
    print(f"Loaded {len(corpus_df)} utterances from {len(corpus_df['session'].unique())} sessions")
    
    # Run all approaches
    approaches = [
        approach_1_simple(corpus_df),
        approach_2_moderate_filtering(corpus_df), 
        approach_3_minimal_filtering(corpus_df)
    ]
    
    # Display comparison
    display_comparison(approaches)
    
    print(f"\nAll results saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()