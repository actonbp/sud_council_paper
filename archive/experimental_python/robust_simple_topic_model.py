#!/usr/bin/env python3
"""
Robust Simple Topic Model - No Gensim Dependencies
Bottom-up topic discovery with robustness testing using only sklearn
"""

import glob, os, re, textwrap
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "../../data"
RESULTS_DIR = "../../results/"

def load_focus_group_data():
    """Load focus group data"""
    print("📥 Loading focus group data for robust analysis...")
    
    paths = glob.glob(os.path.join(DATA_DIR, "*_Focus_Group_full*.csv"))
    rows = []
    for p in paths:
        df = pd.read_csv(p)
        df = df[~df["Speaker"].astype(str).str.match(r"^[A-Z]{2,3}$")]
        df = df[df["Text"].str.split().str.len() >= 8]
        df["session"] = os.path.basename(p)
        df["speaker_id"] = df["Speaker"].astype(str) + "_" + df["session"]
        rows.append(df)
    
    corpus_df = pd.concat(rows, ignore_index=True)
    texts = corpus_df['Text'].astype(str).tolist()
    
    print(f"Loaded {len(texts)} utterances from {len(corpus_df['speaker_id'].unique())} unique speakers")
    return texts, corpus_df

def clean_and_filter_texts(texts):
    """Clean texts with strategic domain filtering"""
    print("\n🧹 Strategic text cleaning and filtering...")
    
    # Strategic removal of most circular terms
    DOMAIN_TERMS = [
        'counselor', 'counselors', 'counseling', 
        'therapist', 'therapists', 'therapy',
        'psychologist', 'psychiatrist'
    ]
    
    # Enhanced stopwords
    STOP_WORDS = set("""
    a about above after again against all am an and any are as at be because been before being
    between both but by could did do does doing down during each few for from further had has
    have having he her here hers herself him himself his how i if in into is it its itself
    just like me more most my myself nor not of off on once only or other our ours ourselves
    out over own same she should so some such than that the their theirs them themselves then
    there these they this those through to too under until up very was we were what when where
    which while who whom why will with you your yours yourself yourselves
    um uh yeah okay kinda sorta right would know think really kind going lot can say 
    definitely want guess something able way actually maybe feel feels felt get got make 
    made see say said sure look looking yes no dont don't thats that's gonna wanna
    re ve ll don didn won isn aren weren hasn haven couldn wouldn shouldn mustn needn mightn
    also even just now well much many still back come came put take took give gave
    go went come came one two three first second next last another other
    thing things stuff whatever really little bit pretty much sort kind
    """.split())
    
    cleaned_texts = []
    terms_removed = 0
    
    for text in texts:
        # Convert to lowercase
        text = text.lower()
        
        # Remove domain terms
        for term in DOMAIN_TERMS:
            if term in text:
                terms_removed += text.count(term)
                text = re.sub(rf'\b{re.escape(term)}\b', ' ', text)
        
        # Basic cleaning
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\d+', ' ', text)      # Remove numbers
        text = re.sub(r'\s+', ' ', text)      # Normalize whitespace
        
        # Tokenize and filter
        tokens = [token for token in text.split() 
                 if len(token) >= 3 and token not in STOP_WORDS and token.isalpha()]
        
        if len(tokens) >= 5:  # Keep only substantial documents
            cleaned_texts.append(' '.join(tokens))
    
    print(f"Removed {terms_removed} circular domain term instances")
    print(f"Cleaned texts: {len(cleaned_texts)} documents retained")
    
    return cleaned_texts

def create_ngram_features(texts):
    """Create n-gram features using sklearn only"""
    print("\n📊 Creating n-gram feature matrix...")
    
    # Use CountVectorizer to identify meaningful n-grams first
    ngram_vectorizer = CountVectorizer(
        ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
        min_df=2,            # Must appear in at least 2 documents
        max_df=0.8,          # Not in more than 80% of documents
        max_features=None
    )
    
    ngram_matrix = ngram_vectorizer.fit_transform(texts)
    ngram_vocab = ngram_vectorizer.get_feature_names_out()
    
    # Filter for meaningful n-grams
    meaningful_ngrams = []
    ngram_scores = ngram_matrix.sum(axis=0).A1  # Document frequency
    
    for i, ngram in enumerate(ngram_vocab):
        score = ngram_scores[i]
        n_words = len(ngram.split())
        
        # Skip generic phrases
        if any(generic in ngram for generic in [
            'little bit', 'pretty much', 'kind like', 'sort like', 'really good',
            'right now', 'long time', 'first time', 'one thing', 'other thing'
        ]):
            continue
        
        # Keep meaningful phrases with good frequency
        if (n_words >= 2 and score >= 3) or (n_words == 1 and score >= 5):
            # Must contain meaningful content words
            if any(meaningful in ngram for meaningful in [
                'family', 'personal', 'experience', 'background', 'help', 'helping',
                'people', 'support', 'career', 'path', 'field', 'work', 'job',
                'responsibility', 'important', 'interest', 'care', 'caring',
                'school', 'education', 'learning', 'knowledge'
            ]):
                meaningful_ngrams.append(ngram)
    
    print(f"Extracted {len(meaningful_ngrams)} meaningful n-grams")
    print(f"Sample n-grams: {meaningful_ngrams[:12]}")
    
    return meaningful_ngrams

def create_weighted_features(texts, meaningful_ngrams):
    """Create weighted TF-IDF matrix"""
    print("\n⚖️ Creating weighted feature matrix...")
    
    def weighted_analyzer(text):
        """Custom analyzer that weights n-grams appropriately"""
        words = text.split()
        features = []
        
        # Add single words
        for word in words:
            if len(word) >= 4:  # Substantial words only
                features.append(word)
        
        # Add meaningful n-grams with higher weight
        for ngram in meaningful_ngrams:
            if ngram in text:
                n_words = len(ngram.split())
                # Weight: bigrams=2x, trigrams=3x
                weight = min(n_words, 3)
                features.extend([ngram.replace(' ', '_')] * weight)
        
        return features
    
    # Create TF-IDF matrix with weighted features
    vectorizer = TfidfVectorizer(
        analyzer=weighted_analyzer,
        min_df=2,
        max_df=0.85,
        max_features=120,
        sublinear_tf=True,
        norm='l2'
    )
    
    X = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    
    # Categorize features
    phrases = [term for term in vocab if '_' in term]
    words = [term for term in vocab if '_' not in term]
    
    print(f"Feature matrix: {X.shape[0]} docs × {X.shape[1]} features")
    print(f"  • {len(phrases)} phrases ({len(phrases)/len(vocab):.1%})")
    print(f"  • {len(words)} words ({len(words)/len(vocab):.1%})")
    
    return X, vocab, vectorizer

def run_topic_model(X, n_topics=4, random_state=42):
    """Run LDA topic model"""
    
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=1000,
        learning_method="batch",
        doc_topic_prior=0.1,
        topic_word_prior=0.01,
        random_state=random_state
    )
    
    doc_topic = lda.fit_transform(X)
    return lda, doc_topic

def interpret_topics(lda, vocab, doc_topic):
    """Interpret topics with meaningful labels"""
    print("\n🔍 Interpreting discovered topics...")
    
    topics_data = []
    assignments = doc_topic.argmax(axis=1)
    
    for topic_idx in range(lda.n_components):
        # Get top terms
        top_indices = lda.components_[topic_idx].argsort()[-15:][::-1]
        top_terms = [vocab[i] for i in top_indices]
        
        # Categorize terms
        phrases = [term for term in top_terms if '_' in term]
        words = [term for term in top_terms if '_' not in term]
        
        # Document statistics
        doc_count = np.sum(assignments == topic_idx)
        percentage = (doc_count / len(doc_topic)) * 100
        
        # Interpret based on content
        all_content = ' '.join(top_terms).lower()
        interpretation = interpret_topic_content(all_content, phrases, words)
        
        topics_data.append({
            'topic_num': topic_idx + 1,
            'interpretation': interpretation,
            'doc_count': doc_count,
            'percentage': percentage,
            'key_phrases': phrases[:4],
            'key_words': words[:6],
            'top_terms': top_terms[:10]
        })
        
        print(f"\nTopic {topic_idx + 1}: {interpretation}")
        print(f"  Documents: {doc_count} ({percentage:.1f}%)")
        if phrases:
            print(f"  Key phrases: {', '.join(phrases[:3])}")
        print(f"  Key words: {', '.join(words[:5])}")
    
    return topics_data

def interpret_topic_content(all_content, phrases, words):
    """Interpret topic based on content patterns"""
    
    # Pattern matching for interpretation
    if any(term in all_content for term in ['family', 'personal', 'experience', 'background']):
        if any(term in all_content for term in ['help', 'support', 'care']):
            return "Personal & Family Experience Driving Service"
        else:
            return "Personal & Family Background Influence"
    
    elif any(term in all_content for term in ['help', 'helping', 'people', 'support', 'care']):
        if any(term in all_content for term in ['responsibility', 'important']):
            return "Service-Oriented Helping with Responsibility"
        else:
            return "People-Focused Helping & Support"
    
    elif any(term in all_content for term in ['career', 'path', 'field', 'work', 'job']):
        if any(term in all_content for term in ['interest', 'future']):
            return "Career Interest & Professional Development"
        else:
            return "Professional Career Considerations"
    
    elif any(term in all_content for term in ['school', 'education', 'learn', 'knowledge']):
        return "Educational Pathway & Academic Preparation"
    
    else:
        return "Mixed Motivational Factors"

def run_robustness_analysis(texts, meaningful_ngrams, n_runs=5):
    """Test topic stability across multiple runs"""
    print(f"\n🔄 Running robustness analysis ({n_runs} runs)...")
    
    all_topic_terms = []
    
    for run in range(n_runs):
        print(f"  Run {run + 1}/{n_runs}...")
        
        # Create features and run model with different seed
        X, vocab, vectorizer = create_weighted_features(texts, meaningful_ngrams)
        lda, doc_topic = run_topic_model(X, n_topics=4, random_state=42 + run)
        
        # Extract top terms for each topic
        run_topics = []
        for topic_idx in range(4):
            top_indices = lda.components_[topic_idx].argsort()[-10:][::-1]
            top_terms = set([vocab[i] for i in top_indices])
            run_topics.append(top_terms)
        
        all_topic_terms.append(run_topics)
    
    # Calculate Jaccard similarity across runs
    topic_stabilities = []
    
    for topic_idx in range(4):
        similarities = []
        for i in range(n_runs):
            for j in range(i+1, n_runs):
                set1 = all_topic_terms[i][topic_idx]
                set2 = all_topic_terms[j][topic_idx]
                if len(set1.union(set2)) > 0:
                    jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
                    similarities.append(jaccard)
        
        topic_stabilities.append(np.mean(similarities) if similarities else 0)
    
    overall_stability = np.mean(topic_stabilities)
    
    # Rating
    if overall_stability >= 0.6:
        rating = "EXCELLENT"
    elif overall_stability >= 0.4:
        rating = "GOOD"
    elif overall_stability >= 0.3:
        rating = "MODERATE"
    else:
        rating = "POOR"
    
    print(f"\nRobustness Results:")
    print(f"  • Overall stability: {overall_stability:.3f} ({rating})")
    print(f"  • Topic-wise stability: {[f'{s:.3f}' for s in topic_stabilities]}")
    
    return {
        'overall_stability': overall_stability,
        'topic_stability': topic_stabilities,
        'rating': rating
    }

def save_results(topics_data, corpus_df, doc_topic, robustness):
    """Save all results"""
    
    # Document assignments
    assignments = doc_topic.argmax(axis=1)
    confidences = doc_topic.max(axis=1)
    
    # Results DataFrame
    results_data = []
    for doc_idx, (assignment, confidence) in enumerate(zip(assignments, confidences)):
        topic_data = topics_data[assignment]
        
        results_data.append({
            'document_id': doc_idx,
            'topic_num': topic_data['topic_num'],
            'topic_interpretation': topic_data['interpretation'],
            'confidence': round(confidence, 3),
            'speaker': corpus_df.iloc[doc_idx]['Speaker'],
            'speaker_id': corpus_df.iloc[doc_idx]['speaker_id'],
            'session': corpus_df.iloc[doc_idx]['session'],
            'text': corpus_df.iloc[doc_idx]['Text']
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(RESULTS_DIR, "robust_simple_topic_assignments.csv"), index=False)
    
    # Topic summary
    summary_data = []
    for topic_data in topics_data:
        summary_data.append({
            'Topic': f"Topic {topic_data['topic_num']}: {topic_data['interpretation']}",
            'Documents': topic_data['doc_count'],
            'Percentage': f"{topic_data['percentage']:.1f}%",
            'Key_Phrases': ', '.join(topic_data['key_phrases']) if topic_data['key_phrases'] else 'None',
            'Key_Words': ', '.join(topic_data['key_words'][:5]),
            'Stability': f"{robustness['topic_stability'][topic_data['topic_num']-1]:.3f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "robust_simple_topic_summary.csv"), index=False)
    
    # Report
    with open(os.path.join(RESULTS_DIR, "robust_simple_topic_report.txt"), 'w') as f:
        f.write("ROBUST SIMPLE TOPIC MODELING - SUD COUNSELING RESEARCH\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("METHODOLOGY - Bottom-Up Discovery:\n")
        f.write("• Strategic domain filtering (removed circular terms only)\n")
        f.write("• N-gram extraction using sklearn CountVectorizer\n")
        f.write("• Weighted TF-IDF (phrases get 2-3x weight)\n")
        f.write("• LDA topic modeling with 4 topics\n")
        f.write(f"• Robustness testing across 5 runs\n\n")
        
        f.write(f"ROBUSTNESS ASSESSMENT:\n")
        f.write(f"• Overall stability: {robustness['overall_stability']:.3f} ({robustness['rating']})\n")
        f.write(f"• Individual topic stability: {[f'{s:.3f}' for s in robustness['topic_stability']]}\n\n")
        
        f.write("DISCOVERED TOPICS:\n")
        f.write("-" * 20 + "\n")
        
        for topic_data in topics_data:
            f.write(f"\n{topic_data['interpretation']} ({topic_data['percentage']:.1f}%)\n")
            f.write(f"  Documents: {topic_data['doc_count']}\n")
            if topic_data['key_phrases']:
                f.write(f"  Key phrases: {', '.join(topic_data['key_phrases'])}\n")
            f.write(f"  Key words: {', '.join(topic_data['key_words'][:5])}\n")
            f.write(f"  Stability: {robustness['topic_stability'][topic_data['topic_num']-1]:.3f}\n")
    
    print(f"\n✅ Results saved to {RESULTS_DIR}")
    
    # Display representative quotes
    print("\n" + "="*70)
    print("ROBUST TOPIC MODELING - REPRESENTATIVE QUOTES")
    print("="*70)
    
    for topic_data in topics_data:
        topic_idx = topic_data['topic_num'] - 1
        topic_probs = doc_topic[:, topic_idx]
        best_doc_idx = topic_probs.argmax()
        best_prob = topic_probs[best_doc_idx]
        best_text = corpus_df.iloc[best_doc_idx]['Text']
        
        print(f"\n>>> {topic_data['interpretation']} <<<")
        print(f"Stability: {robustness['topic_stability'][topic_idx]:.3f} | Confidence: {best_prob:.3f}")
        if topic_data['key_phrases']:
            print(f"Key phrases: {', '.join(topic_data['key_phrases'][:3])}")
        print(f"Quote: {textwrap.fill(best_text, 70)}")

def main():
    """Execute robust simple topic modeling"""
    
    print("🎯 ROBUST SIMPLE TOPIC MODELING")
    print("=" * 50)
    print("Bottom-up discovery with robustness testing")
    
    # Load data
    texts, corpus_df = load_focus_group_data()
    
    # Clean and prepare
    cleaned_texts = clean_and_filter_texts(texts)
    
    # Extract meaningful n-grams
    meaningful_ngrams = create_ngram_features(cleaned_texts)
    
    # Create weighted features
    X, vocab, vectorizer = create_weighted_features(cleaned_texts, meaningful_ngrams)
    
    # Run topic model
    lda, doc_topic = run_topic_model(X, n_topics=4)
    
    # Interpret topics
    topics_data = interpret_topics(lda, vocab, doc_topic)
    
    # Test robustness
    robustness = run_robustness_analysis(cleaned_texts, meaningful_ngrams, n_runs=5)
    
    # Save results
    save_results(topics_data, corpus_df, doc_topic, robustness)
    
    print(f"\n🏆 ROBUST SIMPLE ANALYSIS COMPLETE!")
    print(f"📊 {len(topics_data)} topics discovered through bottom-up analysis")
    print(f"🔄 Robustness: {robustness['rating']} (stability={robustness['overall_stability']:.3f})")
    print(f"📝 {len(meaningful_ngrams)} meaningful n-grams identified")
    print("🎯 Ready for Study 1 individual linkage!")
    
    return topics_data, robustness

if __name__ == "__main__":
    topics, robustness = main()