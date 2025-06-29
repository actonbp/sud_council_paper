#!/usr/bin/env python3
"""
Advanced Topic Modeling Comparison
Test different approaches: BERTopic, Sentence Transformers, and Semantic Clustering
"""

import glob, os, re, textwrap
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

# Core libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Advanced approaches (install if needed)
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("⚠️  sentence-transformers not installed. Installing...")
    os.system("pip install sentence-transformers")
    try:
        from sentence_transformers import SentenceTransformer
        HAS_SENTENCE_TRANSFORMERS = True
    except:
        HAS_SENTENCE_TRANSFORMERS = False

try:
    from bertopic import BERTopic
    HAS_BERTOPIC = True
except ImportError:
    HAS_BERTOPIC = False
    print("⚠️  BERTopic not installed. Installing...")
    os.system("pip install bertopic")
    try:
        from bertopic import BERTopic
        HAS_BERTOPIC = True
    except:
        HAS_BERTOPIC = False

# Configuration
DATA_DIR = "../../data"
RESULTS_DIR = "../../results/"

def load_clean_data():
    """Load and clean focus group data"""
    print("📥 Loading focus group data...")
    
    paths = glob.glob(os.path.join(DATA_DIR, "*_Focus_Group_full*.csv"))
    rows = []
    for p in paths:
        df = pd.read_csv(p)
        df = df[~df["Speaker"].astype(str).str.match(r"^[A-Z]{2,3}$")]
        df = df[df["Text"].str.split().str.len() >= 8]
        df["session"] = os.path.basename(p)
        rows.append(df)
    
    corpus_df = pd.concat(rows, ignore_index=True)
    
    # Clean texts - remove only most circular terms
    CIRCULAR_TERMS = ['counselor', 'counseling', 'therapist', 'therapy', 'mental_health', 'substance_abuse']
    
    cleaned_texts = []
    for text in corpus_df['Text'].astype(str):
        text = text.lower()
        for term in CIRCULAR_TERMS:
            text = re.sub(rf'\b{re.escape(term)}\b', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text.split()) >= 5:  # Keep substantial content
            cleaned_texts.append(text)
    
    print(f"Loaded {len(cleaned_texts)} clean utterances")
    return cleaned_texts, corpus_df

def approach_1_bertopic(texts):
    """BERTopic with sentence transformers"""
    if not HAS_BERTOPIC or not HAS_SENTENCE_TRANSFORMERS:
        print("❌ BERTopic or SentenceTransformers not available")
        return None
        
    print("\n🔥 APPROACH 1: BERTopic with Sentence Transformers")
    print("=" * 55)
    
    # Use a model optimized for semantic similarity
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create BERTopic model focused on meaningful clusters
    topic_model = BERTopic(
        embedding_model=embedding_model,
        nr_topics=4,  # Force 4 topics for comparison
        min_topic_size=10,  # Minimum documents per topic
        verbose=True
    )
    
    # Fit the model
    topics, probs = topic_model.fit_transform(texts)
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    print("\nBERTopic Results:")
    print(topic_info)
    
    # Extract representative words for each topic
    bertopic_results = []
    for topic_id in range(4):
        if topic_id in topic_model.topic_representations_:
            topic_words = topic_model.get_topic(topic_id)
            top_words = [word for word, score in topic_words[:10]]
            doc_count = sum(1 for t in topics if t == topic_id)
            
            # Generate semantic interpretation
            interpretation = interpret_semantic_topic(top_words)
            
            bertopic_results.append({
                'approach': 'BERTopic',
                'topic_id': topic_id,
                'interpretation': interpretation,
                'top_words': top_words,
                'doc_count': doc_count,
                'percentage': (doc_count / len(topics)) * 100
            })
            
            print(f"\nTopic {topic_id}: {interpretation}")
            print(f"  Documents: {doc_count} ({(doc_count/len(topics)*100):.1f}%)")
            print(f"  Top words: {', '.join(top_words[:8])}")
    
    return bertopic_results, topic_model

def approach_2_semantic_clustering(texts):
    """Semantic clustering with sentence transformers"""
    if not HAS_SENTENCE_TRANSFORMERS:
        print("❌ SentenceTransformers not available")
        return None
        
    print("\n🧠 APPROACH 2: Semantic Clustering")
    print("=" * 40)
    
    # Generate semantic embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    
    print(f"Generated embeddings: {embeddings.shape}")
    
    # K-means clustering on embeddings
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Analyze clusters
    semantic_results = []
    for cluster_id in range(4):
        cluster_texts = [texts[i] for i in range(len(texts)) if cluster_labels[i] == cluster_id]
        doc_count = len(cluster_texts)
        
        # Extract key terms from cluster using TF-IDF
        if cluster_texts:
            vectorizer = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1,2))
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = mean_scores.argsort()[-10:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            
            # Generate interpretation
            interpretation = interpret_semantic_topic(top_terms)
            
            semantic_results.append({
                'approach': 'Semantic Clustering',
                'topic_id': cluster_id,
                'interpretation': interpretation,
                'top_words': top_terms,
                'doc_count': doc_count,
                'percentage': (doc_count / len(texts)) * 100
            })
            
            print(f"\nCluster {cluster_id}: {interpretation}")
            print(f"  Documents: {doc_count} ({(doc_count/len(texts)*100):.1f}%)")
            print(f"  Top terms: {', '.join(top_terms[:8])}")
    
    return semantic_results, (embeddings, cluster_labels)

def approach_3_focused_lda(texts):
    """Focused LDA with semantic preprocessing"""
    print("\n📚 APPROACH 3: Focused LDA (Baseline)")
    print("=" * 40)
    
    # Create semantic-focused vectorizer
    def semantic_preprocessor(text):
        # Focus on meaningful semantic content
        words = text.split()
        semantic_words = []
        
        for word in words:
            if (len(word) >= 4 and 
                word not in ['really', 'pretty', 'kinda', 'sorta', 'little', 'very', 'just']):
                semantic_words.append(word)
        
        return ' '.join(semantic_words)
    
    preprocessed_texts = [semantic_preprocessor(text) for text in texts]
    
    # Semantic-focused vectorization
    vectorizer = TfidfVectorizer(
        max_features=200,
        min_df=3,
        max_df=0.7,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    X = vectorizer.fit_transform(preprocessed_texts)
    vocab = vectorizer.get_feature_names_out()
    
    # LDA with semantic focus
    lda = LatentDirichletAllocation(
        n_components=3,  # Try 3 for cleaner separation
        max_iter=1000,
        learning_method="batch",
        doc_topic_prior=0.1,
        topic_word_prior=0.01,
        random_state=42
    )
    
    doc_topic = lda.fit_transform(X)
    assignments = doc_topic.argmax(axis=1)
    
    # Extract topics
    lda_results = []
    for topic_id in range(3):
        top_indices = lda.components_[topic_id].argsort()[-15:][::-1]
        top_words = [vocab[i] for i in top_indices]
        doc_count = sum(assignments == topic_id)
        
        # Generate interpretation
        interpretation = interpret_semantic_topic(top_words)
        
        lda_results.append({
            'approach': 'Focused LDA',
            'topic_id': topic_id,
            'interpretation': interpretation,
            'top_words': top_words,
            'doc_count': doc_count,
            'percentage': (doc_count / len(texts)) * 100
        })
        
        print(f"\nTopic {topic_id}: {interpretation}")
        print(f"  Documents: {doc_count} ({(doc_count/len(texts)*100):.1f}%)")
        print(f"  Top words: {', '.join(top_words[:8])}")
    
    return lda_results, (lda, doc_topic, vocab)

def interpret_semantic_topic(terms):
    """Interpret topic based on semantic patterns"""
    terms_str = ' '.join(terms).lower()
    
    # Your identified core patterns
    if any(word in terms_str for word in ['help', 'helping', 'support', 'care', 'assist', 'serve']):
        if any(word in terms_str for word in ['people', 'others', 'community', 'lives']):
            return "Helping People & Service Motivation"
        else:
            return "Support & Care Orientation"
    
    elif any(word in terms_str for word in ['family', 'parents', 'mom', 'dad', 'personal', 'experience']):
        if any(word in terms_str for word in ['background', 'history', 'grew', 'childhood']):
            return "Family Background & Personal Experience"
        else:
            return "Family & Personal Influence"
    
    elif any(word in terms_str for word in ['school', 'education', 'college', 'university', 'study', 'learn']):
        if any(word in terms_str for word in ['psychology', 'major', 'degree', 'academic']):
            return "Educational & Academic Pathway"
        else:
            return "Education & Learning Focus"
    
    elif any(word in terms_str for word in ['career', 'job', 'work', 'professional', 'field']):
        if any(word in terms_str for word in ['future', 'path', 'choice', 'plan']):
            return "Career Planning & Development"
        else:
            return "Professional & Work Focus"
    
    elif any(word in terms_str for word in ['interested', 'interest', 'passion', 'love', 'enjoy']):
        return "Interest & Passion Motivation"
    
    elif any(word in terms_str for word in ['important', 'meaningful', 'significant', 'responsibility']):
        return "Purpose & Responsibility Awareness"
    
    else:
        # Fallback based on most prominent term
        if terms:
            return f"{terms[0].title()}-Centered Theme"
        else:
            return "Mixed Theme"

def compare_approaches(all_results):
    """Compare results across different approaches"""
    print("\n🔍 APPROACH COMPARISON")
    print("=" * 30)
    
    comparison_data = []
    
    for results in all_results:
        if results:
            approach_name = results[0]['approach']
            num_topics = len(results)
            
            # Check for core patterns
            core_patterns = {'helping': 0, 'family': 0, 'education': 0, 'career': 0}
            
            for result in results:
                interp = result['interpretation'].lower()
                if 'help' in interp or 'service' in interp:
                    core_patterns['helping'] += 1
                if 'family' in interp or 'personal' in interp:
                    core_patterns['family'] += 1  
                if 'education' in interp or 'academic' in interp or 'school' in interp:
                    core_patterns['education'] += 1
                if 'career' in interp or 'professional' in interp or 'work' in interp:
                    core_patterns['career'] += 1
            
            # Topic balance
            percentages = [r['percentage'] for r in results]
            balance_score = 1 - (np.std(percentages) / np.mean(percentages))
            
            comparison_data.append({
                'approach': approach_name,
                'num_topics': num_topics,
                'helping_themes': core_patterns['helping'],
                'family_themes': core_patterns['family'],
                'education_themes': core_patterns['education'],
                'career_themes': core_patterns['career'],
                'balance_score': round(balance_score, 3),
                'captures_core_patterns': sum(core_patterns.values())
            })
            
            print(f"\n{approach_name}:")
            print(f"  Topics: {num_topics}")
            print(f"  Core patterns captured: {sum(core_patterns.values())}")
            print(f"  Helping: {core_patterns['helping']}, Family: {core_patterns['family']}")
            print(f"  Education: {core_patterns['education']}, Career: {core_patterns['career']}")
            print(f"  Balance score: {balance_score:.3f}")
    
    # Find best approach
    if comparison_data:
        best_approach = max(comparison_data, key=lambda x: (x['captures_core_patterns'], x['balance_score']))
        print(f"\n🏆 BEST APPROACH: {best_approach['approach']}")
        print(f"   Captures {best_approach['captures_core_patterns']} core patterns with {best_approach['balance_score']:.3f} balance")
    
    return comparison_data

def save_advanced_results(all_results, comparison_data):
    """Save advanced topic modeling results"""
    
    # Flatten all results
    all_topics = []
    for results in all_results:
        if results:
            all_topics.extend(results)
    
    # Save detailed results
    results_df = pd.DataFrame(all_topics)
    results_df.to_csv(os.path.join(RESULTS_DIR, "advanced_topic_modeling_results.csv"), index=False)
    
    # Save comparison
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(RESULTS_DIR, "approach_comparison.csv"), index=False)
    
    # Detailed report
    with open(os.path.join(RESULTS_DIR, "advanced_topic_modeling_report.txt"), 'w') as f:
        f.write("ADVANCED TOPIC MODELING COMPARISON - SUD COUNSELING RESEARCH\n")
        f.write("=" * 65 + "\n\n")
        
        f.write("OBJECTIVE:\n")
        f.write("Capture the core patterns identified: education, family, helping people\n")
        f.write("Test different approaches beyond traditional LDA\n\n")
        
        f.write("APPROACHES TESTED:\n")
        f.write("-" * 20 + "\n")
        for data in comparison_data:
            f.write(f"\n{data['approach']}:\n")
            f.write(f"  • Topics: {data['num_topics']}\n")
            f.write(f"  • Core patterns captured: {data['captures_core_patterns']}\n")
            f.write(f"  • Balance score: {data['balance_score']}\n")
        
        f.write(f"\nDETAILED RESULTS BY APPROACH:\n")
        f.write("-" * 35 + "\n")
        
        current_approach = None
        for result in all_topics:
            if result['approach'] != current_approach:
                current_approach = result['approach']
                f.write(f"\n{current_approach}:\n")
            
            f.write(f"  Topic {result['topic_id']}: {result['interpretation']} ({result['percentage']:.1f}%)\n")
            f.write(f"    Top terms: {', '.join(result['top_words'][:8])}\n")
    
    print(f"\n✅ Advanced results saved to {RESULTS_DIR}")

def main():
    """Run advanced topic modeling comparison"""
    
    print("🚀 ADVANCED TOPIC MODELING COMPARISON")
    print("=" * 45)
    print("Testing: BERTopic, Semantic Clustering, Focused LDA")
    print("Goal: Better capture education, family, helping patterns")
    
    # Load data
    texts, corpus_df = load_clean_data()
    
    # Run different approaches
    all_results = []
    
    # Approach 1: BERTopic
    try:
        bertopic_results, bertopic_model = approach_1_bertopic(texts)
        all_results.append(bertopic_results)
    except Exception as e:
        print(f"BERTopic failed: {e}")
        all_results.append(None)
    
    # Approach 2: Semantic Clustering  
    try:
        semantic_results, semantic_data = approach_2_semantic_clustering(texts)
        all_results.append(semantic_results)
    except Exception as e:
        print(f"Semantic Clustering failed: {e}")
        all_results.append(None)
    
    # Approach 3: Focused LDA
    try:
        lda_results, lda_data = approach_3_focused_lda(texts)
        all_results.append(lda_results)
    except Exception as e:
        print(f"Focused LDA failed: {e}")
        all_results.append(None)
    
    # Compare approaches
    comparison_data = compare_approaches(all_results)
    
    # Save results
    save_advanced_results(all_results, comparison_data)
    
    print(f"\n🎯 ADVANCED COMPARISON COMPLETE!")
    print("📊 Results show which approach best captures core patterns")
    print("🔍 Focus: education, family, helping people themes")
    
    return all_results, comparison_data

if __name__ == "__main__":
    results, comparison = main()