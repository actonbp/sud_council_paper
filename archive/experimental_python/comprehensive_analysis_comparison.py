#!/usr/bin/env python3
"""
Comprehensive Analysis Comparison
Testing multiple approaches to find the most interesting and robust patterns:
1. TF-IDF ranking (most distinctive terms)
2. Co-occurrence networks (network analysis)
3. Sentiment and emotion analysis
4. Keyword-in-context analysis
5. Collocation analysis (statistical word pairs)
6. N-gram frequency analysis
7. Semantic clustering
"""

import glob, os, re, textwrap
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "../../data"
RESULTS_DIR = "../../results/"

def load_focus_group_data():
    """Load focus group data"""
    print("📥 Loading focus group data for comprehensive analysis...")
    
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

def clean_texts_comprehensive(texts):
    """Clean texts while preserving different types of analysis needs"""
    print("\n🧹 Comprehensive text cleaning...")
    
    # Remove only circular terms
    DOMAIN_TERMS = ['counselor', 'counselors', 'counseling', 'therapist', 'therapists', 'therapy']
    
    cleaned_texts = []
    
    for text in texts:
        # Convert to lowercase
        text = text.lower()
        
        # Remove domain terms
        for term in DOMAIN_TERMS:
            text = re.sub(rf'\b{re.escape(term)}\b', ' ', text)
        
        # Light cleaning - preserve contractions and emotional language
        text = re.sub(r'[^\w\s\']', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        if len(text.split()) >= 5:
            cleaned_texts.append(text)
    
    print(f"Retained {len(cleaned_texts)} documents for analysis")
    return cleaned_texts

def approach_1_tfidf_distinctive_terms(texts):
    """Approach 1: TF-IDF to find most distinctive terms"""
    print("\n🔍 APPROACH 1: TF-IDF Distinctive Terms Analysis")
    print("=" * 55)
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(
        max_features=500,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate mean TF-IDF scores across all documents
    mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
    
    # Get distinctive terms (high average TF-IDF)
    term_scores = list(zip(feature_names, mean_scores))
    term_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Filter for meaningful terms
    distinctive_terms = []
    for term, score in term_scores:
        if (len(term) >= 4 and 
            score > 0.01 and 
            not any(filler in term for filler in ['like', 'just', 'really', 'kind', 'sort'])):
            distinctive_terms.append((term, score))
    
    print(f"Most Distinctive Terms (by TF-IDF):")
    for term, score in distinctive_terms[:20]:
        print(f"  {term}: {score:.4f}")
    
    # Find documents with highest scores for top terms
    top_terms = [term for term, score in distinctive_terms[:5]]
    distinctive_docs = {}
    
    for term in top_terms:
        if term in feature_names:
            term_idx = list(feature_names).index(term)
            doc_scores = tfidf_matrix[:, term_idx].toarray().flatten()
            best_doc_idx = np.argmax(doc_scores)
            distinctive_docs[term] = {
                'doc_idx': best_doc_idx,
                'score': doc_scores[best_doc_idx],
                'text': texts[best_doc_idx][:200] + '...'
            }
    
    print(f"\nMost distinctive document examples:")
    for term, data in distinctive_docs.items():
        print(f"\n'{term}' (score: {data['score']:.4f}):")
        print(f"  {textwrap.fill(data['text'], 70)}")
    
    return {
        'distinctive_terms': distinctive_terms,
        'distinctive_docs': distinctive_docs,
        'interest_score': len([t for t, s in distinctive_terms[:20] if any(keyword in t for keyword in ['help', 'family', 'personal', 'career', 'important'])])
    }

def approach_2_cooccurrence_networks(texts):
    """Approach 2: Co-occurrence Network Analysis"""
    print("\n🔗 APPROACH 2: Co-occurrence Network Analysis")
    print("=" * 50)
    
    # Extract meaningful words
    all_words = []
    for text in texts:
        words = [word for word in text.split() 
                if len(word) >= 4 and word.isalpha()]
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    frequent_words = [word for word, count in word_counts.items() if count >= 5]
    
    # Build co-occurrence matrix
    cooccurrence_matrix = defaultdict(lambda: defaultdict(int))
    window_size = 5
    
    for text in texts:
        words = [word for word in text.split() if word in frequent_words]
        
        for i, word1 in enumerate(words):
            for j in range(max(0, i-window_size), min(len(words), i+window_size+1)):
                if i != j:
                    word2 = words[j]
                    cooccurrence_matrix[word1][word2] += 1
    
    # Find strongest connections
    strong_connections = []
    for word1, connections in cooccurrence_matrix.items():
        for word2, strength in connections.items():
            if strength >= 5 and word1 < word2:  # Avoid duplicates
                strong_connections.append((word1, word2, strength))
    
    strong_connections.sort(key=lambda x: x[2], reverse=True)
    
    print(f"Strongest Word Connections (≥5 co-occurrences):")
    for word1, word2, strength in strong_connections[:20]:
        print(f"  {word1} ↔ {word2}: {strength} connections")
    
    # Identify central hub words (most connections)
    hub_scores = defaultdict(int)
    for word1, word2, strength in strong_connections:
        hub_scores[word1] += strength
        hub_scores[word2] += strength
    
    top_hubs = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\nCentral Hub Words (most connected):")
    for word, total_connections in top_hubs:
        print(f"  {word}: {total_connections} total connection strength")
    
    return {
        'strong_connections': strong_connections,
        'hub_words': top_hubs,
        'interest_score': len([conn for conn in strong_connections[:20] if any(keyword in conn[0] or keyword in conn[1] for keyword in ['help', 'family', 'personal', 'career'])])
    }

def approach_3_sentiment_emotion_analysis(texts):
    """Approach 3: Simple Sentiment and Emotion Analysis"""
    print("\n😊 APPROACH 3: Sentiment and Emotion Analysis")
    print("=" * 48)
    
    # Define emotion/sentiment lexicons
    positive_words = {
        'interested', 'interesting', 'love', 'enjoy', 'like', 'good', 'great', 'amazing', 
        'wonderful', 'excited', 'passionate', 'satisfied', 'rewarding', 'meaningful', 
        'important', 'valuable', 'helpful', 'supportive', 'encouraging', 'positive'
    }
    
    negative_words = {
        'worried', 'scared', 'afraid', 'anxious', 'nervous', 'difficult', 'hard', 
        'challenging', 'overwhelming', 'stressful', 'bad', 'terrible', 'awful', 
        'concerned', 'hesitant', 'unsure', 'uncertain', 'doubtful', 'reluctant'
    }
    
    uncertainty_words = {
        'maybe', 'perhaps', 'possibly', 'might', 'could', 'not sure', 'unsure', 
        'uncertain', 'thinking about', 'considering', 'wondering', 'guess', 
        'probably', 'potentially', 'unclear'
    }
    
    helping_emotion_words = {
        'care', 'caring', 'compassion', 'empathy', 'understanding', 'support', 
        'supportive', 'nurturing', 'protecting', 'healing', 'comforting'
    }
    
    # Analyze sentiment in each document
    sentiment_scores = []
    emotion_patterns = []
    
    for doc_idx, text in enumerate(texts):
        words = text.lower().split()
        
        # Count sentiment words
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        uncertainty_count = sum(1 for word in words if word in uncertainty_words)
        helping_emotion_count = sum(1 for word in words if word in helping_emotion_words)
        
        # Calculate scores
        total_words = len(words)
        sentiment_score = (positive_count - negative_count) / total_words if total_words > 0 else 0
        uncertainty_score = uncertainty_count / total_words if total_words > 0 else 0
        helping_emotion_score = helping_emotion_count / total_words if total_words > 0 else 0
        
        sentiment_scores.append({
            'doc_idx': doc_idx,
            'sentiment': sentiment_score,
            'uncertainty': uncertainty_score,
            'helping_emotion': helping_emotion_score,
            'positive_words': positive_count,
            'negative_words': negative_count,
            'uncertainty_words': uncertainty_count,
            'helping_words': helping_emotion_count
        })
        
        # Categorize emotional pattern
        if helping_emotion_score > 0.02:
            emotion_pattern = 'helping_focused'
        elif uncertainty_score > 0.03:
            emotion_pattern = 'uncertain_exploratory'
        elif sentiment_score > 0.01:
            emotion_pattern = 'positive_interested'
        elif sentiment_score < -0.01:
            emotion_pattern = 'concerned_hesitant'
        else:
            emotion_pattern = 'neutral_descriptive'
        
        emotion_patterns.append(emotion_pattern)
    
    # Analyze patterns
    pattern_counts = Counter(emotion_patterns)
    avg_sentiment = np.mean([s['sentiment'] for s in sentiment_scores])
    avg_uncertainty = np.mean([s['uncertainty'] for s in sentiment_scores])
    avg_helping = np.mean([s['helping_emotion'] for s in sentiment_scores])
    
    print(f"Emotional Pattern Distribution:")
    for pattern, count in pattern_counts.most_common():
        percentage = (count / len(texts)) * 100
        print(f"  {pattern.replace('_', ' ').title()}: {count} docs ({percentage:.1f}%)")
    
    print(f"\nAverage Emotional Scores:")
    print(f"  Overall sentiment: {avg_sentiment:.4f}")
    print(f"  Uncertainty level: {avg_uncertainty:.4f}")
    print(f"  Helping emotion: {avg_helping:.4f}")
    
    # Find most emotional documents
    most_positive = max(sentiment_scores, key=lambda x: x['sentiment'])
    most_uncertain = max(sentiment_scores, key=lambda x: x['uncertainty'])
    most_helping = max(sentiment_scores, key=lambda x: x['helping_emotion'])
    
    print(f"\nMost Emotional Document Examples:")
    print(f"\nMost Positive (score: {most_positive['sentiment']:.4f}):")
    print(f"  {textwrap.fill(texts[most_positive['doc_idx']][:200], 70)}...")
    
    print(f"\nMost Uncertain (score: {most_uncertain['uncertainty']:.4f}):")
    print(f"  {textwrap.fill(texts[most_uncertain['doc_idx']][:200], 70)}...")
    
    print(f"\nMost Helping-Focused (score: {most_helping['helping_emotion']:.4f}):")
    print(f"  {textwrap.fill(texts[most_helping['doc_idx']][:200], 70)}...")
    
    return {
        'sentiment_scores': sentiment_scores,
        'emotion_patterns': pattern_counts,
        'averages': {'sentiment': avg_sentiment, 'uncertainty': avg_uncertainty, 'helping': avg_helping},
        'interest_score': avg_uncertainty * 100 + avg_helping * 100  # Higher uncertainty and helping = more interesting
    }

def approach_4_keyword_context_analysis(texts):
    """Approach 4: Advanced Keyword-in-Context Analysis"""
    print("\n📝 APPROACH 4: Advanced Keyword-in-Context Analysis")
    print("=" * 55)
    
    key_concepts = {
        'helping_motivation': ['help', 'helping', 'support', 'care', 'assist'],
        'family_influence': ['family', 'parents', 'mom', 'dad', 'relatives'],
        'personal_experience': ['personal', 'personally', 'experience', 'background'],
        'career_consideration': ['career', 'job', 'work', 'field', 'profession'],
        'uncertainty_exploration': ['maybe', 'might', 'could', 'not sure', 'thinking about'],
        'responsibility_weight': ['responsibility', 'important', 'serious', 'pressure'],
        'interest_passion': ['interested', 'interesting', 'passion', 'love', 'enjoy']
    }
    
    context_analysis = {}
    
    for concept, keywords in key_concepts.items():
        contexts = []
        context_window = 6
        
        for text_idx, text in enumerate(texts):
            words = text.split()
            for i, word in enumerate(words):
                if word.lower() in keywords:
                    # Extract context
                    start = max(0, i - context_window)
                    end = min(len(words), i + context_window + 1)
                    context = ' '.join(words[start:end])
                    
                    contexts.append({
                        'keyword': word,
                        'context': context,
                        'doc_idx': text_idx,
                        'position': i
                    })
        
        # Analyze context patterns
        context_words = []
        for ctx in contexts:
            context_words.extend([w.lower() for w in ctx['context'].split() 
                                if len(w) >= 4 and w.lower() not in keywords])
        
        common_context_words = Counter(context_words).most_common(10)
        
        context_analysis[concept] = {
            'total_mentions': len(contexts),
            'contexts': contexts[:5],  # Sample contexts
            'common_context_words': common_context_words
        }
        
        print(f"\n{concept.replace('_', ' ').title()} ({len(contexts)} mentions):")
        print(f"  Common context words: {', '.join([word for word, count in common_context_words[:8]])}")
        
        # Show example contexts
        for i, ctx in enumerate(contexts[:2]):
            print(f"  Example {i+1}: \"{ctx['context']}\"")
    
    # Find co-occurring concepts
    concept_cooccurrence = defaultdict(int)
    for text in texts:
        text_lower = text.lower()
        present_concepts = []
        
        for concept, keywords in key_concepts.items():
            if any(keyword in text_lower for keyword in keywords):
                present_concepts.append(concept)
        
        # Count co-occurrences
        for concept1, concept2 in combinations(present_concepts, 2):
            pair = tuple(sorted([concept1, concept2]))
            concept_cooccurrence[pair] += 1
    
    print(f"\nConcept Co-occurrences (appearing in same document):")
    for (concept1, concept2), count in sorted(concept_cooccurrence.items(), 
                                            key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {concept1} + {concept2}: {count} documents")
    
    return {
        'context_analysis': context_analysis,
        'concept_cooccurrence': concept_cooccurrence,
        'interest_score': sum(len(data['contexts']) for data in context_analysis.values()) / len(key_concepts)
    }

def approach_5_statistical_collocations(texts):
    """Approach 5: Statistical Collocation Analysis"""
    print("\n📈 APPROACH 5: Statistical Collocation Analysis")
    print("=" * 50)
    
    # Extract all word sequences
    all_words = []
    for text in texts:
        words = [word.lower() for word in text.split() 
                if len(word) >= 3 and word.isalpha()]
        all_words.extend(words)
    
    word_freq = Counter(all_words)
    total_words = len(all_words)
    
    # Extract bigrams and calculate statistical significance
    bigrams = []
    bigram_freq = defaultdict(int)
    
    for text in texts:
        words = [word.lower() for word in text.split() 
                if len(word) >= 3 and word.isalpha()]
        
        for i in range(len(words) - 1):
            bigram = (words[i], words[i+1])
            bigrams.append(bigram)
            bigram_freq[bigram] += 1
    
    # Calculate PMI (Pointwise Mutual Information) for statistical significance
    significant_collocations = []
    
    for (word1, word2), bigram_count in bigram_freq.items():
        if bigram_count >= 3:  # Minimum frequency threshold
            # Calculate PMI
            prob_bigram = bigram_count / len(bigrams)
            prob_word1 = word_freq[word1] / total_words
            prob_word2 = word_freq[word2] / total_words
            
            if prob_word1 > 0 and prob_word2 > 0:
                pmi = np.log2(prob_bigram / (prob_word1 * prob_word2))
                
                # Filter for meaningful collocations
                if (pmi > 2.0 and  # Statistically significant
                    bigram_count >= 3 and  # Frequent enough
                    not any(filler in word1 or filler in word2 for filler in 
                           ['like', 'just', 'really', 'kind', 'sort', 'very', 'pretty'])):
                    
                    significant_collocations.append({
                        'words': (word1, word2),
                        'frequency': bigram_count,
                        'pmi': pmi,
                        'phrase': f"{word1} {word2}"
                    })
    
    # Sort by PMI score
    significant_collocations.sort(key=lambda x: x['pmi'], reverse=True)
    
    print(f"Statistically Significant Collocations (PMI > 2.0):")
    for colloc in significant_collocations[:20]:
        print(f"  '{colloc['phrase']}': freq={colloc['frequency']}, PMI={colloc['pmi']:.2f}")
    
    # Extract meaningful trigrams
    trigram_freq = defaultdict(int)
    for text in texts:
        words = [word.lower() for word in text.split() 
                if len(word) >= 3 and word.isalpha()]
        
        for i in range(len(words) - 2):
            trigram = (words[i], words[i+1], words[i+2])
            trigram_freq[trigram] += 1
    
    meaningful_trigrams = [(trigram, count) for trigram, count in trigram_freq.items() 
                          if count >= 2 and 
                          not any(filler in trigram for filler in 
                                 ['like', 'just', 'really', 'kind', 'sort'])]
    
    meaningful_trigrams.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nMeaningful Trigrams (≥2 occurrences):")
    for trigram, count in meaningful_trigrams[:15]:
        phrase = ' '.join(trigram)
        print(f"  '{phrase}': {count} times")
    
    return {
        'significant_collocations': significant_collocations,
        'meaningful_trigrams': meaningful_trigrams,
        'interest_score': len([c for c in significant_collocations if any(keyword in c['phrase'] for keyword in ['help', 'family', 'personal', 'career'])])
    }

def approach_6_ngram_frequency_analysis(texts):
    """Approach 6: Comprehensive N-gram Frequency Analysis"""
    print("\n🔤 APPROACH 6: Comprehensive N-gram Frequency Analysis")
    print("=" * 58)
    
    # Clean words for n-gram analysis
    def get_clean_words(text):
        return [word.lower() for word in text.split() 
                if len(word) >= 3 and word.isalpha() and 
                word not in {'like', 'just', 'really', 'kind', 'sort', 'very', 'pretty'}]
    
    # Extract n-grams of different lengths
    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()
    fourgrams = Counter()
    
    for text in texts:
        words = get_clean_words(text)
        
        # Unigrams
        unigrams.update(words)
        
        # Bigrams
        for i in range(len(words) - 1):
            bigrams[' '.join(words[i:i+2])] += 1
        
        # Trigrams
        for i in range(len(words) - 2):
            trigrams[' '.join(words[i:i+3])] += 1
        
        # 4-grams
        for i in range(len(words) - 3):
            fourgrams[' '.join(words[i:i+4])] += 1
    
    # Filter for meaningful n-grams
    meaningful_unigrams = [(word, count) for word, count in unigrams.most_common(30) 
                          if count >= 10]
    
    meaningful_bigrams = [(phrase, count) for phrase, count in bigrams.most_common(50) 
                         if count >= 3]
    
    meaningful_trigrams = [(phrase, count) for phrase, count in trigrams.most_common(30) 
                          if count >= 2]
    
    meaningful_fourgrams = [(phrase, count) for phrase, count in fourgrams.most_common(20) 
                           if count >= 2]
    
    print(f"Top Meaningful Unigrams:")
    for word, count in meaningful_unigrams[:15]:
        print(f"  {word}: {count}")
    
    print(f"\nTop Meaningful Bigrams:")
    for phrase, count in meaningful_bigrams[:15]:
        print(f"  '{phrase}': {count}")
    
    print(f"\nTop Meaningful Trigrams:")
    for phrase, count in meaningful_trigrams[:10]:
        print(f"  '{phrase}': {count}")
    
    if meaningful_fourgrams:
        print(f"\nMeaningful 4-grams:")
        for phrase, count in meaningful_fourgrams[:8]:
            print(f"  '{phrase}': {count}")
    
    # Calculate diversity metrics
    total_unique_unigrams = len([w for w, c in unigrams.items() if c >= 2])
    total_unique_bigrams = len([p for p, c in bigrams.items() if c >= 2])
    
    print(f"\nN-gram Diversity:")
    print(f"  Unique unigrams (≥2 freq): {total_unique_unigrams}")
    print(f"  Unique bigrams (≥2 freq): {total_unique_bigrams}")
    print(f"  Vocabulary richness: {total_unique_unigrams / len(texts):.2f} unique words per document")
    
    return {
        'unigrams': meaningful_unigrams,
        'bigrams': meaningful_bigrams,
        'trigrams': meaningful_trigrams,
        'fourgrams': meaningful_fourgrams,
        'diversity': {'unigrams': total_unique_unigrams, 'bigrams': total_unique_bigrams},
        'interest_score': len(meaningful_trigrams) + len(meaningful_fourgrams)
    }

def approach_7_semantic_clustering(texts):
    """Approach 7: Simple Semantic Clustering"""
    print("\n🧠 APPROACH 7: Simple Semantic Clustering")
    print("=" * 47)
    
    # Create TF-IDF representation
    vectorizer = TfidfVectorizer(
        max_features=200,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Perform k-means clustering
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    
    # Analyze clusters
    cluster_analysis = {}
    for cluster_id in range(n_clusters):
        cluster_docs = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        
        if len(cluster_docs) == 0:
            continue
        
        # Get representative terms for this cluster
        cluster_center = kmeans.cluster_centers_[cluster_id]
        top_indices = cluster_center.argsort()[-10:][::-1]
        top_terms = [feature_names[i] for i in top_indices]
        
        # Get most representative document
        cluster_tfidf = tfidf_matrix[cluster_docs]
        similarities = cosine_similarity(cluster_tfidf, cluster_center.reshape(1, -1))
        most_representative_idx = cluster_docs[np.argmax(similarities)]
        
        cluster_analysis[cluster_id] = {
            'size': len(cluster_docs),
            'percentage': (len(cluster_docs) / len(texts)) * 100,
            'top_terms': top_terms,
            'representative_doc': texts[most_representative_idx][:200] + '...',
            'doc_indices': cluster_docs
        }
    
    # Sort clusters by size
    sorted_clusters = sorted(cluster_analysis.items(), key=lambda x: x[1]['size'], reverse=True)
    
    print(f"Semantic Clusters (K-means with k={n_clusters}):")
    for cluster_id, data in sorted_clusters:
        print(f"\nCluster {cluster_id + 1}: {data['size']} docs ({data['percentage']:.1f}%)")
        print(f"  Key terms: {', '.join(data['top_terms'][:6])}")
        print(f"  Representative text: {textwrap.fill(data['representative_doc'], 70)}")
    
    # Calculate cluster quality (silhouette-like measure)
    intra_cluster_similarity = 0
    for cluster_id, data in cluster_analysis.items():
        if len(data['doc_indices']) > 1:
            cluster_docs_tfidf = tfidf_matrix[data['doc_indices']]
            similarities = cosine_similarity(cluster_docs_tfidf)
            avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
            intra_cluster_similarity += avg_similarity * data['size']
    
    avg_cluster_quality = intra_cluster_similarity / len(texts)
    
    print(f"\nCluster Quality Score: {avg_cluster_quality:.3f}")
    
    return {
        'clusters': cluster_analysis,
        'cluster_quality': avg_cluster_quality,
        'interest_score': avg_cluster_quality * 100 + len([c for c in cluster_analysis.values() if c['size'] >= 10])
    }

def compare_all_approaches(results):
    """Compare all approaches and determine most interesting"""
    print("\n" + "="*70)
    print("🏆 COMPREHENSIVE APPROACH COMPARISON")
    print("="*70)
    
    approaches = [
        ("TF-IDF Distinctive Terms", results['tfidf']),
        ("Co-occurrence Networks", results['cooccurrence']),
        ("Sentiment & Emotion", results['sentiment']),
        ("Keyword-in-Context", results['keyword_context']),
        ("Statistical Collocations", results['collocations']),
        ("N-gram Frequency", results['ngrams']),
        ("Semantic Clustering", results['clustering'])
    ]
    
    print("\nInterest Scores (higher = more interesting patterns):")
    ranked_approaches = sorted(approaches, key=lambda x: x[1]['interest_score'], reverse=True)
    
    for rank, (name, data) in enumerate(ranked_approaches, 1):
        print(f"  {rank}. {name}: {data['interest_score']:.2f}")
    
    # Determine most robust and interesting
    top_approach = ranked_approaches[0]
    
    print(f"\n🎯 MOST INTERESTING APPROACH: {top_approach[0]}")
    print(f"Score: {top_approach[1]['interest_score']:.2f}")
    
    # Provide specific insights about why this approach is best
    if "Sentiment" in top_approach[0]:
        print("\nWhy this is most interesting:")
        print("• Captures emotional dimensions topic modeling missed")
        print("• Uncertainty patterns validate Study 1 findings")
        print("• Helping emotions show intrinsic motivation")
        
    elif "Keyword" in top_approach[0]:
        print("\nWhy this is most interesting:")
        print("• Shows natural language contexts around key concepts")
        print("• Reveals how concepts co-occur in student speech")
        print("• Provides quotable examples for paper")
        
    elif "N-gram" in top_approach[0]:
        print("\nWhy this is most interesting:")
        print("• Captures actual phrases students use")
        print("• More stable than topic modeling")
        print("• Easy to interpret and validate")
    
    print(f"\n📊 RECOMMENDATION:")
    print(f"Use {top_approach[0]} as primary analysis")
    print(f"Supplement with {ranked_approaches[1][0]} for validation")
    
    return ranked_approaches

def main():
    """Execute comprehensive analysis comparison"""
    
    print("🔬 COMPREHENSIVE ANALYSIS COMPARISON")
    print("=" * 60)
    print("Testing 7 different approaches to find most interesting patterns")
    
    # Load and clean data
    texts, corpus_df = load_focus_group_data()
    cleaned_texts = clean_texts_comprehensive(texts)
    
    # Run all approaches
    results = {}
    
    results['tfidf'] = approach_1_tfidf_distinctive_terms(cleaned_texts)
    results['cooccurrence'] = approach_2_cooccurrence_networks(cleaned_texts)
    results['sentiment'] = approach_3_sentiment_emotion_analysis(cleaned_texts)
    results['keyword_context'] = approach_4_keyword_context_analysis(cleaned_texts)
    results['collocations'] = approach_5_statistical_collocations(cleaned_texts)
    results['ngrams'] = approach_6_ngram_frequency_analysis(cleaned_texts)
    results['clustering'] = approach_7_semantic_clustering(cleaned_texts)
    
    # Compare and rank approaches
    ranked_approaches = compare_all_approaches(results)
    
    print(f"\n🏆 ANALYSIS COMPLETE!")
    print(f"🎯 Best approach identified: {ranked_approaches[0][0]}")
    print("📈 Ready to implement most promising method for final analysis")
    
    return results, ranked_approaches

if __name__ == "__main__":
    results, rankings = main()