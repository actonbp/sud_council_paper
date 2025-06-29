#!/usr/bin/env python3
"""
spaCy Enhanced Topic Modeling
Modern approach using spaCy's linguistic features for robust topic discovery
- Parts of speech tagging for meaningful words
- Noun phrase extraction for concepts
- Named entity recognition for context
- Lemmatization for word normalization
- Dependency parsing for relationships
"""

import glob, os, re, textwrap
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
DATA_DIR = "../../data"
RESULTS_DIR = "../../results/"

# Load spaCy model (install with: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Installing spaCy English model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def load_focus_group_data():
    """Load focus group data"""
    print("📥 Loading focus group data for spaCy analysis...")
    
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

def extract_spacy_features(texts):
    """Extract linguistic features using spaCy"""
    print("\n🧠 Extracting spaCy linguistic features...")
    
    # Features to extract
    meaningful_tokens = []
    noun_phrases = []
    named_entities = []
    action_verbs = []
    descriptive_adjectives = []
    doc_features = []
    
    # Domain terms to strategically filter
    DOMAIN_FILTERS = {
        'counselor', 'counselors', 'counseling', 'therapist', 'therapists', 'therapy',
        'psychologist', 'psychiatrist', 'social_work'
    }
    
    print("Processing documents with spaCy...")
    for doc_id, text in enumerate(texts):
        if doc_id % 50 == 0:
            print(f"  Processed {doc_id}/{len(texts)} documents...")
            
        # Process with spaCy
        doc = nlp(text)
        
        # Extract different feature types
        doc_tokens = []
        doc_noun_phrases = []
        doc_entities = []
        doc_verbs = []
        doc_adjectives = []
        
        # 1. Meaningful tokens (lemmatized, filtered by POS)
        for token in doc:
            # Skip if filtered domain term
            if token.lemma_.lower() in DOMAIN_FILTERS:
                continue
                
            # Extract meaningful tokens based on POS and properties
            if (token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and
                not token.is_stop and
                not token.is_punct and
                not token.is_space and
                len(token.lemma_) >= 3 and
                token.lemma_.isalpha()):
                
                doc_tokens.append(token.lemma_.lower())
                
                # Categorize by POS
                if token.pos_ == 'VERB' and token.lemma_ not in ['be', 'have', 'do', 'say', 'get', 'go', 'know', 'think']:
                    doc_verbs.append(token.lemma_.lower())
                elif token.pos_ == 'ADJ' and len(token.lemma_) >= 4:
                    doc_adjectives.append(token.lemma_.lower())
        
        # 2. Noun phrases (meaningful concepts)
        for chunk in doc.noun_chunks:
            # Clean and filter noun phrases
            phrase_text = chunk.text.lower().strip()
            phrase_lemma = " ".join([token.lemma_.lower() for token in chunk 
                                   if not token.is_stop and token.is_alpha])
            
            # Skip if contains domain terms or too short
            if (any(term in phrase_lemma for term in DOMAIN_FILTERS) or
                len(phrase_lemma.split()) < 2 or
                len(phrase_lemma) < 6):
                continue
                
            # Keep meaningful noun phrases
            if any(keyword in phrase_lemma for keyword in [
                'family', 'personal', 'experience', 'background', 'career', 'path',
                'helping', 'people', 'support', 'responsibility', 'interest',
                'education', 'school', 'field', 'work', 'job', 'future'
            ]):
                doc_noun_phrases.append(phrase_lemma.replace(' ', '_'))
        
        # 3. Named entities (persons, organizations, etc.)
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT'] and len(ent.text) >= 3:
                doc_entities.append(f"{ent.text.lower()}_{ent.label_.lower()}")
        
        # Store document features
        doc_features.append({
            'doc_id': doc_id,
            'tokens': doc_tokens,
            'noun_phrases': doc_noun_phrases,
            'entities': doc_entities,
            'verbs': doc_verbs,
            'adjectives': doc_adjectives,
            'total_features': len(doc_tokens) + len(doc_noun_phrases)
        })
        
        # Aggregate across corpus
        meaningful_tokens.extend(doc_tokens)
        noun_phrases.extend(doc_noun_phrases)
        named_entities.extend(doc_entities)
        action_verbs.extend(doc_verbs)
        descriptive_adjectives.extend(doc_adjectives)
    
    print(f"\nspaCy Feature Extraction Complete:")
    print(f"  • Meaningful tokens: {len(set(meaningful_tokens))}")
    print(f"  • Noun phrases: {len(set(noun_phrases))}")
    print(f"  • Named entities: {len(set(named_entities))}")
    print(f"  • Action verbs: {len(set(action_verbs))}")
    print(f"  • Descriptive adjectives: {len(set(descriptive_adjectives))}")
    
    return doc_features, {
        'tokens': meaningful_tokens,
        'noun_phrases': noun_phrases,
        'entities': named_entities,
        'verbs': action_verbs,
        'adjectives': descriptive_adjectives
    }

def create_spacy_feature_matrix(doc_features, corpus_features):
    """Create weighted feature matrix from spaCy features"""
    print("\n📊 Creating spaCy-enhanced feature matrix...")
    
    # Get most frequent and meaningful features
    token_counts = Counter(corpus_features['tokens'])
    phrase_counts = Counter(corpus_features['noun_phrases'])
    verb_counts = Counter(corpus_features['verbs'])
    adj_counts = Counter(corpus_features['adjectives'])
    
    # Select features with minimum frequency
    selected_tokens = [token for token, count in token_counts.items() if count >= 3]
    selected_phrases = [phrase for phrase, count in phrase_counts.items() if count >= 2]
    selected_verbs = [verb for verb, count in verb_counts.items() if count >= 2]
    selected_adjectives = [adj for adj, count in adj_counts.items() if count >= 2]
    
    # Create vocabulary with weights
    vocab_weights = {}
    
    # Single tokens (base weight = 1)
    for token in selected_tokens:
        vocab_weights[token] = 1
    
    # Noun phrases (higher weight = 3)
    for phrase in selected_phrases:
        vocab_weights[phrase] = 3
        
    # Action verbs (moderate weight = 2)
    for verb in selected_verbs:
        vocab_weights[f"action_{verb}"] = 2
        
    # Descriptive adjectives (moderate weight = 2)
    for adj in selected_adjectives:
        vocab_weights[f"desc_{adj}"] = 2
    
    print(f"Selected features:")
    print(f"  • Tokens: {len(selected_tokens)}")
    print(f"  • Noun phrases: {len(selected_phrases)}")
    print(f"  • Action verbs: {len(selected_verbs)}")
    print(f"  • Adjectives: {len(selected_adjectives)}")
    print(f"  • Total vocabulary: {len(vocab_weights)}")
    
    # Create document representations
    documents = []
    for doc_data in doc_features:
        doc_terms = []
        
        # Add tokens
        for token in doc_data['tokens']:
            if token in selected_tokens:
                doc_terms.extend([token] * vocab_weights[token])
        
        # Add noun phrases (weighted)
        for phrase in doc_data['noun_phrases']:
            if phrase in selected_phrases:
                doc_terms.extend([phrase] * vocab_weights[phrase])
        
        # Add action verbs (weighted)
        for verb in doc_data['verbs']:
            verb_key = f"action_{verb}"
            if verb_key in vocab_weights:
                doc_terms.extend([verb_key] * vocab_weights[verb_key])
        
        # Add adjectives (weighted)
        for adj in doc_data['adjectives']:
            adj_key = f"desc_{adj}"
            if adj_key in vocab_weights:
                doc_terms.extend([adj_key] * vocab_weights[adj_key])
        
        documents.append(" ".join(doc_terms))
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(
        min_df=2,
        max_df=0.85,
        max_features=150,
        ngram_range=(1, 1),  # spaCy already handled phrases
        sublinear_tf=True,
        norm='l2'
    )
    
    X = vectorizer.fit_transform(documents)
    vocab = vectorizer.get_feature_names_out()
    
    print(f"Feature matrix: {X.shape[0]} docs × {X.shape[1]} features")
    
    return X, vocab, vectorizer, documents

def run_spacy_topic_model(X, vocab, n_topics=4):
    """Run LDA topic model on spaCy features"""
    print(f"\n🎯 Running spaCy-enhanced topic model (k={n_topics})...")
    
    # LDA with spaCy features
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=1000,
        learning_method="batch",
        doc_topic_prior=0.1,
        topic_word_prior=0.01,
        random_state=42
    )
    
    doc_topic = lda.fit_transform(X)
    
    # Analyze topic distribution
    assignments = doc_topic.argmax(axis=1)
    topic_counts = Counter(assignments)
    
    print(f"Topic distribution: {dict(topic_counts)}")
    
    return lda, doc_topic

def interpret_spacy_topics(lda, vocab, doc_topic):
    """Interpret topics with spaCy-enhanced understanding"""
    print("\n🔍 Interpreting spaCy-enhanced topics...")
    
    topics_data = []
    
    for topic_idx in range(lda.n_components):
        # Get top terms
        top_indices = lda.components_[topic_idx].argsort()[-20:][::-1]
        top_terms = [vocab[i] for i in top_indices]
        
        # Categorize terms by type
        noun_phrases = [term for term in top_terms if "_" in term and not term.startswith(('action_', 'desc_'))]
        action_verbs = [term.replace('action_', '') for term in top_terms if term.startswith('action_')]
        adjectives = [term.replace('desc_', '') for term in top_terms if term.startswith('desc_')]
        single_tokens = [term for term in top_terms if "_" not in term and not term.startswith(('action_', 'desc_'))]
        
        # Document count and percentage
        assignments = doc_topic.argmax(axis=1)
        doc_count = np.sum(assignments == topic_idx)
        percentage = (doc_count / len(doc_topic)) * 100
        
        # Interpret based on linguistic patterns
        interpretation = interpret_linguistic_patterns(noun_phrases, action_verbs, adjectives, single_tokens)
        
        topics_data.append({
            'topic_num': topic_idx + 1,
            'interpretation': interpretation,
            'doc_count': doc_count,
            'percentage': percentage,
            'noun_phrases': noun_phrases[:4],
            'action_verbs': action_verbs[:4],
            'adjectives': adjectives[:4],
            'key_tokens': single_tokens[:6],
            'all_terms': top_terms[:12]
        })
        
        print(f"\nTopic {topic_idx + 1}: {interpretation}")
        print(f"  Documents: {doc_count} ({percentage:.1f}%)")
        if noun_phrases:
            print(f"  Noun phrases: {', '.join(noun_phrases[:3])}")
        if action_verbs:
            print(f"  Action verbs: {', '.join(action_verbs[:3])}")
        if adjectives:
            print(f"  Adjectives: {', '.join(adjectives[:3])}")
        print(f"  Key tokens: {', '.join(single_tokens[:5])}")
    
    return topics_data

def interpret_linguistic_patterns(noun_phrases, action_verbs, adjectives, tokens):
    """Interpret topic based on linguistic patterns"""
    
    # Combine all terms for analysis
    all_content = ' '.join(noun_phrases + action_verbs + adjectives + tokens).lower()
    
    # Pattern-based interpretation
    if any(term in all_content for term in ['family', 'personal', 'experience', 'background', 'parent']):
        if any(term in all_content for term in ['help', 'support', 'care']):
            return "Personal & Family Experience Driving Service"
        else:
            return "Personal & Family Background Influence"
    
    elif any(term in all_content for term in ['help', 'support', 'people', 'care', 'assist']):
        if any(term in all_content for term in ['responsibility', 'important', 'meaningful']):
            return "Service-Oriented Helping with Responsibility"
        else:
            return "People-Focused Helping & Support"
    
    elif any(term in all_content for term in ['career', 'path', 'field', 'professional', 'work', 'job']):
        if any(term in all_content for term in ['interest', 'consider', 'think', 'future']):
            return "Career Exploration & Professional Interest"
        else:
            return "Professional Career Development"
    
    elif any(term in all_content for term in ['school', 'education', 'learn', 'study', 'knowledge']):
        return "Educational Pathway & Academic Preparation"
    
    else:
        return "Mixed Motivational Factors"

def save_spacy_results(topics_data, corpus_df, doc_topic, vocab):
    """Save spaCy-enhanced topic modeling results"""
    
    # Document assignments
    assignments = doc_topic.argmax(axis=1)
    confidences = doc_topic.max(axis=1)
    
    # Create results DataFrame
    results_data = []
    for doc_idx, (assignment, confidence) in enumerate(zip(assignments, confidences)):
        topic_data = topics_data[assignment]
        
        results_data.append({
            'document_id': doc_idx,
            'topic_num': topic_data['topic_num'],
            'topic_interpretation': topic_data['interpretation'],
            'confidence': round(confidence, 3),
            'speaker': corpus_df.iloc[doc_idx]['Speaker'],
            'session': corpus_df.iloc[doc_idx]['session'],
            'text_preview': corpus_df.iloc[doc_idx]['Text'][:200] + '...' if len(corpus_df.iloc[doc_idx]['Text']) > 200 else corpus_df.iloc[doc_idx]['Text']
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(RESULTS_DIR, "spacy_enhanced_topic_assignments.csv"), index=False)
    
    # Topic summary
    summary_data = []
    for topic_data in topics_data:
        summary_data.append({
            'Topic': f"Topic {topic_data['topic_num']}: {topic_data['interpretation']}",
            'Documents': topic_data['doc_count'],
            'Percentage': f"{topic_data['percentage']:.1f}%",
            'Noun_Phrases': ', '.join(topic_data['noun_phrases'][:3]) if topic_data['noun_phrases'] else 'None',
            'Action_Verbs': ', '.join(topic_data['action_verbs'][:3]) if topic_data['action_verbs'] else 'None',
            'Adjectives': ', '.join(topic_data['adjectives'][:3]) if topic_data['adjectives'] else 'None',
            'Key_Tokens': ', '.join(topic_data['key_tokens'][:5])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "spacy_enhanced_topic_summary.csv"), index=False)
    
    # Comprehensive report
    with open(os.path.join(RESULTS_DIR, "spacy_enhanced_topic_report.txt"), 'w') as f:
        f.write("SPACY-ENHANCED TOPIC MODELING - SUD COUNSELING RESEARCH\n")
        f.write("=" * 65 + "\n\n")
        
        f.write("METHODOLOGY - spaCy Linguistic Analysis:\n")
        f.write("• Parts-of-speech tagging for meaningful words\n")
        f.write("• Noun phrase extraction for concepts\n")
        f.write("• Named entity recognition for context\n")
        f.write("• Lemmatization for word normalization\n")
        f.write("• Strategic domain filtering\n")
        f.write("• Weighted feature matrix (phrases=3x, verbs/adj=2x)\n\n")
        
        f.write("LINGUISTIC TOPIC ANALYSIS RESULTS:\n")
        f.write("-" * 35 + "\n")
        
        for topic_data in topics_data:
            f.write(f"\n{topic_data['interpretation']} ({topic_data['percentage']:.1f}%)\n")
            f.write(f"  Documents: {topic_data['doc_count']}\n")
            if topic_data['noun_phrases']:
                f.write(f"  Key concepts: {', '.join(topic_data['noun_phrases'][:4])}\n")
            if topic_data['action_verbs']:
                f.write(f"  Actions: {', '.join(topic_data['action_verbs'][:4])}\n")
            if topic_data['adjectives']:
                f.write(f"  Descriptors: {', '.join(topic_data['adjectives'][:4])}\n")
            f.write(f"  Core tokens: {', '.join(topic_data['key_tokens'][:5])}\n")
    
    print(f"\n✅ spaCy-enhanced results saved to {RESULTS_DIR}")
    
    # Display sample quotes
    print("\n" + "="*70)
    print("SPACY-ENHANCED TOPICS - REPRESENTATIVE QUOTES")
    print("="*70)
    
    for topic_data in topics_data:
        topic_idx = topic_data['topic_num'] - 1
        topic_probs = doc_topic[:, topic_idx]
        best_doc_idx = topic_probs.argmax()
        best_prob = topic_probs[best_doc_idx]
        best_text = corpus_df.iloc[best_doc_idx]['Text']
        
        print(f"\n>>> {topic_data['interpretation']} (P={best_prob:.3f}) <<<")
        if topic_data['noun_phrases']:
            print(f"Key concepts: {', '.join(topic_data['noun_phrases'][:3])}")
        print(f"Quote: {textwrap.fill(best_text, 70)}")
    
    return topics_data

def run_robustness_checks(X, vocab, n_topics=4, n_runs=5):
    """Run robustness checks on spaCy-enhanced model"""
    print(f"\n🔄 Running robustness checks ({n_runs} runs)...")
    
    all_top_terms = []
    coherence_scores = []
    
    for run in range(n_runs):
        # Run with different random seeds
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=1000,
            learning_method="batch",
            doc_topic_prior=0.1,
            topic_word_prior=0.01,
            random_state=42 + run
        )
        
        lda.fit(X)
        
        # Extract top terms for each topic
        run_terms = []
        for topic_idx in range(n_topics):
            top_indices = lda.components_[topic_idx].argsort()[-10:][::-1]
            top_terms = [vocab[i] for i in top_indices]
            run_terms.append(set(top_terms))
        
        all_top_terms.append(run_terms)
        
        # Calculate perplexity as coherence proxy
        coherence_scores.append(-lda.score(X))
    
    # Calculate stability (Jaccard similarity across runs)
    stability_scores = []
    for topic_idx in range(n_topics):
        topic_similarities = []
        for i in range(n_runs):
            for j in range(i+1, n_runs):
                set1 = all_top_terms[i][topic_idx]
                set2 = all_top_terms[j][topic_idx]
                jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
                topic_similarities.append(jaccard)
        stability_scores.append(np.mean(topic_similarities))
    
    avg_stability = np.mean(stability_scores)
    avg_coherence = np.mean(coherence_scores)
    
    print(f"Robustness Results:")
    print(f"  • Average stability (Jaccard): {avg_stability:.3f}")
    print(f"  • Average coherence (perplexity): {avg_coherence:.1f}")
    print(f"  • Topic-wise stability: {[f'{s:.3f}' for s in stability_scores]}")
    
    # Robustness interpretation
    if avg_stability >= 0.6:
        stability_rating = "EXCELLENT"
    elif avg_stability >= 0.4:
        stability_rating = "GOOD"
    elif avg_stability >= 0.3:
        stability_rating = "MODERATE"
    else:
        stability_rating = "POOR"
    
    print(f"  • Overall stability: {stability_rating}")
    
    return {
        'stability': avg_stability,
        'coherence': avg_coherence,
        'topic_stability': stability_scores,
        'rating': stability_rating
    }

def main():
    """Execute spaCy-enhanced topic modeling with robustness checks"""
    
    print("🧠 SPACY-ENHANCED TOPIC MODELING")
    print("=" * 50)
    print("Modern NLP approach using linguistic features")
    
    # Load data
    texts, corpus_df = load_focus_group_data()
    
    # Extract spaCy features
    doc_features, corpus_features = extract_spacy_features(texts)
    
    # Create feature matrix
    X, vocab, vectorizer, documents = create_spacy_feature_matrix(doc_features, corpus_features)
    
    # Run topic model
    lda, doc_topic = run_spacy_topic_model(X, vocab, n_topics=4)
    
    # Interpret topics
    topics_data = interpret_spacy_topics(lda, vocab, doc_topic)
    
    # Run robustness checks
    robustness = run_robustness_checks(X, vocab, n_topics=4, n_runs=5)
    
    # Save results
    save_spacy_results(topics_data, corpus_df, doc_topic, vocab)
    
    print(f"\n🏆 SPACY-ENHANCED ANALYSIS COMPLETE!")
    print(f"🧠 Used advanced linguistic features")
    print(f"📊 {len(topics_data)} interpretable topics discovered")
    print(f"🔄 Robustness: {robustness['rating']} (stability={robustness['stability']:.3f})")
    print("✨ Ready for Study 1 individual linkage!")
    
    return topics_data, robustness

if __name__ == "__main__":
    topics, robustness = main()