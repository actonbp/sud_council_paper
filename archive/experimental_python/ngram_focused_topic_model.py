#!/usr/bin/env python3
"""
N-gram Focused Topic Modeling
Prioritizes meaningful multi-word phrases over single words
"""

import glob, os, re, textwrap
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser

# Configuration
DATA_DIR = "../../data"
RESULTS_DIR = "../../results/"

def load_focus_group_data():
    """Load focus group data"""
    paths = glob.glob(os.path.join(DATA_DIR, "*_Focus_Group_full*.csv"))
    
    rows = []
    for p in paths:
        df = pd.read_csv(p)
        # Remove moderator utterances and short responses
        df = df[~df["Speaker"].astype(str).str.match(r"^[A-Z]{2,3}$")]
        df = df[df["Text"].str.split().str.len() >= 8]
        df["session"] = os.path.basename(p)
        rows.append(df)
    
    corpus_df = pd.concat(rows, ignore_index=True)
    print(f"Loaded {len(corpus_df)} substantive utterances from {len(corpus_df['session'].unique())} sessions")
    
    return corpus_df

def detect_meaningful_phrases(texts):
    """Use Gensim to detect meaningful n-gram phrases"""
    print("\n>>> Detecting meaningful phrases with Gensim...")
    
    # Remove only the most circular terms
    CIRCULAR_TERMS = ['counselor', 'counseling', 'therapist', 'therapy']
    
    # Basic stop words
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
    """.split())
    
    # Clean and tokenize texts for phrase detection
    cleaned_sentences = []
    for text in texts:
        text = text.lower()
        
        # Remove circular terms
        for term in CIRCULAR_TERMS:
            text = text.replace(term, ' ')
        
        # Tokenize and filter
        tokens = simple_preprocess(text, deacc=True, min_len=3, max_len=20)
        meaningful_tokens = [token for token in tokens 
                           if token not in STOP_WORDS and len(token) >= 3 and not token.isdigit()]
        
        if len(meaningful_tokens) >= 5:  # Need sufficient content
            cleaned_sentences.append(meaningful_tokens)
    
    print(f"Prepared {len(cleaned_sentences)} documents for phrase detection")
    
    # Multi-level phrase detection with correct NPMI thresholds
    print("   Detecting bigrams...")
    bigram_model = Phrases(cleaned_sentences, min_count=3, threshold=0.3, delimiter="_", scoring='npmi')
    bigram_phraser = Phraser(bigram_model)
    bigram_sentences = [bigram_phraser[sent] for sent in cleaned_sentences]
    
    print("   Detecting trigrams...")
    trigram_model = Phrases(bigram_sentences, min_count=2, threshold=0.2, delimiter="_", scoring='npmi')
    trigram_phraser = Phraser(trigram_model)
    trigram_sentences = [trigram_phraser[sent] for sent in bigram_sentences]
    
    print("   Detecting 4-grams...")
    fourgram_model = Phrases(trigram_sentences, min_count=2, threshold=0.1, delimiter="_", scoring='npmi')
    fourgram_phraser = Phraser(fourgram_model)
    final_sentences = [fourgram_phraser[sent] for sent in trigram_sentences]
    
    # Extract all discovered phrases
    all_phrases = set()
    for sent in final_sentences:
        for token in sent:
            if "_" in token:  # Multi-word phrase
                all_phrases.add(token)
    
    # Categorize phrases by length
    bigrams = [p for p in all_phrases if len(p.split("_")) == 2]
    trigrams = [p for p in all_phrases if len(p.split("_")) == 3]
    fourgrams = [p for p in all_phrases if len(p.split("_")) >= 4]
    
    print(f"   Discovered {len(bigrams)} meaningful bigrams")
    print(f"   Discovered {len(trigrams)} meaningful trigrams")
    print(f"   Discovered {len(fourgrams)} meaningful 4+ grams")
    print(f"   Sample phrases: {sorted(list(all_phrases))[:15]}")
    
    # Convert back to text documents with phrases
    phrase_docs = []
    for sent in final_sentences:
        phrase_docs.append(" ".join(sent))
    
    return phrase_docs, all_phrases

def create_ngram_weighted_features(docs, discovered_phrases):
    """Create feature matrix that heavily weights n-grams"""
    print("\n>>> Creating n-gram weighted features...")
    
    def ngram_weighted_analyzer(text):
        """Custom analyzer that weights n-grams much higher than single words"""
        tokens = text.split()
        features = []
        
        for token in tokens:
            if "_" in token:  # Multi-word phrase
                n_words = len(token.split("_"))
                if n_words == 2:
                    weight = 3  # Bigrams get 3x weight
                elif n_words == 3:
                    weight = 5  # Trigrams get 5x weight
                elif n_words >= 4:
                    weight = 7  # 4+ grams get 7x weight
                else:
                    weight = 1
                
                # Add the phrase multiple times based on weight
                features.extend([token] * weight)
            else:
                # Single words get added once (lower priority)
                if len(token) >= 4:  # Only substantial single words
                    features.append(token)
        
        return features
    
    # Use TF-IDF with custom analyzer
    vectorizer = TfidfVectorizer(
        analyzer=ngram_weighted_analyzer,
        min_df=2,                    # Lower threshold for phrases
        max_df=0.8,                  # Allow more common phrases
        max_features=250,            # Reasonable vocabulary size
        sublinear_tf=True,
        norm='l2'
    )
    
    X = vectorizer.fit_transform(docs)
    vocab = vectorizer.get_feature_names_out()
    
    # Analyze vocabulary composition
    phrases = [term for term in vocab if "_" in term]
    single_words = [term for term in vocab if "_" not in term]
    
    print(f"Feature matrix: {X.shape[0]} docs × {X.shape[1]} features")
    print(f"  - {len(phrases)} meaningful phrases ({len(phrases)/len(vocab):.1%})")
    print(f"  - {len(single_words)} single words ({len(single_words)/len(vocab):.1%})")
    print(f"Sample phrases in vocabulary: {phrases[:10]}")
    print(f"Sample single words: {single_words[:10]}")
    
    return X, vocab, vectorizer

def run_ngram_topic_model(X, vocab):
    """Run topic model optimized for n-gram interpretation"""
    print("\n>>> Running n-gram focused topic modeling...")
    
    # Test different k values with focus on phrase interpretability
    best_score = -1
    best_k = 3
    best_model = None
    
    for k in [3, 4, 5]:
        print(f"Testing k={k}...")
        
        lda = LatentDirichletAllocation(
            n_components=k,
            max_iter=500,
            learning_method="batch",
            doc_topic_prior=0.1,     # Encourage focused topics
            topic_word_prior=0.01,   # Encourage focused vocabulary
            random_state=42
        )
        
        doc_topic = lda.fit_transform(X)
        
        # Score based on phrase prominence and topic separation
        phrase_score = 0
        for topic_idx in range(k):
            top_indices = lda.components_[topic_idx].argsort()[-15:][::-1]
            top_terms = [vocab[i] for i in top_indices]
            
            # Count meaningful phrases in top terms
            phrases_in_top = sum(1 for term in top_terms[:10] if "_" in term)
            phrase_score += phrases_in_top
        
        # Normalize phrase score
        phrase_score = phrase_score / k
        
        # Topic balance
        assignments = doc_topic.argmax(axis=1)
        topic_counts = [np.sum(assignments == i) for i in range(k)]
        balance = 1 - (np.std(topic_counts) / np.mean(topic_counts))
        
        # Combined score favoring phrase-rich topics
        combined_score = phrase_score * 0.7 + balance * 0.3
        
        print(f"  Phrase score: {phrase_score:.2f}, Balance: {balance:.3f}, Combined: {combined_score:.3f}")
        
        if combined_score > best_score:
            best_score = combined_score
            best_k = k
            best_model = lda
    
    print(f"\n>>> Optimal configuration: k={best_k}, score={best_score:.3f}")
    
    # Fit final model
    doc_topic = best_model.fit_transform(X)
    
    return best_model, doc_topic, best_k

def interpret_ngram_topics(lda, vocab, doc_topic):
    """Interpret topics with focus on meaningful phrases"""
    print("\n>>> Interpreting n-gram focused topics...")
    
    k = lda.n_components
    topics_data = []
    
    for topic_idx in range(k):
        # Get top terms
        top_indices = lda.components_[topic_idx].argsort()[-20:][::-1]
        all_terms = [vocab[i] for i in top_indices]
        
        # Separate phrases and words
        phrases = [term for term in all_terms if "_" in term]
        single_words = [term for term in all_terms if "_" not in term]
        
        # Count documents
        assignments = doc_topic.argmax(axis=1)
        doc_count = np.sum(assignments == topic_idx)
        
        # Generate interpretation based on phrases first, then words
        interpretation = generate_phrase_based_interpretation(phrases, single_words)
        
        topics_data.append({
            'topic_num': topic_idx + 1,
            'interpretation': interpretation,
            'doc_count': doc_count,
            'percentage': (doc_count / len(doc_topic)) * 100,
            'key_phrases': phrases[:5],
            'key_words': single_words[:8],
            'all_terms': all_terms[:12]
        })
        
        print(f"\nTopic {topic_idx + 1}: {interpretation}")
        print(f"  Documents: {doc_count} ({(doc_count/len(doc_topic)*100):.1f}%)")
        if phrases:
            print(f"  Key phrases: {', '.join(phrases[:5])}")
        print(f"  Key words: {', '.join(single_words[:6])}")
    
    return topics_data

def generate_phrase_based_interpretation(phrases, single_words):
    """Generate interpretation prioritizing meaningful phrases"""
    
    # Analyze phrases first
    phrase_text = ' '.join(phrases).lower()
    
    if 'helping_people' in phrase_text or 'help_people' in phrase_text:
        return "Helping People & Service Motivation"
    elif 'mental_health' in phrase_text:
        return "Mental Health Field Interest" 
    elif 'family' in phrase_text and ('experience' in phrase_text or 'background' in phrase_text):
        return "Family Experience & Background"
    elif 'career' in phrase_text or 'job' in phrase_text:
        return "Career & Professional Considerations"
    elif 'school' in phrase_text or 'college' in phrase_text:
        return "Educational & Academic Path"
    
    # Fallback to single word analysis
    words_text = ' '.join(single_words).lower()
    
    if 'help' in words_text or 'helping' in words_text:
        return "Helping & Support Orientation"
    elif 'family' in words_text or 'parents' in words_text:
        return "Family & Personal Background"
    elif 'school' in words_text or 'education' in words_text:
        return "Educational Pathway"
    elif 'people' in words_text and 'work' in words_text:
        return "People-Focused Work Interest"
    else:
        return "Mixed Motivational Theme"

def get_representative_quotes_ngram(corpus_df, doc_topic, topics_data):
    """Get representative quotes for n-gram topics"""
    print("\n" + "="*60)
    print("REPRESENTATIVE QUOTES FOR N-GRAM TOPICS")
    print("="*60)
    
    quotes = []
    for topic_data in topics_data:
        topic_idx = topic_data['topic_num'] - 1
        
        # Find best document for this topic
        topic_probs = doc_topic[:, topic_idx]
        best_doc_idx = topic_probs.argmax()
        best_prob = topic_probs[best_doc_idx]
        
        best_row = corpus_df.iloc[best_doc_idx]
        
        quotes.append({
            'topic': topic_data['interpretation'],
            'probability': round(best_prob, 3),
            'speaker': best_row['Speaker'],
            'session': best_row['session'],
            'text': best_row['Text'],
            'key_phrases': ', '.join(topic_data['key_phrases']) if topic_data['key_phrases'] else 'None'
        })
        
        print(f"\n>>> {topic_data['interpretation']} (P={best_prob:.3f}) <<<")
        if topic_data['key_phrases']:
            print(f"Key phrases: {', '.join(topic_data['key_phrases'][:3])}")
        print(f"Session: {best_row['session']}, Speaker: {best_row['Speaker']}")
        print(f"Quote: {textwrap.fill(best_row['Text'], 80)}")
    
    return quotes

def save_ngram_results(topics_data, quotes, corpus_df, doc_topic, discovered_phrases):
    """Save n-gram focused results"""
    
    # Topics summary
    topics_df = pd.DataFrame([{
        'Topic': f"Topic {t['topic_num']}: {t['interpretation']}",
        'Documents': t['doc_count'],
        'Percentage': f"{t['percentage']:.1f}%",
        'Key_Phrases': ', '.join(t['key_phrases'][:3]) if t['key_phrases'] else 'None',
        'Key_Words': ', '.join(t['key_words'][:6])
    } for t in topics_data])
    
    topics_df.to_csv(os.path.join(RESULTS_DIR, "ngram_topic_summary.csv"), index=False)
    
    # Representative quotes
    quotes_df = pd.DataFrame(quotes)
    quotes_df.to_csv(os.path.join(RESULTS_DIR, "ngram_representative_quotes.csv"), index=False)
    
    # Discovered phrases
    phrases_df = pd.DataFrame({
        'phrase': sorted(list(discovered_phrases)),
        'word_count': [len(phrase.split('_')) for phrase in sorted(list(discovered_phrases))]
    })
    phrases_df.to_csv(os.path.join(RESULTS_DIR, "discovered_meaningful_phrases.csv"), index=False)
    
    # Research report
    with open(os.path.join(RESULTS_DIR, "ngram_topic_analysis_report.txt"), 'w') as f:
        f.write("N-GRAM FOCUSED TOPIC ANALYSIS FOR SUD COUNSELING RESEARCH\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("• Phrase detection: Gensim with NPMI scoring\n")
        f.write("• N-gram weighting: Bigrams (3x), Trigrams (5x), 4+grams (7x)\n")
        f.write("• Topic modeling: LDA optimized for phrase interpretability\n") 
        f.write(f"• Meaningful phrases discovered: {len(discovered_phrases)}\n")
        f.write(f"• Documents analyzed: {len(corpus_df)}\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-" * 30 + "\n")
        for topic_data in topics_data:
            f.write(f"\n{topic_data['interpretation']}\n")
            f.write(f"• Documents: {topic_data['doc_count']} ({topic_data['percentage']:.1f}%)\n")
            if topic_data['key_phrases']:
                f.write(f"• Key phrases: {', '.join(topic_data['key_phrases'])}\n")
            f.write(f"• Key words: {', '.join(topic_data['key_words'])}\n")
        
        f.write(f"\nDISCOVERED MEANINGFUL PHRASES:\n")
        f.write("-" * 30 + "\n")
        bigrams = [p for p in discovered_phrases if len(p.split('_')) == 2]
        trigrams = [p for p in discovered_phrases if len(p.split('_')) == 3]
        fourgrams = [p for p in discovered_phrases if len(p.split('_')) >= 4]
        
        f.write(f"Bigrams ({len(bigrams)}): {', '.join(sorted(bigrams)[:15])}\n")
        f.write(f"Trigrams ({len(trigrams)}): {', '.join(sorted(trigrams)[:10])}\n") 
        f.write(f"4+ grams ({len(fourgrams)}): {', '.join(sorted(fourgrams)[:10])}\n")
    
    print(f"\n✅ N-gram results saved to {RESULTS_DIR}")

def main():
    """Execute n-gram focused topic modeling"""
    
    print("🎯 N-GRAM FOCUSED TOPIC MODELING FOR SUD COUNSELING RESEARCH")
    print("=" * 65)
    
    # Load data
    corpus_df = load_focus_group_data()
    original_texts = corpus_df['Text'].astype(str).tolist()
    
    # Detect meaningful phrases
    phrase_docs, discovered_phrases = detect_meaningful_phrases(original_texts)
    
    # Create n-gram weighted features
    X, vocab, vectorizer = create_ngram_weighted_features(phrase_docs, discovered_phrases)
    
    # Run topic modeling
    lda, doc_topic, k = run_ngram_topic_model(X, vocab)
    
    # Interpret topics
    topics_data = interpret_ngram_topics(lda, vocab, doc_topic)
    
    # Get representative quotes
    quotes = get_representative_quotes_ngram(corpus_df, doc_topic, topics_data)
    
    # Save results
    save_ngram_results(topics_data, quotes, corpus_df, doc_topic, discovered_phrases)
    
    print(f"\n🏆 N-GRAM TOPIC ANALYSIS COMPLETE!")
    print(f"📊 Found {len(topics_data)} phrase-rich themes")
    print(f"🔤 Discovered {len(discovered_phrases)} meaningful phrases")
    print("📄 Prioritized multi-word insights over single words")
    
    return topics_data, quotes, discovered_phrases

if __name__ == "__main__":
    topics, quotes, phrases = main()