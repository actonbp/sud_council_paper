#!/usr/bin/env python3
"""
Clean N-gram Topic Model (k=4)
Focus on meaningful phrases with cleaner topic separation
"""

import glob, os, re, textwrap
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
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
        df = df[~df["Speaker"].astype(str).str.match(r"^[A-Z]{2,3}$")]
        df = df[df["Text"].str.split().str.len() >= 8]
        df["session"] = os.path.basename(p)
        rows.append(df)
    
    corpus_df = pd.concat(rows, ignore_index=True)
    print(f"Loaded {len(corpus_df)} substantive utterances")
    return corpus_df

def detect_meaningful_phrases(texts):
    """Detect meaningful phrases using Gensim"""
    print("\n>>> Detecting meaningful phrases...")
    
    # Remove only circular terms
    CIRCULAR_TERMS = ['counselor', 'counseling', 'therapist', 'therapy']
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
    
    # Clean and tokenize
    cleaned_sentences = []
    for text in texts:
        text = text.lower()
        for term in CIRCULAR_TERMS:
            text = text.replace(term, ' ')
        
        tokens = simple_preprocess(text, deacc=True, min_len=3, max_len=20)
        meaningful_tokens = [token for token in tokens 
                           if token not in STOP_WORDS and len(token) >= 3 and not token.isdigit()]
        
        if len(meaningful_tokens) >= 5:
            cleaned_sentences.append(meaningful_tokens)
    
    print(f"Prepared {len(cleaned_sentences)} documents for phrase detection")
    
    # Phrase detection
    bigram_model = Phrases(cleaned_sentences, min_count=3, threshold=0.3, delimiter="_", scoring='npmi')
    bigram_phraser = Phraser(bigram_model)
    bigram_sentences = [bigram_phraser[sent] for sent in cleaned_sentences]
    
    trigram_model = Phrases(bigram_sentences, min_count=2, threshold=0.2, delimiter="_", scoring='npmi')
    trigram_phraser = Phraser(trigram_model)
    final_sentences = [trigram_phraser[sent] for sent in bigram_sentences]
    
    # Extract phrases
    all_phrases = set()
    for sent in final_sentences:
        for token in sent:
            if "_" in token:
                all_phrases.add(token)
    
    print(f"Discovered {len(all_phrases)} meaningful phrases")
    print(f"Sample phrases: {sorted(list(all_phrases))[:10]}")
    
    phrase_docs = [" ".join(sent) for sent in final_sentences]
    return phrase_docs, all_phrases

def create_phrase_focused_features(docs):
    """Create features heavily weighted toward phrases"""
    print("\n>>> Creating phrase-focused features...")
    
    def phrase_focused_analyzer(text):
        tokens = text.split()
        features = []
        
        for token in tokens:
            if "_" in token:  # Multi-word phrase
                n_words = len(token.split("_"))
                if n_words == 2:
                    weight = 4  # Bigrams get 4x weight
                elif n_words >= 3:
                    weight = 6  # Trigrams+ get 6x weight
                features.extend([token] * weight)
            else:
                # Only include substantial single words
                if len(token) >= 4:
                    features.append(token)
        
        return features
    
    vectorizer = TfidfVectorizer(
        analyzer=phrase_focused_analyzer,
        min_df=2,
        max_df=0.8,
        max_features=200,  # Smaller vocab for cleaner topics
        sublinear_tf=True,
        norm='l2'
    )
    
    X = vectorizer.fit_transform(docs)
    vocab = vectorizer.get_feature_names_out()
    
    phrases = [term for term in vocab if "_" in term]
    words = [term for term in vocab if "_" not in term]
    
    print(f"Feature matrix: {X.shape[0]} docs × {X.shape[1]} features")
    print(f"  - {len(phrases)} phrases ({len(phrases)/len(vocab):.1%})")
    print(f"  - {len(words)} words ({len(words)/len(vocab):.1%})")
    
    return X, vocab, vectorizer

def run_clean_topic_model(X, vocab):
    """Run LDA with k=4 for cleaner separation"""
    print("\n>>> Running clean topic model (k=4)...")
    
    lda = LatentDirichletAllocation(
        n_components=4,
        max_iter=500,
        learning_method="batch",
        doc_topic_prior=0.1,
        topic_word_prior=0.01,
        random_state=42
    )
    
    doc_topic = lda.fit_transform(X)
    return lda, doc_topic

def interpret_clean_topics(lda, vocab, doc_topic):
    """Interpret topics with manual refinement for clarity"""
    print("\n>>> Interpreting clean topics...")
    
    topics_data = []
    
    for topic_idx in range(4):
        top_indices = lda.components_[topic_idx].argsort()[-20:][::-1]
        all_terms = [vocab[i] for i in top_indices]
        
        phrases = [term for term in all_terms if "_" in term]
        words = [term for term in all_terms if "_" not in term]
        
        assignments = doc_topic.argmax(axis=1)
        doc_count = np.sum(assignments == topic_idx)
        
        # Manual interpretation based on key phrases
        interpretation = manual_topic_interpretation(topic_idx, phrases, words)
        
        topics_data.append({
            'topic_num': topic_idx + 1,
            'interpretation': interpretation,
            'doc_count': doc_count,
            'percentage': (doc_count / len(doc_topic)) * 100,
            'key_phrases': phrases[:5],
            'key_words': words[:8],
            'all_terms': all_terms[:15]
        })
        
        print(f"\nTopic {topic_idx + 1}: {interpretation}")
        print(f"  Documents: {doc_count} ({(doc_count/len(doc_topic)*100):.1f}%)")
        if phrases:
            print(f"  Key phrases: {', '.join(phrases[:4])}")
        print(f"  Key words: {', '.join(words[:6])}")
    
    return topics_data

def manual_topic_interpretation(topic_idx, phrases, words):
    """Manually interpret topics for maximum clarity"""
    
    phrase_text = ' '.join(phrases).lower()
    word_text = ' '.join(words).lower()
    all_text = phrase_text + ' ' + word_text
    
    # Look for clear patterns in phrases first
    if 'helping_people' in phrase_text or 'help_people' in phrase_text:
        if 'family' in all_text or 'experience' in all_text:
            return "Helping People through Personal Experience"
        else:
            return "Helping People & Service Motivation"
    
    elif 'career_path' in phrase_text or 'career_field' in phrase_text:
        return "Career Path & Professional Development"
    
    elif 'mental_health' in phrase_text and 'substance' in phrase_text:
        return "Mental Health & Substance Abuse Focus"
    
    elif 'mental_health' in phrase_text:
        if 'field' in all_text or 'work' in all_text:
            return "Mental Health Field Interest"
        else:
            return "Mental Health Awareness & Understanding"
    
    elif 'family' in all_text and ('experience' in all_text or 'background' in all_text):
        return "Family Background & Personal Experience"
    
    elif 'school' in all_text or 'education' in all_text:
        return "Educational & Academic Pathway"
    
    else:
        # Fallback based on most prominent content
        if 'help' in all_text:
            return "Helping & Support Orientation"
        elif 'work' in all_text or 'job' in all_text:
            return "Professional & Career Focus"
        elif 'people' in all_text:
            return "People-Centered Motivation"
        else:
            return f"Mixed Theme {topic_idx + 1}"

def save_clean_results(topics_data, corpus_df, doc_topic, all_phrases):
    """Save clean n-gram results"""
    
    # Get representative quotes
    quotes = []
    for topic_data in topics_data:
        topic_idx = topic_data['topic_num'] - 1
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
    
    # Save main results
    topics_df = pd.DataFrame([{
        'Topic': f"Topic {t['topic_num']}: {t['interpretation']}",
        'Documents': t['doc_count'],
        'Percentage': f"{t['percentage']:.1f}%",
        'Key_Phrases': ', '.join(t['key_phrases'][:4]) if t['key_phrases'] else 'None',
        'Key_Words': ', '.join(t['key_words'][:6])
    } for t in topics_data])
    
    topics_df.to_csv(os.path.join(RESULTS_DIR, "clean_ngram_topics.csv"), index=False)
    
    quotes_df = pd.DataFrame(quotes)
    quotes_df.to_csv(os.path.join(RESULTS_DIR, "clean_ngram_quotes.csv"), index=False)
    
    # Summary report
    with open(os.path.join(RESULTS_DIR, "clean_ngram_analysis_summary.txt"), 'w') as f:
        f.write("CLEAN N-GRAM TOPIC ANALYSIS - SUD COUNSELING RESEARCH\n")
        f.write("=" * 55 + "\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("• Phrase detection with Gensim NPMI scoring\n")
        f.write("• Heavy phrase weighting (4x bigrams, 6x trigrams)\n")
        f.write("• Clean 4-topic LDA model\n")
        f.write(f"• Meaningful phrases discovered: {len(all_phrases)}\n\n")
        
        f.write("TOPIC THEMES:\n")
        f.write("-" * 25 + "\n")
        for i, topic_data in enumerate(topics_data):
            f.write(f"\n{i+1}. {topic_data['interpretation']} ({topic_data['percentage']:.1f}%)\n")
            if topic_data['key_phrases']:
                f.write(f"   Key phrases: {', '.join(topic_data['key_phrases'][:4])}\n")
            f.write(f"   Key words: {', '.join(topic_data['key_words'][:6])}\n")
        
        f.write(f"\nSAMPLE MEANINGFUL PHRASES:\n")
        f.write("-" * 25 + "\n")
        sample_phrases = sorted(list(all_phrases))[:20]
        f.write(f"{', '.join(sample_phrases)}\n")
    
    print(f"\n✅ Clean n-gram results saved")
    
    # Display representative quotes
    print("\n" + "="*60)
    print("REPRESENTATIVE QUOTES - CLEAN N-GRAM TOPICS")
    print("="*60)
    
    for quote in quotes:
        print(f"\n>>> {quote['topic']} (P={quote['probability']}) <<<")
        if quote['key_phrases'] != 'None':
            print(f"Key phrases: {quote['key_phrases']}")
        print(f"Quote: {textwrap.fill(quote['text'], 80)}")
    
    return topics_data, quotes

def main():
    """Execute clean n-gram topic modeling"""
    
    print("🎯 CLEAN N-GRAM TOPIC MODELING (k=4)")
    print("=" * 45)
    
    corpus_df = load_focus_group_data()
    original_texts = corpus_df['Text'].astype(str).tolist()
    
    phrase_docs, all_phrases = detect_meaningful_phrases(original_texts)
    X, vocab, vectorizer = create_phrase_focused_features(phrase_docs)
    lda, doc_topic = run_clean_topic_model(X, vocab)
    topics_data, quotes = save_clean_results(topics_data := interpret_clean_topics(lda, vocab, doc_topic), 
                                           corpus_df, doc_topic, all_phrases)
    
    print(f"\n🏆 CLEAN N-GRAM ANALYSIS COMPLETE!")
    print(f"📊 4 distinct phrase-rich themes")
    print(f"🔤 {len(all_phrases)} meaningful phrases discovered")
    
    return topics_data, quotes

if __name__ == "__main__":
    topics, quotes = main()