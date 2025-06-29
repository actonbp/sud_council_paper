#!/usr/bin/env python3
"""
Final Optimal Topic Model for SUD Counseling Research
Combines best practices from comparison analysis
"""

import glob, os, re, textwrap
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Configuration
DATA_DIR = "../../data"
RESULTS_DIR = "../../results/"

def load_focus_group_data():
    """Load and prepare focus group data"""
    paths = glob.glob(os.path.join(DATA_DIR, "*_Focus_Group_full*.csv"))
    
    rows = []
    for p in paths:
        df = pd.read_csv(p)
        # Remove moderator utterances (all caps speaker codes)
        df = df[~df["Speaker"].astype(str).str.match(r"^[A-Z]{2,3}$")]
        # Remove very short utterances (less than 8 words)
        df = df[df["Text"].str.split().str.len() >= 8]
        df["session"] = os.path.basename(p)
        rows.append(df)
    
    corpus_df = pd.concat(rows, ignore_index=True)
    print(f"Loaded {len(corpus_df)} substantive utterances from {len(corpus_df['session'].unique())} sessions")
    
    return corpus_df

def optimal_text_preprocessing(texts):
    """Optimal text preprocessing based on comparison results"""
    
    # Strategic domain term removal (not too aggressive)
    STRATEGIC_DOMAIN_TERMS = set([
        'counselor', 'counselors', 'counseling',  # Circular job terms
        'therapist', 'therapists', 'therapy',     # Circular profession terms  
        'mental_health',                          # Obvious target phrase
        'substance_abuse'                         # Obvious target phrase
    ])
    
    # Comprehensive stop words (including conversation fillers)
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
    also even just now well much many still back come came put take took give gave
    go went come came one two three first second next last another other
    
    re ve ll don didn won isn aren weren hasn haven couldn wouldn shouldn mustn needn mightn
    """.split())
    
    def clean_text(text):
        """Clean individual text"""
        text = text.lower()
        
        # Remove strategic domain terms
        for term in STRATEGIC_DOMAIN_TERMS:
            text = re.sub(rf'\b{re.escape(term)}\b', ' ', text)
        
        # Clean punctuation and normalize spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    cleaned_texts = [clean_text(text) for text in texts]
    
    print(f"Preprocessing complete:")
    print(f"  - Removed {len(STRATEGIC_DOMAIN_TERMS)} strategic domain terms")
    print(f"  - Using {len(STOP_WORDS)} stop words")
    
    return cleaned_texts, STOP_WORDS

def create_optimal_topic_model(texts, stop_words):
    """Create optimal topic model based on comparison results"""
    
    # TF-IDF with optimal parameters
    vectorizer = TfidfVectorizer(
        stop_words=list(stop_words),
        ngram_range=(1, 2),           # Include meaningful phrases
        min_df=3,                     # Must appear in at least 3 documents
        max_df=0.7,                   # Remove terms in >70% of documents  
        max_features=300,             # Limit vocabulary size
        sublinear_tf=True,            # Use log-scale TF
        norm='l2'                     # L2 normalization
    )
    
    X = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    
    print(f"Feature matrix: {X.shape[0]} documents × {X.shape[1]} features")
    print(f"Sample vocabulary: {list(vocab[:15])}")
    
    # Optimal LDA configuration (k=4 for balance of detail and interpretability)
    optimal_k = 4
    lda = LatentDirichletAllocation(
        n_components=optimal_k,
        max_iter=500,
        learning_method="batch",
        doc_topic_prior=0.1,         # Sparse document-topic distributions
        topic_word_prior=0.01,       # Sparse topic-word distributions  
        random_state=42              # Reproducible results
    )
    
    doc_topic_probs = lda.fit_transform(X)
    
    print(f"Trained LDA model with k={optimal_k} topics")
    
    return lda, doc_topic_probs, vocab, vectorizer, X

def extract_and_interpret_topics(lda, vocab, doc_topic_probs):
    """Extract topics and provide research-oriented interpretations"""
    
    k = lda.n_components
    topics_data = []
    
    # Extract top terms for each topic
    for topic_idx in range(k):
        # Get top 15 terms for analysis, display top 10
        top_indices = lda.components_[topic_idx].argsort()[-15:][::-1]
        all_terms = [vocab[i] for i in top_indices]
        display_terms = all_terms[:10]
        
        # Count documents primarily assigned to this topic
        primary_assignments = doc_topic_probs.argmax(axis=1)
        doc_count = np.sum(primary_assignments == topic_idx)
        
        # Research-oriented interpretation based on term patterns
        interpretation = interpret_topic_for_research(all_terms)
        
        topics_data.append({
            'topic_num': topic_idx + 1,
            'interpretation': interpretation,
            'doc_count': doc_count,
            'percentage': (doc_count / len(doc_topic_probs)) * 100,
            'top_terms': display_terms,
            'all_terms': all_terms
        })
        
        print(f"\nTopic {topic_idx + 1}: {interpretation}")
        print(f"  Documents: {doc_count} ({(doc_count/len(doc_topic_probs)*100):.1f}%)")
        print(f"  Key terms: {', '.join(display_terms[:8])}")
    
    return topics_data

def interpret_topic_for_research(terms):
    """Provide research-focused topic interpretation"""
    terms_str = ' '.join(terms).lower()
    
    # Pattern matching for research themes
    if ('help' in terms_str or 'helping' in terms_str) and ('people' in terms_str):
        if 'family' in terms_str or 'parents' in terms_str:
            return "Altruistic Helping with Family Influence"
        else:
            return "Altruistic Helping Motivation"
    
    elif 'family' in terms_str or 'parents' in terms_str or 'mom' in terms_str:
        if 'experience' in terms_str or 'trauma' in terms_str:
            return "Personal/Family Experience Background"
        else:
            return "Family Background Influence"
    
    elif 'school' in terms_str or 'education' in terms_str or 'college' in terms_str:
        if 'psychology' in terms_str or 'psych' in terms_str:
            return "Academic Psychology Interest"
        else:
            return "Educational Pathway Considerations"
    
    elif 'field' in terms_str or 'job' in terms_str or 'work' in terms_str:
        if 'interesting' in terms_str or 'interest' in terms_str:
            return "Professional Field Interest"
        else:
            return "Career Practical Considerations"
    
    elif 'people' in terms_str and ('different' in terms_str or 'someone' in terms_str):
        return "Interpersonal & Individual Differences"
    
    else:
        # Fallback based on most prominent term
        if terms:
            return f"{terms[0].title()}-Centered Theme"
        else:
            return "Mixed Theme"

def get_representative_quotes(corpus_df, doc_topic_probs, topics_data):
    """Get most representative quote for each topic"""
    
    representative_quotes = []
    
    for topic_data in topics_data:
        topic_idx = topic_data['topic_num'] - 1
        
        # Find document with highest probability for this topic
        topic_probs = doc_topic_probs[:, topic_idx]
        best_doc_idx = topic_probs.argmax()
        best_prob = topic_probs[best_doc_idx]
        
        # Get the original utterance
        best_row = corpus_df.iloc[best_doc_idx]
        
        representative_quotes.append({
            'topic': topic_data['interpretation'],
            'probability': round(best_prob, 3),
            'speaker': best_row['Speaker'],
            'session': best_row['session'],
            'text': best_row['Text']
        })
        
        print(f"\n>>> {topic_data['interpretation']} (P={best_prob:.3f}) <<<")
        print(f"Session: {best_row['session']}, Speaker: {best_row['Speaker']}")
        print(f"Quote: {textwrap.fill(best_row['Text'], 80)}")
    
    return representative_quotes

def save_research_results(topics_data, representative_quotes, corpus_df, doc_topic_probs):
    """Save results formatted for research paper"""
    
    # 1. Topics summary table
    topics_df = pd.DataFrame([{
        'Topic': f"Topic {t['topic_num']}: {t['interpretation']}",
        'Documents': t['doc_count'],
        'Percentage': f"{t['percentage']:.1f}%", 
        'Key_Terms': ', '.join(t['top_terms'][:8])
    } for t in topics_data])
    
    topics_df.to_csv(os.path.join(RESULTS_DIR, "final_topic_summary.csv"), index=False)
    
    # 2. Representative quotes
    quotes_df = pd.DataFrame(representative_quotes)
    quotes_df.to_csv(os.path.join(RESULTS_DIR, "final_representative_quotes.csv"), index=False)
    
    # 3. Document assignments (for validation)
    primary_assignments = doc_topic_probs.argmax(axis=1)
    max_probs = doc_topic_probs.max(axis=1)
    
    assignments_df = pd.DataFrame({
        'document_id': range(len(corpus_df)),
        'primary_topic': [f"Topic {i+1}" for i in primary_assignments],
        'topic_probability': max_probs,
        'speaker': corpus_df['Speaker'].values,
        'session': corpus_df['session'].values,
        'text_preview': [text[:150] + '...' if len(text) > 150 else text 
                        for text in corpus_df['Text'].values]
    })
    
    assignments_df.to_csv(os.path.join(RESULTS_DIR, "final_document_assignments.csv"), index=False)
    
    # 4. Research report
    with open(os.path.join(RESULTS_DIR, "final_topic_analysis_report.txt"), 'w') as f:
        f.write("FINAL OPTIMAL TOPIC ANALYSIS FOR SUD COUNSELING RESEARCH\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("• Text preprocessing: Strategic domain term removal\n") 
        f.write("• Vectorization: TF-IDF with unigrams + bigrams\n")
        f.write("• Topic modeling: Latent Dirichlet Allocation (LDA)\n")
        f.write("• Topics (k): 4 (optimal balance of detail and interpretability)\n")
        f.write(f"• Documents analyzed: {len(corpus_df)}\n\n")
        
        f.write("RESEARCH FINDINGS:\n")
        f.write("-" * 30 + "\n")
        for topic_data in topics_data:
            f.write(f"\n{topic_data['interpretation']}\n")
            f.write(f"• Documents: {topic_data['doc_count']} ({topic_data['percentage']:.1f}%)\n")
            f.write(f"• Key terms: {', '.join(topic_data['top_terms'][:10])}\n")
        
        f.write(f"\nREPRESENTATIVE UTTERANCES:\n")
        f.write("-" * 30 + "\n")
        for quote in representative_quotes:
            f.write(f"\n{quote['topic']} (P={quote['probability']}):\n")
            f.write(f'"{quote['text'][:200]}..."\n')
    
    print(f"\n✅ All results saved to {RESULTS_DIR}")
    print("Files created:")
    print("  • final_topic_summary.csv")
    print("  • final_representative_quotes.csv") 
    print("  • final_document_assignments.csv")
    print("  • final_topic_analysis_report.txt")

def main():
    """Execute optimal topic modeling pipeline"""
    
    print("🎯 FINAL OPTIMAL TOPIC MODELING FOR SUD COUNSELING RESEARCH")
    print("=" * 65)
    
    # Load data
    corpus_df = load_focus_group_data()
    original_texts = corpus_df['Text'].astype(str).tolist()
    
    # Preprocessing  
    cleaned_texts, stop_words = optimal_text_preprocessing(original_texts)
    
    # Topic modeling
    lda, doc_topic_probs, vocab, vectorizer, X = create_optimal_topic_model(cleaned_texts, stop_words)
    
    # Extract and interpret topics
    topics_data = extract_and_interpret_topics(lda, vocab, doc_topic_probs)
    
    # Get representative quotes
    print("\n" + "=" * 65)
    print("REPRESENTATIVE QUOTES FOR EACH TOPIC")
    print("=" * 65)
    representative_quotes = get_representative_quotes(corpus_df, doc_topic_probs, topics_data)
    
    # Save results
    save_research_results(topics_data, representative_quotes, corpus_df, doc_topic_probs)
    
    print("\n🏆 OPTIMAL TOPIC ANALYSIS COMPLETE!")
    print(f"📊 Found {len(topics_data)} distinct themes in SUD counseling interest")
    print("📄 All results formatted for research paper inclusion")
    
    return topics_data, representative_quotes

if __name__ == "__main__":
    topics, quotes = main()