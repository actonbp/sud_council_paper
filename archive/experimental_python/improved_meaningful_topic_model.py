#!/usr/bin/env python3
"""
Improved Meaningful Topic Model
Focuses on semantically meaningful phrases while filtering generic conversation patterns
"""

import glob, os, re, textwrap
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from collections import Counter

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

def detect_meaningful_phrases_improved(texts):
    """Enhanced phrase detection with generic filtering"""
    print("\n>>> Detecting meaningful phrases (improved filtering)...")
    
    # Remove circular terms only
    CIRCULAR_TERMS = ['counselor', 'counseling', 'therapist', 'therapy']
    
    # Comprehensive stop words
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
    """.split())
    
    # Clean and tokenize with longer context windows
    cleaned_sentences = []
    for text in texts:
        text = text.lower()
        
        # Remove circular terms
        for term in CIRCULAR_TERMS:
            text = text.replace(term, ' ')
        
        # Tokenize with focus on meaningful content
        tokens = simple_preprocess(text, deacc=True, min_len=3, max_len=25)
        meaningful_tokens = []
        
        for token in tokens:
            if (token not in STOP_WORDS and 
                len(token) >= 3 and 
                not token.isdigit() and
                not token in ['thing', 'things', 'stuff', 'whatever']):  # Remove generic words
                meaningful_tokens.append(token)
        
        if len(meaningful_tokens) >= 5:
            cleaned_sentences.append(meaningful_tokens)
    
    print(f"Prepared {len(cleaned_sentences)} documents for phrase detection")
    
    # Enhanced phrase detection with stricter thresholds
    print("   Detecting meaningful bigrams...")
    bigram_model = Phrases(cleaned_sentences, min_count=4, threshold=0.4, delimiter="_", scoring='npmi')
    bigram_phraser = Phraser(bigram_model)
    bigram_sentences = [bigram_phraser[sent] for sent in cleaned_sentences]
    
    print("   Detecting meaningful trigrams...")
    trigram_model = Phrases(bigram_sentences, min_count=3, threshold=0.3, delimiter="_", scoring='npmi')
    trigram_phraser = Phraser(trigram_model)
    final_sentences = [trigram_phraser[sent] for sent in bigram_sentences]
    
    # Extract and filter phrases
    all_phrases = set()
    for sent in final_sentences:
        for token in sent:
            if "_" in token:
                all_phrases.add(token)
    
    # Filter out generic phrases
    meaningful_phrases = filter_generic_phrases(all_phrases)
    
    print(f"Total phrases found: {len(all_phrases)}")
    print(f"Meaningful phrases after filtering: {len(meaningful_phrases)}")
    print(f"Sample meaningful phrases: {sorted(list(meaningful_phrases))[:10]}")
    
    phrase_docs = [" ".join(sent) for sent in final_sentences]
    return phrase_docs, meaningful_phrases

def filter_generic_phrases(phrases):
    """Filter out generic conversational phrases"""
    
    # Generic conversation patterns to remove
    GENERIC_PATTERNS = {
        'time_based': ['every_day', 'long_time', 'lot_time', 'right_now', 'first_time'],
        'vague_qualifiers': ['little_bit', 'pretty_much', 'kind_like', 'sort_like', 'lot_like'],
        'generic_opinions': ['good_idea', 'really_good', 'pretty_good', 'really_nice', 'sounds_good'],
        'generic_education': ['high_school', 'went_school', 'school_school', 'back_school'],
        'conversation_fillers': ['mean_like', 'know_like', 'think_like', 'feel_like', 'guess_like'],
        'generic_actions': ['try_get', 'want_get', 'need_get', 'going_try', 'trying_get'],
        'generic_references': ['one_thing', 'other_thing', 'anything_like', 'something_like']
    }
    
    # Flatten all generic patterns
    all_generic = set()
    for category in GENERIC_PATTERNS.values():
        all_generic.update(category)
    
    # Meaningful phrase patterns we want to keep
    MEANINGFUL_PATTERNS = {
        'helping_motivation': ['helping_people', 'help_people', 'helping_others', 'support_people'],
        'family_experience': ['family_member', 'family_experience', 'personal_experience', 'family_background'],
        'professional_concepts': ['mental_health', 'substance_abuse', 'career_path', 'professional_field'],
        'responsibility_awareness': ['big_responsibility', 'huge_responsibility', 'lot_responsibility'],
        'emotional_connection': ['emotional_connection', 'personal_connection', 'emotionally_invested'],
        'field_interest': ['really_interested', 'very_interested', 'find_interesting'],
        'life_impact': ['make_difference', 'making_difference', 'change_lives', 'impact_lives'],
        'career_planning': ['career_path', 'career_choice', 'future_career', 'professional_path']
    }
    
    # Flatten meaningful patterns
    prioritized_meaningful = set()
    for category in MEANINGFUL_PATTERNS.values():
        prioritized_meaningful.update(category)
    
    # Filter phrases
    meaningful_phrases = set()
    
    for phrase in phrases:
        # Always keep prioritized meaningful phrases
        if phrase in prioritized_meaningful:
            meaningful_phrases.add(phrase)
            continue
            
        # Skip generic phrases
        if phrase in all_generic:
            continue
            
        # Apply semantic filters
        words = phrase.split('_')
        
        # Keep if it contains meaningful semantic content
        meaningful_indicators = [
            'help', 'helping', 'support', 'care', 'assist',
            'family', 'personal', 'experience', 'background', 'trauma',
            'mental', 'health', 'substance', 'abuse', 'addiction', 'recovery',
            'career', 'professional', 'field', 'work', 'job',
            'interested', 'passion', 'love', 'enjoy',
            'responsibility', 'important', 'serious', 'difficult',
            'difference', 'impact', 'change', 'lives', 'community',
            'education', 'learning', 'knowledge', 'understanding',
            'emotional', 'psychology', 'behavioral', 'social'
        ]
        
        # Keep if any word has meaningful content
        if any(any(indicator in word for indicator in meaningful_indicators) for word in words):
            # Additional filter: must be substantive (not just connective words)
            substantive_words = [w for w in words if len(w) >= 4]
            if len(substantive_words) >= 1:  # At least one substantive word
                meaningful_phrases.add(phrase)
    
    return meaningful_phrases

def create_semantic_features(docs, meaningful_phrases):
    """Create features with semantic grouping and improved weighting"""
    print("\n>>> Creating semantic features...")
    
    # Group semantically similar phrases
    semantic_groups = {
        'helping_motivation': ['helping_people', 'help_people', 'helping_others', 'support_people', 'trying_help'],
        'family_influence': ['family_member', 'family_experience', 'personal_experience', 'family_background', 'parents_supportive'],
        'field_focus': ['mental_health', 'substance_abuse', 'addiction_treatment', 'behavioral_health'],
        'responsibility_awareness': ['big_responsibility', 'huge_responsibility', 'lot_responsibility', 'serious_responsibility'],
        'career_planning': ['career_path', 'career_choice', 'future_career', 'professional_path', 'career_field'],
        'emotional_connection': ['emotional_connection', 'personal_connection', 'emotionally_invested', 'emotional_support'],
        'life_impact': ['make_difference', 'making_difference', 'change_lives', 'impact_lives', 'help_others']
    }
    
    def semantic_analyzer(text):
        tokens = text.split()
        features = []
        
        for token in tokens:
            if "_" in token and token in meaningful_phrases:
                # Determine semantic weight
                semantic_weight = 1
                
                # Check semantic groups for higher weighting
                for group_name, group_phrases in semantic_groups.items():
                    if token in group_phrases:
                        if group_name in ['helping_motivation', 'family_influence', 'field_focus']:
                            semantic_weight = 8  # High weight for core themes
                        elif group_name in ['responsibility_awareness', 'emotional_connection']:
                            semantic_weight = 6  # Medium-high weight for insights
                        else:
                            semantic_weight = 4  # Medium weight for general themes
                        break
                
                # Default phrase weighting
                if semantic_weight == 1:
                    n_words = len(token.split("_"))
                    semantic_weight = min(n_words * 2, 6)  # 2x per word, max 6x
                
                features.extend([token] * semantic_weight)
                
            elif len(token) >= 5:  # Only substantial single words
                features.append(token)
        
        return features
    
    vectorizer = TfidfVectorizer(
        analyzer=semantic_analyzer,
        min_df=2,
        max_df=0.85,
        max_features=150,  # Smaller, more focused vocabulary
        sublinear_tf=True,
        norm='l2'
    )
    
    X = vectorizer.fit_transform(docs)
    vocab = vectorizer.get_feature_names_out()
    
    # Analyze vocabulary composition
    phrases = [term for term in vocab if "_" in term]
    words = [term for term in vocab if "_" not in term]
    
    print(f"Semantic feature matrix: {X.shape[0]} docs × {X.shape[1]} features")
    print(f"  - {len(phrases)} meaningful phrases ({len(phrases)/len(vocab):.1%})")
    print(f"  - {len(words)} substantial words ({len(words)/len(vocab):.1%})")
    print(f"Key phrases in vocabulary: {[p for p in phrases if any(group in p for group in ['helping', 'family', 'mental', 'career'])]}")
    
    return X, vocab, vectorizer

def run_focused_topic_model(X, vocab):
    """Run LDA with k=3 for clearest interpretation"""
    print("\n>>> Running focused topic model (k=3)...")
    
    lda = LatentDirichletAllocation(
        n_components=3,
        max_iter=1000,               # More iterations for stability
        learning_method="batch",
        doc_topic_prior=0.05,        # Sparser document-topic distributions
        topic_word_prior=0.005,      # Sparser topic-word distributions
        random_state=42
    )
    
    doc_topic = lda.fit_transform(X)
    
    # Check topic separation
    assignments = doc_topic.argmax(axis=1)
    topic_distribution = Counter(assignments)
    print(f"Topic distribution: {dict(topic_distribution)}")
    
    return lda, doc_topic

def interpret_meaningful_topics(lda, vocab, doc_topic):
    """Interpret topics with focus on meaningful phrases"""
    print("\n>>> Interpreting meaningful topics...")
    
    topics_data = []
    
    for topic_idx in range(3):
        top_indices = lda.components_[topic_idx].argsort()[-25:][::-1]
        all_terms = [vocab[i] for i in top_indices]
        
        # Separate phrases and words, prioritize phrases
        phrases = [term for term in all_terms if "_" in term]
        words = [term for term in all_terms if "_" not in term]
        
        assignments = doc_topic.argmax(axis=1)
        doc_count = np.sum(assignments == topic_idx)
        
        # Enhanced interpretation based on meaningful phrases
        interpretation = enhanced_topic_interpretation(phrases, words, topic_idx)
        
        topics_data.append({
            'topic_num': topic_idx + 1,
            'interpretation': interpretation,
            'doc_count': doc_count,
            'percentage': (doc_count / len(doc_topic)) * 100,
            'key_phrases': phrases[:6],
            'key_words': words[:8],
            'top_terms': all_terms[:12]
        })
        
        print(f"\nTopic {topic_idx + 1}: {interpretation}")
        print(f"  Documents: {doc_count} ({(doc_count/len(doc_topic)*100):.1f}%)")
        if phrases:
            print(f"  Key phrases: {', '.join(phrases[:5])}")
        print(f"  Key words: {', '.join(words[:6])}")
    
    return topics_data

def enhanced_topic_interpretation(phrases, words, topic_idx):
    """Enhanced interpretation focusing on core motivations"""
    
    phrase_text = ' '.join(phrases).lower()
    word_text = ' '.join(words).lower()
    all_content = phrase_text + ' ' + word_text
    
    # Priority interpretation based on meaningful phrases
    if any(phrase in phrase_text for phrase in ['helping_people', 'help_people', 'helping_others']):
        if any(word in all_content for word in ['family', 'personal', 'experience']):
            return "Helping Others Through Personal Experience"
        else:
            return "Altruistic Helping & Service Motivation"
    
    elif any(phrase in phrase_text for phrase in ['family_member', 'family_experience', 'personal_experience']):
        return "Personal & Family Background Influence"
    
    elif any(phrase in phrase_text for phrase in ['mental_health', 'substance_abuse']):
        if any(word in all_content for word in ['field', 'professional', 'career']):
            return "Mental Health & Substance Abuse Professional Interest"
        else:
            return "Mental Health & Substance Abuse Awareness"
    
    elif any(phrase in phrase_text for phrase in ['career_path', 'career_choice', 'future_career']):
        return "Career Path & Professional Development"
    
    elif any(phrase in phrase_text for phrase in ['big_responsibility', 'serious_responsibility']):
        return "Professional Responsibility & Commitment Awareness"
    
    # Fallback to word-based interpretation
    elif any(word in all_content for word in ['help', 'helping', 'support']):
        return "Helping & Support Orientation"
    
    elif any(word in all_content for word in ['family', 'parents', 'personal']):
        return "Family & Personal Background"
    
    elif any(word in all_content for word in ['career', 'professional', 'field', 'work']):
        return "Professional & Career Focus"
    
    else:
        return f"Underlying Motivation Theme {topic_idx + 1}"

def save_improved_results(topics_data, corpus_df, doc_topic, meaningful_phrases):
    """Save improved results with comprehensive analysis"""
    
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
            'key_phrases': ', '.join(topic_data['key_phrases'][:4]) if topic_data['key_phrases'] else 'None'
        })
    
    # Save comprehensive results
    topics_df = pd.DataFrame([{
        'Topic': f"Topic {t['topic_num']}: {t['interpretation']}",
        'Documents': t['doc_count'],
        'Percentage': f"{t['percentage']:.1f}%",
        'Key_Phrases': ', '.join(t['key_phrases'][:4]) if t['key_phrases'] else 'None',
        'Key_Words': ', '.join(t['key_words'][:6])
    } for t in topics_data])
    
    topics_df.to_csv(os.path.join(RESULTS_DIR, "improved_meaningful_topics.csv"), index=False)
    
    quotes_df = pd.DataFrame(quotes)
    quotes_df.to_csv(os.path.join(RESULTS_DIR, "improved_meaningful_quotes.csv"), index=False)
    
    # Meaningful phrases analysis
    phrases_df = pd.DataFrame({
        'meaningful_phrase': sorted(list(meaningful_phrases)),
        'word_count': [len(phrase.split('_')) for phrase in sorted(list(meaningful_phrases))],
        'category': [categorize_phrase(phrase) for phrase in sorted(list(meaningful_phrases))]
    })
    phrases_df.to_csv(os.path.join(RESULTS_DIR, "improved_meaningful_phrases.csv"), index=False)
    
    # Comprehensive report
    with open(os.path.join(RESULTS_DIR, "improved_meaningful_analysis_report.txt"), 'w') as f:
        f.write("IMPROVED MEANINGFUL TOPIC ANALYSIS - SUD COUNSELING RESEARCH\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("METHODOLOGY IMPROVEMENTS:\n")
        f.write("• Enhanced phrase filtering (removed generic conversation patterns)\n")
        f.write("• Semantic grouping and weighting of related phrases\n")
        f.write("• Focused 3-topic model for clearest interpretation\n")
        f.write("• Prioritized meaningful content over frequency\n")
        f.write(f"• Meaningful phrases discovered: {len(meaningful_phrases)}\n\n")
        
        f.write("RESEARCH THEMES:\n")
        f.write("-" * 30 + "\n")
        for topic_data in topics_data:
            f.write(f"\n{topic_data['interpretation']} ({topic_data['percentage']:.1f}%)\n")
            if topic_data['key_phrases']:
                f.write(f"  Key phrases: {', '.join(topic_data['key_phrases'][:5])}\n")
            f.write(f"  Key words: {', '.join(topic_data['key_words'][:6])}\n")
        
        f.write(f"\nMEANINGFUL PHRASES BY CATEGORY:\n")
        f.write("-" * 35 + "\n")
        phrase_categories = {}
        for phrase in meaningful_phrases:
            cat = categorize_phrase(phrase)
            if cat not in phrase_categories:
                phrase_categories[cat] = []
            phrase_categories[cat].append(phrase)
        
        for category, phrase_list in sorted(phrase_categories.items()):
            f.write(f"{category}: {', '.join(sorted(phrase_list)[:8])}\n")
    
    print(f"\n✅ Improved meaningful results saved")
    
    # Display results
    print("\n" + "="*70)
    print("IMPROVED MEANINGFUL TOPICS - REPRESENTATIVE QUOTES")
    print("="*70)
    
    for quote in quotes:
        print(f"\n>>> {quote['topic']} (P={quote['probability']}) <<<")
        if quote['key_phrases'] != 'None':
            print(f"Key phrases: {quote['key_phrases']}")
        print(f"Quote: {textwrap.fill(quote['text'], 75)}")
    
    return topics_data, quotes, phrases_df

def categorize_phrase(phrase):
    """Categorize phrases for analysis"""
    if any(word in phrase for word in ['help', 'helping', 'support', 'assist']):
        return "Helping_Motivation"
    elif any(word in phrase for word in ['family', 'personal', 'experience', 'background']):
        return "Personal_Experience"
    elif any(word in phrase for word in ['mental', 'substance', 'health', 'abuse', 'addiction']):
        return "Field_Focus"
    elif any(word in phrase for word in ['career', 'professional', 'field', 'work']):
        return "Career_Planning"
    elif any(word in phrase for word in ['responsibility', 'important', 'serious']):
        return "Professional_Awareness"
    elif any(word in phrase for word in ['emotional', 'connection', 'invested']):
        return "Emotional_Connection"
    else:
        return "Other"

def main():
    """Execute improved meaningful topic modeling"""
    
    print("🎯 IMPROVED MEANINGFUL TOPIC MODELING")
    print("=" * 45)
    
    corpus_df = load_focus_group_data()
    original_texts = corpus_df['Text'].astype(str).tolist()
    
    phrase_docs, meaningful_phrases = detect_meaningful_phrases_improved(original_texts)
    X, vocab, vectorizer = create_semantic_features(phrase_docs, meaningful_phrases)
    lda, doc_topic = run_focused_topic_model(X, vocab)
    topics_data, quotes, phrases_df = save_improved_results(
        interpret_meaningful_topics(lda, vocab, doc_topic), 
        corpus_df, doc_topic, meaningful_phrases)
    
    print(f"\n🏆 IMPROVED MEANINGFUL ANALYSIS COMPLETE!")
    print(f"📊 3 focused themes with rich semantic content")
    print(f"🔤 {len(meaningful_phrases)} high-quality meaningful phrases")
    print("📈 Filtered out generic conversation patterns")
    print("🎯 Semantic grouping for deeper insights")
    
    return topics_data, quotes, meaningful_phrases

if __name__ == "__main__":
    topics, quotes, phrases = main()