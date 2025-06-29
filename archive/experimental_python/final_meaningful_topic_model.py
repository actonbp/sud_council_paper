#!/usr/bin/env python3
"""
Final Meaningful Topic Model
Removes SUD/mental health domain terms while keeping meaningful phrases about motivations
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

def detect_underlying_motivation_phrases(texts):
    """Detect phrases about underlying motivations, not domain terms"""
    print("\n>>> Detecting underlying motivation phrases (domain terms removed)...")
    
    # EXPANDED domain term removal - all SUD/mental health field terms
    DOMAIN_TERMS = [
        # Core field terms
        'counselor', 'counseling', 'therapist', 'therapy', 'therapeutic',
        'mental', 'health', 'substance', 'abuse', 'addiction', 'addicted',
        'drug', 'drugs', 'alcohol', 'alcoholic', 'alcoholism',
        
        # Professional terms
        'treatment', 'rehabilitation', 'rehab', 'recovery', 'clinic', 'clinical',
        'patient', 'patients', 'client', 'clients', 'diagnosis', 'diagnostic',
        'intervention', 'disorder', 'disorders', 'condition', 'conditions',
        'medication', 'medications', 'medicine', 'symptom', 'symptoms',
        
        # Related field terms
        'psychology', 'psychological', 'psychologist', 'psychiatrist', 'psychiatric',
        'social_work', 'social_worker', 'behavioral', 'cognitive'
    ]
    
    # Stop words + conversation fillers
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
    thing things stuff whatever
    """.split())
    
    # Clean and tokenize focusing on motivations, not domain content
    cleaned_sentences = []
    domain_terms_removed = 0
    
    for text in texts:
        text = text.lower()
        
        # Remove all domain terms
        original_length = len(text.split())
        for term in DOMAIN_TERMS:
            if term in text:
                domain_terms_removed += text.count(term)
                text = re.sub(rf'\b{re.escape(term)}\w*\b', ' ', text)
        
        # Tokenize what remains
        tokens = simple_preprocess(text, deacc=True, min_len=3, max_len=25)
        meaningful_tokens = []
        
        for token in tokens:
            if (token not in STOP_WORDS and 
                token not in DOMAIN_TERMS and  # Double-check domain removal
                len(token) >= 3 and 
                not token.isdigit()):
                meaningful_tokens.append(token)
        
        # Only keep if sufficient meaningful content remains
        if len(meaningful_tokens) >= 4:
            cleaned_sentences.append(meaningful_tokens)
    
    print(f"Removed {domain_terms_removed} domain term instances")
    print(f"Prepared {len(cleaned_sentences)} documents with underlying content")
    
    # Enhanced phrase detection for motivational content
    print("   Detecting motivation-focused bigrams...")
    bigram_model = Phrases(cleaned_sentences, min_count=4, threshold=0.5, delimiter="_", scoring='npmi')
    bigram_phraser = Phraser(bigram_model)
    bigram_sentences = [bigram_phraser[sent] for sent in cleaned_sentences]
    
    print("   Detecting motivation-focused trigrams...")
    trigram_model = Phrases(bigram_sentences, min_count=3, threshold=0.4, delimiter="_", scoring='npmi')
    trigram_phraser = Phraser(trigram_model)
    final_sentences = [trigram_phraser[sent] for sent in bigram_sentences]
    
    # Extract phrases and filter for motivational content
    all_phrases = set()
    for sent in final_sentences:
        for token in sent:
            if "_" in token:
                all_phrases.add(token)
    
    # Filter for truly meaningful motivational phrases
    motivational_phrases = filter_for_motivational_content(all_phrases)
    
    print(f"Total phrases found: {len(all_phrases)}")
    print(f"Motivational phrases after filtering: {len(motivational_phrases)}")
    print(f"Sample motivational phrases: {sorted(list(motivational_phrases))[:10]}")
    
    phrase_docs = [" ".join(sent) for sent in final_sentences]
    return phrase_docs, motivational_phrases, domain_terms_removed

def filter_for_motivational_content(phrases):
    """Filter phrases to focus on motivational content only"""
    
    # Motivational phrase patterns we want to keep
    MOTIVATIONAL_PATTERNS = {
        'helping_orientation': [
            'helping_people', 'help_people', 'helping_others', 'support_people',
            'trying_help', 'want_help', 'love_helping', 'enjoy_helping'
        ],
        'family_influence': [
            'family_member', 'family_experience', 'family_background', 'family_history',
            'parents_supportive', 'mom_dad', 'grew_up', 'childhood_experience',
            'personal_experience', 'life_experience', 'own_experience'
        ],
        'career_motivation': [
            'career_path', 'career_choice', 'future_career', 'professional_path',
            'job_security', 'stable_career', 'meaningful_work', 'fulfilling_career'
        ],
        'personal_qualities': [
            'good_listener', 'people_person', 'caring_person', 'empathetic_person',
            'patient_person', 'understanding_person', 'supportive_person'
        ],
        'impact_motivation': [
            'make_difference', 'making_difference', 'positive_impact', 'change_lives',
            'help_community', 'serve_others', 'give_back', 'meaningful_impact'
        ],
        'responsibility_awareness': [
            'big_responsibility', 'huge_responsibility', 'serious_responsibility',
            'important_work', 'challenging_work', 'difficult_work'
        ],
        'emotional_connection': [
            'personal_connection', 'emotional_connection', 'deep_connection',
            'emotionally_invested', 'passionate_about', 'care_deeply'
        ],
        'learning_growth': [
            'always_learning', 'continuous_learning', 'personal_growth',
            'professional_development', 'skill_development', 'knowledge_building'
        ]
    }
    
    # Flatten motivational patterns
    prioritized_motivational = set()
    for category in MOTIVATIONAL_PATTERNS.values():
        prioritized_motivational.update(category)
    
    # Generic phrases to still avoid (non-motivational)
    STILL_GENERIC = {
        'little_bit', 'pretty_much', 'kind_like', 'sort_like', 'lot_like',
        'right_now', 'long_time', 'first_time', 'every_day', 'all_time',
        'really_good', 'pretty_good', 'sounds_good', 'good_idea',
        'high_school', 'went_school', 'back_school', 'school_school',
        'one_thing', 'other_thing', 'anything_like', 'something_like'
    }
    
    motivational_phrases = set()
    
    for phrase in phrases:
        # Always keep prioritized motivational phrases
        if phrase in prioritized_motivational:
            motivational_phrases.add(phrase)
            continue
            
        # Skip still-generic phrases
        if phrase in STILL_GENERIC:
            continue
        
        # Check for motivational content indicators
        words = phrase.split('_')
        
        # Motivational content indicators (underlying motivations)
        motivational_indicators = [
            'help', 'helping', 'support', 'care', 'caring', 'serve', 'service',
            'family', 'personal', 'experience', 'background', 'history', 'grew',
            'parents', 'mom', 'dad', 'childhood', 'life', 'own',
            'career', 'path', 'choice', 'future', 'professional', 'work',
            'difference', 'impact', 'change', 'community', 'lives', 'meaningful',
            'responsibility', 'important', 'serious', 'challenging', 'difficult',
            'connection', 'emotional', 'passionate', 'invested', 'deeply',
            'learning', 'growth', 'development', 'skill', 'knowledge',
            'listener', 'person', 'empathetic', 'patient', 'understanding',
            'stable', 'security', 'fulfilling', 'rewarding', 'satisfying'
        ]
        
        # Keep if contains motivational indicators and is substantive
        if any(any(indicator in word for indicator in motivational_indicators) for word in words):
            # Must have at least one substantial word (4+ characters)
            substantive_words = [w for w in words if len(w) >= 4]
            if len(substantive_words) >= 1:
                motivational_phrases.add(phrase)
    
    return motivational_phrases

def create_motivation_focused_features(docs, motivational_phrases):
    """Create features focused on underlying motivations"""
    print("\n>>> Creating motivation-focused features...")
    
    # Group motivational phrases by theme
    motivation_themes = {
        'helping_service': ['helping_people', 'help_people', 'trying_help', 'support_people', 'serve_others'],
        'family_personal': ['family_member', 'family_experience', 'personal_experience', 'parents_supportive'],
        'career_path': ['career_path', 'career_choice', 'future_career', 'meaningful_work'],
        'impact_difference': ['make_difference', 'making_difference', 'positive_impact', 'change_lives'],
        'responsibility': ['big_responsibility', 'important_work', 'serious_responsibility'],
        'personal_qualities': ['good_listener', 'caring_person', 'people_person', 'empathetic_person'],
        'connection': ['personal_connection', 'emotional_connection', 'emotionally_invested']
    }
    
    def motivation_analyzer(text):
        tokens = text.split()
        features = []
        
        for token in tokens:
            if "_" in token and token in motivational_phrases:
                # High weight for motivational phrases
                motivation_weight = 6  # Default high weight
                
                # Extra weight for core motivation themes
                for theme_name, theme_phrases in motivation_themes.items():
                    if token in theme_phrases:
                        if theme_name in ['helping_service', 'family_personal']:
                            motivation_weight = 10  # Highest weight for core motivations
                        elif theme_name in ['impact_difference', 'responsibility']:
                            motivation_weight = 8   # High weight for insight themes
                        break
                
                features.extend([token] * motivation_weight)
                
            elif len(token) >= 5:  # Only substantial single words
                # Check if single word indicates motivation
                motivation_indicators = ['helping', 'family', 'career', 'personal', 'support', 'caring', 'impact']
                if any(indicator in token for indicator in motivation_indicators):
                    features.append(token)
        
        return features
    
    vectorizer = TfidfVectorizer(
        analyzer=motivation_analyzer,
        min_df=2,
        max_df=0.9,
        max_features=120,  # Smaller for focused analysis
        sublinear_tf=True,
        norm='l2'
    )
    
    X = vectorizer.fit_transform(docs)
    vocab = vectorizer.get_feature_names_out()
    
    # Analyze vocabulary
    phrases = [term for term in vocab if "_" in term]
    words = [term for term in vocab if "_" not in term]
    
    print(f"Motivation-focused matrix: {X.shape[0]} docs × {X.shape[1]} features")
    print(f"  - {len(phrases)} motivational phrases ({len(phrases)/len(vocab):.1%})")
    print(f"  - {len(words)} motivational words ({len(words)/len(vocab):.1%})")
    print(f"Key motivational phrases: {[p for p in phrases if any(theme in p for theme in ['help', 'family', 'career', 'support'])]}")
    
    return X, vocab, vectorizer

def run_underlying_motivation_model(X, vocab):
    """Run LDA focused on underlying motivations"""
    print("\n>>> Running underlying motivation model (k=3)...")
    
    lda = LatentDirichletAllocation(
        n_components=3,
        max_iter=1000,
        learning_method="batch", 
        doc_topic_prior=0.03,        # Very sparse topics
        topic_word_prior=0.003,      # Very sparse vocabulary
        random_state=42
    )
    
    doc_topic = lda.fit_transform(X)
    
    # Check topic separation
    assignments = doc_topic.argmax(axis=1)
    topic_distribution = Counter(assignments)
    print(f"Topic distribution: {dict(topic_distribution)}")
    
    return lda, doc_topic

def interpret_motivation_topics(lda, vocab, doc_topic):
    """Interpret topics focusing purely on underlying motivations"""
    print("\n>>> Interpreting underlying motivation topics...")
    
    topics_data = []
    
    for topic_idx in range(3):
        top_indices = lda.components_[topic_idx].argsort()[-20:][::-1]
        all_terms = [vocab[i] for i in top_indices]
        
        # Separate motivational phrases and words
        phrases = [term for term in all_terms if "_" in term]
        words = [term for term in all_terms if "_" not in term]
        
        assignments = doc_topic.argmax(axis=1)
        doc_count = np.sum(assignments == topic_idx)
        
        # Pure motivation interpretation
        interpretation = pure_motivation_interpretation(phrases, words, topic_idx)
        
        topics_data.append({
            'topic_num': topic_idx + 1,
            'interpretation': interpretation,
            'doc_count': doc_count,
            'percentage': (doc_count / len(doc_topic)) * 100,
            'motivation_phrases': phrases[:5],
            'motivation_words': words[:8],
            'all_terms': all_terms[:12]
        })
        
        print(f"\nTopic {topic_idx + 1}: {interpretation}")
        print(f"  Documents: {doc_count} ({(doc_count/len(doc_topic)*100):.1f}%)")
        if phrases:
            print(f"  Motivation phrases: {', '.join(phrases[:4])}")
        print(f"  Motivation words: {', '.join(words[:6])}")
    
    return topics_data

def pure_motivation_interpretation(phrases, words, topic_idx):
    """Interpret based purely on underlying motivations, not field content"""
    
    phrase_content = ' '.join(phrases).lower()
    word_content = ' '.join(words).lower()
    all_content = phrase_content + ' ' + word_content
    
    # Focus on WHY students are interested, not WHAT they're interested in
    
    # Helping/Service motivation
    if any(phrase in phrase_content for phrase in ['helping_people', 'help_people', 'support_people']):
        if any(word in all_content for word in ['family', 'personal', 'experience']):
            return "Service Motivation from Personal Experience"
        else:
            return "Altruistic Service & Helping Motivation"
    
    # Personal/Family background driving interest
    elif any(phrase in phrase_content for phrase in ['family_member', 'family_experience', 'personal_experience']):
        if any(word in all_content for word in ['help', 'helping', 'support']):
            return "Personal Background Inspiring Helping Others"
        else:
            return "Personal & Family Experience Influence"
    
    # Career-focused motivation
    elif any(phrase in phrase_content for phrase in ['career_path', 'career_choice', 'future_career']):
        return "Career Path & Professional Development Focus"
    
    # Impact/difference motivation
    elif any(phrase in phrase_content for phrase in ['make_difference', 'positive_impact', 'change_lives']):
        return "Desire to Make Meaningful Impact"
    
    # Responsibility awareness
    elif any(phrase in phrase_content for phrase in ['big_responsibility', 'important_work']):
        return "Professional Responsibility & Commitment Understanding"
    
    # Personal connection/emotional investment
    elif any(phrase in phrase_content for phrase in ['personal_connection', 'emotional_connection']):
        return "Emotional Connection & Personal Investment"
    
    # Fall back to word-based motivation analysis
    elif any(word in all_content for word in ['helping', 'help', 'support', 'caring']):
        return "Helping & Caring Motivation"
    
    elif any(word in all_content for word in ['family', 'personal', 'parents', 'experience']):
        return "Personal & Family Background Influence"
    
    elif any(word in all_content for word in ['career', 'professional', 'future', 'path']):
        return "Career & Professional Development"
    
    elif any(word in all_content for word in ['people', 'community', 'others', 'lives']):
        return "People-Focused Service Orientation"
    
    else:
        return f"Underlying Motivation {topic_idx + 1}"

def save_final_results(topics_data, corpus_df, doc_topic, motivational_phrases, domain_terms_removed):
    """Save final motivation-focused results"""
    
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
            'motivation_phrases': ', '.join(topic_data['motivation_phrases'][:3]) if topic_data['motivation_phrases'] else 'None'
        })
    
    # Save main results
    topics_df = pd.DataFrame([{
        'Topic': f"Topic {t['topic_num']}: {t['interpretation']}",
        'Documents': t['doc_count'],
        'Percentage': f"{t['percentage']:.1f}%",
        'Motivation_Phrases': ', '.join(t['motivation_phrases'][:3]) if t['motivation_phrases'] else 'None',
        'Motivation_Words': ', '.join(t['motivation_words'][:6])
    } for t in topics_data])
    
    topics_df.to_csv(os.path.join(RESULTS_DIR, "final_motivation_topics.csv"), index=False)
    
    quotes_df = pd.DataFrame(quotes)
    quotes_df.to_csv(os.path.join(RESULTS_DIR, "final_motivation_quotes.csv"), index=False)
    
    # Motivational phrases analysis
    phrases_df = pd.DataFrame({
        'motivational_phrase': sorted(list(motivational_phrases)),
        'word_count': [len(phrase.split('_')) for phrase in sorted(list(motivational_phrases))],
        'motivation_category': [categorize_motivation_phrase(phrase) for phrase in sorted(list(motivational_phrases))]
    })
    phrases_df.to_csv(os.path.join(RESULTS_DIR, "final_motivational_phrases.csv"), index=False)
    
    # Final comprehensive report
    with open(os.path.join(RESULTS_DIR, "final_motivation_analysis_report.txt"), 'w') as f:
        f.write("FINAL MOTIVATION-FOCUSED TOPIC ANALYSIS - SUD COUNSELING RESEARCH\n")
        f.write("=" * 65 + "\n\n")
        
        f.write("METHODOLOGY - PURE MOTIVATION FOCUS:\n")
        f.write("• Complete removal of SUD/mental health domain terms\n")
        f.write("• Focus on underlying motivations, not field content\n")
        f.write("• Semantic weighting of motivational phrases\n")
        f.write("• 3-topic model for clearest motivation themes\n")
        f.write(f"• Domain term instances removed: {domain_terms_removed}\n")
        f.write(f"• Pure motivational phrases discovered: {len(motivational_phrases)}\n\n")
        
        f.write("UNDERLYING MOTIVATION THEMES:\n")
        f.write("-" * 35 + "\n")
        for topic_data in topics_data:
            f.write(f"\n{topic_data['interpretation']} ({topic_data['percentage']:.1f}%)\n")
            if topic_data['motivation_phrases']:
                f.write(f"  Motivation phrases: {', '.join(topic_data['motivation_phrases'][:4])}\n")
            f.write(f"  Motivation words: {', '.join(topic_data['motivation_words'][:6])}\n")
        
        f.write(f"\nMOTIVATIONAL PHRASES BY CATEGORY:\n")
        f.write("-" * 40 + "\n")
        phrase_categories = {}
        for phrase in motivational_phrases:
            cat = categorize_motivation_phrase(phrase)
            if cat not in phrase_categories:
                phrase_categories[cat] = []
            phrase_categories[cat].append(phrase)
        
        for category, phrase_list in sorted(phrase_categories.items()):
            if phrase_list:  # Only show non-empty categories
                f.write(f"{category}: {', '.join(sorted(phrase_list))}\n")
        
        f.write(f"\nKEY INSIGHT:\n")
        f.write("These results reveal WHY students are drawn to SUD counseling,\n")
        f.write("not just WHAT field they're interested in. The analysis shows\n")
        f.write("underlying motivational drivers free from circular field terminology.\n")
    
    print(f"\n✅ Final motivation-focused results saved")
    
    # Display results
    print("\n" + "="*75)
    print("FINAL MOTIVATION-FOCUSED TOPICS - REPRESENTATIVE QUOTES")
    print("="*75)
    
    for quote in quotes:
        print(f"\n>>> {quote['topic']} (P={quote['probability']}) <<<")
        if quote['motivation_phrases'] != 'None':
            print(f"Motivation phrases: {quote['motivation_phrases']}")
        print(f"Quote: {textwrap.fill(quote['text'], 70)}")
    
    return topics_data, quotes

def categorize_motivation_phrase(phrase):
    """Categorize motivational phrases"""
    if any(word in phrase for word in ['help', 'helping', 'support', 'serve']):
        return "Helping_Service_Motivation"
    elif any(word in phrase for word in ['family', 'personal', 'experience', 'parents']):
        return "Personal_Family_Influence" 
    elif any(word in phrase for word in ['career', 'path', 'professional', 'future']):
        return "Career_Development"
    elif any(word in phrase for word in ['difference', 'impact', 'change', 'lives']):
        return "Impact_Motivation"
    elif any(word in phrase for word in ['responsibility', 'important', 'serious']):
        return "Responsibility_Awareness"
    elif any(word in phrase for word in ['connection', 'emotional', 'passionate']):
        return "Emotional_Connection"
    else:
        return "Other_Motivation"

def main():
    """Execute final motivation-focused topic modeling"""
    
    print("🎯 FINAL MOTIVATION-FOCUSED TOPIC MODELING")
    print("=" * 50)
    print("Focus: WHY students are interested (not WHAT field)")
    
    corpus_df = load_focus_group_data()
    original_texts = corpus_df['Text'].astype(str).tolist()
    
    phrase_docs, motivational_phrases, domain_removed = detect_underlying_motivation_phrases(original_texts)
    X, vocab, vectorizer = create_motivation_focused_features(phrase_docs, motivational_phrases)
    lda, doc_topic = run_underlying_motivation_model(X, vocab)
    topics_data, quotes = save_final_results(
        interpret_motivation_topics(lda, vocab, doc_topic),
        corpus_df, doc_topic, motivational_phrases, domain_removed)
    
    print(f"\n🏆 FINAL MOTIVATION ANALYSIS COMPLETE!")
    print(f"📊 3 pure motivation themes (no domain circularity)")
    print(f"🔤 {len(motivational_phrases)} motivational phrases discovered")
    print(f"🚫 {domain_removed} domain term instances removed")
    print("🎯 Focus: Underlying WHY, not surface WHAT")
    print("✨ Results reveal true student motivations!")
    
    return topics_data, quotes, motivational_phrases

if __name__ == "__main__":
    topics, quotes, phrases = main()