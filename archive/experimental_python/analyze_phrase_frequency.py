#!/usr/bin/env python3
"""
Analyze Phrase Frequency by Topic
Extract and rank the most common phrases within each distinct theme
"""

import glob, os, re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from collections import Counter, defaultdict

# Configuration
DATA_DIR = "../../data"
RESULTS_DIR = "../../results/"

def load_and_process_data():
    """Load data and recreate the optimal model"""
    print("🔄 Recreating optimal balanced model to analyze phrase frequencies...")
    
    # Load data
    paths = glob.glob(os.path.join(DATA_DIR, "*_Focus_Group_full*.csv"))
    rows = []
    for p in paths:
        df = pd.read_csv(p)
        df = df[~df["Speaker"].astype(str).str.match(r"^[A-Z]{2,3}$")]
        df = df[df["Text"].str.split().str.len() >= 8]
        df["session"] = os.path.basename(p)
        rows.append(df)
    
    corpus_df = pd.concat(rows, ignore_index=True)
    original_texts = corpus_df['Text'].astype(str).tolist()
    
    # Strategic filtering (same as optimal model)
    STRATEGIC_REMOVAL = [
        'counselor', 'counselors', 'counseling', 
        'therapist', 'therapists', 'therapy',
        'mental_health', 'substance_abuse', 'substance_use',
        'psychologist', 'psychiatrist', 'social_worker'
    ]
    
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
    
    # Process texts
    cleaned_sentences = []
    for text in original_texts:
        text = text.lower()
        for term in STRATEGIC_REMOVAL:
            text = re.sub(rf'\b{re.escape(term)}\b', ' ', text)
        
        tokens = simple_preprocess(text, deacc=True, min_len=3, max_len=25)
        meaningful_tokens = [token for token in tokens 
                           if (token not in STOP_WORDS and len(token) >= 3 and 
                               not token.isdigit() and 
                               token not in ['thing', 'things', 'stuff', 'whatever', 'really'])]
        
        if len(meaningful_tokens) >= 5:
            cleaned_sentences.append(meaningful_tokens)
    
    # Phrase detection
    bigram_model = Phrases(cleaned_sentences, min_count=3, threshold=0.4, delimiter="_", scoring='npmi')
    bigram_phraser = Phraser(bigram_model)
    bigram_sentences = [bigram_phraser[sent] for sent in cleaned_sentences]
    
    trigram_model = Phrases(bigram_sentences, min_count=2, threshold=0.3, delimiter="_", scoring='npmi')
    trigram_phraser = Phraser(trigram_model)
    final_sentences = [trigram_phraser[sent] for sent in bigram_sentences]
    
    # Create documents with phrases
    phrase_docs = [" ".join(sent) for sent in final_sentences]
    
    # Extract all phrases
    all_phrases = set()
    for sent in final_sentences:
        for token in sent:
            if "_" in token:
                all_phrases.add(token)
    
    return phrase_docs, corpus_df, list(all_phrases), final_sentences

def create_topic_model_for_analysis(phrase_docs):
    """Create the same topic model as optimal balanced version"""
    
    def balanced_analyzer(text):
        tokens = text.split()
        features = []
        
        for token in tokens:
            if "_" in token:
                n_words = len(token.split("_"))
                high_value_patterns = ['help', 'family', 'career', 'personal', 'responsibility', 'connection']
                if any(pattern in token for pattern in high_value_patterns):
                    weight = min(n_words * 3, 8)
                else:
                    weight = min(n_words * 2, 6)
                features.extend([token] * weight)
            elif len(token) >= 4:
                meaningful_single = ['helping', 'family', 'personal', 'career', 'support', 'people', 'important']
                if any(meaningful in token for meaningful in meaningful_single):
                    features.append(token)
        
        return features
    
    vectorizer = TfidfVectorizer(
        analyzer=balanced_analyzer,
        min_df=2,
        max_df=0.85,
        max_features=180,
        sublinear_tf=True,
        norm='l2'
    )
    
    X = vectorizer.fit_transform(phrase_docs)
    vocab = vectorizer.get_feature_names_out()
    
    # LDA model (same parameters as optimal)
    lda = LatentDirichletAllocation(
        n_components=4,
        max_iter=800,
        learning_method="batch",
        doc_topic_prior=0.1,
        topic_word_prior=0.01,
        random_state=42
    )
    
    doc_topic = lda.fit_transform(X)
    
    return lda, doc_topic, vocab, vectorizer, X

def analyze_phrase_frequencies_by_topic(lda, doc_topic, vocab, final_sentences, corpus_df):
    """Analyze phrase frequencies within each topic"""
    print("\n📊 Analyzing phrase frequencies by topic...")
    
    # Get topic assignments
    topic_assignments = doc_topic.argmax(axis=1)
    
    # Topic labels from our optimal model
    topic_labels = [
        "Helping People through Personal Experience",
        "Personal Experience & Background", 
        "Career Planning & Considerations",
        "Professional Field Considerations"
    ]
    
    # Count phrases within each topic
    topic_phrase_counts = defaultdict(lambda: defaultdict(int))
    topic_doc_counts = defaultdict(int)
    
    for doc_idx, topic_id in enumerate(topic_assignments):
        topic_doc_counts[topic_id] += 1
        
        # Count phrases in this document
        if doc_idx < len(final_sentences):
            sent = final_sentences[doc_idx]
            for token in sent:
                if "_" in token:  # It's a phrase
                    topic_phrase_counts[topic_id][token] += 1
    
    # Analyze each topic
    topic_analyses = []
    
    for topic_id in range(4):
        print(f"\n{'='*60}")
        print(f"TOPIC {topic_id + 1}: {topic_labels[topic_id]}")
        print(f"{'='*60}")
        print(f"Documents in this topic: {topic_doc_counts[topic_id]}")
        
        # Get phrase counts for this topic
        phrase_counts = topic_phrase_counts[topic_id]
        
        if phrase_counts:
            # Sort by frequency
            sorted_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\nTop 15 Most Common Phrases:")
            print("-" * 40)
            
            top_phrases_data = []
            for i, (phrase, count) in enumerate(sorted_phrases[:15], 1):
                percentage = (count / sum(phrase_counts.values())) * 100
                print(f"{i:2d}. {phrase:<25} | {count:3d} times ({percentage:4.1f}%)")
                
                top_phrases_data.append({
                    'rank': i,
                    'phrase': phrase,
                    'frequency': count,
                    'percentage': round(percentage, 1),
                    'word_count': len(phrase.split('_'))
                })
            
            # Categorize phrases by type
            phrase_categories = categorize_phrases_by_meaning(sorted_phrases[:15])
            
            print(f"\nPhrase Categories:")
            print("-" * 25)
            for category, phrases in phrase_categories.items():
                if phrases:
                    phrase_list = [f"{phrase} ({count})" for phrase, count in phrases]
                    print(f"{category}: {', '.join(phrase_list)}")
            
            topic_analyses.append({
                'topic_id': topic_id + 1,
                'topic_label': topic_labels[topic_id],
                'doc_count': topic_doc_counts[topic_id],
                'total_phrases': sum(phrase_counts.values()),
                'unique_phrases': len(phrase_counts),
                'top_phrases': top_phrases_data,
                'phrase_categories': phrase_categories
            })
        else:
            print("No phrases found in this topic.")
            topic_analyses.append({
                'topic_id': topic_id + 1,
                'topic_label': topic_labels[topic_id],
                'doc_count': topic_doc_counts[topic_id],
                'total_phrases': 0,
                'unique_phrases': 0,
                'top_phrases': [],
                'phrase_categories': {}
            })
    
    return topic_analyses

def categorize_phrases_by_meaning(sorted_phrases):
    """Categorize phrases by their semantic meaning"""
    
    categories = {
        'Helping & Service': [],
        'Family & Personal': [],
        'Career & Professional': [],
        'Responsibility & Awareness': [],
        'Field & Medical': [],
        'Educational': [],
        'Emotional & Connection': [],
        'Other': []
    }
    
    for phrase, count in sorted_phrases:
        categorized = False
        
        if any(word in phrase for word in ['help', 'helping', 'support', 'trying']):
            categories['Helping & Service'].append((phrase, count))
            categorized = True
        elif any(word in phrase for word in ['family', 'personal', 'parents', 'mom', 'dad']):
            categories['Family & Personal'].append((phrase, count))
            categorized = True
        elif any(word in phrase for word in ['career', 'professional', 'job', 'work']):
            categories['Career & Professional'].append((phrase, count))
            categorized = True
        elif any(word in phrase for word in ['responsibility', 'important', 'serious', 'big']):
            categories['Responsibility & Awareness'].append((phrase, count))
            categorized = True
        elif any(word in phrase for word in ['medical', 'field', 'med', 'nursing']):
            categories['Field & Medical'].append((phrase, count))
            categorized = True
        elif any(word in phrase for word in ['school', 'education', 'learning', 'college']):
            categories['Educational'].append((phrase, count))
            categorized = True
        elif any(word in phrase for word in ['emotional', 'connection', 'feeling', 'invested']):
            categories['Emotional & Connection'].append((phrase, count))
            categorized = True
        
        if not categorized:
            categories['Other'].append((phrase, count))
    
    return categories

def save_phrase_frequency_analysis(topic_analyses):
    """Save detailed phrase frequency analysis"""
    
    # Create comprehensive CSV with all phrase data
    all_phrase_data = []
    for analysis in topic_analyses:
        for phrase_data in analysis['top_phrases']:
            all_phrase_data.append({
                'topic_id': analysis['topic_id'],
                'topic_label': analysis['topic_label'],
                'phrase_rank': phrase_data['rank'],
                'phrase': phrase_data['phrase'],
                'frequency': phrase_data['frequency'],
                'percentage_in_topic': phrase_data['percentage'],
                'word_count': phrase_data['word_count']
            })
    
    phrase_df = pd.DataFrame(all_phrase_data)
    phrase_df.to_csv(os.path.join(RESULTS_DIR, "phrase_frequency_by_topic.csv"), index=False)
    
    # Create topic summary
    topic_summary_data = []
    for analysis in topic_analyses:
        top_5_phrases = [p['phrase'] for p in analysis['top_phrases'][:5]]
        topic_summary_data.append({
            'Topic': f"Topic {analysis['topic_id']}: {analysis['topic_label']}",
            'Documents': analysis['doc_count'],
            'Total_Phrase_Instances': analysis['total_phrases'],
            'Unique_Phrases': analysis['unique_phrases'],
            'Top_5_Phrases': ', '.join(top_5_phrases)
        })
    
    summary_df = pd.DataFrame(topic_summary_data)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "topic_phrase_summary.csv"), index=False)
    
    # Create detailed report
    with open(os.path.join(RESULTS_DIR, "phrase_frequency_analysis_report.txt"), 'w') as f:
        f.write("PHRASE FREQUENCY ANALYSIS BY TOPIC - SUD COUNSELING RESEARCH\n")
        f.write("=" * 65 + "\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("• Analyzed phrase frequencies within each topic cluster\n")
        f.write("• Ranked phrases by frequency within topic\n")
        f.write("• Categorized phrases by semantic meaning\n")
        f.write("• Based on optimal balanced 4-topic model\n\n")
        
        for analysis in topic_analyses:
            f.write(f"TOPIC {analysis['topic_id']}: {analysis['topic_label']}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Documents: {analysis['doc_count']}\n")
            f.write(f"Total phrase instances: {analysis['total_phrases']}\n")
            f.write(f"Unique phrases: {analysis['unique_phrases']}\n\n")
            
            f.write("Top 10 Most Frequent Phrases:\n")
            for i, phrase_data in enumerate(analysis['top_phrases'][:10], 1):
                f.write(f"{i:2d}. {phrase_data['phrase']:<25} | {phrase_data['frequency']:3d} times ({phrase_data['percentage']:4.1f}%)\n")
            
            f.write(f"\nPhrase Categories:\n")
            for category, phrases in analysis['phrase_categories'].items():
                if phrases:
                    phrase_list = [phrase for phrase, count in phrases]
                    f.write(f"  {category}: {', '.join(phrase_list[:5])}\n")
            f.write("\n")
    
    print(f"\n✅ Phrase frequency analysis saved to {RESULTS_DIR}")
    print("Files created:")
    print("  • phrase_frequency_by_topic.csv")
    print("  • topic_phrase_summary.csv") 
    print("  • phrase_frequency_analysis_report.txt")

def create_cross_topic_phrase_comparison(topic_analyses):
    """Compare phrases across topics to find unique vs shared patterns"""
    print(f"\n🔍 Cross-Topic Phrase Comparison:")
    print("=" * 45)
    
    # Collect all phrases and their topic associations
    phrase_topic_map = defaultdict(list)
    
    for analysis in topic_analyses:
        for phrase_data in analysis['top_phrases']:
            phrase = phrase_data['phrase']
            topic_id = analysis['topic_id']
            frequency = phrase_data['frequency']
            phrase_topic_map[phrase].append((topic_id, frequency))
    
    # Find unique vs shared phrases
    unique_phrases = {phrase: topics for phrase, topics in phrase_topic_map.items() if len(topics) == 1}
    shared_phrases = {phrase: topics for phrase, topics in phrase_topic_map.items() if len(topics) > 1}
    
    print(f"\nUnique Phrases (appear in only one topic):")
    print("-" * 40)
    for topic_id in range(1, 5):
        topic_unique = [phrase for phrase, topics in unique_phrases.items() if topics[0][0] == topic_id]
        if topic_unique:
            topic_label = topic_analyses[topic_id-1]['topic_label']
            print(f"\nTopic {topic_id} ({topic_label}):")
            for phrase in topic_unique[:8]:  # Top 8 unique phrases
                freq = unique_phrases[phrase][0][1]
                print(f"  • {phrase} ({freq} times)")
    
    if shared_phrases:
        print(f"\nShared Phrases (appear in multiple topics):")
        print("-" * 40)
        for phrase, topics in list(shared_phrases.items())[:10]:  # Top 10 shared
            topic_info = ", ".join([f"T{topic_id}({freq})" for topic_id, freq in topics])
            print(f"  • {phrase}: {topic_info}")

def main():
    """Execute phrase frequency analysis"""
    
    print("📊 PHRASE FREQUENCY ANALYSIS BY TOPIC")
    print("=" * 45)
    
    # Load and process data
    phrase_docs, corpus_df, all_phrases, final_sentences = load_and_process_data()
    
    # Create topic model
    lda, doc_topic, vocab, vectorizer, X = create_topic_model_for_analysis(phrase_docs)
    
    # Analyze phrase frequencies by topic
    topic_analyses = analyze_phrase_frequencies_by_topic(lda, doc_topic, vocab, final_sentences, corpus_df)
    
    # Cross-topic comparison
    create_cross_topic_phrase_comparison(topic_analyses)
    
    # Save results
    save_phrase_frequency_analysis(topic_analyses)
    
    print(f"\n🏆 PHRASE FREQUENCY ANALYSIS COMPLETE!")
    print(f"📈 Analyzed {len(topic_analyses)} topics with detailed phrase rankings")
    print("📋 Results show most common phrases within each distinct theme")
    
    return topic_analyses

if __name__ == "__main__":
    analysis = main()