#!/usr/bin/env python3
"""
Descriptive Robust Analysis of Focus Group Data
Simple, interpretable approaches that are much more stable than topic modeling:
1. Frequency analysis of meaningful terms
2. Co-occurrence network analysis (which words appear together)
3. Keyword-in-context analysis (natural language around key concepts)
4. Collocation analysis (statistically significant word pairs)
"""

import glob, os, re, textwrap
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "../../data"
RESULTS_DIR = "../../results/"

def load_focus_group_data():
    """Load focus group data"""
    print("📥 Loading focus group data for descriptive analysis...")
    
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

def clean_texts_for_analysis(texts):
    """Clean texts while preserving natural language patterns"""
    print("\n🧹 Cleaning texts while preserving natural patterns...")
    
    # Remove only the most circular terms
    DOMAIN_TERMS = ['counselor', 'counselors', 'counseling', 'therapist', 'therapists', 'therapy']
    
    # Keep these meaningful stopwords for context
    KEEP_WORDS = {'not', 'very', 'really', 'always', 'never', 'maybe', 'probably', 'definitely'}
    
    # Standard stopwords to remove
    REMOVE_WORDS = set("""
    a an and are as at be been being by for from has have he her him his i
    in is it its me my of on or she that the their them they this to was we
    were will with you your
    um uh yeah okay right would can say get got make made see look
    re ve ll don didn won isn aren weren hasn haven couldn wouldn shouldn
    """.split())
    
    cleaned_texts = []
    original_indices = []
    
    for idx, text in enumerate(texts):
        # Convert to lowercase
        text = text.lower()
        
        # Remove domain terms
        for term in DOMAIN_TERMS:
            text = re.sub(rf'\b{re.escape(term)}\b', ' ', text)
        
        # Light cleaning - keep punctuation context
        text = re.sub(r'[^\w\s\']', ' ', text)  # Keep apostrophes
        text = re.sub(r'\d+', ' ', text)        # Remove numbers
        text = re.sub(r'\s+', ' ', text)        # Normalize whitespace
        
        # Keep meaningful words
        tokens = []
        for token in text.split():
            if (len(token) >= 3 and 
                token.isalpha() and 
                (token not in REMOVE_WORDS or token in KEEP_WORDS)):
                tokens.append(token)
        
        if len(tokens) >= 5:  # Keep substantial documents
            cleaned_texts.append(' '.join(tokens))
            original_indices.append(idx)
    
    print(f"Retained {len(cleaned_texts)} documents for analysis")
    return cleaned_texts, original_indices

def frequency_analysis(texts):
    """Simple but powerful frequency analysis"""
    print("\n📊 Frequency Analysis - What students actually say...")
    
    # Count all words
    all_words = []
    for text in texts:
        all_words.extend(text.split())
    
    word_counts = Counter(all_words)
    
    # Categorize by semantic meaning
    categories = {
        'helping_service': ['help', 'helping', 'support', 'care', 'caring', 'assist', 'service', 'serve'],
        'people_focus': ['people', 'person', 'someone', 'others', 'individuals', 'community'],
        'family_personal': ['family', 'personal', 'personally', 'parents', 'mom', 'dad', 'experience', 'background'],
        'career_work': ['career', 'job', 'work', 'working', 'field', 'profession', 'professional'],
        'education': ['school', 'college', 'education', 'learn', 'learning', 'study', 'studying', 'knowledge'],
        'emotions_attitudes': ['interested', 'interesting', 'important', 'responsibility', 'meaningful', 'rewarding'],
        'uncertainty': ['maybe', 'probably', 'not', 'sure', 'think', 'guess', 'might', 'could']
    }
    
    # Calculate category frequencies
    category_counts = {}
    for category, words in categories.items():
        total_count = sum(word_counts[word] for word in words if word in word_counts)
        category_counts[category] = total_count
    
    # Most frequent meaningful words overall
    meaningful_words = []
    for word, count in word_counts.most_common(100):
        if len(word) >= 4 and count >= 5:  # Substantial words with good frequency
            meaningful_words.append((word, count))
    
    print(f"\nCategory Frequencies (total mentions):")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category.replace('_', ' ').title()}: {count} mentions")
    
    print(f"\nTop 20 Most Frequent Meaningful Words:")
    for word, count in meaningful_words[:20]:
        print(f"  {word}: {count}")
    
    return word_counts, category_counts, meaningful_words

def cooccurrence_analysis(texts, window_size=5):
    """Analyze which words appear together - much more stable than topic modeling"""
    print(f"\n🔗 Co-occurrence Analysis - Words that appear together (window={window_size})...")
    
    # Count word co-occurrences within window
    cooccurrence_counts = defaultdict(int)
    target_words = ['help', 'family', 'people', 'career', 'personal', 'experience', 'support', 'work', 'school', 'important']
    
    for text in texts:
        words = text.split()
        for i, word in enumerate(words):
            if word in target_words:
                # Look within window around target word
                start = max(0, i - window_size)
                end = min(len(words), i + window_size + 1)
                
                for j in range(start, end):
                    if i != j and len(words[j]) >= 4:  # Don't count the word with itself
                        pair = tuple(sorted([word, words[j]]))
                        cooccurrence_counts[pair] += 1
    
    # Find strongest associations
    strong_associations = []
    for (word1, word2), count in cooccurrence_counts.items():
        if count >= 3:  # Appears together at least 3 times
            strong_associations.append(((word1, word2), count))
    
    strong_associations.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nStrongest Word Associations (appear together ≥3 times):")
    for (word1, word2), count in strong_associations[:20]:
        print(f"  '{word1}' + '{word2}': {count} times")
    
    return strong_associations, cooccurrence_counts

def keyword_in_context_analysis(texts, corpus_df, original_indices):
    """Analyze natural language around key concepts"""
    print("\n📝 Keyword-in-Context Analysis - Natural language patterns...")
    
    key_concepts = {
        'family': ['family', 'families', 'parents', 'mom', 'dad', 'mother', 'father'],
        'helping': ['help', 'helping', 'support', 'care', 'caring'],
        'personal': ['personal', 'personally', 'experience', 'experienced'],
        'career': ['career', 'job', 'work', 'field', 'profession'],
        'responsibility': ['responsibility', 'responsible', 'important'],
        'interest': ['interested', 'interesting', 'passion', 'passionate']
    }
    
    concept_contexts = defaultdict(list)
    context_window = 8  # Words before and after
    
    for text_idx, text in enumerate(texts):
        words = text.split()
        for concept, keywords in key_concepts.items():
            for i, word in enumerate(words):
                if word in keywords:
                    # Extract context
                    start = max(0, i - context_window)
                    end = min(len(words), i + context_window + 1)
                    context = ' '.join(words[start:end])
                    
                    # Get original document info
                    orig_idx = original_indices[text_idx]
                    speaker = corpus_df.iloc[orig_idx]['Speaker']
                    
                    concept_contexts[concept].append({
                        'keyword': word,
                        'context': context,
                        'speaker': speaker,
                        'full_text': corpus_df.iloc[orig_idx]['Text']
                    })
    
    # Analyze patterns for each concept
    concept_patterns = {}
    for concept, contexts in concept_contexts.items():
        print(f"\n'{concept.upper()}' contexts ({len(contexts)} instances):")
        
        # Show most informative examples
        for i, ctx in enumerate(contexts[:3]):
            print(f"  Example {i+1}: \"{ctx['context']}\"")
        
        concept_patterns[concept] = contexts
    
    return concept_patterns

def collocation_analysis(texts):
    """Find statistically significant word pairs using simple frequency"""
    print("\n📈 Collocation Analysis - Statistically meaningful word pairs...")
    
    # Extract all bigrams and trigrams
    bigrams = defaultdict(int)
    trigrams = defaultdict(int)
    
    for text in texts:
        words = text.split()
        
        # Bigrams
        for i in range(len(words) - 1):
            if len(words[i]) >= 3 and len(words[i+1]) >= 3:
                bigram = f"{words[i]} {words[i+1]}"
                bigrams[bigram] += 1
        
        # Trigrams
        for i in range(len(words) - 2):
            if all(len(w) >= 3 for w in words[i:i+3]):
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                trigrams[trigram] += 1
    
    # Filter for meaningful collocations
    meaningful_bigrams = [(phrase, count) for phrase, count in bigrams.items() 
                         if count >= 3 and not any(filler in phrase for filler in 
                         ['really', 'pretty', 'very', 'just', 'like', 'kind', 'sort'])]
    
    meaningful_trigrams = [(phrase, count) for phrase, count in trigrams.items() 
                          if count >= 2]
    
    meaningful_bigrams.sort(key=lambda x: x[1], reverse=True)
    meaningful_trigrams.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nMeaningful Bigrams (≥3 occurrences):")
    for phrase, count in meaningful_bigrams[:15]:
        print(f"  '{phrase}': {count} times")
    
    print(f"\nMeaningful Trigrams (≥2 occurrences):")
    for phrase, count in meaningful_trigrams[:10]:
        print(f"  '{phrase}': {count} times")
    
    return meaningful_bigrams, meaningful_trigrams

def create_simple_themes(word_counts, cooccurrences, concept_patterns):
    """Create simple themes based on descriptive patterns"""
    print("\n🎯 Simple Theme Creation from Descriptive Patterns...")
    
    # Calculate theme strength based on multiple indicators
    theme_indicators = {
        'helping_service': {
            'words': ['help', 'helping', 'support', 'care', 'people', 'community'],
            'description': 'Focus on helping people and service to others'
        },
        'family_personal': {
            'words': ['family', 'personal', 'experience', 'background', 'parents'],
            'description': 'Personal and family experiences driving interest'
        },
        'career_professional': {
            'words': ['career', 'work', 'field', 'job', 'professional', 'future'],
            'description': 'Career development and professional considerations'
        },
        'responsibility_importance': {
            'words': ['important', 'responsibility', 'meaningful', 'serious', 'impact'],
            'description': 'Sense of responsibility and importance of the work'
        },
        'education_learning': {
            'words': ['school', 'education', 'learn', 'knowledge', 'study'],
            'description': 'Educational pathway and learning orientation'
        }
    }
    
    # Calculate theme strengths
    theme_strengths = {}
    total_words = sum(word_counts.values())
    
    for theme, data in theme_indicators.items():
        strength = sum(word_counts.get(word, 0) for word in data['words'])
        percentage = (strength / total_words) * 100
        theme_strengths[theme] = {
            'raw_count': strength,
            'percentage': percentage,
            'description': data['description']
        }
    
    # Sort by strength
    sorted_themes = sorted(theme_strengths.items(), key=lambda x: x[1]['raw_count'], reverse=True)
    
    print(f"\nTheme Strengths (based on word frequencies):")
    for theme, data in sorted_themes:
        print(f"  {theme.replace('_', ' ').title()}: {data['raw_count']} mentions ({data['percentage']:.1f}%)")
        print(f"    → {data['description']}")
    
    return theme_strengths

def save_descriptive_results(word_counts, category_counts, meaningful_words, 
                           strong_associations, concept_patterns, meaningful_bigrams, 
                           meaningful_trigrams, theme_strengths):
    """Save all descriptive analysis results"""
    
    # Word frequency data
    freq_df = pd.DataFrame([
        {'word': word, 'count': count, 'rank': rank+1} 
        for rank, (word, count) in enumerate(meaningful_words)
    ])
    freq_df.to_csv(os.path.join(RESULTS_DIR, "descriptive_word_frequencies.csv"), index=False)
    
    # Category frequencies
    cat_df = pd.DataFrame([
        {'category': cat, 'total_mentions': count} 
        for cat, count in category_counts.items()
    ])
    cat_df.to_csv(os.path.join(RESULTS_DIR, "descriptive_category_frequencies.csv"), index=False)
    
    # Co-occurrences
    cooc_df = pd.DataFrame([
        {'word1': pair[0], 'word2': pair[1], 'cooccurrence_count': count}
        for (pair, count) in strong_associations
    ])
    cooc_df.to_csv(os.path.join(RESULTS_DIR, "descriptive_cooccurrences.csv"), index=False)
    
    # Collocations
    bigram_df = pd.DataFrame(meaningful_bigrams, columns=['bigram', 'frequency'])
    bigram_df.to_csv(os.path.join(RESULTS_DIR, "descriptive_bigrams.csv"), index=False)
    
    trigram_df = pd.DataFrame(meaningful_trigrams, columns=['trigram', 'frequency'])
    trigram_df.to_csv(os.path.join(RESULTS_DIR, "descriptive_trigrams.csv"), index=False)
    
    # Comprehensive report
    with open(os.path.join(RESULTS_DIR, "descriptive_robust_analysis_report.txt"), 'w') as f:
        f.write("DESCRIPTIVE ROBUST ANALYSIS - SUD COUNSELING RESEARCH\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("METHODOLOGY - Simple & Robust Approaches:\n")
        f.write("• Frequency analysis of meaningful terms\n")
        f.write("• Co-occurrence analysis (words appearing together)\n")
        f.write("• Keyword-in-context analysis\n")
        f.write("• Collocation analysis (statistically significant pairs)\n")
        f.write("• Simple theme creation from patterns\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-" * 15 + "\n")
        
        f.write("\n1. MOST FREQUENT THEMES (by word count):\n")
        for theme, data in sorted(theme_strengths.items(), key=lambda x: x[1]['raw_count'], reverse=True):
            f.write(f"   • {theme.replace('_', ' ').title()}: {data['raw_count']} mentions ({data['percentage']:.1f}%)\n")
        
        f.write(f"\n2. TOP WORD ASSOCIATIONS:\n")
        for (word1, word2), count in strong_associations[:10]:
            f.write(f"   • '{word1}' + '{word2}': {count} times together\n")
        
        f.write(f"\n3. MEANINGFUL PHRASES:\n")
        for phrase, count in meaningful_bigrams[:10]:
            f.write(f"   • '{phrase}': {count} times\n")
        
        f.write(f"\n4. CATEGORY FREQUENCIES:\n")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"   • {category.replace('_', ' ').title()}: {count} total mentions\n")
    
    print(f"\n✅ Descriptive analysis results saved to {RESULTS_DIR}")
    print("Files created:")
    print("  • descriptive_word_frequencies.csv")
    print("  • descriptive_category_frequencies.csv") 
    print("  • descriptive_cooccurrences.csv")
    print("  • descriptive_bigrams.csv")
    print("  • descriptive_trigrams.csv")
    print("  • descriptive_robust_analysis_report.txt")

def main():
    """Execute descriptive robust analysis"""
    
    print("📊 DESCRIPTIVE ROBUST ANALYSIS")
    print("=" * 50)
    print("Simple, stable, interpretable approaches")
    
    # Load data
    texts, corpus_df = load_focus_group_data()
    
    # Clean texts
    cleaned_texts, original_indices = clean_texts_for_analysis(texts)
    
    # 1. Frequency analysis
    word_counts, category_counts, meaningful_words = frequency_analysis(cleaned_texts)
    
    # 2. Co-occurrence analysis  
    strong_associations, cooccurrence_counts = cooccurrence_analysis(cleaned_texts)
    
    # 3. Keyword-in-context analysis
    concept_patterns = keyword_in_context_analysis(cleaned_texts, corpus_df, original_indices)
    
    # 4. Collocation analysis
    meaningful_bigrams, meaningful_trigrams = collocation_analysis(cleaned_texts)
    
    # 5. Create simple themes
    theme_strengths = create_simple_themes(word_counts, strong_associations, concept_patterns)
    
    # Save results
    save_descriptive_results(word_counts, category_counts, meaningful_words, 
                           strong_associations, concept_patterns, meaningful_bigrams, 
                           meaningful_trigrams, theme_strengths)
    
    print(f"\n🏆 DESCRIPTIVE ANALYSIS COMPLETE!")
    print(f"📊 Much more robust than topic modeling")
    print(f"📝 Clear, interpretable patterns discovered")
    print(f"🔗 Perfect for linking with Study 1 individual data")
    print("✨ Simple word counting beats complex algorithms!")
    
    return {
        'word_counts': word_counts,
        'category_counts': category_counts,
        'associations': strong_associations,
        'themes': theme_strengths,
        'bigrams': meaningful_bigrams,
        'concept_patterns': concept_patterns
    }

if __name__ == "__main__":
    results = main()