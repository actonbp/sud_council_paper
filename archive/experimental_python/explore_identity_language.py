#!/usr/bin/env python3
"""
Explore Identity Formation Language in Focus Group Data
Data-driven approach to find actual language patterns before building dictionaries
"""

import glob, os, re
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuration
DATA_DIR = "../../data"
RESULTS_DIR = "../../results/"

def load_focus_group_data():
    """Load focus group data for exploration"""
    print("📥 Loading focus group data for identity language exploration...")
    
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

def explore_commitment_uncertainty_language(texts):
    """Find actual words/phrases that indicate commitment vs uncertainty"""
    print("\n🔍 Exploring commitment vs uncertainty language patterns...")
    
    # Look for potential commitment indicators
    commitment_patterns = [
        r'\b(I am|I\'m)\s+\w+',
        r'\b(I will|I\'ll)\s+\w+',
        r'\b(I want to be|I plan to|I\'m going to)\s+\w+',
        r'\b(my career|my field|my profession)\b',
        r'\b(decided|committed|determined|sure|certain)\b',
        r'\b(always|definitely|absolutely|completely)\b'
    ]
    
    # Look for uncertainty indicators
    uncertainty_patterns = [
        r'\b(maybe|might|could|possibly|perhaps)\b',
        r'\b(not sure|unsure|uncertain|don\'t know)\b',
        r'\b(thinking about|considering|wondering|exploring)\b',
        r'\b(haven\'t decided|still deciding|figuring out)\b',
        r'\b(or|versus|vs|instead of|rather than)\b'
    ]
    
    commitment_examples = []
    uncertainty_examples = []
    
    for i, text in enumerate(texts):
        text_lower = text.lower()
        
        # Find commitment language
        for pattern in commitment_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                # Get surrounding context (20 words before and after)
                words = text_lower.split()
                start_pos = max(0, text_lower[:match.start()].count(' ') - 10)
                end_pos = min(len(words), text_lower[:match.end()].count(' ') + 10)
                context = ' '.join(words[start_pos:end_pos])
                
                commitment_examples.append({
                    'doc_id': i,
                    'pattern': pattern,
                    'match': match.group(),
                    'context': context,
                    'full_text': text
                })
        
        # Find uncertainty language
        for pattern in uncertainty_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                words = text_lower.split()
                start_pos = max(0, text_lower[:match.start()].count(' ') - 10)
                end_pos = min(len(words), text_lower[:match.end()].count(' ') + 10)
                context = ' '.join(words[start_pos:end_pos])
                
                uncertainty_examples.append({
                    'doc_id': i,
                    'pattern': pattern,
                    'match': match.group(),
                    'context': context,
                    'full_text': text
                })
    
    print(f"Found {len(commitment_examples)} commitment language instances")
    print(f"Found {len(uncertainty_examples)} uncertainty language instances")
    
    return commitment_examples, uncertainty_examples

def explore_career_field_mentions(texts):
    """Find how students actually talk about different career fields"""
    print("\n🎓 Exploring career field mentions...")
    
    # Look for career field patterns
    career_patterns = [
        r'\b(psychology|psychologist|psychologists)\b',
        r'\b(social work|social worker|social workers)\b',
        r'\b(nursing|nurse|nurses)\b',
        r'\b(medicine|medical|doctor|doctors|physician)\b',
        r'\b(psychiatry|psychiatrist|psychiatrists)\b',
        r'\b(counseling|counselor|counselors)\b',
        r'\b(therapy|therapist|therapists)\b',
        r'\b(mental health)\b',
        r'\b(substance|addiction|drug|alcohol)\b'
    ]
    
    field_mentions = []
    
    for i, text in enumerate(texts):
        text_lower = text.lower()
        
        for pattern in career_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                # Get surrounding context
                words = text_lower.split()
                start_pos = max(0, text_lower[:match.start()].count(' ') - 15)
                end_pos = min(len(words), text_lower[:match.end()].count(' ') + 15)
                context = ' '.join(words[start_pos:end_pos])
                
                field_mentions.append({
                    'doc_id': i,
                    'field': match.group(),
                    'context': context,
                    'full_text': text
                })
    
    # Count field mentions
    field_counts = Counter([mention['field'] for mention in field_mentions])
    print(f"\nField mention frequencies:")
    for field, count in field_counts.most_common(15):
        print(f"  {field}: {count} mentions")
    
    return field_mentions, field_counts

def explore_comparison_language(texts):
    """Find how students compare different options/paths"""
    print("\n⚖️ Exploring comparison and choice language...")
    
    comparison_patterns = [
        r'\b(versus|vs|compared to|rather than|instead of|or|different from)\b',
        r'\b(better than|worse than|more|less)\s+\w+\s+(than)\b',
        r'\b(either|neither|both|all|none)\b',
        r'\b(option|choice|alternative|path|route|way)\b',
        r'\b(choose|pick|select|decide between)\b'
    ]
    
    comparison_examples = []
    
    for i, text in enumerate(texts):
        text_lower = text.lower()
        
        for pattern in comparison_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                words = text_lower.split()
                start_pos = max(0, text_lower[:match.start()].count(' ') - 12)
                end_pos = min(len(words), text_lower[:match.end()].count(' ') + 12)
                context = ' '.join(words[start_pos:end_pos])
                
                comparison_examples.append({
                    'doc_id': i,
                    'comparison_word': match.group(),
                    'context': context,
                    'full_text': text
                })
    
    print(f"Found {len(comparison_examples)} comparison language instances")
    
    # Show most common comparison words
    comparison_counts = Counter([ex['comparison_word'] for ex in comparison_examples])
    print(f"\nMost common comparison words:")
    for word, count in comparison_counts.most_common(10):
        print(f"  {word}: {count} times")
    
    return comparison_examples

def explore_identity_development_phrases(texts):
    """Look for actual phrases about identity, becoming, growing"""
    print("\n🌱 Exploring identity development language...")
    
    identity_patterns = [
        r'\b(I am|I\'m)\s+[a-z]+\s+(person|student|type)\b',
        r'\b(becoming|growing|developing|learning|changing)\b',
        r'\b(see myself|think of myself|consider myself)\b',
        r'\b(the type of person|kind of person|sort of person)\b',
        r'\b(always been|never been|used to be)\b',
        r'\b(interested in|passionate about|drawn to)\b',
        r'\b(good at|bad at|skilled at|talented)\b'
    ]
    
    identity_examples = []
    
    for i, text in enumerate(texts):
        text_lower = text.lower()
        
        for pattern in identity_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                words = text_lower.split()
                start_pos = max(0, text_lower[:match.start()].count(' ') - 10)
                end_pos = min(len(words), text_lower[:match.end()].count(' ') + 10)
                context = ' '.join(words[start_pos:end_pos])
                
                identity_examples.append({
                    'doc_id': i,
                    'identity_phrase': match.group(),
                    'context': context,
                    'full_text': text
                })
    
    print(f"Found {len(identity_examples)} identity development instances")
    
    return identity_examples

def find_high_uncertainty_documents(uncertainty_examples, texts):
    """Find documents with high concentration of uncertainty language"""
    print("\n📊 Finding high-uncertainty documents...")
    
    # Count uncertainty indicators per document
    doc_uncertainty_counts = Counter([ex['doc_id'] for ex in uncertainty_examples])
    
    # Find documents with multiple uncertainty indicators
    high_uncertainty_docs = [(doc_id, count) for doc_id, count in doc_uncertainty_counts.items() if count >= 3]
    high_uncertainty_docs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Found {len(high_uncertainty_docs)} documents with 3+ uncertainty indicators")
    
    # Show examples
    print(f"\nTop uncertainty documents:")
    for doc_id, count in high_uncertainty_docs[:5]:
        print(f"\nDocument {doc_id} ({count} uncertainty indicators):")
        print(f"  Text: {texts[doc_id][:300]}...")
        
        # Show which uncertainty patterns matched
        matching_patterns = [ex['match'] for ex in uncertainty_examples if ex['doc_id'] == doc_id]
        print(f"  Uncertainty words: {', '.join(set(matching_patterns))}")
    
    return high_uncertainty_docs

def save_exploration_results(commitment_examples, uncertainty_examples, field_mentions, comparison_examples, identity_examples):
    """Save all exploration results for review"""
    
    # Save commitment examples
    commitment_df = pd.DataFrame(commitment_examples)
    if not commitment_df.empty:
        commitment_df.to_csv(os.path.join(RESULTS_DIR, "identity_exploration_commitment_examples.csv"), index=False)
    
    # Save uncertainty examples
    uncertainty_df = pd.DataFrame(uncertainty_examples)
    if not uncertainty_df.empty:
        uncertainty_df.to_csv(os.path.join(RESULTS_DIR, "identity_exploration_uncertainty_examples.csv"), index=False)
    
    # Save field mentions
    field_df = pd.DataFrame(field_mentions)
    if not field_df.empty:
        field_df.to_csv(os.path.join(RESULTS_DIR, "identity_exploration_field_mentions.csv"), index=False)
    
    # Save comparison examples
    comparison_df = pd.DataFrame(comparison_examples)
    if not comparison_df.empty:
        comparison_df.to_csv(os.path.join(RESULTS_DIR, "identity_exploration_comparisons.csv"), index=False)
    
    # Save identity examples
    identity_df = pd.DataFrame(identity_examples)
    if not identity_df.empty:
        identity_df.to_csv(os.path.join(RESULTS_DIR, "identity_exploration_identity_phrases.csv"), index=False)
    
    # Create summary report
    with open(os.path.join(RESULTS_DIR, "identity_language_exploration_report.txt"), 'w') as f:
        f.write("IDENTITY FORMATION LANGUAGE EXPLORATION - STUDY 2 DATA\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("PURPOSE:\n")
        f.write("Explore actual language patterns in focus group data to identify\n")
        f.write("commitment vs uncertainty indicators for Study 1-2 linkage analysis\n\n")
        
        f.write("DATA-DRIVEN FINDINGS:\n")
        f.write("-" * 25 + "\n")
        f.write(f"• Commitment language instances: {len(commitment_examples)}\n")
        f.write(f"• Uncertainty language instances: {len(uncertainty_examples)}\n")
        f.write(f"• Career field mentions: {len(field_mentions)}\n")
        f.write(f"• Comparison language instances: {len(comparison_examples)}\n")
        f.write(f"• Identity development phrases: {len(identity_examples)}\n\n")
        
        if uncertainty_examples:
            uncertainty_words = [ex['match'] for ex in uncertainty_examples]
            uncertainty_counts = Counter(uncertainty_words)
            f.write("Most common uncertainty words:\n")
            for word, count in uncertainty_counts.most_common(10):
                f.write(f"  • {word}: {count} times\n")
        
        f.write(f"\nNEXT STEPS:\n")
        f.write("1. Review CSV files to see actual language patterns\n")
        f.write("2. Identify most meaningful indicators from real data\n")
        f.write("3. Build refined dictionaries based on observed patterns\n")
        f.write("4. Test refined approach on subset of data\n")
    
    print(f"\n✅ Exploration results saved to {RESULTS_DIR}")
    print("Files created:")
    print("  • identity_exploration_commitment_examples.csv")
    print("  • identity_exploration_uncertainty_examples.csv")
    print("  • identity_exploration_field_mentions.csv")
    print("  • identity_exploration_comparisons.csv")
    print("  • identity_exploration_identity_phrases.csv")
    print("  • identity_language_exploration_report.txt")

def main():
    """Execute identity language exploration"""
    
    print("🔍 IDENTITY FORMATION LANGUAGE EXPLORATION")
    print("=" * 50)
    print("Data-driven approach to find actual language patterns")
    
    # Load data
    texts, corpus_df = load_focus_group_data()
    
    # Explore different language patterns
    commitment_examples, uncertainty_examples = explore_commitment_uncertainty_language(texts)
    field_mentions, field_counts = explore_career_field_mentions(texts)
    comparison_examples = explore_comparison_language(texts)
    identity_examples = explore_identity_development_phrases(texts)
    
    # Find high uncertainty documents
    high_uncertainty_docs = find_high_uncertainty_documents(uncertainty_examples, texts)
    
    # Save results
    save_exploration_results(commitment_examples, uncertainty_examples, field_mentions, comparison_examples, identity_examples)
    
    print(f"\n🏆 IDENTITY LANGUAGE EXPLORATION COMPLETE!")
    print(f"📊 Found patterns in {len(texts)} utterances")
    print("🔍 Review CSV files to see actual language used")
    print("📝 Use findings to build data-driven dictionaries")
    
    return {
        'commitment_examples': commitment_examples,
        'uncertainty_examples': uncertainty_examples, 
        'field_mentions': field_mentions,
        'comparison_examples': comparison_examples,
        'identity_examples': identity_examples,
        'high_uncertainty_docs': high_uncertainty_docs
    }

if __name__ == "__main__":
    results = main()