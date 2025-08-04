#!/usr/bin/env python3
"""
08_ai_language_patterns.py
Purpose: Use Gemini to find specific language patterns that distinguish groups
Author: AI Assistant, 2025-08-02
"""

import os
import pandas as pd
import json
import re
from pathlib import Path
from datetime import datetime
from collections import Counter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Gemini
try:
    from google import genai
    USE_NEW_SDK = True
except ImportError:
    import google.generativeai as genai
    USE_NEW_SDK = False

def setup_gemini():
    """Initialize Gemini API"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    if USE_NEW_SDK:
        client = genai.Client(api_key=api_key)
        return client
    else:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        return model

def load_text_data():
    """Load the text data grouped by interest level"""
    
    # Load text data
    text_df = pd.read_csv('results/study2/clean_participant_dataset_fewshot.csv')
    
    # Separate by group
    interested_texts = text_df[text_df['ai_label'] == 'INTERESTED']['combined_text'].tolist()
    not_interested_texts = text_df[text_df['ai_label'] == 'NOT_INTERESTED']['combined_text'].tolist()
    
    # Combine all text for each group
    all_interested_text = " ".join(interested_texts)
    all_not_interested_text = " ".join(not_interested_texts)
    
    print(f"Loaded text from {len(interested_texts)} INTERESTED participants")
    print(f"Total words: {len(all_interested_text.split())}")
    print(f"\nLoaded text from {len(not_interested_texts)} NOT_INTERESTED participants")
    print(f"Total words: {len(all_not_interested_text.split())}")
    
    return interested_texts, not_interested_texts, all_interested_text, all_not_interested_text

def basic_word_frequency_analysis(interested_text, not_interested_text):
    """Perform basic word frequency analysis to verify AI findings"""
    
    # Simple word counting (lowercase, remove punctuation)
    def get_word_counts(text):
        words = re.findall(r'\b[a-z]+\b', text.lower())
        return Counter(words)
    
    interested_counts = get_word_counts(interested_text)
    not_interested_counts = get_word_counts(not_interested_text)
    
    # Find words unique to each group (or much more common)
    interested_unique = []
    not_interested_unique = []
    
    # Calculate relative frequencies
    total_interested = sum(interested_counts.values())
    total_not_interested = sum(not_interested_counts.values())
    
    for word in interested_counts:
        if len(word) > 3:  # Skip short words
            int_freq = interested_counts[word] / total_interested
            not_int_freq = not_interested_counts.get(word, 0) / total_not_interested
            
            if int_freq > 2 * not_int_freq and interested_counts[word] >= 5:
                interested_unique.append((word, interested_counts[word], int_freq / max(not_int_freq, 0.00001)))
    
    for word in not_interested_counts:
        if len(word) > 3:  # Skip short words
            not_int_freq = not_interested_counts[word] / total_not_interested
            int_freq = interested_counts.get(word, 0) / total_interested
            
            if not_int_freq > 2 * int_freq and not_interested_counts[word] >= 5:
                not_interested_unique.append((word, not_interested_counts[word], not_int_freq / max(int_freq, 0.00001)))
    
    # Sort by ratio
    interested_unique.sort(key=lambda x: x[2], reverse=True)
    not_interested_unique.sort(key=lambda x: x[2], reverse=True)
    
    return interested_unique[:20], not_interested_unique[:20]

def find_language_patterns_with_gemini(model_or_client, interested_texts, not_interested_texts):
    """Ask Gemini to find specific language patterns"""
    
    # Sample texts for Gemini (first 5 from each group)
    interested_sample = "\n\n---NEW PARTICIPANT---\n\n".join(interested_texts[:5])
    not_interested_sample = "\n\n---NEW PARTICIPANT---\n\n".join(not_interested_texts[:5])
    
    prompt = f"""
You are an expert in discourse analysis examining language patterns in focus group discussions.

I have transcripts from 40 participants discussing their interest in substance use disorder (SUD) counseling careers.
20 are INTERESTED and 20 are NOT INTERESTED.

Here are samples from each group:

INTERESTED PARTICIPANTS (5 samples):
{interested_sample}

NOT INTERESTED PARTICIPANTS (5 samples):
{not_interested_sample}

TASK: Identify specific LANGUAGE PATTERNS that distinguish the two groups.

Focus on:
1. Specific words or phrases used predominantly by one group
2. Emotional tone differences
3. Types of concerns expressed
4. How they talk about careers/future
5. Personal vs. abstract language use

For each pattern, provide:
- The specific language marker (exact words/phrases)
- Which group uses it more
- Frequency estimate
- What it reveals about their mindset

Format as:
PATTERN 1: [Description]
- Marker: [Exact words/phrases]
- Group: [INTERESTED or NOT_INTERESTED]
- Example quotes: [2-3 brief examples]
- Insight: [What this reveals]

Provide 8-10 specific, verifiable language patterns.
"""
    
    if USE_NEW_SDK:
        response = model_or_client.models.generate_content(
            model="gemini-1.5-pro",
            contents=prompt
        )
        patterns_text = response.text
    else:
        response = model_or_client.generate_content(prompt)
        patterns_text = response.text
    
    return patterns_text

def analyze_emotional_language(model_or_client, interested_text, not_interested_text):
    """Specifically analyze emotional language differences"""
    
    prompt = f"""
Analyze the emotional language in these two groups discussing SUD counseling careers.

GROUP 1 - INTERESTED (20 participants combined):
{interested_text[:3000]}...

GROUP 2 - NOT INTERESTED (20 participants combined):
{not_interested_text[:3000]}...

Analyze:
1. Emotional words used by each group
2. Positive vs negative sentiment
3. Confidence vs uncertainty markers
4. Personal emotional investment language

Provide specific counts and examples where possible.
Format results as clear comparisons between groups.
"""
    
    if USE_NEW_SDK:
        response = model_or_client.models.generate_content(
            model="gemini-1.5-pro",
            contents=prompt
        )
        emotional_analysis = response.text
    else:
        response = model_or_client.generate_content(prompt)
        emotional_analysis = response.text
    
    return emotional_analysis

def find_phrase_patterns(model_or_client, interested_texts, not_interested_texts):
    """Look for multi-word phrases that distinguish groups"""
    
    prompt = f"""
Find EXACT PHRASES (2-5 words) that appear multiple times in one group but rarely in the other.

INTERESTED GROUP (20 participants):
{" ".join(interested_texts[:10])}

NOT INTERESTED GROUP (20 participants):
{" ".join(not_interested_texts[:10])}

List the top 10 phrases for each group that:
1. Appear at least 3 times in that group
2. Are rare or absent in the other group
3. Are meaningful (not just filler phrases)

Format:
INTERESTED GROUP PHRASES:
1. "[exact phrase]" - appears X times (example context)
2. ...

NOT INTERESTED GROUP PHRASES:
1. "[exact phrase]" - appears X times (example context)
2. ...
"""
    
    if USE_NEW_SDK:
        response = model_or_client.models.generate_content(
            model="gemini-1.5-pro",
            contents=prompt
        )
        phrases = response.text
    else:
        response = model_or_client.generate_content(prompt)
        phrases = response.text
    
    return phrases

def main():
    """Main execution function"""
    print("🔤 AI Language Pattern Analysis")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Setup
    model = setup_gemini()
    interested_texts, not_interested_texts, all_interested, all_not_interested = load_text_data()
    
    # Basic frequency analysis
    print("\n📊 Basic Word Frequency Analysis...")
    interested_words, not_interested_words = basic_word_frequency_analysis(all_interested, all_not_interested)
    
    print("\nWords more common in INTERESTED group:")
    for word, count, ratio in interested_words[:10]:
        print(f"  '{word}': {count} times (ratio: {ratio:.1f}x)")
    
    print("\nWords more common in NOT_INTERESTED group:")
    for word, count, ratio in not_interested_words[:10]:
        print(f"  '{word}': {count} times (ratio: {ratio:.1f}x)")
    
    # AI language pattern analysis
    print("\n🤖 Finding language patterns with AI...")
    patterns = find_language_patterns_with_gemini(model, interested_texts, not_interested_texts)
    
    print("\n" + "="*70)
    print("LANGUAGE PATTERNS DISCOVERED:")
    print("="*70)
    print(patterns)
    
    # Emotional language analysis
    print("\n💭 Analyzing emotional language...")
    emotional = analyze_emotional_language(model, all_interested, all_not_interested)
    
    print("\n" + "="*70)
    print("EMOTIONAL LANGUAGE ANALYSIS:")
    print("="*70)
    print(emotional)
    
    # Phrase patterns
    print("\n📝 Finding distinguishing phrases...")
    phrases = find_phrase_patterns(model, interested_texts, not_interested_texts)
    
    print("\n" + "="*70)
    print("PHRASE PATTERNS:")
    print("="*70)
    print(phrases)
    
    # Save all results
    output_dir = Path('results/study2/ai_patterns')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    with open(output_dir / 'language_patterns.txt', 'w') as f:
        f.write(f"Language Pattern Analysis Results\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"{'='*70}\n\n")
        
        f.write("BASIC FREQUENCY ANALYSIS:\n")
        f.write("\nWords more common in INTERESTED:\n")
        for word, count, ratio in interested_words:
            f.write(f"  '{word}': {count} times (ratio: {ratio:.1f}x)\n")
        
        f.write("\nWords more common in NOT_INTERESTED:\n")
        for word, count, ratio in not_interested_words:
            f.write(f"  '{word}': {count} times (ratio: {ratio:.1f}x)\n")
        
        f.write(f"\n\n{'='*70}\n\n")
        f.write("AI-DISCOVERED PATTERNS:\n")
        f.write(patterns)
        
        f.write(f"\n\n{'='*70}\n\n")
        f.write("EMOTIONAL LANGUAGE:\n")
        f.write(emotional)
        
        f.write(f"\n\n{'='*70}\n\n")
        f.write("PHRASE PATTERNS:\n")
        f.write(phrases)
    
    # Save structured results
    results = {
        'timestamp': datetime.now().isoformat(),
        'basic_frequency': {
            'interested_words': [(w, c, float(r)) for w, c, r in interested_words],
            'not_interested_words': [(w, c, float(r)) for w, c, r in not_interested_words]
        },
        'ai_patterns': patterns,
        'emotional_analysis': emotional,
        'phrase_patterns': phrases
    }
    
    with open(output_dir / 'language_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_dir}")

if __name__ == "__main__":
    main()