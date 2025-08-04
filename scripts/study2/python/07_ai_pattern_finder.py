#!/usr/bin/env python3
"""
07_ai_pattern_finder.py
Purpose: Use Gemini to find clear, replicable patterns between interested/not interested
Author: AI Assistant, 2025-08-01
"""

import os
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
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

def prepare_data_for_analysis():
    """Load and prepare the complete dataset for pattern analysis"""
    
    # Load merged data with demographics and labels
    merged_df = pd.read_csv('results/study2/merged_demographics_interest.csv')
    
    # Load text data
    text_df = pd.read_csv('results/study2/clean_participant_dataset_fewshot.csv')
    
    # Merge everything together
    full_df = merged_df.merge(
        text_df[['participant_id', 'combined_text', 'num_utterances']], 
        on='participant_id'
    )
    
    print(f"Loaded data for {len(full_df)} participants")
    print(f"Interested: {sum(full_df['ai_label'] == 'INTERESTED')}")
    print(f"Not Interested: {sum(full_df['ai_label'] == 'NOT_INTERESTED')}")
    
    return full_df

def create_participant_profiles(df):
    """Create readable profiles for each participant"""
    profiles = []
    
    for _, row in df.iterrows():
        profile = f"""
Participant {row['participant_id']} - {row['ai_label']}
Demographics:
- Age: {row['Age']}
- Gender: {'Female' if row['Gener Identity'] == 5 else 'Male' if row['Gener Identity'] == 7 else 'Other'}
- Race: {row['Race']}
- Year in school: {row['Year_in_school']}
- Parent education: {row['Parent_highest_level_education']}

Experience:
- Mental health treatment: {row['Mental_health_treatment']}
- Family/friend substance treatment: {row['Family_friend_substance_use_treatment']}
- Currently employed: {row['Current_employement']}

Environment:
- Safety of area grew up: {row['Safety_area_grew_up']}
- Household income: {row['Household_income']}
- Frequency talk to close connections: {row['Frequency_talk_to_close_connections']}

What they said ({row['num_utterances']} utterances):
{row['combined_text'][:1000]}...
"""
        profiles.append(profile)
    
    return profiles

def find_patterns_with_gemini(model_or_client, profiles, df):
    """Send profiles to Gemini and ask for clear patterns"""
    
    # Create the data summary
    interested_profiles = [p for i, p in enumerate(profiles) if df.iloc[i]['ai_label'] == 'INTERESTED']
    not_interested_profiles = [p for i, p in enumerate(profiles) if df.iloc[i]['ai_label'] == 'NOT_INTERESTED']
    
    prompt = f"""
You are an expert researcher analyzing focus group data about student interest in substance use disorder (SUD) counseling careers.

I have data from 40 participants - 20 who are INTERESTED in SUD counseling careers and 20 who are NOT INTERESTED.

Here are the profiles for all participants:

INTERESTED PARTICIPANTS (20 total):
{"".join(interested_profiles[:5])}
[... and 15 more similar profiles]

NOT INTERESTED PARTICIPANTS (20 total):
{"".join(not_interested_profiles[:5])}
[... and 15 more similar profiles]

TASK: Identify the 5 CLEAREST and MOST REPLICABLE patterns that distinguish interested from not-interested participants.

Requirements:
1. Patterns must be CONCRETE and OBSERVABLE (things we can count or clearly identify)
2. Each pattern should work for at least 80% of the relevant group
3. Focus on patterns that combine demographics with what they say
4. Be specific about numbers, words, or combinations

Format your response as:

PATTERN 1: [Clear description]
- Accuracy: X out of Y participants (Z%)
- How to identify: [Specific criteria]
- Example quotes: [Brief examples]

PATTERN 2: [Clear description]
...and so on for 5 patterns

Focus on patterns that would be easy to replicate in future studies.
"""
    
    # Get response from Gemini
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

def analyze_specific_patterns(model_or_client, df):
    """Ask Gemini to look for specific types of replicable patterns"""
    
    # Prepare structured data
    data_summary = []
    for _, row in df.iterrows():
        data_summary.append({
            'id': row['participant_id'],
            'label': row['ai_label'],
            'has_mh_treatment': row['Mental_health_treatment'] > 1,
            'has_family_su_treatment': row['Family_friend_substance_use_treatment'] > 1,
            'is_sophomore': row['Year_in_school'] == 2,
            'is_employed': row['Current_employement'] == 1,
            'mentions_helping': 'help' in row['combined_text'].lower(),
            'mentions_money': 'money' in row['combined_text'].lower() or 'salary' in row['combined_text'].lower(),
            'mentions_career': 'career' in row['combined_text'].lower(),
            'text_length': len(row['combined_text']),
            'num_utterances': row['num_utterances']
        })
    
    prompt = f"""
Analyze this structured data from 40 participants about SUD counseling career interest.

DATA:
{json.dumps(data_summary, indent=2)}

Find the TOP 5 RULES that predict whether someone is INTERESTED or NOT_INTERESTED.

Example of a good rule:
"If has_mh_treatment=True AND mentions_helping=True, then 90% are INTERESTED"

For each rule, provide:
1. The exact rule (using the data fields)
2. How many participants it applies to
3. Accuracy rate
4. Why this pattern might exist

Focus on rules with at least 80% accuracy that apply to at least 5 participants.
"""
    
    if USE_NEW_SDK:
        response = model_or_client.models.generate_content(
            model="gemini-1.5-pro",
            contents=prompt
        )
        rules_text = response.text
    else:
        response = model_or_client.generate_content(prompt)
        rules_text = response.text
    
    return rules_text

def main():
    """Main execution function"""
    print("🔍 AI Pattern Finder for SUD Counseling Interest")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Setup
    model = setup_gemini()
    df = prepare_data_for_analysis()
    
    # Create profiles
    print("\n📝 Creating participant profiles...")
    profiles = create_participant_profiles(df)
    
    # Find general patterns
    print("\n🤖 Asking Gemini to find clear patterns...")
    patterns = find_patterns_with_gemini(model, profiles, df)
    
    print("\n" + "="*70)
    print("DISCOVERED PATTERNS:")
    print("="*70)
    print(patterns)
    
    # Find specific rules
    print("\n🔍 Looking for specific replicable rules...")
    rules = analyze_specific_patterns(model, df)
    
    print("\n" + "="*70)
    print("REPLICABLE RULES:")
    print("="*70)
    print(rules)
    
    # Save results
    output_dir = Path('results/study2/ai_patterns')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'discovered_patterns.txt', 'w') as f:
        f.write(f"AI Pattern Discovery Results\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"{'='*70}\n\n")
        f.write("GENERAL PATTERNS:\n")
        f.write(patterns)
        f.write(f"\n\n{'='*70}\n\n")
        f.write("SPECIFIC RULES:\n")
        f.write(rules)
    
    print(f"\n✅ Results saved to: {output_dir / 'discovered_patterns.txt'}")
    
    # Also save a structured version for further analysis
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_participants': len(df),
        'n_interested': sum(df['ai_label'] == 'INTERESTED'),
        'n_not_interested': sum(df['ai_label'] == 'NOT_INTERESTED'),
        'general_patterns': patterns,
        'specific_rules': rules
    }
    
    with open(output_dir / 'pattern_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 Structured results saved to: {output_dir / 'pattern_results.json'}")

if __name__ == "__main__":
    main()