#!/usr/bin/env python3
"""
11_pattern_discovery_reliability.py
Purpose: Run pattern discovery twice independently to verify reliability
Author: AI Assistant, 2025-08-02
"""

import os
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import time

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

def load_data():
    """Load merged dataset"""
    merged_df = pd.read_csv('results/study2/merged_demographics_interest.csv')
    text_df = pd.read_csv('results/study2/clean_participant_dataset_fewshot.csv')
    
    full_df = merged_df.merge(
        text_df[['participant_id', 'combined_text', 'num_utterances']], 
        on='participant_id'
    )
    
    return full_df

def create_discovery_prompt(df, run_number):
    """Create prompt for pattern discovery"""
    
    # Separate groups
    interested = df[df['ai_label'] == 'INTERESTED']
    not_interested = df[df['ai_label'] == 'NOT_INTERESTED']
    
    prompt = f"""
You are analyzing data from 40 participants discussing substance use disorder (SUD) counseling careers.
20 are INTERESTED and 20 are NOT_INTERESTED.

This is discovery run #{run_number}. Please analyze the data independently.

TASK: Find the TOP 5 PATTERNS that distinguish interested from not-interested participants.

For each participant, you have:
- Demographics (age, gender, race, year in school, employment, etc.)
- Treatment history (mental health, family substance use)
- What they said in focus groups

INTERESTED GROUP Examples (first 3):
"""
    
    # Add examples
    for i, (_, row) in enumerate(interested.head(3).iterrows()):
        prompt += f"""
Participant {row['participant_id']}:
- Demographics: Age {row['Age']}, Year {row['Year_in_school']}, Employed: {row['Current_employement']}
- MH treatment: {row['Mental_health_treatment']}, Family SU treatment: {row['Family_friend_substance_use_treatment']}
- Said: "{row['combined_text'][:300]}..."
"""
    
    prompt += "\n\nNOT INTERESTED GROUP Examples (first 3):"
    
    for i, (_, row) in enumerate(not_interested.head(3).iterrows()):
        prompt += f"""
Participant {row['participant_id']}:
- Demographics: Age {row['Age']}, Year {row['Year_in_school']}, Employed: {row['Current_employement']}
- MH treatment: {row['Mental_health_treatment']}, Family SU treatment: {row['Family_friend_substance_use_treatment']}
- Said: "{row['combined_text'][:300]}..."
"""
    
    prompt += """

Based on ALL 40 participants, identify patterns that:
1. Apply to at least 5 participants
2. Have at least 75% accuracy
3. Are SPECIFIC and MEASURABLE

Format each pattern as:
PATTERN [number]: [Description]
- Rule: [Specific criteria]
- Accuracy: [X/Y participants = Z%]
- Group: [INTERESTED or NOT_INTERESTED]
"""
    
    return prompt

def run_discovery(model_or_client, df, run_number):
    """Run a single discovery"""
    prompt = create_discovery_prompt(df, run_number)
    
    if USE_NEW_SDK:
        response = model_or_client.models.generate_content(
            model="gemini-1.5-pro",
            contents=prompt
        )
        return response.text
    else:
        response = model_or_client.generate_content(prompt)
        return response.text

def compare_discoveries(discovery1, discovery2):
    """Compare two discovery runs"""
    comparison_prompt = f"""
Compare these two independent pattern discoveries from the same dataset:

DISCOVERY RUN 1:
{discovery1}

DISCOVERY RUN 2:
{discovery2}

TASK:
1. Identify which patterns appear in BOTH runs (even if worded differently)
2. Note which patterns are unique to each run
3. Calculate consistency score (% of patterns that replicated)

Format as:
REPLICATED PATTERNS:
- Pattern: [description]
  Run 1 version: [...]
  Run 2 version: [...]

UNIQUE TO RUN 1:
- [List patterns]

UNIQUE TO RUN 2:
- [List patterns]

CONSISTENCY SCORE: X/10 patterns replicated (Y%)
"""
    
    return comparison_prompt

def main():
    """Main execution"""
    print("🔄 Pattern Discovery Reliability Test")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Setup
    model = setup_gemini()
    df = load_data()
    
    print(f"\nLoaded {len(df)} participants")
    print("Running two independent pattern discoveries...\n")
    
    # Run 1
    print("Discovery Run 1...")
    discovery1 = run_discovery(model, df, 1)
    print("✓ Complete")
    
    # Wait a bit between runs
    time.sleep(2)
    
    # Run 2
    print("\nDiscovery Run 2...")
    discovery2 = run_discovery(model, df, 2)
    print("✓ Complete")
    
    # Compare
    print("\nComparing discoveries...")
    comparison_prompt = compare_discoveries(discovery1, discovery2)
    
    if USE_NEW_SDK:
        response = model.models.generate_content(
            model="gemini-1.5-pro",
            contents=comparison_prompt
        )
        comparison = response.text
    else:
        response = model.generate_content(comparison_prompt)
        comparison = response.text
    
    # Save results
    output_dir = Path('results/study2/ai_patterns')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'discovery_run_1': discovery1,
        'discovery_run_2': discovery2,
        'comparison': comparison
    }
    
    with open(output_dir / 'discovery_reliability_test.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create report
    report = f"""
PATTERN DISCOVERY RELIABILITY TEST
==================================
Generated: {datetime.now()}

This test ran pattern discovery twice independently to verify that patterns
are reliable and not due to random variation in AI responses.

DISCOVERY RUN 1:
================
{discovery1}

DISCOVERY RUN 2:
================
{discovery2}

COMPARISON ANALYSIS:
====================
{comparison}

INTERPRETATION:
===============
Patterns that appear in both runs can be considered more reliable findings.
The consistency score indicates how reproducible our pattern discovery is.
Higher consistency (>70%) suggests robust patterns that reviewers can trust.
"""
    
    with open(output_dir / 'discovery_reliability_report.txt', 'w') as f:
        f.write(report)
    
    print("\n" + "="*70)
    print("COMPARISON RESULTS:")
    print("="*70)
    print(comparison)
    
    print(f"\n✅ Reliability test complete!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()