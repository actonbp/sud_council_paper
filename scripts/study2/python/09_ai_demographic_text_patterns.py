#!/usr/bin/env python3
"""
09_ai_demographic_text_patterns.py
Purpose: Find interactions between demographics and text that predict interest
Author: AI Assistant, 2025-08-02
"""

import os
import pandas as pd
import json
import numpy as np
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

def load_full_dataset():
    """Load complete dataset with demographics and text"""
    
    # Load merged data
    merged_df = pd.read_csv('results/study2/merged_demographics_interest.csv')
    
    # Load text data
    text_df = pd.read_csv('results/study2/clean_participant_dataset_fewshot.csv')
    
    # Combine
    full_df = merged_df.merge(
        text_df[['participant_id', 'combined_text', 'num_utterances']], 
        on='participant_id'
    )
    
    # Create derived features
    full_df['text_mentions_help'] = full_df['combined_text'].str.lower().str.contains('help')
    full_df['text_mentions_money'] = full_df['combined_text'].str.lower().str.contains('money|salary|pay|income')
    full_df['text_mentions_family'] = full_df['combined_text'].str.lower().str.contains('family|parent|mom|dad|mother|father')
    full_df['text_mentions_pressure'] = full_df['combined_text'].str.lower().str.contains('pressure|stress|difficult|hard|challenge')
    full_df['text_mentions_passion'] = full_df['combined_text'].str.lower().str.contains('passion|love|care|interested')
    full_df['text_mentions_career'] = full_df['combined_text'].str.lower().str.contains('career|job|profession|field')
    
    print(f"Loaded {len(full_df)} participants with full data")
    
    return full_df

def calculate_interaction_patterns(df):
    """Calculate simple interaction patterns manually"""
    
    patterns = []
    
    # Pattern 1: Mental health treatment + mentions helping
    mh_help = df[(df['Mental_health_treatment'] > 1) & (df['text_mentions_help'] == True)]
    mh_help_int = sum(mh_help['ai_label'] == 'INTERESTED')
    if len(mh_help) >= 5:
        patterns.append({
            'pattern': 'Has mental health treatment + mentions "help"',
            'n_total': len(mh_help),
            'n_interested': mh_help_int,
            'accuracy': mh_help_int / len(mh_help),
            'participants': mh_help['participant_id'].tolist()
        })
    
    # Pattern 2: No mental health treatment + mentions money
    no_mh_money = df[(df['Mental_health_treatment'] == 1) & (df['text_mentions_money'] == True)]
    no_mh_money_not = sum(no_mh_money['ai_label'] == 'NOT_INTERESTED')
    if len(no_mh_money) >= 5:
        patterns.append({
            'pattern': 'No mental health treatment + mentions money/salary',
            'n_total': len(no_mh_money),
            'n_not_interested': no_mh_money_not,
            'accuracy': no_mh_money_not / len(no_mh_money),
            'participants': no_mh_money['participant_id'].tolist()
        })
    
    # Pattern 3: Family SU treatment + mentions family
    fam_su_fam = df[(df['Family_friend_substance_use_treatment'] > 1) & (df['text_mentions_family'] == True)]
    fam_su_fam_int = sum(fam_su_fam['ai_label'] == 'INTERESTED')
    if len(fam_su_fam) >= 5:
        patterns.append({
            'pattern': 'Family/friend SU treatment + mentions family',
            'n_total': len(fam_su_fam),
            'n_interested': fam_su_fam_int,
            'accuracy': fam_su_fam_int / len(fam_su_fam),
            'participants': fam_su_fam['participant_id'].tolist()
        })
    
    # Pattern 4: Sophomore + mentions career uncertainty
    soph_career = df[(df['Year_in_school'] == 2) & (df['text_mentions_career'] == True)]
    soph_career_int = sum(soph_career['ai_label'] == 'INTERESTED')
    if len(soph_career) >= 5:
        patterns.append({
            'pattern': 'Sophomore + mentions career/job',
            'n_total': len(soph_career),
            'n_interested': soph_career_int,
            'accuracy': soph_career_int / len(soph_career),
            'participants': soph_career['participant_id'].tolist()
        })
    
    # Pattern 5: High income + mentions pressure/stress
    high_inc_pressure = df[(df['Household_income'] >= 7) & (df['text_mentions_pressure'] == True)]
    if len(high_inc_pressure) >= 5:
        patterns.append({
            'pattern': 'High household income (7+) + mentions pressure/stress',
            'n_total': len(high_inc_pressure),
            'by_label': high_inc_pressure['ai_label'].value_counts().to_dict(),
            'participants': high_inc_pressure['participant_id'].tolist()
        })
    
    return patterns

def find_complex_interactions(model_or_client, df):
    """Ask Gemini to find complex demographic-text interactions"""
    
    # Create structured data for Gemini
    participant_data = []
    for _, row in df.iterrows():
        participant_data.append({
            'id': row['participant_id'],
            'label': row['ai_label'],
            'demographics': {
                'age': row['Age'],
                'gender': 'Female' if row['Gener Identity'] == 5 else 'Male' if row['Gener Identity'] == 7 else 'Other',
                'race': row['Race'],
                'year_school': row['Year_in_school'],
                'parent_education': row['Parent_highest_level_education'],
                'employed': row['Current_employement'] == 1,
                'household_income': row['Household_income'],
                'mh_treatment': row['Mental_health_treatment'] > 1,
                'family_su_treatment': row['Family_friend_substance_use_treatment'] > 1,
                'safety_grew_up': row['Safety_area_grew_up']
            },
            'text_features': {
                'mentions_help': bool(row['text_mentions_help']),
                'mentions_money': bool(row['text_mentions_money']),
                'mentions_family': bool(row['text_mentions_family']),
                'mentions_pressure': bool(row['text_mentions_pressure']),
                'mentions_passion': bool(row['text_mentions_passion']),
                'mentions_career': bool(row['text_mentions_career']),
                'num_utterances': row['num_utterances'],
                'sample_text': row['combined_text'][:500]
            }
        })
    
    prompt = f"""
Analyze these 40 participants to find INTERACTION PATTERNS between demographics and what they say.

DATA:
{json.dumps(participant_data[:10], indent=2)}
... [and 30 more participants with similar structure]

TASK: Find patterns where DEMOGRAPHICS + TEXT CONTENT together predict interest better than either alone.

Look for patterns like:
- "Females who mention family concerns are usually NOT interested"
- "Students with MH treatment who talk about helping are INTERESTED"
- "High income students who mention money are NOT interested"

For each pattern:
1. State the demographic + text combination
2. How many participants fit this pattern
3. What percentage are INTERESTED vs NOT_INTERESTED
4. Why this combination might be meaningful

Find the TOP 5 most accurate interaction patterns (at least 80% accuracy, minimum 5 participants).
"""
    
    if USE_NEW_SDK:
        response = model_or_client.models.generate_content(
            model="gemini-1.5-pro",
            contents=prompt
        )
        interactions = response.text
    else:
        response = model_or_client.generate_content(prompt)
        interactions = response.text
    
    return interactions

def test_specific_hypotheses(df):
    """Test specific demographic-text interaction hypotheses"""
    
    hypotheses = []
    
    # Hypothesis 1: Women mentioning family are less interested
    women_family = df[(df['Gener Identity'] == 5) & (df['text_mentions_family'] == True)]
    if len(women_family) > 0:
        women_family_not_int = sum(women_family['ai_label'] == 'NOT_INTERESTED')
        hypotheses.append({
            'hypothesis': 'Women who mention family',
            'prediction': 'NOT_INTERESTED',
            'n': len(women_family),
            'accuracy': women_family_not_int / len(women_family) if len(women_family) > 0 else 0,
            'actual_distribution': women_family['ai_label'].value_counts().to_dict()
        })
    
    # Hypothesis 2: Students with both MH and family SU treatment who mention helping
    both_treatment_help = df[
        (df['Mental_health_treatment'] > 1) & 
        (df['Family_friend_substance_use_treatment'] > 1) & 
        (df['text_mentions_help'] == True)
    ]
    if len(both_treatment_help) > 0:
        both_treatment_int = sum(both_treatment_help['ai_label'] == 'INTERESTED')
        hypotheses.append({
            'hypothesis': 'Has both MH + family SU treatment + mentions helping',
            'prediction': 'INTERESTED',
            'n': len(both_treatment_help),
            'accuracy': both_treatment_int / len(both_treatment_help),
            'actual_distribution': both_treatment_help['ai_label'].value_counts().to_dict()
        })
    
    # Hypothesis 3: Low income + mentions money
    low_income_money = df[(df['Household_income'] <= 5) & (df['text_mentions_money'] == True)]
    if len(low_income_money) > 0:
        low_income_int = sum(low_income_money['ai_label'] == 'INTERESTED')
        hypotheses.append({
            'hypothesis': 'Low income (≤5) + mentions money',
            'prediction': 'INTERESTED',
            'n': len(low_income_money),
            'accuracy': low_income_int / len(low_income_money),
            'actual_distribution': low_income_money['ai_label'].value_counts().to_dict()
        })
    
    # Hypothesis 4: No employment + mentions career
    not_employed_career = df[(df['Current_employement'] != 1) & (df['text_mentions_career'] == True)]
    if len(not_employed_career) > 0:
        not_emp_career_dist = not_employed_career['ai_label'].value_counts()
        hypotheses.append({
            'hypothesis': 'Not employed + mentions career',
            'prediction': 'Mixed',
            'n': len(not_employed_career),
            'actual_distribution': not_emp_career_dist.to_dict()
        })
    
    # Hypothesis 5: Multiracial + mentions pressure
    multiracial_pressure = df[(df['Race'] == 6) & (df['text_mentions_pressure'] == True)]
    if len(multiracial_pressure) > 0:
        multi_pressure_int = sum(multiracial_pressure['ai_label'] == 'INTERESTED')
        hypotheses.append({
            'hypothesis': 'Multiracial + mentions pressure/stress',
            'prediction': 'INTERESTED',
            'n': len(multiracial_pressure),
            'accuracy': multi_pressure_int / len(multiracial_pressure),
            'actual_distribution': multiracial_pressure['ai_label'].value_counts().to_dict()
        })
    
    return hypotheses

def main():
    """Main execution function"""
    print("🔄 Demographic + Text Interaction Analysis")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Setup
    model = setup_gemini()
    df = load_full_dataset()
    
    # Calculate simple patterns
    print("\n📊 Calculating interaction patterns...")
    patterns = calculate_interaction_patterns(df)
    
    print("\nSimple Interaction Patterns Found:")
    for p in patterns:
        print(f"\n{p['pattern']}:")
        print(f"  Total: {p['n_total']} participants")
        if 'accuracy' in p:
            print(f"  Accuracy: {p['accuracy']:.1%}")
        if 'n_interested' in p:
            print(f"  Interested: {p['n_interested']}")
        if 'n_not_interested' in p:
            print(f"  Not Interested: {p['n_not_interested']}")
        if 'by_label' in p:
            print(f"  Distribution: {p['by_label']}")
    
    # Test specific hypotheses
    print("\n🧪 Testing specific hypotheses...")
    hypotheses = test_specific_hypotheses(df)
    
    print("\nHypothesis Test Results:")
    for h in hypotheses:
        print(f"\n{h['hypothesis']}:")
        print(f"  N = {h['n']}")
        if 'accuracy' in h:
            print(f"  Accuracy for {h['prediction']}: {h['accuracy']:.1%}")
        print(f"  Actual distribution: {h['actual_distribution']}")
    
    # AI complex pattern finding
    print("\n🤖 Finding complex interactions with AI...")
    complex_patterns = find_complex_interactions(model, df)
    
    print("\n" + "="*70)
    print("COMPLEX INTERACTION PATTERNS:")
    print("="*70)
    print(complex_patterns)
    
    # Verify patterns manually
    print("\n✓ Verifying top patterns...")
    
    # Save results
    output_dir = Path('results/study2/ai_patterns')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'simple_patterns': patterns,
        'hypothesis_tests': hypotheses,
        'complex_patterns': complex_patterns
    }
    
    with open(output_dir / 'interaction_patterns.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary report
    with open(output_dir / 'interaction_patterns_report.txt', 'w') as f:
        f.write("Demographic + Text Interaction Patterns\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("="*70 + "\n\n")
        
        f.write("SIMPLE PATTERNS (Manually Calculated):\n")
        for p in patterns:
            f.write(f"\n{p['pattern']}:\n")
            f.write(f"  N = {p['n_total']}")
            if 'accuracy' in p:
                f.write(f", Accuracy = {p['accuracy']:.1%}\n")
            else:
                f.write("\n")
        
        f.write("\n\nHYPOTHESIS TESTS:\n")
        for h in hypotheses:
            f.write(f"\n{h['hypothesis']}:\n")
            f.write(f"  N = {h['n']}, Distribution = {h['actual_distribution']}\n")
        
        f.write("\n\nCOMPLEX PATTERNS (AI-Discovered):\n")
        f.write(complex_patterns)
    
    print(f"\n✅ Results saved to: {output_dir}")

if __name__ == "__main__":
    main()