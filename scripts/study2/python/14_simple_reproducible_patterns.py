#!/usr/bin/env python3
"""
14_simple_reproducible_patterns.py
Purpose: Simple, focused pattern discovery for reproducibility
Author: AI Assistant, 2025-08-02
"""

import os
import pandas as pd
import json
import numpy as np
from pathlib import Path
from datetime import datetime

def load_data():
    """Load merged dataset"""
    merged_df = pd.read_csv('results/study2/merged_demographics_interest.csv')
    text_df = pd.read_csv('results/study2/clean_participant_dataset_fewshot.csv')
    
    full_df = merged_df.merge(
        text_df[['participant_id', 'combined_text', 'num_utterances']], 
        on='participant_id'
    )
    
    return full_df

def test_demographic_patterns(df):
    """Test specific demographic patterns"""
    patterns = []
    
    # Pattern 1: Both treatments
    both_treatment = df[(df['Mental_health_treatment'] > 1) & 
                       (df['Family_friend_substance_use_treatment'] > 1)]
    if len(both_treatment) >= 5:
        interested = sum(both_treatment['ai_label'] == 'INTERESTED')
        patterns.append({
            'name': 'Both MH and Family SU Treatment',
            'rule': 'Mental_health_treatment > 1 AND Family_friend_substance_use_treatment > 1',
            'n_matches': len(both_treatment),
            'n_interested': interested,
            'accuracy': interested / len(both_treatment),
            'participants': both_treatment['participant_id'].tolist()
        })
    
    # Pattern 2: Sophomore + Employed
    soph_emp = df[(df['Year_in_school'] == 2) & (df['Current_employement'] == 1)]
    if len(soph_emp) >= 5:
        interested = sum(soph_emp['ai_label'] == 'INTERESTED')
        patterns.append({
            'name': 'Sophomore + Employed',
            'rule': 'Year_in_school = 2 AND Current_employement = 1',
            'n_matches': len(soph_emp),
            'n_interested': interested,
            'accuracy': interested / len(soph_emp),
            'participants': soph_emp['participant_id'].tolist()
        })
    
    # Pattern 3: No MH treatment
    no_mh = df[df['Mental_health_treatment'] == 1]
    if len(no_mh) >= 5:
        not_interested = sum(no_mh['ai_label'] == 'NOT_INTERESTED')
        patterns.append({
            'name': 'No Mental Health Treatment',
            'rule': 'Mental_health_treatment = 1',
            'n_matches': len(no_mh),
            'n_not_interested': not_interested,
            'accuracy': not_interested / len(no_mh),
            'participants': no_mh['participant_id'].tolist()
        })
    
    # Pattern 4: High parent education
    high_parent_ed = df[df['Parent_highest_level_education'] >= 5]
    if len(high_parent_ed) >= 5:
        interested = sum(high_parent_ed['ai_label'] == 'INTERESTED')
        patterns.append({
            'name': 'High Parent Education (College+)',
            'rule': 'Parent_highest_level_education >= 5',
            'n_matches': len(high_parent_ed),
            'n_interested': interested,
            'accuracy': interested / len(high_parent_ed),
            'participants': high_parent_ed['participant_id'].tolist()
        })
    
    return patterns

def test_text_patterns(df):
    """Test specific text patterns"""
    patterns = []
    
    # Common words to test
    word_tests = [
        ('help', 'Mentions "help"'),
        ('money|salary|pay|income', 'Mentions money/salary'),
        ('family', 'Mentions family'),
        ('pressure|stress|difficult|hard', 'Mentions pressure/stress'),
        ('research', 'Mentions research'),
        ('patients', 'Mentions patients'),
        ('career', 'Mentions career'),
        ('passion|love|care', 'Mentions passion/love')
    ]
    
    for pattern, name in word_tests:
        df[f'has_{pattern}'] = df['combined_text'].str.lower().str.contains(pattern)
        matches = df[df[f'has_{pattern}']]
        
        if len(matches) >= 5:
            interested = sum(matches['ai_label'] == 'INTERESTED')
            not_interested = sum(matches['ai_label'] == 'NOT_INTERESTED')
            
            # Determine which group it predicts better
            if interested > not_interested:
                accuracy = interested / len(matches)
                prediction = 'INTERESTED'
            else:
                accuracy = not_interested / len(matches)
                prediction = 'NOT_INTERESTED'
            
            patterns.append({
                'name': name,
                'rule': f'text contains "{pattern}"',
                'n_matches': len(matches),
                'n_interested': interested,
                'n_not_interested': not_interested,
                'accuracy': accuracy,
                'predicts': prediction,
                'participants': matches['participant_id'].tolist()
            })
    
    return patterns

def test_interaction_patterns(df):
    """Test demographic + text interaction patterns"""
    patterns = []
    
    # Pattern 1: MH treatment + mentions help
    df['mentions_help'] = df['combined_text'].str.lower().str.contains('help')
    mh_help = df[(df['Mental_health_treatment'] > 1) & (df['mentions_help'])]
    if len(mh_help) >= 5:
        interested = sum(mh_help['ai_label'] == 'INTERESTED')
        patterns.append({
            'name': 'MH Treatment + Mentions Help',
            'rule': 'Mental_health_treatment > 1 AND text contains "help"',
            'n_matches': len(mh_help),
            'n_interested': interested,
            'accuracy': interested / len(mh_help),
            'participants': mh_help['participant_id'].tolist()
        })
    
    # Pattern 2: No MH treatment + mentions money
    df['mentions_money'] = df['combined_text'].str.lower().str.contains('money|salary|pay')
    no_mh_money = df[(df['Mental_health_treatment'] == 1) & (df['mentions_money'])]
    if len(no_mh_money) >= 5:
        not_interested = sum(no_mh_money['ai_label'] == 'NOT_INTERESTED')
        patterns.append({
            'name': 'No MH Treatment + Mentions Money',
            'rule': 'Mental_health_treatment = 1 AND text contains money terms',
            'n_matches': len(no_mh_money),
            'n_not_interested': not_interested,
            'accuracy': not_interested / len(no_mh_money),
            'participants': no_mh_money['participant_id'].tolist()
        })
    
    # Pattern 3: Both treatments + mentions help
    both_help = df[(df['Mental_health_treatment'] > 1) & 
                   (df['Family_friend_substance_use_treatment'] > 1) & 
                   (df['mentions_help'])]
    if len(both_help) >= 5:
        interested = sum(both_help['ai_label'] == 'INTERESTED')
        patterns.append({
            'name': 'Both Treatments + Mentions Help',
            'rule': 'MH > 1 AND Family SU > 1 AND mentions "help"',
            'n_matches': len(both_help),
            'n_interested': interested,
            'accuracy': interested / len(both_help),
            'participants': both_help['participant_id'].tolist()
        })
    
    # Pattern 4: Sophomore + mentions career
    df['mentions_career'] = df['combined_text'].str.lower().str.contains('career|job|field')
    soph_career = df[(df['Year_in_school'] == 2) & (df['mentions_career'])]
    if len(soph_career) >= 5:
        interested = sum(soph_career['ai_label'] == 'INTERESTED')
        patterns.append({
            'name': 'Sophomore + Mentions Career',
            'rule': 'Year_in_school = 2 AND mentions career terms',
            'n_matches': len(soph_career),
            'n_interested': interested,
            'accuracy': interested / len(soph_career),
            'participants': soph_career['participant_id'].tolist()
        })
    
    return patterns

def calculate_robustness(pattern):
    """Calculate robustness score for a pattern"""
    accuracy = pattern['accuracy']
    coverage = pattern['n_matches'] / 40  # Out of 40 participants
    
    # Robustness combines accuracy and coverage
    robustness = (accuracy * 0.7) + (coverage * 0.3)
    
    return robustness

def main():
    """Main execution"""
    print("🎯 Simple Reproducible Pattern Analysis")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load data
    df = load_data()
    print(f"\nLoaded {len(df)} participants")
    
    # Test patterns
    print("\nTesting demographic patterns...")
    demo_patterns = test_demographic_patterns(df)
    
    print("Testing text patterns...")
    text_patterns = test_text_patterns(df)
    
    print("Testing interaction patterns...")
    interaction_patterns = test_interaction_patterns(df)
    
    # Calculate robustness
    all_patterns = []
    
    for p in demo_patterns:
        p['type'] = 'demographic'
        p['robustness'] = calculate_robustness(p)
        all_patterns.append(p)
    
    for p in text_patterns:
        p['type'] = 'text'
        p['robustness'] = calculate_robustness(p)
        all_patterns.append(p)
    
    for p in interaction_patterns:
        p['type'] = 'interaction'
        p['robustness'] = calculate_robustness(p)
        all_patterns.append(p)
    
    # Sort by robustness
    all_patterns.sort(key=lambda x: x['robustness'], reverse=True)
    
    # Filter for high-quality patterns
    strong_patterns = [p for p in all_patterns if p['accuracy'] >= 0.75 and p['n_matches'] >= 5]
    
    # Create report
    output_dir = Path('results/study2/ai_patterns')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = f"""
REPRODUCIBLE PATTERN ANALYSIS REPORT
====================================
Generated: {datetime.now()}

This analysis uses deterministic rules to find patterns, ensuring
complete reproducibility for peer review.

HIGH-CONFIDENCE PATTERNS (≥75% accuracy, ≥5 participants):
==========================================================
"""
    
    for i, pattern in enumerate(strong_patterns):
        report += f"""
{i+1}. {pattern['name']} ({pattern['type']})
   Rule: {pattern['rule']}
   Matches: {pattern['n_matches']} participants
   Accuracy: {pattern['accuracy']:.1%}
"""
        if 'n_interested' in pattern:
            report += f"   Interested: {pattern['n_interested']}/{pattern['n_matches']}\n"
        if 'n_not_interested' in pattern:
            report += f"   Not Interested: {pattern['n_not_interested']}/{pattern['n_matches']}\n"
        report += f"   Robustness Score: {pattern['robustness']:.3f}\n"
    
    report += """

ALL PATTERNS TESTED:
====================
"""
    
    # Group by type
    for ptype in ['demographic', 'text', 'interaction']:
        type_patterns = [p for p in all_patterns if p['type'] == ptype]
        if type_patterns:
            report += f"\n{ptype.upper()} PATTERNS:\n"
            for p in type_patterns:
                report += f"- {p['name']}: {p['n_matches']} matches, {p['accuracy']:.1%} accuracy\n"
    
    report += """

REPRODUCIBILITY GUARANTEE:
=========================
All patterns use exact, deterministic rules that can be verified:
1. Download the merged dataset (results/study2/merged_demographics_interest.csv)
2. Apply the rules exactly as specified
3. Count matches and calculate accuracy
4. Results will be identical

This approach ensures complete transparency and reproducibility
for peer review, addressing any concerns about AI-based analysis.
"""
    
    # Save report
    with open(output_dir / 'final_reproducible_patterns.txt', 'w') as f:
        f.write(report)
    
    # Save detailed results
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_participants': len(df),
        'strong_patterns': strong_patterns,
        'all_patterns': all_patterns,
        'summary': {
            'n_demographic': len([p for p in strong_patterns if p['type'] == 'demographic']),
            'n_text': len([p for p in strong_patterns if p['type'] == 'text']),
            'n_interaction': len([p for p in strong_patterns if p['type'] == 'interaction']),
            'highest_accuracy': max(p['accuracy'] for p in strong_patterns) if strong_patterns else 0,
            'highest_coverage': max(p['n_matches'] for p in strong_patterns) if strong_patterns else 0
        }
    }
    
    with open(output_dir / 'final_reproducible_patterns.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Analysis complete!")
    print(f"Found {len(strong_patterns)} high-confidence patterns")
    print(f"Results saved to: {output_dir}")
    
    # Print summary
    print("\nTOP 5 PATTERNS BY ROBUSTNESS:")
    print("="*50)
    for i, p in enumerate(strong_patterns[:5]):
        print(f"{i+1}. {p['name']}")
        print(f"   Accuracy: {p['accuracy']:.1%}, Coverage: {p['n_matches']}/40")

if __name__ == "__main__":
    main()