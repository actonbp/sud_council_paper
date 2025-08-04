#!/usr/bin/env python3
"""
02_reliability_test_simple.py
Purpose: Simple test of few-shot LLM labeling reliability
Author: AI Assistant, 2025-08-01
"""

import os
import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Copy the necessary functions from the original script
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

def create_fewshot_labeling_prompt():
    """Same prompt as in 01_label_interest_fewshot.py"""
    # Participant 117 - INTERESTED example
    participant_117_text = """I don't have any personal connection with it, but I've definitely seen it or heard it talked about and like the media and stuff like on TV. I don't know if that counts. Yeah, that's pretty much it. I don't have a personal connection. I've definitely considered it. So I probably say my interest is like an 8 or 9 out of ten. I know that I counseling is interesting to me. I'm just kind of not sure what area to go in, but I would definitely consider substance abuse counseling. I think appealing is kind of just like being able to like, help somebody in a very, like, personal way, like connecting with them and providing support. And I also feel like it's just such a big issue that so many people struggle with that it would feel. It would be nice to be able to help people in that way. I would say what doesn't appeal to me, I feel like it's very intimidating, especially since substance abuse can be such a like. Like difficult. And kind of I know people can relapse and things like that. So I would say intimidating like I would just starting out would be a challenge, Definitely. Okay. Yup. I think personally, even though I know for a fact she would be ultimately supportive, I think my mom there's a lot of like stigma around substance abuse. So I think she would maybe be nervous, but I think ultimately supportive because it's like such a important career. But I think there's just a lot of especially with like drug misuse that I think would make her just nervous. I'm not fully sure what I want to do yet, but I do know I want to major in psychology. So I'm kind of just excited about experience, like in a career, like getting to interact with people and also, again, like helping people is just my number one priority in a career choice. So yeah, but I'm it. I would maybe think about doing something like research related just because I think it's so interesting and I just want to learn more, especially through like, like people. Yeah, that makes sense. I'd say uncertainty right now is just there's so many options and it's kind of a matter of narrowing it down to what I'm interested in. Okay. So I would say that what makes me nervous is the uncertainty. But I'm hoping in like the next year or so to I kind of have a little bit more focus. Yep. So I would say a lot of anxiety also comes from like like making sure I choose the right field. It's that makes sense."""
    
    # Participant 147 - NOT INTERESTED example
    participant_147_text = """I just don't see myself working as like a mental health counselor so I'll be surprised if I work as one. I am now in a nursing major, so I still like a nursing professional. None of my family, like any of my friends, is like substance, like the likes of like alcohol. So I actually don't know a lot about this. Like, people who, like, struggle with just like, sometimes the way they, like, go to get help just because they like, they harm a lot of people. Like they have...they have Like family feelings like even like maybe their sibling. So like, maybe because of that they ask like to get help. So they go look for a professional. Like it's just like a really long journey to get like help...like to stop using like the drugs? So like, maybe alcohol That's like the only information I know about substance abuse. I am less interested in. Just like, as I said earlier. I wasn't raised in an environment like people around me using drugs and alcohol like even until now. So I don't see myself working in this field. And also because like I am, I don't see myself like helping other will be motion. That's why I like going to nursing like I like I remember like emotion. I am a global. Yeah. I think they'll be shocked. Yeah.  Like we don't talk about this, like, a lot. Like I never spoke about, like, mental health before. So if I just came out of the blue and said all I want to become a mental health counselor, they'll be shocked. I think money. I'm like, How much of this is I need a job. I really can't work in a job like where I'm in a position where I can't give, like all my support. Krish, any thoughts on your end? Mine is similar to the other people. Just like personal satisfaction. Like how happy I am, how much I like my job. I think the joy that come from helping others that's like...that was like this the biggest reason the I showed like an interest because like I really like how I help other people and how they walk by me with like a smile or something. So I feel like the joy that comes from with helping other. Like my parents again. ? They were the ones who chose to for me, like to go to nursing. I was, like, looking for choices in the medical field. Well, then I didn't know which one I should go for. Like they are the ones who decide for me. And like it shows, it's like they know me more than how I know myself. So I think, like, I used them most."""
    
    return f"""
You are an expert researcher analyzing focus group discussions about career interests in substance use disorder (SUD) counseling.

Your task: Read ALL utterances from a single participant and determine their overall interest in SUD counseling as a career.

I will provide you with two examples to guide your classification:

EXAMPLE 1 - INTERESTED:
Participant 117 said: "{participant_117_text}"

Classification: INTERESTED
Why: This participant explicitly states their interest is "8 or 9 out of ten" and says "I've definitely considered it." They express genuine personal interest in counseling, discuss what appeals to them (helping people in a personal way), acknowledge challenges (intimidating, relapses), and are actively exploring psychology career options. Despite concerns, they show clear personal consideration and engagement with SUD counseling as a career possibility.

EXAMPLE 2 - NOT INTERESTED:
Participant 147 said: "{participant_147_text}"

Classification: NOT INTERESTED
Why: This participant clearly states "I just don't see myself working as like a mental health counselor" and "I am less interested in." They have no personal connection to substance issues, prefer nursing (physical health) over mental health work, and their parents chose their career path. They explicitly state they don't see themselves helping others with emotional issues and would be "shocked" if they became a mental health counselor.

GUIDELINES FOR CLASSIFICATION:
Based on the examples above, classify participants as:
- "INTERESTED" - Participant shows personal interest, curiosity, or positive consideration of SUD counseling careers (even with some concerns)
- "NOT_INTERESTED" - Participant shows disinterest, negative views, or lacks personal engagement with SUD counseling careers

Key indicators:
- Direct statements of interest level (like "8 or 9 out of ten" or "I don't see myself")
- Personal consideration of the field as a career option
- Genuine engagement with the topic beyond just answering questions
- Clear preference for other career paths without interest in SUD counseling

Remember: Someone can have concerns but still be interested if they show genuine personal consideration (like Example 1). Someone is NOT interested if they clearly state disinterest or preference for other fields without personal engagement with SUD counseling (like Example 2).

Respond with ONLY the label (INTERESTED or NOT_INTERESTED). No explanation needed.

Now classify this participant based on all their utterances:
"""

def label_participant_fewshot(model_or_client, participant_text, max_retries=3):
    """Label a participant's overall interest using few-shot Gemini prompt"""
    prompt = create_fewshot_labeling_prompt() + f'"{participant_text}"'
    
    for attempt in range(max_retries):
        try:
            if USE_NEW_SDK:
                response = model_or_client.models.generate_content(
                    model="gemini-1.5-pro",
                    contents=prompt
                )
                label = response.text.strip().upper()
            else:
                response = model_or_client.generate_content(prompt)
                label = response.text.strip().upper()
            
            valid_labels = ['INTERESTED', 'NOT_INTERESTED']
            if label in valid_labels:
                return label
            else:
                print(f"Invalid label '{label}', retrying...")
                continue
                
        except Exception as e:
            print(f"API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return "ERROR"
    
    return "ERROR"

def test_reliability_simple(n_participants=10, n_runs=5):
    """Simple reliability test using the same function as original labeling"""
    
    # Load participant data
    df = pd.read_csv('results/study2/clean_participant_dataset_fewshot.csv')
    
    # Sample participants
    test_df = df.sample(n=n_participants, random_state=42)
    print(f"Testing reliability on {n_participants} randomly selected participants")
    print(f"Participant IDs: {test_df['participant_id'].tolist()}")
    
    # Setup model
    model = setup_gemini()
    
    # Store results
    results = {}
    
    # Run multiple times
    for run in range(1, n_runs + 1):
        print(f"\n--- Run {run} of {n_runs} ---")
        results[f'Run_{run}'] = []
        
        for idx, row in test_df.iterrows():
            participant_id = row['participant_id']
            combined_text = row['combined_text']
            
            print(f"Run {run}: Participant {participant_id}...", end=' ')
            
            # Use exact same function as original labeling
            label = label_participant_fewshot(model, combined_text)
            results[f'Run_{run}'].append(label)
            print(label)
            
            # Rate limiting
            time.sleep(1)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df['participant_id'] = test_df['participant_id'].values
    
    return results_df

def calculate_simple_metrics(results_df):
    """Calculate simple reliability metrics"""
    n_runs = len([col for col in results_df.columns if col.startswith('Run_')])
    
    # Check consistency for each participant
    consistency_results = []
    
    for idx, row in results_df.iterrows():
        participant_id = row['participant_id']
        labels = [row[f'Run_{i}'] for i in range(1, n_runs + 1) if row[f'Run_{i}'] != 'ERROR']
        
        if len(labels) >= 2:
            # All same?
            all_same = len(set(labels)) == 1
            # Most common label
            if labels:
                most_common = max(set(labels), key=labels.count)
                consistency = labels.count(most_common) / len(labels)
            else:
                consistency = 0
            
            consistency_results.append({
                'participant_id': participant_id,
                'all_same': all_same,
                'consistency': consistency,
                'labels': labels
            })
    
    # Calculate overall metrics
    perfect_agreement = sum([r['all_same'] for r in consistency_results])
    total_tested = len(consistency_results)
    
    # Calculate pairwise Cohen's Kappa
    kappa_scores = []
    label_map = {'INTERESTED': 1, 'NOT_INTERESTED': 0}
    
    for i in range(1, n_runs + 1):
        for j in range(i + 1, n_runs + 1):
            labels_i = [label_map.get(row[f'Run_{i}'], -1) for _, row in results_df.iterrows()]
            labels_j = [label_map.get(row[f'Run_{j}'], -1) for _, row in results_df.iterrows()]
            
            # Filter valid pairs
            valid_pairs = [(li, lj) for li, lj in zip(labels_i, labels_j) if li != -1 and lj != -1]
            
            if valid_pairs:
                labels_i_valid = [p[0] for p in valid_pairs]
                labels_j_valid = [p[1] for p in valid_pairs]
                kappa = cohen_kappa_score(labels_i_valid, labels_j_valid)
                kappa_scores.append(kappa)
    
    return {
        'perfect_agreement': perfect_agreement,
        'total_tested': total_tested,
        'perfect_agreement_pct': (perfect_agreement / total_tested * 100) if total_tested > 0 else 0,
        'mean_kappa': np.mean(kappa_scores) if kappa_scores else 0,
        'consistency_results': consistency_results,
        'kappa_scores': kappa_scores
    }

def main():
    """Main execution function"""
    print("🔬 Testing Few-Shot LLM Labeling Reliability (Simple Version)...")
    print("=" * 60)
    
    # Run reliability test
    results_df = test_reliability_simple(n_participants=10, n_runs=5)
    
    # Save raw results
    results_df.to_csv('results/study2/reliability_test_results.csv', index=False)
    
    # Calculate metrics
    metrics = calculate_simple_metrics(results_df)
    
    # Print results
    print("\n📊 RELIABILITY TEST RESULTS:")
    print("=" * 60)
    print(f"Participants tested: {metrics['total_tested']}")
    print(f"Perfect agreement: {metrics['perfect_agreement']}/{metrics['total_tested']} ({metrics['perfect_agreement_pct']:.1f}%)")
    print(f"Mean Cohen's Kappa: {metrics['mean_kappa']:.3f}")
    
    # Detailed results
    print("\n📋 PARTICIPANT-LEVEL CONSISTENCY:")
    print("-" * 60)
    for result in metrics['consistency_results']:
        print(f"Participant {result['participant_id']}: {result['consistency']:.0%} consistent")
        print(f"  Labels: {result['labels']}")
    
    # Interpretation
    print("\n📈 INTERPRETATION:")
    if metrics['mean_kappa'] >= 0.80:
        print("✅ EXCELLENT reliability (κ ≥ 0.80)")
    elif metrics['mean_kappa'] >= 0.60:
        print("✅ GOOD reliability (κ ≥ 0.60)")
    elif metrics['mean_kappa'] >= 0.40:
        print("⚠️  MODERATE reliability (κ ≥ 0.40)")
    else:
        print("❌ POOR reliability (κ < 0.40)")
    
    print("\n✅ Reliability testing complete!")
    print("Results saved to: results/study2/reliability_test_results.csv")
    
    # Additional note about demographics
    print("\n📝 NOTE ABOUT DEMOGRAPHICS:")
    print("To add demographic analysis, you'll need to:")
    print("1. Obtain the secure demographic file with participant IDs")
    print("2. Merge with clean_participant_dataset_fewshot.csv on participant_id")
    print("3. Analyze descriptive statistics by interest groups")

if __name__ == "__main__":
    main()