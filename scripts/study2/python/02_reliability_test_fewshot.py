#!/usr/bin/env python3
"""
02_reliability_test_fewshot.py
Purpose: Test the reliability of few-shot LLM labeling through multiple runs
Author: AI Assistant, 2025-08-01
"""

import os
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import random
random.seed(42)  # For reproducible sampling

try:
    from google import genai
    USE_NEW_SDK = True
except ImportError:
    import google.generativeai as genai
    USE_NEW_SDK = False
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

def label_participant_with_temperature(model_or_client, participant_text, temperature=0.0, max_retries=3):
    """Label a participant with specific temperature setting"""
    prompt = create_fewshot_labeling_prompt() + f'"{participant_text}"'
    
    for attempt in range(max_retries):
        try:
            if USE_NEW_SDK:
                # Temperature control may need adjustment based on SDK version
                response = model_or_client.models.generate_content(
                    model="gemini-1.5-pro",
                    contents=prompt,
                    generation_config={"temperature": temperature}
                )
                label = response.text.strip().upper()
            else:
                # Old SDK with temperature
                model_or_client._generation_config.temperature = temperature
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

def load_participant_data():
    """Load the clean participant dataset"""
    df = pd.read_csv('results/study2/clean_participant_dataset_fewshot.csv')
    return df

def test_reliability(model, participants_df, n_runs=5, subset_size=None, temperature=0.0):
    """Test reliability by running the same prompt multiple times"""
    
    # Select subset if specified
    if subset_size and subset_size < len(participants_df):
        test_df = participants_df.sample(n=subset_size, random_state=42)
        print(f"Testing reliability on {subset_size} randomly selected participants")
    else:
        test_df = participants_df
        print(f"Testing reliability on all {len(test_df)} participants")
    
    # Initialize results storage
    results = []
    
    # Run multiple labeling iterations
    for run in range(n_runs):
        print(f"\n--- Run {run + 1} of {n_runs} ---")
        run_labels = []
        
        for idx, row in test_df.iterrows():
            participant_id = row['participant_id']
            combined_text = row['combined_text']
            
            print(f"Run {run + 1}: Processing participant {participant_id}...", end=' ')
            
            # Get label from Gemini
            label = label_participant_with_temperature(model, combined_text, temperature)
            run_labels.append(label)
            print(label)
            
            # Rate limiting
            time.sleep(1)
        
        results.append(run_labels)
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results).T
    results_df.columns = [f'Run_{i+1}' for i in range(n_runs)]
    results_df['participant_id'] = test_df['participant_id'].values
    
    return results_df

def calculate_reliability_metrics(results_df):
    """Calculate various reliability metrics"""
    n_runs = len([col for col in results_df.columns if col.startswith('Run_')])
    
    # Convert labels to numeric (INTERESTED=1, NOT_INTERESTED=0)
    label_map = {'INTERESTED': 1, 'NOT_INTERESTED': 0, 'ERROR': -1}
    
    metrics = {}
    
    # Pairwise Cohen's Kappa
    kappa_scores = []
    for i in range(n_runs):
        for j in range(i+1, n_runs):
            run_i = results_df[f'Run_{i+1}'].map(label_map)
            run_j = results_df[f'Run_{j+1}'].map(label_map)
            
            # Filter out errors
            valid_mask = (run_i != -1) & (run_j != -1)
            if valid_mask.sum() > 0:
                kappa = cohen_kappa_score(run_i[valid_mask], run_j[valid_mask])
                kappa_scores.append(kappa)
    
    metrics['mean_kappa'] = np.mean(kappa_scores)
    metrics['std_kappa'] = np.std(kappa_scores)
    metrics['min_kappa'] = np.min(kappa_scores)
    metrics['max_kappa'] = np.max(kappa_scores)
    
    # Percentage agreement
    agreement_counts = []
    for idx, row in results_df.iterrows():
        run_labels = [row[f'Run_{i+1}'] for i in range(n_runs) if row[f'Run_{i+1}'] != 'ERROR']
        if len(run_labels) > 0:
            most_common = max(set(run_labels), key=run_labels.count)
            agreement = run_labels.count(most_common) / len(run_labels)
            agreement_counts.append(agreement)
    
    metrics['mean_agreement'] = np.mean(agreement_counts)
    metrics['perfect_agreement_pct'] = (np.array(agreement_counts) == 1.0).mean() * 100
    
    # Label stability
    label_changes = []
    for idx, row in results_df.iterrows():
        run_labels = [row[f'Run_{i+1}'] for i in range(n_runs) if row[f'Run_{i+1}'] != 'ERROR']
        unique_labels = len(set(run_labels))
        label_changes.append(unique_labels > 1)
    
    metrics['label_instability_pct'] = np.mean(label_changes) * 100
    
    return metrics, kappa_scores

def main():
    """Main execution function"""
    print("🔬 Testing Few-Shot LLM Labeling Reliability...")
    
    # Setup
    model = setup_gemini()
    participants_df = load_participant_data()
    
    # Test 1: Full sample test-retest at temperature=0.0
    print("\n📊 TEST 1: Test-Retest Reliability (Temperature=0.0)")
    print("=" * 50)
    
    results_temp0 = test_reliability(
        model, 
        participants_df, 
        n_runs=5,
        subset_size=10,  # Test on 10 participants to save API calls
        temperature=0.0
    )
    
    # Calculate metrics
    metrics_temp0, kappa_scores = calculate_reliability_metrics(results_temp0)
    
    # Save results
    results_temp0.to_csv('results/study2/reliability_test_temp0.csv', index=False)
    
    # Print results
    print("\n📈 Reliability Metrics (Temperature=0.0):")
    print(f"Mean Cohen's Kappa: {metrics_temp0['mean_kappa']:.3f} (±{metrics_temp0['std_kappa']:.3f})")
    print(f"Range: {metrics_temp0['min_kappa']:.3f} - {metrics_temp0['max_kappa']:.3f}")
    print(f"Mean Agreement: {metrics_temp0['mean_agreement']:.1%}")
    print(f"Perfect Agreement: {metrics_temp0['perfect_agreement_pct']:.1f}% of participants")
    print(f"Label Instability: {metrics_temp0['label_instability_pct']:.1f}% changed labels")
    
    # Interpretation
    print("\n📋 Interpretation:")
    if metrics_temp0['mean_kappa'] >= 0.80:
        print("✅ EXCELLENT reliability (κ ≥ 0.80)")
    elif metrics_temp0['mean_kappa'] >= 0.60:
        print("✅ GOOD reliability (κ ≥ 0.60)")
    elif metrics_temp0['mean_kappa'] >= 0.40:
        print("⚠️  MODERATE reliability (κ ≥ 0.40)")
    else:
        print("❌ POOR reliability (κ < 0.40)")
    
    # Test 2: Temperature sensitivity (optional, commented out to save API calls)
    """
    print("\n📊 TEST 2: Temperature Sensitivity")
    print("=" * 50)
    
    for temp in [0.3, 0.7]:
        print(f"\nTesting temperature={temp}")
        results_temp = test_reliability(
            model, 
            participants_df, 
            n_runs=3,
            subset_size=5,
            temperature=temp
        )
        metrics_temp, _ = calculate_reliability_metrics(results_temp)
        print(f"Mean Kappa at temp={temp}: {metrics_temp['mean_kappa']:.3f}")
    """
    
    # Generate reliability report
    report = {
        'test_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        'n_participants_tested': len(results_temp0),
        'n_runs': 5,
        'temperature': 0.0,
        'mean_kappa': metrics_temp0['mean_kappa'],
        'std_kappa': metrics_temp0['std_kappa'],
        'mean_agreement': metrics_temp0['mean_agreement'],
        'perfect_agreement_pct': metrics_temp0['perfect_agreement_pct'],
        'label_instability_pct': metrics_temp0['label_instability_pct']
    }
    
    # Save report
    pd.DataFrame([report]).to_csv('results/study2/reliability_report.csv', index=False)
    
    print("\n✅ Reliability testing complete!")
    print("Results saved to:")
    print("- results/study2/reliability_test_temp0.csv")
    print("- results/study2/reliability_report.csv")

if __name__ == "__main__":
    main()