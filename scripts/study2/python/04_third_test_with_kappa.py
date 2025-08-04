#!/usr/bin/env python3
"""
04_third_test_with_kappa.py
Purpose: Third test of few-shot labeling and calculate Cohen's Kappa across all three runs
Author: AI Assistant, 2025-08-01
"""

import os
import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from sklearn.metrics import cohen_kappa_score
from itertools import combinations
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import from the original script
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
    """EXACT SAME prompt as in 01_label_interest_fewshot.py"""
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

def calculate_fleiss_kappa(labels_df):
    """Calculate Fleiss' Kappa for multiple raters"""
    # This is a simplified version - for full Fleiss' Kappa, we'd need statsmodels
    # For now, we'll calculate the average of pairwise Cohen's Kappas
    return None  # Placeholder

def main():
    """Main execution function for third test"""
    print("🔄 THIRD TEST: Few-Shot LLM Labeling on All 40 Participants")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Purpose: Third run to calculate Cohen's Kappa across all three runs")
    print("=" * 70)
    
    # Load participant data
    df = pd.read_csv('results/study2/clean_participant_dataset_fewshot.csv')
    print(f"\n📊 Loaded {len(df)} participants")
    
    # Load previous results
    original_df = pd.read_csv('results/study2/clean_participant_dataset_fewshot.csv')
    retest_df = pd.read_csv('results/study2/full_retest_results.csv')
    
    # Setup model
    model = setup_gemini()
    
    # Process all participants
    print("\n🚀 Starting third test labeling...")
    third_test_results = []
    
    for idx, row in df.iterrows():
        participant_id = row['participant_id']
        combined_text = row['combined_text']
        
        print(f"Participant {participant_id}...", end=' ')
        
        # Get new label
        third_label = label_participant_fewshot(model, combined_text)
        print(third_label)
        
        third_test_results.append({
            'participant_id': participant_id,
            'third_test_label': third_label
        })
        
        # Rate limiting
        time.sleep(1)
    
    # Create comprehensive results dataframe
    third_test_df = pd.DataFrame(third_test_results)
    
    # Merge all three runs
    comprehensive_df = original_df[['participant_id', 'ai_label']].copy()
    comprehensive_df = comprehensive_df.rename(columns={'ai_label': 'run1_label'})
    comprehensive_df = comprehensive_df.merge(
        retest_df[['participant_id', 'retest_label']], 
        on='participant_id'
    )
    comprehensive_df = comprehensive_df.rename(columns={'retest_label': 'run2_label'})
    comprehensive_df = comprehensive_df.merge(
        third_test_df, 
        on='participant_id'
    )
    comprehensive_df = comprehensive_df.rename(columns={'third_test_label': 'run3_label'})
    
    # Save comprehensive results
    comprehensive_df.to_csv('results/study2/three_run_comparison.csv', index=False)
    
    # Calculate statistics
    print("\n" + "=" * 70)
    print("📊 THREE-RUN COMPARISON RESULTS:")
    print("=" * 70)
    
    # Distribution for each run
    print("\nLabel Distribution Across Runs:")
    for run in ['run1_label', 'run2_label', 'run3_label']:
        counts = comprehensive_df[run].value_counts()
        interested = counts.get('INTERESTED', 0)
        not_interested = counts.get('NOT_INTERESTED', 0)
        print(f"\n{run}:")
        print(f"  INTERESTED: {interested} ({interested/len(comprehensive_df)*100:.1f}%)")
        print(f"  NOT_INTERESTED: {not_interested} ({not_interested/len(comprehensive_df)*100:.1f}%)")
    
    # Calculate pairwise Cohen's Kappa
    print("\n" + "-" * 50)
    print("PAIRWISE COHEN'S KAPPA:")
    print("-" * 50)
    
    label_map = {'INTERESTED': 1, 'NOT_INTERESTED': 0, 'ERROR': -1}
    kappa_results = []
    
    runs = ['run1_label', 'run2_label', 'run3_label']
    for run1, run2 in combinations(runs, 2):
        # Convert to numeric
        labels1 = [label_map.get(label, -1) for label in comprehensive_df[run1]]
        labels2 = [label_map.get(label, -1) for label in comprehensive_df[run2]]
        
        # Filter valid pairs
        valid_pairs = [(l1, l2) for l1, l2 in zip(labels1, labels2) if l1 != -1 and l2 != -1]
        
        if valid_pairs:
            valid1 = [p[0] for p in valid_pairs]
            valid2 = [p[1] for p in valid_pairs]
            kappa = cohen_kappa_score(valid1, valid2)
            kappa_results.append(kappa)
            print(f"\n{run1} vs {run2}:")
            print(f"  Cohen's Kappa: {kappa:.3f}")
            
            # Show agreement
            agreement = sum(1 for v1, v2 in zip(valid1, valid2) if v1 == v2) / len(valid1)
            print(f"  Agreement: {agreement:.1%}")
    
    # Average Kappa
    if kappa_results:
        avg_kappa = np.mean(kappa_results)
        print(f"\n" + "=" * 50)
        print(f"AVERAGE COHEN'S KAPPA ACROSS ALL PAIRS: {avg_kappa:.3f}")
        
        if avg_kappa >= 0.80:
            print("Interpretation: EXCELLENT reliability (κ ≥ 0.80)")
        elif avg_kappa >= 0.60:
            print("Interpretation: GOOD reliability (κ ≥ 0.60)")
        else:
            print("Interpretation: MODERATE reliability (κ < 0.60)")
    
    # Show participants with any disagreement
    print("\n" + "-" * 50)
    print("PARTICIPANT-LEVEL CONSISTENCY:")
    print("-" * 50)
    
    inconsistent_count = 0
    for idx, row in comprehensive_df.iterrows():
        labels = [row['run1_label'], row['run2_label'], row['run3_label']]
        unique_labels = set(labels) - {'ERROR'}
        
        if len(unique_labels) > 1:
            inconsistent_count += 1
            print(f"\nParticipant {row['participant_id']}:")
            print(f"  Run 1: {row['run1_label']}")
            print(f"  Run 2: {row['run2_label']}")
            print(f"  Run 3: {row['run3_label']}")
    
    consistency_rate = (len(comprehensive_df) - inconsistent_count) / len(comprehensive_df) * 100
    print(f"\n" + "=" * 50)
    print(f"OVERALL CONSISTENCY: {len(comprehensive_df) - inconsistent_count}/{len(comprehensive_df)} participants ({consistency_rate:.1f}%) had identical labels across all 3 runs")
    
    print("\n✅ Three-run comparison complete!")
    print("Results saved to: results/study2/three_run_comparison.csv")

if __name__ == "__main__":
    main()