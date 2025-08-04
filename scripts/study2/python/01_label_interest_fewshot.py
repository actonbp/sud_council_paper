#!/usr/bin/env python3
"""
01_label_interest_fewshot.py
Purpose: Use Google Gemini 1.5 Pro with few-shot learning to label focus group PARTICIPANTS for SUD counseling interest
Author: AI Assistant, 2025-08-01
"""

import os
import pandas as pd
import json
import time
from pathlib import Path
try:
    # Try new google-genai SDK first
    from google import genai
    USE_NEW_SDK = True
except ImportError:
    # Fall back to older SDK
    import google.generativeai as genai
    USE_NEW_SDK = False
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

def setup_gemini():
    """Initialize Gemini API"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    if USE_NEW_SDK:
        # New SDK approach
        client = genai.Client(api_key=api_key)
        return client
    else:
        # Old SDK approach
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        return model

def load_focus_group_data():
    """Load and combine all focus group CSV files"""
    data_dir = Path('data/study2')
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    all_data = []
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        df['session'] = file_path.stem  # Add session identifier
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Filter out moderator utterances (2-3 letter codes like LR, BR, etc.)
    combined_df = combined_df[
        ~combined_df['Speaker'].astype(str).str.match(r'^[A-Z]{2,3}$', na=False)
    ]
    
    # Remove rows with missing text
    combined_df = combined_df.dropna(subset=['Text'])
    
    print(f"Loaded {len(combined_df)} participant utterances from {len(csv_files)} sessions")
    return combined_df

def create_participant_data(df):
    """Group utterances by participant ID and combine all their text"""
    participant_data = []
    
    for participant_id in df['Speaker'].unique():
        participant_utterances = df[df['Speaker'] == participant_id]
        
        # Combine all utterances for this participant
        combined_text = ' '.join(participant_utterances['Text'].astype(str))
        
        # Get sessions this participant appeared in
        sessions = participant_utterances['session'].unique().tolist()
        
        participant_data.append({
            'participant_id': participant_id,
            'sessions': sessions,
            'num_utterances': len(participant_utterances),
            'combined_text': combined_text
        })
    
    participant_df = pd.DataFrame(participant_data)
    print(f"Created participant-level data for {len(participant_df)} participants")
    return participant_df

def create_fewshot_labeling_prompt():
    """Create the few-shot learning prompt for Gemini to label participant interest level"""
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
                # New SDK approach
                response = model_or_client.models.generate_content(
                    model="gemini-1.5-pro",
                    contents=prompt
                )
                label = response.text.strip().upper()
            else:
                # Old SDK approach
                response = model_or_client.generate_content(prompt)
                label = response.text.strip().upper()
            
            # Validate response - only INTERESTED or NOT_INTERESTED
            valid_labels = ['INTERESTED', 'NOT_INTERESTED']
            if label in valid_labels:
                return label
            else:
                print(f"Invalid label '{label}', retrying...")
                continue
                
        except Exception as e:
            print(f"API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return "ERROR"
    
    return "ERROR"

def load_existing_results(cache_file):
    """Load existing results to avoid re-labeling"""
    if cache_file.exists():
        return pd.read_csv(cache_file)
    return pd.DataFrame()

def save_results(df, cache_file):
    """Save results to CSV"""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    print(f"Results saved to {cache_file}")

def main():
    """Main execution function"""
    print("🚀 Starting Gemini participant-level labeling (FEW-SHOT VERSION)...")
    
    # Setup
    model = setup_gemini()
    df = load_focus_group_data()
    participant_df = create_participant_data(df)
    
    # Create cache file for results
    cache_file = Path('results/study2/participant_labels_fewshot.csv')
    existing_results = load_existing_results(cache_file)
    
    # Check which participants already processed
    if not existing_results.empty:
        processed_ids = set(existing_results['participant_id'])
        participants_to_process = participant_df[~participant_df['participant_id'].isin(processed_ids)]
        print(f"Found {len(existing_results)} existing labels, processing {len(participants_to_process)} new participants")
    else:
        participants_to_process = participant_df
        print(f"Processing all {len(participants_to_process)} participants")
    
    # Process participants
    results = []
    for idx, row in participants_to_process.iterrows():
        participant_id = row['participant_id']
        combined_text = row['combined_text']
        sessions = row['sessions']
        num_utterances = row['num_utterances']
        
        print(f"Processing participant {participant_id} ({num_utterances} utterances)...")
        
        # Get label from Gemini
        label = label_participant_fewshot(model, combined_text)
        
        # Store result
        result = {
            'participant_id': participant_id,
            'sessions': str(sessions),  # Convert list to string for CSV
            'num_utterances': num_utterances,
            'gemini_label': label
        }
        results.append(result)
        
        # Rate limiting - be respectful to API
        time.sleep(1)
        
        # Save progress every 5 participants
        if len(results) % 5 == 0:
            temp_df = pd.DataFrame(results)
            if not existing_results.empty:
                temp_df = pd.concat([existing_results, temp_df], ignore_index=True)
            save_results(temp_df, cache_file)
            print(f"Progress saved: {len(results)} new participant labels completed")
    
    # Final save
    if results:
        new_results_df = pd.DataFrame(results)
        if not existing_results.empty:
            final_df = pd.concat([existing_results, new_results_df], ignore_index=True)
        else:
            final_df = new_results_df
        
        save_results(final_df, cache_file)
        
        # Print summary statistics
        label_counts = final_df['gemini_label'].value_counts()
        print("\n📊 Participant Labeling Summary (FEW-SHOT):")
        for label, count in label_counts.items():
            pct = (count / len(final_df)) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")
        
        print(f"\n✅ Few-shot labeling complete! {len(final_df)} total participants labeled")
        print("\nUsing exemplar participants:")
        print("- Participant 117: INTERESTED (8-9/10 interest, psychology major)")
        print("- Participant 147: NOT INTERESTED (nursing major, no personal connection)")
    else:
        print("✅ All participants already processed!")

if __name__ == "__main__":
    main()