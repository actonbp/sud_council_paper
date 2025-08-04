#!/usr/bin/env python3
"""
01_label_interest_liberal.py
Purpose: Use Google Gemini 1.5 Pro to label focus group PARTICIPANTS for SUD counseling interest (LIBERAL VERSION)
Author: AI Assistant, 2025-07-17
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

def create_liberal_labeling_prompt():
    """Create the LIBERAL prompt for Gemini to label participant interest level"""
    return """
You are an expert researcher analyzing focus group discussions about career interests in substance use disorder (SUD) counseling.

Your task: Read ALL utterances from a single participant and determine their overall interest or OPENNESS to SUD counseling as a career.

Based on everything this person said across the focus group(s), classify them as:
- "INTERESTED" - Participant expresses positive interest, curiosity, openness, or any willingness to consider SUD counseling careers
- "NOT_INTERESTED" - Participant expresses clear disinterest, strong concerns, or definitive rejection of SUD counseling careers

LIBERAL GUIDELINES - Err on the side of INTERESTED:
- Look for ANY signs of curiosity, consideration, or openness (even if hesitant)
- Keywords indicating INTERESTED: "I would", "I'm interested", "I could see myself", "I might", "sounds appealing", "I'd consider", "it's interesting", "I'd be open to", "maybe", "possibly", "I think it would be", "I'd give it a try"
- Only label NOT_INTERESTED if there are CLEAR, DEFINITIVE statements of rejection
- NOT_INTERESTED keywords: "definitely not for me", "I would never", "absolutely not interested", "I hate the idea", "I refuse to consider"
- If someone discusses challenges but still seems open or curious, label as INTERESTED
- If someone expresses both interest and concerns, lean toward INTERESTED
- If someone seems neutral or undecided, lean toward INTERESTED

Be LIBERAL in your interpretation - we want to capture anyone who might be open to this career path, even if not explicitly enthusiastic.

Respond with ONLY the label (INTERESTED or NOT_INTERESTED). No explanation needed.

All utterances from this participant:
"""

def label_participant_liberal(model_or_client, participant_text, max_retries=3):
    """Label a participant's overall interest using liberal Gemini prompt"""
    prompt = create_liberal_labeling_prompt() + f'"{participant_text}"'
    
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
    print("🚀 Starting Gemini participant-level labeling (LIBERAL VERSION)...")
    
    # Setup
    model = setup_gemini()
    df = load_focus_group_data()
    participant_df = create_participant_data(df)
    
    # Create cache file for results
    cache_file = Path('results/study2/participant_labels_liberal.csv')
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
        label = label_participant_liberal(model, combined_text)
        
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
        print("\n📊 Participant Labeling Summary (LIBERAL):")
        for label, count in label_counts.items():
            pct = (count / len(final_df)) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")
        
        print(f"\n✅ Liberal labeling complete! {len(final_df)} total participants labeled")
    else:
        print("✅ All participants already processed!")

if __name__ == "__main__":
    main()