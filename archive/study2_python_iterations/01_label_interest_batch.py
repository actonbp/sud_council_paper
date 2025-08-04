#!/usr/bin/env python3
"""
01_label_interest_batch.py
Purpose: Batch process focus group utterances with Gemini 2.5 Pro
Designed to run efficiently in background with progress tracking
"""

import os
import pandas as pd
import json
import time
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
import re
from datetime import datetime
import sys

# Load environment variables
load_dotenv()

def setup_gemini():
    """Initialize Gemini API"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    
    # Try different model names
    model_names = [
        'gemini-2.5-pro',
        'gemini-pro',
        'gemini-1.5-pro',
        'gemini-1.5-flash'
    ]
    
    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            print(f"✅ Successfully initialized model: {model_name}")
            return model
        except Exception as e:
            print(f"❌ Failed to initialize {model_name}: {e}")
            continue
    
    raise ValueError("Could not initialize any Gemini model")

def load_focus_group_data():
    """Load and combine all focus group CSV files"""
    data_dir = Path('data/study2')
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    all_data = []
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        df['session'] = file_path.stem
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Filter out moderator utterances
    combined_df = combined_df[
        ~combined_df['Speaker'].astype(str).str.match(r'^[A-Z]{2,3}$', na=False)
    ]
    
    # Remove rows with missing text
    combined_df = combined_df.dropna(subset=['Text'])
    
    print(f"Loaded {len(combined_df)} participant utterances from {len(csv_files)} sessions")
    return combined_df

def create_batch_prompt(utterances_batch):
    """Create a batch prompt for multiple utterances"""
    prompt = """You are an expert researcher analyzing focus group discussions about career interests in substance use disorder (SUD) counseling.

Your task: For each numbered utterance below, determine if the speaker expresses interest in SUD counseling as a career.

Label each as:
- INTERESTED: Speaker expresses positive interest, curiosity, or attraction to SUD counseling careers
- NOT_INTERESTED: Speaker expresses disinterest, concerns, or negative views about SUD counseling careers  
- NEUTRAL: Speaker discusses the topic without expressing personal interest/disinterest

Guidelines:
- Focus on PERSONAL interest/career intentions, not general opinions
- Look for: "I would", "I'm interested", "I could see myself", "I might"
- Disinterest: "not for me", "wouldn't want to", "not interested", "too difficult"

Respond with ONLY the labels in order, one per line. Example:
NEUTRAL
INTERESTED
NOT_INTERESTED

Utterances to analyze:
"""
    
    for i, text in enumerate(utterances_batch, 1):
        prompt += f"\n{i}. \"{text}\""
    
    return prompt

def process_batch(model, batch_df, batch_size=5):
    """Process utterances in batches"""
    results = []
    
    for i in range(0, len(batch_df), batch_size):
        batch = batch_df.iloc[i:i+batch_size]
        utterances = batch['Text'].tolist()
        
        prompt = create_batch_prompt(utterances)
        
        try:
            response = model.generate_content(prompt)
            labels = response.text.strip().upper().split('\n')
            
            # Validate we got the right number of labels
            if len(labels) == len(utterances):
                for j, row in enumerate(batch.itertuples()):
                    label = labels[j].strip()
                    if label not in ['INTERESTED', 'NOT_INTERESTED', 'NEUTRAL']:
                        label = 'ERROR'
                    
                    results.append({
                        'utterance_id': row.Index,
                        'session': row.session,
                        'speaker': row.Speaker,
                        'text': row.Text,
                        'gemini_label': label
                    })
            else:
                # Fall back to individual processing
                for row in batch.itertuples():
                    label = label_single_utterance(model, row.Text)
                    results.append({
                        'utterance_id': row.Index,
                        'session': row.session,
                        'speaker': row.Speaker,
                        'text': row.Text,
                        'gemini_label': label
                    })
            
            # Progress update
            processed = min(i + batch_size, len(batch_df))
            pct = (processed / len(batch_df)) * 100
            print(f"Progress: {processed}/{len(batch_df)} ({pct:.1f}%)")
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Batch error: {e}, falling back to individual processing")
            # Process individually on error
            for row in batch.itertuples():
                label = label_single_utterance(model, row.Text)
                results.append({
                    'utterance_id': row.Index,
                    'session': row.session,
                    'speaker': row.Speaker,
                    'text': row.Text,
                    'gemini_label': label
                })
    
    return results

def label_single_utterance(model, text, max_retries=3):
    """Label a single utterance as fallback"""
    prompt = f"""Label if this utterance shows interest in SUD counseling careers.
Respond with ONLY: INTERESTED, NOT_INTERESTED, or NEUTRAL

Utterance: "{text}"
"""
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            label = response.text.strip().upper()
            
            if label in ['INTERESTED', 'NOT_INTERESTED', 'NEUTRAL']:
                return label
            else:
                return 'ERROR'
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return "ERROR"
    
    return "ERROR"

def main():
    """Main execution function"""
    print(f"🚀 Starting batch Gemini utterance labeling at {datetime.now()}")
    
    # Setup
    model = setup_gemini()
    df = load_focus_group_data()
    
    # Create output directory
    output_dir = Path('results/study2')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing results
    cache_file = output_dir / 'llm_labels.csv'
    if cache_file.exists():
        existing = pd.read_csv(cache_file)
        processed_ids = set(existing['utterance_id'])
        df = df[~df.index.isin(processed_ids)]
        print(f"Found {len(existing)} existing labels, processing {len(df)} new utterances")
    else:
        existing = pd.DataFrame()
    
    if len(df) == 0:
        print("✅ All utterances already processed!")
        return
    
    # Process in batches
    print(f"Processing {len(df)} utterances in batches...")
    results = process_batch(model, df, batch_size=5)
    
    # Combine with existing results
    new_results_df = pd.DataFrame(results)
    if not existing.empty:
        final_df = pd.concat([existing, new_results_df], ignore_index=True)
    else:
        final_df = new_results_df
    
    # Save results
    final_df.to_csv(cache_file, index=False)
    
    # Save progress log
    log_file = output_dir / 'labeling_log.txt'
    with open(log_file, 'a') as f:
        f.write(f"\n{datetime.now()}: Processed {len(results)} utterances\n")
    
    # Print summary
    label_counts = final_df['gemini_label'].value_counts()
    print("\n📊 Final Labeling Summary:")
    for label, count in label_counts.items():
        pct = (count / len(final_df)) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    print(f"\n✅ Batch labeling complete at {datetime.now()}")
    print(f"Results saved to: {cache_file}")

if __name__ == "__main__":
    main()