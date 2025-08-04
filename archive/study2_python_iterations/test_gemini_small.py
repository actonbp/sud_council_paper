#!/usr/bin/env python3
"""
test_gemini_small.py
Purpose: Test Gemini labeling on just 5 utterances to validate approach
"""

import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

def setup_gemini():
    """Initialize Gemini API"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro')
    return model

def create_labeling_prompt():
    """Create the prompt for Gemini to label utterances"""
    return """
You are an expert researcher analyzing focus group discussions about career interests in substance use disorder (SUD) counseling.

Your task: Read each utterance and determine if the speaker expresses interest in SUD counseling as a career.

Label each utterance as one of:
- "INTERESTED" - Speaker expresses positive interest, curiosity, or attraction to SUD counseling careers
- "NOT_INTERESTED" - Speaker expresses disinterest, concerns, or negative views about SUD counseling careers  
- "NEUTRAL" - Speaker discusses the topic without expressing personal interest/disinterest

Guidelines:
- Focus on PERSONAL interest/career intentions, not general opinions about the field
- Look for keywords like: "I would", "I'm interested", "I could see myself", "I might", "sounds appealing"
- Disinterest keywords: "not for me", "wouldn't want to", "not interested", "too difficult"
- Be conservative - only label INTERESTED/NOT_INTERESTED if there's clear personal positioning

Respond with ONLY the label (INTERESTED, NOT_INTERESTED, or NEUTRAL). No explanation needed.

Utterance to analyze:
"""

def test_small_sample():
    """Test on just 5 utterances"""
    print("🧪 Testing Gemini labeling on small sample...")
    
    # Setup model
    model = setup_gemini()
    
    # Load one CSV file and get first 5 non-moderator utterances
    data_dir = Path('data/study2')
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    # Load first file
    df = pd.read_csv(csv_files[0])
    
    # Filter out moderator utterances
    df = df[~df['Speaker'].astype(str).str.match(r'^[A-Z]{2,3}$', na=False)]
    df = df.dropna(subset=['Text'])
    
    # Get first 5 utterances
    test_utterances = df.head(5)
    
    print(f"Testing on {len(test_utterances)} utterances from {csv_files[0].name}")
    print("-" * 80)
    
    # Test each utterance
    for idx, row in test_utterances.iterrows():
        speaker = row['Speaker']
        text = row['Text']
        
        # Create prompt
        prompt = create_labeling_prompt() + f'"{text}"'
        
        try:
            # Get label from Gemini
            response = model.generate_content(prompt)
            label = response.text.strip().upper()
            
            print(f"\nSpeaker {speaker}:")
            print(f"Text: {text[:100]}...")
            print(f"Label: {label}")
            
        except Exception as e:
            print(f"Error processing utterance {idx}: {e}")
    
    print("\n✅ Small sample test complete!")

if __name__ == "__main__":
    test_small_sample()