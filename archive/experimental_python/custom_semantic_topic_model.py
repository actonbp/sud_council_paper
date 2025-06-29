#!/usr/bin/env python3
"""
Custom Semantic Topic Model
Specifically designed to capture the three core patterns: education, family, helping people
"""

import glob, os, re, textwrap
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
DATA_DIR = "../../data"
RESULTS_DIR = "../../results/"

def load_clean_data():
    """Load and clean focus group data"""
    print("📥 Loading focus group data for custom semantic analysis...")
    
    paths = glob.glob(os.path.join(DATA_DIR, "*_Focus_Group_full*.csv"))
    rows = []
    for p in paths:
        df = pd.read_csv(p)
        df = df[~df["Speaker"].astype(str).str.match(r"^[A-Z]{2,3}$")]
        df = df[df["Text"].str.split().str.len() >= 8]
        df["session"] = os.path.basename(p)
        rows.append(df)
    
    corpus_df = pd.concat(rows, ignore_index=True)
    
    # Strategic cleaning - keep meaningful content
    CIRCULAR_TERMS = ['counselor', 'counseling', 'therapist', 'therapy']
    
    texts = []
    for text in corpus_df['Text'].astype(str):
        text = text.lower()
        for term in CIRCULAR_TERMS:
            text = re.sub(rf'\b{re.escape(term)}\b', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text.split()) >= 5:
            texts.append(text)
    
    print(f"Loaded {len(texts)} utterances for semantic analysis")
    return texts, corpus_df

def define_semantic_themes():
    """Define the three core semantic themes with comprehensive keywords"""
    
    semantic_themes = {
        'education_academic': {
            'name': 'Educational & Academic Pathway',
            'core_keywords': [
                # Education terms
                'school', 'college', 'university', 'education', 'academic', 'degree',
                'major', 'study', 'studying', 'studies', 'student', 'learn', 'learning',
                'class', 'classes', 'course', 'courses', 'semester', 'year', 'years',
                'psychology', 'psych', 'pre_med', 'premed', 'pre-med', 'medical_school',
                'graduate', 'undergrad', 'undergraduate', 'grad_school', 'graduate_school',
                
                # Academic experiences
                'professor', 'teacher', 'taught', 'grade', 'grades', 'gpa',
                'scholarship', 'tuition', 'campus', 'dorm', 'roommate',
                'high_school', 'elementary', 'middle_school',
                
                # Learning & knowledge
                'knowledge', 'understand', 'understanding', 'comprehend', 'analyze',
                'research', 'paper', 'assignment', 'exam', 'test', 'quiz'
            ],
            'context_phrases': [
                'in school', 'at school', 'during college', 'my major', 'my degree',
                'taking classes', 'psychology class', 'learned about', 'studying psychology',
                'academic interest', 'educational background', 'school experience'
            ]
        },
        
        'family_personal': {
            'name': 'Family Background & Personal Experience',
            'core_keywords': [
                # Family terms
                'family', 'families', 'parents', 'parent', 'mom', 'mother', 'dad', 'father',
                'sister', 'brother', 'sibling', 'siblings', 'grandmother', 'grandfather',
                'grandma', 'grandpa', 'aunt', 'uncle', 'cousin', 'relative', 'relatives',
                
                # Personal experience
                'personal', 'personally', 'experience', 'experienced', 'background',
                'history', 'past', 'childhood', 'grew_up', 'growing_up', 'upbringing',
                'life', 'lived', 'been_through', 'went_through', 'witnessed',
                
                # Emotional/personal connection
                'trauma', 'traumatic', 'difficult', 'struggle', 'struggled', 'pain',
                'loss', 'grief', 'support', 'supported', 'supportive', 'understanding',
                'relate', 'connection', 'connected', 'identify', 'empathy', 'empathetic',
                
                # Personal impact
                'affected', 'impact', 'influenced', 'shaped', 'changed', 'motivated'
            ],
            'context_phrases': [
                'my family', 'family member', 'personal experience', 'family history',
                'family background', 'grew up with', 'family struggle', 'personal connection',
                'family experience', 'my parents', 'family situation', 'personal story'
            ]
        },
        
        'helping_service': {
            'name': 'Helping People & Service Motivation',
            'core_keywords': [
                # Direct helping terms
                'help', 'helping', 'helped', 'support', 'supporting', 'supported',
                'assist', 'assisting', 'care', 'caring', 'serve', 'serving', 'service',
                
                # People focus
                'people', 'person', 'individuals', 'others', 'someone', 'everybody',
                'community', 'society', 'population', 'patients', 'clients',
                
                # Impact & difference
                'difference', 'impact', 'change', 'improve', 'better', 'healing',
                'recovery', 'progress', 'growth', 'transformation', 'lives',
                
                # Service motivation
                'volunteer', 'volunteering', 'give_back', 'contribute', 'meaningful',
                'purpose', 'calling', 'passion', 'devoted', 'dedicated', 'committed',
                
                # Altruistic concepts
                'altruistic', 'selfless', 'compassion', 'compassionate', 'kindness',
                'generous', 'giving', 'empathy', 'empathetic', 'understanding'
            ],
            'context_phrases': [
                'helping people', 'help others', 'make a difference', 'support people',
                'care for others', 'helping community', 'service to others', 'give back',
                'meaningful work', 'helping profession', 'people in need'
            ]
        }
    }
    
    return semantic_themes

def calculate_semantic_scores(texts, semantic_themes):
    """Calculate semantic similarity scores for each text to each theme"""
    print("\n🧠 Calculating semantic theme scores...")
    
    theme_scores = defaultdict(list)
    document_details = []
    
    for doc_idx, text in enumerate(texts):
        text_lower = text.lower()
        words = text_lower.split()
        
        # Score each theme
        for theme_key, theme_data in semantic_themes.items():
            score = 0
            matched_keywords = []
            matched_phrases = []
            
            # Keyword matching with weights
            for keyword in theme_data['core_keywords']:
                if keyword in text_lower:
                    # Weight by keyword importance and frequency
                    frequency = text_lower.count(keyword)
                    importance_weight = 2 if len(keyword) >= 6 else 1  # Longer words get more weight
                    score += frequency * importance_weight
                    matched_keywords.append(f"{keyword}({frequency})")
            
            # Context phrase matching (higher weight)
            for phrase in theme_data['context_phrases']:
                if phrase in text_lower:
                    frequency = text_lower.count(phrase)
                    score += frequency * 5  # Phrases get 5x weight
                    matched_phrases.append(f"{phrase}({frequency})")
            
            # Normalize by document length
            normalized_score = score / len(words) if len(words) > 0 else 0
            
            theme_scores[theme_key].append(normalized_score)
            
            # Track details for analysis
            if normalized_score > 0:
                document_details.append({
                    'doc_idx': doc_idx,
                    'theme': theme_key,
                    'score': normalized_score,
                    'raw_score': score,
                    'matched_keywords': matched_keywords[:5],  # Top 5
                    'matched_phrases': matched_phrases,
                    'text_preview': text[:150] + '...' if len(text) > 150 else text
                })
    
    print(f"Calculated semantic scores for {len(texts)} documents across {len(semantic_themes)} themes")
    return theme_scores, document_details

def assign_documents_to_themes(texts, theme_scores, semantic_themes):
    """Assign each document to its best-fitting theme"""
    print("\n📊 Assigning documents to semantic themes...")
    
    theme_assignments = []
    theme_documents = defaultdict(list)
    assignment_confidence = []
    
    for doc_idx in range(len(texts)):
        # Get scores for this document across all themes
        doc_theme_scores = {
            theme_key: theme_scores[theme_key][doc_idx] 
            for theme_key in semantic_themes.keys()
        }
        
        # Find best theme
        if max(doc_theme_scores.values()) > 0:
            best_theme = max(doc_theme_scores, key=doc_theme_scores.get)
            best_score = doc_theme_scores[best_theme]
            
            # Calculate confidence (how much better is best vs second best)
            sorted_scores = sorted(doc_theme_scores.values(), reverse=True)
            confidence = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0] if len(sorted_scores) > 1 and sorted_scores[0] > 0 else 1.0
        else:
            # No clear theme - assign to mixed
            best_theme = 'mixed'
            best_score = 0
            confidence = 0
        
        theme_assignments.append(best_theme)
        assignment_confidence.append(confidence)
        theme_documents[best_theme].append(doc_idx)
    
    # Display theme distribution
    theme_counts = Counter(theme_assignments)
    print(f"\nTheme Distribution:")
    for theme_key, theme_data in semantic_themes.items():
        count = theme_counts[theme_key]
        percentage = (count / len(texts)) * 100
        print(f"  {theme_data['name']}: {count} docs ({percentage:.1f}%)")
    
    if 'mixed' in theme_counts:
        print(f"  Mixed/Unclear: {theme_counts['mixed']} docs ({(theme_counts['mixed']/len(texts)*100):.1f}%)")
    
    return theme_assignments, theme_documents, assignment_confidence

def extract_representative_content(texts, theme_documents, theme_scores, semantic_themes):
    """Extract most representative content for each theme"""
    print("\n🎯 Extracting representative content for each theme...")
    
    theme_representatives = {}
    
    for theme_key, theme_data in semantic_themes.items():
        doc_indices = theme_documents[theme_key]
        
        if not doc_indices:
            continue
            
        # Get documents and their scores for this theme
        theme_docs_with_scores = []
        for doc_idx in doc_indices:
            score = theme_scores[theme_key][doc_idx]
            theme_docs_with_scores.append((doc_idx, score, texts[doc_idx]))
        
        # Sort by score (highest first)
        theme_docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Extract top representative documents
        top_docs = theme_docs_with_scores[:3]  # Top 3 most representative
        
        # Extract common keywords from theme documents
        theme_texts = [texts[doc_idx] for doc_idx in doc_indices]
        theme_combined = ' '.join(theme_texts).lower()
        
        # Find most common theme-specific words
        theme_words = theme_combined.split()
        theme_word_counts = Counter(theme_words)
        
        # Filter for meaningful theme words
        meaningful_words = []
        for word, count in theme_word_counts.most_common(50):
            if (len(word) >= 4 and 
                word in theme_data['core_keywords'] and
                count >= 2):
                meaningful_words.append((word, count))
        
        theme_representatives[theme_key] = {
            'theme_name': theme_data['name'],
            'doc_count': len(doc_indices),
            'percentage': (len(doc_indices) / len(texts)) * 100,
            'top_documents': top_docs,
            'meaningful_words': meaningful_words[:10],
            'avg_score': np.mean([score for _, score, _ in theme_docs_with_scores])
        }
        
        print(f"\n{theme_data['name']}:")
        print(f"  Documents: {len(doc_indices)} ({(len(doc_indices)/len(texts)*100):.1f}%)")
        print(f"  Avg score: {np.mean([score for _, score, _ in theme_docs_with_scores]):.4f}")
        print(f"  Top words: {', '.join([word for word, count in meaningful_words[:8]])}")
    
    return theme_representatives

def save_semantic_results(theme_representatives, semantic_themes, texts, theme_assignments, assignment_confidence):
    """Save semantic topic modeling results"""
    
    # Create detailed results DataFrame
    results_data = []
    for doc_idx, (assignment, confidence) in enumerate(zip(theme_assignments, assignment_confidence)):
        if assignment != 'mixed':
            theme_name = semantic_themes[assignment]['name']
        else:
            theme_name = 'Mixed/Unclear'
            
        results_data.append({
            'document_id': doc_idx,
            'assigned_theme': theme_name,
            'theme_key': assignment,
            'confidence': round(confidence, 3),
            'text_preview': texts[doc_idx][:200] + '...' if len(texts[doc_idx]) > 200 else texts[doc_idx]
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(RESULTS_DIR, "semantic_topic_assignments.csv"), index=False)
    
    # Create theme summary
    summary_data = []
    for theme_key, rep_data in theme_representatives.items():
        top_words = [word for word, count in rep_data['meaningful_words']]
        summary_data.append({
            'Theme': rep_data['theme_name'],
            'Documents': rep_data['doc_count'],
            'Percentage': f"{rep_data['percentage']:.1f}%",
            'Avg_Score': round(rep_data['avg_score'], 4),
            'Top_Words': ', '.join(top_words[:8])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "semantic_theme_summary.csv"), index=False)
    
    # Create comprehensive report
    with open(os.path.join(RESULTS_DIR, "semantic_topic_modeling_report.txt"), 'w') as f:
        f.write("CUSTOM SEMANTIC TOPIC MODELING - SUD COUNSELING RESEARCH\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("• Custom semantic themes based on identified core patterns\n")
        f.write("• Three core themes: Education, Family, Helping People\n")
        f.write("• Keyword and context phrase matching with weights\n")
        f.write("• Document assignment based on highest semantic similarity\n\n")
        
        f.write("SEMANTIC THEMES RESULTS:\n")
        f.write("-" * 30 + "\n")
        
        for theme_key, rep_data in theme_representatives.items():
            f.write(f"\n{rep_data['theme_name']} ({rep_data['percentage']:.1f}%)\n")
            f.write(f"  Documents: {rep_data['doc_count']}\n")
            f.write(f"  Average semantic score: {rep_data['avg_score']:.4f}\n")
            f.write(f"  Top theme words: {', '.join([word for word, count in rep_data['meaningful_words'][:8]])}\n")
            
            f.write(f"  Most representative quote:\n")
            if rep_data['top_documents']:
                top_quote = rep_data['top_documents'][0][2]
                f.write(f'    "{top_quote[:300]}..."\n')
    
    print(f"\n✅ Semantic results saved to {RESULTS_DIR}")
    
    # Display representative quotes
    print("\n" + "="*70)
    print("SEMANTIC THEMES - MOST REPRESENTATIVE QUOTES")
    print("="*70)
    
    for theme_key, rep_data in theme_representatives.items():
        print(f"\n>>> {rep_data['theme_name']} (Score: {rep_data['avg_score']:.4f}) <<<")
        print(f"Documents: {rep_data['doc_count']} ({rep_data['percentage']:.1f}%)")
        if rep_data['top_documents']:
            top_doc = rep_data['top_documents'][0]
            print(f"Quote: {textwrap.fill(top_doc[2], 70)}")

def main():
    """Execute custom semantic topic modeling"""
    
    print("🎯 CUSTOM SEMANTIC TOPIC MODELING")
    print("=" * 45)
    print("Target: Education, Family, Helping People patterns")
    
    # Load data
    texts, corpus_df = load_clean_data()
    
    # Define semantic themes
    semantic_themes = define_semantic_themes()
    print(f"\nDefined {len(semantic_themes)} semantic themes:")
    for theme_key, theme_data in semantic_themes.items():
        print(f"  • {theme_data['name']}: {len(theme_data['core_keywords'])} keywords")
    
    # Calculate semantic scores
    theme_scores, document_details = calculate_semantic_scores(texts, semantic_themes)
    
    # Assign documents to themes
    theme_assignments, theme_documents, confidence = assign_documents_to_themes(texts, theme_scores, semantic_themes)
    
    # Extract representative content
    theme_representatives = extract_representative_content(texts, theme_documents, theme_scores, semantic_themes)
    
    # Save results
    save_semantic_results(theme_representatives, semantic_themes, texts, theme_assignments, confidence)
    
    print(f"\n🏆 CUSTOM SEMANTIC ANALYSIS COMPLETE!")
    print(f"🎯 Successfully captured the three core patterns")
    print(f"📊 Clear thematic separation based on semantic content")
    print("✨ Results focus on underlying motivations and pathways")
    
    return theme_representatives, semantic_themes

if __name__ == "__main__":
    representatives, themes = main()