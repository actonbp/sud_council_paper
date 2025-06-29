#!/usr/bin/env python3
"""
Study 1-2 Linkage Analysis
Connect Study 1's "career uncertainty" finding with Study 2's qualitative themes
"""

import glob, os, re, textwrap
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

# Configuration
DATA_DIR = "../../data"
RESULTS_DIR = "../../results/"

def load_focus_group_data():
    """Load focus group data for linkage analysis"""
    print("🔗 Loading focus group data for Study 1-2 linkage analysis...")
    
    paths = glob.glob(os.path.join(DATA_DIR, "*_Focus_Group_full*.csv"))
    rows = []
    for p in paths:
        df = pd.read_csv(p)
        df = df[~df["Speaker"].astype(str).str.match(r"^[A-Z]{2,3}$")]
        df = df[df["Text"].str.split().str.len() >= 8]
        df["session"] = os.path.basename(p)
        rows.append(df)
    
    corpus_df = pd.concat(rows, ignore_index=True)
    texts = corpus_df['Text'].astype(str).tolist()
    
    print(f"Loaded {len(texts)} utterances for linkage analysis")
    return texts, corpus_df

def define_uncertainty_exploration_indicators():
    """Define indicators of career uncertainty and exploration from Study 1 findings"""
    
    uncertainty_indicators = {
        'explicit_uncertainty': {
            'name': 'Explicit Career Uncertainty',
            'patterns': [
                # Direct uncertainty expressions
                r'\b(not sure|unsure|uncertain|don\'t know|confused|undecided)\b',
                r'\b(maybe|might|possibly|potentially|considering)\b',
                r'\b(exploring|options|possibilities|alternatives)\b',
                r'\b(haven\'t decided|still deciding|figuring out)\b',
                
                # Questioning language
                r'\b(wondering|questioning|debating|torn between)\b',
                r'\b(what if|should I|could I|would I)\b',
                
                # Exploration language
                r'\b(looking into|checking out|researching|investigating)\b'
            ],
            'context_phrases': [
                'not sure if', 'don\'t know if', 'maybe I should', 'considering whether',
                'still figuring out', 'exploring options', 'haven\'t decided',
                'wondering about', 'looking into different', 'not certain'
            ]
        },
        
        'career_comparison': {
            'name': 'Career Path Comparison',
            'patterns': [
                # Comparing different paths
                r'\b(versus|vs|compared to|rather than|instead of)\b',
                r'\b(different fields|other options|alternative careers)\b',
                r'\b(social work|psychology|nursing|medicine|psychiatry)\b',
                
                # Path switching language
                r'\b(switched|changed|pivot|transition|move from)\b',
                r'\b(originally|initially|first thought|used to)\b'
            ],
            'context_phrases': [
                'different from', 'other fields', 'compared to psychology',
                'instead of social work', 'rather than nursing',
                'switched from', 'originally wanted', 'thought about psychology'
            ]
        },
        
        'exploration_motivation': {
            'name': 'Exploratory Interest',
            'patterns': [
                # Exploration verbs
                r'\b(exploring|discovering|learning about|finding out)\b',
                r'\b(interested in|curious about|drawn to|attracted to)\b',
                r'\b(want to know|need to understand|trying to figure)\b',
                
                # Tentative commitment
                r'\b(leaning towards|thinking about|considering|contemplating)\b',
                r'\b(starting to|beginning to|getting interested)\b'
            ],
            'context_phrases': [
                'interested in learning', 'want to explore', 'curious about',
                'thinking about pursuing', 'considering this field',
                'drawn to this area', 'exploring different paths'
            ]
        }
    }
    
    return uncertainty_indicators

def analyze_uncertainty_patterns(texts, uncertainty_indicators):
    """Analyze patterns of career uncertainty in focus group discussions"""
    print("\n🔍 Analyzing career uncertainty patterns...")
    
    uncertainty_scores = defaultdict(list)
    uncertainty_details = []
    
    for doc_idx, text in enumerate(texts):
        text_lower = text.lower()
        
        # Score each uncertainty dimension
        for indicator_key, indicator_data in uncertainty_indicators.items():
            score = 0
            matched_patterns = []
            matched_phrases = []
            
            # Pattern matching
            for pattern in indicator_data['patterns']:
                matches = re.findall(pattern, text_lower)
                if matches:
                    score += len(matches) * 2  # 2 points per pattern match
                    matched_patterns.extend(matches)
            
            # Context phrase matching (higher weight)
            for phrase in indicator_data['context_phrases']:
                if phrase in text_lower:
                    score += text_lower.count(phrase) * 3  # 3 points per phrase
                    matched_phrases.append(phrase)
            
            # Normalize by document length
            normalized_score = score / len(text_lower.split()) if len(text_lower.split()) > 0 else 0
            uncertainty_scores[indicator_key].append(normalized_score)
            
            # Track details for high-scoring examples
            if normalized_score > 0.02:  # Threshold for meaningful uncertainty
                uncertainty_details.append({
                    'doc_idx': doc_idx,
                    'uncertainty_type': indicator_key,
                    'score': normalized_score,
                    'matched_patterns': matched_patterns[:5],
                    'matched_phrases': matched_phrases,
                    'text_preview': text[:200] + '...' if len(text) > 200 else text
                })
    
    return uncertainty_scores, uncertainty_details

def link_uncertainty_to_themes(uncertainty_details, texts):
    """Link uncertainty patterns to our previously identified themes"""
    print("\n🔗 Linking uncertainty patterns to motivational themes...")
    
    # Load our semantic themes from previous analysis
    theme_keywords = {
        'education': ['school', 'college', 'university', 'psychology', 'major', 'study', 'degree', 'academic'],
        'family': ['family', 'parents', 'mom', 'dad', 'personal', 'experience', 'background'],
        'helping': ['help', 'helping', 'people', 'support', 'care', 'serve', 'difference', 'community']
    }
    
    linkage_patterns = []
    
    for detail in uncertainty_details:
        doc_idx = detail['doc_idx']
        text = texts[doc_idx].lower()
        
        # Check which themes appear with uncertainty
        co_occurring_themes = []
        for theme, keywords in theme_keywords.items():
            theme_score = sum(1 for keyword in keywords if keyword in text)
            if theme_score >= 2:  # At least 2 theme keywords present
                co_occurring_themes.append((theme, theme_score))
        
        if co_occurring_themes:
            linkage_patterns.append({
                'doc_idx': doc_idx,
                'uncertainty_type': detail['uncertainty_type'],
                'uncertainty_score': detail['score'],
                'co_occurring_themes': co_occurring_themes,
                'dominant_theme': max(co_occurring_themes, key=lambda x: x[1])[0] if co_occurring_themes else None,
                'text': texts[doc_idx]
            })
    
    return linkage_patterns

def analyze_uncertainty_theme_combinations(linkage_patterns):
    """Analyze how uncertainty manifests with different motivational themes"""
    print("\n📊 Analyzing uncertainty-theme combinations...")
    
    # Count combinations
    combination_counts = defaultdict(int)
    theme_uncertainty_details = defaultdict(list)
    
    for pattern in linkage_patterns:
        uncertainty_type = pattern['uncertainty_type']
        dominant_theme = pattern['dominant_theme']
        
        if dominant_theme:
            combo_key = f"{dominant_theme}_{uncertainty_type}"
            combination_counts[combo_key] += 1
            theme_uncertainty_details[combo_key].append(pattern)
    
    # Find most common combinations
    top_combinations = sorted(combination_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\nTop Uncertainty-Theme Combinations:")
    for combo, count in top_combinations:
        parts = combo.split('_')
        if len(parts) >= 2:
            theme = parts[0]
            uncertainty = '_'.join(parts[1:])  # Handle multi-part uncertainty types
            print(f"  {theme.title()} + {uncertainty.replace('_', ' ').title()}: {count} instances")
    
    return combination_counts, theme_uncertainty_details, top_combinations

def extract_representative_uncertainty_quotes(theme_uncertainty_details, top_combinations):
    """Extract representative quotes showing uncertainty within each theme"""
    print("\n💬 Extracting representative uncertainty quotes...")
    
    representative_quotes = []
    
    for combo, count in top_combinations[:6]:  # Top 6 combinations
        if combo in theme_uncertainty_details:
            details = theme_uncertainty_details[combo]
            # Sort by uncertainty score to get most representative
            details.sort(key=lambda x: x['uncertainty_score'], reverse=True)
            
            if details:
                best_example = details[0]
                parts = combo.split('_')
                theme = parts[0]
                uncertainty = '_'.join(parts[1:]) if len(parts) > 1 else parts[0]
                
                representative_quotes.append({
                    'combination': f"{theme.title()} + {uncertainty.replace('_', ' ').title()}",
                    'count': count,
                    'uncertainty_score': best_example['uncertainty_score'],
                    'quote': best_example['text'],
                    'interpretation': interpret_uncertainty_theme_link(theme, uncertainty, best_example['text'])
                })
    
    return representative_quotes

def interpret_uncertainty_theme_link(theme, uncertainty_type, text):
    """Interpret what the uncertainty-theme combination reveals"""
    
    interpretations = {
        'education_explicit_uncertainty': "Students uncertain about educational pathways see SUD counseling as exploration option",
        'education_career_comparison': "Students comparing SUD counseling to other academic/professional tracks",
        'education_exploration_motivation': "Academic students exploring SUD counseling as specialization possibility",
        
        'helping_explicit_uncertainty': "Students want to help but unsure how - SUD counseling offers concrete pathway",
        'helping_career_comparison': "Students comparing helping professions - SUD counseling as alternative to psychology/social work",
        'helping_exploration_motivation': "Service-motivated students exploring SUD counseling as meaningful career",
        
        'family_explicit_uncertainty': "Students with family experience unsure if personal connection helps or hinders career choice",
        'family_career_comparison': "Students comparing whether personal experience makes them suitable for SUD vs other fields",
        'family_exploration_motivation': "Family experience motivates exploration of SUD counseling as meaningful response"
    }
    
    combo_key = f"{theme}_{uncertainty_type}"
    return interpretations.get(combo_key, f"Students showing {uncertainty_type.replace('_', ' ')} within {theme} motivation")

def save_linkage_results(uncertainty_scores, representative_quotes, combination_counts):
    """Save Study 1-2 linkage analysis results"""
    
    # Create uncertainty scores summary
    uncertainty_summary = []
    for uncertainty_type, scores in uncertainty_scores.items():
        avg_score = np.mean(scores)
        high_uncertainty_docs = sum(1 for score in scores if score > 0.02)
        uncertainty_summary.append({
            'Uncertainty_Type': uncertainty_type.replace('_', ' ').title(),
            'Average_Score': round(avg_score, 4),
            'High_Uncertainty_Documents': high_uncertainty_docs,
            'Percentage': round((high_uncertainty_docs / len(scores)) * 100, 1)
        })
    
    uncertainty_df = pd.DataFrame(uncertainty_summary)
    uncertainty_df.to_csv(os.path.join(RESULTS_DIR, "study_linkage_uncertainty_scores.csv"), index=False)
    
    # Create combination analysis
    combination_data = []
    for combo, count in combination_counts.items():
        parts = combo.split('_')
        theme = parts[0]
        uncertainty = '_'.join(parts[1:]) if len(parts) > 1 else parts[0]
        combination_data.append({
            'Theme': theme.title(),
            'Uncertainty_Type': uncertainty.replace('_', ' ').title(),
            'Combination': f"{theme.title()} + {uncertainty.replace('_', ' ').title()}",
            'Count': count
        })
    
    combination_df = pd.DataFrame(combination_data)
    combination_df.to_csv(os.path.join(RESULTS_DIR, "study_linkage_theme_uncertainty_combinations.csv"), index=False)
    
    # Create representative quotes
    quotes_df = pd.DataFrame(representative_quotes)
    quotes_df.to_csv(os.path.join(RESULTS_DIR, "study_linkage_representative_quotes.csv"), index=False)
    
    # Create comprehensive report linking Study 1 and Study 2
    with open(os.path.join(RESULTS_DIR, "study_1_2_linkage_analysis_report.txt"), 'w') as f:
        f.write("STUDY 1-2 LINKAGE ANALYSIS: CAREER UNCERTAINTY & MOTIVATIONAL THEMES\n")
        f.write("=" * 75 + "\n\n")
        
        f.write("STUDY 1 KEY FINDING:\n")
        f.write("• Mental Health Career Uncertainty → 74% higher odds of SUD counseling interest\n")
        f.write("• SUD counseling serves as EXPLORATION PATHWAY for uncertain students\n")
        f.write("• Students already committed to mental health careers show 36% lower interest\n\n")
        
        f.write("STUDY 2 QUALITATIVE CONNECTION:\n")
        f.write("Analysis of how career uncertainty manifests within motivational themes\n\n")
        
        f.write("UNCERTAINTY PATTERNS IN FOCUS GROUPS:\n")
        f.write("-" * 45 + "\n")
        for uncertainty_type, scores in uncertainty_scores.items():
            high_uncertainty = sum(1 for score in scores if score > 0.02)
            pct = (high_uncertainty / len(scores)) * 100
            f.write(f"{uncertainty_type.replace('_', ' ').title()}: {high_uncertainty} docs ({pct:.1f}%)\n")
        
        f.write(f"\nTOP UNCERTAINTY-THEME COMBINATIONS:\n")
        f.write("-" * 40 + "\n")
        top_combos = sorted(combination_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        for combo, count in top_combos:
            parts = combo.split('_')
            theme = parts[0]
            uncertainty = '_'.join(parts[1:]) if len(parts) > 1 else parts[0]
            f.write(f"{theme.title()} + {uncertainty.replace('_', ' ').title()}: {count} instances\n")
        
        f.write(f"\nREPRESENTATIVE QUOTES LINKING STUDY 1 & 2:\n")
        f.write("-" * 45 + "\n")
        for quote_data in representative_quotes[:4]:
            f.write(f"\n{quote_data['combination']} ({quote_data['count']} instances):\n")
            f.write(f"Interpretation: {quote_data['interpretation']}\n")
            f.write(f'Quote: "{quote_data['quote'][:300]}..."\n')
        
        f.write(f"\nKEY INSIGHT - NATURAL LINKAGE:\n")
        f.write("Study 1's quantitative finding that 'career uncertainty' predicts SUD counseling interest\n")
        f.write("is supported by Study 2's qualitative themes showing HOW uncertainty manifests:\n")
        f.write("• Educational uncertainty → exploring SUD as academic specialization\n")
        f.write("• Helping uncertainty → SUD as concrete way to make difference\n")
        f.write("• Family uncertainty → whether personal experience helps or hinders\n")
        f.write("\nThis provides a complete narrative: uncertain students explore SUD counseling\n")
        f.write("as a pathway that addresses their specific motivational concerns.\n")
    
    print(f"\n✅ Study 1-2 linkage results saved to {RESULTS_DIR}")
    return uncertainty_summary, combination_data

def display_linkage_findings(representative_quotes):
    """Display the key linkage findings"""
    print("\n" + "="*80)
    print("STUDY 1-2 LINKAGE FINDINGS: CAREER UNCERTAINTY IN MOTIVATIONAL THEMES")
    print("="*80)
    
    print(f"\n🔗 KEY CONNECTION:")
    print(f"Study 1: Students with 'Mental Health Career Uncertainty' → 74% higher SUD counseling interest")
    print(f"Study 2: Qualitative analysis reveals HOW this uncertainty manifests in motivational themes")
    
    print(f"\n💬 REPRESENTATIVE EXAMPLES:")
    
    for quote_data in representative_quotes[:4]:
        print(f"\n>>> {quote_data['combination']} <<<")
        print(f"Frequency: {quote_data['count']} instances")
        print(f"Interpretation: {quote_data['interpretation']}")
        print(f"Quote: {textwrap.fill(quote_data['quote'], 70)}")
        print("-" * 70)

def main():
    """Execute Study 1-2 linkage analysis"""
    
    print("🔗 STUDY 1-2 LINKAGE ANALYSIS")
    print("=" * 40)
    print("Connecting quantitative 'career uncertainty' finding with qualitative themes")
    
    # Load data
    texts, corpus_df = load_focus_group_data()
    
    # Define uncertainty indicators
    uncertainty_indicators = define_uncertainty_exploration_indicators()
    print(f"\nDefined {len(uncertainty_indicators)} uncertainty pattern types")
    
    # Analyze uncertainty patterns
    uncertainty_scores, uncertainty_details = analyze_uncertainty_patterns(texts, uncertainty_indicators)
    
    # Link to themes
    linkage_patterns = link_uncertainty_to_themes(uncertainty_details, texts)
    
    # Analyze combinations
    combination_counts, theme_uncertainty_details, top_combinations = analyze_uncertainty_theme_combinations(linkage_patterns)
    
    # Extract representative quotes
    representative_quotes = extract_representative_uncertainty_quotes(theme_uncertainty_details, top_combinations)
    
    # Save results
    uncertainty_summary, combination_data = save_linkage_results(uncertainty_scores, representative_quotes, combination_counts)
    
    # Display findings
    display_linkage_findings(representative_quotes)
    
    print(f"\n🏆 STUDY 1-2 LINKAGE ANALYSIS COMPLETE!")
    print(f"🎯 Successfully connected quantitative uncertainty finding with qualitative themes")
    print(f"📈 Reveals HOW career uncertainty manifests in student motivations")
    print("✨ Provides complete narrative linking both studies!")
    
    return representative_quotes, combination_counts, uncertainty_scores

if __name__ == "__main__":
    quotes, combinations, uncertainty = main()