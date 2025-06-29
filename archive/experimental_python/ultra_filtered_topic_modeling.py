#!/usr/bin/env python3
"""
Ultra-Filtered Topic Modeling
Removes ALL generic terms to find truly specific underlying themes
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Gensim for phrase detection and coherence
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

# Sklearn imports
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = "../../data/"
RESULTS_DIR = "../../results/"
CSV_FILES = [
    "11_4_2024_11am_Focus_Group_full (1).csv",
    "11_6_2024_130pm_Focus_Group_full.csv", 
    "11_8_2024_12pm_Focus_Group_full.csv",
    "11_11_2024_4pm_Focus_Group_full (1).csv",
    "11_12_2024_11am_Focus_Group_full.csv",
    "11_14_2024_4_pm_Focus_group__full.csv",
]

# ULTRA-COMPREHENSIVE FILTERING
# Remove target domain terms + generic career terms + filler words
TARGET_DOMAIN_TERMS = set("""
substance substances abuse abusing abused abuser abusers
drug drugs drugged drugging addiction addicted addictive addict addicts
alcohol alcoholic alcoholism alcoholics drinking drink drinks drunk
counseling counselor counselors counsel counselled counselling
therapy therapist therapists therapeutic therapeutics
mental_health mental health psychological psychology psychologist psychologists
treatment treating treated treatments rehabilitation rehab recovery
sud suds clinic clinical clinician clinicians patient patients client clients
session sessions appointment appointments diagnosis diagnostic diagnose diagnosed
intervention interventions disorder disorders condition conditions
symptom symptoms medication medications medicine medicines
hospital hospitals facility facilities program programs service services
professional professionals profession
""".split())

# GENERIC CAREER/WORK TERMS
GENERIC_CAREER_TERMS = set("""
job jobs work working worked worker workers workplace
career careers field fields profession professional professionals
occupation occupational occupations
employment employed employer employee employees
business businesses company companies organization organizations
industry industries sector sectors
opportunity opportunities position positions role roles
""".split())

# FILLER WORDS AND GENERIC DESCRIPTORS  
FILLER_GENERIC_TERMS = set("""
little bit different pretty stuff things thing
important good bad better best worse worst
nice great awesome terrible horrible amazing
big small large huge tiny
easy hard difficult simple complex
interesting boring fun exciting
general specific particular certain
kind kinds type types sort sorts
way ways manner
""".split())

# Standard stopwords
STOP_WORDS = set("""
a about above after again against all am an and any are as at be because been before being
between both but by could did do does doing down during each few for from further had has
have having he her here hers herself him himself his how i if in into is it its itself
just like me more most my myself nor not of off on once only or other our ours ourselves
out over own same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were what when where
which while who whom why will with you your yours yourself yourselves

um uh yeah okay kinda sorta right would know think really kind going lot can say 
definitely want guess something able way actually maybe feel feels felt get got make 
made see say said sure look looking yes no dont don't thats that's gonna wanna 
like really actually probably maybe perhaps guess suppose might could would should
also even just now well much many still back come came put take took give gave
go went come came one two three first second next last another other
time times someone something anything nothing everything somebody anybody nobody everybody

re ve ll don didn won isn aren weren hasn haven couldn wouldn shouldn mustn needn mightn
""".split())

# Combine all filtering terms
ALL_FILTERED_TERMS = TARGET_DOMAIN_TERMS.union(GENERIC_CAREER_TERMS).union(FILLER_GENERIC_TERMS).union(STOP_WORDS)


class UltraFilteredTopicModeler:
    """Ultra-aggressive filtering to find truly specific underlying content"""
    
    def __init__(self):
        self.filtered_phrases = []
        self.all_terms_removed = {}
        
    def load_data(self):
        """Load and clean data"""
        docs = []
        
        print(">>> Loading focus group data...")
        for fn in CSV_FILES:
            fp = os.path.join(DATA_DIR, fn)
            df = pd.read_csv(fp)
            original_count = len(df)
            
            # Remove moderator rows and empty text
            df = df[~df["Speaker"].astype(str).str.match(r"^[A-Z]{2,3}$")]
            df = df[df["Text"].notna()]
            
            # Keep substantive utterances
            df = df[df["Text"].str.split().str.len() >= 10]
            
            texts = df["Text"].astype(str).tolist()
            docs.extend(texts)
            
            print(f"{fn:45} | {original_count:3} → {len(df):3} utterances")
        
        print(f"Total utterances: {len(docs)}")
        self.raw_docs = docs
        
    def ultra_aggressive_filtering(self):
        """Remove ALL generic terms to find truly specific content"""
        print("\n>>> Ultra-aggressive filtering (removing ALL generic terms)...")
        
        filtered_docs = []
        removed_terms_count = {}
        
        for doc in self.raw_docs:
            # Convert to lowercase for processing
            doc_lower = doc.lower()
            
            # Track what we're removing
            original_words = doc_lower.split()
            
            # Remove ALL filtered terms
            for term in ALL_FILTERED_TERMS:
                if term in doc_lower:
                    # Count removals
                    count = len(re.findall(rf'\b{re.escape(term)}\w*\b', doc_lower))
                    if count > 0:
                        if term not in removed_terms_count:
                            removed_terms_count[term] = 0
                        removed_terms_count[term] += count
                        
                        # Remove the term
                        doc_lower = re.sub(rf'\b{re.escape(term)}\w*\b', ' ', doc_lower)
            
            # Clean up extra spaces
            doc_lower = re.sub(r'\s+', ' ', doc_lower).strip()
            
            # Only keep docs that still have meaningful content after ultra-filtering
            remaining_words = doc_lower.split()
            meaningful_words = [w for w in remaining_words if len(w) >= 4]  # Longer words only
            
            if len(meaningful_words) >= 3:  # Must have at least 3 substantial words left
                filtered_docs.append(doc_lower)
        
        print(f"Documents after ultra-filtering: {len(filtered_docs)}")
        print(f"Total filtered terms: {len(removed_terms_count)}")
        
        # Show most frequently removed terms by category
        domain_removed = {k: v for k, v in removed_terms_count.items() if k in TARGET_DOMAIN_TERMS}
        career_removed = {k: v for k, v in removed_terms_count.items() if k in GENERIC_CAREER_TERMS}
        filler_removed = {k: v for k, v in removed_terms_count.items() if k in FILLER_GENERIC_TERMS}
        
        print(f"\nRemoved {len(domain_removed)} domain terms, {len(career_removed)} career terms, {len(filler_removed)} filler terms")
        
        # Show top removed terms
        top_removed = sorted(removed_terms_count.items(), key=lambda x: x[1], reverse=True)[:20]
        print("Most frequently removed terms:")
        for term, count in top_removed:
            category = "DOMAIN" if term in TARGET_DOMAIN_TERMS else "CAREER" if term in GENERIC_CAREER_TERMS else "FILLER" if term in FILLER_GENERIC_TERMS else "STOP"
            print(f"  - '{term}' ({category}): {count}x")
        
        self.all_terms_removed = removed_terms_count
        return filtered_docs
        
    def detect_ultra_specific_phrases(self, docs):
        """Detect phrases from ultra-filtered, specific content only"""
        print("\n>>> Detecting ultra-specific phrases...")
        
        # Clean and expand contractions
        cleaned_docs = []
        for doc in docs:
            # Expand contractions
            contractions = {
                "don't": "do not", "won't": "will not", "can't": "cannot",
                "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
                "'d": " would", "'m": " am"
            }
            
            for contraction, expansion in contractions.items():
                doc = doc.replace(contraction, expansion)
            
            # Remove punctuation, normalize spaces
            doc = re.sub(r"[^\w\s]", " ", doc)
            doc = re.sub(r"\s+", " ", doc).strip()
            cleaned_docs.append(doc)
        
        # Tokenize for phrase detection - very strict filtering
        sentences = []
        for doc in cleaned_docs:
            tokens = simple_preprocess(doc, deacc=True, min_len=4, max_len=25)  # Longer words only
            # Ultra-strict filtering
            specific_tokens = []
            for token in tokens:
                if (token not in ALL_FILTERED_TERMS and 
                    len(token) >= 4 and  # Must be substantial words
                    not token.isdigit() and
                    not re.match(r'^[a-z]{1,3}$', token)):  # No short words
                    specific_tokens.append(token)
            
            if len(specific_tokens) >= 3:  # Need at least 3 specific words
                sentences.append(specific_tokens)
        
        print(f"Prepared {len(sentences)} documents with ultra-specific content")
        
        # Multi-level phrase detection with lower thresholds (less data now)
        print("   Detecting specific bigrams...")
        bigram_model = Phrases(sentences, min_count=2, threshold=0.2, delimiter="_", scoring='npmi')
        bigram_phraser = Phraser(bigram_model)
        bigram_sentences = [bigram_phraser[sent] for sent in sentences]
        
        print("   Detecting specific trigrams...")
        trigram_model = Phrases(bigram_sentences, min_count=2, threshold=0.15, delimiter="_", scoring='npmi')
        trigram_phraser = Phraser(trigram_model)
        final_sentences = [trigram_phraser[sent] for sent in bigram_sentences]
        
        # Extract truly specific phrases
        all_phrases = set()
        for sent in final_sentences:
            for token in sent:
                if "_" in token:
                    # Double-check no filtered terms
                    phrase_words = token.split("_")
                    if not any(word in ALL_FILTERED_TERMS for word in phrase_words):
                        all_phrases.add(token)
        
        specific_phrases = list(all_phrases)
        bigrams = [p for p in specific_phrases if len(p.split("_")) == 2]
        trigrams = [p for p in specific_phrases if len(p.split("_")) >= 3]
        
        print(f"   Detected {len(bigrams)} ultra-specific bigrams")
        print(f"   Detected {len(trigrams)} ultra-specific trigrams+")
        print(f"   Sample specific phrases: {specific_phrases[:15]}")
        
        self.filtered_phrases = specific_phrases
        self.tokens_with_phrases = final_sentences
        
        # Convert back to documents
        docs_with_phrases = [" ".join(sent) for sent in final_sentences]
        return docs_with_phrases
        
    def create_ultra_specific_features(self, docs):
        """Create feature matrix from ultra-specific content only"""
        print("\n>>> Creating ultra-specific feature matrix...")
        
        def ultra_specific_analyzer(text):
            tokens = text.split()
            features = []
            
            for token in tokens:
                # Triple-check no filtered terms
                if (token not in ALL_FILTERED_TERMS and
                    len(token) >= 4 and  # Substantial words only
                    not token.isdigit() and
                    not re.match(r'^[a-z]{1,3}$', token)):
                    
                    # Weight phrases higher
                    if "_" in token:
                        n_words = len(token.split("_"))
                        weight = min(n_words + 1, 4)  # Cap at 4x weight
                        features.extend([token] * weight)
                    else:
                        features.append(token)
            
            return features
        
        # Use TF-IDF with ultra-specific filtering
        vectorizer = TfidfVectorizer(
            analyzer=ultra_specific_analyzer,
            min_df=2,  # Lower threshold due to aggressive filtering
            max_df=0.7,
            max_features=200,  # Fewer features due to filtering
            sublinear_tf=True,
            norm='l2'
        )
        
        X = vectorizer.fit_transform(docs)
        vocab = vectorizer.get_feature_names_out()
        
        # Final vocabulary quality check
        ultra_clean_vocab_indices = []
        ultra_clean_vocab = []
        for i, term in enumerate(vocab):
            # Check every part of compound terms
            term_parts = term.replace("_", " ").split()
            if (not any(part in ALL_FILTERED_TERMS for part in term_parts) and
                all(len(part) >= 4 for part in term_parts)):  # All parts must be substantial
                ultra_clean_vocab_indices.append(i)
                ultra_clean_vocab.append(term)
        
        if len(ultra_clean_vocab_indices) < len(vocab):
            X = X[:, ultra_clean_vocab_indices]
            vocab = np.array(ultra_clean_vocab)
            print(f"Final ultra-clean vocabulary: {len(ultra_clean_vocab)} terms")
        
        # Analyze vocabulary composition
        ngram_count = sum(1 for term in vocab if "_" in term)
        single_word_count = len(vocab) - ngram_count
        
        print(f"Ultra-specific feature matrix: {X.shape[0]} docs × {X.shape[1]} features")
        print(f"  - {ngram_count} specific phrases ({ngram_count/len(vocab):.1%})")
        print(f"  - {single_word_count} specific words ({single_word_count/len(vocab):.1%})")
        print(f"Sample ultra-specific vocabulary: {list(vocab[:15])}")
        
        return X, vocab, vectorizer
        
    def find_optimal_k_ultra_specific(self, X, vocab, k_range=range(2, 6)):
        """Find optimal k for ultra-specific analysis"""
        print("\n>>> Finding optimal k for ultra-specific themes...")
        
        results = []
        
        for k in k_range:
            print(f"Testing k={k}...")
            
            # Test both algorithms
            for alg_name in ['lda', 'nmf']:
                if alg_name == 'lda':
                    model = LatentDirichletAllocation(
                        n_components=k, max_iter=300, random_state=42,
                        doc_topic_prior=0.2, topic_word_prior=0.02,  # Less focused due to limited data
                        learning_method='batch')
                else:
                    model = NMF(
                        n_components=k, init='nndsvd', max_iter=1500, random_state=42,
                        alpha_W=0.05, alpha_H=0.05, l1_ratio=0.2)  # Less regularization
                
                # Fit model
                doc_topic = model.fit_transform(X)
                assignments = doc_topic.argmax(axis=1)
                
                # Get topics
                topics = []
                for topic_idx, topic in enumerate(model.components_):
                    top_indices = topic.argsort()[-15:][::-1]  # Fewer terms available
                    top_terms = [vocab[i] for i in top_indices]
                    topics.append(top_terms)
                
                # Calculate ultra-specific metrics
                
                # 1. Specificity score (prefer longer, more specific terms)
                specificity = 0
                term_count = 0
                for topic in topics:
                    for term in topic[:10]:
                        if "_" in term:
                            specificity += len(term.split("_")) * 2  # Multi-word bonus
                        else:
                            specificity += len(term) / 5  # Length bonus for single words
                        term_count += 1
                specificity = specificity / term_count if term_count > 0 else 0
                
                # 2. Diversity (no overlap)
                all_terms = []
                for topic in topics:
                    all_terms.extend(topic[:10])
                diversity = len(set(all_terms)) / len(all_terms) if all_terms else 0
                
                # 3. Coherence
                cm_cv = CoherenceModel(topics=topics, texts=self.tokens_with_phrases,
                                     dictionary=Dictionary(self.tokens_with_phrases), coherence='c_v')
                coherence = cm_cv.get_coherence()
                
                # 4. Balance
                topic_counts = [np.sum(assignments == i) for i in range(k)]
                balance = 1 - np.std(topic_counts) / np.mean(topic_counts) if np.mean(topic_counts) > 0 else 0
                
                # Combined ultra-specific score
                ultra_specific_score = (specificity * 0.35 +  # Prioritize specificity
                                      diversity * 0.25 +
                                      coherence * 0.25 +
                                      balance * 0.15)
                
                results.append({
                    'k': k,
                    'algorithm': alg_name,
                    'specificity': specificity,
                    'diversity': diversity,
                    'coherence': coherence,
                    'balance': balance,
                    'ultra_specific_score': ultra_specific_score,
                    'topics': topics,
                    'assignments': assignments,
                    'model': model
                })
                
                print(f"  {alg_name.upper()}: specificity={specificity:.3f}, diversity={diversity:.3f}, "
                      f"coherence={coherence:.3f}, score={ultra_specific_score:.3f}")
        
        # Find best configuration
        best_config = max(results, key=lambda x: x['ultra_specific_score'])
        
        print(f"\n>>> OPTIMAL ULTRA-SPECIFIC CONFIGURATION:")
        print(f"    k={best_config['k']}, algorithm={best_config['algorithm'].upper()}")
        print(f"    Ultra-specific score: {best_config['ultra_specific_score']:.3f}")
        
        return best_config, pd.DataFrame(results)
        
    def interpret_ultra_specific_themes(self, topics, assignments):
        """Interpret themes based on ultra-specific content"""
        print("\n>>> Interpreting ultra-specific themes...")
        
        # Ultra-specific theme patterns (what's left after aggressive filtering)
        specific_patterns = {
            'family_personal': {
                'keywords': ['family', 'parents', 'relatives', 'mother', 'father', 'sister', 'brother', 'grandmother', 'grandfather'],
                'label': 'Family & Personal Experience'
            },
            'helping_care': {
                'keywords': ['helping', 'caring', 'support', 'assist', 'nurture', 'comfort'],
                'label': 'Helping & Care Orientation'
            },
            'learning_education': {
                'keywords': ['learning', 'education', 'teaching', 'studying', 'knowledge', 'understanding'],
                'label': 'Learning & Education Focus'
            },
            'people_relationships': {
                'keywords': ['people', 'relationships', 'connection', 'communication', 'interaction'],
                'label': 'People & Relationships'
            },
            'emotional_feelings': {
                'keywords': ['emotional', 'feelings', 'empathy', 'compassion', 'understanding'],
                'label': 'Emotional & Empathetic Focus'
            },
            'community_social': {
                'keywords': ['community', 'social', 'society', 'neighborhood', 'group'],
                'label': 'Community & Social Focus'
            },
            'experience_background': {
                'keywords': ['experience', 'background', 'history', 'past', 'lived', 'witnessed'],
                'label': 'Experience & Background'
            }
        }
        
        interpretations = []
        theme_analyses = []
        
        for i, topic_terms in enumerate(topics):
            doc_count = sum(assignments == i)
            
            # Separate phrases and single words
            phrases = [t for t in topic_terms[:12] if "_" in t]
            single_words = [t for t in topic_terms[:12] if "_" not in t]
            
            # Score against ultra-specific patterns
            pattern_scores = {}
            for pattern_key, pattern_info in specific_patterns.items():
                score = 0
                
                # Check phrases
                for phrase in phrases:
                    phrase_words = phrase.replace("_", " ").split()
                    for word in phrase_words:
                        for keyword in pattern_info['keywords']:
                            if keyword in word or word in keyword:
                                score += 3  # High phrase bonus
                
                # Check single words
                for word in single_words:
                    for keyword in pattern_info['keywords']:
                        if keyword in word or word in keyword:
                            score += 2  # Word bonus
                
                if score > 0:
                    pattern_scores[pattern_key] = score
            
            # Generate interpretation
            if pattern_scores:
                best_pattern = max(pattern_scores, key=pattern_scores.get)
                interpretation = specific_patterns[best_pattern]['label']
            else:
                # Fallback based on most specific terms
                if phrases:
                    main_concept = phrases[0].replace("_", " ").title()
                    interpretation = f"{main_concept} Theme"
                elif single_words:
                    # Use longest/most specific word
                    longest_word = max(single_words[:3], key=len)
                    interpretation = f"{longest_word.title()}-Focused Theme"
                else:
                    interpretation = "Specific Content Theme"
            
            interpretations.append(f"{interpretation} ({doc_count} docs)")
            
            theme_analyses.append({
                'topic_id': i + 1,
                'interpretation': interpretation,
                'doc_count': doc_count,
                'specific_phrases': phrases[:8],
                'specific_words': single_words[:12],
                'pattern_scores': pattern_scores,
                'specificity_level': len([t for t in topic_terms[:10] if len(t) >= 6])  # Count substantial terms
            })
            
            print(f"\nTopic {i+1}: {interpretation} ({doc_count} docs)")
            if phrases:
                print(f"  Specific phrases: {', '.join(phrases[:5])}")
            print(f"  Specific words: {', '.join(single_words[:10])}")
            print(f"  Specificity level: {theme_analyses[-1]['specificity_level']}/10 substantial terms")
        
        return interpretations, theme_analyses
        
    def create_ultra_specific_visualization(self, results, theme_analyses):
        """Create visualization emphasizing ultra-specific content"""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(20, 16))
        
        gs = fig.add_gridspec(4, 3, hspace=0.5, wspace=0.3)
        
        k = len(theme_analyses)
        colors = plt.cm.Set1(np.linspace(0, 1, k))
        
        # 1. Topic distribution
        ax1 = fig.add_subplot(gs[0, 0])
        doc_counts = [analysis['doc_count'] for analysis in theme_analyses]
        bars = ax1.bar(range(k), doc_counts, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Ultra-Specific Themes\n(All Generic Terms Removed)', fontweight='bold')
        ax1.set_xlabel('Theme')
        ax1.set_ylabel('Documents')
        ax1.set_xticks(range(k))
        ax1.set_xticklabels([f'T{i+1}' for i in range(k)])
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Filtering impact breakdown
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Count different types of removed terms
        domain_count = sum(1 for term in self.all_terms_removed.keys() if term in TARGET_DOMAIN_TERMS)
        career_count = sum(1 for term in self.all_terms_removed.keys() if term in GENERIC_CAREER_TERMS)
        filler_count = sum(1 for term in self.all_terms_removed.keys() if term in FILLER_GENERIC_TERMS)
        
        categories = ['Domain\nTerms', 'Career\nTerms', 'Filler\nWords']
        counts = [domain_count, career_count, filler_count]
        colors_filter = ['lightcoral', 'lightsalmon', 'lightgray']
        
        bars = ax2.bar(categories, counts, color=colors_filter, alpha=0.8)
        ax2.set_title('Filtered Terms by Category', fontweight='bold')
        ax2.set_ylabel('Number of Term Types')
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Specificity levels
        ax3 = fig.add_subplot(gs[0, 2])
        specificity_levels = [analysis['specificity_level'] for analysis in theme_analyses]
        bars = ax3.bar(range(k), specificity_levels, color=colors, alpha=0.8)
        ax3.set_title('Theme Specificity Level', fontweight='bold')
        ax3.set_xlabel('Theme')
        ax3.set_ylabel('Substantial Terms (6+ chars)')
        ax3.set_xticks(range(k))
        ax3.set_xticklabels([f'T{i+1}' for i in range(k)])
        ax3.set_ylim(0, 10)
        
        for bar, level in zip(bars, specificity_levels):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{level}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Most removed terms (top section)
        ax4 = fig.add_subplot(gs[1, :])
        top_removed = sorted(self.all_terms_removed.items(), key=lambda x: x[1], reverse=True)[:15]
        if top_removed:
            terms, counts = zip(*top_removed)
            y_pos = range(len(terms))
            
            # Color by category
            bar_colors = []
            for term in terms:
                if term in TARGET_DOMAIN_TERMS:
                    bar_colors.append('lightcoral')
                elif term in GENERIC_CAREER_TERMS:
                    bar_colors.append('lightsalmon')
                elif term in FILLER_GENERIC_TERMS:
                    bar_colors.append('lightgray')
                else:
                    bar_colors.append('lightblue')
            
            bars = ax4.barh(y_pos, counts, color=bar_colors, alpha=0.8)
            ax4.set_title('Most Frequently Removed Terms (Ultra-Aggressive Filtering)', fontweight='bold')
            ax4.set_xlabel('Frequency')
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels([f'{term}' for term in terms])
            ax4.invert_yaxis()
        
        # 5. Ultra-specific themes display (large section)
        ax5 = fig.add_subplot(gs[2:, :])
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        
        y_positions = np.linspace(0.9, 0.1, k)
        
        for i, analysis in enumerate(theme_analyses):
            y_pos = y_positions[i]
            color = colors[i]
            
            # Theme header
            ax5.text(0.02, y_pos, f"Theme {analysis['topic_id']}: {analysis['interpretation']}", 
                    transform=ax5.transAxes, fontsize=16, fontweight='bold', color=color)
            
            # Specific phrases (if any)
            if analysis['specific_phrases']:
                phrases_text = f"Ultra-specific phrases: {', '.join(analysis['specific_phrases'])}"
                ax5.text(0.02, y_pos - 0.05, phrases_text, 
                        transform=ax5.transAxes, fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='gold', 
                                 alpha=0.9, edgecolor=color, linewidth=2))
            
            # Specific words
            words_text = f"Ultra-specific words: {', '.join(analysis['specific_words'][:15])}"
            y_offset = 0.10 if analysis['specific_phrases'] else 0.05
            ax5.text(0.02, y_pos - y_offset, words_text, 
                    transform=ax5.transAxes, fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', 
                             alpha=0.9, edgecolor=color, linewidth=1))
            
            # Specificity and document info
            stats_text = f"Documents: {analysis['doc_count']} | Specificity: {analysis['specificity_level']}/10 | Phrases: {len(analysis['specific_phrases'])}"
            ax5.text(0.02, y_pos - y_offset - 0.05, stats_text, 
                    transform=ax5.transAxes, fontsize=10, style='italic', color='gray')
        
        # Title
        title = f'Ultra-Specific Analysis: {k} Highly Focused Themes'
        subtitle = f'Removed {len(self.all_terms_removed)} generic terms | Specificity Score: {results["ultra_specific_score"]:.3f}'
        fig.suptitle(f'{title}\n{subtitle}', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = os.path.join(RESULTS_DIR, 'ultra_specific_themes.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Ultra-specific visualization saved: {output_path}")
        
    def save_ultra_specific_results(self, results, theme_analyses, k_analysis_df):
        """Save ultra-specific results"""
        
        # Main report
        with open(os.path.join(RESULTS_DIR, 'ultra_specific_themes_report.txt'), 'w') as f:
            f.write("ULTRA-SPECIFIC THEMES ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("ULTRA-AGGRESSIVE FILTERING CONFIGURATION:\n")
            f.write(f"• Number of ultra-specific themes: {len(theme_analyses)}\n")
            f.write(f"• Algorithm: {results['algorithm'].upper()}\n")
            f.write(f"• Documents analyzed: {len(results['assignments'])}\n")
            f.write(f"• Ultra-specific score: {results['ultra_specific_score']:.3f}\n\n")
            
            f.write("ULTRA-AGGRESSIVE FILTERING IMPACT:\n")
            f.write(f"• Total filtered terms: {len(self.all_terms_removed)}\n")
            f.write(f"• Total term instances removed: {sum(self.all_terms_removed.values())}\n")
            
            # Count by category
            domain_count = sum(1 for term in self.all_terms_removed.keys() if term in TARGET_DOMAIN_TERMS)
            career_count = sum(1 for term in self.all_terms_removed.keys() if term in GENERIC_CAREER_TERMS)
            filler_count = sum(1 for term in self.all_terms_removed.keys() if term in FILLER_GENERIC_TERMS)
            
            f.write(f"• Domain terms removed: {domain_count}\n")
            f.write(f"• Generic career terms removed: {career_count}\n")
            f.write(f"• Filler/generic words removed: {filler_count}\n")
            f.write(f"• Ultra-specific phrases discovered: {len(self.filtered_phrases)}\n\n")
            
            f.write("QUALITY METRICS:\n")
            f.write(f"• Specificity score: {results['specificity']:.3f}\n")
            f.write(f"• Theme diversity: {results['diversity']:.3f}\n")
            f.write(f"• Topic coherence: {results['coherence']:.3f}\n")
            f.write(f"• Theme balance: {results['balance']:.3f}\n\n")
            
            f.write("ULTRA-SPECIFIC THEMES (ALL GENERIC TERMS REMOVED):\n")
            f.write("-" * 80 + "\n")
            
            for analysis in theme_analyses:
                f.write(f"\n{analysis['interpretation']} ({analysis['doc_count']} docs)\n")
                f.write(f"  • Specificity level: {analysis['specificity_level']}/10\n")
                
                if analysis['specific_phrases']:
                    f.write(f"  • Ultra-specific phrases: {', '.join(analysis['specific_phrases'])}\n")
                f.write(f"  • Ultra-specific words: {', '.join(analysis['specific_words'])}\n")
            
            f.write(f"\nTOP REMOVED TERMS (EXAMPLES):\n")
            f.write("-" * 40 + "\n")
            top_removed = sorted(self.all_terms_removed.items(), key=lambda x: x[1], reverse=True)
            for i, (term, count) in enumerate(top_removed[:25]):
                category = "DOMAIN" if term in TARGET_DOMAIN_TERMS else "CAREER" if term in GENERIC_CAREER_TERMS else "FILLER"
                f.write(f"  {i+1:2}. '{term}' ({category}): {count}x\n")
        
        # Analysis details
        k_analysis_df.to_csv(os.path.join(RESULTS_DIR, 'ultra_specific_k_analysis.csv'), index=False)
        
        # Document assignments
        assignments_df = pd.DataFrame({
            'document_id': range(len(results['assignments'])),
            'theme_number': results['assignments'] + 1,
            'ultra_specific_theme': [theme_analyses[i]['interpretation'] for i in results['assignments']],
            'specificity_level': [theme_analyses[i]['specificity_level'] for i in results['assignments']],
            'text_preview': [doc[:300] + '...' if len(doc) > 300 else doc 
                           for doc in self.raw_docs[:len(results['assignments'])]]
        })
        assignments_df.to_csv(os.path.join(RESULTS_DIR, 'ultra_specific_assignments.csv'), index=False)
        
        print("✓ Ultra-specific results saved:")
        print("  • ultra_specific_themes_report.txt")
        print("  • ultra_specific_k_analysis.csv") 
        print("  • ultra_specific_assignments.csv")
        
    def run_ultra_specific_analysis(self):
        """Execute ultra-specific analysis"""
        print("=" * 80)
        print("ULTRA-SPECIFIC THEMES ANALYSIS")
        print("=" * 80)
        
        # Load and ultra-filter data
        self.load_data()
        filtered_docs = self.ultra_aggressive_filtering()
        docs_with_phrases = self.detect_ultra_specific_phrases(filtered_docs)
        X, vocab, vectorizer = self.create_ultra_specific_features(docs_with_phrases)
        
        # Find optimal configuration
        best_config, k_analysis_df = self.find_optimal_k_ultra_specific(X, vocab)
        
        # Interpret ultra-specific themes
        interpretations, theme_analyses = self.interpret_ultra_specific_themes(
            best_config['topics'], best_config['assignments'])
        
        # Display results
        print("\n" + "=" * 80)
        print("ULTRA-SPECIFIC THEMES DISCOVERED")
        print("=" * 80)
        print(f"Optimal k: {len(theme_analyses)}")
        print(f"Algorithm: {best_config['algorithm'].upper()}")
        print(f"Total terms filtered: {len(self.all_terms_removed)}")
        print(f"Ultra-specific score: {best_config['ultra_specific_score']:.3f}")
        
        # Save and visualize
        self.save_ultra_specific_results(best_config, theme_analyses, k_analysis_df)
        self.create_ultra_specific_visualization(best_config, theme_analyses)
        
        return best_config, theme_analyses


if __name__ == "__main__":
    modeler = UltraFilteredTopicModeler()
    results, theme_analyses = modeler.run_ultra_specific_analysis()
    
    print("\n" + "🔥" * 40)
    print("ULTRA-SPECIFIC ANALYSIS COMPLETE!")
    print("🔥" * 40)
    print(f"\nUltra-Aggressive Filtering Achievements:")
    print(f"• Found {len(theme_analyses)} ultra-specific themes")
    print(f"• Removed {len(modeler.all_terms_removed)} different generic terms")
    print(f"• Total generic instances removed: {sum(modeler.all_terms_removed.values())}")
    print(f"• Discovered {len(modeler.filtered_phrases)} ultra-specific phrases")
    print("• Removed domain terms, career words, AND filler language")
    print("• Only the most specific, meaningful content remains!")
    print("\nResult: Pure underlying motivations with zero noise!")
    print("All results saved to results/ directory")