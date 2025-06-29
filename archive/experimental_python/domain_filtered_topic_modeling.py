#!/usr/bin/env python3
"""
Domain-Filtered Topic Modeling
Removes ALL target domain terms to discover underlying themes and motivations
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

# COMPREHENSIVE TARGET DOMAIN FILTERING
# Remove ALL obvious target terms so we can discover underlying motivations
TARGET_DOMAIN_TERMS = set("""
substance substances abuse abusing abused abuser abusers
drug drugs drugged drugging addiction addicted addictive addict addicts
alcohol alcoholic alcoholism alcoholics drinking drink drinks drunk
counseling counselor counselors counsel counselled counselling
therapy therapist therapists therapeutic therapeutics
mental_health mental health psychological psychology psychologist psychologists
treatment treating treated treatments rehabilitation rehab recovery
sud suds
clinic clinical clinician clinicians
patient patients client clients
session sessions appointment appointments
diagnosis diagnostic diagnose diagnosed
intervention interventions
disorder disorders condition conditions
symptom symptoms
medication medications medicine medicines
hospital hospitals facility facilities
program programs service services
professional professionals profession
""".split())

# Generic stopwords
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
made see say said sure look looking good bad yes no dont don't thats that's gonna wanna 
like really actually probably maybe perhaps guess suppose might could would should
also even just now well much many still back come came put take took give gave
go went come came one two three first second next last another other thing things
time times someone something anything nothing everything somebody anybody nobody everybody

re ve ll don didn won isn aren weren hasn haven couldn wouldn shouldn mustn needn mightn
""".split())


class DomainFilteredTopicModeler:
    """Topic modeling that removes target domain terms to find underlying themes"""
    
    def __init__(self):
        self.filtered_phrases = []
        self.domain_terms_removed = []
        
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
            df = df[df["Text"].str.split().str.len() >= 8]
            
            texts = df["Text"].astype(str).tolist()
            docs.extend(texts)
            
            print(f"{fn:45} | {original_count:3} → {len(df):3} utterances")
        
        print(f"Total utterances: {len(docs)}")
        self.raw_docs = docs
        
    def aggressive_domain_filtering(self):
        """Aggressively remove ALL target domain terms"""
        print("\n>>> Aggressive domain term filtering...")
        
        filtered_docs = []
        removed_terms_count = {}
        
        for doc in self.raw_docs:
            # Convert to lowercase for processing
            doc_lower = doc.lower()
            
            # Track what we're removing for analysis
            original_words = doc_lower.split()
            
            # Remove target domain terms
            for term in TARGET_DOMAIN_TERMS:
                if term in doc_lower:
                    # Count removals
                    count = doc_lower.count(term)
                    if term not in removed_terms_count:
                        removed_terms_count[term] = 0
                    removed_terms_count[term] += count
                    
                    # Remove the term
                    doc_lower = re.sub(rf'\b{re.escape(term)}\w*\b', ' ', doc_lower)
            
            # Clean up extra spaces
            doc_lower = re.sub(r'\s+', ' ', doc_lower).strip()
            
            # Only keep docs that still have meaningful content after filtering
            remaining_words = doc_lower.split()
            meaningful_words = [w for w in remaining_words if w not in STOP_WORDS and len(w) >= 3]
            
            if len(meaningful_words) >= 5:  # Must have at least 5 meaningful words left
                filtered_docs.append(doc_lower)
        
        print(f"Documents after domain filtering: {len(filtered_docs)}")
        print(f"Removed {len(removed_terms_count)} different target domain terms")
        
        # Show most frequently removed terms
        top_removed = sorted(removed_terms_count.items(), key=lambda x: x[1], reverse=True)[:15]
        print("Most frequently removed domain terms:")
        for term, count in top_removed:
            print(f"  - '{term}': {count} occurrences")
        
        self.domain_terms_removed = removed_terms_count
        return filtered_docs
        
    def detect_meaningful_phrases(self, docs):
        """Detect meaningful phrases from domain-filtered content"""
        print("\n>>> Detecting meaningful phrases from filtered content...")
        
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
        
        # Tokenize for phrase detection
        sentences = []
        for doc in cleaned_docs:
            tokens = simple_preprocess(doc, deacc=True, min_len=3, max_len=25)
            # Remove domain terms and stopwords
            meaningful_tokens = []
            for token in tokens:
                if (token not in TARGET_DOMAIN_TERMS and 
                    token not in STOP_WORDS and 
                    len(token) >= 3 and
                    not token.isdigit()):
                    meaningful_tokens.append(token)
            
            if len(meaningful_tokens) >= 4:
                sentences.append(meaningful_tokens)
        
        print(f"Prepared {len(sentences)} documents for phrase detection")
        
        # Multi-level phrase detection
        print("   Detecting bigrams...")
        bigram_model = Phrases(sentences, min_count=2, threshold=0.3, delimiter="_", scoring='npmi')
        bigram_phraser = Phraser(bigram_model)
        bigram_sentences = [bigram_phraser[sent] for sent in sentences]
        
        print("   Detecting trigrams...")
        trigram_model = Phrases(bigram_sentences, min_count=2, threshold=0.2, delimiter="_", scoring='npmi')
        trigram_phraser = Phraser(trigram_model)
        final_sentences = [trigram_phraser[sent] for sent in bigram_sentences]
        
        # Extract meaningful phrases (excluding any remaining domain terms)
        all_phrases = set()
        for sent in final_sentences:
            for token in sent:
                if "_" in token:
                    # Check that phrase doesn't contain domain terms
                    phrase_words = token.split("_")
                    if not any(word in TARGET_DOMAIN_TERMS for word in phrase_words):
                        all_phrases.add(token)
        
        # Categorize phrases
        meaningful_phrases = list(all_phrases)
        bigrams = [p for p in meaningful_phrases if len(p.split("_")) == 2]
        trigrams = [p for p in meaningful_phrases if len(p.split("_")) >= 3]
        
        print(f"   Detected {len(bigrams)} meaningful bigrams")
        print(f"   Detected {len(trigrams)} meaningful trigrams+")
        print(f"   Sample meaningful phrases: {meaningful_phrases[:15]}")
        
        self.filtered_phrases = meaningful_phrases
        self.tokens_with_phrases = final_sentences
        
        # Convert back to documents
        docs_with_phrases = [" ".join(sent) for sent in final_sentences]
        return docs_with_phrases
        
    def create_domain_filtered_features(self, docs):
        """Create feature matrix from domain-filtered content"""
        print("\n>>> Creating features from domain-filtered content...")
        
        def domain_filtered_analyzer(text):
            tokens = text.split()
            features = []
            
            for token in tokens:
                # Double-check no domain terms slipped through
                if (token not in TARGET_DOMAIN_TERMS and
                    token not in STOP_WORDS and
                    len(token) >= 3 and
                    not token.isdigit()):
                    
                    # Weight n-grams higher
                    if "_" in token:
                        n_words = len(token.split("_"))
                        weight = min(n_words, 3)  # Cap at 3x weight
                        features.extend([token] * weight)
                    else:
                        features.append(token)
            
            return features
        
        # Use TF-IDF with domain filtering
        vectorizer = TfidfVectorizer(
            analyzer=domain_filtered_analyzer,
            min_df=3,
            max_df=0.6,
            max_features=250,
            sublinear_tf=True,
            norm='l2'
        )
        
        X = vectorizer.fit_transform(docs)
        vocab = vectorizer.get_feature_names_out()
        
        # Final check: ensure no domain terms in vocabulary
        clean_vocab_indices = []
        clean_vocab = []
        for i, term in enumerate(vocab):
            # Check if term or any part contains domain terms
            term_parts = term.replace("_", " ").split()
            if not any(part in TARGET_DOMAIN_TERMS for part in term_parts):
                clean_vocab_indices.append(i)
                clean_vocab.append(term)
        
        if len(clean_vocab_indices) < len(vocab):
            X = X[:, clean_vocab_indices]
            vocab = np.array(clean_vocab)
            print(f"Final vocabulary cleaning: {len(clean_vocab)} clean terms")
        
        # Analyze vocabulary composition
        ngram_count = sum(1 for term in vocab if "_" in term)
        single_word_count = len(vocab) - ngram_count
        
        print(f"Domain-filtered feature matrix: {X.shape[0]} docs × {X.shape[1]} features")
        print(f"  - {ngram_count} meaningful phrases ({ngram_count/len(vocab):.1%})")
        print(f"  - {single_word_count} single words ({single_word_count/len(vocab):.1%})")
        print(f"Sample clean vocabulary: {list(vocab[:20])}")
        
        return X, vocab, vectorizer
        
    def find_optimal_k_domain_filtered(self, X, vocab, k_range=range(2, 7)):
        """Find optimal k for domain-filtered analysis"""
        print("\n>>> Finding optimal k for underlying themes...")
        
        results = []
        
        for k in k_range:
            print(f"Testing k={k}...")
            
            # Test both algorithms
            for alg_name in ['lda', 'nmf']:
                if alg_name == 'lda':
                    model = LatentDirichletAllocation(
                        n_components=k, max_iter=200, random_state=42,
                        doc_topic_prior=0.1, topic_word_prior=0.01,
                        learning_method='batch')
                else:
                    model = NMF(
                        n_components=k, init='nndsvd', max_iter=1000, random_state=42,
                        alpha_W=0.1, alpha_H=0.1, l1_ratio=0.3)
                
                # Fit model
                doc_topic = model.fit_transform(X)
                assignments = doc_topic.argmax(axis=1)
                
                # Get topics
                topics = []
                for topic_idx, topic in enumerate(model.components_):
                    top_indices = topic.argsort()[-20:][::-1]
                    top_terms = [vocab[i] for i in top_indices]
                    topics.append(top_terms)
                
                # Calculate metrics focused on underlying themes
                
                # 1. Thematic diversity (no domain overlap)
                all_terms = []
                for topic in topics:
                    all_terms.extend(topic[:15])
                diversity = len(set(all_terms)) / len(all_terms) if all_terms else 0
                
                # 2. Coherence
                cm_cv = CoherenceModel(topics=topics, texts=self.tokens_with_phrases,
                                     dictionary=Dictionary(self.tokens_with_phrases), coherence='c_v')
                coherence = cm_cv.get_coherence()
                
                # 3. Balance
                topic_counts = [np.sum(assignments == i) for i in range(k)]
                balance = 1 - np.std(topic_counts) / np.mean(topic_counts) if np.mean(topic_counts) > 0 else 0
                
                # 4. Underlying theme quality (prefer longer, more specific terms)
                theme_quality = 0
                for topic in topics:
                    for term in topic[:10]:
                        if "_" in term:
                            theme_quality += len(term.split("_"))  # Multi-word bonus
                        elif len(term) >= 5:
                            theme_quality += 1  # Substantial single word bonus
                theme_quality = theme_quality / (k * 10)  # Normalize
                
                # Combined underlying theme score
                underlying_score = (diversity * 0.3 + 
                                  coherence * 0.25 +
                                  balance * 0.2 +
                                  theme_quality * 0.25)
                
                results.append({
                    'k': k,
                    'algorithm': alg_name,
                    'diversity': diversity,
                    'coherence': coherence,
                    'balance': balance,
                    'theme_quality': theme_quality,
                    'underlying_score': underlying_score,
                    'topics': topics,
                    'assignments': assignments,
                    'model': model
                })
                
                print(f"  {alg_name.upper()}: diversity={diversity:.3f}, coherence={coherence:.3f}, "
                      f"theme_quality={theme_quality:.3f}, score={underlying_score:.3f}")
        
        # Find best configuration
        best_config = max(results, key=lambda x: x['underlying_score'])
        
        print(f"\n>>> OPTIMAL UNDERLYING THEMES CONFIGURATION:")
        print(f"    k={best_config['k']}, algorithm={best_config['algorithm'].upper()}")
        print(f"    Underlying themes score: {best_config['underlying_score']:.3f}")
        
        return best_config, pd.DataFrame(results)
        
    def interpret_underlying_themes(self, topics, assignments):
        """Interpret themes based on underlying motivations (not domain terms)"""
        print("\n>>> Interpreting underlying themes and motivations...")
        
        # Theme patterns based on underlying motivations
        underlying_patterns = {
            'helping_altruism': {
                'keywords': ['help', 'helping', 'support', 'care', 'assist', 'serve', 'volunteer'],
                'label': 'Helping & Altruistic Motivation'
            },
            'personal_growth': {
                'keywords': ['experience', 'grow', 'learning', 'develop', 'understand', 'perspective'],
                'label': 'Personal Growth & Development'
            },
            'family_background': {
                'keywords': ['family', 'parents', 'relatives', 'background', 'home', 'upbringing'],
                'label': 'Family Background & Personal Experience'
            },
            'career_practical': {
                'keywords': ['job', 'career', 'work', 'profession', 'field', 'opportunity', 'future'],
                'label': 'Career & Practical Considerations'
            },
            'education_academic': {
                'keywords': ['school', 'university', 'education', 'study', 'learn', 'academic', 'degree'],
                'label': 'Educational & Academic Path'
            },
            'social_impact': {
                'keywords': ['community', 'society', 'people', 'social', 'impact', 'change', 'difference'],
                'label': 'Social Impact & Community Service'
            },
            'emotional_connection': {
                'keywords': ['emotional', 'feelings', 'empathy', 'understanding', 'connection', 'relate'],
                'label': 'Emotional Connection & Empathy'
            },
            'challenge_interest': {
                'keywords': ['interesting', 'challenging', 'complex', 'difficult', 'engaging', 'fascinating'],
                'label': 'Intellectual Challenge & Interest'
            },
            'life_meaning': {
                'keywords': ['meaningful', 'purpose', 'important', 'significant', 'valuable', 'worthwhile'],
                'label': 'Life Meaning & Purpose'
            }
        }
        
        interpretations = []
        theme_analyses = []
        
        for i, topic_terms in enumerate(topics):
            doc_count = sum(assignments == i)
            
            # Separate phrases and single words
            phrases = [t for t in topic_terms[:15] if "_" in t]
            single_words = [t for t in topic_terms[:15] if "_" not in t]
            
            # Score against underlying motivation patterns
            pattern_scores = {}
            for pattern_key, pattern_info in underlying_patterns.items():
                score = 0
                
                # Check phrases
                for phrase in phrases:
                    phrase_words = phrase.replace("_", " ").split()
                    for word in phrase_words:
                        for keyword in pattern_info['keywords']:
                            if keyword in word or word in keyword:
                                score += 2  # Phrase bonus
                
                # Check single words
                for word in single_words:
                    for keyword in pattern_info['keywords']:
                        if keyword in word or word in keyword:
                            score += 1
                
                if score > 0:
                    pattern_scores[pattern_key] = score
            
            # Generate interpretation
            if pattern_scores:
                best_pattern = max(pattern_scores, key=pattern_scores.get)
                interpretation = underlying_patterns[best_pattern]['label']
            else:
                # Fallback based on most prominent terms
                if phrases:
                    main_concept = phrases[0].replace("_", " ").title()
                    interpretation = f"{main_concept} Focus"
                elif single_words:
                    interpretation = f"{single_words[0].title()}-Related Theme"
                else:
                    interpretation = "Mixed Underlying Theme"
            
            interpretations.append(f"{interpretation} ({doc_count} docs)")
            
            theme_analyses.append({
                'topic_id': i + 1,
                'interpretation': interpretation,
                'doc_count': doc_count,
                'key_phrases': phrases[:5],
                'key_words': single_words[:10],
                'pattern_scores': pattern_scores
            })
            
            print(f"\nTopic {i+1}: {interpretation} ({doc_count} docs)")
            if phrases:
                print(f"  Key phrases: {', '.join(phrases[:5])}")
            print(f"  Key words: {', '.join(single_words[:8])}")
            if pattern_scores:
                top_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"  Top motivations: {', '.join([p[0] for p in top_patterns])}")
        
        return interpretations, theme_analyses
        
    def create_domain_filtered_visualization(self, results, theme_analyses):
        """Create visualization showing underlying themes"""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(20, 14))
        
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
        
        k = len(theme_analyses)
        colors = plt.cm.Set2(np.linspace(0, 1, k))
        
        # 1. Topic distribution
        ax1 = fig.add_subplot(gs[0, 0])
        doc_counts = [analysis['doc_count'] for analysis in theme_analyses]
        bars = ax1.bar(range(k), doc_counts, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Underlying Themes\n(Domain Terms Removed)', fontweight='bold')
        ax1.set_xlabel('Theme')
        ax1.set_ylabel('Documents')
        ax1.set_xticks(range(k))
        ax1.set_xticklabels([f'T{i+1}' for i in range(k)])
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Domain filtering impact
        ax2 = fig.add_subplot(gs[0, 1])
        removed_counts = list(self.domain_terms_removed.values())
        if removed_counts:
            top_removed = sorted(self.domain_terms_removed.items(), key=lambda x: x[1], reverse=True)[:8]
            terms, counts = zip(*top_removed)
            bars = ax2.barh(range(len(terms)), counts, color='lightcoral', alpha=0.8)
            ax2.set_title('Removed Domain Terms', fontweight='bold')
            ax2.set_xlabel('Frequency')
            ax2.set_yticks(range(len(terms)))
            ax2.set_yticklabels([t[:12] for t in terms])
        
        # 3. Theme quality metrics
        ax3 = fig.add_subplot(gs[0, 2])
        metrics = ['Diversity', 'Coherence', 'Balance', 'Theme\nQuality']
        values = [results['diversity'], results['coherence'], 
                 results['balance'], results['theme_quality']]
        
        bars = ax3.bar(metrics, values, color=['coral', 'lightblue', 'lightgreen', 'gold'], alpha=0.8)
        ax3.set_title('Underlying Theme Quality', fontweight='bold')
        ax3.set_ylabel('Score')
        ax3.set_ylim(0, 1)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Phrase vs word composition
        ax4 = fig.add_subplot(gs[0, 3])
        phrase_counts = [len(analysis['key_phrases']) for analysis in theme_analyses]
        word_counts = [len(analysis['key_words']) for analysis in theme_analyses]
        
        x = range(k)
        width = 0.35
        ax4.bar([i - width/2 for i in x], phrase_counts, width, label='Phrases', alpha=0.8)
        ax4.bar([i + width/2 for i in x], word_counts, width, label='Words', alpha=0.8)
        ax4.set_title('Phrases vs Words', fontweight='bold')
        ax4.set_xlabel('Theme')
        ax4.set_ylabel('Count')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'T{i+1}' for i in range(k)])
        ax4.legend()
        
        # 5. Underlying themes display (large section)
        ax5 = fig.add_subplot(gs[1:, :])
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
            
            # Key phrases (if any)
            if analysis['key_phrases']:
                phrases_text = f"Key phrases: {', '.join(analysis['key_phrases'])}"
                ax5.text(0.02, y_pos - 0.05, phrases_text, 
                        transform=ax5.transAxes, fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow', 
                                 alpha=0.9, edgecolor=color, linewidth=2))
            
            # Key words
            words_text = f"Key words: {', '.join(analysis['key_words'][:12])}"
            y_offset = 0.10 if analysis['key_phrases'] else 0.05
            ax5.text(0.02, y_pos - y_offset, words_text, 
                    transform=ax5.transAxes, fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                             alpha=0.9, edgecolor=color, linewidth=1))
            
            # Underlying motivations
            if analysis['pattern_scores']:
                top_motivations = sorted(analysis['pattern_scores'].items(), 
                                       key=lambda x: x[1], reverse=True)[:3]
                motivations_text = f"Underlying motivations: {', '.join([m[0].replace('_', ' ') for m in top_motivations])}"
                ax5.text(0.02, y_pos - y_offset - 0.05, motivations_text, 
                        transform=ax5.transAxes, fontsize=10, style='italic', color='gray')
        
        # Title
        title = f'Domain-Filtered Analysis: {k} Underlying Themes'
        subtitle = f'Removed {len(self.domain_terms_removed)} domain terms | Quality Score: {results["underlying_score"]:.3f}'
        fig.suptitle(f'{title}\n{subtitle}', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = os.path.join(RESULTS_DIR, 'domain_filtered_underlying_themes.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Domain-filtered visualization saved: {output_path}")
        
    def save_domain_filtered_results(self, results, theme_analyses, k_analysis_df):
        """Save domain-filtered results"""
        
        # Main report
        with open(os.path.join(RESULTS_DIR, 'domain_filtered_underlying_themes_report.txt'), 'w') as f:
            f.write("DOMAIN-FILTERED UNDERLYING THEMES ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("DOMAIN FILTERING CONFIGURATION:\n")
            f.write(f"• Number of underlying themes: {len(theme_analyses)}\n")
            f.write(f"• Algorithm: {results['algorithm'].upper()}\n")
            f.write(f"• Documents analyzed: {len(results['assignments'])}\n")
            f.write(f"• Underlying themes score: {results['underlying_score']:.3f}\n\n")
            
            f.write("DOMAIN FILTERING IMPACT:\n")
            f.write(f"• Target domain terms removed: {len(self.domain_terms_removed)}\n")
            f.write(f"• Total domain term occurrences removed: {sum(self.domain_terms_removed.values())}\n")
            f.write(f"• Meaningful phrases discovered: {len(self.filtered_phrases)}\n\n")
            
            f.write("QUALITY METRICS:\n")
            f.write(f"• Theme diversity: {results['diversity']:.3f}\n")
            f.write(f"• Topic coherence: {results['coherence']:.3f}\n")
            f.write(f"• Theme balance: {results['balance']:.3f}\n")
            f.write(f"• Theme quality: {results['theme_quality']:.3f}\n\n")
            
            f.write("UNDERLYING THEMES (DOMAIN TERMS REMOVED):\n")
            f.write("-" * 80 + "\n")
            
            for analysis in theme_analyses:
                f.write(f"\n{analysis['interpretation']} ({analysis['doc_count']} docs)\n")
                
                if analysis['key_phrases']:
                    f.write(f"  • Key phrases: {', '.join(analysis['key_phrases'])}\n")
                f.write(f"  • Key words: {', '.join(analysis['key_words'])}\n")
                
                if analysis['pattern_scores']:
                    top_motivations = sorted(analysis['pattern_scores'].items(), 
                                           key=lambda x: x[1], reverse=True)[:3]
                    f.write(f"  • Top motivations: {', '.join([m[0].replace('_', ' ') for m in top_motivations])}\n")
            
            f.write(f"\nREMOVED DOMAIN TERMS:\n")
            f.write("-" * 40 + "\n")
            top_removed = sorted(self.domain_terms_removed.items(), key=lambda x: x[1], reverse=True)
            for term, count in top_removed[:20]:
                f.write(f"  • '{term}': {count} occurrences\n")
        
        # Analysis details
        k_analysis_df.to_csv(os.path.join(RESULTS_DIR, 'domain_filtered_k_analysis.csv'), index=False)
        
        # Document assignments
        assignments_df = pd.DataFrame({
            'document_id': range(len(results['assignments'])),
            'theme_number': results['assignments'] + 1,
            'underlying_theme': [theme_analyses[i]['interpretation'] for i in results['assignments']],
            'text_preview': [doc[:250] + '...' if len(doc) > 250 else doc 
                           for doc in self.raw_docs[:len(results['assignments'])]]
        })
        assignments_df.to_csv(os.path.join(RESULTS_DIR, 'domain_filtered_assignments.csv'), index=False)
        
        print("✓ Domain-filtered results saved:")
        print("  • domain_filtered_underlying_themes_report.txt")
        print("  • domain_filtered_k_analysis.csv") 
        print("  • domain_filtered_assignments.csv")
        
    def run_domain_filtered_analysis(self):
        """Execute domain-filtered analysis to find underlying themes"""
        print("=" * 80)
        print("DOMAIN-FILTERED UNDERLYING THEMES ANALYSIS")
        print("=" * 80)
        
        # Load and filter data
        self.load_data()
        filtered_docs = self.aggressive_domain_filtering()
        docs_with_phrases = self.detect_meaningful_phrases(filtered_docs)
        X, vocab, vectorizer = self.create_domain_filtered_features(docs_with_phrases)
        
        # Find optimal configuration
        best_config, k_analysis_df = self.find_optimal_k_domain_filtered(X, vocab)
        
        # Interpret underlying themes
        interpretations, theme_analyses = self.interpret_underlying_themes(
            best_config['topics'], best_config['assignments'])
        
        # Display results
        print("\n" + "=" * 80)
        print("UNDERLYING THEMES DISCOVERED (DOMAIN TERMS REMOVED)")
        print("=" * 80)
        print(f"Optimal k: {len(theme_analyses)}")
        print(f"Algorithm: {best_config['algorithm'].upper()}")
        print(f"Domain terms removed: {len(self.domain_terms_removed)}")
        print(f"Underlying themes score: {best_config['underlying_score']:.3f}")
        
        # Save and visualize
        self.save_domain_filtered_results(best_config, theme_analyses, k_analysis_df)
        self.create_domain_filtered_visualization(best_config, theme_analyses)
        
        return best_config, theme_analyses


if __name__ == "__main__":
    modeler = DomainFilteredTopicModeler()
    results, theme_analyses = modeler.run_domain_filtered_analysis()
    
    print("\n" + "🎯" * 35)
    print("DOMAIN-FILTERED UNDERLYING THEMES ANALYSIS COMPLETE!")
    print("🎯" * 35)
    print(f"\nKey Achievements:")
    print(f"• Discovered {len(theme_analyses)} underlying themes")
    print(f"• Removed {len(modeler.domain_terms_removed)} different target domain terms")
    print(f"• Total domain term instances removed: {sum(modeler.domain_terms_removed.values())}")
    print(f"• Found {len(modeler.filtered_phrases)} meaningful phrases")
    print("• Reveals WHY students are interested, not just WHAT they're interested in")
    print("\nFocus: Underlying motivations and themes, not obvious target terms!")
    print("All results saved to results/ directory")