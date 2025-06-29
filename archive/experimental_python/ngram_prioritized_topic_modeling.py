#!/usr/bin/env python3
"""
N-gram Prioritized Topic Modeling
Heavily prioritizes meaningful 2-3+ word phrases over single words
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Gensim for advanced phrase detection
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

# Domain filtering (keep counseling/mental health terms - they're meaningful)
DOMAIN_RE = re.compile(
    r"\b(substance|abuse|addict\w*|drug\w*|alcohol\w*)\b",  
    re.I,
)

# Comprehensive stopwords
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


class NgramPrioritizedTopicModeler:
    """Topic modeling that heavily prioritizes meaningful n-grams"""
    
    def __init__(self):
        self.meaningful_phrases = []
        self.tokens_with_phrases = []
        
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
            
            # Keep only substantive utterances (>= 12 words for better n-gram detection)
            df = df[df["Text"].str.split().str.len() >= 12]
            
            texts = df["Text"].astype(str).tolist()
            docs.extend(texts)
            
            print(f"{fn:45} | {original_count:3} → {len(df):3} rich utterances")
        
        print(f"Total rich utterances: {len(docs)}")
        self.raw_docs = docs
        
        # Clean text
        self.clean_docs = [DOMAIN_RE.sub(" ", doc.lower()) for doc in docs]
        
    def advanced_ngram_detection(self):
        """Advanced n-gram detection with multiple passes"""
        print("\n>>> Advanced n-gram detection (prioritizing phrases)...")
        
        # Step 1: Clean and expand contractions properly
        cleaned_docs = []
        for doc in self.clean_docs:
            # Proper contraction expansion
            contractions = {
                "don't": "do not", "won't": "will not", "can't": "cannot",
                "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
                "'d": " would", "'m": " am", "let's": "let us"
            }
            
            for contraction, expansion in contractions.items():
                doc = doc.replace(contraction, expansion)
            
            # Remove punctuation, normalize spaces
            doc = re.sub(r"[^\w\s]", " ", doc)
            doc = re.sub(r"\s+", " ", doc).strip()
            cleaned_docs.append(doc)
        
        # Step 2: Tokenize for phrase detection
        sentences = []
        for doc in cleaned_docs:
            tokens = simple_preprocess(doc, deacc=True, min_len=3, max_len=25)
            # Filter meaningful tokens
            meaningful_tokens = [t for t in tokens if t not in STOP_WORDS and len(t) >= 3]
            if len(meaningful_tokens) >= 5:
                sentences.append(meaningful_tokens)
        
        print(f"Prepared {len(sentences)} documents for phrase detection")
        
        # Step 3: Multi-level phrase detection
        print("   Detecting bigrams...")
        bigram_model = Phrases(sentences, min_count=2, threshold=0.3, delimiter="_", scoring='npmi')
        bigram_phraser = Phraser(bigram_model)
        bigram_sentences = [bigram_phraser[sent] for sent in sentences]
        
        print("   Detecting trigrams...")
        trigram_model = Phrases(bigram_sentences, min_count=2, threshold=0.2, delimiter="_", scoring='npmi')
        trigram_phraser = Phraser(trigram_model)
        final_sentences = [trigram_phraser[sent] for sent in bigram_sentences]
        
        print("   Detecting 4-grams...")
        fourgram_model = Phrases(final_sentences, min_count=2, threshold=0.1, delimiter="_", scoring='npmi')
        fourgram_phraser = Phraser(fourgram_model)
        final_sentences = [fourgram_phraser[sent] for sent in final_sentences]
        
        # Step 4: Extract and analyze detected phrases
        all_phrases = set()
        for sent in final_sentences:
            for token in sent:
                if "_" in token:
                    all_phrases.add(token)
        
        # Categorize phrases by length and meaningfulness
        bigrams = [p for p in all_phrases if len(p.split("_")) == 2]
        trigrams = [p for p in all_phrases if len(p.split("_")) == 3]
        fourgrams = [p for p in all_phrases if len(p.split("_")) >= 4]
        
        print(f"   Detected {len(bigrams)} bigrams, {len(trigrams)} trigrams, {len(fourgrams)} 4+ grams")
        
        # Show most meaningful phrases
        print("   Most meaningful bigrams:", bigrams[:15])
        print("   Most meaningful trigrams:", trigrams[:10])
        print("   Most meaningful 4+ grams:", fourgrams[:5])
        
        # Store for later use
        self.meaningful_phrases = list(all_phrases)
        self.tokens_with_phrases = final_sentences
        
        # Convert back to documents
        docs_with_ngrams = [" ".join(sent) for sent in final_sentences]
        
        return docs_with_ngrams
        
    def create_ngram_weighted_features(self, docs):
        """Create feature matrix that heavily weights n-grams"""
        print("\n>>> Creating n-gram weighted feature matrix...")
        
        # Custom analyzer that weights n-grams higher
        def ngram_weighted_analyzer(text):
            tokens = text.split()
            features = []
            
            # Add all tokens but weight n-grams much higher
            for token in tokens:
                if "_" in token:
                    # N-gram detected - add multiple times for higher weight
                    n_words = len(token.split("_"))
                    if n_words >= 4:
                        weight = 5  # 4+ grams get 5x weight
                    elif n_words == 3:
                        weight = 3  # trigrams get 3x weight
                    else:
                        weight = 2  # bigrams get 2x weight
                    
                    features.extend([token] * weight)
                else:
                    # Single word - add once (if meaningful)
                    if (len(token) >= 4 and 
                        token not in STOP_WORDS and
                        not token.isdigit()):
                        features.append(token)
            
            return features
        
        # Use TF-IDF with custom analyzer
        vectorizer = TfidfVectorizer(
            analyzer=ngram_weighted_analyzer,
            min_df=3,  # Must appear in at least 3 documents
            max_df=0.7,  # Can't appear in more than 70% of docs
            max_features=300,  # Allow more features for n-grams
            sublinear_tf=True,
            norm='l2'
        )
        
        X = vectorizer.fit_transform(docs)
        vocab = vectorizer.get_feature_names_out()
        
        # Analyze vocabulary composition
        ngram_count = sum(1 for term in vocab if "_" in term)
        single_word_count = len(vocab) - ngram_count
        
        print(f"Feature matrix: {X.shape[0]} docs × {X.shape[1]} features")
        print(f"  - {ngram_count} n-grams ({ngram_count/len(vocab):.1%})")
        print(f"  - {single_word_count} single words ({single_word_count/len(vocab):.1%})")
        
        # Show sample n-grams
        ngrams_in_vocab = [term for term in vocab if "_" in term]
        print(f"Sample n-grams in vocabulary: {ngrams_in_vocab[:20]}")
        
        return X, vocab, vectorizer
        
    def find_optimal_k_ngram_focused(self, X, vocab, k_range=range(2, 6)):
        """Find optimal k focusing on n-gram interpretability"""
        print("\n>>> Finding optimal k with n-gram focus...")
        
        results = []
        
        for k in k_range:
            print(f"Testing k={k}...")
            
            # Test both algorithms
            for alg_name in ['lda', 'nmf']:
                if alg_name == 'lda':
                    model = LatentDirichletAllocation(
                        n_components=k, max_iter=200, random_state=42,
                        doc_topic_prior=0.05,  # More focused topics
                        topic_word_prior=0.01,  # More focused words per topic
                        learning_method='batch')
                else:
                    model = NMF(
                        n_components=k, init='nndsvd', max_iter=1000, random_state=42,
                        alpha_W=0.05, alpha_H=0.05, l1_ratio=0.2)  # Less regularization
                
                # Fit model
                doc_topic = model.fit_transform(X)
                assignments = doc_topic.argmax(axis=1)
                
                # Get topics
                topics = []
                for topic_idx, topic in enumerate(model.components_):
                    top_indices = topic.argsort()[-20:][::-1]
                    top_terms = [vocab[i] for i in top_indices]
                    topics.append(top_terms)
                
                # Calculate n-gram focused metrics
                
                # 1. N-gram ratio (higher is better)
                all_terms = []
                ngram_count = 0
                for topic in topics:
                    for term in topic[:15]:
                        all_terms.append(term)
                        if "_" in term:
                            ngram_count += 1
                
                ngram_ratio = ngram_count / len(all_terms) if all_terms else 0
                
                # 2. N-gram meaningfulness (longer phrases score higher)
                ngram_score = 0
                for topic in topics:
                    for term in topic[:10]:
                        if "_" in term:
                            n_words = len(term.split("_"))
                            ngram_score += n_words  # More words = higher score
                
                ngram_score = ngram_score / (k * 10)  # Normalize by max possible
                
                # 3. Topic diversity
                unique_terms = len(set(all_terms))
                diversity = unique_terms / len(all_terms) if all_terms else 0
                
                # 4. Coherence
                cm_cv = CoherenceModel(topics=topics, texts=self.tokens_with_phrases, 
                                     dictionary=Dictionary(self.tokens_with_phrases), coherence='c_v')
                coherence = cm_cv.get_coherence()
                
                # 5. Balance
                topic_counts = [np.sum(assignments == i) for i in range(k)]
                balance = 1 - np.std(topic_counts) / np.mean(topic_counts) if np.mean(topic_counts) > 0 else 0
                
                # Combined n-gram focused score
                ngram_focused_score = (ngram_ratio * 0.3 +  # Prioritize n-grams
                                     ngram_score * 0.25 +   # Longer phrases
                                     diversity * 0.2 + 
                                     coherence * 0.15 +
                                     balance * 0.1)
                
                results.append({
                    'k': k,
                    'algorithm': alg_name,
                    'ngram_ratio': ngram_ratio,
                    'ngram_score': ngram_score,
                    'diversity': diversity,
                    'coherence': coherence,
                    'balance': balance,
                    'ngram_focused_score': ngram_focused_score,
                    'topics': topics,
                    'assignments': assignments,
                    'model': model
                })
                
                print(f"  {alg_name.upper()}: ngrams={ngram_ratio:.1%}, diversity={diversity:.3f}, "
                      f"score={ngram_focused_score:.3f}")
        
        # Find best configuration
        best_config = max(results, key=lambda x: x['ngram_focused_score'])
        
        print(f"\n>>> OPTIMAL N-GRAM FOCUSED CONFIGURATION:")
        print(f"    k={best_config['k']}, algorithm={best_config['algorithm'].upper()}")
        print(f"    N-gram ratio: {best_config['ngram_ratio']:.1%}")
        print(f"    N-gram score: {best_config['ngram_score']:.3f}")
        print(f"    Combined score: {best_config['ngram_focused_score']:.3f}")
        
        return best_config, pd.DataFrame(results)
        
    def analyze_ngram_topics(self, topics, assignments):
        """Analyze topics with focus on n-gram content"""
        print("\n>>> Analyzing n-gram rich topics...")
        
        topic_analyses = []
        
        for i, topic_terms in enumerate(topics):
            doc_count = sum(assignments == i)
            
            # Separate n-grams by length
            bigrams = [t for t in topic_terms if "_" in t and len(t.split("_")) == 2]
            trigrams = [t for t in topic_terms if "_" in t and len(t.split("_")) == 3]
            fourgrams = [t for t in topic_terms if "_" in t and len(t.split("_")) >= 4]
            single_words = [t for t in topic_terms if "_" not in t]
            
            # Calculate n-gram richness
            total_ngrams = len(bigrams) + len(trigrams) + len(fourgrams)
            ngram_richness = total_ngrams / len(topic_terms[:15]) if topic_terms else 0
            
            # Generate interpretation based on n-grams first, then single words
            interpretation = self.interpret_ngram_topic(bigrams, trigrams, fourgrams, single_words)
            
            analysis = {
                'topic_id': i + 1,
                'doc_count': doc_count,
                'interpretation': interpretation,
                'bigrams': bigrams[:8],
                'trigrams': trigrams[:5],
                'fourgrams': fourgrams[:3],
                'key_single_words': single_words[:10],
                'ngram_richness': ngram_richness,
                'total_ngrams': total_ngrams
            }
            
            topic_analyses.append(analysis)
            
            print(f"\nTopic {i+1}: {interpretation} ({doc_count} docs)")
            print(f"  N-gram richness: {ngram_richness:.1%} ({total_ngrams} n-grams)")
            if bigrams:
                print(f"  Key bigrams: {', '.join(bigrams[:5])}")
            if trigrams:
                print(f"  Key trigrams: {', '.join(trigrams[:3])}")
            if fourgrams:
                print(f"  Key 4+ grams: {', '.join(fourgrams[:2])}")
            print(f"  Supporting words: {', '.join(single_words[:8])}")
        
        return topic_analyses
        
    def interpret_ngram_topic(self, bigrams, trigrams, fourgrams, single_words):
        """Generate interpretation prioritizing n-grams"""
        
        # Define meaningful n-gram patterns
        ngram_patterns = {
            'mental_health': ['mental health', 'mental_health', 'psychological', 'emotional'],
            'helping_orientation': ['helping people', 'helping_people', 'help others', 'make difference'],
            'career_path': ['career path', 'career_path', 'professional development', 'job opportunities'],
            'family_experience': ['family experience', 'family_experience', 'personal experience', 'lived experience'],
            'education_training': ['education training', 'school program', 'learning experience', 'academic'],
            'clinical_practice': ['clinical practice', 'therapy sessions', 'therapeutic', 'counseling'],
            'practical_concerns': ['job market', 'financial', 'work environment', 'practical']
        }
        
        # Score based on n-grams first
        pattern_scores = {}
        all_ngrams = bigrams + trigrams + fourgrams
        
        for pattern_name, pattern_terms in ngram_patterns.items():
            score = 0
            for ngram in all_ngrams:
                ngram_clean = ngram.replace("_", " ")
                for pattern_term in pattern_terms:
                    if pattern_term in ngram_clean or ngram_clean in pattern_term:
                        # Weight by n-gram length
                        weight = len(ngram.split("_"))
                        score += weight
            
            # Also check single words but with lower weight
            for word in single_words[:5]:
                for pattern_term in pattern_terms:
                    if word in pattern_term:
                        score += 0.5
                        
            if score > 0:
                pattern_scores[pattern_name] = score
        
        # Generate interpretation
        if pattern_scores:
            best_pattern = max(pattern_scores, key=pattern_scores.get)
            interpretations = {
                'mental_health': 'Mental Health & Psychological Focus',
                'helping_orientation': 'Helping & Service Motivation',
                'career_path': 'Career Development & Professional Path',
                'family_experience': 'Personal & Family Experience',
                'education_training': 'Education & Training Focus',
                'clinical_practice': 'Clinical Practice & Therapy',
                'practical_concerns': 'Practical & Economic Considerations'
            }
            return interpretations.get(best_pattern, f"{best_pattern.title()} Theme")
        
        # Fallback to most meaningful n-gram
        if all_ngrams:
            main_ngram = all_ngrams[0].replace("_", " ").title()
            return f"{main_ngram} Theme"
        
        # Last resort - single words
        if single_words:
            return f"{single_words[0].title()}-Related Theme"
        
        return "Mixed Discussion Theme"
        
    def create_ngram_visualization(self, results, topic_analyses):
        """Create visualization emphasizing n-gram content"""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(20, 16))
        
        gs = fig.add_gridspec(4, 3, hspace=0.5, wspace=0.3)
        
        k = len(topic_analyses)
        colors = plt.cm.Set3(np.linspace(0, 1, k))
        
        # 1. Topic distribution
        ax1 = fig.add_subplot(gs[0, 0])
        doc_counts = [analysis['doc_count'] for analysis in topic_analyses]
        bars = ax1.bar(range(k), doc_counts, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Document Distribution\n(N-gram Focused Topics)', fontweight='bold')
        ax1.set_xlabel('Topic')
        ax1.set_ylabel('Documents')
        ax1.set_xticks(range(k))
        ax1.set_xticklabels([f'T{i+1}' for i in range(k)])
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 2. N-gram richness
        ax2 = fig.add_subplot(gs[0, 1])
        richness = [analysis['ngram_richness'] for analysis in topic_analyses]
        bars = ax2.bar(range(k), richness, color=colors, alpha=0.8)
        ax2.set_title('N-gram Richness per Topic', fontweight='bold')
        ax2.set_xlabel('Topic')
        ax2.set_ylabel('N-gram Ratio')
        ax2.set_xticks(range(k))
        ax2.set_xticklabels([f'T{i+1}' for i in range(k)])
        ax2.set_ylim(0, 1)
        
        for bar, val in zip(bars, richness):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 3. N-gram type distribution
        ax3 = fig.add_subplot(gs[0, 2])
        bigram_counts = [len(analysis['bigrams']) for analysis in topic_analyses]
        trigram_counts = [len(analysis['trigrams']) for analysis in topic_analyses]
        fourgram_counts = [len(analysis['fourgrams']) for analysis in topic_analyses]
        
        x = range(k)
        width = 0.25
        ax3.bar([i - width for i in x], bigram_counts, width, label='Bigrams', alpha=0.8)
        ax3.bar(x, trigram_counts, width, label='Trigrams', alpha=0.8)
        ax3.bar([i + width for i in x], fourgram_counts, width, label='4+ grams', alpha=0.8)
        ax3.set_title('N-gram Types per Topic', fontweight='bold')
        ax3.set_xlabel('Topic')
        ax3.set_ylabel('Count')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'T{i+1}' for i in range(k)])
        ax3.legend()
        
        # 4. N-gram showcase (large section)
        ax4 = fig.add_subplot(gs[1:, :])
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        y_positions = np.linspace(0.9, 0.1, k)
        
        for i, analysis in enumerate(topic_analyses):
            y_pos = y_positions[i]
            color = colors[i]
            
            # Topic header
            ax4.text(0.02, y_pos, f"Topic {analysis['topic_id']}: {analysis['interpretation']}", 
                    transform=ax4.transAxes, fontsize=16, fontweight='bold', color=color)
            
            # N-gram showcase
            y_offset = 0.05
            
            if analysis['fourgrams']:
                text = f"4+ word phrases: {', '.join(analysis['fourgrams'])}"
                ax4.text(0.02, y_pos - y_offset, text, 
                        transform=ax4.transAxes, fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='gold', 
                                 alpha=0.9, edgecolor=color, linewidth=2))
                y_offset += 0.05
            
            if analysis['trigrams']:
                text = f"3-word phrases: {', '.join(analysis['trigrams'])}"
                ax4.text(0.02, y_pos - y_offset, text, 
                        transform=ax4.transAxes, fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgreen', 
                                 alpha=0.9, edgecolor=color, linewidth=2))
                y_offset += 0.05
            
            if analysis['bigrams']:
                text = f"2-word phrases: {', '.join(analysis['bigrams'])}"
                ax4.text(0.02, y_pos - y_offset, text, 
                        transform=ax4.transAxes, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', 
                                 alpha=0.9, edgecolor=color, linewidth=1))
                y_offset += 0.05
            
            if analysis['key_single_words']:
                text = f"Supporting words: {', '.join(analysis['key_single_words'])}"
                ax4.text(0.02, y_pos - y_offset, text, 
                        transform=ax4.transAxes, fontsize=11,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                 alpha=0.9, edgecolor='gray', linewidth=1))
                y_offset += 0.05
            
            # Stats
            stats = f"Documents: {analysis['doc_count']} | N-gram richness: {analysis['ngram_richness']:.1%} | Total n-grams: {analysis['total_ngrams']}"
            ax4.text(0.02, y_pos - y_offset, stats, 
                    transform=ax4.transAxes, fontsize=10, style='italic', color='gray')
        
        # Title
        title = f'N-gram Prioritized Topic Analysis: {k} Phrase-Rich Themes'
        subtitle = f'N-gram Ratio: {results["ngram_ratio"]:.1%} | N-gram Score: {results["ngram_score"]:.3f} | Overall Score: {results["ngram_focused_score"]:.3f}'
        fig.suptitle(f'{title}\n{subtitle}', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = os.path.join(RESULTS_DIR, 'ngram_prioritized_topics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ N-gram visualization saved: {output_path}")
        
    def save_ngram_results(self, results, topic_analyses, k_analysis_df):
        """Save n-gram focused results"""
        
        # Main report
        with open(os.path.join(RESULTS_DIR, 'ngram_prioritized_report.txt'), 'w') as f:
            f.write("N-GRAM PRIORITIZED TOPIC ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("N-GRAM FOCUSED CONFIGURATION:\n")
            f.write(f"• Number of topics: {len(topic_analyses)}\n")
            f.write(f"• Algorithm: {results['algorithm'].upper()}\n")
            f.write(f"• Documents analyzed: {len(results['assignments'])}\n")
            f.write(f"• N-gram focused score: {results['ngram_focused_score']:.3f}\n\n")
            
            f.write("N-GRAM METRICS:\n")
            f.write(f"• N-gram ratio: {results['ngram_ratio']:.1%}\n")
            f.write(f"• N-gram quality score: {results['ngram_score']:.3f}\n")
            f.write(f"• Topic diversity: {results['diversity']:.3f}\n")
            f.write(f"• Topic coherence: {results['coherence']:.3f}\n")
            f.write(f"• Total meaningful phrases detected: {len(self.meaningful_phrases)}\n\n")
            
            f.write("PHRASE-RICH THEMES:\n")
            f.write("-" * 80 + "\n")
            
            for analysis in topic_analyses:
                f.write(f"\n{analysis['interpretation']} ({analysis['doc_count']} docs)\n")
                f.write(f"  • N-gram richness: {analysis['ngram_richness']:.1%}\n")
                
                if analysis['fourgrams']:
                    f.write(f"  • 4+ word phrases: {', '.join(analysis['fourgrams'])}\n")
                if analysis['trigrams']:
                    f.write(f"  • 3-word phrases: {', '.join(analysis['trigrams'])}\n")
                if analysis['bigrams']:
                    f.write(f"  • 2-word phrases: {', '.join(analysis['bigrams'])}\n")
                if analysis['key_single_words']:
                    f.write(f"  • Supporting words: {', '.join(analysis['key_single_words'])}\n")
        
        # Analysis details
        k_analysis_df.to_csv(os.path.join(RESULTS_DIR, 'ngram_k_analysis.csv'), index=False)
        
        # Document assignments
        assignments_df = pd.DataFrame({
            'document_id': range(len(results['assignments'])),
            'topic_number': results['assignments'] + 1,
            'topic_interpretation': [topic_analyses[i]['interpretation'] for i in results['assignments']],
            'ngram_richness': [topic_analyses[i]['ngram_richness'] for i in results['assignments']],
            'text_preview': [doc[:250] + '...' if len(doc) > 250 else doc 
                           for doc in self.raw_docs[:len(results['assignments'])]]
        })
        assignments_df.to_csv(os.path.join(RESULTS_DIR, 'ngram_topic_assignments.csv'), index=False)
        
        print("✓ N-gram results saved:")
        print("  • ngram_prioritized_report.txt")
        print("  • ngram_k_analysis.csv") 
        print("  • ngram_topic_assignments.csv")
        
    def run_ngram_analysis(self):
        """Execute n-gram prioritized analysis"""
        print("=" * 80)
        print("N-GRAM PRIORITIZED TOPIC MODELING")
        print("=" * 80)
        
        # Load and process data
        self.load_data()
        docs_with_ngrams = self.advanced_ngram_detection()
        X, vocab, vectorizer = self.create_ngram_weighted_features(docs_with_ngrams)
        
        # Find optimal configuration
        best_config, k_analysis_df = self.find_optimal_k_ngram_focused(X, vocab)
        
        # Analyze topics with n-gram focus
        topic_analyses = self.analyze_ngram_topics(best_config['topics'], best_config['assignments'])
        
        # Display results
        print("\n" + "=" * 80)
        print("N-GRAM PRIORITIZED TOPICS IDENTIFIED")
        print("=" * 80)
        print(f"Optimal k: {len(topic_analyses)}")
        print(f"Algorithm: {best_config['algorithm'].upper()}")
        print(f"N-gram ratio: {best_config['ngram_ratio']:.1%}")
        print(f"N-gram score: {best_config['ngram_score']:.3f}")
        print(f"Total meaningful phrases: {len(self.meaningful_phrases)}")
        
        # Save and visualize
        self.save_ngram_results(best_config, topic_analyses, k_analysis_df)
        self.create_ngram_visualization(best_config, topic_analyses)
        
        return best_config, topic_analyses


if __name__ == "__main__":
    modeler = NgramPrioritizedTopicModeler()
    results, topic_analyses = modeler.run_ngram_analysis()
    
    print("\n" + "🔤" * 35)
    print("N-GRAM PRIORITIZED ANALYSIS COMPLETE!")
    print("🔤" * 35)
    print(f"\nKey N-gram Achievements:")
    print(f"• {len(topic_analyses)} topics prioritizing meaningful phrases")
    print(f"• {results['ngram_ratio']:.1%} of top terms are multi-word phrases")
    print(f"• Detected phrases: 2-word, 3-word, and 4+ word combinations")
    print(f"• {len(modeler.meaningful_phrases)} total meaningful phrases found")
    print("• Weighted n-grams 2-5x higher than single words")
    print("\nFocus: Multi-word concepts over single terms!")
    print("All results saved to results/ directory")