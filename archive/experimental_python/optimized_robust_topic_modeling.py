#!/usr/bin/env python3
"""
Optimized Robust Topic Modeling for SUD Focus Groups
Improved parameters for better topic separation and validation
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Gensim imports
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF, LatentDirichletAllocation
from scipy.optimize import linear_sum_assignment

# Suppress warnings
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

# Domain-specific term filtering (more selective)
DOMAIN_RE = re.compile(
    r"\b(substance|abuse|addict\w*|drug\w*|alcohol\w*)\b",  # Keep counseling terms
    flags=re.I,
)

# More focused stopwords (less aggressive)
EXTRA_STOP = set("""
gonna wanna yeah um uh okay dont don't thats that's heres here's theres there's
would could should can just also even let's lets well much many though
like really actually maybe pretty basically bit lot kind_of sort_of
think feel know want going get got make made see say said look looking
yes no people person someone something anything nothing
probably definitely obviously clearly certainly guess suppose might
exactly totally completely absolutely somewhat somehow
""".split())


class OptimizedTopicModeler:
    """Optimized topic modeling with better separation"""
    
    def __init__(self, n_topics=5):
        self.n_topics = n_topics
        
    def load_data(self):
        """Load and clean CSV files"""
        docs = []
        
        print("\n>>> Loading focus group data...")
        for fn in CSV_FILES:
            fp = os.path.join(DATA_DIR, fn)
            df = pd.read_csv(fp)
            raw = len(df)
            
            # Filter out moderator rows and empty text
            df = df[df["Text"].notna()]
            df = df[~df["Speaker"].astype(str).str.fullmatch(r"[A-Z]{2,3}")]
            
            # Keep only substantive utterances (>= 5 words)
            df = df[df["Text"].str.split().str.len() >= 5]
            
            texts = df["Text"].astype(str).tolist()
            docs.extend(texts)
            
            print(f"{fn:45} | {raw:3} → {len(df):3} utterances")
        
        print(f"Total substantive utterances: {len(docs)}")
        self.raw_docs = docs
        
        # Conservative domain filtering (keep more content)
        self.docs_clean = [DOMAIN_RE.sub(" ", d.lower()) for d in docs]
        
    def advanced_phrase_detection(self):
        """Improved phrase detection"""
        print(">>> Advanced phrase detection...")
        
        # Tokenize with better preprocessing
        sentences = []
        for doc in self.docs_clean:
            # Remove punctuation but keep apostrophes
            clean_doc = re.sub(r"[^\w\s']", " ", doc)
            tokens = simple_preprocess(clean_doc, deacc=True, min_len=2, max_len=20)
            if len(tokens) >= 3:  # Keep only docs with 3+ meaningful tokens
                sentences.append(tokens)
        
        print(f"Processed {len(sentences)} documents for phrase detection")
        
        # More sensitive phrase detection
        bigram = Phrases(sentences, min_count=2, threshold=8, delimiter="_")
        trigram = Phrases(bigram[sentences], threshold=7, delimiter="_")
        phraser = Phraser(trigram)
        
        # Apply phrases
        docs_phr = [" ".join(phraser[s]) for s in sentences]
        self.tokens = [phraser[s] for s in sentences]
        
        # Show detected phrases
        phrases = [phrase for phrase in phraser.phrasegrams.keys() if "_" in phrase]
        print(f"Detected {len(phrases)} key phrases: {phrases[:20]}")
        
        return docs_phr
        
    def create_optimized_features(self, docs):
        """Create better feature matrix for topic separation"""
        stop_full = list(ENGLISH_STOP_WORDS.union(EXTRA_STOP))
        
        # More permissive parameters for better topic diversity
        vectorizer = TfidfVectorizer(
            min_df=2,  # More permissive - words in 2+ docs
            max_df=0.90,  # Less aggressive filtering
            stop_words=stop_full,
            token_pattern=r"(?u)\b\w[\w_]+\b",
            sublinear_tf=True,
            max_features=300,  # More features
            ngram_range=(1, 1),  # Focus on unigrams + our detected phrases
            lowercase=True,
            strip_accents='unicode'
        )
            
        X = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()
        
        print(f"Feature matrix: {X.shape[0]} docs × {X.shape[1]} features")
        print(f"Matrix density: {X.nnz / (X.shape[0] * X.shape[1]):.3f}")
        
        return X, feature_names, vectorizer
        
    def fit_optimized_nmf(self, X, n_topics, random_state=42):
        """NMF with better parameters for topic separation"""
        model = NMF(
            n_components=n_topics,
            init='nndsvd',
            max_iter=1000,  # More iterations
            random_state=random_state,
            alpha_W=0.01,   # Less regularization for more diversity
            alpha_H=0.01,
            l1_ratio=0.1,   # Less L1 penalty
            beta_loss='frobenius'
        )
        W = model.fit_transform(X)
        H = model.components_
        return model, W, H
        
    def fit_optimized_lda(self, X, n_topics, random_state=42):
        """LDA with better parameters"""
        model = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=50,
            learning_method='online',
            learning_offset=50.,
            learning_decay=0.7,
            doc_topic_prior=0.1,  # More focused topics per document
            topic_word_prior=0.01,  # More focused words per topic
            random_state=random_state,
            n_jobs=-1
        )
        W = model.fit_transform(X)
        H = model.components_
        return model, W, H
        
    def get_top_terms(self, components, feature_names, n_terms=15):
        """Extract top terms for each topic"""
        topics = []
        for topic_idx, topic in enumerate(components):
            top_indices = topic.argsort()[-n_terms:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            topics.append(top_terms)
        return topics
        
    def calculate_stability(self, X, feature_names, algorithm='nmf', n_seeds=15):
        """Test stability across random seeds"""
        print(f">>> Stability test across {n_seeds} seeds for {algorithm.upper()}...")
        
        reference_topics = None
        all_jaccard_scores = []
        
        for seed in range(n_seeds):
            if algorithm == 'nmf':
                _, _, H = self.fit_optimized_nmf(X, self.n_topics, seed)
            else:
                _, _, H = self.fit_optimized_lda(X, self.n_topics, seed)
                
            topics = self.get_top_terms(H, feature_names)
            
            if seed == 0:
                reference_topics = topics
            else:
                # Calculate Jaccard similarities
                jaccard_matrix = np.zeros((self.n_topics, self.n_topics))
                for i, ref_topic in enumerate(reference_topics):
                    ref_set = set(ref_topic[:10])
                    for j, test_topic in enumerate(topics):
                        test_set = set(test_topic[:10])
                        jaccard = len(ref_set & test_set) / len(ref_set | test_set) if len(ref_set | test_set) > 0 else 0
                        jaccard_matrix[i, j] = jaccard
                
                # Optimal matching
                row_ind, col_ind = linear_sum_assignment(-jaccard_matrix)
                matched_scores = [jaccard_matrix[i, j] for i, j in zip(row_ind, col_ind)]
                all_jaccard_scores.append(matched_scores)
                
        if all_jaccard_scores:
            jaccard_array = np.array(all_jaccard_scores)
            mean_stability = jaccard_array.mean(axis=0)
            overall_stability = mean_stability.mean()
        else:
            mean_stability = np.ones(self.n_topics)
            overall_stability = 1.0
        
        return overall_stability, mean_stability
        
    def calculate_coherence(self, topics):
        """Calculate topic coherence"""
        if not hasattr(self, 'tokens'):
            self.tokens = [simple_preprocess(d) for d in self.docs_clean]
            
        dictionary = Dictionary(self.tokens)
        
        # Both coherence measures
        cm_umass = CoherenceModel(
            topics=topics, texts=self.tokens, dictionary=dictionary, coherence='u_mass')
        cm_cv = CoherenceModel(
            topics=topics, texts=self.tokens, dictionary=dictionary, coherence='c_v')
        
        return cm_umass.get_coherence(), cm_cv.get_coherence()
        
    def calculate_topic_diversity(self, topics):
        """Topic diversity score"""
        all_terms = []
        for topic in topics:
            all_terms.extend(topic[:10])
            
        unique_terms = len(set(all_terms))
        total_terms = len(all_terms)
        return unique_terms / total_terms if total_terms > 0 else 0
        
    def detailed_topic_analysis(self, topics, doc_assignments, feature_names):
        """Detailed analysis of topic quality"""
        print("\n>>> Detailed Topic Analysis")
        
        # Topic interpretations
        theme_mapping = {
            'helping': ['help', 'support', 'care', 'assist', 'service', 'give_back'],
            'personal': ['family', 'personal', 'experience', 'life', 'grew_up', 'background'],
            'education': ['school', 'class', 'course', 'learning', 'education', 'training'],
            'career': ['career', 'job', 'profession', 'work', 'field', 'professional'],
            'clinical': ['clinical', 'therapy', 'client', 'session', 'practice', 'hours'],
            'emotional': ['stress', 'burnout', 'emotional', 'challenging', 'difficult', 'cope'],
            'practical': ['requirements', 'certification', 'license', 'supervised', 'credential']
        }
        
        interpretations = []
        for i, topic_terms in enumerate(topics):
            doc_count = sum(doc_assignments == i)
            
            # Score against themes
            theme_scores = {}
            for theme, keywords in theme_mapping.items():
                score = sum(1 for term in topic_terms[:10] for keyword in keywords if keyword in term)
                if score > 0:
                    theme_scores[theme] = score
            
            # Best theme
            if theme_scores:
                best_theme = max(theme_scores, key=theme_scores.get)
                interpretation = f"{best_theme.title()} Theme"
            else:
                interpretation = "General Discussion"
                
            interpretations.append(f"{interpretation} ({doc_count} docs)")
            
            print(f"\nTopic {i+1}: {interpretation}")
            print(f"  Documents: {doc_count}")
            print(f"  Top terms: {', '.join(topic_terms[:12])}")
            
        return interpretations
        
    def visualize_comprehensive_results(self, results):
        """Create comprehensive visualization"""
        plt.style.use('default')
        fig = plt.figure(figsize=(18, 14))
        
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
        
        # Colors
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_topics))
        
        # 1. Topic sizes
        ax1 = fig.add_subplot(gs[0, 0])
        topic_counts = results['topic_counts']
        bars = ax1.bar(range(len(topic_counts)), topic_counts, color=colors)
        ax1.set_title('Documents per Topic')
        ax1.set_xlabel('Topic')
        ax1.set_ylabel('Count')
        ax1.set_xticks(range(len(topic_counts)))
        ax1.set_xticklabels([f'T{i+1}' for i in range(len(topic_counts))])
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', 
                    ha='center', va='bottom')
        
        # 2. Stability scores
        ax2 = fig.add_subplot(gs[0, 1])
        stability = results['topic_stability']
        bars = ax2.bar(range(len(stability)), stability, color=colors)
        ax2.axhline(y=0.7, color='red', linestyle='--', label='Good threshold')
        ax2.set_title('Topic Stability')
        ax2.set_xlabel('Topic')
        ax2.set_ylabel('Jaccard Score')
        ax2.set_xticks(range(len(stability)))
        ax2.set_xticklabels([f'T{i+1}' for i in range(len(stability))])
        ax2.legend()
        ax2.set_ylim(0, 1.05)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', 
                    ha='center', va='bottom')
        
        # 3. Algorithm comparison
        ax3 = fig.add_subplot(gs[0, 2])
        if 'comparison' in results:
            comp = results['comparison']
            metrics = ['Stability', 'Coherence', 'Diversity']
            
            lda_vals = [comp['lda']['stability'], comp['lda']['coherence_cv'], comp['lda']['diversity']]
            nmf_vals = [comp['nmf']['stability'], comp['nmf']['coherence_cv'], comp['nmf']['diversity']]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax3.bar(x - width/2, lda_vals, width, label='LDA', alpha=0.8)
            ax3.bar(x + width/2, nmf_vals, width, label='NMF', alpha=0.8)
            ax3.set_title('Algorithm Comparison')
            ax3.set_xticks(x)
            ax3.set_xticklabels(metrics)
            ax3.legend()
        
        # 4. Quality metrics
        ax4 = fig.add_subplot(gs[0, 3])
        metrics_names = ['Stability', 'Coherence', 'Diversity']
        metrics_vals = [results['overall_stability'], results['coherence_cv'], results['diversity']]
        
        bars = ax4.bar(metrics_names, metrics_vals, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax4.set_title('Overall Quality Metrics')
        ax4.set_ylabel('Score')
        for bar, val in zip(bars, metrics_vals):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height, f'{val:.3f}', 
                    ha='center', va='bottom')
        
        # 5. Topic words (large text display)
        ax5 = fig.add_subplot(gs[1:, :])
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        
        # Display topics with interpretations
        y_start = 0.95
        y_step = 0.18
        
        for i, (topic, interp) in enumerate(zip(results['topics'], results['interpretations'])):
            y_pos = y_start - i * y_step
            color = colors[i]
            
            # Topic header
            ax5.text(0.02, y_pos, f"Topic {i+1}: {interp}", 
                    transform=ax5.transAxes, fontsize=14, weight='bold', color=color)
            
            # Top terms with weights (if available)
            terms_text = f"Top terms: {', '.join(topic[:15])}"
            ax5.text(0.02, y_pos - 0.04, terms_text, 
                    transform=ax5.transAxes, fontsize=11, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Stability info
            stability_text = f"Stability: {results['topic_stability'][i]:.3f} | Docs: {results['topic_counts'][i]}"
            ax5.text(0.02, y_pos - 0.08, stability_text, 
                    transform=ax5.transAxes, fontsize=10, style='italic', color='gray')
        
        # Title and summary
        summary = (f"Algorithm: {results['algorithm'].upper()} | "
                  f"Stability: {results['overall_stability']:.3f} | "
                  f"Coherence: {results['coherence_cv']:.3f} | "
                  f"Diversity: {results['diversity']:.3f}")
        
        fig.suptitle(f'Robust Topic Modeling Results\n{summary}', fontsize=16, y=0.98)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'optimized_topic_modeling_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved: {os.path.join(RESULTS_DIR, 'optimized_topic_modeling_results.png')}")
        
    def save_comprehensive_results(self, results):
        """Save all results"""
        # Main report
        with open(os.path.join(RESULTS_DIR, 'optimized_topic_analysis.txt'), 'w') as f:
            f.write("OPTIMIZED ROBUST TOPIC MODELING ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"CONFIGURATION:\n")
            f.write(f"• Algorithm: {results['algorithm'].upper()}\n")
            f.write(f"• Number of topics: {results['n_topics']}\n")
            f.write(f"• Documents: {len(results['doc_assignments'])}\n")
            f.write(f"• Features: {len(results['feature_names'])}\n\n")
            
            f.write(f"QUALITY ASSESSMENT:\n")
            f.write(f"• Overall stability: {results['overall_stability']:.3f}\n")
            f.write(f"• Coherence (C_v): {results['coherence_cv']:.3f}\n")
            f.write(f"• Coherence (UMass): {results['coherence_umass']:.3f}\n")
            f.write(f"• Topic diversity: {results['diversity']:.3f}\n\n")
            
            f.write(f"TOPIC BREAKDOWN:\n")
            f.write("-" * 60 + "\n")
            
            for i, (topic, interp) in enumerate(zip(results['topics'], results['interpretations'])):
                f.write(f"\nTopic {i+1}: {interp}\n")
                f.write(f"  Stability: {results['topic_stability'][i]:.3f}\n")
                f.write(f"  Document count: {results['topic_counts'][i]}\n")
                f.write(f"  Top 15 terms: {', '.join(topic)}\n")
        
        # CSV files
        pd.DataFrame({
            'metric': ['algorithm', 'n_topics', 'overall_stability', 'coherence_cv', 
                      'coherence_umass', 'diversity'],
            'value': [results['algorithm'], results['n_topics'], results['overall_stability'],
                     results['coherence_cv'], results['coherence_umass'], results['diversity']]
        }).to_csv(os.path.join(RESULTS_DIR, 'optimized_metrics.csv'), index=False)
        
        # Document assignments
        pd.DataFrame({
            'doc_id': range(len(results['doc_assignments'])),
            'topic': results['doc_assignments'] + 1,
            'topic_name': [results['interpretations'][t] for t in results['doc_assignments']],
            'text_preview': [doc[:120] + '...' if len(doc) > 120 else doc 
                           for doc in self.raw_docs[:len(results['doc_assignments'])]]
        }).to_csv(os.path.join(RESULTS_DIR, 'optimized_assignments.csv'), index=False)
        
    def run_analysis(self):
        """Run complete optimized analysis"""
        print("=" * 70)
        print("OPTIMIZED ROBUST TOPIC MODELING")
        print("=" * 70)
        
        # Load and preprocess
        self.load_data()
        docs_phr = self.advanced_phrase_detection()
        X, feature_names, vectorizer = self.create_optimized_features(docs_phr)
        
        # Compare algorithms
        print("\n>>> Algorithm Comparison")
        comparison = {}
        
        for alg_name, alg_func in [('lda', self.fit_optimized_lda), ('nmf', self.fit_optimized_nmf)]:
            print(f"\nTesting {alg_name.upper()}...")
            
            model, W, H = alg_func(X, self.n_topics)
            topics = self.get_top_terms(H, feature_names)
            doc_assignments = W.argmax(axis=1)
            
            # Metrics
            overall_stability, topic_stability = self.calculate_stability(X, feature_names, alg_name)
            umass, cv = self.calculate_coherence(topics)
            diversity = self.calculate_topic_diversity(topics)
            
            comparison[alg_name] = {
                'stability': overall_stability,
                'coherence_cv': cv, 
                'coherence_umass': umass,
                'diversity': diversity,
                'topics': topics,
                'doc_assignments': doc_assignments,
                'topic_stability': topic_stability,
                'model': model
            }
            
            print(f"  Stability: {overall_stability:.3f} | Coherence: {cv:.3f} | Diversity: {diversity:.3f}")
        
        # Select best algorithm
        def score_algorithm(metrics):
            return (metrics['stability'] * 0.4 + 
                   metrics['coherence_cv'] * 0.3 + 
                   metrics['diversity'] * 0.3)
        
        best_alg = max(comparison, key=lambda x: score_algorithm(comparison[x]))
        print(f"\n>>> Best algorithm: {best_alg.upper()}")
        
        # Final results
        best = comparison[best_alg]
        topic_counts = [np.sum(best['doc_assignments'] == i) for i in range(self.n_topics)]
        
        interpretations = self.detailed_topic_analysis(
            best['topics'], best['doc_assignments'], feature_names)
        
        results = {
            'algorithm': best_alg,
            'n_topics': self.n_topics,
            'topics': best['topics'],
            'doc_assignments': best['doc_assignments'],
            'topic_counts': topic_counts,
            'overall_stability': best['stability'],
            'topic_stability': best['topic_stability'],
            'coherence_cv': best['coherence_cv'],
            'coherence_umass': best['coherence_umass'],
            'diversity': best['diversity'],
            'interpretations': interpretations,
            'feature_names': feature_names,
            'comparison': comparison
        }
        
        # Save and visualize
        self.save_comprehensive_results(results)
        self.visualize_comprehensive_results(results)
        
        # Final summary
        print("\n" + "=" * 70)
        print("FINAL OPTIMIZED RESULTS")
        print("=" * 70)
        print(f"Algorithm: {best_alg.upper()}")
        print(f"Overall stability: {best['stability']:.3f}")
        print(f"Coherence: {best['coherence_cv']:.3f}")
        print(f"Diversity: {best['diversity']:.3f}")
        print(f"Topic separation: {'GOOD' if best['diversity'] > 0.7 else 'MODERATE' if best['diversity'] > 0.4 else 'POOR'}")
        
        return results


if __name__ == "__main__":
    modeler = OptimizedTopicModeler(n_topics=5)
    results = modeler.run_analysis()
    
    print("\n✓ OPTIMIZED ANALYSIS COMPLETE")
    print("Files saved:")
    print("  • optimized_topic_analysis.txt - Detailed report")
    print("  • optimized_metrics.csv - Quality metrics")
    print("  • optimized_assignments.csv - Document classifications")
    print("  • optimized_topic_modeling_results.png - Comprehensive visualization")