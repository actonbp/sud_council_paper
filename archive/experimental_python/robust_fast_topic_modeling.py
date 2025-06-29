#!/usr/bin/env python3
"""
Fast Robust Topic Modeling Pipeline for SUD Focus Groups
Uses phrase detection + TF-IDF + both LDA and NMF with comprehensive validation
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path

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

# Domain-specific term filtering
DOMAIN_RE = re.compile(
    r"\b(substance|abuse|addict\w*|drug\w*|alcohol\w*|counsel\w*|mental\s+health)\b",
    flags=re.I,
)

# Extended stopwords
EXTRA_STOP = set("""
feel think thing stuff kind gonna wanna like know work job family friends time
really actually maybe pretty basically bit lot kind_of sort_of yeah um uh okay
going go get got make made see say said sure look looking good bad right wrong
yes no dont don't want wanted people person someone something anything nothing
way ways mean means meant thats that's thing things heres here's theres there's
probably definitely obviously clearly certainly guess suppose probably might
exactly totally completely absolutely fairly quite rather somewhat somehow
student students class classes course courses just also even would could should
can could've would've should've let's lets still well much many though
""".split())


class FastRobustTopicModeler:
    """Fast and robust topic modeling with comprehensive validation"""
    
    def __init__(self, n_topics=5):
        self.n_topics = n_topics
        self.results = {}
        
    def load_data(self):
        """Load and perform basic cleaning of CSV files"""
        docs = []
        
        print("\n>>> Loading data files...")
        for fn in CSV_FILES:
            fp = os.path.join(DATA_DIR, fn)
            df = pd.read_csv(fp)
            raw = len(df)
            
            # Filter out moderator rows
            df = df[df["Text"].notna()]
            df = df[~df["Speaker"].astype(str).str.fullmatch(r"[A-Z]{2,3}")]
            
            texts = df["Text"].astype(str).tolist()
            docs.extend(texts)
            
            print(f"{fn:45} | rows {raw:3} → kept {len(df):3}")
        
        print(f"Total student utterances = {len(docs)}")
        self.raw_docs = docs
        
        # Remove domain-specific terms
        self.docs_clean = [DOMAIN_RE.sub("", d.lower()) for d in docs]
        
    def detect_phrases(self):
        """Detect phrases using Gensim"""
        print(">>> Detecting phrases with Gensim...")
        
        # Tokenize
        sentences = [simple_preprocess(d, deacc=True) for d in self.docs_clean]
        
        # Detect bigrams and trigrams
        bigram = Phrases(sentences, min_count=3, threshold=10, delimiter="_")
        trigram = Phrases(bigram[sentences], threshold=9, delimiter="_")
        phraser = Phraser(trigram)
        
        # Apply phrase detection
        docs_phr = [" ".join(phraser[s]) for s in sentences]
        self.tokens = [phraser[s] for s in sentences]  # For coherence later
        
        return docs_phr
        
    def create_feature_matrix(self, docs):
        """Create TF-IDF matrix"""
        stop_full = list(ENGLISH_STOP_WORDS.union(EXTRA_STOP))
        
        vectorizer = TfidfVectorizer(
            min_df=3, 
            max_df=0.80,
            stop_words=stop_full,
            token_pattern=r"(?u)\b\w[\w_]+\b",  # Allow underscores for phrases
            sublinear_tf=True,
            max_features=150
        )
            
        X = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()
        
        print(f"Feature matrix: {X.shape[0]} docs × {X.shape[1]} terms")
        return X, feature_names, vectorizer
        
    def fit_nmf(self, X, n_topics, random_state=42):
        """Fit NMF model"""
        model = NMF(
            n_components=n_topics,
            init='nndsvd',
            max_iter=500,
            random_state=random_state,
            alpha_W=0.1,
            alpha_H=0.1,
            l1_ratio=0.5
        )
        W = model.fit_transform(X)
        H = model.components_
        return model, W, H
        
    def fit_lda(self, X, n_topics, random_state=42):
        """Fit LDA model"""
        model = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=100,
            learning_method='online',
            learning_offset=50.,
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
        
    def calculate_stability(self, X, feature_names, algorithm='nmf', n_seeds=20):
        """Calculate topic stability across random seeds"""
        print(f">>> Testing stability across {n_seeds} random seeds for {algorithm.upper()}...")
        
        reference_topics = None
        all_jaccard_scores = []
        
        for seed in range(n_seeds):
            if algorithm == 'nmf':
                _, _, H = self.fit_nmf(X, self.n_topics, seed)
            else:
                _, _, H = self.fit_lda(X, self.n_topics, seed)
                
            topics = self.get_top_terms(H, feature_names)
            
            if seed == 0:
                reference_topics = topics
            else:
                # Calculate pairwise Jaccard similarity
                jaccard_matrix = np.zeros((self.n_topics, self.n_topics))
                for i, ref_topic in enumerate(reference_topics):
                    ref_set = set(ref_topic[:10])  # Top 10 terms
                    for j, test_topic in enumerate(topics):
                        test_set = set(test_topic[:10])
                        if len(ref_set | test_set) > 0:
                            jaccard = len(ref_set & test_set) / len(ref_set | test_set)
                        else:
                            jaccard = 0
                        jaccard_matrix[i, j] = jaccard
                
                # Find best matching using Hungarian algorithm
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
        """Calculate topic coherence using Gensim"""
        if not hasattr(self, 'tokens'):
            self.tokens = [simple_preprocess(d) for d in self.docs_clean]
            
        dictionary = Dictionary(self.tokens)
        
        # U_mass coherence (intrinsic)
        cm_umass = CoherenceModel(
            topics=topics,
            texts=self.tokens,
            dictionary=dictionary,
            coherence='u_mass'
        )
        
        # C_v coherence  
        cm_cv = CoherenceModel(
            topics=topics,
            texts=self.tokens,
            dictionary=dictionary,
            coherence='c_v'
        )
        
        return cm_umass.get_coherence(), cm_cv.get_coherence()
        
    def calculate_topic_diversity(self, topics):
        """Calculate topic diversity (uniqueness of terms across topics)"""
        all_terms = []
        for topic in topics:
            all_terms.extend(topic[:10])  # Top 10 terms
            
        unique_terms = len(set(all_terms))
        total_terms = len(all_terms)
        diversity = unique_terms / total_terms if total_terms > 0 else 0
        
        return diversity
        
    def bootstrap_confidence(self, X, feature_names, algorithm='nmf', n_bootstrap=30):
        """Calculate bootstrap confidence intervals for topic assignments"""
        print(f">>> Bootstrap confidence intervals ({n_bootstrap} iterations)...")
        
        n_docs = X.shape[0]
        assignment_counts = np.zeros((n_docs, self.n_topics))
        
        for i in range(n_bootstrap):
            # Resample documents
            indices = np.random.choice(n_docs, n_docs, replace=True)
            X_boot = X[indices]
            
            # Fit model
            if algorithm == 'nmf':
                _, W, _ = self.fit_nmf(X_boot, self.n_topics, random_state=i)
            else:
                _, W, _ = self.fit_lda(X_boot, self.n_topics, random_state=i)
                
            # Track assignments for original documents
            for orig_idx, boot_idx in enumerate(indices):
                if boot_idx < n_docs:
                    topic = W[orig_idx].argmax()
                    assignment_counts[boot_idx, topic] += 1
                    
        # Calculate confidence as proportion of times assigned to most common topic
        confidence_scores = []
        for doc_idx in range(n_docs):
            if assignment_counts[doc_idx].sum() > 0:
                confidence = assignment_counts[doc_idx].max() / assignment_counts[doc_idx].sum()
            else:
                confidence = 0
            confidence_scores.append(confidence)
                
        return np.array(confidence_scores)
        
    def find_optimal_k(self, X, feature_names, k_range=range(3, 9)):
        """Find optimal number of topics using combined metrics"""
        print("\n>>> Finding optimal number of topics...")
        
        metrics = {
            'k': [],
            'coherence_cv': [],
            'coherence_umass': [],
            'diversity': [],
            'stability': []
        }
        
        for k in k_range:
            print(f"Testing k={k}...")
            
            # Fit NMF model (typically more stable)
            _, _, H = self.fit_nmf(X, k)
            topics = self.get_top_terms(H, feature_names)
            
            # Calculate metrics
            umass, cv = self.calculate_coherence(topics)
            diversity = self.calculate_topic_diversity(topics)
            stability, _ = self.calculate_stability(X, feature_names, 'nmf', n_seeds=10)
            
            metrics['k'].append(k)
            metrics['coherence_cv'].append(cv)
            metrics['coherence_umass'].append(umass)
            metrics['diversity'].append(diversity)
            metrics['stability'].append(stability)
            
        return pd.DataFrame(metrics)
        
    def interpret_topics(self, topics, doc_assignments):
        """Generate interpretations for topics based on key themes"""
        interpretations = []
        
        # Common themes in counseling/psychology
        theme_keywords = {
            'helping/service': ['help', 'support', 'assist', 'care', 'service', 'give', 'provide', 'difference'],
            'personal/family': ['family', 'personal', 'experience', 'life', 'story', 'background', 'grew'],
            'education/training': ['school', 'class', 'course', 'learn', 'study', 'education', 'training', 'program'],
            'career/professional': ['career', 'job', 'profession', 'work', 'field', 'opportunity', 'professional'],
            'emotional/burnout': ['feel', 'emotion', 'stress', 'burnout', 'mental', 'cope', 'challenging'],
            'clinical/practice': ['clinical', 'practice', 'therapy', 'session', 'client', 'patient', 'hours'],
            'requirements/licensing': ['license', 'certification', 'requirement', 'hours', 'credential', 'supervised']
        }
        
        for i, topic_terms in enumerate(topics):
            # Count theme matches in top 10 terms
            theme_scores = {}
            for theme, keywords in theme_keywords.items():
                score = sum(1 for term in topic_terms[:10] 
                          for keyword in keywords if keyword in term.lower())
                if score > 0:
                    theme_scores[theme] = score
                    
            # Generate interpretation
            if theme_scores:
                top_theme = max(theme_scores, key=theme_scores.get)
                interpretation = f"{top_theme.title()}"
            else:
                interpretation = "Mixed theme"
                
            # Add document count
            doc_count = sum(doc_assignments == i)
            interpretations.append(f"{interpretation} ({doc_count} docs)")
            
        return interpretations
        
    def visualize_results(self, results_dict):
        """Create comprehensive visualization"""
        plt.style.use('default')
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # Colors for consistency
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_topics))
        
        # 1. Topic distribution
        ax1 = fig.add_subplot(gs[0, 0])
        topic_counts = results_dict['topic_counts']
        bars = ax1.bar(range(len(topic_counts)), topic_counts, color=colors)
        ax1.set_xlabel('Topic')
        ax1.set_ylabel('Number of Documents')
        ax1.set_title('Document Distribution')
        ax1.set_xticks(range(len(topic_counts)))
        ax1.set_xticklabels([f'T{i+1}' for i in range(len(topic_counts))])
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 2. Stability scores
        ax2 = fig.add_subplot(gs[0, 1])
        stability_scores = results_dict['topic_stability']
        bars = ax2.bar(range(len(stability_scores)), stability_scores, color=colors)
        ax2.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='Good stability')
        ax2.set_xlabel('Topic')
        ax2.set_ylabel('Stability Score')
        ax2.set_title('Topic Stability (Jaccard)')
        ax2.set_xticks(range(len(stability_scores)))
        ax2.set_xticklabels([f'T{i+1}' for i in range(len(stability_scores))])
        ax2.set_ylim(0, 1.05)
        ax2.legend()
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 3. Confidence distribution
        ax3 = fig.add_subplot(gs[0, 2])
        confidence = results_dict['bootstrap_confidence']
        ax3.hist(confidence, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        ax3.set_xlabel('Confidence Score')
        ax3.set_ylabel('Number of Documents')
        ax3.set_title('Bootstrap Confidence')
        mean_conf = confidence.mean()
        ax3.axvline(x=mean_conf, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_conf:.2f}')
        ax3.legend()
        
        # 4. Algorithm comparison
        if 'algorithm_comparison' in results_dict:
            ax4 = fig.add_subplot(gs[1, 0])
            comp = results_dict['algorithm_comparison']
            metrics = ['Stability', 'Coherence', 'Diversity']
            x = np.arange(len(metrics))
            width = 0.35
            
            lda_scores = [comp['lda']['stability'], comp['lda']['coherence_cv'], comp['lda']['diversity']]
            nmf_scores = [comp['nmf']['stability'], comp['nmf']['coherence_cv'], comp['nmf']['diversity']]
            
            # Normalize scores for comparison
            max_scores = [max(lda_scores[i], nmf_scores[i]) for i in range(len(metrics))]
            lda_norm = [lda_scores[i]/max_scores[i] if max_scores[i] > 0 else 0 for i in range(len(metrics))]
            nmf_norm = [nmf_scores[i]/max_scores[i] if max_scores[i] > 0 else 0 for i in range(len(metrics))]
            
            ax4.bar(x - width/2, lda_norm, width, label='LDA', alpha=0.8)
            ax4.bar(x + width/2, nmf_norm, width, label='NMF', alpha=0.8)
            ax4.set_xlabel('Metrics')
            ax4.set_ylabel('Normalized Score')
            ax4.set_title('Algorithm Comparison')
            ax4.set_xticks(x)
            ax4.set_xticklabels(metrics)
            ax4.legend()
            ax4.set_ylim(0, 1.1)
        
        # 5. K selection results
        if 'k_selection' in results_dict:
            ax5 = fig.add_subplot(gs[1, 1])
            k_df = results_dict['k_selection']
            
            # Normalize metrics for plotting
            stability_norm = k_df['stability'] / k_df['stability'].max()
            coherence_norm = (k_df['coherence_cv'] - k_df['coherence_cv'].min()) / (k_df['coherence_cv'].max() - k_df['coherence_cv'].min())
            diversity_norm = k_df['diversity'] / k_df['diversity'].max()
            
            ax5.plot(k_df['k'], stability_norm, 'o-', label='Stability', linewidth=2)
            ax5.plot(k_df['k'], coherence_norm, 's-', label='Coherence', linewidth=2)
            ax5.plot(k_df['k'], diversity_norm, '^-', label='Diversity', linewidth=2)
            
            ax5.set_xlabel('Number of Topics (k)')
            ax5.set_ylabel('Normalized Score')
            ax5.set_title('K Selection Metrics')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            ax5.set_xticks(k_df['k'])
            ax5.axvline(x=self.n_topics, color='red', linestyle=':', alpha=0.7, 
                       label=f'Selected k={self.n_topics}')
        
        # 6. Topics visualization (word cloud style)
        ax6 = fig.add_subplot(gs[1, 2])
        topics = results_dict['topics']
        interpretations = results_dict.get('interpretations', [f"Topic {i+1}" for i in range(len(topics))])
        
        # Create text display
        y_positions = np.linspace(0.9, 0.1, len(topics))
        for i, (topic, interp) in enumerate(zip(topics, interpretations)):
            color = colors[i]
            topic_text = f"{interp}\n  {', '.join(topic[:8])}"
            ax6.text(0.05, y_positions[i], topic_text, transform=ax6.transAxes,
                    fontsize=10, color=color, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Topic Interpretations & Top Terms', fontsize=12)
        
        # 7. Performance metrics summary (bottom row)
        ax7 = fig.add_subplot(gs[2, :])
        metrics_text = f"""
        ROBUST TOPIC MODELING RESULTS SUMMARY
        
        Algorithm: {results_dict['algorithm'].upper()}
        Number of Topics: {results_dict['n_topics']}
        
        Quality Metrics:
        • Overall Stability: {results_dict['overall_stability']:.3f} (Jaccard similarity across seeds)
        • Topic Coherence (C_v): {results_dict['coherence_cv']:.3f} (semantic coherence)
        • Topic Coherence (UMass): {results_dict['coherence_umass']:.3f} (intrinsic coherence)
        • Topic Diversity: {results_dict['diversity']:.3f} (term uniqueness across topics)
        • Mean Bootstrap Confidence: {results_dict['bootstrap_confidence'].mean():.3f} (assignment certainty)
        
        Stability by Topic: {', '.join([f'T{i+1}={s:.2f}' for i, s in enumerate(results_dict['topic_stability'])])}
        """
        
        ax7.text(0.05, 0.95, metrics_text, transform=ax7.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        ax7.set_xlim(0, 1)
        ax7.set_ylim(0, 1)
        ax7.axis('off')
        
        # Overall title
        fig.suptitle(f'Comprehensive Robust Topic Modeling Analysis', fontsize=16, y=0.98)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'robust_comprehensive_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Comprehensive visualization saved to: {os.path.join(RESULTS_DIR, 'robust_comprehensive_results.png')}")
        
    def save_results(self, results_dict):
        """Save comprehensive results"""
        # Save detailed topic report
        with open(os.path.join(RESULTS_DIR, 'robust_comprehensive_report.txt'), 'w') as f:
            f.write("ROBUST TOPIC MODELING COMPREHENSIVE REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"CONFIGURATION:\n")
            f.write(f"Algorithm: {results_dict['algorithm'].upper()}\n")
            f.write(f"Number of topics: {results_dict['n_topics']}\n")
            f.write(f"Documents analyzed: {len(results_dict['doc_assignments'])}\n")
            f.write(f"Features extracted: {len(results_dict['feature_names'])}\n\n")
            
            f.write(f"QUALITY METRICS:\n")
            f.write(f"Overall stability: {results_dict['overall_stability']:.3f}\n")
            f.write(f"Coherence (C_v): {results_dict['coherence_cv']:.3f}\n")
            f.write(f"Coherence (UMass): {results_dict['coherence_umass']:.3f}\n")
            f.write(f"Topic diversity: {results_dict['diversity']:.3f}\n")
            f.write(f"Mean confidence: {results_dict['bootstrap_confidence'].mean():.3f}\n\n")
            
            f.write(f"TOPICS WITH INTERPRETATIONS:\n")
            f.write("-" * 60 + "\n")
            
            interpretations = results_dict.get('interpretations', [])
            for i, topic in enumerate(results_dict['topics']):
                stability = results_dict['topic_stability'][i]
                count = results_dict['topic_counts'][i]
                interp = interpretations[i] if i < len(interpretations) else f"Topic {i+1}"
                
                f.write(f"\nTopic {i+1}: {interp}\n")
                f.write(f"  Documents: {count} | Stability: {stability:.3f}\n")
                f.write(f"  Top 15 terms: {', '.join(topic)}\n")
                
        # Save metrics CSV
        metrics_df = pd.DataFrame({
            'metric': ['algorithm', 'n_topics', 'overall_stability', 'coherence_cv', 
                      'coherence_umass', 'diversity', 'mean_confidence'],
            'value': [results_dict['algorithm'], results_dict['n_topics'],
                     results_dict['overall_stability'], results_dict['coherence_cv'],
                     results_dict['coherence_umass'], results_dict['diversity'],
                     results_dict['bootstrap_confidence'].mean()]
        })
        metrics_df.to_csv(os.path.join(RESULTS_DIR, 'robust_comprehensive_metrics.csv'), index=False)
        
        # Save document assignments
        assignments_df = pd.DataFrame({
            'doc_id': range(len(results_dict['doc_assignments'])),
            'topic': results_dict['doc_assignments'] + 1,  # 1-indexed
            'confidence': results_dict['bootstrap_confidence'],
            'text_preview': [doc[:150] + '...' if len(doc) > 150 else doc 
                           for doc in self.raw_docs[:len(results_dict['doc_assignments'])]]
        })
        assignments_df.to_csv(os.path.join(RESULTS_DIR, 'robust_comprehensive_assignments.csv'), index=False)
        
        # Save k-selection results
        if 'k_selection' in results_dict:
            results_dict['k_selection'].to_csv(
                os.path.join(RESULTS_DIR, 'k_selection_comprehensive.csv'), index=False)
            
    def run_full_pipeline(self):
        """Execute the complete robust topic modeling pipeline"""
        print("="*70)
        print("FAST ROBUST TOPIC MODELING PIPELINE")
        print("="*70)
        
        # Load data
        self.load_data()
        
        # Detect phrases and create features
        docs_phr = self.detect_phrases()
        X, feature_names, vectorizer = self.create_feature_matrix(docs_phr)
        
        # Find optimal k
        k_df = self.find_optimal_k(X, feature_names)
        print("\nK selection results:")
        print(k_df.round(3))
        
        # Compare LDA vs NMF
        print("\n>>> Comparing LDA vs NMF...")
        comparison = {}
        
        for alg_name, alg_func in [('nmf', self.fit_nmf), ('lda', self.fit_lda)]:
            print(f"\nEvaluating {alg_name.upper()}...")
            
            # Fit model
            model, W, H = alg_func(X, self.n_topics)
            topics = self.get_top_terms(H, feature_names)
            
            # Calculate metrics
            overall_stability, topic_stability = self.calculate_stability(
                X, feature_names, alg_name, n_seeds=15)
            umass, cv = self.calculate_coherence(topics)
            diversity = self.calculate_topic_diversity(topics)
            confidence = self.bootstrap_confidence(X, feature_names, alg_name, n_bootstrap=30)
            
            comparison[alg_name] = {
                'stability': overall_stability,
                'coherence_cv': cv,
                'coherence_umass': umass,
                'diversity': diversity,
                'confidence': confidence.mean(),
                'model': model,
                'W': W,
                'H': H,
                'topics': topics,
                'topic_stability': topic_stability
            }
            
            print(f"{alg_name.upper()} - Stability: {overall_stability:.3f}, "
                  f"Coherence: {cv:.3f}, Diversity: {diversity:.3f}")
        
        # Select best algorithm based on combined score
        def combined_score(metrics):
            return (metrics['stability'] * 0.35 + 
                   metrics['coherence_cv'] * 0.25 + 
                   metrics['diversity'] * 0.20 +
                   metrics['confidence'] * 0.20)
        
        best_alg = max(comparison, key=lambda x: combined_score(comparison[x]))
        print(f"\n>>> Best algorithm: {best_alg.upper()} "
              f"(score: {combined_score(comparison[best_alg]):.3f})")
        
        # Get final results from best algorithm
        best_results = comparison[best_alg]
        doc_assignments = best_results['W'].argmax(axis=1)
        topic_counts = [np.sum(doc_assignments == i) for i in range(self.n_topics)]
        
        # Generate interpretations
        interpretations = self.interpret_topics(best_results['topics'], doc_assignments)
        
        # Compile comprehensive results
        final_results = {
            'algorithm': best_alg,
            'n_topics': self.n_topics,
            'topics': best_results['topics'],
            'doc_assignments': doc_assignments,
            'topic_counts': topic_counts,
            'overall_stability': best_results['stability'],
            'topic_stability': best_results['topic_stability'],
            'coherence_cv': best_results['coherence_cv'],
            'coherence_umass': best_results['coherence_umass'],
            'diversity': best_results['diversity'],
            'bootstrap_confidence': self.bootstrap_confidence(X, feature_names, best_alg, n_bootstrap=50),
            'interpretations': interpretations,
            'feature_names': feature_names,
            'k_selection': k_df,
            'algorithm_comparison': comparison
        }
        
        # Print comprehensive summary
        print("\n" + "="*70)
        print("COMPREHENSIVE RESULTS SUMMARY")
        print("="*70)
        print(f"Selected Algorithm: {best_alg.upper()}")
        print(f"Number of topics: {self.n_topics}")
        print(f"Overall stability: {best_results['stability']:.3f}")
        print(f"Coherence (C_v): {best_results['coherence_cv']:.3f}")
        print(f"Topic diversity: {best_results['diversity']:.3f}")
        
        print("\nFinal Topic Themes:")
        for i, (topic, interp) in enumerate(zip(best_results['topics'], interpretations)):
            print(f"\n{interp}")
            print(f"  Stability: {best_results['topic_stability'][i]:.3f}")
            print(f"  Top terms: {', '.join(topic[:12])}")
            
        # Save and visualize
        self.save_results(final_results)
        self.visualize_results(final_results)
        
        return final_results


if __name__ == "__main__":
    # Run the comprehensive pipeline
    modeler = FastRobustTopicModeler(n_topics=5)
    results = modeler.run_full_pipeline()
    
    print("\n" + "="*70)
    print("✓ COMPREHENSIVE ANALYSIS COMPLETE")
    print("="*70)
    print("All results saved to:", RESULTS_DIR)
    print("• robust_comprehensive_report.txt: Detailed analysis report")
    print("• robust_comprehensive_metrics.csv: Performance metrics")  
    print("• robust_comprehensive_assignments.csv: Document-topic assignments")
    print("• k_selection_comprehensive.csv: K optimization results")
    print("• robust_comprehensive_results.png: Full visualization suite")