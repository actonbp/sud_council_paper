#!/usr/bin/env python3
"""
Simplified Robust Topic Modeling Pipeline for SUD Focus Groups
Uses only sklearn and standard libraries
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

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.model_selection import KFold
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon

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


class SimpleRobustTopicModeler:
    """Simplified robust topic modeling using only sklearn"""
    
    def __init__(self, n_topics=5):
        self.n_topics = n_topics
        self.results = {}
        
    def load_data(self):
        """Load and perform basic cleaning of CSV files"""
        docs = []
        metadata = []
        
        print("\n>>> Loading data files...")
        for fn in CSV_FILES:
            fp = os.path.join(DATA_DIR, fn)
            df = pd.read_csv(fp)
            raw = len(df)
            
            # Filter out moderator rows
            df = df[df["Text"].notna()]
            df = df[~df["Speaker"].astype(str).str.fullmatch(r"[A-Z]{2,3}")]
            
            # Extract text and metadata
            texts = df["Text"].astype(str).tolist()
            docs.extend(texts)
            
            print(f"{fn:45} | rows {raw:3} → kept {len(df):3}")
        
        print(f"Total student utterances = {len(docs)}")
        self.raw_docs = docs
        
        # Remove domain-specific terms
        self.docs_clean = [DOMAIN_RE.sub("", d.lower()) for d in docs]
        
    def detect_phrases(self, docs, min_count=3):
        """Simple bigram detection based on co-occurrence"""
        # Tokenize documents
        tokenized = []
        for doc in docs:
            tokens = re.findall(r'\b[a-z]+\b', doc.lower())
            tokenized.append(tokens)
        
        # Count bigrams
        bigram_counts = Counter()
        for tokens in tokenized:
            for i in range(len(tokens)-1):
                bigram = f"{tokens[i]}_{tokens[i+1]}"
                bigram_counts[bigram] += 1
        
        # Keep frequent bigrams
        frequent_bigrams = {bg: count for bg, count in bigram_counts.items() 
                           if count >= min_count}
        
        # Apply bigrams to documents
        docs_with_phrases = []
        for doc in docs:
            for bigram, count in sorted(frequent_bigrams.items(), 
                                       key=lambda x: x[1], reverse=True):
                word1, word2 = bigram.split('_')
                pattern = rf'\b{word1}\s+{word2}\b'
                doc = re.sub(pattern, bigram, doc, flags=re.I)
            docs_with_phrases.append(doc)
            
        return docs_with_phrases
        
    def create_feature_matrix(self, docs, use_tfidf=True, detect_phrases=True):
        """Create document-term matrix"""
        if detect_phrases:
            print(">>> Detecting phrases...")
            docs = self.detect_phrases(docs)
            
        stop_full = ENGLISH_STOP_WORDS.union(EXTRA_STOP)
        
        if use_tfidf:
            vectorizer = TfidfVectorizer(
                min_df=3, 
                max_df=0.80,
                stop_words=stop_full,
                token_pattern=r"(?u)\b\w[\w_]+\b",  # Allow underscores for phrases
                sublinear_tf=True,
                max_features=150
            )
        else:
            vectorizer = CountVectorizer(
                min_df=3,
                max_df=0.80,
                stop_words=stop_full,
                token_pattern=r"(?u)\b\w[\w_]+\b",
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
        print(f"\n>>> Testing stability across {n_seeds} random seeds...")
        
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
        
    def calculate_coherence_pmi(self, topics, X, vectorizer):
        """Calculate PMI-based coherence"""
        # Get document-term matrix as boolean
        X_bool = (X > 0).astype(int)
        
        coherences = []
        for topic in topics:
            topic_words = topic[:10]
            
            # Get word indices
            word_indices = []
            for word in topic_words:
                if word in vectorizer.vocabulary_:
                    word_indices.append(vectorizer.vocabulary_[word])
                    
            if len(word_indices) < 2:
                coherences.append(0)
                continue
                
            # Calculate PMI for word pairs
            pmi_scores = []
            n_docs = X_bool.shape[0]
            
            for i in range(len(word_indices)):
                for j in range(i+1, len(word_indices)):
                    w1_idx = word_indices[i]
                    w2_idx = word_indices[j]
                    
                    # Count occurrences
                    p_w1 = X_bool[:, w1_idx].sum() / n_docs
                    p_w2 = X_bool[:, w2_idx].sum() / n_docs
                    p_w1_w2 = (X_bool[:, w1_idx] * X_bool[:, w2_idx]).sum() / n_docs
                    
                    # Calculate PMI
                    if p_w1_w2 > 0 and p_w1 > 0 and p_w2 > 0:
                        pmi = np.log(p_w1_w2 / (p_w1 * p_w2))
                        pmi_scores.append(pmi)
                        
            if pmi_scores:
                coherences.append(np.mean(pmi_scores))
            else:
                coherences.append(0)
                
        return np.mean(coherences)
        
    def calculate_topic_diversity(self, topics):
        """Calculate topic diversity (uniqueness of terms across topics)"""
        all_terms = []
        for topic in topics:
            all_terms.extend(topic[:10])  # Top 10 terms
            
        unique_terms = len(set(all_terms))
        total_terms = len(all_terms)
        diversity = unique_terms / total_terms if total_terms > 0 else 0
        
        return diversity
        
    def bootstrap_confidence(self, X, feature_names, algorithm='nmf', n_bootstrap=50):
        """Calculate bootstrap confidence intervals for topic assignments"""
        print(f"\n>>> Bootstrap confidence intervals ({n_bootstrap} iterations)...")
        
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
        
    def find_optimal_k(self, X, feature_names, vectorizer, k_range=range(3, 9)):
        """Find optimal number of topics"""
        print("\n>>> Finding optimal number of topics...")
        
        metrics = {
            'k': [],
            'coherence': [],
            'diversity': [],
            'stability': [],
            'reconstruction_error': []
        }
        
        for k in k_range:
            print(f"Testing k={k}...")
            
            # Fit NMF model (typically more stable than LDA)
            model, W, H = self.fit_nmf(X, k)
            topics = self.get_top_terms(H, feature_names)
            
            # Calculate metrics
            coherence = self.calculate_coherence_pmi(topics, X, vectorizer)
            diversity = self.calculate_topic_diversity(topics)
            stability, _ = self.calculate_stability(X, feature_names, 'nmf', n_seeds=10)
            reconstruction_error = model.reconstruction_err_
            
            metrics['k'].append(k)
            metrics['coherence'].append(coherence)
            metrics['diversity'].append(diversity)
            metrics['stability'].append(stability)
            metrics['reconstruction_error'].append(reconstruction_error)
            
        return pd.DataFrame(metrics)
        
    def visualize_results(self, results_dict):
        """Create visualization of results"""
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        fig = plt.figure(figsize=(16, 10))
        
        # Create a 2x3 grid
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Topic distribution
        ax1 = fig.add_subplot(gs[0, 0])
        topic_counts = results_dict['topic_counts']
        colors = plt.cm.Set3(np.linspace(0, 1, len(topic_counts)))
        bars = ax1.bar(range(len(topic_counts)), topic_counts, color=colors)
        ax1.set_xlabel('Topic')
        ax1.set_ylabel('Number of Documents')
        ax1.set_title('Document Distribution Across Topics')
        ax1.set_xticks(range(len(topic_counts)))
        ax1.set_xticklabels([f'T{i+1}' for i in range(len(topic_counts))])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 2. Stability scores
        ax2 = fig.add_subplot(gs[0, 1])
        stability_scores = results_dict['topic_stability']
        bars = ax2.bar(range(len(stability_scores)), stability_scores, color=colors)
        ax2.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='Good stability threshold')
        ax2.set_xlabel('Topic')
        ax2.set_ylabel('Stability Score')
        ax2.set_title('Topic Stability (Jaccard Similarity)')
        ax2.set_xticks(range(len(stability_scores)))
        ax2.set_xticklabels([f'T{i+1}' for i in range(len(stability_scores))])
        ax2.set_ylim(0, 1.05)
        ax2.legend()
        
        # Add value labels
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
        ax3.set_title('Bootstrap Confidence Distribution')
        mean_conf = confidence.mean()
        ax3.axvline(x=mean_conf, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_conf:.2f}')
        ax3.legend()
        
        # 4. Topic words visualization
        ax4 = fig.add_subplot(gs[1, :])
        topics = results_dict['topics']
        
        # Create text display of top terms
        y_pos = 0.9
        for i, topic in enumerate(topics):
            color = colors[i]
            # Show top 12 terms
            topic_text = f"Topic {i+1}: " + ", ".join(topic[:12])
            ax4.text(0.02, y_pos, topic_text, transform=ax4.transAxes,
                    fontsize=11, color=color, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            y_pos -= 0.18
            
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Top Terms by Topic', fontsize=14, pad=20)
        
        # 5. Add k-selection plot if available
        if 'k_selection' in results_dict:
            # Create inset plot
            ax5 = fig.add_axes([0.65, 0.08, 0.3, 0.25])  # [left, bottom, width, height]
            k_df = results_dict['k_selection']
            
            # Normalize metrics
            stability_norm = k_df['stability'] / k_df['stability'].max()
            coherence_norm = (k_df['coherence'] - k_df['coherence'].min()) / (k_df['coherence'].max() - k_df['coherence'].min())
            diversity_norm = k_df['diversity'] / k_df['diversity'].max()
            
            ax5.plot(k_df['k'], stability_norm, 'o-', label='Stability', linewidth=2)
            ax5.plot(k_df['k'], coherence_norm, 's-', label='Coherence', linewidth=2)
            ax5.plot(k_df['k'], diversity_norm, '^-', label='Diversity', linewidth=2)
            
            ax5.set_xlabel('Number of Topics (k)')
            ax5.set_ylabel('Normalized Score')
            ax5.set_title('K Selection Metrics')
            ax5.legend(loc='best', fontsize='small')
            ax5.grid(True, alpha=0.3)
            ax5.set_xticks(k_df['k'])
            
            # Highlight selected k
            ax5.axvline(x=self.n_topics, color='red', linestyle=':', alpha=0.7)
        
        # Overall title
        fig.suptitle(f'Robust Topic Modeling Results (k={self.n_topics}, {results_dict["algorithm"].upper()})',
                    fontsize=16, y=0.98)
        
        # Add summary statistics as text
        summary_text = (f"Overall Stability: {results_dict['overall_stability']:.3f} | "
                       f"Coherence: {results_dict['coherence']:.3f} | "
                       f"Diversity: {results_dict['diversity']:.3f} | "
                       f"Mean Confidence: {mean_conf:.3f}")
        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'robust_topic_modeling_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Visualization saved to: {os.path.join(RESULTS_DIR, 'robust_topic_modeling_results.png')}")
        
    def save_results(self, results_dict):
        """Save all results to files"""
        # Save topics
        with open(os.path.join(RESULTS_DIR, 'robust_topics.txt'), 'w') as f:
            f.write(f"Robust Topic Modeling Results\n")
            f.write(f"Algorithm: {results_dict['algorithm'].upper()}\n")
            f.write(f"Number of topics: {results_dict['n_topics']}\n")
            f.write(f"Overall stability: {results_dict['overall_stability']:.3f}\n")
            f.write(f"Coherence: {results_dict['coherence']:.3f}\n")
            f.write(f"Diversity: {results_dict['diversity']:.3f}\n")
            f.write(f"="*60 + "\n\n")
            
            for i, topic in enumerate(results_dict['topics']):
                stability = results_dict['topic_stability'][i]
                count = results_dict['topic_counts'][i]
                f.write(f"Topic {i+1} ({count} documents, stability={stability:.2f}):\n")
                f.write(f"  Top 15 terms: {', '.join(topic)}\n\n")
                
        # Save detailed metrics
        metrics_df = pd.DataFrame({
            'metric': ['overall_stability', 'coherence', 'diversity', 'mean_confidence'],
            'value': [
                results_dict['overall_stability'],
                results_dict['coherence'],
                results_dict['diversity'],
                results_dict['bootstrap_confidence'].mean()
            ]
        })
        metrics_df.to_csv(os.path.join(RESULTS_DIR, 'robust_metrics.csv'), index=False)
        
        # Save document assignments with confidence
        assignments_df = pd.DataFrame({
            'doc_id': range(len(results_dict['doc_assignments'])),
            'topic': results_dict['doc_assignments'] + 1,  # 1-indexed
            'confidence': results_dict['bootstrap_confidence'],
            'text_preview': [doc[:100] + '...' if len(doc) > 100 else doc 
                           for doc in self.raw_docs[:len(results_dict['doc_assignments'])]]
        })
        assignments_df.to_csv(os.path.join(RESULTS_DIR, 'robust_assignments.csv'), index=False)
        
        # Save k-selection results if available
        if 'k_selection' in results_dict:
            results_dict['k_selection'].to_csv(
                os.path.join(RESULTS_DIR, 'k_selection_results.csv'), index=False)
            
    def interpret_topics(self, topics, doc_assignments):
        """Generate interpretations for topics based on terms"""
        interpretations = []
        
        # Common themes in counseling/psychology
        theme_keywords = {
            'helping': ['help', 'support', 'assist', 'care', 'service', 'give', 'provide'],
            'personal': ['family', 'personal', 'experience', 'life', 'story', 'background'],
            'education': ['school', 'class', 'course', 'learn', 'study', 'education', 'training'],
            'career': ['career', 'job', 'profession', 'work', 'field', 'opportunity'],
            'emotional': ['feel', 'emotion', 'stress', 'burnout', 'mental', 'cope'],
            'clinical': ['clinical', 'practice', 'therapy', 'session', 'client', 'patient'],
            'requirements': ['license', 'certification', 'requirement', 'hours', 'credential']
        }
        
        for i, topic_terms in enumerate(topics):
            # Count theme matches
            theme_scores = {}
            for theme, keywords in theme_keywords.items():
                score = sum(1 for term in topic_terms[:10] 
                          for keyword in keywords if keyword in term.lower())
                if score > 0:
                    theme_scores[theme] = score
                    
            # Generate interpretation
            if theme_scores:
                top_theme = max(theme_scores, key=theme_scores.get)
                interpretation = f"{top_theme.capitalize()}-focused theme"
            else:
                interpretation = "Mixed theme"
                
            # Add document count
            doc_count = sum(doc_assignments == i)
            interpretations.append(f"{interpretation} ({doc_count} docs)")
            
        return interpretations
        
    def run_full_pipeline(self):
        """Execute the complete robust topic modeling pipeline"""
        print("="*60)
        print("ROBUST TOPIC MODELING PIPELINE (SIMPLIFIED)")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Create feature matrix with phrase detection
        X, feature_names, vectorizer = self.create_feature_matrix(
            self.docs_clean, use_tfidf=True, detect_phrases=True)
        
        # Find optimal k
        k_df = self.find_optimal_k(X, feature_names, vectorizer)
        print("\nK selection results:")
        print(k_df)
        
        # Select optimal k based on combined score
        k_df['combined_score'] = (
            k_df['stability'] * 0.4 + 
            k_df['diversity'] * 0.3 + 
            (k_df['coherence'] - k_df['coherence'].min()) / (k_df['coherence'].max() - k_df['coherence'].min()) * 0.3
        )
        optimal_k = k_df.loc[k_df['combined_score'].idxmax(), 'k']
        print(f"\nOptimal k based on combined metrics: {optimal_k}")
        
        # But allow override to user preference
        if self.n_topics != optimal_k:
            print(f"Using user-specified k={self.n_topics} instead")
            
        # Compare LDA vs NMF
        print("\n>>> Comparing LDA vs NMF...")
        comparison_results = {}
        
        for alg_name, alg_func in [('nmf', self.fit_nmf), ('lda', self.fit_lda)]:
            print(f"\nTesting {alg_name.upper()}...")
            
            # Fit model
            model, W, H = alg_func(X, self.n_topics)
            topics = self.get_top_terms(H, feature_names)
            doc_assignments = W.argmax(axis=1)
            
            # Calculate metrics
            overall_stability, topic_stability = self.calculate_stability(
                X, feature_names, alg_name, n_seeds=15)
            coherence = self.calculate_coherence_pmi(topics, X, vectorizer)
            diversity = self.calculate_topic_diversity(topics)
            confidence = self.bootstrap_confidence(X, feature_names, alg_name, n_bootstrap=30)
            
            # Count documents per topic
            topic_counts = [np.sum(doc_assignments == i) for i in range(self.n_topics)]
            
            comparison_results[alg_name] = {
                'model': model,
                'W': W,
                'H': H,
                'topics': topics,
                'doc_assignments': doc_assignments,
                'topic_counts': topic_counts,
                'overall_stability': overall_stability,
                'topic_stability': topic_stability,
                'coherence': coherence,
                'diversity': diversity,
                'bootstrap_confidence': confidence,
                'score': overall_stability * 0.4 + coherence * 0.3 + diversity * 0.3
            }
            
            print(f"{alg_name.upper()} - Stability: {overall_stability:.3f}, "
                  f"Coherence: {coherence:.3f}, Diversity: {diversity:.3f}")
        
        # Select best algorithm
        best_alg = max(comparison_results, key=lambda x: comparison_results[x]['score'])
        print(f"\n>>> Best algorithm: {best_alg.upper()}")
        
        # Get final results
        final_results = comparison_results[best_alg]
        final_results['algorithm'] = best_alg
        final_results['n_topics'] = self.n_topics
        final_results['feature_names'] = feature_names
        final_results['k_selection'] = k_df
        
        # Generate interpretations
        interpretations = self.interpret_topics(
            final_results['topics'], 
            final_results['doc_assignments']
        )
        
        # Print summary
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        print(f"Algorithm: {best_alg.upper()}")
        print(f"Number of topics: {self.n_topics}")
        print(f"Overall stability: {final_results['overall_stability']:.3f}")
        print(f"Coherence: {final_results['coherence']:.3f}")
        print(f"Diversity: {final_results['diversity']:.3f}")
        print(f"Mean confidence: {final_results['bootstrap_confidence'].mean():.3f}")
        
        print("\nTopics with interpretations:")
        for i, (topic, interp) in enumerate(zip(final_results['topics'], interpretations)):
            print(f"\nTopic {i+1}: {interp}")
            print(f"  Stability: {final_results['topic_stability'][i]:.2f}")
            print(f"  Top terms: {', '.join(topic[:10])}")
            
        # Save and visualize
        self.save_results(final_results)
        self.visualize_results(final_results)
        
        return final_results


if __name__ == "__main__":
    # Run the simplified robust pipeline
    modeler = SimpleRobustTopicModeler(n_topics=5)
    results = modeler.run_full_pipeline()
    
    print("\n✓ All results saved to:", RESULTS_DIR)
    print("  - robust_topics.txt: Topic term lists and interpretations")
    print("  - robust_metrics.csv: Performance metrics")  
    print("  - robust_assignments.csv: Document-topic assignments")
    print("  - k_selection_results.csv: K optimization results")
    print("  - robust_topic_modeling_results.png: Comprehensive visualization")