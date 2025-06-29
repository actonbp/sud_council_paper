#!/usr/bin/env python3
"""
Robust Topic Modeling Pipeline for SUD Focus Groups
Combines best practices from existing scripts with comprehensive validation
"""

import os
import re
import hashlib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# NLP imports
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.model_selection import KFold
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from scipy.stats import bootstrap

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

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
""".split())

class RobustTopicModeler:
    """Main class for robust topic modeling pipeline"""
    
    def __init__(self, n_topics=5, preprocessing='enhanced', algorithm='both'):
        self.n_topics = n_topics
        self.preprocessing = preprocessing  # 'basic', 'enhanced', 'phrasy'
        self.algorithm = algorithm  # 'lda', 'nmf', 'both'
        self.lemmatizer = WordNetLemmatizer()
        self.results = {}
        
    def load_data(self):
        """Load and perform basic cleaning of CSV files"""
        docs = []
        metadata = []
        
        print("\n>>> Loading data files...")
        for fn in CSV_FILES:
            fp = os.path.join(DATA_DIR, fn)
            md5 = hashlib.md5(open(fp, "rb").read()).hexdigest()[:8]
            df = pd.read_csv(fp)
            raw = len(df)
            
            # Filter out moderator rows
            df = df[df["Text"].notna()]
            df = df[~df["Speaker"].astype(str).str.fullmatch(r"[A-Z]{2,3}")]
            
            # Extract text and metadata
            texts = df["Text"].astype(str).tolist()
            docs.extend(texts)
            
            # Store metadata for each utterance
            for idx, row in df.iterrows():
                metadata.append({
                    'file': fn,
                    'speaker': row.get('Speaker', 'Unknown'),
                    'timestamp': row.get('In', ''),
                    'original_idx': idx
                })
            
            print(f"{fn:45} {md5} | rows {raw:3} → kept {len(df):3}")
        
        print(f"Total student utterances = {len(docs)}")
        self.raw_docs = docs
        self.metadata = metadata[:len(docs)]  # Ensure same length
        
        # Remove domain-specific terms
        self.docs_clean = [DOMAIN_RE.sub("", d.lower()) for d in docs]
        
    def preprocess_basic(self):
        """Basic preprocessing with stopword removal"""
        stop_full = list(ENGLISH_STOP_WORDS.union(EXTRA_STOP))
        return self.docs_clean, stop_full
        
    def preprocess_enhanced(self):
        """Enhanced preprocessing with lemmatization and POS filtering"""
        stop_full = list(ENGLISH_STOP_WORDS.union(EXTRA_STOP))
        processed_docs = []
        
        print("\n>>> Enhanced preprocessing with lemmatization...")
        for doc in self.docs_clean:
            # Tokenize
            tokens = simple_preprocess(doc, deacc=True)
            
            # POS tagging
            tagged = pos_tag(tokens)
            
            # Keep only nouns, verbs, adjectives
            filtered = [word for word, pos in tagged 
                       if pos.startswith(('NN', 'VB', 'JJ')) and 
                       word not in stop_full and len(word) > 2]
            
            # Lemmatize
            lemmatized = [self.lemmatizer.lemmatize(word) for word in filtered]
            
            processed_docs.append(' '.join(lemmatized))
            
        return processed_docs, stop_full
        
    def preprocess_phrasy(self):
        """Phrase-aware preprocessing"""
        print("\n>>> Phrase detection preprocessing...")
        sentences = [simple_preprocess(d, deacc=True) for d in self.docs_clean]
        
        # Detect bigrams and trigrams
        bigram = Phrases(sentences, min_count=3, threshold=10, delimiter="_")
        trigram = Phrases(bigram[sentences], threshold=9, delimiter="_")
        phraser = Phraser(trigram)
        
        # Apply phrase detection
        docs_phr = [" ".join(phraser[s]) for s in sentences]
        self.tokens = [phraser[s] for s in sentences]  # For coherence later
        
        stop_full = list(ENGLISH_STOP_WORDS.union(EXTRA_STOP))
        return docs_phr, stop_full
        
    def create_feature_matrix(self, docs, stop_words, use_tfidf=True):
        """Create document-term matrix"""
        if use_tfidf:
            vectorizer = TfidfVectorizer(
                min_df=3, 
                max_df=0.80,
                stop_words=stop_words,
                token_pattern=r"(?u)\b\w[\w'_]+\b",
                sublinear_tf=True,
                max_features=200
            )
        else:
            vectorizer = CountVectorizer(
                min_df=3,
                max_df=0.80,
                stop_words=stop_words,
                token_pattern=r"(?u)\b\w[\w'_]+\b",
                max_features=200
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
        jaccard_scores = []
        
        for seed in range(n_seeds):
            if algorithm == 'nmf':
                _, _, H = self.fit_nmf(X, self.n_topics, seed)
            else:
                _, _, H = self.fit_lda(X, self.n_topics, seed)
                
            topics = self.get_top_terms(H, feature_names)
            
            if seed == 0:
                reference_topics = topics
            else:
                # Calculate Jaccard similarity with reference
                seed_scores = []
                for ref_topic in reference_topics:
                    ref_set = set(ref_topic)
                    max_jaccard = 0
                    for test_topic in topics:
                        test_set = set(test_topic)
                        jaccard = len(ref_set & test_set) / len(ref_set | test_set)
                        max_jaccard = max(max_jaccard, jaccard)
                    seed_scores.append(max_jaccard)
                jaccard_scores.append(seed_scores)
                
        jaccard_array = np.array(jaccard_scores)
        mean_stability = jaccard_array.mean(axis=0)
        overall_stability = mean_stability.mean()
        
        return overall_stability, mean_stability
        
    def calculate_coherence(self, topics, texts=None):
        """Calculate topic coherence using Gensim"""
        if texts is None:
            texts = [simple_preprocess(d) for d in self.docs_clean]
            
        dictionary = Dictionary(texts)
        
        # U_mass coherence (doesn't need external corpus)
        cm_umass = CoherenceModel(
            topics=topics,
            texts=texts,
            dictionary=dictionary,
            coherence='u_mass'
        )
        
        # C_v coherence
        cm_cv = CoherenceModel(
            topics=topics,
            texts=texts,
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
        diversity = unique_terms / total_terms
        
        return diversity
        
    def bootstrap_confidence(self, X, feature_names, algorithm='nmf', n_bootstrap=100):
        """Calculate bootstrap confidence intervals for topic assignments"""
        print(f"\n>>> Bootstrap confidence intervals ({n_bootstrap} iterations)...")
        
        n_docs = X.shape[0]
        topic_assignments = []
        
        for i in range(n_bootstrap):
            # Resample documents
            indices = np.random.choice(n_docs, n_docs, replace=True)
            X_boot = X[indices]
            
            # Fit model
            if algorithm == 'nmf':
                _, W, _ = self.fit_nmf(X_boot, self.n_topics, random_state=i)
            else:
                _, W, _ = self.fit_lda(X_boot, self.n_topics, random_state=i)
                
            # Get assignments
            assignments = W.argmax(axis=1)
            topic_assignments.append(assignments)
            
        # Calculate confidence for each document's assignment
        confidence_scores = []
        for doc_idx in range(n_docs):
            doc_assignments = [ta[doc_idx] if doc_idx < len(ta) else -1 
                             for ta in topic_assignments]
            doc_assignments = [a for a in doc_assignments if a != -1]
            
            if doc_assignments:
                # Most common assignment
                main_topic = max(set(doc_assignments), key=doc_assignments.count)
                confidence = doc_assignments.count(main_topic) / len(doc_assignments)
                confidence_scores.append(confidence)
            else:
                confidence_scores.append(0)
                
        return np.array(confidence_scores)
        
    def find_optimal_k(self, X, feature_names, k_range=range(3, 10)):
        """Find optimal number of topics using multiple metrics"""
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
            
            # Fit model
            if self.algorithm in ['nmf', 'both']:
                _, W, H = self.fit_nmf(X, k)
            else:
                _, W, H = self.fit_lda(X, k)
                
            topics = self.get_top_terms(H, feature_names)
            
            # Calculate metrics
            umass, cv = self.calculate_coherence(topics)
            diversity = self.calculate_topic_diversity(topics)
            stability, _ = self.calculate_stability(X, feature_names, 
                                                  'nmf' if self.algorithm == 'nmf' else 'lda',
                                                  n_seeds=10)
            
            metrics['k'].append(k)
            metrics['coherence_cv'].append(cv)
            metrics['coherence_umass'].append(umass)
            metrics['diversity'].append(diversity)
            metrics['stability'].append(stability)
            
        return pd.DataFrame(metrics)
        
    def visualize_results(self, results_dict):
        """Create comprehensive visualization of results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Robust Topic Modeling Results', fontsize=16)
        
        # 1. Topic distribution
        ax = axes[0, 0]
        topic_counts = results_dict['topic_counts']
        ax.bar(range(len(topic_counts)), topic_counts)
        ax.set_xlabel('Topic')
        ax.set_ylabel('Number of Documents')
        ax.set_title('Document Distribution Across Topics')
        
        # 2. Stability scores
        ax = axes[0, 1]
        stability_scores = results_dict['topic_stability']
        ax.bar(range(len(stability_scores)), stability_scores)
        ax.axhline(y=0.8, color='r', linestyle='--', label='Good stability')
        ax.set_xlabel('Topic')
        ax.set_ylabel('Stability Score')
        ax.set_title('Topic Stability (Jaccard)')
        ax.legend()
        
        # 3. Confidence distribution
        ax = axes[0, 2]
        confidence = results_dict['bootstrap_confidence']
        ax.hist(confidence, bins=20, edgecolor='black')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Number of Documents')
        ax.set_title('Bootstrap Confidence Distribution')
        ax.axvline(x=confidence.mean(), color='r', linestyle='--', 
                  label=f'Mean: {confidence.mean():.2f}')
        ax.legend()
        
        # 4. Top terms heatmap
        ax = axes[1, 0]
        # Create term importance matrix
        topics = results_dict['topics']
        n_terms = 10
        term_matrix = []
        all_terms = []
        
        for topic in topics:
            topic_terms = topic[:n_terms]
            all_terms.extend(topic_terms)
            
        all_terms = list(set(all_terms))[:20]  # Top 20 unique terms
        
        for topic in topics:
            row = [1 if term in topic[:n_terms] else 0 for term in all_terms]
            term_matrix.append(row)
            
        im = ax.imshow(term_matrix, cmap='Blues', aspect='auto')
        ax.set_xticks(range(len(all_terms)))
        ax.set_xticklabels(all_terms, rotation=45, ha='right')
        ax.set_yticks(range(len(topics)))
        ax.set_yticklabels([f'Topic {i+1}' for i in range(len(topics))])
        ax.set_title('Top Terms by Topic')
        
        # 5. Coherence comparison
        ax = axes[1, 1]
        if 'algorithm_comparison' in results_dict:
            comp = results_dict['algorithm_comparison']
            x = range(len(comp['algorithms']))
            ax.plot(x, comp['coherence_cv'], 'o-', label='C_v')
            ax.plot(x, comp['coherence_umass_normalized'], 's-', label='U_mass (norm)')
            ax.set_xticks(x)
            ax.set_xticklabels(comp['algorithms'])
            ax.set_ylabel('Coherence Score')
            ax.set_title('Algorithm Comparison')
            ax.legend()
            
        # 6. K selection
        ax = axes[1, 2]
        if 'k_selection' in results_dict:
            k_df = results_dict['k_selection']
            ax2 = ax.twinx()
            
            ax.plot(k_df['k'], k_df['coherence_cv'], 'b-', label='Coherence')
            ax2.plot(k_df['k'], k_df['stability'], 'r-', label='Stability')
            
            ax.set_xlabel('Number of Topics (k)')
            ax.set_ylabel('Coherence (C_v)', color='b')
            ax2.set_ylabel('Stability', color='r')
            ax.set_title('Optimal K Selection')
            ax.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='r')
            
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'robust_topic_modeling_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_results(self, results_dict):
        """Save all results to files"""
        # Save topics
        with open(os.path.join(RESULTS_DIR, 'robust_topics.txt'), 'w') as f:
            for i, topic in enumerate(results_dict['topics']):
                f.write(f"Topic {i+1}: {', '.join(topic)}\n")
                
        # Save detailed results
        results_df = pd.DataFrame({
            'metric': ['overall_stability', 'coherence_cv', 'coherence_umass', 
                      'diversity', 'mean_confidence'],
            'value': [
                results_dict['overall_stability'],
                results_dict['coherence_cv'],
                results_dict['coherence_umass'],
                results_dict['diversity'],
                results_dict['bootstrap_confidence'].mean()
            ]
        })
        results_df.to_csv(os.path.join(RESULTS_DIR, 'robust_metrics.csv'), index=False)
        
        # Save document assignments with confidence
        assignments_df = pd.DataFrame({
            'doc_id': range(len(results_dict['doc_assignments'])),
            'topic': results_dict['doc_assignments'],
            'confidence': results_dict['bootstrap_confidence'],
            'text': self.raw_docs[:len(results_dict['doc_assignments'])]
        })
        assignments_df.to_csv(os.path.join(RESULTS_DIR, 'robust_assignments.csv'), index=False)
        
    def run_full_pipeline(self):
        """Execute the complete robust topic modeling pipeline"""
        print("="*60)
        print("ROBUST TOPIC MODELING PIPELINE")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Preprocessing
        if self.preprocessing == 'basic':
            processed_docs, stop_words = self.preprocess_basic()
        elif self.preprocessing == 'enhanced':
            processed_docs, stop_words = self.preprocess_enhanced()
        else:  # phrasy
            processed_docs, stop_words = self.preprocess_phrasy()
            
        # Create feature matrix
        X, feature_names, vectorizer = self.create_feature_matrix(
            processed_docs, stop_words, use_tfidf=True)
        
        # Find optimal k
        k_df = self.find_optimal_k(X, feature_names)
        print("\nK selection results:")
        print(k_df)
        
        # Store k selection results
        self.results['k_selection'] = k_df
        
        # Compare algorithms if requested
        if self.algorithm == 'both':
            print("\n>>> Comparing LDA vs NMF...")
            comparison = {
                'algorithms': ['LDA', 'NMF'],
                'coherence_cv': [],
                'coherence_umass': [],
                'coherence_umass_normalized': [],
                'stability': [],
                'diversity': []
            }
            
            for alg in ['lda', 'nmf']:
                if alg == 'lda':
                    model, W, H = self.fit_lda(X, self.n_topics)
                else:
                    model, W, H = self.fit_nmf(X, self.n_topics)
                    
                topics = self.get_top_terms(H, feature_names)
                umass, cv = self.calculate_coherence(topics)
                diversity = self.calculate_topic_diversity(topics)
                stability, _ = self.calculate_stability(X, feature_names, alg, n_seeds=10)
                
                comparison['coherence_cv'].append(cv)
                comparison['coherence_umass'].append(umass)
                comparison['coherence_umass_normalized'].append((umass + 10) / 10)  # Normalize
                comparison['stability'].append(stability)
                comparison['diversity'].append(diversity)
                
            self.results['algorithm_comparison'] = comparison
            
            # Select best algorithm
            best_idx = np.argmax(np.array(comparison['coherence_cv']) + 
                               np.array(comparison['stability']))
            best_alg = ['lda', 'nmf'][best_idx]
            print(f"\nBest algorithm: {best_alg.upper()}")
        else:
            best_alg = self.algorithm
            
        # Fit final model with best settings
        print(f"\n>>> Fitting final {best_alg.upper()} model with k={self.n_topics}...")
        if best_alg == 'lda':
            model, W, H = self.fit_lda(X, self.n_topics)
        else:
            model, W, H = self.fit_nmf(X, self.n_topics)
            
        # Extract topics
        topics = self.get_top_terms(H, feature_names)
        doc_assignments = W.argmax(axis=1)
        
        # Calculate all metrics
        overall_stability, topic_stability = self.calculate_stability(
            X, feature_names, best_alg, n_seeds=20)
        umass, cv = self.calculate_coherence(topics)
        diversity = self.calculate_topic_diversity(topics)
        confidence = self.bootstrap_confidence(X, feature_names, best_alg, n_bootstrap=50)
        
        # Count documents per topic
        topic_counts = [np.sum(doc_assignments == i) for i in range(self.n_topics)]
        
        # Compile results
        final_results = {
            'algorithm': best_alg,
            'n_topics': self.n_topics,
            'topics': topics,
            'doc_assignments': doc_assignments,
            'topic_counts': topic_counts,
            'overall_stability': overall_stability,
            'topic_stability': topic_stability,
            'coherence_cv': cv,
            'coherence_umass': umass,
            'diversity': diversity,
            'bootstrap_confidence': confidence,
            'feature_names': feature_names,
            'k_selection': k_df,
            'algorithm_comparison': self.results.get('algorithm_comparison')
        }
        
        # Print summary
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        print(f"Algorithm: {best_alg.upper()}")
        print(f"Number of topics: {self.n_topics}")
        print(f"Overall stability: {overall_stability:.3f}")
        print(f"Coherence (C_v): {cv:.3f}")
        print(f"Coherence (U_mass): {umass:.3f}")
        print(f"Topic diversity: {diversity:.3f}")
        print(f"Mean bootstrap confidence: {confidence.mean():.3f}")
        
        print("\nTopics:")
        for i, topic in enumerate(topics):
            print(f"Topic {i+1} ({topic_counts[i]} docs, stability={topic_stability[i]:.2f}): "
                  f"{', '.join(topic[:10])}")
            
        # Save and visualize
        self.save_results(final_results)
        self.visualize_results(final_results)
        
        return final_results


if __name__ == "__main__":
    # Run the robust pipeline
    modeler = RobustTopicModeler(
        n_topics=5,
        preprocessing='phrasy',  # Try 'basic', 'enhanced', or 'phrasy'
        algorithm='both'  # Compare both LDA and NMF
    )
    
    results = modeler.run_full_pipeline()
    
    print("\n✓ Results saved to:", RESULTS_DIR)
    print("  - robust_topics.txt: Topic term lists")
    print("  - robust_metrics.csv: Performance metrics")
    print("  - robust_assignments.csv: Document-topic assignments")
    print("  - robust_topic_modeling_results.png: Visualizations")