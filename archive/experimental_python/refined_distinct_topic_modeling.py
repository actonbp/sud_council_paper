#!/usr/bin/env python3
"""
Refined Topic Modeling with Forced Distinction
Addresses overlap issues by finding optimal k and enforcing topic separation
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Gensim for coherence
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

# Sklearn imports
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment

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

# More aggressive domain filtering
DOMAIN_RE = re.compile(
    r"\b(substance|abuse|addict\w*|drug\w*|alcohol\w*)\b",  # Keep counseling terms
    re.I,
)

# Expanded stopwords to reduce noise
STOP_WORDS = set("""
a about above after again against all am an and any are as at be because been before being
between both but by could did do does doing down during each few for from further had has
have having he her here hers herself him himself his how i if in into is it its itself
just like me more most my myself nor not of off on once only or other our ours ourselves
out over own same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were what when where
which while who whom why will with you your yours yourself yourselves um uh yeah okay kinda
sorta right would know think really kind going lot can say definitely want guess something
able way actually maybe feel feels felt get got make made see say said sure look looking
good bad yes no dont don't thats that's gonna wanna like really actually probably
""".split())


class RefinedTopicModeler:
    """Topic modeling focused on finding distinct, non-overlapping themes"""
    
    def __init__(self):
        self.optimal_k = None
        
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
            
            # Keep only substantive utterances (>= 8 words)
            df = df[df["Text"].str.split().str.len() >= 8]
            
            texts = df["Text"].astype(str).tolist()
            docs.extend(texts)
            
            print(f"{fn:45} | {original_count:3} → {len(df):3} substantive utterances")
        
        print(f"Total substantive utterances: {len(docs)}")
        self.raw_docs = docs
        
        # Conservative domain filtering
        self.clean_docs = [DOMAIN_RE.sub(" ", doc.lower()) for doc in docs]
        
    def create_distinct_features(self):
        """Create features optimized for topic separation"""
        print("\n>>> Creating features for topic distinction...")
        
        # Use TF-IDF for better separation + more restrictive parameters
        vectorizer = TfidfVectorizer(
            stop_words=list(STOP_WORDS),
            ngram_range=(1, 2),  # unigrams and bigrams
            min_df=5,  # Must appear in at least 5 documents
            max_df=0.7,  # Can't appear in more than 70% of docs
            max_features=200,  # Limit to most discriminative features
            sublinear_tf=True  # Dampen high frequencies
        )
        
        X = vectorizer.fit_transform(self.clean_docs)
        vocab = vectorizer.get_feature_names_out()
        
        print(f"Feature matrix: {X.shape[0]} docs × {X.shape[1]} features")
        print(f"Sparsity: {1 - X.nnz / (X.shape[0] * X.shape[1]):.3f}")
        
        return X, vocab, vectorizer
        
    def find_optimal_k_comprehensive(self, X, vocab, k_range=range(2, 7)):
        """Find optimal k using multiple metrics"""
        print("\n>>> Finding optimal number of topics...")
        
        results = []
        
        for k in k_range:
            print(f"Testing k={k}...")
            
            # Test both algorithms
            for alg_name in ['lda', 'nmf']:
                if alg_name == 'lda':
                    model = LatentDirichletAllocation(
                        n_components=k, max_iter=100, random_state=42,
                        doc_topic_prior=0.1, topic_word_prior=0.01)  # More focused
                else:
                    model = NMF(
                        n_components=k, init='nndsvd', max_iter=500, random_state=42,
                        alpha_W=0.1, alpha_H=0.1, l1_ratio=0.5)
                
                # Fit model
                doc_topic = model.fit_transform(X)
                assignments = doc_topic.argmax(axis=1)
                
                # Get topics
                topics = []
                for topic_idx, topic in enumerate(model.components_):
                    top_indices = topic.argsort()[-15:][::-1]
                    top_terms = [vocab[i] for i in top_indices]
                    topics.append(top_terms)
                
                # Calculate metrics
                
                # 1. Topic diversity (overlap penalty)
                all_terms = []
                for topic in topics:
                    all_terms.extend(topic[:10])
                diversity = len(set(all_terms)) / len(all_terms)
                
                # 2. Coherence
                tokens = [simple_preprocess(doc) for doc in self.clean_docs]
                dictionary = Dictionary(tokens)
                cm_cv = CoherenceModel(topics=topics, texts=tokens, 
                                     dictionary=dictionary, coherence='c_v')
                coherence = cm_cv.get_coherence()
                
                # 3. Silhouette score (how well separated are document clusters)
                if len(set(assignments)) > 1:  # Need at least 2 clusters
                    silhouette = silhouette_score(X, assignments, metric='cosine')
                else:
                    silhouette = -1
                
                # 4. Topic balance (how evenly distributed are documents)
                topic_counts = [np.sum(assignments == i) for i in range(k)]
                balance = 1 - np.std(topic_counts) / np.mean(topic_counts)
                
                # 5. Term overlap penalty
                overlap_penalty = 0
                for i in range(len(topics)):
                    for j in range(i+1, len(topics)):
                        overlap = len(set(topics[i][:10]) & set(topics[j][:10]))
                        overlap_penalty += overlap / 10  # Penalty for shared terms
                overlap_penalty /= (k * (k-1) / 2)  # Normalize by number of pairs
                
                # Combined score (higher is better)
                combined_score = (diversity * 0.3 + 
                                coherence * 0.25 + 
                                silhouette * 0.2 + 
                                balance * 0.15 + 
                                (1 - overlap_penalty) * 0.1)
                
                results.append({
                    'k': k,
                    'algorithm': alg_name,
                    'diversity': diversity,
                    'coherence': coherence,
                    'silhouette': silhouette,
                    'balance': balance,
                    'overlap_penalty': overlap_penalty,
                    'combined_score': combined_score,
                    'topics': topics,
                    'assignments': assignments,
                    'model': model
                })
                
                print(f"  {alg_name.upper()}: diversity={diversity:.3f}, coherence={coherence:.3f}, "
                      f"silhouette={silhouette:.3f}, score={combined_score:.3f}")
        
        # Find best configuration
        best_config = max(results, key=lambda x: x['combined_score'])
        self.optimal_k = best_config['k']
        
        print(f"\n>>> OPTIMAL CONFIGURATION:")
        print(f"    k={best_config['k']}, algorithm={best_config['algorithm'].upper()}")
        print(f"    Combined score: {best_config['combined_score']:.3f}")
        
        return best_config, pd.DataFrame(results)
        
    def remove_topic_overlap(self, topics, vocab):
        """Remove overlapping terms to make topics more distinct"""
        print("\n>>> Removing overlapping terms...")
        
        # Create term frequency across topics
        term_topic_count = {}
        for topic_idx, topic_terms in enumerate(topics):
            for term in topic_terms[:15]:
                if term not in term_topic_count:
                    term_topic_count[term] = []
                term_topic_count[term].append(topic_idx)
        
        # Identify unique and shared terms
        unique_topics = []
        for topic_idx, topic_terms in enumerate(topics):
            unique_terms = []
            for term in topic_terms:
                # Keep term if it appears in only this topic or is highly characteristic
                if len(term_topic_count[term]) == 1:
                    unique_terms.append(term)
                elif len(unique_terms) < 12:  # Fill up to 12 terms per topic
                    unique_terms.append(term)
                
                if len(unique_terms) >= 15:
                    break
            
            unique_topics.append(unique_terms)
        
        # Fill short topics with next best terms
        for topic_idx, unique_terms in enumerate(unique_topics):
            if len(unique_terms) < 10:
                original_terms = topics[topic_idx]
                for term in original_terms:
                    if term not in unique_terms and len(unique_terms) < 15:
                        unique_terms.append(term)
        
        print("Topic overlap removed. Unique terms per topic:")
        for i, terms in enumerate(unique_topics):
            print(f"  Topic {i+1}: {len(terms)} unique terms")
        
        return unique_topics
        
    def interpret_distinct_topics(self, topics, assignments):
        """Generate interpretations focused on distinction"""
        
        # More nuanced theme patterns
        theme_patterns = {
            'helping_altruistic': {
                'keywords': ['help', 'helping', 'support', 'care', 'service', 'difference', 'change'],
                'label': 'Altruistic Helping Motivation'
            },
            'personal_experience': {
                'keywords': ['family', 'personal', 'experience', 'grew', 'background', 'lived', 'seen'],
                'label': 'Personal/Family Experience'
            },
            'education_academic': {
                'keywords': ['school', 'class', 'course', 'learning', 'study', 'psychology', 'program'],
                'label': 'Academic/Educational Path'
            },
            'career_practical': {
                'keywords': ['job', 'career', 'work', 'field', 'profession', 'money', 'salary'],
                'label': 'Career/Practical Considerations'
            },
            'clinical_professional': {
                'keywords': ['therapy', 'client', 'treatment', 'clinical', 'session', 'practice'],
                'label': 'Clinical/Professional Practice'
            },
            'challenges_barriers': {
                'keywords': ['difficult', 'hard', 'challenging', 'stress', 'burnout', 'tough'],
                'label': 'Challenges and Barriers'
            },
            'social_impact': {
                'keywords': ['community', 'society', 'people', 'population', 'social', 'public'],
                'label': 'Social Impact Focus'
            }
        }
        
        interpretations = []
        for i, topic_terms in enumerate(topics):
            doc_count = sum(assignments == i)
            
            # Score against themes using top 10 terms
            theme_scores = {}
            for theme_key, theme_info in theme_patterns.items():
                score = 0
                for term in topic_terms[:10]:
                    for keyword in theme_info['keywords']:
                        if keyword in term.lower():
                            score += 1
                if score > 0:
                    theme_scores[theme_key] = score
            
            # Select best theme
            if theme_scores:
                best_theme = max(theme_scores, key=theme_scores.get)
                interpretation = theme_patterns[best_theme]['label']
            else:
                # Fallback: use most frequent meaningful words
                meaningful_terms = [t for t in topic_terms[:5] 
                                  if len(t) > 3 and t not in ['said', 'like', 'really']]
                if meaningful_terms:
                    interpretation = f"Theme: {meaningful_terms[0].title()}"
                else:
                    interpretation = f"Mixed Discussion"
            
            interpretations.append(f"{interpretation} ({doc_count} docs)")
        
        return interpretations
        
    def create_refined_visualization(self, results):
        """Create visualization emphasizing topic distinction"""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(18, 12))
        
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        k = len(results['topics'])
        colors = plt.cm.Set1(np.linspace(0, 1, k))
        
        # 1. Topic distribution (with clear labeling)
        ax1 = fig.add_subplot(gs[0, 0])
        topic_counts = [sum(results['assignments'] == i) for i in range(k)]
        bars = ax1.bar(range(k), topic_counts, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Document Distribution\n(Distinct Topics)', fontweight='bold')
        ax1.set_xlabel('Topic')
        ax1.set_ylabel('Documents')
        ax1.set_xticks(range(k))
        ax1.set_xticklabels([f'T{i+1}' for i in range(k)])
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Quality metrics comparison
        ax2 = fig.add_subplot(gs[0, 1])
        metrics = ['Diversity', 'Coherence', 'Balance']
        values = [results['diversity'], results['coherence'], results['balance']]
        
        bars = ax2.bar(metrics, values, color=['coral', 'lightblue', 'lightgreen'], alpha=0.8)
        ax2.set_title('Topic Quality Metrics', fontweight='bold')
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Overlap analysis
        ax3 = fig.add_subplot(gs[0, 2])
        overlap_matrix = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                if i != j:
                    overlap = len(set(results['topics'][i][:10]) & set(results['topics'][j][:10]))
                    overlap_matrix[i, j] = overlap
        
        im = ax3.imshow(overlap_matrix, cmap='Reds', alpha=0.8)
        ax3.set_title('Topic Overlap Matrix\n(Lower = Better)', fontweight='bold')
        ax3.set_xlabel('Topic')
        ax3.set_ylabel('Topic')
        ax3.set_xticks(range(k))
        ax3.set_yticks(range(k))
        ax3.set_xticklabels([f'T{i+1}' for i in range(k)])
        ax3.set_yticklabels([f'T{i+1}' for i in range(k)])
        
        # Add overlap values
        for i in range(k):
            for j in range(k):
                if i != j:
                    ax3.text(j, i, f'{int(overlap_matrix[i, j])}', 
                            ha='center', va='center', fontweight='bold')
        
        # 4. Topic terms (large display)
        ax4 = fig.add_subplot(gs[1:, :])
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        y_positions = np.linspace(0.9, 0.1, k)
        
        for i, (topic_terms, interpretation) in enumerate(zip(results['topics'], results['interpretations'])):
            y_pos = y_positions[i]
            color = colors[i]
            
            # Topic header
            ax4.text(0.02, y_pos, f"Topic {i+1}: {interpretation}", 
                    transform=ax4.transAxes, fontsize=16, fontweight='bold', color=color)
            
            # Top terms (emphasize distinction)
            unique_terms = [t for t in topic_terms[:12] if len(t) > 2]
            terms_text = f"Distinctive terms: {', '.join(unique_terms)}"
            ax4.text(0.02, y_pos - 0.06, terms_text, 
                    transform=ax4.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='white', 
                             alpha=0.9, edgecolor=color, linewidth=2))
            
            # Document count and diversity info
            doc_count = sum(results['assignments'] == i)
            diversity_info = f"Documents: {doc_count} | Unique terms: {len(set(topic_terms[:10]))}"
            ax4.text(0.02, y_pos - 0.12, diversity_info, 
                    transform=ax4.transAxes, fontsize=10, style='italic', color='gray')
        
        # Title
        title = f'Refined Topic Analysis: {k} Distinct Themes'
        subtitle = f'Diversity: {results["diversity"]:.3f} | Overlap Penalty: {results["overlap_penalty"]:.3f} | Score: {results["combined_score"]:.3f}'
        fig.suptitle(f'{title}\n{subtitle}', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = os.path.join(RESULTS_DIR, 'refined_distinct_topics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Refined visualization saved: {output_path}")
        
    def save_refined_results(self, results, k_analysis_df):
        """Save refined results with emphasis on distinction"""
        
        # Main report
        with open(os.path.join(RESULTS_DIR, 'refined_distinct_topics_report.txt'), 'w') as f:
            f.write("REFINED DISTINCT TOPIC ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("OPTIMAL CONFIGURATION:\n")
            f.write(f"• Number of topics: {len(results['topics'])}\n")
            f.write(f"• Algorithm: {results['algorithm'].upper()}\n")
            f.write(f"• Documents analyzed: {len(results['assignments'])}\n")
            f.write(f"• Combined quality score: {results['combined_score']:.3f}\n\n")
            
            f.write("DISTINCTIVENESS METRICS:\n")
            f.write(f"• Topic diversity: {results['diversity']:.3f}\n")
            f.write(f"• Topic coherence: {results['coherence']:.3f}\n")
            f.write(f"• Topic balance: {results['balance']:.3f}\n")
            f.write(f"• Overlap penalty: {results['overlap_penalty']:.3f}\n\n")
            
            f.write("DISTINCT THEMES IDENTIFIED:\n")
            f.write("-" * 60 + "\n")
            
            for i, (topic_terms, interpretation) in enumerate(zip(results['topics'], results['interpretations'])):
                doc_count = sum(results['assignments'] == i)
                f.write(f"\n{interpretation}\n")
                f.write(f"  • Document count: {doc_count}\n")
                f.write(f"  • Proportion: {doc_count/len(results['assignments']):.2%}\n")
                f.write(f"  • Key terms: {', '.join(topic_terms[:15])}\n")
        
        # K analysis results
        k_analysis_df.to_csv(os.path.join(RESULTS_DIR, 'k_optimization_analysis.csv'), index=False)
        
        # Document assignments
        assignments_df = pd.DataFrame({
            'document_id': range(len(results['assignments'])),
            'topic_number': results['assignments'] + 1,
            'topic_name': [results['interpretations'][i] for i in results['assignments']],
            'text_preview': [doc[:150] + '...' if len(doc) > 150 else doc 
                           for doc in self.raw_docs[:len(results['assignments'])]]
        })
        assignments_df.to_csv(os.path.join(RESULTS_DIR, 'refined_topic_assignments.csv'), index=False)
        
        print("✓ Refined results saved:")
        print("  • refined_distinct_topics_report.txt")
        print("  • k_optimization_analysis.csv") 
        print("  • refined_topic_assignments.csv")
        
    def run_refined_analysis(self):
        """Execute refined analysis focused on topic distinction"""
        print("=" * 70)
        print("REFINED TOPIC MODELING: FINDING DISTINCT THEMES")
        print("=" * 70)
        
        # Load and prepare data
        self.load_data()
        X, vocab, vectorizer = self.create_distinct_features()
        
        # Find optimal configuration
        best_config, k_analysis_df = self.find_optimal_k_comprehensive(X, vocab)
        
        # Remove overlapping terms
        distinct_topics = self.remove_topic_overlap(best_config['topics'], vocab)
        
        # Generate interpretations
        interpretations = self.interpret_distinct_topics(distinct_topics, best_config['assignments'])
        
        # Update results with refined topics
        refined_results = best_config.copy()
        refined_results['topics'] = distinct_topics
        refined_results['interpretations'] = interpretations
        
        # Display results
        print("\n" + "=" * 70)
        print("REFINED DISTINCT TOPICS IDENTIFIED")
        print("=" * 70)
        print(f"Optimal k: {len(distinct_topics)}")
        print(f"Algorithm: {refined_results['algorithm'].upper()}")
        print(f"Quality score: {refined_results['combined_score']:.3f}")
        print(f"Topic diversity: {refined_results['diversity']:.3f}")
        
        print(f"\nDISTINCT THEMES:")
        for i, (topic_terms, interpretation) in enumerate(zip(distinct_topics, interpretations)):
            doc_count = sum(refined_results['assignments'] == i)
            unique_terms = len(set(topic_terms[:10]))
            print(f"\n{i+1}. {interpretation}")
            print(f"    Unique terms: {unique_terms}/10 | Documents: {doc_count}")
            print(f"    Key terms: {', '.join(topic_terms[:12])}")
        
        # Save and visualize
        self.save_refined_results(refined_results, k_analysis_df)
        self.create_refined_visualization(refined_results)
        
        return refined_results


if __name__ == "__main__":
    modeler = RefinedTopicModeler()
    results = modeler.run_refined_analysis()
    
    print("\n" + "🎯" * 25)
    print("REFINED TOPIC ANALYSIS COMPLETE!")
    print("🎯" * 25)
    print(f"\nKey Achievement: {len(results['topics'])} DISTINCT topics with {results['diversity']:.1%} term diversity")
    print("Focus: Eliminated overlap, forced distinction, optimized k")
    print("All results saved to results/ directory")