#!/usr/bin/env python3
"""
Final Robust Topic Modeling for SUD Focus Groups
Based on working parameters with comprehensive validation
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Gensim for coherence
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

# Sklearn imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
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

# Domain-specific filtering (same as working script)
DOMAIN_RE = re.compile(
    r"\b(substance|abuse|addict\w*|drug\w*|alcohol\w*|counsel\w*|mental\s+health)\b",
    re.I,
)

# Working stopwords list (from successful script)
STOP_WORDS = set("""
a about above after again against all am an and any are as at be because been before being
between both but by could did do does doing down during each few for from further had has
have having he her here hers herself him himself his how i if in into is it its itself
just like me more most my myself nor not of off on once only or other our ours ourselves
out over own same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were what when where
which while who whom why will with you your yours yourself yourselves um uh yeah okay kinda
sorta right would know think really kind going lot can say definitely want guess something
able way actually maybe feel feels felt
""".split())


class FinalRobustTopicModeler:
    """Final robust topic modeling using proven working parameters"""
    
    def __init__(self, n_topics=5):
        self.n_topics = n_topics
        
    def load_data(self):
        """Load CSV files using working approach"""
        docs = []
        file_info = []
        
        print(">>> Loading focus group data...")
        
        for fn in CSV_FILES:
            fp = os.path.join(DATA_DIR, fn)
            df = pd.read_csv(fp)
            original_count = len(df)
            
            # Remove moderator rows (working approach)
            df = df[~df["Speaker"].astype(str).str.match(r"^[A-Z]{2,3}$")]
            df = df[df["Text"].notna()]
            
            texts = df["Text"].astype(str).tolist()
            docs.extend(texts)
            
            file_info.append({
                'file': fn,
                'original': original_count,
                'filtered': len(df),
                'utterances': len(texts)
            })
            
            print(f"{fn:45} | {original_count:3} → {len(df):3} utterances")
        
        print(f"Total utterances: {len(docs)}")
        self.raw_docs = docs
        
        # Clean text (working approach)
        self.clean_docs = [DOMAIN_RE.sub("", doc.lower()) for doc in docs]
        
        return file_info
        
    def create_working_features(self):
        """Create feature matrix using proven working parameters"""
        print("\n>>> Creating feature matrix with working parameters...")
        
        # Use exact parameters from working script
        vectorizer = CountVectorizer(
            stop_words=list(STOP_WORDS),
            ngram_range=(1, 2),  # unigrams + bigrams
            min_df=3
        )
        
        X = vectorizer.fit_transform(self.clean_docs)
        vocab = vectorizer.get_feature_names_out()
        
        print(f"Vocabulary size: {len(vocab)} terms")
        print(f"Document-term matrix: {X.shape}")
        print(f"Matrix sparsity: {1 - X.nnz / (X.shape[0] * X.shape[1]):.3f}")
        
        return X, vocab, vectorizer
        
    def fit_robust_lda(self, X, random_state=42):
        """Fit LDA with working parameters"""
        lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=500,
            learning_method="batch",  # Working parameter
            random_state=random_state
        )
        doc_topic = lda.fit_transform(X)
        return lda, doc_topic
        
    def fit_robust_nmf(self, X, random_state=42):
        """Fit NMF for comparison"""
        nmf = NMF(
            n_components=self.n_topics,
            init='nndsvd',
            max_iter=500,
            random_state=random_state
        )
        doc_topic = nmf.fit_transform(X)
        return nmf, doc_topic
        
    def get_top_terms(self, model, vocab, n_terms=15):
        """Extract top terms per topic"""
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            top_indices = topic.argsort()[-n_terms:][::-1]
            top_terms = [vocab[i] for i in top_indices]
            topics.append(top_terms)
        return topics
        
    def calculate_stability_robust(self, X, vocab, algorithm='lda', n_seeds=20):
        """Comprehensive stability testing"""
        print(f">>> Stability analysis across {n_seeds} seeds for {algorithm.upper()}...")
        
        reference_topics = None
        jaccard_scores = []
        
        for seed in range(n_seeds):
            if algorithm == 'lda':
                model, _ = self.fit_robust_lda(X, seed)
            else:
                model, _ = self.fit_robust_nmf(X, seed)
                
            topics = self.get_top_terms(model, vocab)
            
            if seed == 0:
                reference_topics = topics
            else:
                # Jaccard similarity matrix
                jaccard_matrix = np.zeros((self.n_topics, self.n_topics))
                for i, ref_topic in enumerate(reference_topics):
                    ref_set = set(ref_topic[:10])
                    for j, test_topic in enumerate(topics):
                        test_set = set(test_topic[:10])
                        if len(ref_set | test_set) > 0:
                            jaccard = len(ref_set & test_set) / len(ref_set | test_set)
                        else:
                            jaccard = 0
                        jaccard_matrix[i, j] = jaccard
                
                # Hungarian algorithm for optimal matching
                row_ind, col_ind = linear_sum_assignment(-jaccard_matrix)
                seed_scores = [jaccard_matrix[i, j] for i, j in zip(row_ind, col_ind)]
                jaccard_scores.append(seed_scores)
        
        if jaccard_scores:
            jaccard_array = np.array(jaccard_scores)
            topic_stability = jaccard_array.mean(axis=0)
            overall_stability = topic_stability.mean()
        else:
            topic_stability = np.ones(self.n_topics)
            overall_stability = 1.0
            
        return overall_stability, topic_stability
        
    def calculate_coherence_robust(self, topics):
        """Calculate coherence using Gensim"""
        # Tokenize for gensim
        tokens = [simple_preprocess(doc) for doc in self.clean_docs]
        dictionary = Dictionary(tokens)
        
        # Calculate both coherence measures
        cm_umass = CoherenceModel(
            topics=topics, texts=tokens, dictionary=dictionary, coherence='u_mass')
        cm_cv = CoherenceModel(
            topics=topics, texts=tokens, dictionary=dictionary, coherence='c_v')
        
        return cm_umass.get_coherence(), cm_cv.get_coherence()
        
    def calculate_diversity(self, topics):
        """Topic diversity metric"""
        all_terms = []
        for topic in topics:
            all_terms.extend(topic[:10])
        unique_terms = len(set(all_terms))
        total_terms = len(all_terms)
        return unique_terms / total_terms if total_terms > 0 else 0
        
    def interpret_topics_robust(self, topics, doc_assignments):
        """Robust topic interpretation"""
        
        # Extended theme mapping for better interpretation
        theme_patterns = {
            'helping_service': {
                'keywords': ['help', 'helping', 'support', 'care', 'service', 'assist', 'give_back', 'difference'],
                'label': 'Helping & Service Orientation'
            },
            'personal_experience': {
                'keywords': ['family', 'personal', 'experience', 'life', 'grew', 'background', 'story', 'lived'],
                'label': 'Personal & Family Experience'
            },
            'education_training': {
                'keywords': ['school', 'class', 'course', 'learn', 'education', 'training', 'program', 'study'],
                'label': 'Education & Training'
            },
            'career_professional': {
                'keywords': ['career', 'job', 'work', 'profession', 'field', 'professional', 'opportunity'],
                'label': 'Career & Professional Aspects'
            },
            'clinical_practice': {
                'keywords': ['clinical', 'therapy', 'client', 'patient', 'session', 'practice', 'treatment'],
                'label': 'Clinical Practice'
            },
            'emotional_challenges': {
                'keywords': ['stress', 'burnout', 'emotional', 'difficult', 'challenging', 'tough', 'hard'],
                'label': 'Emotional Challenges & Burnout'
            },
            'requirements_logistics': {
                'keywords': ['requirements', 'license', 'certification', 'hours', 'supervised', 'credential'],
                'label': 'Requirements & Logistics'
            }
        }
        
        interpretations = []
        for i, topic_terms in enumerate(topics):
            doc_count = sum(doc_assignments == i)
            
            # Score against theme patterns
            theme_scores = {}
            for theme_key, theme_info in theme_patterns.items():
                score = 0
                for term in topic_terms[:12]:  # Check top 12 terms
                    for keyword in theme_info['keywords']:
                        if keyword in term.lower():
                            score += 1
                if score > 0:
                    theme_scores[theme_key] = score
            
            # Select best theme
            if theme_scores:
                best_theme_key = max(theme_scores, key=theme_scores.get)
                interpretation = theme_patterns[best_theme_key]['label']
            else:
                interpretation = "General Discussion"
                
            interpretations.append(f"{interpretation} ({doc_count} docs)")
            
        return interpretations
        
    def create_final_visualization(self, results):
        """Create publication-quality visualization"""
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # Grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)
        
        # Color scheme
        colors = plt.cm.Set2(np.linspace(0, 1, self.n_topics))
        
        # 1. Topic distribution
        ax1 = fig.add_subplot(gs[0, 0])
        topic_counts = results['topic_counts']
        bars = ax1.bar(range(len(topic_counts)), topic_counts, color=colors, alpha=0.8)
        ax1.set_title('Documents per Topic', fontsize=12, weight='bold')
        ax1.set_xlabel('Topic')
        ax1.set_ylabel('Document Count')
        ax1.set_xticks(range(len(topic_counts)))
        ax1.set_xticklabels([f'T{i+1}' for i in range(len(topic_counts))])
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Stability scores
        ax2 = fig.add_subplot(gs[0, 1])
        stability = results['topic_stability']
        bars = ax2.bar(range(len(stability)), stability, color=colors, alpha=0.8)
        ax2.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='Good threshold')
        ax2.set_title('Topic Stability', fontsize=12, weight='bold')
        ax2.set_xlabel('Topic')
        ax2.set_ylabel('Jaccard Similarity')
        ax2.set_xticks(range(len(stability)))
        ax2.set_xticklabels([f'T{i+1}' for i in range(len(stability))])
        ax2.legend()
        ax2.set_ylim(0, 1.05)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Algorithm comparison
        ax3 = fig.add_subplot(gs[0, 2])
        if 'comparison' in results:
            comp = results['comparison']
            algorithms = list(comp.keys())
            metrics = ['Stability', 'Coherence', 'Diversity']
            
            x = np.arange(len(algorithms))
            width = 0.25
            
            stab_vals = [comp[alg]['stability'] for alg in algorithms]
            coh_vals = [comp[alg]['coherence_cv'] for alg in algorithms] 
            div_vals = [comp[alg]['diversity'] for alg in algorithms]
            
            ax3.bar(x - width, stab_vals, width, label='Stability', alpha=0.8)
            ax3.bar(x, coh_vals, width, label='Coherence', alpha=0.8)
            ax3.bar(x + width, div_vals, width, label='Diversity', alpha=0.8)
            
            ax3.set_title('Algorithm Comparison', fontsize=12, weight='bold')
            ax3.set_xlabel('Algorithm')
            ax3.set_ylabel('Score')
            ax3.set_xticks(x)
            ax3.set_xticklabels([alg.upper() for alg in algorithms])
            ax3.legend()
        
        # 4. Quality metrics radar
        ax4 = fig.add_subplot(gs[0, 3])
        metrics_names = ['Stability', 'Coherence\n(C_v)', 'Diversity']
        metrics_vals = [results['overall_stability'], results['coherence_cv'], results['diversity']]
        
        bars = ax4.bar(metrics_names, metrics_vals, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.8)
        ax4.set_title('Overall Quality Metrics', fontsize=12, weight='bold')
        ax4.set_ylabel('Score')
        ax4.set_ylim(0, 1)
        for bar, val in zip(bars, metrics_vals):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Topic terms matrix (large section)
        ax5 = fig.add_subplot(gs[1:3, :])
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        
        # Display each topic with interpretation
        y_positions = np.linspace(0.95, 0.05, len(results['topics']))
        
        for i, (topic, interp) in enumerate(zip(results['topics'], results['interpretations'])):
            y_pos = y_positions[i]
            color = colors[i]
            
            # Topic header with interpretation
            header_text = f"Topic {i+1}: {interp}"
            ax5.text(0.02, y_pos, header_text, transform=ax5.transAxes,
                    fontsize=14, weight='bold', color=color)
            
            # Stability and document info
            info_text = f"Stability: {results['topic_stability'][i]:.3f} | Documents: {results['topic_counts'][i]}"
            ax5.text(0.02, y_pos - 0.04, info_text, transform=ax5.transAxes,
                    fontsize=11, color='gray', style='italic')
            
            # Top terms (formatted nicely)
            terms_text = f"Key terms: {', '.join(topic[:18])}"
            ax5.text(0.02, y_pos - 0.08, terms_text, transform=ax5.transAxes,
                    fontsize=11, wrap=True,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor=color))
        
        # 6. Summary statistics (bottom section)
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        
        summary_text = f"""
        COMPREHENSIVE TOPIC MODELING RESULTS SUMMARY
        
        📊 METHODOLOGY: {results['algorithm'].upper()} with {results['n_topics']} topics | {len(results['doc_assignments'])} documents | {len(results['feature_names'])} features
        
        🔍 QUALITY ASSESSMENT:
        • Overall Stability: {results['overall_stability']:.3f} (consistency across random seeds)
        • Topic Coherence (C_v): {results['coherence_cv']:.3f} (semantic meaningfulness) 
        • Topic Coherence (UMass): {results['coherence_umass']:.3f} (intrinsic coherence)
        • Topic Diversity: {results['diversity']:.3f} (term uniqueness across topics)
        
        ✅ ROBUSTNESS: {'EXCELLENT' if results['overall_stability'] > 0.8 else 'GOOD' if results['overall_stability'] > 0.6 else 'MODERATE'}
        📈 COHERENCE: {'HIGH' if results['coherence_cv'] > 0.5 else 'MODERATE' if results['coherence_cv'] > 0.3 else 'LOW'}
        🎯 SEPARATION: {'EXCELLENT' if results['diversity'] > 0.8 else 'GOOD' if results['diversity'] > 0.6 else 'MODERATE'}
        """
        
        ax6.text(0.02, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.6", facecolor='lightgray', alpha=0.9))
        
        # Main title
        title = f'Robust Topic Modeling Analysis: SUD Counseling Focus Groups'
        subtitle = f'Algorithm: {results["algorithm"].upper()} | Quality Score: {(results["overall_stability"] + results["coherence_cv"] + results["diversity"])/3:.3f}'
        fig.suptitle(f'{title}\n{subtitle}', fontsize=18, y=0.98, weight='bold')
        
        plt.tight_layout()
        
        # Save high-quality version
        output_path = os.path.join(RESULTS_DIR, 'final_robust_topic_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ High-quality visualization saved: {output_path}")
        
    def save_comprehensive_results(self, results):
        """Save all results comprehensively"""
        
        # 1. Main analysis report
        with open(os.path.join(RESULTS_DIR, 'final_robust_analysis_report.txt'), 'w') as f:
            f.write("FINAL ROBUST TOPIC MODELING ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY:\n")
            f.write(f"• Algorithm: {results['algorithm'].upper()}\n")
            f.write(f"• Topics identified: {results['n_topics']}\n")
            f.write(f"• Documents analyzed: {len(results['doc_assignments'])}\n")
            f.write(f"• Vocabulary features: {len(results['feature_names'])}\n")
            f.write(f"• Overall quality score: {(results['overall_stability'] + results['coherence_cv'] + results['diversity'])/3:.3f}\n\n")
            
            f.write("ROBUSTNESS METRICS:\n")
            f.write(f"• Stability (seed consistency): {results['overall_stability']:.3f}\n")
            f.write(f"• Coherence (semantic): {results['coherence_cv']:.3f}\n")
            f.write(f"• Coherence (intrinsic): {results['coherence_umass']:.3f}\n")
            f.write(f"• Topic diversity: {results['diversity']:.3f}\n\n")
            
            f.write("IDENTIFIED THEMES:\n")
            f.write("-" * 80 + "\n")
            
            for i, (topic, interp) in enumerate(zip(results['topics'], results['interpretations'])):
                f.write(f"\n{interp}\n")
                f.write(f"  • Stability score: {results['topic_stability'][i]:.3f}\n")
                f.write(f"  • Document count: {results['topic_counts'][i]}\n")
                f.write(f"  • Representative terms: {', '.join(topic[:20])}\n")
                
        # 2. Detailed metrics CSV
        metrics_data = []
        
        # Overall metrics
        overall_metrics = [
            ('algorithm', results['algorithm']),
            ('n_topics', results['n_topics']),
            ('total_documents', len(results['doc_assignments'])),
            ('vocabulary_size', len(results['feature_names'])),
            ('overall_stability', results['overall_stability']),
            ('coherence_cv', results['coherence_cv']),
            ('coherence_umass', results['coherence_umass']),
            ('topic_diversity', results['diversity']),
            ('quality_score', (results['overall_stability'] + results['coherence_cv'] + results['diversity'])/3)
        ]
        
        for metric, value in overall_metrics:
            metrics_data.append({
                'metric_type': 'overall',
                'metric_name': metric,
                'value': value,
                'topic_id': None
            })
            
        # Topic-specific metrics
        for i in range(results['n_topics']):
            topic_metrics = [
                (f'topic_{i+1}_stability', results['topic_stability'][i]),
                (f'topic_{i+1}_document_count', results['topic_counts'][i]),
                (f'topic_{i+1}_proportion', results['topic_counts'][i] / len(results['doc_assignments']))
            ]
            
            for metric, value in topic_metrics:
                metrics_data.append({
                    'metric_type': 'topic_specific',
                    'metric_name': metric,
                    'value': value,
                    'topic_id': i + 1
                })
        
        pd.DataFrame(metrics_data).to_csv(
            os.path.join(RESULTS_DIR, 'final_comprehensive_metrics.csv'), index=False)
        
        # 3. Document assignments with full details
        assignments_data = []
        for i, doc_assignment in enumerate(results['doc_assignments']):
            assignments_data.append({
                'document_id': i + 1,
                'topic_number': doc_assignment + 1,
                'topic_name': results['interpretations'][doc_assignment],
                'topic_stability': results['topic_stability'][doc_assignment],
                'text_preview': self.raw_docs[i][:200] + '...' if len(self.raw_docs[i]) > 200 else self.raw_docs[i],
                'text_length': len(self.raw_docs[i]),
                'cleaned_text_preview': self.clean_docs[i][:200] + '...' if len(self.clean_docs[i]) > 200 else self.clean_docs[i]
            })
            
        pd.DataFrame(assignments_data).to_csv(
            os.path.join(RESULTS_DIR, 'final_document_assignments.csv'), index=False)
        
        # 4. Topic terms with weights (if available)
        topic_terms_data = []
        for topic_id, topic_terms in enumerate(results['topics']):
            for rank, term in enumerate(topic_terms):
                topic_terms_data.append({
                    'topic_id': topic_id + 1,
                    'topic_name': results['interpretations'][topic_id],
                    'term_rank': rank + 1,
                    'term': term,
                    'topic_stability': results['topic_stability'][topic_id]
                })
                
        pd.DataFrame(topic_terms_data).to_csv(
            os.path.join(RESULTS_DIR, 'final_topic_terms.csv'), index=False)
        
        print("✓ Comprehensive results saved:")
        print("  • final_robust_analysis_report.txt - Executive summary")
        print("  • final_comprehensive_metrics.csv - All quality metrics")
        print("  • final_document_assignments.csv - Document-topic mapping")
        print("  • final_topic_terms.csv - Topic terms with rankings")
        
    def run_complete_analysis(self):
        """Execute the complete robust analysis pipeline"""
        print("=" * 80)
        print("FINAL ROBUST TOPIC MODELING ANALYSIS")
        print("=" * 80)
        
        # Load data
        file_info = self.load_data()
        
        # Create features using working parameters
        X, vocab, vectorizer = self.create_working_features()
        
        # Test both algorithms
        print("\n>>> Algorithm Comparison & Selection")
        comparison = {}
        
        for alg_name, alg_func in [('lda', self.fit_robust_lda), ('nmf', self.fit_robust_nmf)]:
            print(f"\nEvaluating {alg_name.upper()}...")
            
            # Fit model
            model, doc_topic = alg_func(X)
            topics = self.get_top_terms(model, vocab)
            doc_assignments = doc_topic.argmax(axis=1)
            
            # Calculate comprehensive metrics
            overall_stability, topic_stability = self.calculate_stability_robust(X, vocab, alg_name)
            umass, cv = self.calculate_coherence_robust(topics)
            diversity = self.calculate_diversity(topics)
            
            comparison[alg_name] = {
                'model': model,
                'doc_topic': doc_topic,
                'topics': topics,
                'doc_assignments': doc_assignments,
                'stability': overall_stability,
                'topic_stability': topic_stability,
                'coherence_cv': cv,
                'coherence_umass': umass,
                'diversity': diversity,
                'quality_score': (overall_stability + cv + diversity) / 3
            }
            
            print(f"  Results: Stability={overall_stability:.3f} | Coherence={cv:.3f} | Diversity={diversity:.3f}")
        
        # Select best algorithm
        best_alg = max(comparison, key=lambda x: comparison[x]['quality_score'])
        print(f"\n>>> SELECTED ALGORITHM: {best_alg.upper()}")
        print(f"    Quality Score: {comparison[best_alg]['quality_score']:.3f}")
        
        # Extract final results
        best_results = comparison[best_alg]
        topic_counts = [np.sum(best_results['doc_assignments'] == i) for i in range(self.n_topics)]
        
        # Generate interpretations
        interpretations = self.interpret_topics_robust(
            best_results['topics'], best_results['doc_assignments'])
        
        # Compile final results
        final_results = {
            'algorithm': best_alg,
            'n_topics': self.n_topics,
            'topics': best_results['topics'],
            'doc_assignments': best_results['doc_assignments'],
            'topic_counts': topic_counts,
            'overall_stability': best_results['stability'],
            'topic_stability': best_results['topic_stability'],
            'coherence_cv': best_results['coherence_cv'],
            'coherence_umass': best_results['coherence_umass'],
            'diversity': best_results['diversity'],
            'interpretations': interpretations,
            'feature_names': vocab,
            'comparison': comparison,
            'file_info': file_info
        }
        
        # Display final summary
        print("\n" + "=" * 80)
        print("FINAL ANALYSIS RESULTS")
        print("=" * 80)
        print(f"📊 Selected Method: {best_alg.upper()}")
        print(f"🎯 Overall Quality Score: {final_results['quality_score']:.3f}" if 'quality_score' in final_results else f"🎯 Quality Score: {(final_results['overall_stability'] + final_results['coherence_cv'] + final_results['diversity'])/3:.3f}")
        print(f"🔄 Stability: {final_results['overall_stability']:.3f}")
        print(f"🧠 Coherence: {final_results['coherence_cv']:.3f}")
        print(f"🌟 Diversity: {final_results['diversity']:.3f}")
        
        print(f"\n📋 IDENTIFIED THEMES:")
        for i, interp in enumerate(interpretations):
            stability = final_results['topic_stability'][i]
            print(f"   {i+1}. {interp} [Stability: {stability:.3f}]")
        
        # Save comprehensive results
        self.save_comprehensive_results(final_results)
        self.create_final_visualization(final_results)
        
        return final_results


if __name__ == "__main__":
    # Execute final robust analysis
    modeler = FinalRobustTopicModeler(n_topics=5)
    results = modeler.run_complete_analysis()
    
    print("\n" + "🎉" * 25)
    print("ROBUST TOPIC MODELING ANALYSIS COMPLETE!")
    print("🎉" * 25)
    print(f"\nFinal Quality Assessment:")
    quality_score = (results['overall_stability'] + results['coherence_cv'] + results['diversity']) / 3
    print(f"• Overall Quality Score: {quality_score:.3f}")
    print(f"• Robustness Level: {'EXCELLENT' if quality_score > 0.7 else 'GOOD' if quality_score > 0.5 else 'MODERATE'}")
    print("\nAll results saved to results/ directory 📁")