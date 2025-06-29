#!/usr/bin/env python3
"""
Clean Topic Modeling with Meaningful Terms
Focuses on substantive words and meaningful ngrams, filters out artifacts
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Gensim for coherence and phrases
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
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

# Conservative domain filtering - keep counseling terms since they're meaningful
DOMAIN_RE = re.compile(
    r"\b(substance|abuse|addict\w*|drug\w*|alcohol\w*)\b",  
    re.I,
)

# Comprehensive stopwords including artifacts
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

re ve ll don didn won isn aren weren hasn haven couldn wouldn shouldn mustn needn mightn
won ve ll re don didn hasn haven weren aren shouldn wouldn couldn mustn needn mightn

also even just now well much many still back still come came put take took give gave
go went come came one two three first second next last another other thing things
time times someone something anything nothing everything somebody anybody nobody everybody
""".split())


class CleanMeaningfulTopicModeler:
    """Topic modeling focused on meaningful, interpretable terms"""
    
    def __init__(self):
        self.optimal_k = None
        
    def load_data(self):
        """Load and clean data with better filtering"""
        docs = []
        
        print(">>> Loading focus group data...")
        for fn in CSV_FILES:
            fp = os.path.join(DATA_DIR, fn)
            df = pd.read_csv(fp)
            original_count = len(df)
            
            # Remove moderator rows and empty text
            df = df[~df["Speaker"].astype(str).str.match(r"^[A-Z]{2,3}$")]
            df = df[df["Text"].notna()]
            
            # Keep only substantive utterances (>= 10 words)
            df = df[df["Text"].str.split().str.len() >= 10]
            
            texts = df["Text"].astype(str).tolist()
            docs.extend(texts)
            
            print(f"{fn:45} | {original_count:3} → {len(df):3} substantive utterances")
        
        print(f"Total substantive utterances: {len(docs)}")
        self.raw_docs = docs
        
        # Clean text (conservative domain filtering)
        self.clean_docs = [DOMAIN_RE.sub(" ", doc.lower()) for doc in docs]
        
    def clean_and_detect_phrases(self):
        """Advanced text cleaning and meaningful phrase detection"""
        print("\n>>> Advanced text cleaning and phrase detection...")
        
        # Step 1: Clean contractions and artifacts
        cleaned_docs = []
        for doc in self.clean_docs:
            # Expand contractions properly
            doc = re.sub(r"don't", "do not", doc)
            doc = re.sub(r"won't", "will not", doc)
            doc = re.sub(r"can't", "cannot", doc)
            doc = re.sub(r"n't", " not", doc)
            doc = re.sub(r"'re", " are", doc)
            doc = re.sub(r"'ve", " have", doc)
            doc = re.sub(r"'ll", " will", doc)
            doc = re.sub(r"'d", " would", doc)
            doc = re.sub(r"'m", " am", doc)
            
            # Remove punctuation but keep spaces
            doc = re.sub(r"[^\w\s]", " ", doc)
            
            # Remove extra whitespace
            doc = re.sub(r"\s+", " ", doc).strip()
            
            cleaned_docs.append(doc)
        
        # Step 2: Tokenize with length filtering
        sentences = []
        for doc in cleaned_docs:
            tokens = simple_preprocess(doc, deacc=True, min_len=3, max_len=20)
            # Filter out obvious artifacts and very short words
            meaningful_tokens = []
            for token in tokens:
                if (len(token) >= 3 and 
                    token not in STOP_WORDS and
                    not re.match(r'^[a-z]{1,2}$', token) and  # Remove 1-2 letter words
                    not token.isdigit()):  # Remove pure numbers
                    meaningful_tokens.append(token)
            
            if len(meaningful_tokens) >= 5:  # Keep only docs with meaningful content
                sentences.append(meaningful_tokens)
        
        print(f"Cleaned to {len(sentences)} documents with meaningful content")
        
        # Step 3: Detect meaningful phrases
        bigram = Phrases(sentences, min_count=3, threshold=15, delimiter="_")
        trigram = Phrases(bigram[sentences], threshold=12, delimiter="_")
        phraser = Phraser(trigram)
        
        # Apply phrase detection
        docs_with_phrases = [" ".join(phraser[s]) for s in sentences]
        self.tokens = [phraser[s] for s in sentences]
        
        # Show detected meaningful phrases
        phrases = [phrase for phrase in phraser.phrasegrams.keys() if "_" in phrase]
        meaningful_phrases = [p for p in phrases if len(p.split('_')[0]) >= 3 and len(p.split('_')[1]) >= 3]
        print(f"Detected {len(meaningful_phrases)} meaningful phrases:")
        for phrase in meaningful_phrases[:15]:
            print(f"  - {phrase}")
        
        return docs_with_phrases
        
    def create_meaningful_features(self, docs):
        """Create feature matrix focused on meaningful terms"""
        print("\n>>> Creating meaningful feature matrix...")
        
        # Custom token pattern to ensure meaningful words
        def meaningful_tokenizer(text):
            # Split on whitespace and filter
            tokens = text.split()
            meaningful_tokens = []
            for token in tokens:
                if (len(token) >= 3 and 
                    token not in STOP_WORDS and
                    not re.match(r'^[a-z]{1,2}$', token) and
                    not token.isdigit() and
                    not re.match(r'^[a-z]*_[a-z]$', token)):  # Remove partial phrases
                    meaningful_tokens.append(token)
            return meaningful_tokens
        
        # Use TF-IDF with custom tokenizer
        vectorizer = TfidfVectorizer(
            tokenizer=meaningful_tokenizer,
            lowercase=False,  # Already lowercased
            min_df=4,  # Must appear in at least 4 documents
            max_df=0.6,  # Can't appear in more than 60% of docs
            max_features=250,  # Focus on most meaningful features
            sublinear_tf=True,
            norm='l2'
        )
        
        X = vectorizer.fit_transform(docs)
        vocab = vectorizer.get_feature_names_out()
        
        # Verify vocabulary quality
        print(f"Feature matrix: {X.shape[0]} docs × {X.shape[1]} meaningful features")
        print(f"Sample meaningful terms: {list(vocab[:20])}")
        
        # Filter out any remaining artifacts
        good_vocab_indices = []
        good_vocab = []
        for i, term in enumerate(vocab):
            if (len(term) >= 3 and 
                not re.match(r'^[a-z]{1,2}$', term) and
                term not in STOP_WORDS):
                good_vocab_indices.append(i)
                good_vocab.append(term)
        
        if len(good_vocab_indices) < len(vocab):
            X = X[:, good_vocab_indices]
            vocab = np.array(good_vocab)
            print(f"Filtered to {X.shape[1]} high-quality terms")
        
        return X, vocab, vectorizer
        
    def find_optimal_k_meaningful(self, X, vocab, k_range=range(3, 7)):
        """Find optimal k focusing on meaningful separation"""
        print("\n>>> Finding optimal number of meaningful topics...")
        
        results = []
        
        for k in k_range:
            print(f"Testing k={k}...")
            
            # Test both algorithms
            for alg_name in ['lda', 'nmf']:
                if alg_name == 'lda':
                    model = LatentDirichletAllocation(
                        n_components=k, max_iter=150, random_state=42,
                        doc_topic_prior=0.1, topic_word_prior=0.01,
                        learning_method='batch')
                else:
                    model = NMF(
                        n_components=k, init='nndsvd', max_iter=800, random_state=42,
                        alpha_W=0.1, alpha_H=0.1, l1_ratio=0.3)
                
                # Fit model
                doc_topic = model.fit_transform(X)
                assignments = doc_topic.argmax(axis=1)
                
                # Get topics with meaningful terms
                topics = []
                for topic_idx, topic in enumerate(model.components_):
                    top_indices = topic.argsort()[-20:][::-1]
                    top_terms = [vocab[i] for i in top_indices]
                    # Filter for truly meaningful terms
                    meaningful_terms = [t for t in top_terms if len(t) >= 3 and '_' not in t or len(t.split('_')[0]) >= 3]
                    topics.append(meaningful_terms[:15])
                
                # Calculate meaningfulness metrics
                
                # 1. Term meaningfulness (average length, presence of phrases)
                all_terms = []
                phrase_count = 0
                for topic in topics:
                    for term in topic[:10]:
                        all_terms.append(term)
                        if '_' in term:
                            phrase_count += 1
                
                avg_term_length = np.mean([len(term) for term in all_terms])
                phrase_ratio = phrase_count / len(all_terms) if all_terms else 0
                
                # 2. Topic diversity (uniqueness)
                unique_terms = len(set(all_terms))
                diversity = unique_terms / len(all_terms) if all_terms else 0
                
                # 3. Coherence
                cm_cv = CoherenceModel(topics=topics, texts=self.tokens, 
                                     dictionary=Dictionary(self.tokens), coherence='c_v')
                coherence = cm_cv.get_coherence()
                
                # 4. Balance
                topic_counts = [np.sum(assignments == i) for i in range(k)]
                balance = 1 - np.std(topic_counts) / np.mean(topic_counts) if np.mean(topic_counts) > 0 else 0
                
                # Combined meaningfulness score
                meaningful_score = (diversity * 0.3 + 
                                  coherence * 0.25 + 
                                  balance * 0.2 + 
                                  (avg_term_length - 3) / 10 * 0.15 +  # Longer terms bonus
                                  phrase_ratio * 0.1)  # Phrase bonus
                
                results.append({
                    'k': k,
                    'algorithm': alg_name,
                    'diversity': diversity,
                    'coherence': coherence,
                    'balance': balance,
                    'avg_term_length': avg_term_length,
                    'phrase_ratio': phrase_ratio,
                    'meaningful_score': meaningful_score,
                    'topics': topics,
                    'assignments': assignments,
                    'model': model
                })
                
                print(f"  {alg_name.upper()}: diversity={diversity:.3f}, coherence={coherence:.3f}, "
                      f"term_length={avg_term_length:.1f}, score={meaningful_score:.3f}")
        
        # Find best configuration
        best_config = max(results, key=lambda x: x['meaningful_score'])
        self.optimal_k = best_config['k']
        
        print(f"\n>>> OPTIMAL MEANINGFUL CONFIGURATION:")
        print(f"    k={best_config['k']}, algorithm={best_config['algorithm'].upper()}")
        print(f"    Meaningful score: {best_config['meaningful_score']:.3f}")
        
        return best_config, pd.DataFrame(results)
        
    def validate_topic_meaningfulness(self, topics):
        """Validate that topics contain meaningful, interpretable terms"""
        print("\n>>> Validating topic meaningfulness...")
        
        validated_topics = []
        for i, topic_terms in enumerate(topics):
            # Filter for meaningful terms only
            meaningful_terms = []
            for term in topic_terms:
                if (len(term) >= 3 and 
                    not re.match(r'^[a-z]{1,2}$', term) and
                    term not in STOP_WORDS and
                    not term.isdigit()):
                    meaningful_terms.append(term)
            
            validated_topics.append(meaningful_terms[:15])
            print(f"  Topic {i+1}: {len(meaningful_terms)} meaningful terms")
            print(f"    Top terms: {', '.join(meaningful_terms[:10])}")
        
        return validated_topics
        
    def interpret_meaningful_topics(self, topics, assignments):
        """Generate interpretations based on meaningful terms"""
        
        # Enhanced theme patterns focused on meaningful concepts
        theme_patterns = {
            'helping_motivation': {
                'keywords': ['help', 'helping', 'support', 'care', 'assistance', 'service'],
                'label': 'Helping Motivation & Service'
            },
            'personal_family': {
                'keywords': ['family', 'personal', 'experience', 'background', 'parents', 'relatives'],
                'label': 'Personal & Family Experience'
            },
            'education_training': {
                'keywords': ['school', 'education', 'training', 'learning', 'program', 'course'],
                'label': 'Education & Training'
            },
            'career_professional': {
                'keywords': ['career', 'profession', 'professional', 'job', 'work', 'field'],
                'label': 'Career & Professional Development'
            },
            'clinical_therapy': {
                'keywords': ['therapy', 'therapeutic', 'clinical', 'treatment', 'counseling', 'session'],
                'label': 'Clinical & Therapeutic Practice'
            },
            'mental_health': {
                'keywords': ['mental_health', 'psychological', 'emotional', 'wellness', 'health'],
                'label': 'Mental Health Focus'
            },
            'challenges_barriers': {
                'keywords': ['difficult', 'challenging', 'hard', 'struggle', 'barriers', 'obstacles'],
                'label': 'Challenges & Barriers'
            },
            'social_community': {
                'keywords': ['community', 'society', 'social', 'people', 'population', 'public'],
                'label': 'Social & Community Impact'
            }
        }
        
        interpretations = []
        for i, topic_terms in enumerate(topics):
            doc_count = sum(assignments == i)
            
            # Score against meaningful theme patterns
            theme_scores = {}
            for theme_key, theme_info in theme_patterns.items():
                score = 0
                for term in topic_terms[:12]:  # Check top 12 meaningful terms
                    term_lower = term.lower()
                    for keyword in theme_info['keywords']:
                        if keyword in term_lower or term_lower in keyword:
                            score += 1
                if score > 0:
                    theme_scores[theme_key] = score
            
            # Select best theme or generate from top terms
            if theme_scores:
                best_theme = max(theme_scores, key=theme_scores.get)
                interpretation = theme_patterns[best_theme]['label']
            else:
                # Generate interpretation from most meaningful terms
                top_meaningful = [t for t in topic_terms[:5] if len(t) >= 4]
                if top_meaningful:
                    interpretation = f"Theme: {top_meaningful[0].title()}-Related"
                else:
                    interpretation = "Mixed Discussion Theme"
            
            interpretations.append(f"{interpretation} ({doc_count} docs)")
        
        return interpretations
        
    def create_meaningful_visualization(self, results):
        """Create visualization emphasizing meaningful content"""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(20, 14))
        
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
        
        k = len(results['topics'])
        colors = plt.cm.Set2(np.linspace(0, 1, k))
        
        # 1. Topic distribution
        ax1 = fig.add_subplot(gs[0, 0])
        topic_counts = [sum(results['assignments'] == i) for i in range(k)]
        bars = ax1.bar(range(k), topic_counts, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Document Distribution\n(Meaningful Topics)', fontweight='bold')
        ax1.set_xlabel('Topic')
        ax1.set_ylabel('Documents')
        ax1.set_xticks(range(k))
        ax1.set_xticklabels([f'T{i+1}' for i in range(k)])
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Meaningfulness metrics
        ax2 = fig.add_subplot(gs[0, 1])
        metrics = ['Diversity', 'Coherence', 'Term\nLength', 'Phrases']
        values = [results['diversity'], results['coherence'], 
                 (results['avg_term_length'] - 3) / 10, results['phrase_ratio']]
        
        bars = ax2.bar(metrics, values, color=['coral', 'lightblue', 'lightgreen', 'gold'], alpha=0.8)
        ax2.set_title('Meaningfulness Metrics', fontweight='bold')
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Term length distribution
        ax3 = fig.add_subplot(gs[0, 2])
        all_terms = []
        for topic in results['topics']:
            all_terms.extend(topic[:10])
        
        term_lengths = [len(term) for term in all_terms]
        ax3.hist(term_lengths, bins=range(3, max(term_lengths)+2), alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('Term Length Distribution', fontweight='bold')
        ax3.set_xlabel('Term Length (characters)')
        ax3.set_ylabel('Frequency')
        
        # 4. Phrase analysis
        ax4 = fig.add_subplot(gs[0, 3])
        phrase_counts = []
        single_word_counts = []
        for topic in results['topics']:
            phrases = sum(1 for term in topic[:10] if '_' in term)
            single_words = sum(1 for term in topic[:10] if '_' not in term)
            phrase_counts.append(phrases)
            single_word_counts.append(single_words)
        
        x = range(k)
        width = 0.35
        ax4.bar([i - width/2 for i in x], single_word_counts, width, label='Single Words', alpha=0.8)
        ax4.bar([i + width/2 for i in x], phrase_counts, width, label='Phrases', alpha=0.8)
        ax4.set_title('Words vs Phrases per Topic', fontweight='bold')
        ax4.set_xlabel('Topic')
        ax4.set_ylabel('Count')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'T{i+1}' for i in range(k)])
        ax4.legend()
        
        # 5. Topic terms display (large section)
        ax5 = fig.add_subplot(gs[1:, :])
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        
        y_positions = np.linspace(0.9, 0.1, k)
        
        for i, (topic_terms, interpretation) in enumerate(zip(results['topics'], results['interpretations'])):
            y_pos = y_positions[i]
            color = colors[i]
            
            # Topic header
            ax5.text(0.02, y_pos, f"Topic {i+1}: {interpretation}", 
                    transform=ax5.transAxes, fontsize=16, fontweight='bold', color=color)
            
            # Separate phrases and single words
            phrases = [t for t in topic_terms[:15] if '_' in t]
            single_words = [t for t in topic_terms[:15] if '_' not in t]
            
            # Display phrases first (more meaningful)
            if phrases:
                phrases_text = f"Key phrases: {', '.join(phrases[:8])}"
                ax5.text(0.02, y_pos - 0.05, phrases_text, 
                        transform=ax5.transAxes, fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow', 
                                 alpha=0.9, edgecolor=color, linewidth=2))
            
            # Display single words
            if single_words:
                words_text = f"Important terms: {', '.join(single_words[:12])}"
                ax5.text(0.02, y_pos - 0.10, words_text, 
                        transform=ax5.transAxes, fontsize=11,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                 alpha=0.9, edgecolor=color, linewidth=1))
            
            # Stats
            doc_count = sum(results['assignments'] == i)
            stats_text = f"Documents: {doc_count} | Avg term length: {np.mean([len(t) for t in topic_terms[:10]]):.1f} chars"
            ax5.text(0.02, y_pos - 0.15, stats_text, 
                    transform=ax5.transAxes, fontsize=10, style='italic', color='gray')
        
        # Title
        title = f'Meaningful Topic Analysis: {k} Interpretable Themes'
        subtitle = f'Avg Term Length: {results["avg_term_length"]:.1f} | Phrases: {results["phrase_ratio"]:.1%} | Score: {results["meaningful_score"]:.3f}'
        fig.suptitle(f'{title}\n{subtitle}', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = os.path.join(RESULTS_DIR, 'meaningful_clean_topics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Meaningful topic visualization saved: {output_path}")
        
    def save_meaningful_results(self, results, k_analysis_df):
        """Save results emphasizing meaningful content"""
        
        # Main report
        with open(os.path.join(RESULTS_DIR, 'meaningful_clean_topics_report.txt'), 'w') as f:
            f.write("MEANINGFUL CLEAN TOPIC ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("MEANINGFUL CONFIGURATION:\n")
            f.write(f"• Number of topics: {len(results['topics'])}\n")
            f.write(f"• Algorithm: {results['algorithm'].upper()}\n")
            f.write(f"• Documents analyzed: {len(results['assignments'])}\n")
            f.write(f"• Meaningful score: {results['meaningful_score']:.3f}\n\n")
            
            f.write("MEANINGFULNESS METRICS:\n")
            f.write(f"• Topic diversity: {results['diversity']:.3f}\n")
            f.write(f"• Topic coherence: {results['coherence']:.3f}\n")
            f.write(f"• Average term length: {results['avg_term_length']:.1f} characters\n")
            f.write(f"• Phrase ratio: {results['phrase_ratio']:.1%}\n")
            f.write(f"• Topic balance: {results['balance']:.3f}\n\n")
            
            f.write("MEANINGFUL THEMES:\n")
            f.write("-" * 70 + "\n")
            
            for i, (topic_terms, interpretation) in enumerate(zip(results['topics'], results['interpretations'])):
                doc_count = sum(results['assignments'] == i)
                phrases = [t for t in topic_terms if '_' in t]
                single_words = [t for t in topic_terms if '_' not in t]
                
                f.write(f"\n{interpretation}\n")
                f.write(f"  • Document count: {doc_count} ({doc_count/len(results['assignments']):.1%})\n")
                f.write(f"  • Key phrases: {', '.join(phrases[:8])}\n")
                f.write(f"  • Important terms: {', '.join(single_words[:12])}\n")
                f.write(f"  • Term characteristics: {len(phrases)} phrases, {len(single_words)} single words\n")
        
        # Analysis details
        k_analysis_df.to_csv(os.path.join(RESULTS_DIR, 'meaningful_k_analysis.csv'), index=False)
        
        # Clean document assignments
        assignments_df = pd.DataFrame({
            'document_id': range(len(results['assignments'])),
            'topic_number': results['assignments'] + 1,
            'topic_interpretation': [results['interpretations'][i] for i in results['assignments']],
            'text_preview': [doc[:200] + '...' if len(doc) > 200 else doc 
                           for doc in self.raw_docs[:len(results['assignments'])]]
        })
        assignments_df.to_csv(os.path.join(RESULTS_DIR, 'meaningful_topic_assignments.csv'), index=False)
        
        print("✓ Meaningful results saved:")
        print("  • meaningful_clean_topics_report.txt")
        print("  • meaningful_k_analysis.csv") 
        print("  • meaningful_topic_assignments.csv")
        
    def run_meaningful_analysis(self):
        """Execute meaningful topic analysis"""
        print("=" * 75)
        print("MEANINGFUL CLEAN TOPIC MODELING")
        print("=" * 75)
        
        # Load and clean data
        self.load_data()
        docs_clean = self.clean_and_detect_phrases()
        X, vocab, vectorizer = self.create_meaningful_features(docs_clean)
        
        # Find optimal configuration with meaningful metrics
        best_config, k_analysis_df = self.find_optimal_k_meaningful(X, vocab)
        
        # Validate and clean topics
        meaningful_topics = self.validate_topic_meaningfulness(best_config['topics'])
        
        # Generate interpretations
        interpretations = self.interpret_meaningful_topics(meaningful_topics, best_config['assignments'])
        
        # Update results
        meaningful_results = best_config.copy()
        meaningful_results['topics'] = meaningful_topics
        meaningful_results['interpretations'] = interpretations
        
        # Display results
        print("\n" + "=" * 75)
        print("MEANINGFUL CLEAN TOPICS IDENTIFIED")
        print("=" * 75)
        print(f"Optimal k: {len(meaningful_topics)}")
        print(f"Algorithm: {meaningful_results['algorithm'].upper()}")
        print(f"Meaningful score: {meaningful_results['meaningful_score']:.3f}")
        print(f"Average term length: {meaningful_results['avg_term_length']:.1f} characters")
        print(f"Phrase ratio: {meaningful_results['phrase_ratio']:.1%}")
        
        print(f"\nMEANINGFUL THEMES:")
        for i, (topic_terms, interpretation) in enumerate(zip(meaningful_topics, interpretations)):
            doc_count = sum(meaningful_results['assignments'] == i)
            phrases = [t for t in topic_terms[:10] if '_' in t]
            single_words = [t for t in topic_terms[:10] if '_' not in t]
            
            print(f"\n{i+1}. {interpretation}")
            if phrases:
                print(f"    Key phrases: {', '.join(phrases)}")
            if single_words:
                print(f"    Main terms: {', '.join(single_words[:8])}")
            print(f"    Documents: {doc_count} | Avg term length: {np.mean([len(t) for t in topic_terms[:10]]):.1f}")
        
        # Save and visualize
        self.save_meaningful_results(meaningful_results, k_analysis_df)
        self.create_meaningful_visualization(meaningful_results)
        
        return meaningful_results


if __name__ == "__main__":
    modeler = CleanMeaningfulTopicModeler()
    results = modeler.run_meaningful_analysis()
    
    print("\n" + "🎯" * 30)
    print("MEANINGFUL TOPIC ANALYSIS COMPLETE!")
    print("🎯" * 30)
    print(f"\nKey Achievements:")
    print(f"• {len(results['topics'])} topics with meaningful, interpretable terms")
    print(f"• Average term length: {results['avg_term_length']:.1f} characters")
    print(f"• {results['phrase_ratio']:.1%} meaningful phrases detected")
    print(f"• Cleaned out artifacts like 've', 'don', 're'")
    print("• Focused on substantive ngrams and content words")
    print("\nAll results saved to results/ directory")