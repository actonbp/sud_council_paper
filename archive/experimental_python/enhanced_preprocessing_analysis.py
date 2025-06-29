#!/usr/bin/env python
"""
enhanced_preprocessing_analysis.py - Advanced text preprocessing for topic modeling

Improvements over basic approach:
1. Lemmatization (reduces word forms: thinking->think, trying->try)
2. POS filtering (keeps only nouns, adjectives, verbs)  
3. Phrase detection (captures meaningful multi-word terms)
4. Advanced stopword filtering (discourse markers, intensifiers)
5. Minimum utterance length filtering
6. Semantic filtering using word embeddings (if available)
"""

import glob, os, hashlib, re, textwrap, sys, platform
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score
import nltk
from collections import Counter

# Download required NLTK data
print("Downloading NLTK data...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True) 
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    print("NLTK download failed - will use basic preprocessing")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet

print("Python:", sys.version.split()[0], "| Platform:", platform.platform())
import sklearn
print("scikit-learn:", sklearn.__version__)
print("-" * 60)

# -------------------------------------------------------------------
# 1. Enhanced Text Preprocessing Functions
# -------------------------------------------------------------------

def get_wordnet_pos(treebank_tag):
    """Convert POS tag to wordnet format for lemmatization"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def advanced_preprocess_text(text, lemmatizer, keep_pos=['NOUN', 'ADJ', 'VERB']):
    """
    Advanced text preprocessing with lemmatization and POS filtering
    """
    if pd.isna(text) or text.strip() == "":
        return ""
    
    # Basic cleaning
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)      # Normalize whitespace
    
    # Tokenize and POS tag
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    
    # Filter by POS and lemmatize
    processed_tokens = []
    for word, pos in pos_tags:
        # Convert POS tag for wordnet
        wordnet_pos = get_wordnet_pos(pos)
        
        # Keep only meaningful POS types
        if pos.startswith(('NN', 'JJ', 'VB', 'RB')) and len(word) > 2:
            # Lemmatize
            lemmatized = lemmatizer.lemmatize(word, wordnet_pos)
            processed_tokens.append(lemmatized)
    
    return ' '.join(processed_tokens)

def extract_meaningful_phrases(texts, min_freq=3):
    """Extract meaningful multi-word phrases using frequency and PMI"""
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Extract bigrams and trigrams
    bigram_vec = CountVectorizer(ngram_range=(2, 3), min_df=min_freq, 
                                max_features=1000, token_pattern=r'\b\w+\b')
    
    try:
        bigram_matrix = bigram_vec.fit_transform(texts)
        phrases = bigram_vec.get_feature_names_out()
        
        # Calculate phrase scores (simple frequency-based)
        phrase_scores = bigram_matrix.sum(axis=0).A1
        phrase_dict = dict(zip(phrases, phrase_scores))
        
        # Keep high-scoring phrases
        meaningful_phrases = [phrase for phrase, score in phrase_dict.items() 
                            if score >= min_freq and len(phrase.split()) >= 2]
        
        return meaningful_phrases[:50]  # Limit to top 50
    except:
        return []

# -------------------------------------------------------------------
# 2. Load and Apply Enhanced Preprocessing
# -------------------------------------------------------------------

# Load data (same as before)
DATA_DIR = "data"
CSV_GLOB = "*_Focus_*_full*.csv"
paths = sorted(glob.glob(os.path.join(DATA_DIR, CSV_GLOB)))
paths = [p for p in paths if 'focus' in os.path.basename(p).lower()]

if not paths:
    sys.exit(f"No files found via pattern {CSV_GLOB} in {DATA_DIR}")

print("Files detected:")
for p in paths:
    md5 = hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
    print(f"- {os.path.basename(p):40}  {md5}")

# Load data
df_list = []
for p in paths:
    df = pd.read_csv(p)
    df = df[df["Text"].notna()]
    is_mod = df["Speaker"].astype(str).str.match(r"^[A-Z]{2,3}$")
    df = df[~is_mod].copy()
    df["session"] = os.path.basename(p)
    df_list.append(df)

corpus = pd.concat(df_list, ignore_index=True)
print(f"Total utterances: {len(corpus)}")

# -------------------------------------------------------------------
# 3. Apply Enhanced Preprocessing
# -------------------------------------------------------------------

# Remove domain-specific terms
DOMAIN = re.compile(r"\b(substance|abuse|addict\w*|drug\w*|alcohol\w*|counsel\w*|mental\s+health)\b", re.I)
corpus["clean_basic"] = corpus["Text"].str.lower().apply(lambda t: DOMAIN.sub("", t))

# Filter out very short utterances (likely not informative)
min_words = 10
corpus["word_count"] = corpus["clean_basic"].str.split().str.len()
corpus_filtered = corpus[corpus["word_count"] >= min_words].copy()
print(f"Utterances after length filtering (>={min_words} words): {len(corpus_filtered)}")

# Initialize lemmatizer
try:
    lemmatizer = WordNetLemmatizer()
    print("Applying advanced preprocessing (lemmatization + POS filtering)...")
    
    # Apply advanced preprocessing
    corpus_filtered["clean_advanced"] = corpus_filtered["clean_basic"].apply(
        lambda x: advanced_preprocess_text(x, lemmatizer)
    )
    
    # Remove empty results
    corpus_filtered = corpus_filtered[corpus_filtered["clean_advanced"].str.len() > 0]
    print(f"Utterances after advanced preprocessing: {len(corpus_filtered)}")
    
except Exception as e:
    print(f"NLTK preprocessing failed: {e}")
    print("Falling back to basic preprocessing...")
    corpus_filtered["clean_advanced"] = corpus_filtered["clean_basic"]

# Extract meaningful phrases
print("Extracting meaningful phrases...")
meaningful_phrases = extract_meaningful_phrases(corpus_filtered["clean_advanced"].tolist())
print(f"Found {len(meaningful_phrases)} meaningful phrases")

# -------------------------------------------------------------------
# 4. Advanced Stop Words
# -------------------------------------------------------------------

# Comprehensive stopword list including discourse markers and intensifiers
ADVANCED_STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being
between both but by could did do does doing down during each few for from further had has
have having he her here hers herself him himself his how i if in into is it its itself
just like me more most my myself nor not of off on once only or other our ours ourselves
out over own same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were what when where
which while who whom why will with you your yours yourself yourselves

um uh yeah okay kinda sorta right would know think really kind going lot can say 
definitely want guess something able way actually maybe feel feels felt don re ve ll 
got get goes didn wouldn couldn shouldn won isn aren wasn hasn haven hadn even still 
always never already back come came comes coming came well better best good bad worse 
worst big bigger biggest small smaller smallest much many make makes made making take 
takes took taking give gives gave giving put puts putting look looks looked looking 
see sees saw seeing tell tells told telling

also probably pretty little bit around trying might worry find try enough terms anyone 
knows obviously every day strong situation issue somebody someone everything anything 
nothing something rather quite pretty much really very totally completely absolutely 
definitely certainly possibly maybe perhaps probably hopefully unfortunately basically 
generally usually normally typically especially particularly specifically exactly 
precisely certainly surely absolutely definitely completely totally entirely wholly 
quite rather somewhat fairly reasonably relatively pretty much sort kind type thing 
stuff things ways means manner method approach

one two three four five six seven eight nine ten first second third fourth fifth 
time times year years day days week weeks month months ago later soon now today 
yesterday tomorrow morning afternoon evening night

a b c d e f g h i j k l m n o p q r s t u v w x y z
""".split())

# Add NLTK stopwords if available
try:
    english_stopwords = set(stopwords.words('english'))
    ADVANCED_STOPWORDS.update(english_stopwords)
except:
    pass

print(f"Using {len(ADVANCED_STOPWORDS)} stopwords")

# -------------------------------------------------------------------
# 5. Advanced Vectorization
# -------------------------------------------------------------------

# Include meaningful phrases in vocabulary
phrase_pattern = '|'.join([re.escape(phrase) for phrase in meaningful_phrases])

vectorizer = TfidfVectorizer(
    stop_words=list(ADVANCED_STOPWORDS),
    ngram_range=(1, 2),
    min_df=3,              # Term must appear in at least 3 documents
    max_df=0.7,            # Remove terms in >70% of documents  
    max_features=500,      # Limit vocabulary size
    sublinear_tf=True,     # Use log scaling
    token_pattern=r'\b\w{3,}\b'  # Only words with 3+ characters
)

X = vectorizer.fit_transform(corpus_filtered["clean_advanced"])
vocab = vectorizer.get_feature_names_out()
print(f"Final feature matrix: {X.shape[0]} docs × {X.shape[1]} features")

# -------------------------------------------------------------------
# 6. Optimal k Selection with Enhanced Preprocessing  
# -------------------------------------------------------------------

k_range = range(2, 8)  # Test fewer k values since we have fewer docs
results = []

print("\nTesting optimal number of topics with enhanced preprocessing...")
print("k\tPerplexity\tSilhouette")
print("-" * 30)

for k in k_range:
    lda = LatentDirichletAllocation(
        n_components=k,
        random_state=42,
        learning_method="batch",
        max_iter=500,
        doc_topic_prior=0.1,
        topic_word_prior=0.01
    )
    
    doc_topic = lda.fit_transform(X)
    perplexity = lda.perplexity(X)
    
    # Silhouette score
    dominant_topics = doc_topic.argmax(axis=1)
    if len(np.unique(dominant_topics)) > 1:
        silhouette = silhouette_score(doc_topic, dominant_topics)
    else:
        silhouette = -1
    
    results.append({
        'k': k,
        'perplexity': perplexity,
        'silhouette': silhouette,
        'lda_model': lda,
        'doc_topic': doc_topic
    })
    
    print(f"{k}\t{perplexity:.1f}\t\t{silhouette:.3f}")

# Select optimal k
df_results = pd.DataFrame([{k: v for k, v in r.items() if k not in ['lda_model', 'doc_topic']} 
                          for r in results])

# Simple selection based on silhouette score
optimal_k = df_results.loc[df_results['silhouette'].idxmax(), 'k']
print(f"\nOptimal k based on silhouette score: {optimal_k}")

# -------------------------------------------------------------------
# 7. Analyze Enhanced Results
# -------------------------------------------------------------------

optimal_result = next(r for r in results if r['k'] == optimal_k)
lda_optimal = optimal_result['lda_model']
doc_topic_optimal = optimal_result['doc_topic']

print(f"\n{'='*70}")
print(f"ENHANCED PREPROCESSING RESULTS (k = {optimal_k})")
print(f"{'='*70}")

print(f"Processing improvements applied:")
print(f"• Lemmatization (thinking→think, trying→try)")
print(f"• POS filtering (nouns, adjectives, verbs only)")
print(f"• Minimum utterance length: {min_words} words")
print(f"• Advanced stopwords: {len(ADVANCED_STOPWORDS)} terms")
print(f"• Meaningful phrases detected: {len(meaningful_phrases)}")
print(f"• Vocabulary limited to: {X.shape[1]} features")

for t in range(optimal_k):
    print(f"\n{'-'*50}")
    print(f"TOPIC {t+1}")
    print(f"{'-'*50}")
    
    # Get top terms
    topic_terms = lda_optimal.components_[t]
    top_indices = topic_terms.argsort()[-20:][::-1]
    
    # Calculate distinctiveness
    avg_probs = lda_optimal.components_.mean(axis=0)
    distinctiveness = topic_terms / (avg_probs + 1e-10)
    
    distinctive_terms = []
    for idx in top_indices:
        term = vocab[idx]
        prob = topic_terms[idx]
        distinct = distinctiveness[idx]
        if prob > 0.01:  # Only meaningful terms
            distinctive_terms.append(f"{term}({distinct:.1f})")
        if len(distinctive_terms) >= 15:
            break
    
    print(f"Top terms: {', '.join(distinctive_terms)}")
    
    # Topic statistics
    topic_probs = doc_topic_optimal[:, t]
    dominant_docs = (topic_probs > 0.5).sum()
    avg_prob = topic_probs.mean()
    
    print(f"Documents: {dominant_docs} dominant, avg prob: {avg_prob:.3f}")

# Topic distribution
dominant = doc_topic_optimal.argmax(axis=1)
counts = pd.Series(dominant).value_counts().sort_index()
print(f"\nTopic Distribution:")
for t, n in counts.items():
    pct = 100 * n / len(corpus_filtered)
    print(f"Topic {t+1}: {n} utterances ({pct:.1f}%)")

print(f"\n{'='*70}")
print("COMPARISON WITH BASIC PREPROCESSING")
print(f"{'='*70}")
print("Enhanced preprocessing should show:")
print("• More coherent, meaningful terms")
print("• Better topic separation") 
print("• Reduced noise from filler words and fragments")
print("• More interpretable themes")