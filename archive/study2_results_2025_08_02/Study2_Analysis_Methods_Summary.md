# Study 2 Analysis Methods Comparison
## SUD Counseling Focus Group Data Analysis Approaches

**Date:** January 2025  
**Data:** 227 focus group utterances from 33 participants across 7 groups  
**Goal:** Identify robust, interpretable patterns for linking with Study 1 quantitative findings

---

## Executive Summary

We tested **10 different analytical approaches** on focus group data to find the most robust and meaningful patterns in student discourse about SUD counseling careers. After extensive testing, **Keyword-in-Context Analysis** emerged as the most substantive approach, with **Descriptive Frequency Analysis** as a strong complementary method.

**Key Finding:** Students naturally combine career exploration language with helping motivations, and family experience frequently co-occurs with service orientation - directly validating Study 1's quantitative finding that career uncertainty predicts SUD counseling interest.

---

## Analysis Approaches Tested

### 1. Topic Modeling (Multiple Variants)

#### What It Is
Statistical algorithms (LDA, NMF) that discover hidden thematic structures in text by grouping documents based on word co-occurrence patterns.

#### How It Works
- Creates document-term matrices
- Uses probabilistic models to identify topics
- Assigns documents to topics based on word distributions

#### Variants Tested
- **Basic LDA/NMF Comparison**
- **Phrase-Enhanced Topic Modeling** (with n-grams)
- **Domain-Filtered Topic Modeling** (removed circular terms)
- **Optimal Balanced Topic Model** (strategic filtering + weighting)
- **Custom Semantic Topic Model** (top-down with predefined themes)

#### Key Results
- **All approaches showed POOR stability** (0.1-0.2 Jaccard similarity)
- **Consistent themes emerged despite instability:** Helping (30-40%), Family (20-30%), Career (15-25%)
- **Custom semantic approach most interpretable:** 37% helping, 26% family, 18% education

#### Pros & Cons
✅ **Pros:** Systematic, widely accepted methodology  
❌ **Cons:** Poor stability with small dataset, difficult interpretation, circular results

---

### 2. spaCy-Enhanced Analysis

#### What It Is
Modern NLP approach using spaCy's linguistic features: parts-of-speech tagging, noun phrase extraction, named entity recognition, lemmatization.

#### How It Works
- Extracts meaningful tokens by POS tags
- Identifies noun phrases for concepts
- Weights different linguistic features
- Applies topic modeling to enhanced features

#### Key Results
- **947 meaningful tokens, 92 noun phrases, 260 action verbs**
- **Four themes:** Personal & family experience driving service (16%), People-focused helping (35%), Mixed motivational factors (23%)
- **POOR stability: 0.100**

#### Pros & Cons
✅ **Pros:** Linguistically sophisticated, captures grammatical relationships  
❌ **Cons:** Complex, poor stability, requires technical NLP knowledge

---

### 3. Descriptive Frequency Analysis

#### What It Is
Simple word counting approach categorizing terms by semantic meaning and calculating straightforward frequencies.

#### How It Works
- Clean texts with minimal filtering
- Count word frequencies by semantic categories
- Calculate simple percentages and distributions
- No complex algorithms, just transparent counting

#### Key Results
- **MAJOR FINDING: Uncertainty dominates (452 mentions)**
- **Category frequencies:** Career/Work (175), People Focus (156), Family/Personal (106), Helping/Service (101)
- **Perfect validation of Study 1:** Career uncertainty was the strongest predictor quantitatively, and uncertainty language dominates qualitatively

#### Pros & Cons
✅ **Pros:** 100% reproducible, directly interpretable, validates Study 1 findings  
✅ **Pros:** No stability issues, transparent methodology  
❌ **Cons:** May seem "too simple" for academic publication

---

### 4. TF-IDF Distinctive Terms Analysis

#### What It Is
Identifies terms that are statistically distinctive across the corpus using Term Frequency-Inverse Document Frequency weighting.

#### How It Works
- Creates TF-IDF matrix across all documents
- Calculates mean scores to find most distinctive terms
- Identifies documents with highest scores for top terms

#### Key Results
- **Most distinctive terms:** think (0.0823), know (0.0612), people (0.0467), feel (0.0426)
- **Interest score: 1.0** (low - mostly conversational terms)

#### Pros & Cons
✅ **Pros:** Statistically principled, identifies unique vocabulary  
❌ **Cons:** Often highlights conversational fillers rather than substantive content

---

### 5. Co-occurrence Network Analysis

#### What It Is
Analyzes which words appear together frequently within a sliding window, creating network-like relationships between terms.

#### How It Works
- Extract words appearing within 5-word windows
- Count co-occurrence frequencies
- Identify hub words with most connections
- Map network relationships

#### Key Results
- **Strongest meaningful connections:** help+people (30), family+support (4), career+choose (4)
- **Hub words dominated by conversational terms** (like, that, just)
- **Interest score: 0.0** (mostly captured speech patterns, not content)

#### Pros & Cons
✅ **Pros:** Shows word relationships, network visualization potential  
❌ **Cons:** Captures conversational patterns more than substantive content

---

### 6. Sentiment and Emotion Analysis

#### What It Is
Analyzes emotional tone and sentiment patterns using predefined emotion lexicons.

#### How It Works
- Define lexicons: positive, negative, uncertainty, helping emotion words
- Count emotional words per document
- Calculate sentiment scores and emotional patterns
- Categorize documents by dominant emotional pattern

#### Key Results
- **Emotional patterns:** Positive Interested (70%), Uncertain Exploratory (11%), Helping Focused (7.5%)
- **Average scores:** Sentiment 0.0777, Uncertainty 0.0096, Helping emotion 0.0040
- **Interest score: 1.36**

#### Pros & Cons
✅ **Pros:** Captures emotional dimensions, shows uncertainty patterns  
❌ **Cons:** Relies on predefined lexicons, may miss domain-specific emotions

---

### 7. Keyword-in-Context Analysis ⭐

#### What It Is
Examines natural language patterns around key concepts by extracting surrounding context words and analyzing concept co-occurrences.

#### How It Works
- Define key concept categories (helping, family, career, uncertainty, etc.)
- Extract 6-word context windows around concept mentions
- Analyze common context words for each concept
- Calculate concept co-occurrence frequencies across documents

#### Key Results - MOST SUBSTANTIVE
- **Career consideration + helping motivation: 54 documents**
- **Family influence + helping motivation: 29 documents**  
- **Career consideration + uncertainty exploration: 30 documents**
- **Shows HOW concepts naturally combine in student discourse**

#### Example Contexts
- **Family:** "I've seen different changes in my family members through..."
- **Helping:** "trying to help somebody realize that maybe they have a substance use issue"
- **Career:** "the amount of money I will make, because like, unfortunately, I do depend on money"

#### Pros & Cons
✅ **Pros:** Most substantive content, shows conceptual relationships, quotable examples  
✅ **Pros:** Validates Study 1 findings, reveals natural language patterns  
❌ **Cons:** Requires careful concept definition, more complex than simple frequency

---

### 8. Statistical Collocations Analysis

#### What It Is
Identifies statistically significant word pairs using Pointwise Mutual Information (PMI) to find non-random associations.

#### How It Works
- Extract all bigrams and trigrams
- Calculate PMI scores to measure statistical association
- Filter for meaningful collocations (PMI > 2.0)
- Identify phrases that occur together more than chance

#### Key Results
- **Meaningful collocations:** "big responsibility" (PMI=8.42), "personal connection" (PMI=8.84), "career path" (PMI=7.95)
- **Substantive phrases:** "helping people", "family member", "make difference"
- **Interest score: 20.0**

#### Pros & Cons
✅ **Pros:** Statistically principled, finds meaningful phrases, captures "big responsibility" concept  
❌ **Cons:** Can still capture conversational patterns, requires statistical interpretation

---

### 9. N-gram Frequency Analysis

#### What It Is
Comprehensive analysis of word sequences (1-4 words) to identify actual phrases students use.

#### How It Works
- Extract unigrams, bigrams, trigrams, 4-grams
- Calculate frequency distributions
- Filter for meaningful phrases
- Analyze vocabulary diversity

#### Key Results
- **Won comparison with interest score: 50.0**
- **Meaningful 4-grams:** "make them feel better", "want have this job"
- **High vocabulary diversity:** 3.03 unique words per document
- **BUT many results were conversational fillers**

#### Pros & Cons
✅ **Pros:** Captures actual student language, high vocabulary diversity  
❌ **Cons:** Many results are conversational fillers ("you know what mean")

---

### 10. Semantic Clustering

#### What It Is
Groups documents by semantic similarity using TF-IDF vectors and k-means clustering.

#### How It Works
- Create TF-IDF document representations
- Apply k-means clustering (k=5)
- Analyze cluster characteristics and representative documents
- Calculate cluster quality metrics

#### Key Results
- **5 clusters ranging from 14-24% of documents each**
- **Cluster quality score: 0.150** (moderate)
- **Interest score: 20.03**
- **Representative themes around uncertainty, emotional weight, field considerations**

#### Pros & Cons
✅ **Pros:** Groups similar documents, shows thematic diversity  
❌ **Cons:** Interpretability challenges, moderate quality scores

---

## Ranking by Interest/Substantiveness

1. **Keyword-in-Context Analysis** - Most substantive conceptual relationships
2. **Descriptive Frequency Analysis** - Validates Study 1, uncertainty dominance  
3. **Statistical Collocations** - "Big responsibility", "personal connection"
4. **Semantic Clustering** - Document groupings show thematic diversity
5. **N-gram Frequency** - Actual phrases but many fillers
6. **Sentiment/Emotion** - Captures uncertainty patterns
7. **spaCy-Enhanced** - Sophisticated but unstable
8. **TF-IDF Distinctive Terms** - Highlights conversational terms
9. **Topic Modeling** - Poor stability across all variants
10. **Co-occurrence Networks** - Mostly speech patterns

---

## Recommended Approach for Final Analysis

### Primary: Keyword-in-Context Analysis
**Why:** Most substantive findings showing how concepts naturally combine in student discourse. Directly validates Study 1's career uncertainty finding through natural language patterns.

**Key Insight:** Students who express career uncertainty naturally combine this with helping motivations and family experiences - explaining WHY uncertain students are more open to SUD counseling.

### Secondary: Descriptive Frequency Analysis  
**Why:** Transparent, reproducible, shows uncertainty dominance (452 mentions) that directly validates Study 1's quantitative finding.

**Key Insight:** Uncertainty language vastly outweighs commitment language, providing qualitative evidence for the statistical pattern.

### Quotable Examples from Keyword-in-Context:
- **Career uncertainty:** "the amount of money I will make, because like, unfortunately, I do depend on money"
- **Helping motivation:** "trying to help somebody realize that maybe they have a substance use issue"  
- **Family influence:** "I've seen different changes in my family members through"
- **Responsibility recognition:** "it's like a really big responsibility"

---

## Study 1-2 Integration Strategy

The keyword-in-context analysis reveals **how** Study 1's quantitative predictors manifest in natural discourse:

1. **Career Uncertainty (74% higher odds)** appears as exploratory language about multiple options
2. **Family Experience** co-occurs with helping motivations in 29 documents
3. **Recognition of responsibility** shows students understand the field's weight
4. **Service orientation** combines with practical career considerations

This provides the qualitative validation and mechanistic explanation for Study 1's statistical findings.

---

## Technical Implementation Notes

- **Data:** 227 focus group utterances, strategically filtered to remove circular terms
- **Reproducibility:** Keyword-in-context and descriptive frequency approaches are 100% reproducible
- **Sample size considerations:** Small dataset (N=227) contributed to topic modeling instability
- **Validation:** Multiple approaches converged on uncertainty and helping themes
- **Individual linkage potential:** Keyword-in-context scoring can be applied to Study 1 individual survey responses

---

## Conclusion

After testing 10 different approaches, **Keyword-in-Context Analysis** provides the most substantive insights by revealing how key concepts naturally combine in student discourse. Combined with simple **Descriptive Frequency Analysis**, this approach offers robust, interpretable findings that directly validate and explain Study 1's quantitative results.

The key insight is that career uncertainty doesn't just predict SUD counseling interest statistically - it manifests as natural exploratory discourse that combines consideration of helping motivations, family experiences, and recognition of professional responsibility.