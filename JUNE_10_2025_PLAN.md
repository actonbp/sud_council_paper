# June 10, 2025 Research Plan - Study 2 Topic Modeling Approach

## Overview
We are transitioning Study 2 from hierarchical clustering to **Latent Dirichlet Allocation (LDA) topic modeling** using the `topicmodels` package, following the tidytext framework principles from [Tidy Topic Modeling](https://juliasilge.github.io/tidytext/articles/topic_modeling.html) and [Supervised Machine Learning for Text Analysis in R](https://smltar.com).

## Current Status
- **Study 1**: Complete and robust (L1-regularized logistic regression, ROC AUC = 0.787)
- **Study 2**: Transitioning from hierarchical clustering to LDA topic modeling
- **Manuscript**: Complete draft exists, Study 2 section needs updating with new approach

## Study 2 Implementation Plan

### Phase 1: Data Preprocessing Enhancement
1. **Moderator Removal**
   - Identify and remove facilitator utterances (likely Speaker 1 or first speaker per session)
   - Keep only participant responses for analysis

2. **Text Cleaning Pipeline**
   - Tokenization using `unnest_tokens()`
   - Remove standard stop words + custom focus group terms ("um", "like", "you know", etc.)
   - Apply Porter stemming via `SnowballC::wordStem()`

3. **Less Conservative Filtering**
   - Include utterances mentioning:
     - Counseling terms: counsel*, therap*, help*, support*, mental health
     - Substance terms: substance, drug*, alcohol*, addict*, abuse*
     - Career terms: career*, profession*, field*, job*, work*
   - Target: ~40-50% of utterances (vs current 19.7%)

### Phase 2: Topic Modeling Implementation
1. **Document-Term Matrix Creation**
   - Use `cast_dtm()` from tidytext to create DTM
   - Each utterance = document
   - Terms = filtered, stemmed words

2. **LDA Model Exploration**
   - Test multiple k values (2-8 topics)
   - Use `topicmodels::LDA()` with seed for reproducibility
   - Evaluate models using:
     - Perplexity
     - Topic coherence
     - Interpretability of resulting topics

3. **Topic Analysis**
   - Extract topic-term probabilities using `tidy()`
   - Identify top terms per topic
   - Calculate document-topic probabilities with `augment()`
   - Create visualizations of topic distributions

### Phase 3: Research Team Interpretation
- Present word clusters from each topic
- Research team assigns meaningful theme names
- Connect themes to Study 1 findings
- Update manuscript with new thematic analysis

## Technical Implementation

### New Script Structure
1. `study2_text_preprocessing.R` (keep and modify)
   - Add moderator removal
   - Implement less conservative filtering
   - Enhanced stop word list

2. `study2_topic_modeling.R` (new)
   - DTM creation
   - LDA model fitting with multiple k values
   - Model evaluation metrics
   - Export topic-term matrices for interpretation

3. `study2_topic_visualization.R` (new)
   - Topic distribution plots
   - Word clouds per topic
   - Document-topic heatmaps
   - Publication-ready figures

### Package Requirements Update
Add to `r_package_requirements.R`:
```r
# Topic modeling packages
library(topicmodels)
library(ldatuning)  # For selecting optimal k
```

## Timeline
- **Week 1**: Complete preprocessing updates and moderator removal
- **Week 2**: Implement topic modeling and test k values
- **Week 3**: Generate visualizations and prepare for team interpretation
- **Week 4**: Update manuscript with new findings

## Expected Outcomes
- 3-6 interpretable topics emerging from focus group data
- Higher utterance inclusion rate (40-50% vs 19.7%)
- Data-driven themes without researcher-imposed categories
- Clear connection to career uncertainty findings from Study 1

## Key Differences from Previous Approach
| Aspect | Old (Clustering) | New (Topic Modeling) |
|--------|------------------|---------------------|
| Method | Hierarchical clustering | LDA topic modeling |
| Filtering | Conservative (19.7%) | Inclusive (40-50%) |
| Theme Discovery | Co-occurrence patterns | Probabilistic topics |
| Interpretation | Mathematical clusters | Word probability distributions |
| Validation | Silhouette analysis | Perplexity, coherence |

## Next Steps
1. Review and clean `study2_text_preprocessing.R`
2. Create new topic modeling script
3. Test on sample data
4. Full analysis once approach validated

## References
- Silge, J., & Robinson, D. (2017). Text Mining with R: A Tidy Approach
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation
- [tidytext LDA vignette](https://juliasilge.github.io/tidytext/articles/topic_modeling.html) 