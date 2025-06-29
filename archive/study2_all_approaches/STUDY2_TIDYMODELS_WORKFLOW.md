# Study 2: Tidymodels Workflow Guide - Iterative Approach

## Overview
**Modern tidymodels-based text analysis pipeline** for robust topic modeling of focus group discussions about SUD counseling career interest.

**Key Innovation**: **ITERATIVE PARAMETER TUNING** - Look at topics, assess quality, adjust parameters, repeat until topics are good. This addresses the core problem of getting interpretable, non-overlapping topics.

## Analysis Approach
- **Framework**: tidymodels + textrecipes with iterative refinement
- **Method**: LDA topic modeling with quality-driven parameter optimization
- **Philosophy**: Look at results → Assess quality → Adjust parameters → Repeat
- **Advantages**: Ensures good topics before proceeding to publication

## Prerequisites

1. **Install Required Packages**:
   ```r
   source("../r_package_requirements.R")
   ```

2. **Verify Data Available**:
   - `data/focus_group_substantive.csv` (from study2_data_preparation.R)

## 🔄 ITERATIVE WORKFLOW (RECOMMENDED)

### Phase 1: Quick Iteration to Find Good Topics

#### Step 1A: Initial Parameter Testing
**Script**: `study2_iterative_tuning.R`

**Purpose**: 
- Quick testing of different parameter combinations
- Immediate preview of resulting topics
- Fast iteration without full analysis overhead
- Identifies promising parameter ranges

**How to use**:
1. Edit the parameter values at the top of the script
2. Run the script to see topic previews
3. Assess if topics look good
4. If not, adjust parameters and re-run
5. Repeat until topics are interpretable and distinct

**Run**:
```r
source("study2_iterative_tuning.R")
```

**Expected Runtime**: 3-5 minutes per iteration

**What you get**:
- Quick preview of topics with top terms
- Simple overlap and generic term checks
- Parameter recommendations
- Best parameter combination for that run

#### Step 1B: Topic Quality Assessment (Optional)
**Script**: `study2_topic_quality_assessment.R` 

**Purpose**: 
- Detailed analysis of topic quality if you have results
- Overlap analysis, coherence scoring, generic term detection
- Specific recommendations for parameter adjustments

**Run**: (Only after you have some results)
```r
source("study2_topic_quality_assessment.R")
```

**Decision Point**: Keep iterating until topics look good in the quick preview!

---

### Phase 2: Final Analysis with Good Parameters

#### Step 2A: Final Analysis
**Script**: `study2_final_analysis.R`

**Purpose**: 
- Runs final analysis with the good parameters you found
- Comprehensive results extraction and processing
- Creates manuscript-ready outputs

**How to use**:
1. Edit the parameter values at the top based on your iteration results
2. Run the final analysis
3. Review the comprehensive results

**Run**:
```r
source("study2_final_analysis.R")
```

**Expected Runtime**: 3-5 minutes

**Outputs Created**:
- `results/r/study2_final/final_topic_term_probabilities.csv`
- `results/r/study2_final/final_topic_summaries.csv`
- `results/r/study2_final/manuscript_table_final_topics.csv`
- Complete model objects and metadata

#### Step 2B: Final Quality Check
**Script**: `study2_topic_quality_assessment.R`

**Purpose**: 
- Detailed quality assessment of final results
- Publication-ready quality metrics
- Final validation before proceeding to visualizations

---

### Phase 3: Publication Materials

#### Step 3: Create Visualizations  
**Script**: `study2_tidymodels_visualizations.R`

**Purpose**:
- Publication-ready figures from final results
- Manuscript tables formatted for publication
- High-resolution plots for paper

**Run**: (Only after final analysis is complete and topics look good)
```r
source("study2_tidymodels_visualizations.R")
```

## Key Innovations

### 1. Hyperparameter Tuning
Unlike traditional topic modeling, this approach tunes:
- **k (number of topics)**: 2-4 range for small datasets
- **max_tokens**: Vocabulary size (15-30 terms)
- **min_freq**: Term frequency threshold (3-5 occurrences)

### 2. Robust Cross-Validation
- 3-fold CV with 3 repeats (9 total fits per combination)
- Proper model selection using held-out data
- Prevents overfitting to small dataset

### 3. Comprehensive Preprocessing
- Removes SUD-specific terms to focus on underlying themes
- Eliminates focus group conversation artifacts
- Aggressive stopword filtering for cleaner topics

### 4. Publication-Ready Outputs
- Manuscript tables in CSV format
- High-resolution figures (300 DPI)
- Complete model metadata for reproducibility

## Expected Results

### Model Selection
- **Optimal k**: Typically 2-3 topics for focus group data
- **Clean separation**: Reduced overlap through tuned preprocessing
- **Interpretable topics**: Meaningful term distributions

### Topic Quality
- Higher coherence through parameter optimization
- Better separation via cross-validated selection
- Reduced overlap compared to previous approaches

## Troubleshooting

### Common Issues

**"Package not found" errors**
- Run `source("../r_package_requirements.R")` first
- Ensure tidymodels and textrecipes are installed

**"Data not found" errors**  
- Run `study2_data_preparation.R` first to create substantive dataset

**Long runtime**
- Reduce tuning grid size by modifying parameter ranges
- Consider using fewer CV repeats (change to repeats = 2)

**Memory issues**
- Reduce max_tokens parameter range
- Close other R sessions

### Parameter Adjustment

If topics are still overlapping:
```r
# In study2_tidymodels_analysis.R, modify:

# More conservative vocabulary
max_tokens = c(10, 15, 20)  # Smaller vocabularies

# Higher frequency thresholds  
min_freq = c(4, 5, 6)       # More stringent filtering

# Fewer topics
k = 2:3                     # Simpler models
```

## Integration with Manuscript

### Methods Section Updates
1. Replace hierarchical clustering description with tidymodels approach
2. Add hyperparameter tuning methodology
3. Include cross-validation details

### Results Section Updates
1. Use `manuscript_table_topic_characteristics.csv` for topic descriptions
2. Include model selection figure
3. Report optimal parameters and validation metrics

### Advantages to Highlight
- **Data-driven model selection**: Cross-validated k selection
- **Robust preprocessing**: Tuned parameter optimization
- **Reproducible framework**: tidymodels ecosystem
- **Small dataset appropriate**: Conservative parameter ranges

## File Dependencies

```
study2_tidymodels_analysis.R requires:
├── data/focus_group_substantive.csv (from study2_data_preparation.R)

study2_tidymodels_visualizations.R requires:  
├── results/r/study2_tidymodels/*.csv (from Step 1)
```

## Expected Timeline

- **Step 1**: 5-15 minutes (analysis + tuning)
- **Step 2**: 2-5 minutes (visualization)
- **Total**: 7-20 minutes for complete pipeline

Following this workflow produces a robust, publication-ready topic modeling analysis that addresses the interpretability and overlap issues identified in previous approaches.