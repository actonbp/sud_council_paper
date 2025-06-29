# Study 2: LDA Topic Modeling Workflow Guide

## Overview
This guide walks through the Study 2 analysis pipeline for LDA topic modeling of focus group discussions about SUD counseling career interest.

**Analysis Approach**: Latent Dirichlet Allocation (LDA) topic modeling using tidytext framework, following the June 10, 2025 plan.

## Prerequisites

1. **Install Required Packages**:
   ```r
   source("../r_package_requirements.R")
   ```

2. **Verify Data Available**:
   - Focus group files should be in `data/focus_group/` directory
   - Files should follow pattern `*_processed.csv`

## Step-by-Step Workflow

### Step 1: Data Preparation
**Script**: `study2_data_preparation.R`

**Purpose**: 
- Combines all focus group files
- Identifies and removes moderator utterances (first speaker in each session)
- Creates cleaned participant-only dataset

**Run**:
```r
source("study2_data_preparation.R")
```

**Outputs Created**:
- `data/focus_group_substantive.csv` - Main dataset for analysis
- `data/focus_group_sessions_summary.csv` - Session overview
- `data/focus_group_sample_review.csv` - Sample for quality review

**Verification**: Check that moderator removal worked properly by reviewing the sample file.

---

### Step 2: LDA Topic Modeling
**Script**: `study2_lda_topic_modeling.R`

**Purpose**:
- Applies less conservative SUD filtering (targeting 40-50% of utterances)
- Determines optimal number of topics using multiple statistical metrics
- Fits final LDA model and extracts topic-term and document-topic probabilities

**Run**:
```r
source("study2_lda_topic_modeling.R")
```

**Outputs Created**:
- `results/r/study2_lda_modeling/lda_model.rds` - Fitted LDA model
- `results/r/study2_lda_modeling/lda_topic_terms.csv` - Topic-term probabilities (β)
- `results/r/study2_lda_modeling/lda_document_topics.csv` - Document-topic probabilities (γ)
- `results/r/study2_lda_modeling/lda_topic_summaries.csv` - Topic summaries with preliminary themes
- `results/r/study2_lda_modeling/lda_tuning_results.csv` - Model selection metrics
- `results/r/study2_lda_modeling/lda_model_metadata.csv` - Analysis metadata

**Key Decision Point**: Review the topic summaries and consider if the optimal k makes sense. You can modify the k_values in the script if needed.

---

### Step 3: Create Visualizations
**Script**: `study2_lda_visualizations.R`

**Purpose**:
- Creates publication-ready figures for manuscript
- Generates manuscript-ready tables (CSV format)
- Produces method comparison visualizations

**Run**:
```r
source("study2_lda_visualizations.R")
```

**Outputs Created**:
- `results/figures/study2_lda_topic_selection.png` - Model selection metrics
- `results/figures/study2_lda_top_terms.png` - Top terms by topic
- `results/figures/study2_lda_document_topics.png` - Document-topic distributions
- `results/figures/study2_lda_summary_table.png` - Summary table visualization
- `results/figures/study2_lda_method_comparison.png` - Comparison with previous approach
- `results/figures/study2_lda_manuscript_table.csv` - Table for manuscript
- `results/figures/study2_lda_analysis_summary.csv` - Results summary for text

## After Analysis: Next Steps

### 1. Review and Name Topics
- Open `results/r/study2_lda_modeling/lda_topic_summaries.csv`
- Review the `preliminary_theme` column and top terms
- Research team should assign meaningful, specific names to each topic

### 2. Update Manuscript
The manuscript needs to be updated in several sections:

**Methods Section**: Replace hierarchical clustering methodology with LDA approach
**Results Section**: Replace clustering results with topic modeling results  
**Discussion Section**: Interpret topics instead of clusters
**Tables**: Use `results/figures/study2_lda_manuscript_table.csv`
**Figures**: Incorporate the new PNG visualizations

### 3. Consider Alternative Models
If the topics don't make substantive sense:
- Try different k values by modifying `k_values` in step 2
- Consider adjusting the SUD filtering terms
- Review the preprocessing steps for improvements

## Troubleshooting

### Common Issues

**"focus_group_substantive.csv not found"**
- Run Step 1 first to create this file

**"No SUD utterances detected"**  
- Check the `sud_terms` list in Step 2 script
- Verify your focus group data contains relevant terminology

**"LDA model files not found"**
- Ensure Step 2 completed successfully before running Step 3
- Check for error messages in the R console

**Visualization errors**
- Ensure all required packages are installed
- Check that the results directory structure exists

### Getting Help

1. Check the console output for specific error messages
2. Verify all prerequisite files exist in expected locations  
3. Ensure all required R packages are installed and loaded
4. Review each script's comments for parameter explanations

## Expected Timeline

- **Step 1**: ~2-3 minutes (data preparation)
- **Step 2**: ~5-10 minutes (LDA modeling with tuning)  
- **Step 3**: ~3-5 minutes (visualization creation)
- **Total**: ~10-18 minutes for complete analysis

## File Dependencies

```
Step 1 requires:
├── data/focus_group/*.csv (processed focus group files)

Step 2 requires:  
├── data/focus_group_substantive.csv (from Step 1)

Step 3 requires:
├── results/r/study2_lda_modeling/*.csv (from Step 2)
```

Following this workflow will produce a complete LDA topic modeling analysis ready for manuscript integration. 