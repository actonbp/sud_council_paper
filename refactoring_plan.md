# Refactoring Plan: Python to R with tidymodels

## Project Overview
This plan outlines the transition from Python-based machine learning analyses to R with tidymodels for the SUD Counselor Paper project. The goal is to maintain all existing Python code while creating equivalent R implementations that will be used for the final paper outputs using the APA Quarto package.

## Current Structure
- Python scripts for preprocessing and model training
- Results stored in results/ directory
- Quarto document with mix of Python and R code for paper generation
- Study 1 (quantitative) complete in Python
- Study 2 (qualitative) incomplete

## Target Structure
- Keep Python code as reference/backup
- Implement all analyses in R with tidymodels
- Use APA Quarto package for consistent formatting
- R code handles all visualizations and tables in the paper
- Clean integration of both Study 1 (quantitative) and Study 2 (qualitative)

## Refactoring Phases

### Phase 1: Setup and Organization
1. Create scripts/r/ directory for R scripts
2. Create results/r/ directory for R-generated outputs
3. Define tidymodels package dependencies
4. Create R project file and set up renv for package management

### Phase 2: Data Preprocessing in R
1. Create 01_preprocess_survey.R
   - Read raw survey data
   - Clean and filter responses
   - Rename variables for clarity
   - Save processed data

2. Create 02_preprocess_for_ml.R
   - Create binary target variable
   - Handle sparse categories
   - Perform feature encoding
   - Split data (train/test)
   - Save processed splits

### Phase 3: Modeling with tidymodels
1. Create 03_logistic_regression.R
   - Primary model based on L1-regularized logistic regression
   - Hyperparameter tuning
   - Feature selection
   - Performance evaluation

2. Create 04_additional_models.R (optional)
   - Alternative models (Random Forest, XGBoost)
   - Comparative evaluation

### Phase 4: Results Analysis and Visualization
1. Create 05_results_visualization.R
   - Generate plots: ROC curves, confusion matrices
   - Feature importance visualization
   - Coefficient tables

### Phase 5: Integration with Quarto Document
1. Update sud_council_paper.qmd
   - Replace Python code chunks with R equivalents
   - Ensure APA formatting is consistent
   - Integrate plots and tables from R outputs

### Phase 6: Study 2 Analysis (Qualitative)
1. Create 06_qualitative_analysis.R
   - Approach 1: Traditional thematic analysis with R
   - Approach 2: Text embedding and clustering
     - Use embeddings to identify themes
     - Visualize theme clusters
     - Extract representative quotes

## Implementation Details

### Data Pipeline Consistency
To ensure consistency between Python and R results:
1. Use same random seeds
2. Apply identical preprocessing steps
3. Validate feature encoding produces same structure
4. Compare model outputs (coefficients, predictions)

### tidymodels Implementation Notes
For the L1-regularized logistic regression model:
```r
# Recipe
log_reg_recipe <- recipe(interest_dv ~ ., data = train_data) %>%
  step_normalize(all_numeric_predictors())

# Model
log_reg_spec <- logistic_reg(
  penalty = tune(),
  mixture = 1  # Lasso/L1
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# Workflow
log_reg_workflow <- workflow() %>%
  add_recipe(log_reg_recipe) %>%
  add_model(log_reg_spec)

# Tuning
log_reg_grid <- grid_regular(
  penalty(range = c(-5, 3), trans = log10_trans()),
  levels = 10
)
```

### Results Transition
For each result/output from Python to be replaced:
1. Confusion matrix visualization
2. ROC curve
3. Feature importance plot
4. Classification report
5. Coefficients table

### Study 2 Approach
For the qualitative analysis using embeddings:
1. Transcribe interviews (if not already done)
2. Use R packages for text processing:
   - `text2vec` or `textfeatures` for word embeddings
   - `quanteda` for text analysis
   - Consider OpenAI API for embeddings (via `openai` R package)
3. Use clustering algorithms to identify themes:
   - K-means or hierarchical clustering
   - t-SNE or UMAP for visualization
4. Extract and summarize themes with representative quotes

## Timeline and Milestones
1. **Week 1:** Phase 1-2 (Setup and Data Preprocessing)
2. **Week 2:** Phase 3 (Modeling)
3. **Week 3:** Phase 4-5 (Results and Quarto Integration)
4. **Week 4:** Phase 6 (Study 2 Analysis)
5. **Week 5:** Final refinements and documentation

## Key Considerations
1. **Consistency:** Ensure R results match Python results
2. **Reproducibility:** Maintain clear pipeline documentation
3. **Efficiency:** Leverage tidymodels for streamlined analysis
4. **Flexibility:** Allow for easy addition of new models or analyses
5. **APA Compatibility:** Ensure all outputs work with APA Quarto
6. **Version Control:** Commit changes incrementally to track progress

## Next Steps
1. Setup R environment and dependencies
2. Begin implementing Phase 1 tasks
3. Create skeleton scripts for each R component
4. Validate data preprocessing consistency