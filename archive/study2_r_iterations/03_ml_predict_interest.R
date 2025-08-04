#!/usr/bin/env Rscript
# 03_ml_predict_interest.R ---------------------------------------------------
# Purpose: Supervised ML to predict SUD counseling interest from text features
# Author : AI Assistant, 2025-06-30
# -------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(feather)
  library(glmnet)
  library(vip)
})

# -------------------------------------------------------------------------
# 1. Load processed data --------------------------------------------------
# -------------------------------------------------------------------------

features_path <- 'results/study2/tfidf_features.feather'
if (!file.exists(features_path)) {
  stop('TF-IDF features not found. Run 02_prepare_features.R first.')
}

df_ml <- read_feather(features_path)

message('Loaded dataset: ', nrow(df_ml), ' utterances x ', ncol(df_ml), ' columns')

# Check outcome distribution
outcome_dist <- df_ml %>%
  count(gemini_label, interested_binary) %>%
  mutate(pct = round(100 * n / sum(n), 1))

print(outcome_dist)

# -------------------------------------------------------------------------
# 2. Prepare modeling data ------------------------------------------------
# -------------------------------------------------------------------------

# Get feature column names
feature_names <- read_lines('results/study2/feature_names.txt')

# Create modeling dataset
modeling_data <- df_ml %>%
  # Convert outcome to factor for classification
  mutate(
    interested = factor(interested_binary, levels = c(0, 1), labels = c("No", "Yes")),
    speaker_id = factor(speaker)  # Include speaker as potential feature
  ) %>%
  select(utterance_id, interested, speaker_id, all_of(feature_names))

# Check for class imbalance
class_balance <- modeling_data %>% count(interested)
message('Class balance: ', paste(class_balance$n, collapse = ' vs '))

# -------------------------------------------------------------------------
# 3. Train/test split -----------------------------------------------------
# -------------------------------------------------------------------------

set.seed(2025)

# Stratified split to maintain class balance
data_split <- initial_split(modeling_data, prop = 0.8, strata = interested)
train_data <- training(data_split)
test_data <- testing(data_split)

message('Training set: ', nrow(train_data), ' utterances')
message('Test set: ', nrow(test_data), ' utterances')

# -------------------------------------------------------------------------
# 4. Model specification --------------------------------------------------
# -------------------------------------------------------------------------

# L1-regularized logistic regression (consistent with Study 1)
lasso_spec <- logistic_reg(
  penalty = tune(),
  mixture = 1  # Lasso (L1 regularization)
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# Create recipe
text_recipe <- recipe(interested ~ ., data = train_data) %>%
  step_rm(utterance_id) %>%                    # Remove ID column
  step_dummy(speaker_id) %>%                   # One-hot encode speaker
  step_normalize(all_numeric_predictors()) %>% # Normalize TF-IDF features
  step_zv(all_predictors())                    # Remove zero-variance predictors

# -------------------------------------------------------------------------
# 5. Cross-validation setup ----------------------------------------------
# -------------------------------------------------------------------------

# 5-fold CV (smaller than Study 1 due to smaller sample size)
cv_folds <- vfold_cv(train_data, v = 5, strata = interested, repeats = 3)

# Create workflow
lasso_workflow <- workflow() %>%
  add_recipe(text_recipe) %>%
  add_model(lasso_spec)

# Parameter grid for tuning
penalty_grid <- grid_regular(penalty(range = c(-5, 0)), levels = 20)

# -------------------------------------------------------------------------
# 6. Model tuning and training -------------------------------------------
# -------------------------------------------------------------------------

message('Starting hyperparameter tuning...')

# Tune hyperparameters
lasso_tune <- lasso_workflow %>%
  tune_grid(
    resamples = cv_folds,
    grid = penalty_grid,
    metrics = metric_set(accuracy, roc_auc, sensitivity, specificity),
    control = control_grid(save_pred = TRUE, verbose = FALSE)
  )

# Select best model
best_params <- lasso_tune %>%
  select_best(metric = "roc_auc")

message('Best penalty parameter: ', best_params$penalty)

# Finalize workflow with best parameters
final_workflow <- lasso_workflow %>%
  finalize_workflow(best_params)

# Fit final model on full training set
final_fit <- final_workflow %>%
  fit(train_data)

# -------------------------------------------------------------------------
# 7. Model evaluation ----------------------------------------------------
# -------------------------------------------------------------------------

# Cross-validation performance
cv_metrics <- lasso_tune %>%
  collect_metrics() %>%
  filter(penalty == best_params$penalty)

message('\nCross-validation performance:')
print(cv_metrics)

# Test set predictions
test_predictions <- final_fit %>%
  predict(test_data, type = "prob") %>%
  bind_cols(
    final_fit %>% predict(test_data),
    test_data %>% select(interested)
  )

# Test set metrics
test_metrics <- test_predictions %>%
  metrics(truth = interested, estimate = .pred_class, .pred_Yes)

message('\nTest set performance:')
print(test_metrics)

# -------------------------------------------------------------------------
# 8. Extract important features ------------------------------------------
# -------------------------------------------------------------------------

# Get model coefficients
model_coefs <- final_fit %>%
  extract_fit_parsnip() %>%
  tidy() %>%
  filter(term != "(Intercept)") %>%
  arrange(desc(abs(estimate)))

# Clean up term names (remove recipe prefixes)
model_coefs <- model_coefs %>%
  mutate(
    clean_term = str_remove_all(term, "^[^_]*_"),  # Remove step prefixes
    abs_coef = abs(estimate)
  ) %>%
  filter(abs_coef > 0)  # Only non-zero coefficients

# Top predictive words
top_words <- model_coefs %>%
  filter(!str_detect(clean_term, "speaker_id")) %>%  # Exclude speaker effects
  arrange(desc(abs_coef)) %>%
  slice_head(n = 20)

message('\nTop 20 predictive terms:')
print(top_words %>% select(clean_term, estimate, abs_coef))

# -------------------------------------------------------------------------
# 9. Save results ---------------------------------------------------------
# -------------------------------------------------------------------------

dir.create('results/study2', showWarnings = FALSE, recursive = TRUE)

# Save model performance
performance_summary <- tibble(
  metric = c("cv_roc_auc", "cv_accuracy", "test_roc_auc", "test_accuracy"),
  value = c(
    cv_metrics$mean[cv_metrics$.metric == "roc_auc"],
    cv_metrics$mean[cv_metrics$.metric == "accuracy"],
    test_metrics$.estimate[test_metrics$.metric == "roc_auc"],
    test_metrics$.estimate[test_metrics$.metric == "accuracy"]
  )
)

write_csv(performance_summary, 'results/study2/model_performance.csv')

# Save coefficients/word importance
write_csv(model_coefs, 'results/study2/word_importance.csv')
write_csv(top_words, 'results/study2/top_predictive_words.csv')

# Save predictions for further analysis
predictions_with_text <- test_data %>%
  select(utterance_id, interested) %>%
  left_join(df_ml %>% select(utterance_id, text, session, speaker), 
            by = "utterance_id") %>%
  bind_cols(
    test_predictions %>% select(.pred_class, .pred_Yes)
  )

write_csv(predictions_with_text, 'results/study2/test_predictions.csv')

# Save trained model
saveRDS(final_fit, 'results/study2/trained_model.rds')

# -------------------------------------------------------------------------
# 10. Generate interpretable summary -------------------------------------
# -------------------------------------------------------------------------

# Words that increase interest probability
interest_words <- top_words %>%
  filter(estimate > 0) %>%
  arrange(desc(estimate)) %>%
  pull(clean_term)

# Words that decrease interest probability  
disinterest_words <- top_words %>%
  filter(estimate < 0) %>%
  arrange(estimate) %>%
  pull(clean_term)

# Create summary report
summary_report <- glue::glue("
STUDY 2: Text Analysis Results Summary
=====================================

Dataset: {nrow(df_ml)} labeled utterances
Training: {nrow(train_data)} utterances
Testing: {nrow(test_data)} utterances

Model Performance:
- Cross-validation ROC AUC: {round(performance_summary$value[1], 3)}
- Test set ROC AUC: {round(performance_summary$value[3], 3)}
- Test set accuracy: {round(performance_summary$value[4], 3)}

Top words predicting INTEREST in SUD counseling:
{paste(head(interest_words, 10), collapse = ', ')}

Top words predicting DISINTEREST in SUD counseling:
{paste(head(disinterest_words, 10), collapse = ', ')}

Active features: {nrow(model_coefs)} out of {length(feature_names)} total features
Sparsity achieved: {round(100 * (1 - nrow(model_coefs)/length(feature_names)), 1)}%

")

writeLines(summary_report, 'results/study2/analysis_summary.txt')

message('\n✅ Supervised ML analysis complete!')
message('Results saved to results/study2/:')
message('  - model_performance.csv (key metrics)')
message('  - word_importance.csv (all coefficients)')
message('  - top_predictive_words.csv (most important terms)')
message('  - test_predictions.csv (detailed predictions)')
message('  - trained_model.rds (fitted model)')
message('  - analysis_summary.txt (interpretable summary)')

# Print final summary
cat(summary_report)