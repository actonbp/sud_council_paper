#!/usr/bin/env Rscript
# 03_ml_predict_interest_participant.R ----------------------------------------
# Purpose: Supervised ML to predict SUD counseling interest from participant text
# Author : AI Assistant, 2025-08-01
# Note   : Adapted for small sample (N=40 participants)
# -------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(arrow)
  library(glmnet)
  library(vip)
  library(here)
})

# -------------------------------------------------------------------------
# 1. Load processed data --------------------------------------------------
# -------------------------------------------------------------------------

features_path <- 'results/study2/tfidf_participant.feather'
if (!file.exists(features_path)) {
  stop('TF-IDF features not found. Run 02_prepare_features_participant.R first.')
}

df_ml <- read_feather(features_path)

message('Loaded dataset: ', nrow(df_ml), ' participants x ', ncol(df_ml), ' columns')

# Check outcome distribution
outcome_dist <- df_ml %>%
  count(ai_label, interested_binary) %>%
  mutate(pct = round(100 * n / sum(n), 1))

print(outcome_dist)

# -------------------------------------------------------------------------
# 2. Prepare modeling data ------------------------------------------------
# -------------------------------------------------------------------------

# Get feature column names
feature_names <- read_lines('results/study2/participant_feature_names.txt')

# Create modeling dataset
modeling_data <- df_ml %>%
  # Convert outcome to factor for classification
  mutate(
    interested = factor(interested_binary, levels = c(0, 1), labels = c("No", "Yes"))
  ) %>%
  select(participant_id, interested, all_of(feature_names))

# Check for zero variance features
zero_var_features <- modeling_data %>%
  select(all_of(feature_names)) %>%
  summarise(across(everything(), ~var(.x, na.rm = TRUE))) %>%
  pivot_longer(everything()) %>%
  filter(value == 0) %>%
  pull(name)

if (length(zero_var_features) > 0) {
  message('Removing ', length(zero_var_features), ' zero-variance features')
  feature_names <- setdiff(feature_names, zero_var_features)
}

# -------------------------------------------------------------------------
# 3. Model specification --------------------------------------------------
# -------------------------------------------------------------------------

# L1-regularized logistic regression with elastic net option
elastic_spec <- logistic_reg(
  penalty = tune(),
  mixture = tune()  # Allow both L1 and L2
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# Create recipe
text_recipe <- recipe(interested ~ ., data = modeling_data) %>%
  step_rm(participant_id) %>%                    # Remove ID column
  step_normalize(all_numeric_predictors()) %>%   # Normalize TF-IDF features
  step_zv(all_predictors())                      # Remove zero-variance predictors

# -------------------------------------------------------------------------
# 4. Cross-validation setup ----------------------------------------------
# -------------------------------------------------------------------------

set.seed(2025)

# With only 40 participants, use 5-fold CV with 5 repeats
cv_folds <- vfold_cv(modeling_data, v = 5, strata = interested, repeats = 5)

message('Cross-validation: 5-fold with 5 repeats (25 total resamples)')
message('Each fold has ~8 participants, stratified by outcome')

# Create workflow
elastic_workflow <- workflow() %>%
  add_recipe(text_recipe) %>%
  add_model(elastic_spec)

# Parameter grid - include both penalty and mixture
param_grid <- grid_regular(
  penalty(range = c(-5, 1)),
  mixture(range = c(0, 1)),
  levels = c(20, 5)  # 20 penalty values, 5 mixture values
)

# -------------------------------------------------------------------------
# 5. Model tuning and training -------------------------------------------
# -------------------------------------------------------------------------

message('\nStarting hyperparameter tuning...')
message('Testing ', nrow(param_grid), ' parameter combinations')

# Tune hyperparameters
elastic_tune <- elastic_workflow %>%
  tune_grid(
    resamples = cv_folds,
    grid = param_grid,
    metrics = metric_set(accuracy, roc_auc, sensitivity, specificity),
    control = control_grid(save_pred = TRUE, verbose = FALSE)
  )

# Show top 10 models
top_models <- elastic_tune %>%
  show_best(metric = "roc_auc", n = 10)

message('\nTop 10 models by ROC AUC:')
print(top_models)

# Select best model
best_params <- elastic_tune %>%
  select_best(metric = "roc_auc")

message('\nBest parameters:')
message('  Penalty: ', round(best_params$penalty, 5))
message('  Mixture: ', round(best_params$mixture, 2), 
        ' (0=Ridge, 1=Lasso, between=Elastic Net)')

# -------------------------------------------------------------------------
# 6. Evaluate best model with CV --------------------------------------
# -------------------------------------------------------------------------

# Get detailed CV metrics for best model
best_metrics <- elastic_tune %>%
  collect_metrics() %>%
  filter(
    penalty == best_params$penalty,
    mixture == best_params$mixture
  )

message('\nBest model cross-validation performance:')
print(best_metrics)

# Get predictions from best model
best_predictions <- elastic_tune %>%
  collect_predictions() %>%
  filter(
    penalty == best_params$penalty,
    mixture == best_params$mixture
  )

# Calculate confidence intervals via bootstrap
bootstrap_metrics <- best_predictions %>%
  group_by(id, id2) %>%  # Group by repeat and fold
  roc_auc(truth = interested, .pred_Yes) %>%
  ungroup()

roc_ci <- quantile(bootstrap_metrics$.estimate, c(0.025, 0.975))

message(sprintf('\nROC AUC: %.3f [95%% CI: %.3f-%.3f]',
                best_metrics$mean[best_metrics$.metric == "roc_auc"],
                roc_ci[1], roc_ci[2]))

# -------------------------------------------------------------------------
# 7. Final model on all data --------------------------------------------
# -------------------------------------------------------------------------

# Finalize workflow with best parameters
final_workflow <- elastic_workflow %>%
  finalize_workflow(best_params)

# Fit final model on all data
final_fit <- final_workflow %>%
  fit(modeling_data)

# -------------------------------------------------------------------------
# 8. Extract important features ------------------------------------------
# -------------------------------------------------------------------------

# Get model coefficients
model_coefs <- final_fit %>%
  extract_fit_parsnip() %>%
  tidy() %>%
  filter(term != "(Intercept)") %>%
  mutate(
    abs_estimate = abs(estimate),
    direction = ifelse(estimate > 0, "Increases interest", "Decreases interest")
  ) %>%
  arrange(desc(abs_estimate))

# Non-zero coefficients only
active_coefs <- model_coefs %>%
  filter(abs_estimate > 0)

message('\nActive features: ', nrow(active_coefs), ' out of ', 
        length(feature_names), ' (', 
        round(100 * nrow(active_coefs)/length(feature_names), 1), '% selected)')

# Top predictive terms
top_terms <- active_coefs %>%
  slice_head(n = 30)

message('\nTop 15 terms increasing interest:')
top_terms %>%
  filter(direction == "Increases interest") %>%
  slice_head(n = 15) %>%
  select(term, estimate, abs_estimate) %>%
  print()

message('\nTop 15 terms decreasing interest:')
top_terms %>%
  filter(direction == "Decreases interest") %>%
  slice_head(n = 15) %>%
  select(term, estimate, abs_estimate) %>%
  print()

# -------------------------------------------------------------------------
# 9. Stability analysis via bootstrap -------------------------------------
# -------------------------------------------------------------------------

message('\nRunning bootstrap stability analysis (100 iterations)...')

# Bootstrap to assess coefficient stability
set.seed(2025)
n_boot <- 100
boot_coefs <- tibble()

pb <- txtProgressBar(min = 0, max = n_boot, style = 3)
for (i in 1:n_boot) {
  # Resample with replacement
  boot_indices <- sample(1:nrow(modeling_data), replace = TRUE)
  boot_data <- modeling_data[boot_indices, ]
  
  # Fit model
  boot_fit <- try(
    final_workflow %>% fit(boot_data),
    silent = TRUE
  )
  
  if (!inherits(boot_fit, "try-error")) {
    # Extract coefficients
    boot_coef_i <- boot_fit %>%
      extract_fit_parsnip() %>%
      tidy() %>%
      filter(term != "(Intercept)") %>%
      mutate(boot_iter = i)
    
    boot_coefs <- bind_rows(boot_coefs, boot_coef_i)
  }
  
  setTxtProgressBar(pb, i)
}
close(pb)

# Summarize bootstrap results
boot_summary <- boot_coefs %>%
  group_by(term) %>%
  summarise(
    mean_coef = mean(estimate),
    sd_coef = sd(estimate),
    pct_nonzero = 100 * mean(estimate != 0),
    pct_positive = 100 * mean(estimate > 0),
    pct_negative = 100 * mean(estimate < 0),
    .groups = 'drop'
  ) %>%
  arrange(desc(pct_nonzero))

message('\n\nMost stable features (selected in >50% of bootstraps):')
boot_summary %>%
  filter(pct_nonzero > 50) %>%
  print(n = 20)

# -------------------------------------------------------------------------
# 10. Save results --------------------------------------------------------
# -------------------------------------------------------------------------

dir.create('results/study2', showWarnings = FALSE, recursive = TRUE)

# Save model performance
performance_summary <- tibble(
  metric = c("cv_roc_auc", "cv_roc_auc_lower", "cv_roc_auc_upper",
             "cv_accuracy", "cv_sensitivity", "cv_specificity",
             "best_penalty", "best_mixture", "n_active_features"),
  value = c(
    best_metrics$mean[best_metrics$.metric == "roc_auc"],
    roc_ci[1], roc_ci[2],
    best_metrics$mean[best_metrics$.metric == "accuracy"],
    best_metrics$mean[best_metrics$.metric == "sensitivity"],
    best_metrics$mean[best_metrics$.metric == "specificity"],
    best_params$penalty,
    best_params$mixture,
    nrow(active_coefs)
  )
)

write_csv(performance_summary, 'results/study2/participant_model_performance.csv')

# Save coefficients
write_csv(active_coefs, 'results/study2/participant_word_importance.csv')
write_csv(boot_summary, 'results/study2/bootstrap_coefficient_stability.csv')

# Save model
saveRDS(final_fit, 'results/study2/participant_trained_model.rds')

# -------------------------------------------------------------------------
# 11. Generate interpretable summary --------------------------------------
# -------------------------------------------------------------------------

# Create summary report
summary_report <- glue::glue("
STUDY 2: Participant-Level Text Analysis Results
===============================================

Dataset: {nrow(df_ml)} participants ({sum(df_ml$interested_binary)} interested, {sum(!df_ml$interested_binary)} not interested)
Features: {length(feature_names)} TF-IDF terms

Model Selection:
- Tested {nrow(param_grid)} hyperparameter combinations
- Best penalty: {round(best_params$penalty, 5)}
- Best mixture: {round(best_params$mixture, 2)} ({ifelse(best_params$mixture == 0, 'Ridge', ifelse(best_params$mixture == 1, 'Lasso', 'Elastic Net'))})

Cross-Validation Performance (5-fold, 5 repeats):
- ROC AUC: {round(best_metrics$mean[best_metrics$.metric == 'roc_auc'], 3)} [95% CI: {round(roc_ci[1], 3)}-{round(roc_ci[2], 3)}]
- Accuracy: {round(best_metrics$mean[best_metrics$.metric == 'accuracy'], 3)}
- Sensitivity: {round(best_metrics$mean[best_metrics$.metric == 'sensitivity'], 3)}
- Specificity: {round(best_metrics$mean[best_metrics$.metric == 'specificity'], 3)}

Feature Selection:
- Active features: {nrow(active_coefs)} out of {length(feature_names)} ({round(100 * nrow(active_coefs)/length(feature_names), 1)}%)
- Features stable in >75% bootstraps: {sum(boot_summary$pct_nonzero > 75)}

Key Terms Predicting INTEREST:
{paste(head(active_coefs$term[active_coefs$direction == 'Increases interest'], 10), collapse = ', ')}

Key Terms Predicting DISINTEREST:
{paste(head(active_coefs$term[active_coefs$direction == 'Decreases interest'], 10), collapse = ', ')}

")

writeLines(summary_report, 'results/study2/participant_analysis_summary.txt')

message('\n✅ Participant-level ML analysis complete!')
message('Results saved to results/study2/:')
message('  - participant_model_performance.csv (metrics)')
message('  - participant_word_importance.csv (coefficients)')
message('  - bootstrap_coefficient_stability.csv (stability analysis)')
message('  - participant_trained_model.rds (fitted model)')
message('  - participant_analysis_summary.txt (summary)')

# Print final summary
cat(summary_report)