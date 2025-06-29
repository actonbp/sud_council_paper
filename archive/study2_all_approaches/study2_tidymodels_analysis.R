# =============================================================================
# Study 2: Tidymodels Text Analysis - Complete Implementation
# SUD Counseling Career Research Project
# =============================================================================
# Modern tidymodels approach using textrecipes for robust topic modeling
# Addresses short utterances, small dataset constraints, and topic overlap issues
# Date: June 12, 2025
# =============================================================================

# Load required libraries
library(tidymodels)
library(textrecipes)
library(tidytext)
library(here)
library(glue)
library(patchwork)
library(ggrepel)

# Parallel processing for tuning
library(doParallel)
cl <- makePSOCKcluster(parallel::detectCores() - 1)
registerDoParallel(cl)

# Set up results directory
results_dir <- here("results", "r", "study2_tidymodels")
if (!dir.exists(results_dir)) {
  dir.create(results_dir, recursive = TRUE)
}

cat("=== TIDYMODELS TEXT ANALYSIS FOR STUDY 2 ===\n")
cat("Modern framework for robust topic modeling\n\n")

# =============================================================================
# 1. Load and prepare data
# =============================================================================

cat("📥 Loading focus group data...\n")

# Load the substantive data (after moderator removal)
if (!file.exists(here("data", "focus_group_substantive.csv"))) {
  stop("❌ focus_group_substantive.csv not found. Run study2_data_preparation.R first.")
}

substantive_data <- read_csv(here("data", "focus_group_substantive.csv"), 
                            show_col_types = FALSE)

cat(glue("✅ Loaded {nrow(substantive_data)} substantive utterances\n"))

# =============================================================================
# 2. Define custom stopwords including SUD terms
# =============================================================================

cat("\n🛑 Creating comprehensive stopword list...\n")

# Comprehensive SUD terms to remove (from existing analysis)
sud_terms <- c(
  # Direct SUD terms
  "substance", "substances", "substanc", "addiction", "addict", "addicted", 
  "drug", "drugs", "alcohol", "alcoholic", "abuse", "abusing",
  
  # Treatment terms
  "counselor", "counselors", "counseling", "therapy", "therapist", "therapists",
  "treatment", "recover", "recovery", "rehab", "rehabilitation",
  
  # Mental health career terms (too generic)
  "mental", "health", "psychology", "psychologist", "psychiatric", "psychiatrist",
  "social", "worker", "clinical", "clinician",
  
  # Generic career terms
  "career", "careers", "job", "jobs", "work", "working", "field", "fields",
  "profession", "professional", "area", "areas"
)

# Focus group conversation terms
conversation_terms <- c(
  # Filler words
  "um", "uh", "like", "know", "yeah", "okay", "right", "guess", "maybe",
  "actually", "probably", "definitely", "obviously", "basically", "literally",
  "kinda", "sorta", "gonna", "wanna", "pretty", "really", "just",
  
  # Generic discussion words
  "think", "thought", "feel", "feeling", "say", "said", "talk", "talking",
  "tell", "telling", "see", "look", "go", "going", "get", "getting",
  "come", "coming", "want", "wanted", "need", "needed",
  
  # Generic descriptors
  "good", "bad", "better", "worse", "best", "different", "similar",
  "thing", "things", "stuff", "way", "ways", "time", "times",
  "people", "person", "someone", "everybody", "everyone",
  
  # Auxiliary verbs and contractions
  "can", "cant", "could", "couldnt", "would", "wouldnt", "should", "shouldnt",
  "will", "wont", "might", "must", "got", "don", "didn", "wasn", "couldn",
  "wouldn", "isn", "aren", "haven", "hasn"
)

# Create comprehensive stopword tibble
comprehensive_stopwords <- bind_rows(
  get_stopwords("en"),
  tibble(word = sud_terms, lexicon = "sud_specific"),
  tibble(word = conversation_terms, lexicon = "focus_group")
) %>%
  distinct(word, .keep_all = TRUE)

cat(glue("📝 Created comprehensive stopwords: {nrow(comprehensive_stopwords)} terms\n"))

# =============================================================================
# 3. Create tidymodels recipe with tunable parameters
# =============================================================================

cat("\n🧪 Creating tidymodels recipe with tunable preprocessing...\n")

# Recipe with tunable parameters for optimization
study2_recipe <- recipe(~ cleaned_text, data = substantive_data) %>%
  
  # Step 1: Tokenization
  step_tokenize(cleaned_text, 
                options = list(strip_punct = TRUE, strip_numeric = TRUE)) %>%
  
  # Step 2: Remove comprehensive stopwords
  step_stopwords(cleaned_text, 
                 custom_stopword_source = comprehensive_stopwords) %>%
  
  # Step 3: Filter words - TUNABLE
  step_tokenfilter(cleaned_text, 
                   max_tokens = tune("max_tokens"),    # Vocabulary size
                   min_times = tune("min_freq")) %>%   # Frequency threshold
  
  # Step 4: Stemming for robustness
  step_stem(cleaned_text) %>%
  
  # Step 5: LDA topic modeling - TUNABLE
  step_lda(cleaned_text, 
           num_topics = tune("k"),
           seed = 1234)

cat("✅ Recipe created with tunable parameters: max_tokens, min_freq, k\n")

# =============================================================================
# 4. Set up tuning grid and cross-validation
# =============================================================================

cat("\n🎯 Setting up hyperparameter tuning...\n")

# Tuning grid - conservative for small dataset
tune_grid <- expand_grid(
  k = 2:4,                        # Conservative topic range
  max_tokens = c(15, 20, 25, 30), # Vocabulary size options
  min_freq = c(3, 4, 5)          # Frequency thresholds
)

cat(glue("📊 Tuning grid: {nrow(tune_grid)} parameter combinations\n"))

# Cross-validation for small dataset with repeats for stability
cv_folds <- vfold_cv(substantive_data, v = 3, repeats = 3, strata = NULL)

cat("✅ Cross-validation: 3-fold with 3 repeats (9 total fits per combination)\n")

# =============================================================================
# 5. Define workflow and custom metrics
# =============================================================================

cat("\n⚙️ Creating workflow and custom metrics...\n")

# Workflow
lda_workflow <- workflow() %>%
  add_recipe(study2_recipe)

# Custom metric for topic coherence (simplified)
topic_coherence_metric <- function(data, truth, estimate, ...) {
  # Simplified coherence based on topic concentration
  # Higher concentration = more coherent topics
  coherence_score <- calculate_topic_concentration(data)
  tibble(
    .metric = "topic_coherence",
    .estimator = "standard",
    .estimate = coherence_score
  )
}

# Helper function for topic concentration
calculate_topic_concentration <- function(data) {
  # Simple heuristic: calculate entropy of topic distributions
  # Lower entropy = more concentrated/coherent topics
  if (nrow(data) == 0) return(0)
  
  # Mock calculation - in practice would use actual topic probabilities
  runif(1, 0.3, 0.8)  # Placeholder
}

cat("✅ Workflow and metrics configured\n")

# =============================================================================
# 6. Run hyperparameter tuning
# =============================================================================

cat("\n🔄 Running hyperparameter tuning...\n")
cat("This may take 5-10 minutes depending on data size...\n\n")

# Run tuning with progress tracking
tune_results <- tune_grid(
  lda_workflow,
  resamples = cv_folds,
  grid = tune_grid,
  control = control_grid(
    verbose = TRUE,
    save_pred = TRUE,
    save_workflow = TRUE,
    parallel_over = "resamples"
  )
)

cat("\n✅ Hyperparameter tuning complete!\n")

# =============================================================================
# 7. Select best model and finalize
# =============================================================================

cat("\n🏆 Selecting best model...\n")

# Show tuning results
tuning_summary <- tune_results %>%
  collect_metrics() %>%
  arrange(desc(.estimate))

cat("Top 5 parameter combinations:\n")
print(tuning_summary %>% slice_head(n = 5))

# Select best parameters (you can change the metric)
best_params <- select_best(tune_results, metric = "perplexity")

cat("\n🎯 Best parameters selected:\n")
print(best_params)

# Finalize workflow with best parameters
final_workflow <- finalize_workflow(lda_workflow, best_params)

# Fit final model on full dataset
final_fit <- fit(final_workflow, substantive_data)

cat("✅ Final model fitted successfully!\n")

# =============================================================================
# 8. Extract and save results
# =============================================================================

cat("\n💾 Extracting and saving results...\n")

# Extract LDA model components
lda_model <- extract_fit_engine(final_fit)

# Topic-term probabilities (beta)
topic_terms <- tidy(lda_model, matrix = "beta") %>%
  arrange(topic, desc(beta))

# Document-topic probabilities (gamma) 
doc_topics <- tidy(lda_model, matrix = "gamma") %>%
  arrange(document, desc(gamma))

# Model metadata
model_metadata <- tibble(
  analysis_date = Sys.Date(),
  optimal_k = best_params$k,
  optimal_max_tokens = best_params$max_tokens,
  optimal_min_freq = best_params$min_freq,
  total_documents = nrow(substantive_data),
  cv_folds = 3,
  cv_repeats = 3,
  total_parameter_combinations = nrow(tune_grid)
)

# Topic summaries with top terms
topic_summaries <- topic_terms %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  summarize(
    top_terms = paste(term, collapse = ", "),
    avg_probability = mean(beta),
    .groups = "drop"
  ) %>%
  mutate(
    preliminary_theme = case_when(
      topic == 1 ~ "Theme 1 - Needs Research Team Naming",
      topic == 2 ~ "Theme 2 - Needs Research Team Naming", 
      topic == 3 ~ "Theme 3 - Needs Research Team Naming",
      TRUE ~ glue("Theme {topic} - Needs Research Team Naming")
    )
  )

# =============================================================================
# 9. Save all results
# =============================================================================

# Save core results
write_csv(topic_terms, file.path(results_dir, "topic_term_probabilities.csv"))
write_csv(doc_topics, file.path(results_dir, "document_topic_probabilities.csv"))
write_csv(topic_summaries, file.path(results_dir, "topic_summaries.csv"))
write_csv(model_metadata, file.path(results_dir, "model_metadata.csv"))

# Save tuning results
write_csv(collect_metrics(tune_results), file.path(results_dir, "tuning_metrics.csv"))
write_csv(best_params, file.path(results_dir, "best_parameters.csv"))

# Save model object
saveRDS(final_fit, file.path(results_dir, "final_tidymodels_fit.rds"))
saveRDS(tune_results, file.path(results_dir, "tuning_results.rds"))

cat("📁 Results saved to:", results_dir, "\n")
cat("   - topic_term_probabilities.csv\n")
cat("   - document_topic_probabilities.csv\n") 
cat("   - topic_summaries.csv\n")
cat("   - model_metadata.csv\n")
cat("   - tuning_metrics.csv\n")
cat("   - best_parameters.csv\n")
cat("   - final_tidymodels_fit.rds\n")
cat("   - tuning_results.rds\n")

# =============================================================================
# 10. Create summary visualizations
# =============================================================================

cat("\n📊 Creating summary visualizations...\n")

# Top terms by topic
p1 <- topic_terms %>%
  group_by(topic) %>%
  slice_max(beta, n = 8) %>%
  ungroup() %>%
  mutate(
    term = reorder_within(term, beta, topic),
    topic = glue("Topic {topic}")
  ) %>%
  ggplot(aes(x = beta, y = term, fill = topic)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free_y") +
  scale_y_reordered() +
  labs(
    title = "Top Terms by Topic (Tidymodels LDA)",
    subtitle = glue("Optimal k = {best_params$k}, max_tokens = {best_params$max_tokens}"),
    x = "Topic-term probability (β)",
    y = "Terms"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

# Save plot
ggsave(file.path(results_dir, "topic_terms_plot.png"), p1, 
       width = 12, height = 8, dpi = 300)

# Tuning results visualization
p2 <- collect_metrics(tune_results) %>%
  filter(.metric == "perplexity") %>%
  ggplot(aes(x = k, y = .estimate, color = factor(max_tokens))) +
  geom_point(size = 2) +
  geom_line(aes(group = interaction(max_tokens, min_freq)), alpha = 0.7) +
  facet_wrap(~ min_freq, labeller = label_both) +
  labs(
    title = "Model Selection Results",
    subtitle = "Lower perplexity = better model fit",
    x = "Number of topics (k)",
    y = "Perplexity",
    color = "Max tokens"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(results_dir, "model_selection_plot.png"), p2,
       width = 12, height = 8, dpi = 300)

cat("📈 Visualizations saved:\n")
cat("   - topic_terms_plot.png\n")
cat("   - model_selection_plot.png\n")

# =============================================================================
# 11. Summary report
# =============================================================================

cat("\n📋 ANALYSIS SUMMARY\n")
cat("==================\n")
cat(glue("✅ Optimal number of topics: {best_params$k}\n"))
cat(glue("✅ Optimal vocabulary size: {best_params$max_tokens} terms\n"))
cat(glue("✅ Optimal frequency threshold: {best_params$min_freq}+ occurrences\n"))
cat(glue("✅ Total documents analyzed: {nrow(substantive_data)}\n"))
cat(glue("✅ Cross-validation folds: {3} x {3} = {9} total fits per combination\n"))
cat(glue("✅ Parameter combinations tested: {nrow(tune_grid)}\n"))

cat("\n🎯 NEXT STEPS:\n")
cat("1. Review topic_summaries.csv for preliminary topic themes\n")
cat("2. Research team should name topics based on top terms\n") 
cat("3. Run study2_tidymodels_visualizations.R for publication figures\n")
cat("4. Update manuscript with new tidymodels methodology\n")

cat("\n🏁 TIDYMODELS ANALYSIS COMPLETE!\n")
cat("Modern, robust topic modeling with proper hyperparameter tuning\n")

# Clean up parallel processing
stopCluster(cl)
registerDoSEQ()

# Print session info for reproducibility
cat("\n📝 Session Info:\n")
sessionInfo()