# =============================================================================
# Study 2: Final Analysis with Optimized Parameters  
# SUD Counseling Career Research Project
# =============================================================================
# Runs final analysis with parameters determined through iterative tuning
# Date: June 12, 2025
# =============================================================================

library(tidymodels)
library(textrecipes)
library(tidytext)
library(here)
library(glue)

cat("=== FINAL TIDYMODELS ANALYSIS ===\n")
cat("Using optimized parameters from iterative tuning\n\n")

# =============================================================================
# 🎯 OPTIMIZED PARAMETERS - SET THESE BASED ON ITERATION RESULTS
# =============================================================================

# ⚠️ EDIT THESE VALUES BASED ON YOUR ITERATIVE TUNING RESULTS:

FINAL_K <- 3                    # Number of topics  
FINAL_MAX_TOKENS <- 20          # Vocabulary size
FINAL_MIN_FREQ <- 4             # Minimum term frequency

# Additional stopwords found during iteration
ADDITIONAL_STOPWORDS <- c(
  # Add any problematic terms identified during iteration:
  # "help", "work", "feel", "think", etc.
)

cat(glue("🎯 Final parameters: k={FINAL_K}, max_tokens={FINAL_MAX_TOKENS}, min_freq={FINAL_MIN_FREQ}\n"))
cat(glue("🛑 Additional stopwords: {length(ADDITIONAL_STOPWORDS)} terms\n\n"))

# =============================================================================
# Setup and data loading
# =============================================================================

# Set up results directory
results_dir <- here("results", "r", "study2_final")
if (!dir.exists(results_dir)) {
  dir.create(results_dir, recursive = TRUE)
}

# Load data
if (!file.exists(here("data", "focus_group_substantive.csv"))) {
  stop("❌ focus_group_substantive.csv not found. Run study2_data_preparation.R first.")
}

substantive_data <- read_csv(here("data", "focus_group_substantive.csv"), 
                            show_col_types = FALSE)

cat(glue("✅ Loaded {nrow(substantive_data)} substantive utterances\n"))

# =============================================================================
# Comprehensive stopword list
# =============================================================================

# Base stopwords (same as iterative tuning)
sud_terms <- c(
  "substance", "substances", "substanc", "addiction", "addict", "addicted", 
  "drug", "drugs", "alcohol", "alcoholic", "abuse", "abusing",
  "counselor", "counselors", "counseling", "therapy", "therapist", "therapists",
  "treatment", "recover", "recovery", "rehab", "rehabilitation",
  "mental", "health", "psychology", "psychologist", "psychiatric", "psychiatrist",
  "social", "worker", "clinical", "clinician",
  "career", "careers", "job", "jobs", "work", "working", "field", "fields",
  "profession", "professional", "area", "areas"
)

conversation_terms <- c(
  "um", "uh", "like", "know", "yeah", "okay", "right", "guess", "maybe",
  "actually", "probably", "definitely", "obviously", "basically", "literally",
  "kinda", "sorta", "gonna", "wanna", "pretty", "really", "just",
  "think", "thought", "feel", "feeling", "say", "said", "talk", "talking",
  "tell", "telling", "see", "look", "go", "going", "get", "getting",
  "come", "coming", "want", "wanted", "need", "needed",
  "good", "bad", "better", "worse", "best", "different", "similar",
  "thing", "things", "stuff", "way", "ways", "time", "times",
  "people", "person", "someone", "everybody", "everyone",
  "can", "cant", "could", "couldnt", "would", "wouldnt", "should", "shouldnt",
  "will", "wont", "might", "must", "got", "don", "didn", "wasn", "couldn",
  "wouldn", "isn", "aren", "haven", "hasn"
)

# Comprehensive stopwords
comprehensive_stopwords <- bind_rows(
  get_stopwords("en"),
  tibble(word = sud_terms, lexicon = "sud_specific"),
  tibble(word = conversation_terms, lexicon = "focus_group"),
  tibble(word = ADDITIONAL_STOPWORDS, lexicon = "final_custom")
) %>%
  distinct(word, .keep_all = TRUE)

cat(glue("🛑 Total stopwords: {nrow(comprehensive_stopwords)} terms\n"))

# =============================================================================
# Create final recipe with optimized parameters
# =============================================================================

cat("\n🧪 Creating final recipe with optimized parameters...\n")

final_recipe <- recipe(~ cleaned_text, data = substantive_data) %>%
  step_tokenize(cleaned_text, 
                options = list(strip_punct = TRUE, strip_numeric = TRUE)) %>%
  step_stopwords(cleaned_text, 
                 custom_stopword_source = comprehensive_stopwords) %>%
  step_tokenfilter(cleaned_text, 
                   max_tokens = FINAL_MAX_TOKENS,
                   min_times = FINAL_MIN_FREQ) %>%
  step_stem(cleaned_text) %>%
  step_lda(cleaned_text, 
           num_topics = FINAL_K,
           seed = 1234)

# Create workflow
final_workflow <- workflow() %>%
  add_recipe(final_recipe)

cat("✅ Final recipe and workflow created\n")

# =============================================================================
# Fit final model
# =============================================================================

cat("\n🚀 Fitting final model...\n")

final_fit <- fit(final_workflow, substantive_data)

cat("✅ Final model fitted successfully!\n")

# =============================================================================
# Extract comprehensive results
# =============================================================================

cat("\n📊 Extracting results...\n")

# Extract LDA model
lda_model <- extract_fit_engine(final_fit)

# Topic-term probabilities (beta)
topic_terms <- tidy(lda_model, matrix = "beta") %>%
  arrange(topic, desc(beta))

# Document-topic probabilities (gamma)
doc_topics <- tidy(lda_model, matrix = "gamma") %>%
  arrange(document, desc(gamma))

# Calculate topic summaries
topic_summaries <- topic_terms %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  summarize(
    top_terms = paste(term[1:8], collapse = ", "),
    top_terms_with_probs = paste(glue("{term[1:8]} ({round(beta[1:8]*100, 1)}%)"), collapse = ", "),
    avg_probability = mean(beta),
    max_probability = max(beta),
    .groups = "drop"
  ) %>%
  mutate(
    preliminary_theme = glue("Topic {topic} - [Research Team to Name]")
  )

# Model metadata
model_metadata <- tibble(
  analysis_date = Sys.Date(),
  final_k = FINAL_K,
  final_max_tokens = FINAL_MAX_TOKENS,
  final_min_freq = FINAL_MIN_FREQ,
  total_documents = nrow(substantive_data),
  vocabulary_size = length(unique(topic_terms$term)),
  additional_stopwords_count = length(ADDITIONAL_STOPWORDS),
  methodology = "Tidymodels LDA with iteratively optimized parameters"
)

# Topic prevalence analysis
topic_prevalence <- doc_topics %>%
  group_by(topic) %>%
  summarize(
    avg_gamma = mean(gamma),
    documents_dominant = sum(gamma > 0.5),
    total_docs = n(),
    prevalence_pct = documents_dominant / total_docs * 100,
    .groups = "drop"
  ) %>%
  arrange(desc(avg_gamma))

cat("✅ Results extracted and processed\n")

# =============================================================================
# Display results preview
# =============================================================================

cat("\n📋 FINAL RESULTS PREVIEW\n")
cat("========================\n")

for (i in 1:nrow(topic_summaries)) {
  topic_info <- topic_summaries[i, ]
  prevalence_info <- topic_prevalence[topic_prevalence$topic == topic_info$topic, ]
  
  cat(glue("\n🎯 TOPIC {topic_info$topic} (Prevalence: {round(prevalence_info$avg_gamma * 100, 1)}%)\n"))
  cat(glue("Top terms: {topic_info$top_terms}\n"))
  cat(glue("Max term probability: {round(topic_info$max_probability * 100, 1)}%\n"))
}

cat("\n📊 TOPIC QUALITY SUMMARY\n")
cat("========================\n")

# Quick quality metrics
avg_max_prob <- mean(topic_summaries$max_probability)
vocab_size <- length(unique(topic_terms$term))

cat(glue("📈 Average max term probability: {round(avg_max_prob * 100, 1)}%\n"))
cat(glue("📚 Final vocabulary size: {vocab_size} terms\n"))
cat(glue("🎯 Topics created: {FINAL_K}\n"))
cat(glue("📄 Documents analyzed: {nrow(substantive_data)}\n"))

# =============================================================================
# Save all results
# =============================================================================

cat("\n💾 Saving final results...\n")

# Core results
write_csv(topic_terms, file.path(results_dir, "final_topic_term_probabilities.csv"))
write_csv(doc_topics, file.path(results_dir, "final_document_topic_probabilities.csv"))
write_csv(topic_summaries, file.path(results_dir, "final_topic_summaries.csv"))
write_csv(topic_prevalence, file.path(results_dir, "final_topic_prevalence.csv"))
write_csv(model_metadata, file.path(results_dir, "final_model_metadata.csv"))

# Save model object
saveRDS(final_fit, file.path(results_dir, "final_fitted_model.rds"))

# Create manuscript table
manuscript_table <- topic_summaries %>%
  left_join(topic_prevalence, by = "topic") %>%
  select(
    Topic = topic,
    `Top Terms` = top_terms,
    `Max Term Probability` = max_probability,
    `Average Prevalence` = avg_gamma,
    `Preliminary Theme` = preliminary_theme
  ) %>%
  mutate(
    `Max Term Probability` = round(`Max Term Probability`, 3),
    `Average Prevalence` = round(`Average Prevalence`, 3)
  )

write_csv(manuscript_table, file.path(results_dir, "manuscript_table_final_topics.csv"))

cat("📁 Final results saved to:", results_dir, "\n")
cat("Files created:\n")
cat("   - final_topic_term_probabilities.csv\n")
cat("   - final_document_topic_probabilities.csv\n")
cat("   - final_topic_summaries.csv\n")
cat("   - final_topic_prevalence.csv\n")
cat("   - final_model_metadata.csv\n")
cat("   - final_fitted_model.rds\n")
cat("   - manuscript_table_final_topics.csv\n")

# =============================================================================
# Next steps
# =============================================================================

cat("\n🎯 NEXT STEPS\n")
cat("=============\n")
cat("1. Review the topic previews above\n")
cat("2. Run study2_topic_quality_assessment.R for detailed quality analysis\n")
cat("3. If quality is good, proceed to visualizations\n")
cat("4. Research team should assign meaningful names to topics\n")
cat("5. Update manuscript with final results\n")

cat(glue("\n✅ Final analysis complete with k={FINAL_K} topics!\n"))
cat("Ready for quality assessment and manuscript integration.\n")

# Print session info for reproducibility
cat("\n📝 Session Info:\n")
sessionInfo()