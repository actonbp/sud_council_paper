# =============================================================================
# Study 2: LDA Topic Modeling Analysis
# SUD Counseling Career Research Project
# =============================================================================
# Following tidytext best practices from Julia Silge and David Robinson
# Two-step approach: 
# 1. Identify utterances mentioning SUD counseling
# 2. Remove SUD terms from those utterances and model remaining themes
# =============================================================================

# Load required libraries
library(tidyverse)
library(tidytext)
library(topicmodels)
library(here)
library(SnowballC)

# Set up results directory
results_dir <- here("results", "r", "study2_lda_modeling")
if (!dir.exists(results_dir)) {
  dir.create(results_dir, recursive = TRUE)
}

# =============================================================================
# 1. Load and prepare data
# =============================================================================

cat("Loading focus group data...\n")

# Load the substantive data from step 1
substantive_data <- read_csv(here("data", "focus_group_substantive.csv"))

cat("Loaded", nrow(substantive_data), "substantive utterances\n")

# =============================================================================
# 2. Step 1: Identify SUD-related utterances
# =============================================================================

cat("\\nStep 1: Identifying SUD-related utterances...\n")

# Create initial tokens to identify SUD content
initial_tokens <- substantive_data %>%
  unnest_tokens(word, cleaned_text) %>%
  mutate(word_stem = wordStem(word, language = "english"))

# Define terms for broader career/SUD discussion identification
relevant_terms <- c(
  # Direct SUD terms
  "substanc", "addict", "drug", "alcohol", "abus", "depend", "recover",
  # SUD counseling terms
  "counsel", "therap", "treat", "rehab", "clinic",
  # Career terms
  "career", "job", "work", "profession", "field", "major", "studi",
  # Education terms
  "school", "colleg", "univers", "degre", "program", "train",
  # Interest/motivation terms
  "interest", "motiv", "passion", "want", "like", "enjoy",
  # General helping/mental health terms
  "help", "mental", "health", "psychol", "psychiatr"
)

# Identify utterances that mention career/SUD topics (much broader)
relevant_utterances <- initial_tokens %>%
  filter(
    # Look for any career or SUD-related content
    str_detect(word_stem, paste(relevant_terms, collapse = "|"))
  ) %>%
  distinct(response_id) %>%
  pull(response_id)

cat("Found", length(relevant_utterances), "utterances mentioning career/SUD topics")
cat(" (", round(length(relevant_utterances)/nrow(substantive_data)*100, 1), "% of total)\\n")

# =============================================================================
# 3. Step 2: Remove SUD terms and create topic modeling dataset
# =============================================================================

cat("\\nStep 2: Removing SUD terms and preparing for topic modeling...\n")

# Filter to career/SUD-mentioning utterances only, then remove obvious terms
word_counts <- substantive_data %>%
  filter(response_id %in% relevant_utterances) %>%
  # Tokenize
  unnest_tokens(word, cleaned_text) %>%
  # Apply stemming
  mutate(word_stem = wordStem(word, language = "english")) %>%
  # Remove stop words
  anti_join(stop_words, by = "word") %>%
     # Remove obvious career/SUD terms (this is the key step!)
   filter(
     !str_detect(word_stem, "substanc|addict|drug|alcohol|abus|depend|recover|counsel|therap|treat|rehab|clinic"),
     # Also remove career-related terms that are too obvious
     !word_stem %in% c("mental", "health", "job", "career"),
     # Also remove very common discussion words that don't add insight
     !word_stem %in% c("like", "think", "know", "say", "go", "get", "want", 
                       "thing", "time", "way", "come", "see", "look", "feel",
                       "talk", "tell", "ask", "mean", "kind", "sort", "really",
                       "yeah", "guess", "don", "lot", "ve", "peopl", "person",
                       "didn", "ll", "realiz")
   ) %>%
  # Count word stems by response
  count(response_id, word_stem, sort = TRUE) %>%
  # Remove very rare terms (appear in <3 documents to ensure meaningful topics)
  group_by(word_stem) %>%
  filter(n() >= 3) %>%
  ungroup()

cat("After removing SUD terms:")
cat("\\n- Unique terms:", n_distinct(word_counts$word_stem))
cat("\\n- Documents:", n_distinct(word_counts$response_id))
cat("\\n- Total tokens:", sum(word_counts$n), "\\n")

# =============================================================================
# 4. Create Document-Term Matrix
# =============================================================================

cat("\\nCreating document-term matrix...\n")

# Cast to document-term matrix for topicmodels
focus_dtm <- word_counts %>%
  cast_dtm(response_id, word_stem, n)

cat("DTM dimensions:", dim(focus_dtm)[1], "documents x", dim(focus_dtm)[2], "terms\n")

# Check if we have enough data for meaningful topic modeling
if (dim(focus_dtm)[2] < 10) {
  stop("Too few terms remaining (", dim(focus_dtm)[2], "). Consider relaxing filtering criteria.")
}

# =============================================================================
# 5. Model Selection: Try different numbers of topics
# =============================================================================

cat("\\nTesting different numbers of topics...\n")

# Try different k values
k_values <- 2:6  # Smaller range since we have fewer terms now
model_results <- tibble(
  k = k_values,
  model = map(k_values, ~{
    cat("Fitting LDA model with k =", .x, "...\n")
    LDA(focus_dtm, k = .x, control = list(seed = 1234))
  }),
  perplexity = map_dbl(model, perplexity)
)

cat("\\nModel comparison results:\n")
print(model_results %>% select(k, perplexity))

# Select optimal k (lowest perplexity)
optimal_k <- model_results$k[which.min(model_results$perplexity)]
optimal_model <- model_results$model[[which.min(model_results$perplexity)]]

cat("\\nOptimal number of topics:", optimal_k, "\n")
cat("Best perplexity:", min(model_results$perplexity), "\n")

# =============================================================================
# 6. Extract and save topic information
# =============================================================================

cat("\\nExtracting topic information...\n")

# Extract per-topic-per-word probabilities (beta)
topic_terms <- tidy(optimal_model, matrix = "beta")

# Extract per-document-per-topic probabilities (gamma)  
topic_documents <- tidy(optimal_model, matrix = "gamma")

# Extract word assignments
word_assignments <- augment(optimal_model, data = focus_dtm)

# Get top terms per topic
top_terms_per_topic <- topic_terms %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  ungroup() %>%
  arrange(topic, -beta)

cat("\\nContextual themes when discussing SUD counseling:\n")
top_terms_per_topic %>%
  group_by(topic) %>%
  summarise(
    top_terms = paste(head(term, 5), collapse = ", "),
    .groups = "drop"
  ) %>%
  mutate(interpretation = case_when(
    topic == 1 ~ "Theme 1",
    topic == 2 ~ "Theme 2", 
    topic == 3 ~ "Theme 3",
    topic == 4 ~ "Theme 4",
    TRUE ~ paste("Theme", topic)
  )) %>%
  print()

# =============================================================================
# 7. Save results
# =============================================================================

cat("\\nSaving results...\n")

# Save model comparison
write_csv(model_results %>% select(k, perplexity), 
          file.path(results_dir, "model_comparison.csv"))

# Save topic-term probabilities
write_csv(topic_terms, 
          file.path(results_dir, "topic_term_probabilities.csv"))

# Save document-topic probabilities
write_csv(topic_documents, 
          file.path(results_dir, "document_topic_probabilities.csv"))

# Save word assignments
write_csv(word_assignments, 
          file.path(results_dir, "word_topic_assignments.csv"))

# Save top terms per topic
write_csv(top_terms_per_topic, 
          file.path(results_dir, "top_terms_per_topic.csv"))

# Save model summary with methodology details
model_summary <- tibble(
  analysis_date = Sys.Date(),
  approach = "Two-step: Identify SUD utterances, remove SUD terms, model remaining themes",
  relevant_utterances_identified = length(relevant_utterances),
  detection_rate = round(length(relevant_utterances)/nrow(substantive_data)*100, 1),
  optimal_k = optimal_k,
  best_perplexity = min(model_results$perplexity),
  n_documents = nrow(focus_dtm),
  n_terms = ncol(focus_dtm),
  total_word_tokens = sum(word_counts$n)
)

write_csv(model_summary, 
          file.path(results_dir, "model_summary.csv"))

cat("\\n=== Study 2 LDA Topic Modeling Complete ===\n")
cat("Approach: Contextual themes in career counseling discussions\n")
cat("Relevant utterances identified:", length(relevant_utterances), 
    "(", round(length(relevant_utterances)/nrow(substantive_data)*100, 1), "%)\n")
cat("Optimal model: k =", optimal_k, "topics\n")
cat("Perplexity:", round(min(model_results$perplexity), 2), "\n")
cat("Results saved to:", results_dir, "\n")
cat("Ready for visualization step!\n") 