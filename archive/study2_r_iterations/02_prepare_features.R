#!/usr/bin/env Rscript
# 02_prepare_features.R -------------------------------------------------------
# Purpose: Prepare TF-IDF features from labeled utterances for supervised ML
# Author : AI Assistant, 2025-06-30
# -------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidytext)
  library(Matrix)
  library(feather)
})

# -------------------------------------------------------------------------
# 1. Load labeled data ----------------------------------------------------
# -------------------------------------------------------------------------

llm_labels_path <- 'results/study2/llm_labels.csv'
if (!file.exists(llm_labels_path)) {
  stop('LLM labels file not found. Run 01_label_interest_gemini.py first.')
}

df_labeled <- read_csv(llm_labels_path, show_col_types = FALSE)

message('Loaded ', nrow(df_labeled), ' labeled utterances')

# Check label distribution
label_counts <- df_labeled %>% count(gemini_label, sort = TRUE)
print(label_counts)

# -------------------------------------------------------------------------
# 2. Text preprocessing ---------------------------------------------------
# -------------------------------------------------------------------------

# Define stop words (including domain-specific terms)
data("stop_words", package = "tidytext")

custom_stop_words <- c(
  stop_words$word,
  "counselor", "counselors", "counseling", "therapy", "therapist", 
  "mental", "health", "substance", "addiction", "sud", "field",
  "uh", "um", "like", "yeah", "you", "know", "kinda", "sorta"
)

# Tokenize and clean
df_tokens <- df_labeled %>%
  # Filter out error labels for now
  filter(gemini_label %in% c('INTERESTED', 'NOT_INTERESTED', 'NEUTRAL')) %>%
  
  # Tokenize
  unnest_tokens(word, text, token = "words", to_lower = TRUE) %>%
  
  # Clean tokens
  mutate(
    word = str_remove_all(word, "[0-9]+"),           # Remove numbers
    word = str_remove_all(word, "[^a-z'\\-]"),       # Keep only letters, apostrophes, hyphens
    word = str_trim(word)
  ) %>%
  
  # Filter
  filter(
    nchar(word) >= 3,                                # At least 3 characters
    word != "",                                      # Not empty
    !word %in% custom_stop_words,                    # Not stop words
    str_detect(word, "^[a-z][a-z'\\-]*$")           # Valid format
  )

message('After tokenization: ', nrow(df_tokens), ' tokens from ', 
        length(unique(df_tokens$utterance_id)), ' utterances')

# -------------------------------------------------------------------------
# 3. Create binary outcome variable --------------------------------------
# -------------------------------------------------------------------------

# Convert to binary: INTERESTED vs everything else
df_tokens <- df_tokens %>%
  mutate(
    interested_binary = ifelse(gemini_label == 'INTERESTED', 1, 0),
    not_interested_binary = ifelse(gemini_label == 'NOT_INTERESTED', 1, 0)
  )

# Check class balance
outcome_counts <- df_tokens %>%
  distinct(utterance_id, interested_binary) %>%
  count(interested_binary)

message('Binary outcome distribution:')
print(outcome_counts)

# -------------------------------------------------------------------------
# 4. TF-IDF Feature Engineering ------------------------------------------
# -------------------------------------------------------------------------

# Calculate term frequencies by utterance
tf_by_utterance <- df_tokens %>%
  count(utterance_id, word, name = "n") %>%
  group_by(utterance_id) %>%
  mutate(
    total_words = sum(n),
    tf = n / total_words
  ) %>%
  ungroup()

# Calculate IDF
idf_values <- df_tokens %>%
  distinct(utterance_id, word) %>%
  count(word, name = "docs_with_term") %>%
  mutate(
    total_docs = length(unique(df_tokens$utterance_id)),
    idf = log(total_docs / docs_with_term)
  )

# Combine TF and IDF
tfidf_scores <- tf_by_utterance %>%
  left_join(idf_values, by = "word") %>%
  mutate(tf_idf = tf * idf) %>%
  select(utterance_id, word, tf_idf)

# -------------------------------------------------------------------------
# 5. Create feature matrix -----------------------------------------------
# -------------------------------------------------------------------------

# Filter to most informative terms (optional)
top_terms <- tfidf_scores %>%
  group_by(word) %>%
  summarise(
    mean_tfidf = mean(tf_idf),
    docs_with_term = n(),
    .groups = 'drop'
  ) %>%
  filter(
    docs_with_term >= 2,          # Appears in at least 2 documents
    mean_tfidf > 0.01             # Minimum TF-IDF threshold
  ) %>%
  arrange(desc(mean_tfidf)) %>%
  slice_head(n = 500)             # Top 500 terms

message('Selected ', nrow(top_terms), ' most informative terms')

# Create sparse matrix
tfidf_filtered <- tfidf_scores %>%
  filter(word %in% top_terms$word)

# Cast to wide format (utterance x terms matrix)
tfidf_matrix <- tfidf_filtered %>%
  pivot_wider(
    id_cols = utterance_id,
    names_from = word,
    values_from = tf_idf,
    values_fill = 0
  )

# Add outcome variables and metadata
outcome_data <- df_labeled %>%
  filter(gemini_label %in% c('INTERESTED', 'NOT_INTERESTED', 'NEUTRAL')) %>%
  mutate(
    interested_binary = ifelse(gemini_label == 'INTERESTED', 1, 0),
    not_interested_binary = ifelse(gemini_label == 'NOT_INTERESTED', 1, 0)
  ) %>%
  select(utterance_id, session, speaker, gemini_label, 
         interested_binary, not_interested_binary)

# Combine features with outcomes
final_dataset <- outcome_data %>%
  left_join(tfidf_matrix, by = "utterance_id") %>%
  # Replace any remaining NAs with 0
  mutate(across(where(is.numeric), ~replace_na(.x, 0)))

message('Final dataset: ', nrow(final_dataset), ' utterances x ', 
        ncol(final_dataset) - 6, ' features')

# -------------------------------------------------------------------------
# 6. Save processed data --------------------------------------------------
# -------------------------------------------------------------------------

dir.create('results/study2', showWarnings = FALSE, recursive = TRUE)

# Save as feather for R (fast loading)
write_feather(final_dataset, 'results/study2/tfidf_features.feather')

# Save feature names
feature_names <- setdiff(names(final_dataset), 
                        c('utterance_id', 'session', 'speaker', 'gemini_label', 
                          'interested_binary', 'not_interested_binary'))

write_lines(feature_names, 'results/study2/feature_names.txt')

# Save summary statistics
summary_stats <- tibble(
  metric = c(
    'total_utterances',
    'interested_count',
    'not_interested_count', 
    'neutral_count',
    'total_features',
    'avg_features_per_doc',
    'sparsity'
  ),
  value = c(
    nrow(final_dataset),
    sum(final_dataset$interested_binary),
    sum(final_dataset$not_interested_binary),
    sum(final_dataset$gemini_label == 'NEUTRAL'),
    length(feature_names),
    mean(rowSums(final_dataset[feature_names] > 0)),
    round(1 - mean(final_dataset[feature_names] > 0), 3)
  )
)

write_csv(summary_stats, 'results/study2/feature_summary.csv')

# Print summary
message('\n✅ Feature preparation complete!')
message('Results saved to:')
message('  - results/study2/tfidf_features.feather (main dataset)')
message('  - results/study2/feature_names.txt (feature list)')
message('  - results/study2/feature_summary.csv (summary stats)')

print(summary_stats)