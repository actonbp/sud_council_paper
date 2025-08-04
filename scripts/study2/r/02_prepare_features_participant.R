#!/usr/bin/env Rscript
# 02_prepare_features_participant.R -------------------------------------------
# Purpose: Prepare TF-IDF features from participant-level labeled data
# Author : AI Assistant, 2025-08-01
# -------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidytext)
  library(Matrix)
  library(arrow)
})

# -------------------------------------------------------------------------
# 1. Load participant-level labeled data ----------------------------------
# -------------------------------------------------------------------------

participant_labels_path <- 'results/study2/clean_participant_dataset_fewshot.csv'
if (!file.exists(participant_labels_path)) {
  stop('Participant labels file not found. Run Python labeling scripts first.')
}

df_participants <- read_csv(participant_labels_path, show_col_types = FALSE)

message('Loaded ', nrow(df_participants), ' participants')

# Check label distribution
label_counts <- df_participants %>% count(ai_label, sort = TRUE)
message('\nLabel distribution:')
print(label_counts)

# -------------------------------------------------------------------------
# 2. Text preprocessing ---------------------------------------------------
# -------------------------------------------------------------------------

# Define stop words (including domain-specific terms)
data("stop_words", package = "tidytext")

custom_stop_words <- c(
  stop_words$word,
  "yeah", "uh", "um", "hmm", "oh", "ah", "eh",
  "gonna", "gotta", "wanna", "kinda", "sorta",
  "okay", "ok", "alright", "right",
  "actually", "basically", "literally", "obviously",
  "stuff", "thing", "things"
)

# Tokenize and clean
df_tokens <- df_participants %>%
  # Create unique ID and binary outcome
  mutate(
    doc_id = participant_id,
    interested_binary = ifelse(ai_label == 'INTERESTED', 1, 0)
  ) %>%
  
  # Tokenize combined text
  unnest_tokens(word, combined_text, token = "words", to_lower = TRUE) %>%
  
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

message('\nAfter tokenization: ', nrow(df_tokens), ' tokens from ', 
        length(unique(df_tokens$doc_id)), ' participants')

# -------------------------------------------------------------------------
# 3. TF-IDF Feature Engineering ------------------------------------------
# -------------------------------------------------------------------------

# Calculate term frequencies by participant
tf_by_participant <- df_tokens %>%
  count(doc_id, word, name = "n") %>%
  group_by(doc_id) %>%
  mutate(
    total_words = sum(n),
    tf = n / total_words
  ) %>%
  ungroup()

# Calculate IDF
idf_values <- df_tokens %>%
  distinct(doc_id, word) %>%
  count(word, name = "docs_with_term") %>%
  mutate(
    total_docs = length(unique(df_tokens$doc_id)),
    idf = log(total_docs / docs_with_term)
  )

# Combine TF and IDF
tfidf_scores <- tf_by_participant %>%
  left_join(idf_values, by = "word") %>%
  mutate(tf_idf = tf * idf) %>%
  select(doc_id, word, tf_idf)

# -------------------------------------------------------------------------
# 4. Feature selection ----------------------------------------------------
# -------------------------------------------------------------------------

# Filter to most informative terms
top_terms <- tfidf_scores %>%
  group_by(word) %>%
  summarise(
    mean_tfidf = mean(tf_idf),
    max_tfidf = max(tf_idf),
    docs_with_term = n(),
    .groups = 'drop'
  ) %>%
  filter(
    docs_with_term >= 3,          # Appears in at least 3 participants (7.5%)
    docs_with_term <= 37          # Not in more than 92.5% of participants
  ) %>%
  arrange(desc(mean_tfidf)) %>%
  slice_head(n = 1000)            # Top 1000 terms

message('\nSelected ', nrow(top_terms), ' features after filtering')

# Show top 20 terms
message('\nTop 20 terms by mean TF-IDF:')
top_terms %>% slice_head(n = 20) %>% print()

# -------------------------------------------------------------------------
# 5. Create feature matrix -----------------------------------------------
# -------------------------------------------------------------------------

# Filter to selected terms
tfidf_filtered <- tfidf_scores %>%
  filter(word %in% top_terms$word)

# Cast to wide format (participant x terms matrix)
tfidf_matrix <- tfidf_filtered %>%
  pivot_wider(
    id_cols = doc_id,
    names_from = word,
    values_from = tf_idf,
    values_fill = 0
  )

# Add outcome variables and metadata
outcome_data <- df_participants %>%
  mutate(
    doc_id = participant_id,
    interested_binary = ifelse(ai_label == 'INTERESTED', 1, 0)
  ) %>%
  select(doc_id, participant_id, sessions, num_utterances, 
         ai_label, interested_binary)

# Combine features with outcomes
final_dataset <- outcome_data %>%
  left_join(tfidf_matrix, by = "doc_id") %>%
  # Replace any remaining NAs with 0
  mutate(across(where(is.numeric) & !c(doc_id, participant_id), ~replace_na(.x, 0)))

message('\nFinal dataset: ', nrow(final_dataset), ' participants x ', 
        ncol(final_dataset) - 6, ' features')

# -------------------------------------------------------------------------
# 6. Save processed data --------------------------------------------------
# -------------------------------------------------------------------------

dir.create('results/study2', showWarnings = FALSE, recursive = TRUE)

# Save as feather for fast loading
write_feather(final_dataset, 'results/study2/tfidf_participant.feather')

# Save as CSV for compatibility
write_csv(final_dataset, 'results/study2/tfidf_participant.csv')

# Save feature names
feature_names <- setdiff(names(final_dataset), 
                        c('doc_id', 'participant_id', 'sessions', 'num_utterances',
                          'ai_label', 'interested_binary'))

write_lines(feature_names, 'results/study2/participant_feature_names.txt')

# Save summary statistics
summary_stats <- tibble(
  metric = c(
    'total_participants',
    'interested_count',
    'not_interested_count', 
    'total_features',
    'avg_features_per_participant',
    'sparsity'
  ),
  value = c(
    nrow(final_dataset),
    sum(final_dataset$interested_binary),
    sum(final_dataset$interested_binary == 0),
    length(feature_names),
    mean(rowSums(final_dataset[feature_names] > 0)),
    round(1 - mean(as.matrix(final_dataset[feature_names]) > 0), 3)
  )
)

write_csv(summary_stats, 'results/study2/participant_feature_summary.csv')

# -------------------------------------------------------------------------
# 7. Exploratory analysis of key terms ------------------------------------
# -------------------------------------------------------------------------

# Get top discriminative terms
interested_ids <- outcome_data$doc_id[outcome_data$interested_binary == 1]
not_interested_ids <- outcome_data$doc_id[outcome_data$interested_binary == 0]

discriminative_terms <- tfidf_scores %>%
  filter(word %in% top_terms$word) %>%
  mutate(group = ifelse(doc_id %in% interested_ids, "interested", "not_interested")) %>%
  group_by(word, group) %>%
  summarise(
    mean_tfidf = mean(tf_idf),
    n_docs = n(),
    .groups = 'drop'
  ) %>%
  pivot_wider(
    names_from = group,
    values_from = c(mean_tfidf, n_docs),
    values_fill = list(mean_tfidf = 0, n_docs = 0)
  ) %>%
  mutate(
    diff_tfidf = mean_tfidf_interested - mean_tfidf_not_interested,
    ratio_tfidf = ifelse(mean_tfidf_not_interested > 0, 
                        mean_tfidf_interested / mean_tfidf_not_interested, 
                        NA_real_)
  ) %>%
  filter(!is.na(ratio_tfidf))

# Top terms for interested group
message('\nTop 15 terms associated with INTERESTED:')
discriminative_terms %>%
  arrange(desc(diff_tfidf)) %>%
  slice_head(n = 15) %>%
  select(word, mean_tfidf_interested, mean_tfidf_not_interested, diff_tfidf) %>%
  print()

# Top terms for not interested group  
message('\nTop 15 terms associated with NOT_INTERESTED:')
discriminative_terms %>%
  arrange(diff_tfidf) %>%
  slice_head(n = 15) %>%
  select(word, mean_tfidf_interested, mean_tfidf_not_interested, diff_tfidf) %>%
  print()

write_csv(discriminative_terms, 'results/study2/discriminative_terms.csv')

# Print summary
message('\n✅ Feature preparation complete!')
message('Results saved to:')
message('  - results/study2/tfidf_participant.feather (main dataset)')
message('  - results/study2/tfidf_participant.csv (CSV version)')
message('  - results/study2/participant_feature_names.txt (feature list)')
message('  - results/study2/participant_feature_summary.csv (summary stats)')
message('  - results/study2/discriminative_terms.csv (term analysis)')

print(summary_stats)