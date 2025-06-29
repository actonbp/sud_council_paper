# CO-OCCURRENCE FREQUENCY TABLE
# Extract all word pair co-occurrences with frequencies

library(tidyverse)
library(here)
library(tidytext)
library(widyr)

cat("=== CO-OCCURRENCE FREQUENCY TABLE ===\n")
cat("All word pairs with their co-occurrence frequencies\n\n")

# Load the preprocessed data
stems_with_session <- read_csv(here("data", "focus_group_tokens_preprocessed.csv"), show_col_types = FALSE)

# Use the same parameters as the meaningful clustering analysis
MIN_FREQ <- 3
TOP_N_WORDS <- 25

# Get word frequencies
all_word_frequencies <- stems_with_session %>%
  count(word_stem, sort = TRUE)

word_frequencies <- all_word_frequencies %>%
  filter(n >= MIN_FREQ)

# Focus on top words (same as clustering analysis)
top_words <- word_frequencies %>%
  slice_head(n = TOP_N_WORDS) %>%
  pull(word_stem)

cat("Analyzing co-occurrences among top", TOP_N_WORDS, "words\n")
cat("Top words:", paste(head(top_words, 10), collapse = ", "), "...\n\n")

# Create co-occurrence pairs
word_pairs <- stems_with_session %>%
  filter(word_stem %in% top_words) %>%
  pairwise_count(word_stem, response_id, sort = TRUE) %>%
  filter(n >= 2)  # Keep pairs that co-occur at least 2 times

cat("📊 CO-OCCURRENCE FREQUENCY TABLE:\n")
cat("=================================\n")
cat("Total word pairs:", nrow(word_pairs), "\n")
cat("Frequency range:", min(word_pairs$n), "to", max(word_pairs$n), "\n\n")

# Create formatted table
cooccurrence_table <- word_pairs %>%
  mutate(
    word_pair = paste(item1, "←→", item2),
    rank = row_number()
  ) %>%
  select(rank, word_pair, frequency = n, item1, item2)

cat("RANKED CO-OCCURRENCE PAIRS (Top 50):\n")
cat("====================================\n")
cat("Rank | Word Pair                    | Frequency\n")
cat("-----|------------------------------|----------\n")

for (i in 1:min(50, nrow(cooccurrence_table))) {
  cat(sprintf("%4d | %-28s | %9d\n", 
              cooccurrence_table$rank[i], 
              cooccurrence_table$word_pair[i], 
              cooccurrence_table$frequency[i]))
}

if (nrow(cooccurrence_table) > 50) {
  cat("... and", nrow(cooccurrence_table) - 50, "more pairs\n")
}

cat("\n📈 FREQUENCY DISTRIBUTION:\n")
cat("=========================\n")

freq_dist <- cooccurrence_table %>%
  count(frequency, name = "n_pairs") %>%
  arrange(desc(frequency))

cat("Frequency | Number of Pairs\n")
cat("----------|----------------\n")
for (i in 1:min(15, nrow(freq_dist))) {
  cat(sprintf("%9d | %15d\n", freq_dist$frequency[i], freq_dist$n_pairs[i]))
}

cat("\n🔍 HIGHEST CO-OCCURRENCE PAIRS:\n")
cat("===============================\n")

top_pairs <- cooccurrence_table %>%
  slice_head(n = 10)

for (i in 1:nrow(top_pairs)) {
  cat(sprintf("%d. %s ←→ %s (co-occurs %d times)\n", 
              i, 
              top_pairs$item1[i], 
              top_pairs$item2[i], 
              top_pairs$frequency[i]))
}

cat("\n💾 SAVING COMPLETE TABLE:\n")
cat("========================\n")

# Save the complete table
write_csv(cooccurrence_table, here("results", "cooccurrence_frequency_table.csv"))

cat("✅ Complete table saved to: results/cooccurrence_frequency_table.csv\n")
cat("Total pairs saved:", nrow(cooccurrence_table), "\n")

# Summary statistics
cat("\n📊 SUMMARY STATISTICS:\n")
cat("=====================\n")
cat("Mean co-occurrence frequency:", round(mean(cooccurrence_table$frequency), 2), "\n")
cat("Median co-occurrence frequency:", median(cooccurrence_table$frequency), "\n")
cat("Standard deviation:", round(sd(cooccurrence_table$frequency), 2), "\n")
cat("25th percentile:", quantile(cooccurrence_table$frequency, 0.25), "\n")
cat("75th percentile:", quantile(cooccurrence_table$frequency, 0.75), "\n")

# Show individual word frequencies for context
cat("\n📝 INDIVIDUAL WORD FREQUENCIES (for context):\n")
cat("=============================================\n")

word_freq_context <- word_frequencies %>%
  filter(word_stem %in% top_words) %>%
  arrange(desc(n)) %>%
  mutate(rank = row_number())

cat("Rank | Word      | Frequency\n")
cat("-----|-----------|----------\n")
for (i in 1:nrow(word_freq_context)) {
  cat(sprintf("%4d | %-9s | %9d\n", 
              word_freq_context$rank[i], 
              word_freq_context$word_stem[i], 
              word_freq_context$n[i]))
}

cat("\n✅ Analysis complete!\n") 