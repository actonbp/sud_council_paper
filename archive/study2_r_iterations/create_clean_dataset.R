#!/usr/bin/env Rscript
# Create clean dataset with participant utterances and AI labels
# Author: AI Assistant, 2025-07-17

library(dplyr)
library(readr)
library(purrr)

# Load all focus group data
data_dir <- "data/study2"
csv_files <- list.files(data_dir, pattern = "*.csv", full.names = TRUE)

# Read and combine all sessions
all_utterances <- map_dfr(csv_files, function(file) {
  df <- read_csv(file, show_col_types = FALSE)
  df$session <- tools::file_path_sans_ext(basename(file))
  return(df)
})

# Filter out moderators (2-3 letter codes like LR, BR)
participant_utterances <- all_utterances %>%
  filter(!grepl("^[A-Z]{2,3}$", Speaker)) %>%
  filter(!is.na(Text))

# Combine all utterances by participant
participant_combined <- participant_utterances %>%
  group_by(Speaker) %>%
  summarise(
    sessions = paste(unique(session), collapse = "; "),
    num_utterances = n(),
    combined_text = paste(Text, collapse = " "),
    .groups = "drop"
  ) %>%
  rename(participant_id = Speaker)

# Load AI labels (keep only successful labels, remove duplicates)
ai_labels <- read_csv("results/study2/participant_labels.csv", show_col_types = FALSE) %>%
  filter(gemini_label %in% c("INTERESTED", "NOT_INTERESTED")) %>%
  distinct(participant_id, .keep_all = TRUE) %>%
  mutate(participant_id = as.character(participant_id))

# Merge utterances with AI labels
final_dataset <- participant_combined %>%
  left_join(ai_labels, by = "participant_id") %>%
  select(participant_id, sessions.x, num_utterances.x, combined_text, gemini_label) %>%
  rename(
    sessions = sessions.x,
    num_utterances = num_utterances.x,
    ai_label = gemini_label
  ) %>%
  filter(!is.na(ai_label))

# Save clean dataset
write_csv(final_dataset, "results/study2/clean_participant_dataset.csv")

# Print summary
cat("📊 Clean Dataset Summary:\n")
cat("Total participants:", nrow(final_dataset), "\n")
cat("AI Labels:\n")
print(table(final_dataset$ai_label))
cat("\nDataset saved to: results/study2/clean_participant_dataset.csv\n")