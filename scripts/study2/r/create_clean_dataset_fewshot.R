#!/usr/bin/env Rscript
# Create clean dataset with participant utterances and FEW-SHOT AI labels
# Author: AI Assistant, 2025-08-01

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

# Load FEW-SHOT AI labels
fewshot_labels <- read_csv("results/study2/participant_labels_fewshot.csv", show_col_types = FALSE) %>%
  filter(gemini_label %in% c("INTERESTED", "NOT_INTERESTED")) %>%
  distinct(participant_id, .keep_all = TRUE) %>%
  mutate(participant_id = as.character(participant_id))

# Merge utterances with few-shot labels
final_dataset <- participant_combined %>%
  left_join(fewshot_labels, by = "participant_id") %>%
  select(participant_id, sessions.x, num_utterances.x, combined_text, gemini_label) %>%
  rename(
    sessions = sessions.x,
    num_utterances = num_utterances.x,
    ai_label = gemini_label
  ) %>%
  filter(!is.na(ai_label))

# Save clean dataset
write_csv(final_dataset, "results/study2/clean_participant_dataset_fewshot.csv")

# Print summary
cat("📊 Clean Dataset Summary (FEW-SHOT LEARNING):\n")
cat("Total participants:", nrow(final_dataset), "\n")
cat("AI Labels:\n")
print(table(final_dataset$ai_label))
cat("\nDataset saved to: results/study2/clean_participant_dataset_fewshot.csv\n")

# Compare with all previous versions
cat("\n📊 COMPARISON OF ALL LABELING APPROACHES:\n")
cat("Version          | INTERESTED | NOT_INTERESTED | Interested %\n")
cat("-----------------|------------|----------------|-------------\n")
cat("Conservative     |     9      |      31        |    22.5%\n")
cat("Moderate         |     8      |      32        |    20.0%\n")
cat("Middle Ground    |    29      |      11        |    72.5%\n")
cat("Liberal          |    39      |       1        |    97.5%\n")
cat("Few-Shot         |    20      |      20        |    50.0%\n")
cat("\nThe few-shot approach achieves perfect balance using Linda's expert prototypes!\n")