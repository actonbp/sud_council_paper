#!/usr/bin/env Rscript
# Create clean dataset with participant utterances and LIBERAL AI labels
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

# Load LIBERAL AI labels
liberal_labels <- read_csv("results/study2/participant_labels_liberal.csv", show_col_types = FALSE) %>%
  filter(gemini_label %in% c("INTERESTED", "NOT_INTERESTED")) %>%
  distinct(participant_id, .keep_all = TRUE) %>%
  mutate(participant_id = as.character(participant_id))

# Load CONSERVATIVE AI labels for comparison
conservative_labels <- read_csv("results/study2/participant_labels.csv", show_col_types = FALSE) %>%
  filter(gemini_label %in% c("INTERESTED", "NOT_INTERESTED")) %>%
  distinct(participant_id, .keep_all = TRUE) %>%
  mutate(participant_id = as.character(participant_id))

# Merge utterances with both label versions
final_dataset <- participant_combined %>%
  left_join(liberal_labels, by = "participant_id") %>%
  left_join(conservative_labels, by = "participant_id", suffix = c("_liberal", "_conservative")) %>%
  select(participant_id, sessions, num_utterances.x, combined_text, 
         gemini_label_liberal, gemini_label_conservative) %>%
  rename(
    num_utterances = num_utterances.x,
    ai_label_liberal = gemini_label_liberal,
    ai_label_conservative = gemini_label_conservative
  ) %>%
  filter(!is.na(ai_label_liberal))

# Save clean dataset with both label versions
write_csv(final_dataset, "results/study2/clean_participant_dataset_liberal.csv")

# Print comparison summary
cat("📊 Liberal vs Conservative Labeling Comparison:\n")
cat("Total participants:", nrow(final_dataset), "\n\n")

cat("LIBERAL Labels:\n")
print(table(final_dataset$ai_label_liberal))

cat("\nCONSERVATIVE Labels:\n")
print(table(final_dataset$ai_label_conservative))

# Show participants who changed labels
label_changes <- final_dataset %>%
  filter(ai_label_liberal != ai_label_conservative) %>%
  select(participant_id, ai_label_conservative, ai_label_liberal)

cat("\nParticipants who changed from Conservative to Liberal:\n")
print(label_changes)

cat("\nDataset saved to: results/study2/clean_participant_dataset_liberal.csv\n")