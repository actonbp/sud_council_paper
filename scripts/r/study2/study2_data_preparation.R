# =============================================================================
# Study 2: Data Preparation - Moderator Removal & Substantive Content
# SUD Counseling Career Research Project
# =============================================================================
# Creates focus_group_substantive.csv by removing moderator utterances
# Following June 10, 2025 Plan - Enhanced preprocessing for LDA topic modeling
# =============================================================================

library(tidyverse)
library(here)

cat("=== Study 2: Data Preparation ===\n")

# Set up directories
data_dir <- here("data")
output_file <- file.path(data_dir, "focus_group_substantive.csv")

# =============================================================================
# STEP 1: Load and Combine Focus Group Data
# =============================================================================

cat("Loading focus group files...\n")

# Get all processed focus group files
focus_group_files <- list.files(
  file.path(data_dir, "focus_group"),
  pattern = "*_processed\\.csv$",
  full.names = TRUE
)

cat("Found", length(focus_group_files), "focus group files:\n")
cat("•", basename(focus_group_files), sep = "\n• ")
cat("\n")

# Load and combine all focus group data
focus_group_data <- map_dfr(focus_group_files, function(file) {
  cat("Processing:", basename(file), "\n")
  
  data <- read_csv(file, show_col_types = FALSE) %>%
    mutate(
      source_file = basename(file),
      session_id = str_extract(basename(file), "^[^_]+_[^_]+_[^_]+")
    )
  
  cat("  Loaded", nrow(data), "rows\n")
  return(data)
})

cat("\nCombined data:", nrow(focus_group_data), "total utterances\n")

# =============================================================================
# STEP 2: Identify and Remove Moderator Utterances
# =============================================================================

cat("\n=== Moderator Identification and Removal ===\n")

# Analyze speaker patterns by session
speaker_analysis <- focus_group_data %>%
  group_by(session_id, Speaker) %>%
  summarise(
    n_utterances = n(),
    avg_length = mean(str_length(Text), na.rm = TRUE),
    first_utterance = min(row_number(), na.rm = TRUE),
    sample_text = paste(head(Text, 2), collapse = " | "),
    .groups = "drop"
  ) %>%
  arrange(session_id, first_utterance)

cat("Speaker analysis by session:\n")
print(speaker_analysis)

# Identify moderators (Speaker 1 pattern based on previous analysis)
moderator_speakers <- focus_group_data %>%
  group_by(session_id) %>%
  slice_min(row_number(), n = 1) %>%
  ungroup() %>%
  select(session_id, moderator_speaker = Speaker)

cat("\nIdentified moderators by session:\n")
print(moderator_speakers)

# Remove moderator utterances
participant_data <- focus_group_data %>%
  left_join(moderator_speakers, by = "session_id") %>%
  filter(Speaker != moderator_speaker) %>%
  select(-moderator_speaker)

cat("\nModerator removal results:\n")
cat("• Original utterances:", nrow(focus_group_data), "\n")
cat("• After moderator removal:", nrow(participant_data), "\n")
cat("• Removed utterances:", nrow(focus_group_data) - nrow(participant_data), "\n")
cat("• Retention rate:", round(nrow(participant_data) / nrow(focus_group_data) * 100, 1), "%\n")

# =============================================================================
# STEP 3: Clean and Standardize Data
# =============================================================================

cat("\n=== Data Cleaning and Standardization ===\n")

# Clean and standardize the participant data
substantive_data <- participant_data %>%
  # Remove empty or very short utterances
  filter(
    !is.na(Text),
    str_length(Text) > 10,
    !str_detect(Text, "^\\s*$")
  ) %>%
  # Clean text
  mutate(
    cleaned_text = Text %>%
      str_to_lower() %>%
      str_replace_all("[[:punct:]]", " ") %>%
      str_replace_all("\\s+", " ") %>%
      str_trim(),
    
    # Create unique response ID
    response_id = row_number(),
    
    # Extract session info
    session_date = str_extract(session_id, "\\d+_\\d+_\\d+"),
    session_time = str_extract(session_id, "\\d+[ap]m"),
    
    # Speaker as participant ID within session
    participant_id = paste(session_id, Speaker, sep = "_")
  ) %>%
  # Remove any remaining problematic rows
  filter(str_length(cleaned_text) > 5) %>%
  # Select final columns
  select(
    response_id,
    session_id,
    session_date,
    session_time,
    speaker = Speaker,
    participant_id,
    text = Text,
    cleaned_text,
    source_file
  )

cat("Data cleaning results:\n")
cat("• Utterances after cleaning:", nrow(substantive_data), "\n")
cat("• Final retention rate:", round(nrow(substantive_data) / nrow(focus_group_data) * 100, 1), "%\n")

# =============================================================================
# STEP 4: Data Quality Checks
# =============================================================================

cat("\n=== Data Quality Assessment ===\n")

# Session-level summary
session_summary <- substantive_data %>%
  group_by(session_id) %>%
  summarise(
    n_utterances = n(),
    n_participants = n_distinct(speaker),
    avg_utterance_length = mean(str_length(cleaned_text)),
    date = first(session_date),
    time = first(session_time),
    .groups = "drop"
  ) %>%
  arrange(session_id)

cat("Session-level summary:\n")
print(session_summary)

# Overall quality metrics
cat("\nOverall data quality:\n")
cat("• Total sessions:", n_distinct(substantive_data$session_id), "\n")
cat("• Total participants:", n_distinct(substantive_data$participant_id), "\n")
cat("• Average utterances per session:", round(mean(session_summary$n_utterances), 1), "\n")
cat("• Average participants per session:", round(mean(session_summary$n_participants), 1), "\n")
cat("• Average utterance length:", round(mean(str_length(substantive_data$cleaned_text)), 1), "characters\n")

# Text length distribution
text_lengths <- str_length(substantive_data$cleaned_text)
cat("• Text length quartiles:", paste(quantile(text_lengths), collapse = ", "), "\n")

# =============================================================================
# STEP 5: Save Substantive Data
# =============================================================================

cat("\n=== Saving Substantive Data ===\n")

# Save the substantive data file
write_csv(substantive_data, output_file)

cat("Substantive data saved to:", output_file, "\n")

# Save session summary for reference
session_summary_file <- file.path(data_dir, "focus_group_sessions_summary.csv")
write_csv(session_summary, session_summary_file)

cat("Session summary saved to:", session_summary_file, "\n")

# =============================================================================
# STEP 6: Create Sample for Review
# =============================================================================

# Create a sample for manual review
sample_data <- substantive_data %>%
  group_by(session_id) %>%
  slice_sample(n = 3) %>%
  ungroup() %>%
  select(session_id, speaker, text, cleaned_text) %>%
  arrange(session_id, speaker)

sample_file <- file.path(data_dir, "focus_group_sample_review.csv")
write_csv(sample_data, sample_file)

cat("Sample data for review saved to:", sample_file, "\n")

# =============================================================================
# SUMMARY REPORT
# =============================================================================

cat("\n" %+% strrep("=", 60), "\n")
cat("STUDY 2 DATA PREPARATION - SUMMARY REPORT\n")
cat(strrep("=", 60), "\n\n")

cat("Input Data:\n")
cat("• Focus group files processed:", length(focus_group_files), "\n")
cat("• Total original utterances:", nrow(focus_group_data), "\n\n")

cat("Moderator Removal:\n")
cat("• Moderator utterances removed:", nrow(focus_group_data) - nrow(participant_data), "\n")
cat("• Participant utterances retained:", nrow(participant_data), "\n\n")

cat("Data Cleaning:\n")
cat("• Final substantive utterances:", nrow(substantive_data), "\n")
cat("• Overall retention rate:", round(nrow(substantive_data) / nrow(focus_group_data) * 100, 1), "%\n\n")

cat("Output Files:\n")
cat("• ", output_file, "\n")
cat("• ", session_summary_file, "\n")
cat("• ", sample_file, "\n\n")

cat("Next Steps:\n")
cat("• Review sample data for quality\n")
cat("• Run LDA topic modeling analysis\n")
cat("• Validate moderator removal was successful\n\n")

cat("Data preparation completed successfully!\n")
cat(strrep("=", 60), "\n") 