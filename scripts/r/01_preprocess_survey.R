# 01_preprocess_survey.R
# This script mirrors the functionality of the Python 01_preprocess_survey.py script
# It performs initial data cleaning and variable selection for the SUD Counselor Paper

# Load required packages
library(tidyverse)
library(here)

# Configuration
input_file <- here("data", "survey", "survey_raw.csv")
output_file <- here("data", "processed", "survey_processed.csv")
vars_to_include_file <- here("data", "survey", "initial_analysis_vars_to_include.csv")

# Create output directory if it doesn't exist
dir.create(here("data", "processed"), recursive = TRUE, showWarnings = FALSE)

cat("Starting survey data preprocessing...\n")

# Load raw survey data
cat(paste0("Loading raw survey data from ", input_file, "...\n"))
tryCatch({
  survey_raw <- read_csv(input_file, show_col_types = FALSE)
  cat(paste0("Loaded ", nrow(survey_raw), " rows and ", ncol(survey_raw), " columns.\n"))
}, error = function(e) {
  stop(paste0("Error loading input file: ", e$message))
})

# Load variables to include
cat(paste0("Loading variables list from ", vars_to_include_file, "...\n"))
tryCatch({
  vars_to_include <- read_csv(vars_to_include_file, show_col_types = FALSE)
  cat(paste0("Loaded ", nrow(vars_to_include), " variables to include.\n"))
}, error = function(e) {
  stop(paste0("Error loading variables inclusion list: ", e$message))
})

# Filter responses based on completion metrics
cat("Filtering responses based on completion metrics...\n")
survey_filtered <- survey_raw %>%
  filter(
    Progress == 100,
    Finished == TRUE,
    `CarelessResponderDC` == 0  # Ensures minimum time spent
  )
cat(paste0("Retained ", nrow(survey_filtered), " out of ", nrow(survey_raw), " responses.\n"))

# Select and rename variables based on inclusion list
cat("Selecting and renaming variables...\n")
# Create a named vector for renaming
var_mapping <- vars_to_include %>%
  select(original_var_name, new_var_name) %>%
  deframe()

# Filter to variables in the inclusion list
survey_selected <- survey_filtered %>%
  select(any_of(vars_to_include$original_var_name))

# Rename variables
survey_renamed <- survey_selected %>%
  rename(!!!var_mapping)

cat(paste0("Selected and renamed ", ncol(survey_renamed), " variables.\n"))

# Recode variables based on their types
cat("Recoding variables based on their types...\n")

# Function to recode Likert/ordinal variables
recode_likert <- function(variable, recode_map) {
  # Replace "I prefer not to answer" with NA, then recode
  variable %>%
    na_if("I prefer not to answer") %>%
    recode(!!!recode_map) 
}

# Process each variable based on its type from the inclusion list
survey_processed <- survey_renamed

# First identify the types
numeric_vars <- vars_to_include %>% 
  filter(var_type == "Numeric") %>% 
  pull(new_var_name)

ordinal_vars <- vars_to_include %>% 
  filter(var_type == "Ordinal/Likert") %>% 
  pull(new_var_name)

# For each ordinal variable, apply specific recoding if needed
# Example recoding for SUD counselor interest
if("sud_counselor_interest" %in% names(survey_processed)) {
  survey_processed <- survey_processed %>%
    mutate(
      sud_counselor_interest = recode_likert(
        sud_counselor_interest,
        c(
          "Not interested" = 1,
          "Slightly interested" = 2,
          "Moderately interested" = 3,
          "Definitely interested" = 4
        )
      )
    )
}

# Example recoding for SUD counselor familiarity
if("sud_counselor_familiarity" %in% names(survey_processed)) {
  survey_processed <- survey_processed %>%
    mutate(
      sud_counselor_familiarity = recode_likert(
        sud_counselor_familiarity,
        c(
          "Not at all familiar" = 1,
          "Slightly familiar" = 2,
          "Moderately familiar" = 3,
          "Very familiar" = 4,
          "Extremely familiar" = 5
        )
      )
    )
}

# Example recoding for stress variables
stress_vars <- grep("^stress_", names(survey_processed), value = TRUE)
if(length(stress_vars) > 0) {
  survey_processed <- survey_processed %>%
    mutate(across(
      all_of(stress_vars),
      ~ recode_likert(
        .,
        c(
          "Not at all stressful" = 1,
          "Slightly stressful" = 2,
          "Moderately stressful" = 3,
          "Very stressful" = 4,
          "Extremely stressful" = 5
        )
      )
    ))
}

# Add other specific variable recodings as needed

# Ensure numeric variables are actually numeric
survey_processed <- survey_processed %>%
  mutate(across(all_of(numeric_vars), as.numeric))

# Add progress indicators for tracking through later analysis
survey_processed <- survey_processed %>%
  mutate(
    careless_responder = as.integer(`CarelessResponderDC` > 0),
    completed = as.integer(Finished == TRUE),
    progress = Progress
  )

# Save processed data
cat(paste0("Saving processed data to ", output_file, "...\n"))
write_csv(survey_processed, output_file)
cat(paste0("Saved processed data with ", nrow(survey_processed), " rows and ", ncol(survey_processed), " columns.\n"))

cat("Survey preprocessing complete.\n")